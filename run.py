import json
import os
import pathlib
import warnings
from collections import OrderedDict
from typing import Dict, List, Sequence, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from asteroid.losses.sdr import SingleSrcNegSDR
from pytorch_lightning import seed_everything
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch import Tensor
from torch.utils.data import DataLoader

# local imports
import _datasets as D
import _models as M


warnings.filterwarnings('ignore')

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = os.path.dirname(os.path.realpath(__file__))


def train_predictor(
    hidden_size: int,
    num_layers: int,
    learning_rate: float,
    batch_size: int = 100,
    mixture_snr: Union[float, Sequence[float]] = (-10., 10.),
    ray_report: bool = False,
    save_to_local: bool = True,
):

    # fix seed
    seed_everything(0)

    # create local output directory
    output_dir: str = ''
    if save_to_local:
        output_dir = os.path.join(ROOT, 'weights', 'snr_predictor',
                                  f'{num_layers}x{hidden_size:04d}')
        pathlib.Path(output_dir).mkdir(0o777, True, True)
        print('[INFO] Will copy results to {}.'.format(output_dir))
        if os.path.exists(os.path.join(output_dir, 'checkpoint')):
            print('[WARN] This will overwrite an existing checkpoint.')
    else:
        print('[INFO] Will not copy best checkpoint / test results locally.')

    # select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    # instantiate predictor
    model = M.PredictorGRU(hidden_size, num_layers).to(device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # instantiate predictor loss
    criterion = torch.nn.MSELoss()

    # setup metrics + state dict
    (num_batches, current_epoch, best_epoch) = (0, 0, 0)
    losses: List[float] = []
    best_loss: float = 1e30
    best_state_dict: Optional[Dict[str, Tensor]] = None

    # setup training dataloader
    train_dataloader = DataLoader(
        D.Mixtures(
            speaker_ids=D.speaker_ids_tr,
            utterance_split='all',
            premixture_split='train',
            premixture_snr=mixture_snr,
        ), batch_size)

    # prepare validation set
    val_dataloader = DataLoader(
        D.Mixtures(
            speaker_ids=D.speaker_ids_vl,
            utterance_split='all',
            premixture_split='val',
            premixture_snr=mixture_snr,
        ), 100)
    (vl_x, _, vl_snrs) = next(iter(val_dataloader))
    vl_x = vl_x.to(device)
    vl_snrs = vl_snrs.to(device)

    #
    # begin training loop
    #

    for (tr_x, _, tr_snrs) in train_dataloader:

        current_epoch += 1
        num_batches += batch_size

        # zero parameter gradients
        optimizer.zero_grad()

        # move data to GPU
        tr_x = tr_x.to(device)
        tr_snrs = tr_snrs.to(device)

        # forward propagation + calculate loss
        loss = criterion(model(tr_x), tr_snrs)

        # backward propagation + optimize
        loss.backward()
        optimizer.step()

        # only validate every few epochs
        if current_epoch % 10:
            continue

        # validate
        with torch.no_grad():
            vl_loss = float(criterion(model(vl_x), vl_snrs))
            losses.append(vl_loss)

        if ray_report:
            # send intermediate results back to Ray
            tune.report(num_batches=num_batches,
                best_iter=best_epoch//10, best_loss=best_loss)

        if vl_loss < best_loss:
            best_epoch = current_epoch
            best_loss = vl_loss
            best_state_dict = model.state_dict()
            if ray_report:
                with tune.checkpoint_dir(best_epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'checkpoint')
                    torch.save(best_state_dict, path)
                    path = os.path.join(checkpoint_dir, 'vl_loss.json')
                    with open(path, 'w') as fp:
                        json.dump(losses, fp, indent=2, sort_keys=True)

        # check for convergence
        if (current_epoch - best_epoch > 1000):
            break

    #
    # end training loop
    #

    # save the model parameters to the local results folders
    if output_dir:
        path = os.path.join(output_dir, 'checkpoint')
        torch.save(best_state_dict, path)
        path = os.path.join(output_dir, 'vl_loss.json')
        with open(path, 'w') as fp:
            json.dump(losses, fp, indent=2, sort_keys=True)

    # return the trial dictionary
    return {
        'num_batches': best_epoch * batch_size,
        'best_iter': best_epoch//10,
        'best_loss': best_loss,
        'losses': losses,
    }


def ray_tune_predictor(
    num_gpus: float = 0.25,
):

    def _ray_tune_predictor(config: dict):
        return train_predictor(
            hidden_size=config.get('hidden_size', 1024),
            num_layers=config.get('num_layers', 2),
            learning_rate=config.get('learning_rate', 1e-3),
            batch_size=config.get('batch_size', 100),
            mixture_snr=config.get('mixture_snr', (-10., 10.)),
            ray_report=True,
            save_to_local=config.get('save_to_local', False),
        )

    # create a configuration dictionary which will vary across every trial
    config = {
        'hidden_size': tune.grid_search([64, 128, 256, 512, 1024]),
        'num_layers': tune.grid_search([2, 3]),
        'learning_rate': tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        'batch_size': 128,
        'mixture_snr': (-10., 10.),
        'save_to_local': False
    }
    print(config)

    # use Tune to queue up trials in parallel on the GPUs
    tune.run(
        _ray_tune_predictor,
        config=config,
        keep_checkpoints_num=1,
        log_to_file='log.txt',
        queue_trials=True,
        resources_per_trial={'gpu': num_gpus},
        progress_reporter=CLIReporter(
            parameter_columns=['hidden_size', 'num_layers', 'learning_rate'],
            metric_columns=['num_batches', 'best_iter', 'best_loss'],
            max_report_frequency=10,
        ),
        verbose=1
    )

    print('Finished `ray_tune_predictor(' +
          ', '.join([f'{k}={v}' for (k,v) in config.items()]) + ')`.')


def train_denoiser(
    speaker_id: str,
    hidden_size: int,
    num_layers: int,
    learning_rate: float,
    batch_size: int = 100,
    premixture_snr: Union[float, Sequence[float]] = (0., 10.),
    mixture_snr: Union[float, Sequence[float]] = (-5., 5.),
    ray_report: bool = False,
    save_to_local: bool = True,
    predict_snr: bool = False,
    run_test: bool = False,
):

    # fix seed
    seed_everything(0)

    # create local output directory
    output_dir: str = ''
    if save_to_local:
        output_dir = os.path.join(ROOT, 'weights', speaker_id,
                                  'denoiser'+('_snr' if predict_snr else ''),
                                  f'{num_layers}x{hidden_size:04d}')
        pathlib.Path(output_dir).mkdir(0o777, True, True)
        with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
            json.dump(dict(
                speaker_id=speaker_id,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate,
                batch_size=batch_size,
                premixture_snr=premixture_snr,
                mixture_snr=mixture_snr,
                ray_report=ray_report,
                save_to_local=save_to_local,
                predict_snr=predict_snr,
                run_test=run_test,
            ), fp, indent=2, sort_keys=True)
        print('[INFO] Will copy results to {}.'.format(output_dir))
        if os.path.exists(os.path.join(output_dir, 'checkpoint')):
            print('[WARN] This will overwrite an existing checkpoint.')
    else:
        print('[INFO] Will not copy best checkpoint / test results locally.')

    # select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    # instantiate denoiser
    denoiser = M.DenoiserGRU(hidden_size, num_layers).to(device)

    # instantiate predictor
    if predict_snr:
        predictor = M.PredictorGRU(1024, 3).to(device)
        predictor.load_state_dict(torch.load(
            os.path.join(ROOT, 'weights/snr_predictor/3x1024/checkpoint')),
            strict=False)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=denoiser.parameters(), lr=learning_rate)

    # instantiate denoiser loss
    criterion = torch.nn.MSELoss()
    if predict_snr:
        criterion = M.SegmentalLoss('mse')

    # instantiate metric function
    loss_sisdr = SingleSrcNegSDR('sisdr')

    # setup metrics + state dict
    (num_batches, current_epoch, best_epoch) = (0, 0, 0)
    losses: List[float] = []
    best_loss: float = 1e30
    best_state_dict: Optional[Dict[str, Tensor]] = None

    # setup training dataloader
    train_dataloader = DataLoader(
        D.Mixtures(
            speaker_ids=[speaker_id],
            utterance_split='pretrain',
            premixture_split='test',
            premixture_snr=premixture_snr,
            mixture_split='train',
            mixture_snr=mixture_snr,
        ), batch_size)

    # prepare validation set
    val_dataloader = DataLoader(
        D.Mixtures(
            speaker_ids=[speaker_id],
            utterance_split='preval',
            # premixture_split='test',
            # premixture_snr=premixture_snr,
            mixture_split='train',
            mixture_snr=mixture_snr,
        ), 100)
    (vl_x, vl_s, _) = next(iter(val_dataloader))
    vl_x = vl_x.to(device)
    vl_s = vl_s.to(device)

    #
    # begin training loop
    #
    for (tr_x, tr_s, _) in train_dataloader:

        current_epoch += 1
        num_batches += batch_size

        # zero parameter gradients
        optimizer.zero_grad()

        # move data to GPU
        tr_x = tr_x.to(device)
        tr_s = tr_s.to(device)

        # forward propagation
        tr_y = denoiser(tr_x)

        # calculate loss
        min_len = min(tr_s.shape[-1], tr_y.shape[-1])
        tr_s = tr_s[..., :min_len]
        tr_y = tr_y[..., :min_len]

        if predict_snr:

            # predict SNR on the pseudo-source
            predicted_snrs = predictor(tr_s)
            weights = D.logistic(predicted_snrs)

            # apply the weights frame by frame
            loss = criterion(tr_y, tr_s, weights).mean()

        else:
            loss = criterion(tr_y, tr_s).mean()

        # backward propagation + optimize
        loss.backward()
        optimizer.step()

        # only validate every few epochs
        if current_epoch % 10:
            continue

        # validate
        with torch.no_grad():
            vl_y = denoiser(vl_x)
            min_len = min(vl_s.shape[-1], vl_y.shape[-1])
            vl_x = vl_x[..., :min_len]
            vl_s = vl_s[..., :min_len]
            vl_y = vl_y[..., :min_len]
            sisdr_in = loss_sisdr(vl_x, vl_s)
            sisdr_out = loss_sisdr(vl_y, vl_s)
            vl_loss = float((sisdr_out-sisdr_in).mean())
            losses.append(vl_loss)

        if ray_report:
            # send intermediate results back to Ray
            tune.report(num_batches=num_batches,
                best_iter=best_epoch//10, best_loss=best_loss)

        if vl_loss < best_loss:
            best_epoch = current_epoch
            best_loss = vl_loss
            best_state_dict = denoiser.state_dict()
            if ray_report:
                with tune.checkpoint_dir(best_epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'checkpoint')
                    torch.save(best_state_dict, path)
                    path = os.path.join(checkpoint_dir, 'vl_loss.json')
                    with open(path, 'w') as fp:
                        json.dump(losses, fp, indent=2, sort_keys=True)

        # check for convergence
        if (current_epoch - best_epoch > 1000):
            break

    #
    # end training loop
    #

    # run a test on the best model
    te_result = None
    if run_test:
        te_result = test_denoiser(
            speaker_id=speaker_id,
            hidden_size=hidden_size,
            num_layers=num_layers,
            state_dict=best_state_dict,
        )
        if output_dir:
            path = os.path.join(output_dir, 'te_result.json')
            with open(path, 'w') as fp:
                json.dump(te_result, fp, indent=2, sort_keys=True)
        if ray_report:
            with tune.checkpoint_dir(best_epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'te_result.json')
                with open(path, 'w') as fp:
                    json.dump(te_result, fp, indent=2, sort_keys=True)

    # save the model parameters to the local results folders
    if output_dir:
        path = os.path.join(output_dir, 'checkpoint')
        torch.save(best_state_dict, path)
        path = os.path.join(output_dir, 'vl_loss.json')
        with open(path, 'w') as fp:
            json.dump(losses, fp, indent=2, sort_keys=True)

    # return the trial dictionary
    return {
        'num_batches': best_epoch * batch_size,
        'best_iter': best_epoch//10,
        'best_loss': best_loss,
        'losses': losses,
    }


def test_denoiser(
    speaker_id: str,
    hidden_size: int,
    num_layers: int,
    mixture_snr: Union[float, Sequence[float]] = (-5., 5.),
    state_dict: Optional[Dict[str, Tensor]] = None,
    checkpoint_dir: Optional[str] = None,
):

    # sanity check arguments
    if not state_dict and not checkpoint_dir:
        raise ValueError('Expected either `checkpoint_dir` or `state_dict`.')
    if checkpoint_dir and not os.path.isdir(checkpoint_dir):
        raise ValueError(f'Invalid folder `checkpoint_dir`: {checkpoint_dir}.')
    if state_dict and not isinstance(state_dict, OrderedDict):
        raise ValueError(f'Expected a valid PyTorch `state_dict`.')
    if isinstance(speaker_id, (tuple, list)):
        raise ValueError(f'Expected a single string for `speaker_id`.')

    loss_sisdr = SingleSrcNegSDR('sisdr')

    # wrap with the no-gradient context manager
    with torch.no_grad():

        # fix seed
        seed_everything(0)

        # select device
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'

        # instantiate model
        denoiser = M.DenoiserGRU(hidden_size, num_layers).to(device)
        if checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            state_dict = torch.load(path)
        denoiser.load_state_dict(state_dict)

        te_result = {}

        test_dataloader = DataLoader(
            D.Mixtures(
                speaker_ids=[speaker_id],
                utterance_split='test',
                mixture_split='test',
                mixture_snr=mixture_snr,
            ), 100)
        (te_x, te_s, _) = next(iter(test_dataloader))
        te_x = te_x.to(device)
        te_s = te_s.to(device)

        # run test
        te_y = denoiser(te_x)

        min_len = min(te_x.shape[-1], te_y.shape[-1])
        te_x = te_x[..., :min_len]
        te_s = te_s[..., :min_len]
        te_y = te_y[..., :min_len]
        sisdr_in = loss_sisdr(te_x, te_s)
        sisdr_out = loss_sisdr(te_y, te_s)

        te_result[speaker_id] = str(float((sisdr_in-sisdr_out).mean()))+' dB'

    return te_result


def ray_tune_denoiser(
    num_gpus: float = .33,
):

    def _ray_tune_denoiser(config: dict):
        config['ray_report'] = True
        config['save_to_local'] = True
        config['run_test'] = True
        return train_denoiser(**config)

    # create a configuration dictionary which will vary across every trial
    config = {
        'predict_snr': tune.grid_search([False, True]),
        'speaker_id': tune.grid_search(list(D.speaker_ids_te)),
        'hidden_size': tune.grid_search([64, 128, 256, 512]),
        'num_layers': 2,
        'learning_rate': 1e-3,
        'batch_size': 128,
        'premixture_snr': (0., 10.),
        'mixture_snr': (-5., 5.),
    }
    print(config)

    # use Tune to queue up trials in parallel on the GPUs
    tune.run(
        _ray_tune_denoiser,
        config=config,
        keep_checkpoints_num=1,
        log_to_file='log.txt',
        queue_trials=True,
        resources_per_trial={'gpu': num_gpus},
        progress_reporter=CLIReporter(
            # parameter_columns=['hidden_size', 'learning_rate',
                               # 'speaker_id', 'predict_snr'],
            # metric_columns=['num_batches', 'best_iter', 'best_loss'],
            max_report_frequency=10,
        ),
        verbose=1
    )

    print('Finished `ray_tune_denoiser(' +
          ', '.join([f'{k}={v}' for (k,v) in config.items()]) + ')`.')


def main():
    ray_tune_denoiser()


if __name__ == '__main__':
    main()
