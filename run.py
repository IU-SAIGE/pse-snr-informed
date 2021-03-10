import json
import os
import warnings
from typing import List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wv
import torch
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
    best_loss: float = np.inf
    best_state_dict: Optional[Dict[str, Tensor]] = None

    # setup training dataloader
    train_dataloader = DataLoader(
        D.Mixtures(
            speaker_ids=D.speaker_ids_tr,
            premixture_set='train',
            premixture_snr=mixture_snr,
            target_snrs=True
        ), batch_size)

    # prepare validation set
    val_dataloader = DataLoader(
        D.Mixtures(
            speaker_ids=D.speaker_ids_vl,
            premixture_set='val',
            premixture_snr=mixture_snr,
            target_snrs=True
        ), 100)
    (vl_x, vl_snrs) = next(iter(val_dataloader))
    vl_x = vl_x.to(device)
    vl_snrs = vl_snrs.to(device)

    #
    # begin training loop
    #

    for (tr_x, tr_snrs) in train_dataloader:

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
            tune.report(num_batches=num_batches, vl_loss=vl_loss)

        if vl_loss < best_loss:
            best_epoch = current_epoch
            best_loss = vl_loss
            best_state_dict = model.state_dict()
            with tune.checkpoint_dir(best_epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save(best_state_dict, path)

        # check for convergence
        if (current_epoch - best_epoch > 1000):
            break

    #
    # end training loop
    #

    # save the model parameters to the local results folders
    if output_dir:
        torch.save(best_state_dict, os.path.join(output_dir, 'checkpoint'))
        with open(os.path.join(output_dir, 'vl_loss.json'), 'w') as fp:
            json.dump(losses, fp, indent=2, sort_keys=True)

    # return the trial dictionary
    return {
        'num_batches': best_epoch * batch_size,
        'vl_loss': best_loss,
        'losses': losses,
    }


def ray_tune_predictor(
    num_gpus: float = 0.5,
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

    reporter = CLIReporter(
        parameter_columns=['hidden_size', 'num_layers',
                           'learning_rate', 'batch_size'],
        metric_columns=['num_batches', 'vl_loss'],
        max_report_frequency=10,
    )

    # use Tune to queue up trials in parallel on the GPUs
    tune.run(
        _ray_tune_predictor,
        config=config,
        keep_checkpoints_num=1,
        log_to_file='log.txt',
        queue_trials=True,
        resources_per_trial={'gpu': num_gpus},
        verbose=1
    )

    print('Finished `ray_tune_predictor(' +
          ', '.join([f'{k}={v}' for (k,v) in config.items()]) + ')`.')


def main():
    ray_tune_predictor()


if __name__ == '__main__':
    main()
