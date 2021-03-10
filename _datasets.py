import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from torch.utils.data import IterableDataset


max_duration: int = 3  # seconds
sample_rate: int = 16000  # Hz

data_root: str = '/media/sdc1/'

_df_types = dict(
    channel=str, chapter_id=str, clip_id=str, data_type=str, duration=float,
    is_sparse=bool, set_id=str, speaker_id=str, utterance_id=str,
    freesound_id=str,
)

EPS = 1e-30


def create_df_librispeech(
    root_directory: str,
    csv_path: str = 'corpora/librispeech.csv'
):
    """Creates a Pandas DataFrame with files from the LibriSpeech corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.openslr.org/12/>`_.
    """
    assert os.path.isdir(root_directory)
    df = pd.read_csv(csv_path, dtype=_df_types)
    df = df[(df.set_id == 'train-clean-100')
            & (df.duration > max_duration)]
    df.loc[:, 'max_offset'] = (df.duration - max_duration) * sample_rate
    df.loc[:, 'max_offset'] = df['max_offset'].astype(int)
    df.loc[:, 'partition'] = 'pretrain'
    for speaker_id in df.speaker_id.unique():
        _mask = (df['speaker_id'] == speaker_id)
        _last_row = _mask[::-1].idxmax()
        df.loc[_last_row-25:_last_row-20, 'partition'] = 'prevalidation'
        df.loc[_last_row-20:_last_row-10, 'partition'] = 'finetune'
        df.loc[_last_row-10:_last_row-5, 'partition'] = 'validation'
        df.loc[_last_row-5:_last_row, 'partition'] = 'test'
    df.loc[:, 'filepath'] = (
        root_directory + '/' + df.set_id + '/' + df.speaker_id + '/'
        + df.chapter_id + '/' + df.speaker_id + '-' + df.chapter_id
        + '-' + df.utterance_id + '.wav'
    )
    assert all(df.filepath.apply(os.path.isfile))
    return df


def create_df_musan(
    root_directory: str,
    csv_path: str = 'corpora/musan.csv'
):
    """Creates a Pandas DataFrame with files from the MUSAN corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.openslr.org/17/>`_.
    """
    assert os.path.isdir(root_directory)
    df = pd.read_csv(csv_path, dtype=_df_types)
    df = df[df.duration > max_duration]
    df = df.sample(frac=1, random_state=0)
    df.loc[:, 'max_offset'] = (df.duration - max_duration) * sample_rate
    df.loc[:, 'max_offset'] = df['max_offset'].astype(int)
    df.loc[:, 'filepath'] = (
        root_directory + '/' + df.data_type + '/' + df.set_id + '/'
        + df.data_type + '-' + df.set_id + '-' + df.clip_id + '.wav'
    )
    df.loc[:, 'split'] = df.set_id
    df.split = df.split.replace({'free-sound': 'train', 'sound-bible': 'test'})
    assert all(df.filepath.apply(os.path.isfile))
    return df


def create_df_fsd50k(
    root_directory: str,
    csv_path: str = 'corpora/fsd50k.csv'
):
    """Creates a Pandas DataFrame with files from the FSD50K corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://zenodo.org/record/4060432/>`_.
    """
    assert os.path.isdir(root_directory)
    _dir_map = dict(train='FSD50K.dev_audio',
                    val='FSD50K.dev_audio',
                    test='FSD50K.eval_audio')
    df = pd.read_csv(csv_path, dtype=_df_types)
    df = df[df.duration > max_duration]
    df = df.sample(frac=1, random_state=0)
    df.loc[:, 'max_offset'] = (df.duration - max_duration) * sample_rate
    df.loc[:, 'max_offset'] = df['max_offset'].astype(int)
    df.loc[:, 'filepath'] = (
        root_directory + '/' + df.split.replace(_dir_map) + '/'
        + df.freesound_id + '.wav'
    )
    assert all(df.filepath.apply(os.path.isfile))
    return df


def logistic(v, alpha: float = 1., beta: float = 1.):
    return (1 / (1 + alpha * np.exp(-beta*v)))


def segmental_snr(
    estimate: np.ndarray,
    target: np.ndarray,
    segment_size: int = 1024,
    hop_length: int = 256,
    center: bool = True,
    pad_mode: str = 'reflect'
):
    """Estimates segmental signal-to-noise ratios on a frame-by-frame basis."""
    if center:
        estimate = np.pad(estimate, int(segment_size//2), mode=pad_mode)
        target = np.pad(target, int(segment_size//2), mode=pad_mode)

    s = (target)                    # source
    r = (target - estimate)         # residual
    w = np.hanning(segment_size)    # window

    seg_snrs = []

    for i in range(0, s.shape[-1]-segment_size, hop_length):
        s_i = s[..., i:i+segment_size] * w
        r_i = r[..., i:i+segment_size] * w
        s_i = (s_i**2).sum(axis=-1) + EPS
        r_i = (r_i**2).sum(axis=-1) + EPS
        snr_i = 10*np.log10(s_i/r_i)
        seg_snrs.append(snr_i)

    return np.stack(seg_snrs).T


class Mixtures(IterableDataset):

    def __init__(
        self,
        speaker_ids: Sequence[str],
        premixture_set: Optional[str] = None,
        premixture_snr: Optional[Union[float, Tuple[float, float]]] = None,
        mixture_set: Optional[str] = None,
        mixture_snr: Optional[Union[float, Tuple[float, float]]] = None,
        dataset_duration: Optional[float] = None,
        utterance_duration: float = 1.,
        target_snrs: bool = False,
    ):
        # parse and sanity check arguments
        self.sanity_check(locals())

        # setup internal corpora
        self.df_s = librispeech.query(f'speaker_id in {self.speaker_ids}')
        self.df_m = fsd50k.query(f'split == "{self.premixture_set}"')
        self.df_n = musan.query(f'split == "{self.mixture_set}"')

        # setup corpora pointers
        self.reset()

        # if dealing with a single-speaker, load all the audio data in advance
        self.speech = []
        if len(self.speaker_ids) == 1:

            for filepath in self.df_s.filepath.tolist():
                (_, s) = wavfile.read(filepath)
                self.speech.append(s)
            self.speech = np.concatenate(self.speech)

            # truncate the corpus if specified
            if dataset_duration:
                num_samples_dataset = int(dataset_duration * sample_rate)
                self.speech = self.speech[:num_samples_dataset]

            # normalize
            self.speech = self.speech / (self.speech.std() + EPS)

    def __iter__(self):
        return self

    def reset(self):
        self.rng = np.random.default_rng(0)
        (self.s_idx, self.m_idx, self.n_idx) = (-1, -1, -1)

    def sanity_check(self, args: dict):

        # verify speaker ID(s)
        if not isinstance(args['speaker_ids'], (list, set)):
            raise ValueError('Expected a list or set of speaker IDs.')
        if len(args['speaker_ids']) < 1:
            raise ValueError('Expected one or more speaker IDs.')
        if not set(args['speaker_ids']).issubset(set(librispeech.speaker_id)):
            raise ValueError('Invalid speaker IDs, must be within LibriSpeech.')
        self.speaker_ids = args['speaker_ids']

        # missing pairs of arguments
        if args['premixture_set'] != None and args['premixture_snr'] == None:
            raise ValueError('Missing argument `premixture_snr`.')
        if args['premixture_set'] == None and args['premixture_snr'] != None:
            raise ValueError('Missing argument `premixture_set`.')
        if args['mixture_set'] != None and args['mixture_snr'] == None:
            raise ValueError('Missing argument `mixture_snr`.')
        if args['mixture_set'] == None and args['mixture_snr'] != None:
            raise ValueError('Missing argument `mixture_set`.')

        # unpack mixture SNR values
        if isinstance(args['premixture_snr'], tuple):
            self.premixture_snr_min = float(min(args['premixture_snr']))
            self.premixture_snr_max = float(max(args['premixture_snr']))
        elif isinstance(args['premixture_snr'], (float, int)):
            self.premixture_snr_min = float(args['premixture_snr'])
            self.premixture_snr_max = float(args['premixture_snr'])
        elif args['premixture_snr'] is None:
            self.premixture_snr_min = None
            self.premixture_snr_max = None
        else:
            raise ValueError('Expected `premixture_snr` to be a float type or '
                             'a tuple of floats.')
        if isinstance(args['mixture_snr'], tuple):
            self.mixture_snr_min = float(min(args['mixture_snr']))
            self.mixture_snr_max = float(max(args['mixture_snr']))
        elif isinstance(args['mixture_snr'], (float, int)):
            self.mixture_snr_min = float(args['mixture_snr'])
            self.mixture_snr_max = float(args['mixture_snr'])
        elif args['mixture_snr'] is None:
            self.mixture_snr_min = None
            self.mixture_snr_max = None
        else:
            raise ValueError('Expected `mixture_snr` to be a float type or '
                             'a tuple of floats.')

        # verify corpus sets
        if args['premixture_snr'] is not None:
            if not (args['premixture_set'] in ('train', 'val', 'test')):
                raise ValueError('Expected `premixture_set` to be '
                                 '"train", "val", or "test".')
        if args['mixture_snr'] is not None:
            if args['mixture_set'] == 'free-sound':
                args['mixture_set'] = 'train'
            elif args['mixture_set'] == 'sound-bible':
                args['mixture_set'] = 'test'
            elif not (args['mixture_set'] in ('train', 'test')):
                raise ValueError('Expected `mixture_set` to be '
                                 '"train" or "test".')
        self.premixture_set = args['premixture_set'] or ''
        self.mixture_set = args['mixture_set'] or ''

        # verify target specification
        if not isinstance(args['target_snrs'], bool):
            raise ValueError('Expected `target_snrs` to be True or False.')
        self.target_snrs = args['target_snrs']

        # verify utterance duration
        if not isinstance(args['utterance_duration'], (int, float)):
            raise ValueError('Expected `utterance_duration` to be a number.')
        self.utterance_duration = args['utterance_duration']

        # verify dataset duration
        if not isinstance(args['dataset_duration'], (int, float, type(None))):
            raise ValueError('Expected `dataset_duration` to be a number.')
        self.dataset_duration = args['dataset_duration']

    def __next__(self):

        length = int(self.utterance_duration * sample_rate)

        # slice from speech array, randomly offset, truncate, normalize, and mix
        s: np.ndarray = np.zeros(length)
        if len(self.speaker_ids) > 1:
            self.s_idx = (self.s_idx + 1) % len(self.df_s)
            offset_s = self.df_s.max_offset.iloc[self.s_idx]
            (_, _s) = wavfile.read(self.df_s.filepath.iloc[self.s_idx])
            s = _s[offset_s:offset_s+length]
        else:
            offset_s = max(1, len(self.speech) - length)
            offset_s = self.rng.integers(0, offset_s)
            s = self.speech[offset_s:offset_s+length]
        s = s / (EPS + s.std())
        x = p = s

        # read premixture noise, randomly offset, truncate, normalize, and mix
        m: np.ndarray = np.zeros(length)
        if len(self.df_m) > 0:
            self.m_idx = (self.m_idx + 1) % len(self.df_m)
            offset_m = max(1, self.df_m.max_offset.iloc[self.m_idx])
            offset_m = self.rng.integers(0, offset_m)
            (_, m) = wavfile.read(self.df_m.filepath.iloc[self.m_idx])
            m = m[offset_m:offset_m+length]
            m = m / (EPS + m.std())
            snr = self.rng.uniform(self.premixture_snr_min,
                                   self.premixture_snr_max)
            p = s + (m * 10 ** (-snr / 20.))
            x = p

        # read deformation noise, randomly offset, truncate, normalize, and mix
        n: np.ndarray = np.zeros(length)
        if len(self.df_n) > 0:
            self.n_idx = (self.n_idx + 1) % len(self.df_n)
            offset_n = max(1, self.df_m.max_offset.iloc[self.n_idx])
            offset_n = self.rng.integers(0, offset_n)
            (_, n) = wavfile.read(self.df_n.filepath.iloc[self.n_idx])
            n = n[offset_n:offset_n+length]
            n = n / (EPS + n.std())
            snr = self.rng.uniform(self.mixture_snr_min,
                                   self.mixture_snr_max)
            x = p + (n * 10 ** (-snr / 20.))

        # create output tuple
        sample = ()
        scale_factor = EPS + max(abs(x.min()), abs(x.max()))
        if self.target_snrs:
            sample = (
                torch.Tensor(x) / scale_factor,     # noise-injected premixture
                torch.Tensor(segmental_snr(x, s)),  # frame-by-frame SNRs
            )
        else:
            sample = (
                torch.Tensor(x) / scale_factor,  # noise-injected premixture
                torch.Tensor(p) / scale_factor,  # premixture (or clean speech)
            )
        return sample

    # end of class


_sf = ('_8khz' if sample_rate == 8000 else '')
librispeech = create_df_librispeech(os.path.join(data_root, 'librispeech'+_sf))
fsd50k = create_df_fsd50k(os.path.join(data_root, 'fsd50k'+_sf))
musan = create_df_musan(os.path.join(data_root, 'musan'+_sf))

speakers_vl = pd.read_csv('speakers/validation.csv', dtype=_df_types)
speakers_te = pd.read_csv('speakers/test.csv', dtype=_df_types)
speaker_ids_vl = set(speakers_vl.speaker_id)
speaker_ids_te = set(speakers_te.speaker_id)
speaker_ids_tr = set(librispeech.speaker_id) - speaker_ids_vl - speaker_ids_te
speaker_ids_vl = sorted(speaker_ids_vl)
speaker_ids_te = sorted(speaker_ids_te)
speaker_ids_tr = sorted(speaker_ids_tr)
