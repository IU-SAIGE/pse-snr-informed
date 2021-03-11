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
    df.loc[:, 'split'] = 'pretrain'
    for speaker_id in df.speaker_id.unique():
        _mask = (df['speaker_id'] == speaker_id)
        _last_row = _mask[::-1].idxmax()
        df.loc[_last_row-25:_last_row-20, 'split'] = 'preval'
        df.loc[_last_row-20:_last_row-10, 'split'] = 'train'
        df.loc[_last_row-10:_last_row-5, 'split'] = 'val'
        df.loc[_last_row-5:_last_row, 'split'] = 'test'
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
        utterance_split: str,
        premixture_split: Optional[str] = None,
        premixture_snr: Optional[Union[float, Tuple[float, float]]] = None,
        mixture_split: Optional[str] = None,
        mixture_snr: Optional[Union[float, Tuple[float, float]]] = None,
        dataset_duration: Optional[float] = None,
        utterance_duration: float = 1.
    ):
        # parse and sanity check arguments
        self.sanity_check(locals())

        # setup internal corpora
        self.df_s = librispeech.query(f'speaker_id in {self.speaker_ids}')
        if self.utterance_split != 'all':
            self.df_s = self.df_s.query(f'split == "{self.utterance_split}"')
        self.df_m = fsd50k.query(f'split == "{self.premixture_split}"')
        self.df_n = musan.query(f'split == "{self.mixture_split}"')

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

    def __repr__(self):
        '''Returns dataset constructor parameters.'''
        repr_str = f'{self.__class__.__name__}('
        repr_str += f'speaker_ids={self.speaker_ids}'
        repr_str += f', utterance_split=\'{self.utterance_split}\''
        if self.premixture_split is not '':
            repr_str += f', premixture_split=\'{self.premixture_split}\''
        if self.premixture_snr_min is not None:
            snr_str = f'({self.premixture_snr_min}, {self.premixture_snr_max})'
            if self.premixture_snr_min == self.premixture_snr_max:
                snr_str = f'{self.premixture_snr_min}'
            repr_str += f', premixture_snr={snr_str}'
        if self.mixture_split is not '':
            repr_str += f', mixture_split=\'{self.mixture_split}\''
        if self.mixture_snr_min is not None:
            snr_str = f'({self.mixture_snr_min}, {self.mixture_snr_max})'
            if self.mixture_snr_min == self.mixture_snr_max:
                snr_str = f'{self.mixture_snr_min}'
            repr_str += f', mixture_snr={snr_str}'
        if self.dataset_duration is not None:
            repr_str += f', dataset_duration={self.dataset_duration}'
        repr_str += f', utterance_duration={self.utterance_duration}'
        repr_str += ')'
        return repr_str

    def __next__(self):
        '''Generates mixture, source, and frame-by-frame SNR estimates.'''

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
            offset_n = max(1, self.df_n.max_offset.iloc[self.n_idx])
            offset_n = self.rng.integers(0, offset_n)
            (_, n) = wavfile.read(self.df_n.filepath.iloc[self.n_idx])
            n = n[offset_n:offset_n+length]
            n = n / (EPS + n.std())
            snr = self.rng.uniform(self.mixture_snr_min,
                                   self.mixture_snr_max)
            x = p + (n * 10 ** (-snr / 20.))

        # create output tuple
        scale_factor = EPS + np.abs(x).max()
        return (
            torch.Tensor(x) / scale_factor,     # mixture signal
            torch.Tensor(p) / scale_factor,     # premixture signal
            torch.Tensor(segmental_snr(x, s)),  # frame-by-frame SNRs
        )

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
        if args['premixture_split'] != None and args['premixture_snr'] == None:
            raise ValueError('Missing argument `premixture_snr`.')
        if args['premixture_split'] == None and args['premixture_snr'] != None:
            raise ValueError('Missing argument `premixture_split`.')
        if args['mixture_split'] != None and args['mixture_snr'] == None:
            raise ValueError('Missing argument `mixture_snr`.')
        if args['mixture_split'] == None and args['mixture_snr'] != None:
            raise ValueError('Missing argument `mixture_split`.')

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
        if not (args['utterance_split'] in
            ('all', 'pretrain', 'preval', 'train', 'val', 'test')):
            raise ValueError('Expected `utterance_split` to be either "all", '
                             '"pretrain", "preval", "train", "val", or "test".')
        if args['premixture_snr'] is not None:
            if not (args['premixture_split'] in ('train', 'val', 'test')):
                raise ValueError('Expected `premixture_split` to be either '
                                 '"train", "val", or "test".')
        if args['mixture_snr'] is not None:
            if args['mixture_split'] == 'free-sound':
                args['mixture_split'] = 'train'
            elif args['mixture_split'] == 'sound-bible':
                args['mixture_split'] = 'test'
            elif not (args['mixture_split'] in ('train', 'test')):
                raise ValueError('Expected `mixture_split` to be either '
                                 '"train" or "test".')
        self.utterance_split = args['utterance_split']
        self.premixture_split = args['premixture_split'] or ''
        self.mixture_split = args['mixture_split'] or ''

        # verify utterance duration
        if not isinstance(args['utterance_duration'], (int, float)):
            raise ValueError('Expected `utterance_duration` to be a number.')
        self.utterance_duration = args['utterance_duration']

        # verify dataset duration
        if not isinstance(args['dataset_duration'], (int, float, type(None))):
            raise ValueError('Expected `dataset_duration` to be a number.')
        self.dataset_duration = args['dataset_duration']

    def statistics(self):
        def duration(seconds, granularity=1):
            result = []
            for (word, count) in [('hrs', 3600), ('mins', 60), ('secs', 1)]:
                value = round(seconds / count)
                if value:
                    seconds -= value * count
                    if value == 1:
                        word = word.rstrip('s')
                    result.append("{:.0f} {}".format(value, word))
            return ', '.join(result[:granularity])
        s_str = f'\u2022 Speaker IDs: {self.speaker_ids}'
        s_str += f'\n\u2022 Utterance Partition: \'{self.utterance_split}\''
        s_str += f'\n  - # of utterance files: {len(self.df_s)}'
        s_str += f'\n  - Total Duration: {duration(sum(self.df_s.duration))}'
        if self.premixture_split != '':
            s_str += f'\n\u2022 Pre-noise Partition: \'{self.premixture_split}\''
            s_str += f'\n  - # of noises files: {len(self.df_m)}'
            s_str += f'\n  - Total Duration: {duration(sum(self.df_m.duration))}'
            _ = f'uniform({self.premixture_snr_min}, {self.premixture_snr_max})'
            if self.premixture_snr_min == self.premixture_snr_max:
                _ = f'{self.premixture_snr_min}'
            s_str += f'\n  - SNRs: {_}'
        if self.mixture_split != '':
            s_str += f'\n\u2022 Post-noise Partition: \'{self.mixture_split}\''
            s_str += f'\n  - # of noise files: {len(self.df_n)}'
            s_str += f'\n  - Total Duration: {duration(sum(self.df_n.duration))}'
            _ = f'uniform({self.mixture_snr_min}, {self.mixture_snr_max})'
            if self.mixture_snr_min == self.mixture_snr_max:
                _ = f'{self.mixture_snr_min}'
            s_str += f'\n  - SNRs: {_}'
        return s_str

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
