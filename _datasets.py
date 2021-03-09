import os
from typing import List, Optional, Set, Tuple, Union

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
    eps = 1e-30

    for i in range(0, s.shape[-1]-segment_size, hop_length):
        s_i = s[..., i:i+segment_size] * w
        r_i = r[..., i:i+segment_size] * w
        s_i = (s_i**2).sum(axis=-1) + eps
        r_i = (r_i**2).sum(axis=-1) + eps
        snr_i = 10*np.log10(s_i/r_i)
        seg_snrs.append(snr_i)

    return np.stack(seg_snrs).T


## experiment dataloaders

class SpeakerAgnosticMixtures(IterableDataset):

    def __init__(
        self,
        speaker_ids: Union[List[str], Set[str]],
        utterance_duration: Optional[float] = 1.,  # seconds
        mixture_snr: Union[float, Tuple[float, float]] = (-5., 10.),  # dB
        seg_snrs: bool = False
    ):
        self.rng = np.random.default_rng(0)
        (self.s_idx, self.n_idx) = (-1, -1)

        self.speaker_ids = speaker_ids
        if isinstance(mixture_snr, Tuple):
            self.mixture_snr_min = min(mixture_snr)
            self.mixture_snr_max = max(mixture_snr)
        else:
            self.mixture_snr_min = self.mixture_snr_max = mixture_snr
        self.df_s = librispeech.query(f'speaker_id in {speaker_ids}')
        self.df_n = fsd50k.query(f'split == "train"')
        self.utterance_duration = utterance_duration
        self.seg_snrs = seg_snrs

    def __iter__(self):
        return self

    def reset(self):
        self.rng = np.random.default_rng(0)
        (self.s_idx, self.n_idx) = (-1, -1)

    def __next__(self):

        # increment pointers
        self.s_idx = (self.s_idx + 1) % len(self.df_s)
        self.n_idx = (self.n_idx + 1) % len(self.df_n)

        length = int(self.utterance_duration * sample_rate)
        offset_s = self.df_s.max_offset.iloc[self.s_idx]
        offset_n = self.df_n.max_offset.iloc[self.n_idx]
        if offset_s > 0:
            offset_s = self.rng.integers(0, offset_s)
        if offset_n > 0:
            offset_n = self.rng.integers(0, offset_n)

        # read speech file, offset and truncate, then normalize
        (_, s) = wavfile.read(self.df_s.filepath.iloc[self.s_idx])
        s = s[offset_s:offset_s+length]
        s = s / (1e-8 + s.std())

        # read noise file, offset and truncate, then normalize
        (_, n) = wavfile.read(self.df_n.filepath.iloc[self.n_idx])
        n = n[offset_n:offset_n+length]
        n = n / (1e-8 + n.std())

        # mix the signals
        snr = self.rng.uniform(self.mixture_snr_min, self.mixture_snr_max)
        x = s + (n * 10**(-snr/20.))

        # create output tuple
        scale_factor = 1e-8 + max(abs(x.min()), abs(x.max()))

        if self.seg_snrs:
            sample = (
                torch.Tensor(x) / scale_factor,
                torch.Tensor(segmental_snr(estimate=x, target=s))
            )
        else:
            sample = (
                torch.Tensor(x) / scale_factor,
                torch.Tensor(s) / scale_factor,
                torch.Tensor(n) / scale_factor,
            )

        return sample


class SpeakerSpecificMixtures(IterableDataset):

    def __init__(
            self,
            speaker_id: str,
            speech_subset: str,
            noise_subset: str = 'free-sound',
            dataset_duration: Optional[int] = None,
            utterance_duration: Optional[int] = 1,
            premixture_snr: Optional[Union[float, Tuple[float, float]]] = None,
            mixture_snr: Union[float, Tuple[float, float]] = (-5, 5)
    ):
        # sanity check for inputs
        if noise_subset not in musan.set_id.unique():
            raise ValueError(f'Invalid noise subset \'{noise_subset}\'. '
                             f'Allowed values: {set(musan.set_id.unique())}.')
        if speech_subset not in librispeech.partition.unique():
            raise ValueError(f'Invalid speech subset \'{speech_subset}\'. '
                             f'Allowed values: '
                             f'{set(librispeech.partition.unique())}.')
        if speaker_id not in librispeech.speaker_id.unique():
            raise ValueError(f'Invalid LibriSpeech speaker ID \'{speaker_id}\'.')

        self.speaker_id = speaker_id
        self.speech_subset = speech_subset
        self.rng = np.random.default_rng(0)
        self.m_idx = -1
        self.n_idx = -1

        # unpack mixture SNR values
        if isinstance(mixture_snr, Tuple):
            self.mixture_snr_min = min(mixture_snr)
            self.mixture_snr_max = max(mixture_snr)
        else:
            self.mixture_snr_min = self.mixture_snr_max = mixture_snr
        if isinstance(premixture_snr, Tuple):
            self.premixture_snr_min = min(premixture_snr)
            self.premixture_snr_max = max(premixture_snr)
        else:
            self.premixture_snr_min = self.premixture_snr_max = premixture_snr

        # load internal corpora
        self.df_s = librispeech.query(f'speaker_id == "{speaker_id}" and ' +
                                      f'partition == "{speech_subset}"')
        self.df_m = demand
        self.df_n = musan.query(f'set_id == "{noise_subset}"')

        # pre-load speech data
        self.speech = []
        for filepath in self.df_s.filepath.tolist():
            (_, s) = wavfile.read(filepath)
            self.speech.append(s)
        self.speech = np.concatenate(self.speech)
        if dataset_duration:
            self.speech = self.speech[:(dataset_duration * sample_rate)]
        self.speech = self.speech / (1e-8 + self.speech.std())
        self.utterance_duration = utterance_duration
        self.max_offset = len(self.speech) - (utterance_duration * sample_rate)

    def __iter__(self):
        return self

    def reset(self):
        self.rng = np.random.default_rng(0)
        self.m_idx = -1
        self.n_idx = -1

    def __next__(self):

        # increment pointers
        self.m_idx = (self.m_idx + 1) % len(self.df_m)
        self.n_idx = (self.n_idx + 1) % len(self.df_n)

        length = self.utterance_duration * sample_rate
        offset_s = 0
        if self.max_offset > 0:
            offset_s = self.rng.integers(0, self.max_offset)
        offset_m = self.rng.integers(0, self.df_m.max_offset.iloc[self.m_idx])
        offset_n = self.rng.integers(0, self.df_n.max_offset.iloc[self.n_idx])

        # slice from speech array, offset and truncate, then normalize
        s = self.speech[offset_s:offset_s+length]
        s = s / (1e-8 + s.std())
        p = s

        # add premixture
        if self.premixture_snr_min is not None:
            (_, m) = wavfile.read(self.df_m.filepath.iloc[self.m_idx])
            m = m[offset_m:offset_m+length]
            m = m / (1e-8 + m.std())
            snr = self.rng.uniform(self.premixture_snr_min,
                                   self.premixture_snr_max)
            p = s + (m * 10 ** (-snr / 20.))

        # read noise file, offset and truncate, then normalize
        (_, n) = wavfile.read(self.df_n.filepath.iloc[self.n_idx])
        n = n[offset_n:offset_n+length]
        n = n / (1e-8 + n.std())

        # mix the signals
        snr = self.rng.uniform(self.mixture_snr_min, self.mixture_snr_max)
        x = p + (n * 10 ** (-snr / 20.))

        # create output tuple
        scale_factor = 1e-8 + max(abs(x.min()), abs(x.max()))
        sample = (
            torch.Tensor(x) / scale_factor,  # noise-injected premixture
            torch.Tensor(p) / scale_factor,  # premixture (or clean speech)
            torch.Tensor(n) / scale_factor,  # added noise
        )

        return sample


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
