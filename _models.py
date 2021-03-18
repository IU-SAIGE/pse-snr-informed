from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid.models.conv_tasnet import ConvTasNet
from torch.nn.modules.loss import _Loss


DEVICE: str = ('cuda:0' if torch.cuda.is_available() else 'cpu')
FFT_SIZE: int = 1024
HOP_SIZE: int = 256
WINDOW: torch.Tensor = torch.hann_window(FFT_SIZE).to(DEVICE)
EPS: float = 1e-8


def stft(waveform: torch.Tensor):
    """Calculates the Short-time Fourier transform (STFT)."""

    # perform the short-time Fourier transform
    spectrogram = torch.stft(
        waveform, FFT_SIZE, HOP_SIZE, window=WINDOW, return_complex=False
    )

    # swap seq_len & feature_dim of the spectrogram (for RNN processing)
    spectrogram = spectrogram.permute(0, 2, 1, 3)

    # calculate the magnitude spectrogram
    magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 +
                                       spectrogram[..., 1] ** 2)

    return (spectrogram, magnitude_spectrogram)


def istft(spectrogram: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """Calculates the inverse Short-time Fourier transform (ISTFT)."""

    # apply a time-frequency mask if provided
    if mask is not None:
        spectrogram[..., 0] *= mask
        spectrogram[..., 1] *= mask

    # swap seq_len & feature_dim of the spectrogram (undo RNN processing)
    spectrogram = spectrogram.permute(0, 2, 1, 3)

    # perform the inverse short-time Fourier transform
    waveform = torch.istft(
        spectrogram, FFT_SIZE, HOP_SIZE, window=WINDOW
    )

    return waveform


class PredictorGRU(nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers

        # layers
        self.rnn = nn.GRU(
            input_size=int(FFT_SIZE//2+1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dnn = nn.Linear(
            in_features=self.hidden_size,
            out_features=1
        )
        self.name = (self.__class__.__name__ +
                     f'_{hidden_size:04d}x{num_layers:02d}')

    def forward(self, waveform: torch.Tensor):

        # convert to time-frequency domain
        (_, magnitude_spectrogram) = stft(waveform)

        # generate frame-by-frame SNR predictions
        predicted_snrs = self.dnn(self.rnn(magnitude_spectrogram)[0]).reshape(
            -1, magnitude_spectrogram.shape[1])

        return predicted_snrs


class DenoiserGRU(nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers

        # create a neural network which predicts a TF binary ratio mask
        self.encoder = nn.GRU(
            input_size=int(FFT_SIZE//2+1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_size,
                out_features=int(FFT_SIZE//2+1)
            ),
            nn.Sigmoid()
        )
        self.name = (self.__class__.__name__ +
                     f'_{hidden_size:04d}x{num_layers:02d}')

    def forward(self, waveform: torch.Tensor):

        # convert to time-frequency domain
        (spectrogram, magnitude_spectrogram) = stft(waveform)

        # generate a time-frequency mask
        predicted_mask = self.decoder(self.encoder(magnitude_spectrogram)[0])
        predicted_mask = predicted_mask.reshape_as(magnitude_spectrogram)

        # convert back to time domain
        estimate = istft(spectrogram, predicted_mask)

        return estimate


class SegmentalLoss(_Loss):
    '''Loss function applied to audio segmented frame by frame.'''

    def __init__(
        self,
        loss_type: str = 'sisdr',
        reduction: str = 'none',
        segment_size: int = 1024,
        hop_length: int = 256,
        windowing: bool = True,
        centering: bool = True,
        pad_mode: str = 'reflect'
    ):
        super().__init__(reduction=reduction)
        assert loss_type in ('mse', 'snr', 'sisdr', 'sdsdr')
        assert pad_mode in ('constant', 'reflect')
        assert isinstance(centering, bool)
        assert isinstance(windowing, bool)
        assert segment_size > hop_length > 0

        self.loss_type = loss_type
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.pad_mode = pad_mode

        self.centering = centering
        self.windowing = windowing

        self.unfold = nn.Unfold(
            kernel_size=(1, segment_size),
            stride=(1, hop_length)
        )
        self.window = torch.hann_window(self.segment_size).view(1, 1, -1)

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        assert target.size() == estimate.size()
        assert target.ndim == 2
        assert self.segment_size < target.size()[-1]

        # subtract signal means
        target -= torch.mean(target, dim=1, keepdim=True)
        estimate -= torch.mean(estimate, dim=1, keepdim=True)

        # center the signals using padding
        if self.centering:
            signal_dim = target.dim()
            ext_shape = [1] * (3 - signal_dim) + list(target.size())
            p = int(self.segment_size // 2)
            target = F.pad(target.view(ext_shape), [p, p], self.pad_mode)
            target = target.view(target.shape[-signal_dim:])
            estimate = F.pad(estimate.view(ext_shape), [p, p], self.pad_mode)
            estimate = estimate.view(estimate.shape[-signal_dim:])

        # use unfold to construct overlapping frames out of inputs
        n_batch = target.size()[0]
        target = self.unfold(target.view(n_batch,1,1,-1)).permute(0,2,1)
        estimate = self.unfold(estimate.view(n_batch,1,1,-1)).permute(0,2,1)

        # window all the frames
        if self.windowing:
            self.window = self.window.to(target.device)
            target = torch.multiply(target, self.window)
            estimate = torch.multiply(estimate, self.window)

        # MSE loss
        if self.loss_type == 'mse':
            losses = ((target - estimate)**2).sum(dim=2)
            losses /= self.segment_size

        # SDR based loss
        else:

            if self.loss_type == 'snr':
                scaled_target = target
            else:
                dot = (estimate * target).sum(dim=2, keepdim=True)
                s_target_energy = (target ** 2).sum(dim=2, keepdim=True) + EPS
                scaled_target = dot * target / s_target_energy

            if self.loss_type == 'sisdr':
                e_noise = estimate - scaled_target
            else:
                e_noise = estimate - target

            losses = (scaled_target ** 2).sum(dim=2)
            losses = losses / ((e_noise ** 2).sum(dim=2) + EPS)
            losses = -10 * torch.log10(losses + EPS)

        # apply weighting (if provided)
        if weights is not None:
            assert losses.size() == weights.size()
            weights = weights.detach()
            losses = torch.multiply(losses, weights).mean(dim=1)

        if self.reduction == 'mean':
            losses = losses.mean()

        return losses
