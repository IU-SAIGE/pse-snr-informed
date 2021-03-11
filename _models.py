from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid.losses.mse import SingleSrcMSE
from asteroid.losses.sdr import SingleSrcNegSDR
from asteroid.losses.stoi import NegSTOILoss as SingleSrcNegSTOI
from asteroid.models.conv_tasnet import ConvTasNet
from torch.nn.modules.loss import _Loss


EPS = 1e-30


class PredictorGRU(nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.fft_size: int = 1024
        self.hop_length: int = 256

        # layers
        self.rnn = nn.GRU(
            input_size=int(self.fft_size // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dnn = nn.Linear(
            in_features=self.hidden_size,
            out_features=1
        )
        self.window = nn.Parameter(torch.hann_window(self.fft_size), False)
        self.name = (self.__class__.__name__ +
                     f'_{hidden_size:04d}x{num_layers:02d}')

    def stft(
        self,
        waveform: torch.Tensor
    ):
        """Calculates the Short-time Fourier transform (STFT)."""

        # perform the short-time Fourier transform
        spectrogram = torch.stft(
            waveform, self.fft_size, self.hop_length, window=self.window,
            return_complex=False
        )

        # swap seq_len & feature_dim of the spectrogram (for RNN processing)
        spectrogram = spectrogram.permute(0, 2, 1, 3)

        # calculate the magnitude spectrogram
        magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 +
                                           spectrogram[..., 1] ** 2)

        return magnitude_spectrogram

    def forward(self, waveform):
        X_magnitude = self.stft(waveform)
        y = self.dnn(self.rnn(X_magnitude)[0])
        predicted_snrs = y.reshape(-1, X_magnitude.shape[1])

        return predicted_snrs


class DenoiserGRU(nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.fft_size: int = 1024
        self.hop_length: int = 256

        # create a neural network which predicts a TF binary ratio mask
        self.encoder = nn.GRU(
            input_size=int(self.fft_size // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_size,
                out_features=int(self.fft_size // 2 + 1)
            ),
            nn.Sigmoid()
        )
        self.window = nn.Parameter(torch.hann_window(self.fft_size), False)
        self.name = (self.__class__.__name__ +
                     f'_{hidden_size:04d}x{num_layers:02d}')

    def stft(
        self,
        waveform: torch.Tensor
    ):
        """Calculates the Short-time Fourier transform (STFT)."""

        # perform the short-time Fourier transform
        spectrogram = torch.stft(
            waveform, self.fft_size, self.hop_length, window=self.window,
            return_complex=False
        )

        # swap seq_len & feature_dim of the spectrogram (for RNN processing)
        spectrogram = spectrogram.permute(0, 2, 1, 3)

        # calculate the magnitude spectrogram
        magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 +
                                           spectrogram[..., 1] ** 2)

        return spectrogram, magnitude_spectrogram

    def istft(
        self,
        spectrogram: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """Calculates the inverse Short-time Fourier transform (ISTFT)."""

        # apply a time-frequency mask if provided
        if mask is not None:
            spectrogram[..., 0] *= mask
            spectrogram[..., 1] *= mask

        # swap seq_len & feature_dim of the spectrogram (undo RNN processing)
        spectrogram = spectrogram.permute(0, 2, 1, 3)

        # perform the inverse short-time Fourier transform
        waveform = torch.istft(
            spectrogram, self.fft_size, self.hop_length, window=self.window
        )

        return waveform

    def forward(self, waveform):
        # convert waveform to spectrogram
        (X, X_magnitude) = self.stft(waveform)

        # generate a time-frequency mask
        H = self.encoder(X_magnitude)[0]
        Y = self.decoder(H)
        Y = Y.reshape_as(X_magnitude)

        # convert masked spectrogram back to waveform
        denoised = self.istft(X, mask=Y)
        residual = self.istft(X.clone(), mask=(1 - Y.clone()))

        return denoised, residual


class DenoiserCTN(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = ConvTasNet(n_src=2,
            n_filters=128,       # N
                                 # L?
            bn_chan=128,         # B
            hid_chan=256,        # H
            skip_chan=128,       # Sc
            conv_kernel_size=3,  # P
            n_blocks=7,          # X
            n_repeats=2,         # R
            sample_rate=16000,
        )
        self.name = self.__class__.__name__

    def forward(self, waveform):
        output = self.network(waveform)
        denoised = output[..., 0, :]
        residual = output[..., 1, :]
        return denoised, residual


class SDRLoss(_Loss):
    '''Loss function based on SI-SDR.'''

    def __init__(
        self,
        sdr_type: str = 'sisdr',
        reduction: str = 'none'
    ):
        super().__init__(reduction=reduction)
        assert sdr_type in ('snr', 'sisdr', 'sdsdr')
        self.sdr_type = sdr_type

    def forward(self, estimate, target):
        assert target.size() == estimate.size()

        # subtract signal means
        mean_source = torch.mean(target, dim=1, keepdim=True)
        mean_estimate = torch.mean(estimate, dim=1, keepdim=True)
        target = target - mean_source
        estimate = estimate - mean_estimate

        # SDR numerator
        if self.sdr_type != 'snr':
            dot = torch.sum(estimate * target, dim=1, keepdim=True)
            s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + EPS
            scaled_target = dot * target / s_target_energy
        else:
            scaled_target = target

        # SDR denominator
        if self.sdr_type != 'sisdr':
            e_noise = estimate - target
        else:
            e_noise = estimate - scaled_target

        losses = torch.sum(scaled_target ** 2, dim=1)
        losses = losses / (torch.sum(e_noise ** 2, dim=1) + EPS)
        losses = 10 * torch.log10(losses + EPS)
        losses = losses.mean() if self.reduction == 'mean' else losses

        return -losses


class SegSDRLoss(_Loss):
    '''Loss function based on SI-SDR segmented frame by frame.'''

    def __init__(
        self,
        sdr_type: str = 'sisdr',
        reduction: str = 'none',
        segment_size: int = 1024,
        hop_length: int = 256,
        center: bool = True,
        pad_mode: str = 'reflect'
    ):
        super().__init__(reduction=reduction)
        assert sdr_type in ('snr', 'sisdr', 'sdsdr')
        assert pad_mode in ('constant', 'reflect')
        assert isinstance(center, bool)
        assert segment_size > hop_length > 0

        self.sdr_type = sdr_type
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.window = torch.hann_window(self.segment_size).view(1, 1, -1)

    def forward(self, estimate, target, weights=None):
        assert target.size() == estimate.size()
        assert target.ndim == 2
        assert self.segment_size < target.size()[-1]

        # subtract signal means
        mean_source = torch.mean(target, dim=1, keepdim=True)
        mean_estimate = torch.mean(estimate, dim=1, keepdim=True)
        target = target - mean_source
        estimate = estimate - mean_estimate

        if self.center:
            signal_dim = target.dim()
            ext_shape = [1] * (3 - signal_dim) + list(target.size())
            p = int(self.segment_size // 2)
            target = F.pad(target.view(ext_shape), [p, p], self.pad_mode)
            target = target.view(target.shape[-signal_dim:])
            estimate = F.pad(estimate.view(ext_shape), [p, p], self.pad_mode)
            estimate = estimate.view(estimate.shape[-signal_dim:])

        # use stride tricks to construct overlapping frames out of inputs
        (n_batch, n_samples) = target.size()
        n_frames = (n_samples - self.segment_size) // self.hop_length + 1
        target = torch.as_strided(
            target,
            size=(n_batch, n_frames, self.segment_size),
            stride=(n_samples, self.hop_length, 1))
        estimate = torch.as_strided(
            estimate,
            size=(n_batch, n_frames, self.segment_size),
            stride=(n_samples, self.hop_length, 1))

        # window all the frames
        target *= self.window
        estimate *= self.window

        # apply weighting (if provided)
        if weights is not None:
            assert weights.numel() == n_frames
            weights = weights.view(1, -1, 1)
            target *= weights
            estimate *= weights

        # collapse batch and time axes
        target = target.reshape(-1, self.segment_size)
        estimate = estimate.reshape(-1, self.segment_size)

        # SDR numerator
        if self.sdr_type != 'snr':
            dot = torch.sum(estimate * target, dim=1, keepdim=True)
            s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + EPS
            scaled_target = dot * target / s_target_energy
        else:
            scaled_target = target

        # SDR denominator
        if self.sdr_type != 'sisdr':
            e_noise = estimate - target
        else:
            e_noise = estimate - scaled_target

        losses = torch.sum(scaled_target ** 2, dim=1)
        losses = losses / (torch.sum(e_noise ** 2, dim=1) + EPS)
        losses = 10 * torch.log10(losses + EPS)
        losses = losses.view(n_batch, -1)
        losses = losses.mean() if self.reduction == 'mean' else losses

        return -losses
