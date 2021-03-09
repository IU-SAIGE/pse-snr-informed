from typing import Optional

import torch
import torch.nn as nn
from asteroid.losses.mse import SingleSrcMSE
from asteroid.losses.sdr import SingleSrcNegSDR
from asteroid.losses.stoi import NegSTOILoss as SingleSrcNegSTOI
from asteroid.models.conv_tasnet import ConvTasNet


class PredictorGRU(nn.Module):

    # STFT parameters
    fft_size: int = 1024
    hop_length: int = 256

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
    # STFT parameters
    fft_size: int = 1024
    hop_length: int = 256

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
