from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from nnAudio import features  # type: ignore

from training.parameters import WAVEFORM_SAMPLING_RATE


def stft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Perform STFT and convert to magnitude spectrogram.
    ---
    Args:
        x (torch.Tensor):
            Input signal tensor (B, T).
        fft_size (int):
            FFT size.
        hop_size (int):
            Hop size.
        win_length (int):
            Window length.
        window (Optional[torch.Tensor]):
            Window function type.

    Returns:
        torch.Tensor:
            Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """
        Calculate forward propagation.
        ---
        Args:
            x_mag (torch.Tensor):
                Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (torch.Tensor):
                Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            torch.Tensor:
                Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")  # type: ignore


class LogSTFTMagnitudeLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """
        Calculate forward propagation.
        ---
        Args:
            x_mag (torch.Tensor):
                Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (torch.Tensor):
                Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            torch.Tensor:
                Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        window: str = "hann_window",
    ) -> None:
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate forward propagation.
        ---
        Args:
            x (torch.Tensor):
                Predicted signal (B, T).
            y (torch.Tensor):
                Groundtruth signal (B, T).

        Returns:
            torch.Tensor:
                Spectral convergence loss value.
            torch.Tensor:
                Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)  # type: ignore
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)  # type: ignore
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann_window",
        factor_sc: float = 0.1,
        factor_mag: float = 0.1,
    ) -> None:
        """
        Initialize Multi resolution STFT loss module.
        ---
        Args:
            fft_sizes (List[int]):
                List of FFT sizes.
            hop_sizes (List[int]):
                List of hop sizes.
            win_lengths (List[int]):
                List of window lengths.
            window (str):
                Window function type.
            factor (float):
                A balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate forward propagation.
        ---
        Args:
            x (torch.Tensor):
                Predicted signal (B, T).
            y (torch.Tensor):
                Groundtruth signal (B, T).

        Returns:
            float:
                Multi resolution spectral convergence loss value.
            float:
                Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss


def cqt(
    x: torch.Tensor, f_min: float, f_max: float, bins_per_octave: int, hop_length: int
) -> torch.Tensor:
    cqt_layer = features.CQT2010v2(
        sr=WAVEFORM_SAMPLING_RATE,
        hop_length=hop_length,
        fmin=f_min,
        fmax=f_max,
        bins_per_octave=bins_per_octave,
        verbose=False,
    ).to(x.get_device())

    return torch.clamp(cqt_layer(x), min=1e-7)


class LogCQTMagnitudeLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(LogCQTMagnitudeLoss, self).__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """
        Calculate forward propagation.
        ---
        Args:
            x_mag (torch.Tensor):
                Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (torch.Tensor):
                Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            torch.Tensor:
                CQT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class CQTLoss(torch.nn.Module):
    def __init__(
        self, f_min: float, f_max: float, bins_per_octave: int, hop_length: int
    ) -> None:
        super(CQTLoss, self).__init__()
        self.f_min = f_min
        self.f_max = f_max
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.cqt_magnitude_loss = LogCQTMagnitudeLoss()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate forward propagation.
        ---
        Args:
            x (torch.Tensor):
                Predicted signal (B, T).
            y (torch.Tensor):
                Groundtruth signal (B, T).

        Returns:
            torch.Tensor:
                Spectral convergence loss value.
            torch.Tensor:
                Log STFT magnitude loss value.
        """
        x_mag = cqt(x, self.f_min, self.f_max, self.bins_per_octave, self.hop_length)
        y_mag = cqt(y, self.f_min, self.f_max, self.bins_per_octave, self.hop_length)

        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.cqt_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionCQTLoss(torch.nn.Module):
    def __init__(
        self,
        f_min: List[float] = [32.70, 32.70, 32.70],
        f_max: List[float] = [4186.009, 4186.009, 4186.009],
        bins_per_octave: List[int] = [48, 36, 24],
        hop_sizes: List[int] = [512, 128, 64],
        factor_sc: float = 0.1,
        factor_mag: float = 0.1,
    ) -> None:
        """
        Initialize Multi resolution STFT loss module.
        ---
        Args:
            fft_sizes (List[float]):
                List of FFT sizes.
            hop_sizes (List[float]):
                List of hop sizes.
            win_lengths (List[int]):
                List of window lengths.
            window (str):
                Window function type.
            factor (float):
                A balancing factor across different losses.
        """
        super(MultiResolutionCQTLoss, self).__init__()
        assert len(f_min) == len(f_max) == len(bins_per_octave) == len(hop_sizes)
        self.cqt_losses = torch.nn.ModuleList()
        for f_mi, f_ma, bpo, hs in zip(f_min, f_max, bins_per_octave, hop_sizes):
            self.cqt_losses += [CQTLoss(f_mi, f_ma, bpo, hs)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate forward propagation.
        ---
        Args:
            x (torch.Tensor):
                Predicted signal (B, T).
            y (torch.Tensor):
                Ground truth signal (B, T).

        Returns:
            torch.Tensor:
                Multi resolution spectral convergence loss value.
            torch.Tensor:
                Multi resolution CQT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.cqt_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.cqt_losses)
        mag_loss /= len(self.cqt_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss
