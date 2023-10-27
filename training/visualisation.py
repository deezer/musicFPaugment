from typing import Any, Optional

import librosa  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
from librosa import display

from training.parameters import WAVEFORM_SAMPLING_RATE


def spectrogram(waveform: torch.Tensor, amplitude=False, device="cpu") -> Any:
    """
    STFT layer used by the DL model. This function uses torchaudio STFT algorithm and is obtained
    using the same parameters as the STFT used in Audfprint
    """
    window = torch.tensor(np.hanning(512 + 2)[1:-1]).to(device)

    specgram = torch.stft(
        waveform,
        n_fft=512,
        hop_length=256,
        window=window,
        return_complex=True,
    ).to(device)

    specgram = torch.absolute(specgram).to(device)
    specgram /= torch.max(specgram).to(device)

    if amplitude == True:
        specgram = librosa.amplitude_to_db(
            specgram.cpu().squeeze(0).detach().numpy(), ref=np.max
        )

    return specgram


def plot_spectrogram(
    spec: npt.NDArray[np.float32], save_path: Optional[str] = None, amplitude=False
) -> Any:
    if amplitude == True:
        spec = librosa.amplitude_to_db(
            spec.cpu().squeeze(0).detach().numpy(), ref=np.max
        )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = display.specshow(
        spec,
        sr=WAVEFORM_SAMPLING_RATE,
        hop_length=256,
        n_fft=512,
        x_axis="time",
        y_axis="fft",
        ax=ax,
    )
    ax.set_title("STFT spectrum")
    _ = plt.colorbar(im, ax=ax, format="%+2.0f dB")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return fig
