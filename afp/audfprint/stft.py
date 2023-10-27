"""
Provide stft to avoid librosa dependency.
This implementation is based on routines from
https://github.com/tensorflow/models/blob/master/research/audioset/mel_features.py
"""

# from __future__ import division

from typing import Literal, Optional

import numpy as np
import numpy.typing as npt


def stft(
    signal: npt.NDArray[np.float32],
    n_fft: int,
    hop_length: Optional[int] = None,
    window: Optional[npt.NDArray[np.float32]] = None,
) -> npt.NDArray[np.complex128]:
    """
    Calculate the short-time Fourier transform.
    ---
    Args:
        signal (npt.NDArray[np.float32]):
            1D np.array of the input time-domain signal.
        n_fft (int):
            Size of the FFT to apply.
        hop_length (Optional[int]):
            Advance (in samples) between each frame passed to FFT. Defaults to half the window length.
        window (Optional[npt.NDArray[np.float32]]):
            Length of each block of samples to pass to FFT, or vector of window values.

    Returns:
        npt.NDArray[np.complex128]:
              2D np.array where each column contains the complex values of the fft_length/2+1 unique values
              of the FFT for the corresponding frame of input samples ("spectrogram transposition").
    """
    if window is None:
        window_length = n_fft
        window = np.hanning(window_length + 2)[1:-1]
    else:
        window_length = len(window)

    if hop_length is None:
        hop_length = window_length // 2

    # Default librosa STFT behavior.
    pad_mode: Literal["reflect"] = "reflect"

    signal = np.pad(signal, (n_fft // 2), mode=pad_mode)
    num_samples = signal.shape[0]
    num_frames = 1 + ((num_samples - window_length) // hop_length)
    shape = (num_frames, window_length) + signal.shape[1:]
    strides = (signal.strides[0] * hop_length,) + signal.strides
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    # Apply frame window to each frame. We use a periodic Hann (cosine of period
    # window_length) instead of the symmetric Hann of np.hanning (period
    # window_length-1).
    windowed_frames = frames * window
    return np.fft.rfft(windowed_frames, n_fft).transpose()
