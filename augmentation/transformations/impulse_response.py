import random
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.fft import irfft, rfft
from torch.nn.utils.rnn import pad_sequence

from augmentation.transform import BaseWaveformTransform, EmptyPathException
from augmentation.utils import Audio, ObjectDict, find_audio_files_in_paths


class ApplyImpulseResponse(BaseWaveformTransform):
    """
    Convolve the given audio with impulse responses.
    """

    # Note: This transform has only partial support for multichannel audio. IRs that are not
    # mono get mixed down to mono before they are convolved with all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        ir_paths: List[str],
        sample_rate: int,
        convolve_mode: str = "full",
        compensate_for_propagation_delay: bool = False,
        p: float = 0.5,
    ) -> None:
        """
        ir_paths (List[Path]):
            Either a path to a folder with audio files or a list of paths to audio files.
        compensate_for_propagation_delay (str):
            Convolving audio with a RIR normally
            introduces a bit of delay, especially when the peak absolute amplitude in the
            RIR is not in the very beginning. When compensate_for_propagation_delay is
            set to True, the returned slices of audio will be offset to compensate for
            this delay.
        """

        super().__init__(
            p=p,
            sample_rate=sample_rate,
        )

        # TODO: check that one can read audio files
        self.ir_paths = find_audio_files_in_paths(ir_paths)

        self.audio = Audio(sample_rate=sample_rate, mono=True)

        if len(self.ir_paths) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        self.convolve_mode = convolve_mode
        self.compensate_for_propagation_delay = compensate_for_propagation_delay

    def randomize_parameters(
        self,
        samples: Tensor,
    ) -> None:
        batch_size, _, _ = samples.shape
        random_ir_paths = random.choices(self.ir_paths, k=batch_size)

        self.transform_parameters["ir"] = pad_sequence(
            [self.audio(ir_path).transpose(0, 1) for ir_path in random_ir_paths],
            batch_first=True,
            padding_value=0.0,
        ).transpose(1, 2)

        self.transform_parameters["ir_paths"] = random_ir_paths

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        _, num_channels, num_samples = samples.shape

        ir = self.transform_parameters["ir"].to(samples.device)

        convolved_samples = convolve(
            samples, ir.expand(-1, num_channels, -1), mode=self.convolve_mode
        )

        ## Normalize output :
        peak_values = torch.max(torch.abs(convolved_samples), dim=2).values
        peak_values = peak_values.expand(-1, convolved_samples.shape[2]).reshape(
            convolved_samples.shape[0], 1, convolved_samples.shape[2]
        )
        convolved_samples /= peak_values

        if self.compensate_for_propagation_delay:
            propagation_delays = ir.abs().argmax(dim=2, keepdim=False)[:, 0]
            convolved_samples = torch.stack(
                [
                    convolved_sample[
                        :, propagation_delay : propagation_delay + num_samples
                    ]
                    for convolved_sample, propagation_delay in zip(
                        convolved_samples, propagation_delays
                    )
                ],
                dim=0,
            )

            return ObjectDict(
                samples=convolved_samples,
                sample_rate=sample_rate,
            )

        else:
            return ObjectDict(
                samples=convolved_samples[..., :num_samples],
                sample_rate=sample_rate,
            )


def convolve(signal: torch.Tensor, kernel: torch.Tensor, mode: str = "full") -> Any:
    """
    Computes the 1-d convolution of signal by kernel using FFTs.
    The two arguments should have the same rightmost dim, but may otherwise be
    arbitrarily broadcastable.

    Note: This function was originally copied from the https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits

    signal (torch.Tensor):
        A signal to convolve.
    kernel (torch.Tensor):
        A convolution kernel.
    mode (str):
        One of: 'full', 'valid', 'same'.

    Return (torch.Tensor):
        A tensor with broadcasted shape. Letting ``m = signal.size(-1)``
        and ``n = kernel.size(-1)``, the rightmost size of the result will be:
        ``m + n - 1`` if mode is 'full';
        ``max(m, n) - min(m, n) + 1`` if mode is 'valid'; or
        ``max(m, n)`` if mode is 'same'.
    """
    m = signal.size(-1)
    n = kernel.size(-1)
    if mode == "full":
        truncate = m + n - 1
    elif mode == "valid":
        truncate = max(m, n) - min(m, n) + 1
    elif mode == "same":
        truncate = max(m, n)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    # Compute convolution using fft.
    padded_size = m + n - 1
    # Round up for cheaper fft.
    fast_ftt_size = next_fast_len(padded_size)
    f_signal = rfft(signal, n=fast_ftt_size)
    f_kernel = rfft(kernel, n=fast_ftt_size)
    f_result = f_signal * f_kernel
    result = irfft(f_result, n=fast_ftt_size)

    start_idx = (padded_size - truncate) // 2
    return result[..., start_idx : start_idx + truncate]


_NEXT_FAST_LEN: Dict[int, Any] = {}


def next_fast_len(size: int) -> Any:
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Note: This function was originally copied from the https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits

    size (int):
        A positive number.

    Returns (int):
        A possibly larger number.
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1
