from typing import Optional

import julius
import torch
from torch import Tensor

from augmentation.transform import BaseWaveformTransform
from augmentation.utils import (
    ObjectDict,
    convert_frequencies_to_mels,
    convert_mels_to_frequencies,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LowPassFilter(BaseWaveformTransform):
    """
    Apply low-pass filtering to the input audio.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        min_cutoff_freq: float = 150.0,
        max_cutoff_freq: float = 7500.0,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        min_cutoff_freq (int):
            Minimum cutoff frequency in hertz
        max_cutoff_freq (int):
            Maximum cutoff frequency in hertz
        """
        super().__init__(
            p=p,
            sample_rate=sample_rate,
        )

        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        if self.min_cutoff_freq > self.max_cutoff_freq:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")

    def randomize_parameters(
        self,
        samples: Tensor,
    ) -> None:
        """
        samples:
            (batch_size, num_channels, num_samples)
        """
        batch_size, _, _ = samples.shape

        # Sample frequencies uniformly in mel space, then convert back to frequency
        dist = torch.distributions.Uniform(  # type: ignore
            low=torch.ceil(
                convert_frequencies_to_mels(
                    torch.tensor(
                        self.min_cutoff_freq,
                        dtype=torch.float32,
                        device=samples.device,
                    )
                )
            ),
            high=torch.floor(
                convert_frequencies_to_mels(
                    torch.tensor(
                        self.max_cutoff_freq,
                        dtype=torch.float32,
                        device=samples.device,
                    )
                )
            ),
            validate_args=True,
        )
        self.transform_parameters["cutoff_freq"] = convert_mels_to_frequencies(
            dist.sample(sample_shape=(batch_size,))  # type: ignore
        )

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        batch_size, _, _ = samples.shape

        if sample_rate is None:
            sample_rate = self.sample_rate

        cutoffs_as_fraction_of_sample_rate = (
            self.transform_parameters["cutoff_freq"] / sample_rate
        )
        # TODO: Instead of using a for loop, perform batched compute to speed things up
        for i in range(batch_size):
            try:
                samples[i] = julius.lowpass_filter(  # type: ignore
                    samples[i], cutoffs_as_fraction_of_sample_rate[i].item(), fft=False
                )
            except ValueError:
                raise ValueError(
                    "Buggy cutoff freq. {}. \nRest of vector:\n{} \nCutoff freq.: {}".format(
                        cutoffs_as_fraction_of_sample_rate[i].item(),
                        cutoffs_as_fraction_of_sample_rate,
                        self.transform_parameters["cutoff_freq"],
                    )
                )

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )


class HighPassFilter(LowPassFilter):
    """
    Apply high-pass filtering to the input audio.
    """

    def __init__(
        self,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2400.0,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        min_cutoff_freq (int):
            Minimum cutoff frequency in hertz
        max_cutoff_freq (int):
            Maximum cutoff frequency in hertz
        """

        super().__init__(
            min_cutoff_freq,
            max_cutoff_freq,
            p=p,
            sample_rate=sample_rate,
        )

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        perturbed = super().apply_transform(
            samples=samples.clone(),
            sample_rate=sample_rate,
        )

        perturbed.samples = samples - perturbed.samples
        return perturbed
