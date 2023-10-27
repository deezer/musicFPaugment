from typing import Any, Optional

import julius
import torch
from torch import Tensor

from augmentation.transform import BaseWaveformTransform
from augmentation.utils import (
    ObjectDict,
    convert_frequencies_to_mels,
    convert_mels_to_frequencies,
)


class BandPassFilter(BaseWaveformTransform):
    """
    Apply band-pass filtering to the input audio.
    """

    supported_modes = {"per_batch", "per_example"}

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        min_center_frequency: int = 200,
        max_center_frequency: int = 4000,
        min_bandwidth_fraction: float = 0.5,
        max_bandwidth_fraction: float = 1.99,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        min_center_frequency (int):
            Minimum center frequency in hertz
        max_center_frequency (int):
            Maximum center frequency in hertz
        min_bandwidth_fraction (float):
            Minimum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        max_bandwidth_fraction (float):
            Maximum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        """
        super().__init__(
            p=p,
            sample_rate=sample_rate,
        )

        self.min_center_frequency = min_center_frequency
        self.max_center_frequency = max_center_frequency
        self.min_bandwidth_fraction = min_bandwidth_fraction
        self.max_bandwidth_fraction = max_bandwidth_fraction

        if max_center_frequency < min_center_frequency:
            raise ValueError(
                f"max_center_frequency ({max_center_frequency}) should be larger than "
                f"min_center_frequency ({min_center_frequency})."
            )

        if min_bandwidth_fraction <= 0.0:
            raise ValueError("min_bandwidth_fraction must be a positive number")

        if max_bandwidth_fraction < min_bandwidth_fraction:
            raise ValueError(
                f"max_bandwidth_fraction ({max_bandwidth_fraction}) should be larger than "
                f"min_bandwidth_fraction ({min_bandwidth_fraction})."
            )

        if max_bandwidth_fraction >= 2.0:
            raise ValueError(
                f"max_bandwidth_fraction ({max_bandwidth_fraction}) should be smaller than 2.0,"
                f"since otherwise low_cut_frequency of the band can be smaller than 0 Hz."
            )

    def randomize_parameters(
        self,
        samples: Tensor,
    ) -> None:
        """
        samples (Tensor):
            (batch_size, num_channels, num_samples)
        """

        batch_size, _, _ = samples.shape

        # Sample frequencies uniformly in mel space, then convert back to frequency
        def get_dist(min_freq: float, max_freq: float) -> Any:
            dist = torch.distributions.Uniform(  # type: ignore
                low=convert_frequencies_to_mels(
                    torch.tensor(
                        min_freq,
                        dtype=torch.float32,
                        device=samples.device,
                    )
                ),
                high=convert_frequencies_to_mels(
                    torch.tensor(
                        max_freq,
                        dtype=torch.float32,
                        device=samples.device,
                    )
                ),
                validate_args=True,
            )
            return dist

        center_dist = get_dist(self.min_center_frequency, self.max_center_frequency)
        self.transform_parameters["center_freq"] = convert_mels_to_frequencies(
            center_dist.sample(sample_shape=(batch_size,))
        )

        bandwidth_dist = torch.distributions.Uniform(  # type: ignore
            low=torch.tensor(
                self.min_bandwidth_fraction, dtype=torch.float32, device=samples.device
            ),
            high=torch.tensor(
                self.max_bandwidth_fraction, dtype=torch.float32, device=samples.device
            ),
        )
        self.transform_parameters["bandwidth"] = bandwidth_dist.sample(
            sample_shape=(batch_size,)  # type: ignore
        )

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        batch_size, _, _ = samples.shape

        low_cutoffs_as_fraction_of_sample_rate = (
            self.transform_parameters["center_freq"]
            * (1 - 0.5 * self.transform_parameters["bandwidth"])
            / sample_rate
        )
        high_cutoffs_as_fraction_of_sample_rate = (
            self.transform_parameters["center_freq"]
            * (1 + 0.5 * self.transform_parameters["bandwidth"])
            / sample_rate
        )

        for i in range(batch_size):
            samples[i] = julius.bandpass_filter(  # type: ignore
                samples[i],
                cutoff_low=low_cutoffs_as_fraction_of_sample_rate[i].item(),
                cutoff_high=high_cutoffs_as_fraction_of_sample_rate[i].item(),
                fft=False,
            )

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )


class BandStopFilter(BandPassFilter):
    """
    Apply band-stop filtering to the input audio. Also known as notch filter,
    band reject filter and frequency mask.
    """

    def __init__(
        self,
        min_center_frequency: int = 200,
        max_center_frequency: int = 4000,
        min_bandwidth_fraction: float = 0.5,
        max_bandwidth_fraction: float = 1.99,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ):
        """
        min_center_frequency (int):
            Minimum center frequency in hertz
        max_center_frequency (int):
            Maximum center frequency in hertz
        min_bandwidth_fraction (float):
            Minimum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        max_bandwidth_fraction (float):
            Maximum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        """

        super().__init__(
            min_center_frequency,
            max_center_frequency,
            min_bandwidth_fraction,
            max_bandwidth_fraction,
            p=p,
            sample_rate=sample_rate,
        )

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        perturbed = super().apply_transform(
            samples.clone(),
            sample_rate,
        )

        perturbed.samples = samples - perturbed.samples
        return perturbed
