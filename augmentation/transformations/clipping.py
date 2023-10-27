from typing import Optional

import torch
from torch import Tensor

from augmentation.transform import BaseWaveformTransform
from augmentation.utils import ObjectDict


class Clipping(BaseWaveformTransform):

    """
    Distort signal by clipping a random percentage of points

    The percentage of points that will be clipped is drawn from a uniform distribution between
    the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
    30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.
    """

    supports_multichannel = True
    requires_sample_rate = False

    def __init__(
        self,
        min_percentile_threshold: float = 0.0,
        max_percentile_threshold: float = 1.0,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> None:
        super().__init__(
            p=p,
            sample_rate=sample_rate,
        )
        assert 0 <= min_percentile_threshold
        assert 1 >= max_percentile_threshold
        assert min_percentile_threshold <= max_percentile_threshold

        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold
        if self.min_percentile_threshold >= self.max_percentile_threshold:
            raise ValueError(
                "max_percentile_threshold must be higher than min_percentile_threshold"
            )

    def randomize_parameters(
        self,
        samples: Tensor,
    ) -> None:
        distribution = torch.distributions.Uniform(  # type: ignore
            low=torch.tensor(
                self.min_percentile_threshold,
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.max_percentile_threshold,
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )
        selected_batch_size = samples.size(0)
        self.transform_parameters["percentile_threshold"] = distribution.sample(
            sample_shape=(selected_batch_size,)  # type: ignore
        ).unsqueeze(1)

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        lower_percentile_threshold = (
            self.transform_parameters["percentile_threshold"] / 2
        )

        lower_threshold = torch.transpose(
            torch.quantile(
                samples[:, 0, :], lower_percentile_threshold.reshape(-1)
            ).expand(samples.shape[2], samples.shape[0]),
            0,
            1,
        )

        upper_threshold = torch.transpose(
            torch.quantile(
                samples[:, 0, :], 1 - lower_percentile_threshold.reshape(-1)
            ).expand(samples.shape[2], samples.shape[0]),
            0,
            1,
        )

        samples = torch.clip(
            samples[:, 0, :], min=lower_threshold, max=upper_threshold
        ).unsqueeze(1)

        result = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )

        return result
