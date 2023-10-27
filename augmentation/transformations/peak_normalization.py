from typing import Optional

import torch
from torch import Tensor

from augmentation.transform import BaseWaveformTransform
from augmentation.utils import ObjectDict


class PeakNormalization(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in each audio snippet
    in the batch becomes 0 dBFS, i.e. the loudest level allowed if all samples must be between
    -1 and 1.

    This transform has an alternative mode (apply_to="only_too_loud_sounds") where it only
    applies to audio snippets that have extreme values outside the [-1, 1] range. This is useful
    for avoiding digital clipping in audio that is too loud, while leaving other audio
    untouched.
    """

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ):
        super().__init__(
            p=p,
            sample_rate=sample_rate,
        )

    def randomize_parameters(
        self,
        samples: Tensor,
    ) -> None:
        # Compute the most extreme value of each multichannel audio snippet in the batch
        most_extreme_values, _ = torch.max(torch.abs(samples), dim=-1)
        most_extreme_values, _ = torch.max(most_extreme_values, dim=-1)

        # Avoid division by zero
        self.transform_parameters["selector"] = most_extreme_values > 0.0

        if self.transform_parameters["selector"].any():
            self.transform_parameters["divisors"] = torch.reshape(
                most_extreme_values[self.transform_parameters["selector"]], (-1, 1, 1)
            )

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        if "divisors" in self.transform_parameters:
            samples[self.transform_parameters["selector"]] /= self.transform_parameters[
                "divisors"
            ]

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )
