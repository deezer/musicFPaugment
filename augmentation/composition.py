import random
from typing import List, Optional, Tuple, Union

import torch.nn
from torch import Tensor

from augmentation.transform import BaseWaveformTransform
from augmentation.utils import ObjectDict


class BaseCompose(torch.nn.Module):
    """This class can apply a sequence of transforms to waveforms."""

    def __init__(
        self,
        transforms: List[
            torch.nn.Module
        ],  # FIXME: do we really want to support regular nn.Module?
        shuffle: bool = False,
        p: float = 1.0,
    ):
        """
        :param transforms: List of waveform transform instances
        :param shuffle: Should the order of transforms be shuffled?
        :param p: The probability of applying the Compose to the given batch.
        """
        super().__init__()
        self.p = p

        self.shuffle = shuffle
        self.are_parameters_frozen = False

        self.transforms = torch.nn.ModuleList(transforms)
        for tfm in self.transforms:
            tfm.output_type = "dict"

    def freeze_parameters(self, seed: int = 0) -> None:
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        self.are_parameters_frozen = True
        for transform in self.transforms:
            transform.freeze_parameters(seed)

    def unfreeze_parameters(self) -> None:
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False
        for transform in self.transforms:
            transform.unfreeze_parameters()


class Compose(BaseCompose):
    def forward(
        self,
        samples: Optional[Tensor] = None,
        sample_rate: Optional[int] = None,
    ) -> ObjectDict:
        inputs = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )

        transform_indexes = list(range(len(self.transforms)))
        if self.shuffle:
            random.shuffle(transform_indexes)
        for i in transform_indexes:
            tfm = self.transforms[i]
            if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                inputs = self.transforms[i](**inputs)

        return inputs


class SomeOf(BaseCompose):
    """
    SomeOf randomly picks several of the given transforms and applies them.
    The number of transforms to be applied can be chosen as follows:

      - Pick exactly n transforms
        Example: pick exactly 2 of the transforms
                 `SomeOf(2, [transform1, transform2, transform3])`

      - Pick between a minimum and maximum number of transforms
        Example: pick 1 to 3 of the transforms
                 `SomeOf((1, 3), [transform1, transform2, transform3])`

        Example: Pick 2 to all of the transforms
                 `SomeOf((2, None), [transform1, transform2, transform3])`
    """

    def __init__(
        self,
        num_transforms: Union[int, Tuple[int, int]],
        transforms: List[torch.nn.Module],
        p: float = 1.0,
    ):
        super().__init__(
            transforms=transforms,
            p=p,
        )

        self.transform_indexes: List[int] = []
        self.num_transforms = num_transforms
        self.all_transforms_indexes = list(range(len(self.transforms)))

        if isinstance(num_transforms, tuple):
            self.min_num_transforms = num_transforms[0]
            self.max_num_transforms = (
                num_transforms[1] if num_transforms[1] else len(transforms)
            )
        else:
            self.min_num_transforms = self.max_num_transforms = num_transforms

        assert self.min_num_transforms >= 1, "min_num_transforms must be >= 1"
        assert self.min_num_transforms <= len(
            transforms
        ), "num_transforms must be <= len(transforms)"
        assert self.max_num_transforms <= len(
            transforms
        ), "max_num_transforms must be <= len(transforms)"

    def randomize_parameters(self) -> None:
        num_transforms_to_apply = random.randint(
            self.min_num_transforms, self.max_num_transforms
        )
        self.transform_indexes = sorted(
            random.sample(self.all_transforms_indexes, num_transforms_to_apply)
        )

    def forward(
        self,
        samples: Optional[Tensor] = None,
        sample_rate: Optional[int] = None,
    ) -> ObjectDict:
        inputs = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )

        if random.random() < self.p:
            self.randomize_parameters()
            for i in self.transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                    inputs = self.transforms[i](**inputs)

        return inputs


class OneOf(SomeOf):
    """
    OneOf randomly picks one of the given transforms and applies it.
    """

    def __init__(
        self,
        transforms: List[torch.nn.Module],
        p: float = 1.0,
    ):
        super().__init__(
            num_transforms=1,
            transforms=transforms,
            p=p,
        )
