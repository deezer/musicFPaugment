import random
import warnings
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Bernoulli

from augmentation.utils import ObjectDict, is_multichannel


class MultichannelAudioNotSupportedException(Exception):
    pass


class EmptyPathException(Exception):
    pass


class ModeNotSupportedException(Exception):
    pass


class BaseWaveformTransform(torch.nn.Module):
    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ):
        """
        p:
            The probability of the transform being applied to a batch/example/channel
            (see mode and p_mode). This number must be in the range [0.0, 1.0].
        sample_rate:
            sample_rate can be set either here or when
            calling the transform.

        """
        super().__init__()
        assert 0.0 <= p <= 1.0
        self._p = p
        self.sample_rate = sample_rate
        self.transform_parameters: Dict[Any, Any] = {}
        self.are_parameters_frozen = False
        self.bernoulli_distribution = Bernoulli(self._p)  # type: ignore

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, p: float) -> None:
        self._p = p
        # Update the Bernoulli distribution accordingly
        self.bernoulli_distribution = Bernoulli(self._p)  # type: ignore

    def forward(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> Any:
        batch_size, num_channels, num_samples = samples.shape

        if not isinstance(samples, Tensor) or len(samples.shape) != 3:
            raise RuntimeError(
                "torch-audiomentations expects three-dimensional input tensors, with"
                " dimension ordering like [batch_size, num_channels, num_samples]. If your"
                " audio is mono, you can use a shape like [batch_size, 1, num_samples]."
            )

        if batch_size * num_channels * num_samples == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(
                    self.__class__.__name__
                )
            )
            output = ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
            )
            return output.samples if self.output_type == "tensor" else output

        if is_multichannel(samples):
            if num_channels > num_samples:
                warnings.warn(
                    "Multichannel audio must have channels first, not channels last. In"
                    " other words, the shape must be (batch size, channels, samples), not"
                    " (batch_size, samples, channels)"
                )

            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException(
                    "{} only supports mono audio, not multichannel audio".format(
                        self.__class__.__name__
                    )
                )

        self.transform_parameters = {
            "should_apply": self.bernoulli_distribution.sample(  # type: ignore
                sample_shape=(batch_size,)
            ).to(torch.bool)
        }

        if self.transform_parameters["should_apply"].any():
            cloned_samples = samples.clone()

            selected_samples = cloned_samples[self.transform_parameters["should_apply"]]

            self.randomize_parameters(
                samples=selected_samples,
            )

            perturbed: ObjectDict = self.apply_transform(
                samples=selected_samples,
                sample_rate=sample_rate,
            )

            cloned_samples[
                self.transform_parameters["should_apply"]
            ] = perturbed.samples

            output = ObjectDict(
                samples=cloned_samples,
                sample_rate=perturbed.sample_rate,
            )
            return output

        output = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )
        return output

    def _forward_unimplemented(self, *inputs: Any) -> None:
        # Avoid IDE error message like "Class ... must implement all abstract methods"
        # See also https://github.com/python/mypy/issues/8795#issuecomment-691658758
        pass

    def randomize_parameters(self, samples: Tensor) -> None:
        pass

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        raise NotImplementedError()

    def serialize_parameters(self) -> None:
        """Return the parameters as a JSON-serializable dict."""
        raise NotImplementedError()
        # TODO: Clone the params and convert any tensors into json-serializable lists
        # return self.transform_parameters

    def freeze_parameters(self, seed: int = 0) -> None:
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True
        random.seed(seed)
        torch.manual_seed(seed)

    def unfreeze_parameters(self) -> None:
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False
