from math import ceil
from typing import Optional

import torch
from torch import Tensor
from torch.fft import irfft, rfft

from augmentation.transform import BaseWaveformTransform
from augmentation.utils import Audio, ObjectDict, calculate_rms


def _gen_noise(
    f_decay: float,
    num_samples: int,
    device: str,
    sample_rate: Optional[int] = None,
) -> Tensor:
    """
    Generate colored noise with f_decay decay using torch.fft
    """
    if sample_rate is None:
        sample_rate = 44100

    noise = torch.normal(
        0.0,
        1.0,
        (sample_rate,),
        device=device,
    )
    spec = rfft(noise)
    mask = 1 / (
        torch.linspace(1, (sample_rate / 2) ** 0.5, spec.shape[0], device=device)
        ** f_decay
    )
    spec *= mask
    noise = Audio.rms_normalize(irfft(spec).unsqueeze(0)).squeeze()
    noise = torch.cat([noise] * int(ceil(num_samples / sample_rate)))
    return noise[:num_samples]


class AddColoredNoise(BaseWaveformTransform):
    """
    Add colored noises to the input audio.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        min_f_decay: float = -2.0,
        max_f_decay: float = 2.0,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ):
        """
        min_snr_in_db (float):
            minimum SNR in dB.
        max_snr_in_db (float):
            maximium SNR in dB.
        min_f_decay (float):
            defines the minimum frequency power decay (1/f**f_decay).
            Typical values are "white noise" (f_decay=0), "pink noise" (f_decay=1),
            "brown noise" (f_decay=2), "blue noise (f_decay=-1)" and "violet noise"
            (f_decay=-2)
        max_f_decay (float):
            defines the maximum power decay (1/f**f_decay) for non-white noises.
        """

        super().__init__(
            p=p,
            sample_rate=sample_rate,
        )

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.min_f_decay = min_f_decay
        self.max_f_decay = max_f_decay
        if self.min_f_decay > self.max_f_decay:
            raise ValueError("min_f_decay must not be greater than max_f_decay")

    def randomize_parameters(
        self,
        samples: Tensor,
    ) -> None:
        """
        selected_samples:
            (batch_size, num_channels, num_samples)
        """
        batch_size, _, _ = samples.shape

        # (batch_size, ) SNRs
        for param, mini, maxi in [
            ("snr_in_db", self.min_snr_in_db, self.max_snr_in_db),
            ("f_decay", self.min_f_decay, self.max_f_decay),
        ]:
            dist = torch.distributions.Uniform(  # type: ignore
                low=torch.tensor(mini, dtype=torch.float32, device=samples.device),
                high=torch.tensor(maxi, dtype=torch.float32, device=samples.device),
                validate_args=True,
            )
            self.transform_parameters[param] = dist.sample(sample_shape=(batch_size,))  # type: ignore

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        noise = torch.stack(
            [
                _gen_noise(
                    self.transform_parameters["f_decay"][i],
                    num_samples,
                    str(samples.device),
                    self.sample_rate,
                )
                for i in range(batch_size)
            ]
        )

        noise_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        samples = samples + noise_rms.unsqueeze(-1) * noise.view(
            batch_size, 1, num_samples
        ).expand(-1, num_channels, -1)

        ## Normalize output :
        peak_values = torch.max(torch.abs(samples), dim=2).values
        peak_values = peak_values.expand(-1, samples.shape[2]).reshape(
            samples.shape[0], 1, samples.shape[2]
        )
        samples /= peak_values

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )
