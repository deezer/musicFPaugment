import random
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from augmentation.transform import BaseWaveformTransform, EmptyPathException
from augmentation.utils import Audio, ObjectDict, calculate_rms


def samplePairing(audio1: Any, audio2: Any) -> Any:
    return (audio1 + audio2) / 2


class AddBackgroundNoise(BaseWaveformTransform):
    """
    Add background noise to the input audio.
    """

    # Note: This transform has only partial support for multichannel audio. Noises that are not
    # mono get mixed down to mono before they are added to all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        background_paths: Dict[str, List[str]],
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        p: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        background_paths(List[str]):
            Either a path to a folder with audio files or a list of paths
            to audio files.
        min_snr_in_db (float):
            minimum SNR in dB.
        max_snr_in_db (float):
            maximium SNR in dB.
        """

        super().__init__(
            p=p,
            sample_rate=sample_rate,
        )
        # TODO: check that one can read audio files
        # self.background_paths = find_audio_files_in_paths(background_paths)
        self.background_paths = background_paths

        self.background_paths_to_update = self.background_paths.copy()

        if sample_rate is not None:
            self.audio = Audio(sample_rate=sample_rate, mono=True)

        if len(self.background_paths) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

    def random_background(self, audio: Audio, target_num_samples: int) -> torch.Tensor:
        pieces = []

        missing_num_samples = target_num_samples

        while missing_num_samples > 0:
            selected_scene = random.choice(list(self.background_paths_to_update.keys()))
            # if len(self.background_paths_to_update[selected_scene]) == 0:
            #     self.background_paths_to_update[selected_scene] = self.background_paths[selected_scene].copy()

            background_path = random.choice(
                self.background_paths_to_update[str(selected_scene)]
            )
            # self.background_paths_to_update[str(selected_scene)].remove(background_path)

            if len(list(background_path)) == 2:  # handle mixup
                background_num_samples = min(
                    audio.get_num_samples(background_path[0]),
                    audio.get_num_samples(background_path[1]),
                )

                if background_num_samples >= missing_num_samples:
                    num_samples = missing_num_samples
                    sample_offset1 = random.randint(
                        0, background_num_samples - missing_num_samples
                    )
                    background_samples1 = audio(
                        background_path[0],
                        sample_offset=sample_offset1,
                        num_samples=num_samples,
                    )

                    sample_offset2 = random.randint(
                        0, background_num_samples - missing_num_samples
                    )
                    background_samples2 = audio(
                        background_path[1],
                        sample_offset=sample_offset2,
                        num_samples=num_samples,
                    )

                    background_samples = samplePairing(
                        background_samples1, background_samples2
                    )
                    missing_num_samples = 0
                else:
                    background_samples1 = audio(background_path[0])
                    background_samples2 = audio(background_path[0])
                    background_samples = samplePairing(
                        background_samples1, background_samples2
                    )
                    missing_num_samples -= background_num_samples

                pieces.append(background_samples)
            else:  # normal case
                background_num_samples = audio.get_num_samples(background_path)

                if background_num_samples >= missing_num_samples:
                    sample_offset = random.randint(
                        0, background_num_samples - missing_num_samples
                    )

                    num_samples = missing_num_samples
                    background_samples = audio(
                        background_path,
                        sample_offset=sample_offset,
                        num_samples=num_samples,
                    )

                    missing_num_samples = 0
                else:
                    background_samples = audio(background_path)
                    missing_num_samples -= background_num_samples

                pieces.append(background_samples)
        return audio.rms_normalize(
            torch.cat([audio.rms_normalize(piece) for piece in pieces], dim=1)
        )

    def randomize_parameters(
        self,
        samples: Tensor,
    ) -> None:
        """
        samples (Tensor):
            (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = samples.shape

        self.transform_parameters["background"] = torch.stack(
            [self.random_background(self.audio, num_samples) for _ in range(batch_size)]
        )

        if self.min_snr_in_db == self.max_snr_in_db:
            self.transform_parameters["snr_in_db"] = torch.full(
                size=(batch_size,),
                fill_value=self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            )

        else:
            snr_distribution = torch.distributions.Uniform(  # type: ignore
                low=torch.tensor(
                    self.min_snr_in_db,
                    dtype=torch.float32,
                    device=samples.device,
                ),
                high=torch.tensor(
                    self.max_snr_in_db,
                    dtype=torch.float32,
                    device=samples.device,
                ),
                validate_args=True,
            )
            self.transform_parameters["snr_in_db"] = snr_distribution.sample(
                sample_shape=(batch_size,)  # type: ignore
            )

    def apply_transform(
        self,
        samples: Tensor,
        sample_rate: int,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)

        background = self.transform_parameters["background"].to(samples.device)

        # (batch_size, num_channels)
        background_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        samples = samples + background_rms.unsqueeze(-1) * background.view(
            batch_size, 1, num_samples
        ).expand(-1, num_channels, -1)

        ## Normalization :
        peak_values = torch.max(torch.abs(samples), dim=2).values
        peak_values = peak_values.expand(-1, samples.shape[2]).reshape(
            samples.shape[0], 1, samples.shape[2]
        )
        samples /= peak_values

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
        )
