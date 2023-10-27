import os
from typing import Any, Dict, List

import torch

from augmentation.composition import Compose
from augmentation.constants import DEFAULT_PARAMETERS, IMPULSE_RESPONSE_DIR
from augmentation.transformations.background_noise import AddBackgroundNoise
from augmentation.transformations.clipping import Clipping
from augmentation.transformations.gain import Gain
from augmentation.transformations.impulse_response import ApplyImpulseResponse
from augmentation.transformations.pass_filters import HighPassFilter, LowPassFilter
from augmentation.transformations.peak_normalization import PeakNormalization


class AugmentFP(object):
    """
    Music augmentation class for audio fingerprinting.
    """

    def __init__(
        self,
        background_paths: Dict[str, List[str]],
        sample_rate: int,
        parameters: Dict[str, float] = DEFAULT_PARAMETERS,
        impulse_response_dir: str = IMPULSE_RESPONSE_DIR,
    ) -> None:
        """
        Args:
            background_paths (List[str]):
                Background noise paths.
            sample_rate (int):
                Self-explanatory.
            parameters (Dict[str, float]):
                Augmentation pipeline parameters.
            impulse_response_dir (str):
                Impulse reponse directory.
        """

        ir_paths = [
            os.path.join(impulse_response_dir, f)
            for f in os.listdir(impulse_response_dir)
            if f.endswith(".wav")
        ]

        self.augmentation_pipeline = Compose(
            transforms=[
                HighPassFilter(
                    p=parameters["proba_cutoff_freq1"],
                    min_cutoff_freq=parameters["min_cutoff_freq1"],
                    max_cutoff_freq=parameters["max_cutoff_freq1"],
                    sample_rate=sample_rate,
                ),
                # Reverb:
                ApplyImpulseResponse(
                    ir_paths,
                    sample_rate=sample_rate,
                    p=parameters["proba_ir_response"],
                ),
                # Background Noise :
                AddBackgroundNoise(
                    background_paths,
                    p=parameters["proba_snr_in_db"],
                    min_snr_in_db=parameters["min_snr_in_db"],
                    max_snr_in_db=parameters["max_snr_in_db"],
                    sample_rate=sample_rate,
                ),
                Gain(
                    p=parameters["proba_gain_in_db"],
                    min_gain_in_db=parameters["min_gain_in_db"],
                    max_gain_in_db=parameters["max_gain_in_db"],
                ),
                Clipping(
                    p=parameters["proba_percentile_threshold"],
                    min_percentile_threshold=0,
                    max_percentile_threshold=parameters["max_percentile_threshold"],
                ),
                LowPassFilter(
                    p=parameters["proba_cutoff_freq2"],
                    min_cutoff_freq=parameters["min_cutoff_freq2"],
                    max_cutoff_freq=parameters["max_cutoff_freq2"],
                    sample_rate=sample_rate,
                ),
                HighPassFilter(
                    p=parameters["proba_cutoff_freq3"],
                    min_cutoff_freq=parameters["min_cutoff_freq3"],
                    max_cutoff_freq=parameters["max_cutoff_freq3"],
                    sample_rate=sample_rate,
                ),
                # Normalization:
                PeakNormalization(p=1),
            ]
        )

    def __call__(self, waveform: torch.Tensor) -> Any:
        augmented_audio = self.augmentation_pipeline(waveform.unsqueeze(0)).samples
        return augmented_audio.squeeze(0)

    def batch_augment(self, waveforms: torch.Tensor) -> Any:
        augmented_audios = self.augmentation_pipeline(waveforms).samples
        return augmented_audios.squeeze(0)
