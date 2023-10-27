from typing import Any, Dict

import torch
import torchaudio  # type: ignore

from training.parameters import WAVEFORM_SAMPLING_RATE
from training.visualisation import plot_spectrogram, spectrogram


def monitor_losses(
    writer: Any, losses: Dict[str, Any], datas: str = "train", epoch: int = 0
) -> None:
    for key in losses.keys():
        writer.add_scalar("{}/{}".format(str(datas), str(key)), losses[str(key)], epoch)


def monitor_metrics(
    writer: Any, metrics: Dict[str, Any], datas: str = "val", epoch: int = 0
) -> None:
    for key in metrics.keys():
        writer.add_scalar(
            "{}/{}".format(str(datas), str(key)), metrics[str(key)], epoch
        )


def monitor_audios(
    writer: Any,
    clean_audios: torch.Tensor,
    augmented_audios: torch.Tensor,
    predicted_audios: torch.Tensor,
    epoch: int,
    datas: str = "train",
    sample_rate: int = WAVEFORM_SAMPLING_RATE,
) -> None:
    for i in range(0, 3):
        writer.add_audio(
            "{}_{}/{}".format(str(datas), "clean", str(i)),
            clean_audios[i],
            epoch,
            sample_rate,
        )
        writer.add_audio(
            "{}_{}/{}".format(str(datas), "augmented", str(i)),
            augmented_audios[i],
            epoch,
            sample_rate,
        )
        writer.add_audio(
            "{}_{}/{}".format(str(datas), "denoised", str(i)),
            predicted_audios[i],
            epoch,
            sample_rate,
        )
        torchaudio.save(
            f"/workspace/mp3s/clean_{i}.wav",
            clean_audios[i].unsqueeze(0),
            sample_rate,
        )
        torchaudio.save(
            f"/workspace/mp3s/aug_{i}.wav",
            augmented_audios[i].unsqueeze(0),
            sample_rate,
        )
        torchaudio.save(
            f"/workspace/mp3s/denoised_{i}.wav",
            predicted_audios[i].unsqueeze(0),
            sample_rate,
        )


def monitor_specs(
    writer: Any,
    clean_audios: torch.Tensor,
    augmented_audios: torch.Tensor,
    predicted_audios: torch.Tensor,
    epoch: int,
    datas: str = "train",
) -> None:
    clean_specs = spectrogram(clean_audios, amplitude=True)
    augmented_specs = spectrogram(augmented_audios, amplitude=True)
    predicted_specs = spectrogram(predicted_audios, amplitude=True)

    for i in range(0, 3):
        fig_clean = plot_spectrogram(clean_specs[i], f"/workspace/mp3s/clean_{i}.png")
        writer.add_figure(
            "{}_{}/{}".format(str(datas), "clean", str(i)),
            fig_clean,
            epoch,
        )

        fig_aug = plot_spectrogram(augmented_specs[i], f"/workspace/mp3s/aug_{i}.png")
        writer.add_figure(
            "{}_{}/{}".format(str(datas), "augmented", str(i)),
            fig_aug,
            epoch,
        )

        fig_pred = plot_spectrogram(
            predicted_specs[i], f"/workspace/mp3s/denoised_{i}.png"
        )
        writer.add_figure(
            "{}_{}/{}".format(str(datas), "denoised", str(i)),
            fig_pred,
            epoch,
        )
