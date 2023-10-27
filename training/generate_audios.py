import json
import random

import tensorflow as tf  # type: ignore
import torch
import torchaudio  # type: ignore
from torchmetrics import PeakSignalNoiseRatio

from augmentation import AugmentFP
from training.jamendo import get_jamendo_data
from training.model import Demucs
from training.parameters import (
    BATCH_SIZE,
    CKPT_PATH,
    DURATION,
    MODEL,
    VAL_STEPS,
    WAVEFORM_SAMPLING_RATE,
)
from training.train import EarlyStopping
from training.unet import UNet
from training.visualisation import plot_spectrogram, spectrogram

device = "cpu"

tf.config.set_visible_devices([], "GPU")

psnr = PeakSignalNoiseRatio(average="micro").to(device)

if MODEL == "demucs":
    model = Demucs().to(device)
if MODEL == "unet":
    model = UNet(1, 1).to(device)

_ = EarlyStopping()

weights_path = CKPT_PATH + "best_epoch.pt"
model.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])
model.eval()

with open("training/splits/val.json", "r") as f:
    noise_paths = json.load(f)

af = AugmentFP(noise_paths, WAVEFORM_SAMPLING_RATE)

__, val_ids = get_jamendo_data(
    "/workspace/mtg-jamendo-dataset/", num_val=VAL_STEPS * BATCH_SIZE
)
waveform, sr = torchaudio.load(random.choice(val_ids))
waveform = waveform.mean(axis=0)
waveform = torchaudio.transforms.Resample(sr, WAVEFORM_SAMPLING_RATE)(waveform)
nb_samples_segment = WAVEFORM_SAMPLING_RATE * DURATION
start = random.randint(0, waveform.shape[0] - nb_samples_segment)
waveform = waveform[start : start + nb_samples_segment].unsqueeze(0)

aug = af(waveform)

if MODEL == "demucs":
    with torch.inference_mode():  # type: ignore
        predicted = model(aug.unsqueeze(0).float()).squeeze(0)

    torchaudio.save(
        "/workspace/src/mp3s/clean.wav",
        waveform,
        WAVEFORM_SAMPLING_RATE,
    )
    torchaudio.save(
        "/workspace/src/mp3s/aug.wav",
        aug,
        WAVEFORM_SAMPLING_RATE,
    )
    torchaudio.save(
        "/workspace/src/mp3s/denoised.wav",
        predicted,
        WAVEFORM_SAMPLING_RATE,
    )

    print("Generated audios!")

    spec_clean = spectrogram(waveform, amplitude=True)
    spec_aug = spectrogram(aug, amplitude=True)
    spec_denoised = spectrogram(predicted)

    _ = plot_spectrogram(spec_clean, "/workspace/src/mp3s/clean.png")
    _ = plot_spectrogram(spec_aug, "/workspace/src/mp3s/aug.png")
    _ = plot_spectrogram(
        spec_denoised, "/workspace/src/mp3s/denoised.png", amplitude=True
    )

    print("Generated spectrograms!")

if MODEL == "unet":
    # Unet setup: s
    spec_clean = spectrogram(waveform, amplitude=True)
    spec_noisy = spectrogram(aug)

    with torch.inference_mode():  # type: ignore
        spec_denoised = model(spec_noisy.unsqueeze(1).float()).squeeze(0)

    _ = plot_spectrogram(spec_clean, "/workspace/src/mp3s/clean.png")
    _ = plot_spectrogram(spec_noisy, "/workspace/src/mp3s/aug.png", amplitude=True)
    _ = plot_spectrogram(
        spec_denoised, "/workspace/src/mp3s/denoised.png", amplitude=True
    )

    print("Generated spectrograms!")
