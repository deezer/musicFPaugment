import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.utils import Progbar  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchmetrics import PeakSignalNoiseRatio

from training.callbacks import (
    monitor_audios,
    monitor_losses,
    monitor_metrics,
    monitor_specs,
)
from training.jamendo import get_data_loaders
from training.loss import MultiResolutionSTFTLoss
from training.model import Demucs
from training.parameters import (
    BATCH_SIZE,
    CKPT_NAME,
    CKPT_PATH,
    EARLY_STOP,
    FACTOR,
    FACTOR_MAG,
    FACTOR_SC,
    LEARNING_RATE,
    MIN_DELTA,
    MODEL,
    NB_EPOCHS,
    PATIENCE,
    RUN_VAL,
    TRAIN_STEPS,
    VAL_STEPS,
)
from training.unet import UNet
from training.utils import (
    fix_random_seeds,
    remove_val_cache,
    set_gpus,
    update_progress_bar,
)
from training.visualisation import spectrogram
import argparse


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        train_steps: int,
        val_loader: DataLoader[Any],
        val_steps: int,
        criterion: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        early_stopping: Any,
        nb_epochs: int,
        device: str,
        metadatas: Dict[str, str],
        checkpoint: str,
        monitoring: bool = False,
        save: bool = False,
        input_type: str = "audio",
    ) -> None:
        """
        Initialize trainer with used parameters.
        ---
        Args:
            model (nn.Module):
                PyTorch model
            train_loader (DataLoader[Any]):
                Dataloader.
            val_loader (DataLoader[Any]):
                Dataloader.
            criterion (Dict[str, Any]):
                Loss.
            optimizer (torch.optim.Optimizer):
                PyTorch optimizer.
            scheduler (nn.Module):
                PyTorch scheduler.
            early_stopping (Any):
                Early stopping
            nb_epochs (int):
                Number of epochs.
            monitoring (bool, optional):
                Defaults to False.
            save (bool, optional):
                Defaults to False.
            radar_loss (bool, optional):
                Defaults to False.
            checkpoint (str):
                Model checkpoint path.
        """
        self.device = device
        self.model = model.to(self.device)
        self.checkpoint = checkpoint
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.nb_epochs = nb_epochs
        self.monitoring = monitoring
        self.save = save
        self.scheduler = scheduler
        self.criterion = criterion
        self.early_stopping = early_stopping
        self.metadatas = metadatas
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.input_type = input_type

        self.criterion["l1"] = self.criterion["l1"].to(device)
        self.criterion["mrsl"] = self.criterion["mrsl"].to(device)

        self.train_loader_iter = iter(self.train_loader)
        self.val_loader_iter = iter(self.val_loader)

        self.validation_metrics = {
            "psnr": PeakSignalNoiseRatio(average="micro").to(self.device),
        }

        self.epoch_start = 1
        self.min_valid_loss = np.inf

        if os.path.exists(self.checkpoint + "last_epoch.pt"):
            print(f"\nLoading checkpoint {self.checkpoint}last_epoch.pt.")

            self.model.load_state_dict(
                torch.load(self.checkpoint + "last_epoch.pt", map_location=device)[
                    "model_state_dict"
                ]
            )
            self.epoch_start = torch.load(
                self.checkpoint + "last_epoch.pt", map_location="cpu"
            )["epoch"]

            self.min_valid_loss = torch.load(
                self.checkpoint + "last_epoch.pt", map_location="cpu"
            )["best_val_loss"]

            print(f"Minimum validation loss is {self.min_valid_loss}...")
            print(f"Epoch is {self.epoch_start}...")

            self.optimizer.load_state_dict(
                torch.load(self.checkpoint + "last_epoch.pt", map_location=device)[
                    "optimizer_state_dict"
                ]
            )
            self.scheduler.load_state_dict(
                torch.load(self.checkpoint + "last_epoch.pt", map_location=device)[
                    "scheduler_state_dict"
                ]
            )
            self.early_stopping = torch.load(
                self.checkpoint + "last_epoch.pt", map_location="cpu"
            )["early_stopping"]

        elif not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)

        if self.monitoring:
            self.writer = SummaryWriter(  # type: ignore
                "/workspace/src/monitoring/" + str(self.metadatas["name"])
            )

    def training_loop(self) -> None:
        min_valid_loss = self.min_valid_loss

        for epoch in range(self.epoch_start, self.nb_epochs):
            if self.early_stopping.early_stop:
                break

            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            print(f"\n\nEpoch {epoch}: {time.time() - t0:.2f} training seconds")
            print(f"Training Loss: {train_loss['loss']}")

            t0 = time.time()
            val_losses, val_metrics = self.validation_epoch()
            print(f"\nEpoch {epoch}: {time.time() - t0:.2f} validation seconds")
            print(f"Validation Loss: {val_losses}")
            print(f"Validation Metrics: {val_metrics}")

            self.early_stopping(val_losses["loss"])

            if min_valid_loss > val_losses["loss"]:
                print(
                    f"Validation loss decreased from {min_valid_loss} to {val_losses['loss']}."
                )
                min_valid_loss = val_losses["loss"]

                if self.save:
                    print("Saving the best model...")
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "best_val_loss": min_valid_loss,
                        },
                        self.checkpoint + "best_epoch.pt",
                    )

            if self.save:
                print("Saving the last model...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "early_stopping": self.early_stopping,
                        "train_loss": train_loss,
                        "val_losses": val_losses,
                        "best_val_loss": min_valid_loss,
                    },
                    self.checkpoint + "last_epoch.pt",
                )

            """gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()"""

            if self.monitoring:
                monitor_losses(
                    self.writer,
                    train_loss,
                    datas="train",
                    epoch=epoch,
                )
                monitor_losses(
                    self.writer,
                    val_losses,
                    datas="val",
                    epoch=epoch,
                )
                monitor_metrics(self.writer, val_metrics, datas="val", epoch=epoch)

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        train_loss = 0.0

        if self.input_type == "audio":
            train_l1_loss = 0.0
            train_sc_loss = 0.0
            train_mag_loss = 0.0

        self.model.train()

        progress_bar = Progbar(self.train_steps)

        for idx in range(1, self.train_steps):
            clean_audios, augmented_audios = next(self.train_loader_iter)

            clean_audios = clean_audios.squeeze(-1).to(self.device)
            augmented_audios = augmented_audios.unsqueeze(1).squeeze(-1).to(self.device)

            if self.input_type == "spec":
                clean_specs = spectrogram(clean_audios, device=self.device).to(
                    self.device
                )
                augmented_specs = spectrogram(
                    augmented_audios.squeeze(1), device=self.device
                ).to(self.device)

                predicted_specs = self.model(
                    augmented_specs.unsqueeze(1).float()
                ).squeeze(1)

            if self.input_type == "audio":
                predicted_audios = self.model(augmented_audios).squeeze(1)

            with torch.autograd.set_detect_anomaly(True):  # type: ignore
                if self.input_type == "spec":
                    loss = self.criterion["l1"](predicted_specs, clean_specs)

                    update_progress_bar(
                        progress_bar,
                        {
                            "loss": loss.item(),
                        },
                        idx,
                        "train",
                    )

                if self.input_type == "audio":
                    l1_loss = self.criterion["l1"](predicted_audios, clean_audios)
                    sc_loss, mag_loss = self.criterion["mrsl"](
                        predicted_audios, clean_audios
                    )

                    loss = l1_loss + sc_loss + mag_loss
                    train_l1_loss += l1_loss.item()
                    train_sc_loss += sc_loss.item()
                    train_mag_loss += mag_loss.item()

                    update_progress_bar(
                        progress_bar,
                        {
                            "loss": loss.item(),
                            "l1": l1_loss.item(),
                            "sc": sc_loss.item(),
                            "mag": mag_loss.item(),
                        },
                        idx,
                        "train",
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if self.monitoring:
                augmented_audios = augmented_audios.squeeze(1)

                if epoch % 10 == 0 and idx == 1:
                    if self.input_type == "audio":
                        monitor_audios(
                            self.writer,
                            clean_audios.detach().cpu(),
                            augmented_audios.detach().cpu(),
                            predicted_audios.detach().cpu(),
                            epoch=epoch,
                            datas="train",
                        )
                        monitor_specs(
                            self.writer,
                            clean_audios.detach().cpu(),
                            augmented_audios.detach().cpu(),
                            predicted_audios.detach().cpu(),
                            epoch=epoch,
                            datas="train",
                        )

        train_loss = train_loss / self.train_steps

        train_losses = {
            "loss": train_loss,
        }

        if self.input_type == "audio":
            train_l1_loss = train_l1_loss / self.train_steps
            train_sc_loss = train_sc_loss / self.train_steps
            train_mag_loss = train_mag_loss / self.train_steps

            train_losses = {
                "loss": train_loss,
                "l1_loss": train_l1_loss,
                "sc_loss": train_sc_loss,
                "mag_loss": train_mag_loss,
            }

        return train_losses

    def validation_epoch(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validation epoch
        """
        val_loss = 0.0
        if self.input_type == "audio":
            val_l1_loss = 0.0
            val_sc_loss = 0.0
            val_mag_loss = 0.0
        psnr_metric = 0.0

        self.model.eval()

        with torch.no_grad():
            progress_bar = Progbar(self.val_steps)

            for idx in range(1, self.val_steps):
                clean_audios, augmented_audios = next(self.val_loader_iter)

                clean_audios = clean_audios.squeeze(-1).to(self.device)
                augmented_audios = (
                    augmented_audios.unsqueeze(1).squeeze(-1).to(self.device)
                )

                if self.input_type == "spec":
                    clean_specs = spectrogram(clean_audios, device=self.device).to(
                        self.device
                    )
                    augmented_specs = spectrogram(
                        augmented_audios.squeeze(1), device=self.device
                    ).to(self.device)

                    predicted_specs = self.model(
                        augmented_specs.unsqueeze(1).float()
                    ).squeeze(1)

                    loss = self.criterion["l1"](clean_specs, predicted_specs)

                    psnr_metric += self.validation_metrics["psnr"](
                        predicted_specs, clean_specs
                    ).item()

                    update_progress_bar(
                        progress_bar,
                        {
                            "all": loss.item(),
                        },
                        idx,
                        "val",
                    )

                if self.input_type == "audio":
                    predicted_audios = self.model(augmented_audios).squeeze(1)

                    l1_loss = self.criterion["l1"](predicted_audios, clean_audios)
                    sc_loss, mag_loss = self.criterion["mrsl"](
                        predicted_audios, clean_audios
                    )

                    loss = l1_loss + sc_loss + mag_loss

                    psnr_metric += self.validation_metrics["psnr"](
                        predicted_audios, clean_audios
                    ).item()

                    val_l1_loss += l1_loss.item()
                    val_sc_loss += sc_loss.item()
                    val_mag_loss += mag_loss.item()

                    update_progress_bar(
                        progress_bar,
                        {
                            "all": loss.item(),
                            "l1": l1_loss.item(),
                            "sc": sc_loss.item(),
                            "mag": mag_loss.item(),
                        },
                        idx,
                        "val",
                    )

                val_loss += loss.item()

        val_loss = val_loss / self.val_steps

        val_losses = {
            "loss": val_loss,
        }

        if self.input_type == "audio":
            val_l1_loss = val_l1_loss / self.val_steps
            val_sc_loss = val_sc_loss / self.val_steps
            val_mag_loss = val_mag_loss / self.val_steps
            val_losses = {
                "loss": val_loss,
                "l1_loss": val_l1_loss,
                "sc_loss": val_sc_loss,
                "mag_loss": val_mag_loss,
            }
        psnr_metric = psnr_metric / self.val_steps

        self.scheduler.step(val_loss)

        val_metrics = {
            "psnr": psnr_metric,
        }

        return val_losses, val_metrics

    def start_epoch(self) -> None:
        start_loss = 0.0

        if self.input_type == "audio":
            start_l1_loss = 0.0
            start_sc_loss = 0.0
            start_mag_loss = 0.0

        psnr_metric = 0.0

        with torch.no_grad():
            progress_bar = Progbar(self.val_steps)

            for idx in range(self.val_steps):
                clean_audios, augmented_audios = next(self.val_loader_iter)

                clean_audios = clean_audios.squeeze(-1).to(self.device)
                augmented_audios = (
                    augmented_audios.unsqueeze(1).squeeze(-1).to(self.device)
                )

                if self.input_type == "spec":
                    clean_specs = spectrogram(clean_audios, device=self.device).to(
                        self.device
                    )
                    augmented_specs = spectrogram(
                        augmented_audios.squeeze(1), device=self.device
                    ).to(self.device)

                    loss = self.criterion["l1"](augmented_specs.squeeze(1), clean_specs)

                    psnr_metric += self.validation_metrics["psnr"](
                        augmented_specs.squeeze(1), clean_specs
                    ).item()

                    update_progress_bar(
                        progress_bar,
                        {
                            "l1": loss.item(),
                        },
                        idx,
                        "val",
                    )

                elif self.input_type == "audio":
                    l1_loss = self.criterion["l1"](
                        augmented_audios.squeeze(1), clean_audios
                    )
                    sc_loss, mag_loss = self.criterion["mrsl"](
                        augmented_audios.squeeze(1), clean_audios
                    )

                    loss = l1_loss + sc_loss + mag_loss

                    psnr_metric += self.validation_metrics["psnr"](
                        augmented_audios.squeeze(1), clean_audios
                    ).item()

                    update_progress_bar(
                        progress_bar,
                        {
                            "all": loss.item(),
                            "l1": l1_loss.item(),
                            "sc": sc_loss.item(),
                            "mag": mag_loss.item(),
                        },
                        idx,
                        "val",
                    )

                    start_l1_loss += l1_loss.item()
                    start_sc_loss += sc_loss.item()
                    start_mag_loss += mag_loss.item()

                start_loss += loss.item()

        start_loss = start_loss / self.val_steps

        if self.input_type == "audio":
            start_l1_loss = start_l1_loss / self.val_steps
            start_sc_loss = start_sc_loss / self.val_steps
            start_mag_loss = start_mag_loss / self.val_steps
            start_losses = {
                "loss": start_loss,
                "l1_loss": start_l1_loss,
                "sc_loss": start_sc_loss,
                "mag_loss": start_mag_loss,
            }
        elif self.input_type == "spec":
            start_losses = {
                "loss": start_loss,
            }

        psnr_metric = psnr_metric / self.val_steps

        start_metrics = {
            "psnr": psnr_metric,
        }

        print(f"\nStart Loss: {start_losses}")
        print(f"\nStart Metrics: {start_metrics}")

        if self.monitoring:
            monitor_losses(
                self.writer,
                start_losses,
                datas="val",
                epoch=0,
            )
            monitor_metrics(self.writer, start_metrics, datas="val", epoch=0)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """
        patience (int):
            Number of epochs to wait before stopping when loss is not improving.
        min_delta (float):
            Minimum difference between new loss and old loss for new loss to be considered as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("-inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if (
            self.best_loss == float("-inf")
            or self.best_loss - val_loss > self.min_delta
        ):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="", default="unet")

    args = parser.parse_args()
    
    fix_random_seeds()
    remove_val_cache()
    
    if str(args.model) == "unet":      
        MODEL = "unet"
        LEARNING_RATE = 1e-3
        CKPT_NAME = f"{MODEL}_lr_{LEARNING_RATE}_BS_{BATCH_SIZE}"
        CKPT_PATH = f"{os.path.dirname(__file__)}/checkpoints/{CKPT_NAME}/"        
        
    if str(args.model) == "demucs":      
        MODEL = "demucs"
        LEARNING_RATE = 5e-4
        CKPT_NAME = f"{MODEL}_lr_{LEARNING_RATE}_BS_{BATCH_SIZE}"
        CKPT_PATH = f"{os.path.dirname(__file__)}/checkpoints/{CKPT_NAME}/"        
                
    device = set_gpus()

    train_loader, val_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        val_steps=VAL_STEPS,
        run_val=RUN_VAL,
    )

    if MODEL == "unet":
        model = UNet(1, 1, rate=0.05)
        audio_type = "spec"
    if MODEL == "demucs":
        model = Demucs()
        audio_type = "audio"

    metadatas = {
        "name": CKPT_NAME,
        "model": MODEL,
    }
    criterion = {
        "l1": torch.nn.L1Loss(reduction="mean"),
        "mrsl": MultiResolutionSTFTLoss(factor_sc=FACTOR_SC, factor_mag=FACTOR_MAG),
    }

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=FACTOR,
        patience=PATIENCE,
    )
    early_stopping = EarlyStopping(patience=EARLY_STOP, min_delta=MIN_DELTA)

    trainer = Trainer(
        model,
        train_loader,
        TRAIN_STEPS,
        val_loader,
        VAL_STEPS,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        NB_EPOCHS,
        device,
        metadatas,
        monitoring=False,
        save=False,
        checkpoint=CKPT_PATH,
        input_type=audio_type,
    )

    trainer.start_epoch()
    trainer.training_loop()
