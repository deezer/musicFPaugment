import math

import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio

psnr = PeakSignalNoiseRatio(average="micro")


class Recall(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicted: torch.Tensor, gt: torch.Tensor, device: str = "cpu"
    ) -> float:
        """
        Compute the recall between predicted binary peak mask and ground thruth.
        ---
        Args:
            predicted (torch.Tensor):
                Predicted mask.
            gt (torch.Tensor):
                Ground truth.
            device (str):
                "cpu", "gpu", ...
        """

        kernel = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).to(device)
        indexes = torch.nonzero(gt)

        if len(indexes) == 0:
            return 0.0

        retrieved = 0.0

        for i in range(len(indexes)):
            batch, f, t = (
                int(indexes[i][0].item()),
                int(indexes[i][1].item()),
                int(indexes[i][2].item()),
            )

            if t == 0:
                if f == 0:
                    retrieved += torch.sum(
                        predicted[batch, f : f + 2, t : t + 2] * kernel[:2, :2]
                    ).item()
                elif f + 1 >= predicted.shape[1]:
                    retrieved += torch.sum(
                        predicted[batch, f - 1 : f + 1, t : t + 2] * kernel[:2, :2]
                    ).item()
                else:
                    retrieved += torch.sum(
                        predicted[batch, f - 1 : f + 2, t : t + 2] * kernel[:, :2]
                    ).item()

            elif t + 1 >= predicted.shape[2]:
                if f == 0:
                    retrieved += torch.sum(
                        predicted[batch, f : f + 2, t - 1 : t + 1] * kernel[:2, :2]
                    ).item()
                elif f + 1 >= predicted.shape[1]:
                    retrieved += torch.sum(
                        predicted[batch, f - 1 : f + 1, t - 1 : t + 1] * kernel[:2, :2]
                    ).item()
                else:
                    retrieved += torch.sum(
                        predicted[batch, f - 1 : f + 2, t - 1 : t + 1] * kernel[:, :2]
                    ).item()
            else:
                if f == 0:
                    retrieved += torch.sum(
                        predicted[batch, f : f + 2, t - 1 : t + 2] * kernel[:2, :]
                    ).item()
                elif f + 1 >= predicted.shape[1]:
                    retrieved += torch.sum(
                        predicted[batch, f - 1 : f + 1, t - 1 : t + 2] * kernel[:2, :]
                    ).item()
                else:
                    retrieved += torch.sum(
                        predicted[batch, f - 1 : f + 2, t - 1 : t + 2] * kernel
                    ).item()

        return retrieved / len(indexes)


class Precision(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicted: torch.Tensor, gt: torch.Tensor, device: str = "cpu"
    ) -> float:
        """
        Compute the precision between predicted binary peak mask and ground thruth.
        ---
        Args:
            predicted (torch.Tensor):
                Predicted mask.
            gt (torch.Tensor):
                Ground truth.
            device (str):
                "cpu", "gpu", ...
        """

        kernel = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).to(device)
        indexes = torch.nonzero(predicted)

        if len(indexes) == 0:
            return 0.0

        relevant = 0.0
        for i in range(len(indexes)):
            batch, f, t = (
                int(indexes[i][0].item()),
                int(indexes[i][1].item()),
                int(indexes[i][2].item()),
            )

            if t == 0:
                if f == 0:
                    relevant += torch.sum(
                        gt[batch, f : f + 2, t : t + 2] * kernel[:2, :2]
                    ).item()
                elif f + 1 >= predicted.shape[1]:
                    relevant += torch.sum(
                        gt[batch, f - 1 : f + 1, t : t + 2] * kernel[:2, :2]
                    ).item()
                else:
                    relevant += torch.sum(
                        gt[batch, f - 1 : f + 2, t : t + 2] * kernel[:, :2]
                    ).item()

            elif t + 1 >= predicted.shape[2]:
                if f == 0:
                    relevant += torch.sum(
                        gt[batch, f : f + 2, t - 1 : t + 1] * kernel[:2, :2]
                    ).item()
                elif f + 1 >= predicted.shape[1]:
                    relevant += torch.sum(
                        gt[batch, f - 1 : f + 1, t - 1 : t + 1] * kernel[:2, :2]
                    ).item()
                else:
                    relevant += torch.sum(
                        gt[batch, f - 1 : f + 2, t - 1 : t + 1] * kernel[:, :2]
                    ).item()
            else:
                if f == 0:
                    relevant += torch.sum(
                        gt[batch, f : f + 2, t - 1 : t + 2] * kernel[:2, :]
                    ).item()
                elif f + 1 >= predicted.shape[1]:
                    relevant += torch.sum(
                        gt[batch, f - 1 : f + 1, t - 1 : t + 2] * kernel[:2, :]
                    ).item()
                else:
                    relevant += torch.sum(
                        gt[batch, f - 1 : f + 2, t - 1 : t + 2] * kernel
                    ).item()

        return relevant / len(indexes)


class F1score(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.prec = Precision()
        self.rec = Recall()

    def forward(
        self, predicted: torch.Tensor, gt: torch.Tensor, device: str = "cpu"
    ) -> float:
        """
        Compute the F1-score between predicted binary peak mask and ground thruth.
        ---
        Args:
            predicted (torch.Tensor):
                Predicted mask.
            gt (torch.Tensor):
                Ground truth.
            device (str):
                "cpu", "gpu", ...
        """
        p = self.prec(predicted, gt, device)
        r = self.rec(predicted, gt, device)

        if math.isclose(p + r, 0.0):
            return 0.0
        else:
            return float(2.0 * (p * r) / (p + r))
