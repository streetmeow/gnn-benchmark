
import torch


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float((y_true == y_pred).sum().item() / max(1, y_true.numel()))
