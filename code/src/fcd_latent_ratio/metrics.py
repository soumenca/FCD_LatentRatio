from __future__ import annotations

import torch


def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean()


def iou_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean()


def precision_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    true_positive = (pred * target).sum(dim=dims)
    predicted_positive = pred.sum(dim=dims)
    return ((true_positive + 1e-6) / (predicted_positive + 1e-6)).mean()


def recall_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    true_positive = (pred * target).sum(dim=dims)
    target_positive = target.sum(dim=dims)
    return ((true_positive + 1e-6) / (target_positive + 1e-6)).mean()
