from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


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


def sensitivity_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return recall_score_from_logits(logits, target, threshold=threshold)


def specificity_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    true_negative = ((1.0 - pred) * (1.0 - target)).sum(dim=dims)
    target_negative = (1.0 - target).sum(dim=dims)
    return ((true_negative + 1e-6) / (target_negative + 1e-6)).mean()


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if not mask_a.any() and not mask_b.any():
        return np.zeros(1, dtype=np.float32)
    if not mask_a.any() or not mask_b.any():
        return np.array([np.inf], dtype=np.float32)

    structure = ndimage.generate_binary_structure(mask_a.ndim, 1)
    surface_a = np.logical_xor(mask_a, ndimage.binary_erosion(mask_a, structure=structure, border_value=0))
    surface_b = np.logical_xor(mask_b, ndimage.binary_erosion(mask_b, structure=structure, border_value=0))

    distances_to_b = ndimage.distance_transform_edt(~surface_b)
    distances_to_a = ndimage.distance_transform_edt(~surface_a)

    return np.concatenate(
        [
            distances_to_b[surface_a],
            distances_to_a[surface_b],
        ]
    ).astype(np.float32)


def dice_score_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    target = target_mask.astype(np.float32) > 0.5
    intersection = float(np.logical_and(pred, target).sum())
    denom = float(pred.sum() + target.sum())
    return (2.0 * intersection + 1e-6) / (denom + 1e-6)


def iou_score_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    target = target_mask.astype(np.float32) > 0.5
    intersection = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    return (intersection + 1e-6) / (union + 1e-6)


def precision_score_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    target = target_mask.astype(np.float32) > 0.5
    true_positive = float(np.logical_and(pred, target).sum())
    predicted_positive = float(pred.sum())
    return (true_positive + 1e-6) / (predicted_positive + 1e-6)


def recall_score_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    target = target_mask.astype(np.float32) > 0.5
    true_positive = float(np.logical_and(pred, target).sum())
    target_positive = float(target.sum())
    return (true_positive + 1e-6) / (target_positive + 1e-6)


def sensitivity_score_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    return recall_score_from_masks(pred_mask, target_mask)


def specificity_score_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = pred_mask.astype(np.float32) > 0.5
    target = target_mask.astype(np.float32) > 0.5
    true_negative = float(np.logical_and(~pred, ~target).sum())
    target_negative = float((~target).sum())
    return (true_negative + 1e-6) / (target_negative + 1e-6)


def hd95_score_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    distances = _surface_distances(pred_mask.astype(np.uint8), target_mask.astype(np.uint8))
    if not np.all(np.isfinite(distances)):
        return float("inf")
    return float(np.percentile(distances, 95))


def hd95_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).detach().cpu().numpy().astype(np.uint8)
    target_np = (target > 0.5).detach().cpu().numpy().astype(np.uint8)

    values: list[float] = []
    for pred_sample, target_sample in zip(pred, target_np):
        pred_mask = np.squeeze(pred_sample, axis=0)
        target_mask = np.squeeze(target_sample, axis=0)
        distances = _surface_distances(pred_mask, target_mask)
        values.append(float(np.percentile(distances, 95)))
    return torch.tensor(float(np.mean(values)), dtype=torch.float32)
