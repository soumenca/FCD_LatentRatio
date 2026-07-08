from __future__ import annotations

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss
