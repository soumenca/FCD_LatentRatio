from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33, smooth: float = 1e-5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).reshape(-1)
        target = target.reshape(-1)

        true_pos = (probs * target).sum()
        false_neg = ((1.0 - probs) * target).sum()
        false_pos = (probs * (1.0 - target)).sum()

        tversky_index = (true_pos + self.smooth) / (
            true_pos + (self.alpha * false_neg) + (self.beta * false_pos) + self.smooth
        )
        return torch.pow(1.0 - tversky_index, 1.0 / self.gamma)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = probs * target + (1.0 - probs) * (1.0 - target)
        alpha_factor = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        return (alpha_factor * modulating_factor * ce_loss).mean()


class FocalTverskyFocalLoss(nn.Module):
    def __init__(
        self,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 1.33,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-5,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal = SigmoidFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            smooth=smooth,
        )
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal(logits, target)
        loss_tversky = self.tversky(logits, target)
        return (self.focal_weight * loss_focal) + (self.tversky_weight * loss_tversky)


def build_loss(config: dict | None) -> nn.Module:
    config = config or {}
    name = str(config.get("name", "dice_bce")).lower()

    if name == "dice_bce":
        return DiceBCELoss(bce_weight=float(config.get("bce_weight", 0.5)))

    if name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=float(config.get("alpha", 0.7)),
            beta=float(config.get("beta", 0.3)),
            gamma=float(config.get("gamma", 1.33)),
            smooth=float(config.get("smooth", 1e-5)),
        )

    if name in {"focal_tversky_focal", "ftl_focal", "focal_tversky_combo"}:
        return FocalTverskyFocalLoss(
            tversky_alpha=float(config.get("alpha", 0.7)),
            tversky_beta=float(config.get("beta", 0.3)),
            tversky_gamma=float(config.get("gamma", 1.33)),
            focal_alpha=float(config.get("focal_alpha", 0.25)),
            focal_gamma=float(config.get("focal_gamma", 2.0)),
            smooth=float(config.get("smooth", 1e-5)),
            focal_weight=float(config.get("focal_weight", 1.0)),
            tversky_weight=float(config.get("tversky_weight", 1.0)),
        )

    raise ValueError(f"Unsupported loss name: {name}")
