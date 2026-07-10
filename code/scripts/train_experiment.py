#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
SRC = CODE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fcd_latent_ratio.training import fit_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline or CRIL-based FCD segmentation experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON experiment config.")
    parser.add_argument("--data-root", type=Path, help="Optional override for config['data_root'].")
    parser.add_argument("--output-root", type=Path, help="Optional override for config['output_root'].")
    parser.add_argument("--epochs", type=int, help="Optional override for config['epochs'].")
    parser.add_argument("--fold-index", type=int, help="Optional zero-based CV fold index to run by itself.")
    parser.add_argument(
        "--loss-name",
        choices=["dice_bce", "focal_tversky", "focal_tversky_focal"],
        help="Optional loss override. Default stays whatever the config specifies.",
    )
    parser.add_argument("--loss-bce-weight", type=float, help="Override Dice+BCE BCE weight.")
    parser.add_argument("--loss-alpha", type=float, help="Override Tversky alpha or combo Tversky alpha.")
    parser.add_argument("--loss-beta", type=float, help="Override Tversky beta or combo Tversky beta.")
    parser.add_argument("--loss-gamma", type=float, help="Override Focal Tversky gamma.")
    parser.add_argument("--loss-smooth", type=float, help="Override smoothing term for Tversky-style losses.")
    parser.add_argument("--loss-focal-alpha", type=float, help="Override focal alpha for the combo loss.")
    parser.add_argument("--loss-focal-gamma", type=float, help="Override focal gamma for the combo loss.")
    parser.add_argument("--loss-focal-weight", type=float, help="Override focal component weight for the combo loss.")
    parser.add_argument(
        "--loss-tversky-weight",
        type=float,
        help="Override Focal Tversky component weight for the combo loss.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.data_root is not None:
        config["data_root"] = str(args.data_root)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.fold_index is not None:
        config["fold_index"] = args.fold_index
    loss_config = dict(config.get("loss", {}))
    if args.loss_name is not None:
        loss_config["name"] = args.loss_name
    if args.loss_bce_weight is not None:
        loss_config["bce_weight"] = args.loss_bce_weight
    if args.loss_alpha is not None:
        loss_config["alpha"] = args.loss_alpha
    if args.loss_beta is not None:
        loss_config["beta"] = args.loss_beta
    if args.loss_gamma is not None:
        loss_config["gamma"] = args.loss_gamma
    if args.loss_smooth is not None:
        loss_config["smooth"] = args.loss_smooth
    if args.loss_focal_alpha is not None:
        loss_config["focal_alpha"] = args.loss_focal_alpha
    if args.loss_focal_gamma is not None:
        loss_config["focal_gamma"] = args.loss_focal_gamma
    if args.loss_focal_weight is not None:
        loss_config["focal_weight"] = args.loss_focal_weight
    if args.loss_tversky_weight is not None:
        loss_config["tversky_weight"] = args.loss_tversky_weight
    if loss_config:
        config["loss"] = loss_config
    output_dir = fit_experiment(config=config, repo_root=REPO_ROOT)
    print(f"Finished experiment. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
