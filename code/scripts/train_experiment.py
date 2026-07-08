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
    output_dir = fit_experiment(config=config, repo_root=REPO_ROOT)
    print(f"Finished experiment. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
