#!/usr/bin/env python3
"""Compute compact segmentation metrics from saved fold validation predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
SRC = CODE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fcd_latent_ratio.data import build_subject_index


METRIC_NAMES = ("dice", "iou", "sensitivity", "precision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute compact per-subject and paper-ready segmentation summaries."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing fold_* folders.")
    parser.add_argument("--data-root", type=Path, help="Optional override for config.snapshot.json data_root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional override for analysis outputs. Defaults to <run-dir>/prediction_analysis.",
    )
    parser.add_argument("--subject-ids", nargs="+", help="Optional subset of subject IDs to analyze.")
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float, both_empty_value: float = 1.0) -> float:
    if denominator == 0:
        return both_empty_value
    return numerator / denominator


def load_binary_mask(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return np.asarray(image.get_fdata(), dtype=np.float32) > 0.5


def compute_case_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())

    pred_voxels = float(pred.sum())
    gt_voxels = float(gt.sum())

    dice = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp, both_empty_value=1.0 if gt_voxels == 0 else 0.0)
    sensitivity = safe_divide(tp, tp + fn)

    return {
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "precision": precision,
        "pred_nonempty": float(pred_voxels > 0),
        "gt_nonempty": float(gt_voxels > 0),
    }


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.run_dir / "prediction_analysis"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname) for fieldname in fieldnames})


def _subject_group(subject_id: str) -> str:
    if subject_id.startswith("FCD_"):
        return "FCD"
    if subject_id.startswith("CON_"):
        return "HC"
    raise ValueError(f"Unsupported subject ID prefix for grouping: {subject_id}")


def _metric_summary(rows: list[dict[str, object]], split_name: str) -> dict[str, object]:
    summary: dict[str, object] = {"split": split_name, "num_subjects": len(rows)}
    for metric_name in METRIC_NAMES:
        values = [float(row[metric_name]) for row in rows]
        summary[f"{metric_name}_mean"] = float(mean(values)) if values else float("nan")
        summary[f"{metric_name}_std"] = float(pstdev(values)) if len(values) > 1 else (0.0 if values else float("nan"))
    return summary


def _counts_summary(rows: list[dict[str, object]], split_name: str) -> dict[str, object]:
    hc_rows = [row for row in rows if row["group"] == "HC"]
    fcd_rows = [row for row in rows if row["group"] == "FCD"]
    return {
        "split": split_name,
        "num_hc_subjects": len(hc_rows),
        "num_fcd_subjects": len(fcd_rows),
        "hc_predicted_nonzero_mask_count": int(sum(float(row["pred_voxels"]) > 0.0 for row in hc_rows)),
        "fcd_zero_mask_count": int(sum(float(row["pred_voxels"]) == 0.0 for row in fcd_rows)),
        "fcd_zero_dice_count": int(sum(float(row["dice"]) == 0.0 for row in fcd_rows)),
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = _resolve_output_dir(args).resolve()
    config_path = run_dir / "config.snapshot.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")

    config = json.loads(config_path.read_text())
    data_root = args.data_root or Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    subjects = build_subject_index(
        data_root,
        include_controls=bool(config.get("include_controls", False)),
        dataset_format=config.get("dataset_format", "subject_dirs"),
        t1_channel_index=int(config.get("t1_channel_index", 0)),
        flair_channel_index=int(config.get("flair_channel_index", 1)),
        t1_flair_ratio_channel_index=(
            int(config["t1_flair_ratio_channel_index"])
            if config.get("t1_flair_ratio_channel_index") is not None
            else None
        ),
        flair_t1_ratio_channel_index=(
            int(config["flair_t1_ratio_channel_index"])
            if config.get("flair_t1_ratio_channel_index") is not None
            else None
        ),
    )
    subject_map = {subject.subject_id: subject for subject in subjects}
    requested_subject_ids = set(args.subject_ids or [])

    fold_dirs = sorted(path for path in run_dir.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {run_dir}")

    rows: list[dict[str, object]] = []
    for fold_dir in fold_dirs:
        manifest_path = fold_dir / "validation" / "manifest.json"
        if not manifest_path.exists():
            print(f"Skipping {fold_dir.name}: missing manifest {manifest_path}")
            continue

        manifest = json.loads(manifest_path.read_text())
        for item in manifest:
            subject_id = item["subject_id"]
            if requested_subject_ids and subject_id not in requested_subject_ids:
                continue
            subject = subject_map.get(subject_id)
            if subject is None:
                raise FileNotFoundError(f"Subject '{subject_id}' from {manifest_path} was not found under {data_root}")
            if subject.label_path is None or not subject.label_path.exists():
                raise FileNotFoundError(f"Missing ground-truth label for '{subject_id}'")

            prediction_path = Path(item["prediction_path"])
            if not prediction_path.is_absolute():
                prediction_path = (fold_dir / "validation" / prediction_path.name).resolve()
            if not prediction_path.exists():
                raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")

            pred_mask = load_binary_mask(prediction_path)
            gt_mask = load_binary_mask(subject.label_path)
            if pred_mask.shape != gt_mask.shape:
                raise ValueError(f"Shape mismatch for {subject_id}: pred {pred_mask.shape} vs gt {gt_mask.shape}")

            group = _subject_group(subject_id)
            row = {
                "fold": fold_dir.name,
                "subject_id": subject_id,
                "group": group,
                **compute_case_metrics(pred_mask, gt_mask),
            }
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No prediction/label pairs were processed from {run_dir}")

    rows.sort(key=lambda row: (str(row["fold"]), str(row["subject_id"])))
    fcd_rows = [row for row in rows if row["group"] == "FCD"]
    hc_rows = [row for row in rows if row["group"] == "HC"]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_subjects_csv = output_dir / "segmentation_metrics_all_subjects.csv"
    fcd_csv = output_dir / "segmentation_metrics_fcd_subjects.csv"
    hc_csv = output_dir / "segmentation_metrics_hc_subjects.csv"
    summary_csv = output_dir / "segmentation_metrics_foldwise_summary.csv"
    counts_csv = output_dir / "segmentation_metrics_error_counts.csv"

    subject_fieldnames = ["fold", "subject_id", "group", "pred_voxels", "gt_voxels", *METRIC_NAMES]
    _write_csv(all_subjects_csv, rows, subject_fieldnames)
    _write_csv(fcd_csv, fcd_rows, subject_fieldnames)
    _write_csv(hc_csv, hc_rows, subject_fieldnames)

    foldwise_rows = [
        _metric_summary([row for row in fcd_rows if row["fold"] == fold_dir.name], fold_dir.name)
        for fold_dir in fold_dirs
        if any(row["fold"] == fold_dir.name for row in fcd_rows)
    ]
    foldwise_rows.append(_metric_summary(fcd_rows, "overall"))
    summary_fieldnames = ["split", "num_subjects", "dice_mean", "dice_std", "iou_mean", "iou_std", "sensitivity_mean", "sensitivity_std", "precision_mean", "precision_std"]
    _write_csv(summary_csv, foldwise_rows, summary_fieldnames)

    count_rows = [_counts_summary([row for row in rows if row["fold"] == fold_dir.name], fold_dir.name) for fold_dir in fold_dirs if any(row["fold"] == fold_dir.name for row in rows)]
    count_rows.append(_counts_summary(rows, "overall"))
    count_fieldnames = [
        "split",
        "num_hc_subjects",
        "num_fcd_subjects",
        "hc_predicted_nonzero_mask_count",
        "fcd_zero_mask_count",
        "fcd_zero_dice_count",
    ]
    _write_csv(counts_csv, count_rows, count_fieldnames)

    print(f"Wrote all-subject metrics to {all_subjects_csv}")
    print(f"Wrote FCD-only metrics to {fcd_csv}")
    print(f"Wrote HC-only metrics to {hc_csv}")
    print(f"Wrote fold-wise and overall summary to {summary_csv}")
    print(f"Wrote HC/FCD error counts to {counts_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
