#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
SRC = CODE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fcd_latent_ratio.data import build_subject_index
from fcd_latent_ratio.metrics import (
    dice_score_from_masks,
    hd95_score_from_masks,
    iou_score_from_masks,
    precision_score_from_masks,
    recall_score_from_masks,
    sensitivity_score_from_masks,
    specificity_score_from_masks,
)


METRIC_NAMES = ("dice", "hd95", "iou", "precision", "recall", "sensitivity", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved predicted masks and compute segmentation metrics.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run output directory containing fold_*/validation manifests.")
    parser.add_argument("--data-root", type=Path, help="Optional override for config.snapshot.json data_root.")
    parser.add_argument("--output-dir", type=Path, help="Optional override for analysis outputs. Defaults to <run-dir>/prediction_analysis.")
    parser.add_argument(
        "--subject-ids",
        nargs="+",
        help="Optional subset of subject IDs to analyze.",
    )
    return parser.parse_args()


def _load_mask(path: Path) -> np.ndarray:
    return (np.asarray(nib.load(str(path)).get_fdata()) > 0.5).astype(np.uint8)


def _aggregate_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float] | int]:
    summary: dict[str, dict[str, float] | int] = {"num_subjects": len(rows)}
    for metric_name in METRIC_NAMES:
        values = [float(row[metric_name]) for row in rows if metric_name in row and math.isfinite(float(row[metric_name]))]
        if not values:
            summary[metric_name] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        summary[metric_name] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _summary_row(split_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    aggregated = _aggregate_rows(rows)
    lesion_positive_rows = [row for row in rows if bool(row["is_lesion_positive"])]
    control_rows = [row for row in rows if bool(row["is_control"])]
    missed_fcd_rows = [row for row in rows if bool(row["is_missed_fcd"])]
    fp_control_rows = [row for row in rows if bool(row["is_fp_control"])]
    row: dict[str, object] = {
        "split": split_name,
        "num_subjects": aggregated["num_subjects"],
        "num_lesion_positive": len(lesion_positive_rows),
        "num_controls": len(control_rows),
        "num_missed_fcd": len(missed_fcd_rows),
        "num_fp_controls": len(fp_control_rows),
    }
    for metric_name in METRIC_NAMES:
        metric_summary = aggregated[metric_name]
        row[f"{metric_name}_mean"] = metric_summary["mean"]
        row[f"{metric_name}_std"] = metric_summary["std"]
    return row


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.run_dir / "prediction_analysis"


def _select_fields(row: dict[str, object], fieldnames: list[str]) -> dict[str, object]:
    return {fieldname: row.get(fieldname) for fieldname in fieldnames}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.snapshot.json"
    if not config_path.exists():
        raise SystemExit(f"Missing config snapshot: {config_path}")

    config = json.loads(config_path.read_text())
    data_root = args.data_root or Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

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

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_subject_rows: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []

    for fold_dir in sorted(path for path in run_dir.glob("fold_*") if path.is_dir()):
        manifest_path = fold_dir / "validation" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        fold_rows: list[dict[str, float | str | int]] = []
        for item in manifest:
            subject_id = item["subject_id"]
            if requested_subject_ids and subject_id not in requested_subject_ids:
                continue
            if subject_id not in subject_map:
                raise SystemExit(f"Subject '{subject_id}' from {manifest_path} was not found under {data_root}")
            subject = subject_map[subject_id]
            if subject.label_path is None or not subject.label_path.exists():
                raise SystemExit(f"Ground-truth label is missing for subject '{subject_id}'")

            prediction_path = Path(item["prediction_path"])
            if not prediction_path.is_absolute():
                prediction_path = (fold_dir / "validation" / prediction_path.name).resolve()
            if not prediction_path.exists():
                raise SystemExit(f"Prediction file does not exist: {prediction_path}")

            pred_mask = _load_mask(prediction_path)
            target_mask = _load_mask(subject.label_path)
            if pred_mask.shape != target_mask.shape:
                raise SystemExit(
                    f"Shape mismatch for subject '{subject_id}': prediction {pred_mask.shape} vs label {target_mask.shape}"
                )

            row = {
                "fold": fold_dir.name,
                "subject_id": subject_id,
                "dataset": subject.dataset,
                "cohort_role": subject.cohort_role,
                "prediction_path": str(prediction_path),
                "label_path": str(subject.label_path),
                "pred_voxels": int(pred_mask.sum()),
                "label_voxels": int(target_mask.sum()),
                "dice": dice_score_from_masks(pred_mask, target_mask),
                "hd95": hd95_score_from_masks(pred_mask, target_mask),
                "iou": iou_score_from_masks(pred_mask, target_mask),
                "precision": precision_score_from_masks(pred_mask, target_mask),
                "recall": recall_score_from_masks(pred_mask, target_mask),
                "sensitivity": sensitivity_score_from_masks(pred_mask, target_mask),
                "specificity": specificity_score_from_masks(pred_mask, target_mask),
            }
            row["is_lesion_positive"] = row["label_voxels"] > 0
            row["is_control"] = subject.cohort_role == "control"
            row["is_missed_fcd"] = row["is_lesion_positive"] and row["pred_voxels"] == 0
            row["is_fp_control"] = row["is_control"] and row["pred_voxels"] > 0
            per_subject_rows.append(row)
            fold_rows.append(row)

        fold_summaries.append(
            {
                "fold": fold_dir.name,
                "metrics": _aggregate_rows(fold_rows),
            }
        )

    if not per_subject_rows:
        raise SystemExit(f"No predictions were analyzed under {run_dir}")

    csv_path = output_dir / "per_subject_metrics.csv"
    fieldnames = [
        "fold",
        "subject_id",
        "dataset",
        "cohort_role",
        "pred_voxels",
        "label_voxels",
        "is_lesion_positive",
        "is_control",
        "is_missed_fcd",
        "is_fp_control",
        *METRIC_NAMES,
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_subject_rows)

    paper_summary_rows = [_summary_row(str(fold_summary["fold"]), [row for row in per_subject_rows if row["fold"] == fold_summary["fold"]]) for fold_summary in fold_summaries]
    paper_summary_rows.append(_summary_row("overall", per_subject_rows))
    paper_csv_path = output_dir / "foldwise_overall_metrics.csv"
    paper_fieldnames = [
        "split",
        "num_subjects",
        "num_lesion_positive",
        "num_controls",
        "num_missed_fcd",
        "num_fp_controls",
    ]
    for metric_name in METRIC_NAMES:
        paper_fieldnames.extend([f"{metric_name}_mean", f"{metric_name}_std"])
    with paper_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paper_fieldnames)
        writer.writeheader()
        writer.writerows(paper_summary_rows)

    failure_csv_path = output_dir / "paper_error_cases.csv"
    failure_fieldnames = [
        "fold",
        "subject_id",
        "dataset",
        "cohort_role",
        "pred_voxels",
        "label_voxels",
        "is_missed_fcd",
        "is_fp_control",
        "dice",
        "hd95",
    ]
    failure_rows = [
        _select_fields(row, failure_fieldnames)
        for row in per_subject_rows
        if bool(row["is_missed_fcd"]) or bool(row["is_fp_control"])
    ]
    with failure_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)

    summary = {
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "num_subjects": len(per_subject_rows),
        "aggregate_metrics": _aggregate_rows(per_subject_rows),
        "num_lesion_positive": sum(1 for row in per_subject_rows if bool(row["is_lesion_positive"])),
        "num_controls": sum(1 for row in per_subject_rows if bool(row["is_control"])),
        "num_missed_fcd": sum(1 for row in per_subject_rows if bool(row["is_missed_fcd"])),
        "num_fp_controls": sum(1 for row in per_subject_rows if bool(row["is_fp_control"])),
        "folds": fold_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote per-subject metrics to {csv_path}")
    print(f"Wrote fold-wise and overall metrics to {paper_csv_path}")
    print(f"Wrote missed-FCD / FP-control cases to {failure_csv_path}")
    print(f"Wrote aggregate summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
