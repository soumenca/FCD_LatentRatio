#!/usr/bin/env python3
"""Compute segmentation metrics from saved fold validation predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
import sys

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
SRC = CODE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fcd_latent_ratio.data import build_subject_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute voxel-wise segmentation metrics for saved fold validation "
            "predictions and export per-case plus paper-ready summaries."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing fold_* folders with validation/manifest.json files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional override for config.snapshot.json data_root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional override for analysis outputs. Defaults to <run-dir>/prediction_analysis.",
    )
    parser.add_argument(
        "--subject-ids",
        nargs="+",
        help="Optional subset of subject IDs to analyze.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float, both_empty_value: float = 1.0) -> float:
    if denominator == 0:
        return both_empty_value
    return numerator / denominator


def load_binary_mask(path: Path) -> tuple[np.ndarray, float]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    spacing = image.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(spacing))
    return data > 0.5, voxel_volume_mm3


def compute_case_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, voxel_volume_mm3: float) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    pred_voxels = float(pred.sum())
    gt_voxels = float(gt.sum())

    dice = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp, both_empty_value=1.0 if gt_voxels == 0 else 0.0)
    recall = safe_divide(tp, tp + fn)
    sensitivity = recall
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    fpr = safe_divide(fp, fp + tn, both_empty_value=0.0)
    fnr = safe_divide(fn, fn + tp, both_empty_value=0.0)
    volume_diff_voxels = pred_voxels - gt_voxels
    abs_volume_diff_voxels = abs(volume_diff_voxels)
    volume_diff_mm3 = volume_diff_voxels * voxel_volume_mm3
    abs_volume_diff_mm3 = abs(volume_diff_mm3)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "pred_volume_mm3": pred_voxels * voxel_volume_mm3,
        "gt_volume_mm3": gt_voxels * voxel_volume_mm3,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "volume_diff_voxels": volume_diff_voxels,
        "abs_volume_diff_voxels": abs_volume_diff_voxels,
        "volume_diff_mm3": volume_diff_mm3,
        "abs_volume_diff_mm3": abs_volume_diff_mm3,
        "gt_nonempty": float(gt_voxels > 0),
        "pred_nonempty": float(pred_voxels > 0),
    }


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else float("nan")


def aggregate_metrics(rows: list[dict[str, object]], group_name: str) -> dict[str, object]:
    metric_cols = [
        "dice",
        "iou",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "pred_volume_mm3",
        "gt_volume_mm3",
        "volume_diff_mm3",
        "abs_volume_diff_mm3",
    ]
    summary: dict[str, object] = {
        "group": group_name,
        "num_cases": int(len(rows)),
        "num_gt_positive_cases": int(sum(float(row["gt_nonempty"]) for row in rows)),
        "num_pred_positive_cases": int(sum(float(row["pred_nonempty"]) for row in rows)),
        "num_zero_dice_cases": int(sum(float(row["dice"]) == 0.0 for row in rows)),
        "zero_dice_rate": float(sum(float(row["dice"]) == 0.0 for row in rows) / len(rows)) if rows else 0.0,
        "num_zero_dice_and_pred_positive_cases": int(
            sum((float(row["dice"]) == 0.0) and (float(row["pred_nonempty"]) == 1.0) for row in rows)
        ),
        "num_zero_dice_and_pred_empty_cases": int(
            sum((float(row["dice"]) == 0.0) and (float(row["pred_nonempty"]) == 0.0) for row in rows)
        ),
        "num_overlap_detected_cases": int(sum(float(row["dice"]) > 0.0 for row in rows)),
    }
    for col in metric_cols:
        values = [float(row[col]) for row in rows]
        summary[f"{col}_mean"] = _mean(values)
        summary[f"{col}_std"] = _std(values)
        summary[f"{col}_median"] = _median(values)
    return summary


def aggregate_positive_dice_metrics(rows: list[dict[str, object]], group_name: str) -> dict[str, object]:
    filtered = [row for row in rows if float(row["dice"]) > 0.0]
    summary: dict[str, object] = {
        "group": f"{group_name}_dice_gt_0",
        "num_cases": int(len(filtered)),
        "detected_only_rate": float(len(filtered) / len(rows)) if rows else 0.0,
    }
    if not filtered:
        return summary

    metric_cols = [
        "dice",
        "iou",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "pred_volume_mm3",
        "gt_volume_mm3",
        "volume_diff_mm3",
        "abs_volume_diff_mm3",
    ]
    for col in metric_cols:
        values = [float(row[col]) for row in filtered]
        summary[f"{col}_mean"] = _mean(values)
        summary[f"{col}_std"] = _std(values)
        summary[f"{col}_median"] = _median(values)
    return summary


def build_paper_summary_rows(
    fcd_summary: dict[str, object],
    control_summary: dict[str, object],
    fcd_positive_dice_summary: dict[str, object],
    control_positive_dice_summary: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    main_rows = [
        {
            "group": "FCD",
            "num_cases": fcd_summary["num_cases"],
            "dice_mean": fcd_summary["dice_mean"],
            "dice_std": fcd_summary["dice_std"],
            "sensitivity_mean": fcd_summary["sensitivity_mean"],
            "sensitivity_std": fcd_summary["sensitivity_std"],
            "precision_mean": fcd_summary["precision_mean"],
            "precision_std": fcd_summary["precision_std"],
            "zero_dice_cases": fcd_summary["fcd_zero_dice_cases"],
            "zero_dice_rate": fcd_summary["fcd_zero_dice_rate"],
            "control_cases_with_predicted_lesion": float("nan"),
            "control_predicted_lesion_rate": float("nan"),
        },
        {
            "group": "control",
            "num_cases": control_summary["num_cases"],
            "dice_mean": float("nan"),
            "dice_std": float("nan"),
            "sensitivity_mean": float("nan"),
            "sensitivity_std": float("nan"),
            "precision_mean": float("nan"),
            "precision_std": float("nan"),
            "zero_dice_cases": float("nan"),
            "zero_dice_rate": float("nan"),
            "control_cases_with_predicted_lesion": control_summary["control_cases_with_predicted_lesion"],
            "control_predicted_lesion_rate": control_summary["control_predicted_lesion_rate"],
        },
    ]

    detected_only_rows = [
        {
            "group": "FCD_detected_only",
            "num_cases_dice_gt_0": fcd_positive_dice_summary["num_cases"],
            "detected_only_rate": fcd_positive_dice_summary["detected_only_rate"],
            "dice_mean": fcd_positive_dice_summary.get("dice_mean", float("nan")),
            "dice_std": fcd_positive_dice_summary.get("dice_std", float("nan")),
            "sensitivity_mean": fcd_positive_dice_summary.get("sensitivity_mean", float("nan")),
            "sensitivity_std": fcd_positive_dice_summary.get("sensitivity_std", float("nan")),
            "precision_mean": fcd_positive_dice_summary.get("precision_mean", float("nan")),
            "precision_std": fcd_positive_dice_summary.get("precision_std", float("nan")),
        },
        {
            "group": "control_detected_only",
            "num_cases_dice_gt_0": control_positive_dice_summary["num_cases"],
            "detected_only_rate": control_positive_dice_summary["detected_only_rate"],
            "dice_mean": control_positive_dice_summary.get("dice_mean", float("nan")),
            "dice_std": control_positive_dice_summary.get("dice_std", float("nan")),
            "sensitivity_mean": control_positive_dice_summary.get("sensitivity_mean", float("nan")),
            "sensitivity_std": control_positive_dice_summary.get("sensitivity_std", float("nan")),
            "precision_mean": control_positive_dice_summary.get("precision_mean", float("nan")),
            "precision_std": control_positive_dice_summary.get("precision_std", float("nan")),
        },
    ]
    return main_rows, detected_only_rows


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.run_dir / "prediction_analysis"


def _select_fields(row: dict[str, object], fieldnames: list[str]) -> dict[str, object]:
    return {fieldname: row.get(fieldname) for fieldname in fieldnames}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([_select_fields(row, fieldnames) for row in rows])


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
    fold_summaries: list[dict[str, object]] = []
    fold_positive_dice_summaries: list[dict[str, object]] = []

    for fold_dir in fold_dirs:
        manifest_path = fold_dir / "validation" / "manifest.json"
        if not manifest_path.exists():
            print(f"Skipping {fold_dir.name}: missing manifest {manifest_path}")
            continue

        manifest = json.loads(manifest_path.read_text())
        fold_rows: list[dict[str, object]] = []
        for item in manifest:
            subject_id = item["subject_id"]
            if requested_subject_ids and subject_id not in requested_subject_ids:
                continue
            if subject_id not in subject_map:
                raise FileNotFoundError(f"Subject '{subject_id}' from {manifest_path} was not found under {data_root}")
            subject = subject_map[subject_id]
            if subject.label_path is None or not subject.label_path.exists():
                raise FileNotFoundError(f"Missing ground-truth label for '{subject_id}'")

            prediction_path = Path(item["prediction_path"])
            if not prediction_path.is_absolute():
                prediction_path = (fold_dir / "validation" / prediction_path.name).resolve()
            if not prediction_path.exists():
                raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")

            pred_mask, pred_voxel_volume = load_binary_mask(prediction_path)
            gt_mask, gt_voxel_volume = load_binary_mask(subject.label_path)
            if pred_mask.shape != gt_mask.shape:
                raise ValueError(
                    f"Shape mismatch for {subject_id}: pred {pred_mask.shape} vs gt {gt_mask.shape}"
                )

            voxel_volume_mm3 = gt_voxel_volume
            if not np.isclose(pred_voxel_volume, gt_voxel_volume):
                print(
                    f"Warning: voxel volume mismatch for {subject_id}; using ground-truth "
                    f"volume {gt_voxel_volume:.6f} mm^3"
                )

            metrics = compute_case_metrics(pred_mask, gt_mask, voxel_volume_mm3)
            row = {
                "fold": fold_dir.name,
                "case_id": subject_id,
                "group": "control" if subject.cohort_role == "control" else "FCD",
                "dataset": subject.dataset,
                "cohort_role": subject.cohort_role,
                "prediction_path": str(prediction_path),
                "label_path": str(subject.label_path),
                **metrics,
            }
            rows.append(row)
            fold_rows.append(row)

        if fold_rows:
            fold_summaries.append(aggregate_metrics(fold_rows, fold_dir.name))
            fold_positive_dice_summaries.append(aggregate_positive_dice_metrics(fold_rows, fold_dir.name))

    if not rows:
        raise RuntimeError(f"No prediction/label pairs were processed from {run_dir}")

    rows.sort(key=lambda row: (str(row["fold"]), str(row["case_id"])))
    overall_summary = aggregate_metrics(rows, "overall")

    fcd_rows = [row for row in rows if row["group"] == "FCD"]
    control_rows = [row for row in rows if row["group"] == "control"]
    fcd_summary = aggregate_metrics(fcd_rows, "FCD")
    control_summary = aggregate_metrics(control_rows, "control")
    overall_positive_dice_summary = aggregate_positive_dice_metrics(rows, "overall")
    fcd_positive_dice_summary = aggregate_positive_dice_metrics(fcd_rows, "FCD")
    control_positive_dice_summary = aggregate_positive_dice_metrics(control_rows, "control")

    fcd_summary["fcd_zero_dice_cases"] = fcd_summary["num_zero_dice_cases"]
    fcd_summary["fcd_zero_dice_rate"] = fcd_summary["zero_dice_rate"]
    control_summary["control_cases_with_predicted_lesion"] = control_summary["num_pred_positive_cases"]
    control_summary["control_predicted_lesion_rate"] = safe_divide(
        float(control_summary["num_pred_positive_cases"]),
        float(control_summary["num_cases"]),
        both_empty_value=0.0,
    )

    paper_rows, paper_detected_only_rows = build_paper_summary_rows(
        fcd_summary,
        control_summary,
        fcd_positive_dice_summary,
        control_positive_dice_summary,
    )

    error_case_rows = [
        row for row in rows if (row["group"] == "FCD" and float(row["dice"]) == 0.0)
        or (row["group"] == "control" and float(row["pred_nonempty"]) == 1.0)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_csv = output_dir / "segmentation_metrics_per_case.csv"
    fcd_per_case_csv = output_dir / "segmentation_metrics_per_case_fcd.csv"
    control_per_case_csv = output_dir / "segmentation_metrics_per_case_control.csv"
    per_fold_csv = output_dir / "segmentation_metrics_per_fold.csv"
    positive_dice_csv = output_dir / "segmentation_metrics_dice_gt_0_summary.csv"
    paper_summary_csv = output_dir / "segmentation_metrics_paper_summary.csv"
    paper_detected_only_csv = output_dir / "segmentation_metrics_paper_summary_detected_only.csv"
    error_cases_csv = output_dir / "segmentation_metrics_paper_error_cases.csv"
    summary_json = output_dir / "segmentation_metrics_summary.json"

    csv_excluded_fields = {"prediction_path", "label_path"}
    common_fieldnames = [field for field in rows[0].keys() if field not in csv_excluded_fields]
    _write_csv(per_case_csv, rows, common_fieldnames)
    _write_csv(fcd_per_case_csv, fcd_rows, common_fieldnames)
    _write_csv(control_per_case_csv, control_rows, common_fieldnames)
    _write_csv(per_fold_csv, fold_summaries + [fcd_summary, control_summary, overall_summary], list((fold_summaries + [fcd_summary, control_summary, overall_summary])[0].keys()))
    _write_csv(
        positive_dice_csv,
        fold_positive_dice_summaries + [fcd_positive_dice_summary, control_positive_dice_summary, overall_positive_dice_summary],
        list((fold_positive_dice_summaries + [fcd_positive_dice_summary, control_positive_dice_summary, overall_positive_dice_summary])[0].keys()),
    )
    _write_csv(paper_summary_csv, paper_rows, list(paper_rows[0].keys()))
    _write_csv(paper_detected_only_csv, paper_detected_only_rows, list(paper_detected_only_rows[0].keys()))
    if error_case_rows:
        _write_csv(error_cases_csv, error_case_rows, common_fieldnames)
    else:
        _write_csv(error_cases_csv, [], common_fieldnames)

    summary_json.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "data_root": str(data_root),
                "folds": fold_summaries,
                "folds_dice_gt_0": fold_positive_dice_summaries,
                "fcd_summary": fcd_summary,
                "control_summary": control_summary,
                "fcd_summary_dice_gt_0": fcd_positive_dice_summary,
                "control_summary_dice_gt_0": control_positive_dice_summary,
                "overall_dice_gt_0": overall_positive_dice_summary,
                "overall": overall_summary,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote per-case metrics to {per_case_csv}")
    print(f"Wrote FCD per-case metrics to {fcd_per_case_csv}")
    print(f"Wrote control per-case metrics to {control_per_case_csv}")
    print(f"Wrote per-fold metrics to {per_fold_csv}")
    print(f"Wrote Dice>0 summary metrics to {positive_dice_csv}")
    print(f"Wrote paper summary metrics to {paper_summary_csv}")
    print(f"Wrote paper detected-only metrics to {paper_detected_only_csv}")
    print(f"Wrote paper error cases to {error_cases_csv}")
    print(f"Wrote summary JSON to {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
