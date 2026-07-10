from __future__ import annotations

import json
import random
from statistics import mean, pstdev
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    Patch3DSegmentationDataset,
    SubjectSample,
    build_input_channels,
    build_cross_validation_folds,
    build_subject_index,
    load_subject_arrays,
    pad_if_needed,
    split_subjects,
    zscore_inside_mask,
)
from .losses import build_loss
from .metrics import (
    dice_score_from_logits,
    hd95_score_from_logits,
    iou_score_from_logits,
    precision_score_from_logits,
    recall_score_from_logits,
    sensitivity_score_from_logits,
    specificity_score_from_logits,
)
from .models import build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _subject_map_by_id(subjects: list[SubjectSample]) -> dict[str, SubjectSample]:
    subject_map: dict[str, SubjectSample] = {}
    for subject in subjects:
        if subject.subject_id in subject_map:
            raise ValueError(f"Duplicate subject_id found: {subject.subject_id}")
        subject_map[subject.subject_id] = subject
    return subject_map


def _plan_from_subjects(
    subjects: list[SubjectSample],
    num_folds: int,
    val_ratio: float,
    seed: int,
) -> dict:
    folds = build_cross_validation_folds(subjects, num_folds=num_folds, seed=seed)
    remaining_fraction = 1.0 - (1.0 / num_folds)
    adjusted_val_ratio = val_ratio / remaining_fraction if remaining_fraction > 0 else val_ratio
    adjusted_val_ratio = min(max(adjusted_val_ratio, 0.0), 0.5)
    adjusted_train_ratio = 1.0 - adjusted_val_ratio

    fold_plans: list[dict] = []
    for fold_index, test_subjects in enumerate(folds):
        dev_subjects = [subject for idx, fold in enumerate(folds) if idx != fold_index for subject in fold]
        train_subjects, val_subjects, _ = split_subjects(
            dev_subjects,
            train_ratio=adjusted_train_ratio,
            val_ratio=adjusted_val_ratio,
            seed=seed + fold_index,
        )
        fold_plans.append(
            {
                "fold_index": fold_index,
                "train_subject_ids": [subject.subject_id for subject in train_subjects],
                "val_subject_ids": [subject.subject_id for subject in val_subjects],
                "test_subject_ids": [subject.subject_id for subject in test_subjects],
            }
        )

    return {
        "num_folds": num_folds,
        "val_ratio": val_ratio,
        "seed": seed,
        "folds": fold_plans,
    }


def _load_or_create_split_plan(
    subjects: list[SubjectSample],
    split_file: Path | None,
    num_folds: int,
    val_ratio: float,
    seed: int,
) -> dict:
    if split_file is None:
        return _plan_from_subjects(subjects, num_folds=num_folds, val_ratio=val_ratio, seed=seed)

    if split_file.exists():
        plan = json.loads(split_file.read_text())
    else:
        plan = _plan_from_subjects(subjects, num_folds=num_folds, val_ratio=val_ratio, seed=seed)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(plan, indent=2) + "\n")

    if int(plan.get("num_folds", -1)) != num_folds:
        raise SystemExit(f"Split file {split_file} has num_folds={plan.get('num_folds')} but config requests {num_folds}")

    available_subject_ids = set(_subject_map_by_id(subjects))
    for fold_plan in plan.get("folds", []):
        for key in ("train_subject_ids", "val_subject_ids", "test_subject_ids"):
            missing = [subject_id for subject_id in fold_plan.get(key, []) if subject_id not in available_subject_ids]
            if missing:
                raise SystemExit(
                    f"Split file {split_file} references subject_ids not present in current dataset for {key}: {missing[:5]}"
                )
    return plan


def _format_run_token(value: object) -> str:
    text = str(value).strip().lower()
    safe_chars: list[str] = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            safe_chars.append(char)
        elif char == ".":
            safe_chars.append("p")
    return "".join(safe_chars).strip("_-") or "na"


def _build_run_name(config: dict) -> str:
    experiment_name = _format_run_token(config["experiment_name"])
    epochs = int(config.get("epochs", 80))
    loss_config = config.get("loss", {}) or {}
    loss_name = _format_run_token(loss_config.get("name", "dice_bce"))

    loss_alias_map = {
        "dice_bce": "db",
        "focal_tversky": "ft",
        "focal_tversky_focal": "ftf",
        "ftl_focal": "ftf",
        "focal_tversky_combo": "ftf",
    }
    loss_alias = loss_alias_map.get(loss_name, loss_name)

    tokens = [experiment_name, loss_alias]
    if loss_alias == "db":
        bce_weight = float(loss_config.get("bce_weight", 0.5))
        if abs(bce_weight - 0.5) > 1e-8:
            tokens.append(f"bw{_format_run_token(bce_weight)}")
    elif loss_alias in {"ft", "ftf"}:
        alpha = float(loss_config.get("alpha", 0.7))
        beta = float(loss_config.get("beta", 0.3))
        gamma = float(loss_config.get("gamma", 1.33))
        if (alpha, beta, gamma) != (0.7, 0.3, 1.33):
            tokens.append(f"a{_format_run_token(alpha)}")
            tokens.append(f"b{_format_run_token(beta)}")
            tokens.append(f"g{_format_run_token(gamma)}")
        if loss_alias == "ftf":
            focal_alpha = float(loss_config.get("focal_alpha", 0.25))
            focal_gamma = float(loss_config.get("focal_gamma", 2.0))
            if (focal_alpha, focal_gamma) != (0.25, 2.0):
                tokens.append(f"fa{_format_run_token(focal_alpha)}")
                tokens.append(f"fg{_format_run_token(focal_gamma)}")
    selected_fold_index = config.get("fold_index")
    if selected_fold_index is not None:
        tokens.append(f"f{int(selected_fold_index) + 1:02d}")
    tokens.append(f"e{epochs}")
    return "__".join(tokens)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    subjects: list[SubjectSample],
    batch_size: int,
    patch_size: tuple[int, int, int],
    input_mode: str,
    positive_patch_prob: float,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = Patch3DSegmentationDataset(
        subjects=subjects,
        patch_size=patch_size,
        input_mode=input_mode,
        positive_patch_prob=positive_patch_prob,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        label = batch["label"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return {"loss": total_loss / max(1, steps)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_hd95 = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_sensitivity = 0.0
    total_specificity = 0.0
    steps = 0
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        label = batch["label"].to(device=device, dtype=torch.float32)
        logits = model(image)
        loss = criterion(logits, label)
        total_loss += float(loss.item())
        total_dice += float(dice_score_from_logits(logits, label).item())
        total_hd95 += float(hd95_score_from_logits(logits, label).item())
        total_iou += float(iou_score_from_logits(logits, label).item())
        total_precision += float(precision_score_from_logits(logits, label).item())
        total_recall += float(recall_score_from_logits(logits, label).item())
        total_sensitivity += float(sensitivity_score_from_logits(logits, label).item())
        total_specificity += float(specificity_score_from_logits(logits, label).item())
        steps += 1
    n = max(1, steps)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "hd95": total_hd95 / n,
        "iou": total_iou / n,
        "precision": total_precision / n,
        "recall": total_recall / n,
        "sensitivity": total_sensitivity / n,
        "specificity": total_specificity / n,
    }


def _aggregate_metric_dicts(rows: list[dict[str, float]]) -> dict[str, dict[str, float] | None]:
    if not rows:
        return {}

    aggregated: dict[str, dict[str, float] | None] = {}
    for key in rows[0]:
        values = [row[key] for row in rows if key in row]
        if not values:
            aggregated[key] = None
            continue
        aggregated[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return aggregated


def _plot_training_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history if "train_loss" in row]
    val_loss = [float(row["val_loss"]) for row in history if "val_loss" in row]
    val_dice = [float(row["val_dice"]) for row in history if "val_dice" in row]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs[: len(train_loss)], train_loss, label="train loss", color="#1f77b4", linewidth=2)
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label="val loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if val_dice:
        axes[1].plot(epochs[: len(val_dice)], val_dice, label="val dice", color="#2ca02c", linewidth=2)
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    if val_dice:
        axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _compute_sliding_window_starts(size: int, window: int) -> list[int]:
    if size <= window:
        return [0]

    stride = max(1, window // 2)
    starts = list(range(0, size - window + 1, stride))
    last_start = size - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@torch.no_grad()
def _predict_subject_mask(
    model: torch.nn.Module,
    subject: SubjectSample,
    input_mode: str,
    patch_size: tuple[int, int, int],
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    arrays = load_subject_arrays(subject)
    t1 = zscore_inside_mask(arrays["t1"], arrays["brain_mask"])
    flair = zscore_inside_mask(arrays["flair"], arrays["brain_mask"])
    t1_flair_ratio = arrays["t1_flair_ratio"]
    flair_t1_ratio = arrays["flair_t1_ratio"]

    original_shape = t1.shape
    target_shape = tuple(max(size, patch) for size, patch in zip(original_shape, patch_size))
    t1 = pad_if_needed(t1, target_shape)
    flair = pad_if_needed(flair, target_shape)
    if t1_flair_ratio is not None:
        t1_flair_ratio = pad_if_needed(t1_flair_ratio.astype(np.float32), target_shape)
    if flair_t1_ratio is not None:
        flair_t1_ratio = pad_if_needed(flair_t1_ratio.astype(np.float32), target_shape)

    image = build_input_channels(
        t1,
        flair,
        input_mode=input_mode,
        t1_flair_ratio=t1_flair_ratio,
        flair_t1_ratio=flair_t1_ratio,
    )
    image_tensor = torch.from_numpy(image[None, ...]).to(device=device, dtype=torch.float32)

    logits_sum = torch.zeros((1, 1, *target_shape), device=device, dtype=torch.float32)
    logits_count = torch.zeros_like(logits_sum)
    z_starts = _compute_sliding_window_starts(target_shape[0], patch_size[0])
    y_starts = _compute_sliding_window_starts(target_shape[1], patch_size[1])
    x_starts = _compute_sliding_window_starts(target_shape[2], patch_size[2])

    model.eval()
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = image_tensor[
                    :,
                    :,
                    z:z + patch_size[0],
                    y:y + patch_size[1],
                    x:x + patch_size[2],
                ]
                patch_logits = model(patch)
                logits_sum[
                    :,
                    :,
                    z:z + patch_size[0],
                    y:y + patch_size[1],
                    x:x + patch_size[2],
                ] += patch_logits
                logits_count[
                    :,
                    :,
                    z:z + patch_size[0],
                    y:y + patch_size[1],
                    x:x + patch_size[2],
                ] += 1.0

    mean_logits = logits_sum / torch.clamp(logits_count, min=1.0)
    probabilities = torch.sigmoid(mean_logits)[0, 0].detach().cpu().numpy()
    prediction = (probabilities >= threshold).astype(np.uint8)
    return prediction[: original_shape[0], : original_shape[1], : original_shape[2]]


def _export_fold_masks(
    model: torch.nn.Module,
    subjects: list[SubjectSample],
    fold_dir: Path,
    input_mode: str,
    patch_size: tuple[int, int, int],
    device: torch.device,
) -> list[dict[str, str]]:
    validation_dir = fold_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, str]] = []
    for subject in subjects:
        prediction = _predict_subject_mask(
            model=model,
            subject=subject,
            input_mode=input_mode,
            patch_size=patch_size,
            device=device,
        )
        reference_path = subject.label_path if subject.label_path and subject.label_path.exists() else subject.t1_path
        if reference_path is None:
            raise FileNotFoundError(f"No reference image found for subject '{subject.subject_id}'")
        reference_image = nib.load(str(reference_path))
        prediction_image = nib.Nifti1Image(prediction.astype(np.uint8), affine=reference_image.affine, header=reference_image.header.copy())
        prediction_image.set_data_dtype(np.uint8)
        prediction_path = validation_dir / f"{subject.subject_id}_pred.nii.gz"
        nib.save(prediction_image, str(prediction_path))
        exported.append(
            {
                "subject_id": subject.subject_id,
                "prediction_path": str(prediction_path),
            }
        )

    (validation_dir / "manifest.json").write_text(json.dumps(exported, indent=2) + "\n")
    return exported


def _run_fold(
    fold_index: int,
    total_folds: int,
    train_subjects: list[SubjectSample],
    val_subjects: list[SubjectSample],
    test_subjects: list[SubjectSample],
    config: dict,
    output_dir: Path,
) -> dict:
    patch_size = tuple(int(v) for v in config.get("patch_size", [96, 96, 96]))
    input_mode = config.get("input_mode", "t1_flair")
    batch_size = int(config.get("batch_size", 2))
    num_workers = int(config.get("num_workers", 0))
    positive_patch_prob = float(config.get("positive_patch_prob", 0.7))

    fold_dir = output_dir / f"fold_{fold_index + 1:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        train_subjects,
        batch_size=batch_size,
        patch_size=patch_size,
        input_mode=input_mode,
        positive_patch_prob=positive_patch_prob,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        val_subjects,
        batch_size=batch_size,
        patch_size=patch_size,
        input_mode=input_mode,
        positive_patch_prob=positive_patch_prob,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        test_subjects,
        batch_size=batch_size,
        patch_size=patch_size,
        input_mode=input_mode,
        positive_patch_prob=positive_patch_prob,
        num_workers=num_workers,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(**config["model"]).to(device)
    criterion = build_loss(config.get("loss"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    history: list[dict[str, float]] = []
    best_val_dice = -1.0
    best_checkpoint = fold_dir / "best_model.pt"
    epochs = int(config.get("epochs", 80))
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device) if val_subjects else {}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        current_score = val_metrics.get("dice", train_metrics["loss"] * -1.0)
        if not best_checkpoint.exists() or current_score > best_val_dice:
            if "dice" in val_metrics:
                best_val_dice = current_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_dice": best_val_dice if best_val_dice >= 0 else None,
                    "fold_index": fold_index + 1,
                    "num_folds": total_folds,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device) if test_subjects else {}
    exported_fold_masks = _export_fold_masks(
        model=model,
        subjects=test_subjects,
        fold_dir=fold_dir,
        input_mode=input_mode,
        patch_size=patch_size,
        device=device,
    )
    fold_summary = {
        "fold_index": fold_index + 1,
        "num_folds": total_folds,
        "input_mode": input_mode,
        "num_train_subjects": len(train_subjects),
        "num_val_subjects": len(val_subjects),
        "num_test_subjects": len(test_subjects),
        "best_val_dice": best_val_dice if best_val_dice >= 0 else None,
        "validation_predictions_dir": str(fold_dir / "validation"),
        "num_validation_predictions": len(exported_fold_masks),
        "training_plot_path": str(fold_dir / "training_curve.png"),
        "test_metrics": test_metrics,
    }

    (fold_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    _plot_training_history(history, fold_dir / "training_curve.png")
    (fold_dir / "summary.json").write_text(json.dumps(fold_summary, indent=2) + "\n")
    (fold_dir / "split.json").write_text(
        json.dumps(
            {
                "fold_index": fold_index + 1,
                "num_folds": total_folds,
                "train_subjects": [subject.subject_id for subject in train_subjects],
                "val_subjects": [subject.subject_id for subject in val_subjects],
                "test_subjects": [subject.subject_id for subject in test_subjects],
            },
            indent=2,
        )
        + "\n"
    )
    return fold_summary


def fit_experiment(config: dict, repo_root: Path) -> Path:
    seed_everything(int(config.get("seed", 42)))

    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    dataset_format = config.get("dataset_format", "subject_dirs")
    experiment_name = config["experiment_name"]
    run_name = _build_run_name(config)
    output_root = Path(config.get("output_root", "data/outputs"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    split_file = config.get("split_file")
    split_file_path = Path(split_file) if split_file is not None else None
    if split_file_path is not None and not split_file_path.is_absolute():
        split_file_path = repo_root / split_file_path
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = build_subject_index(
        data_root,
        include_controls=bool(config.get("include_controls", False)),
        dataset_format=dataset_format,
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
    if not subjects:
        raise SystemExit(f"No subjects found under {data_root}")
    num_folds = int(config.get("num_folds", 5))
    val_ratio = float(config.get("val_ratio", 0.15))
    selected_fold_index = config.get("fold_index")
    seed = int(config.get("seed", 42))
    split_plan = _load_or_create_split_plan(
        subjects,
        split_file=split_file_path,
        num_folds=num_folds,
        val_ratio=val_ratio,
        seed=seed,
    )
    if selected_fold_index is not None:
        selected_fold_index = int(selected_fold_index)
        if selected_fold_index < 0 or selected_fold_index >= num_folds:
            raise SystemExit(f"fold_index must be between 0 and {num_folds - 1}, got {selected_fold_index}")

    subject_map = _subject_map_by_id(subjects)
    fold_summaries: list[dict] = []
    for fold_plan in split_plan["folds"]:
        fold_index = int(fold_plan["fold_index"])
        if selected_fold_index is not None and fold_index != selected_fold_index:
            continue
        train_subjects = [subject_map[subject_id] for subject_id in fold_plan["train_subject_ids"]]
        val_subjects = [subject_map[subject_id] for subject_id in fold_plan["val_subject_ids"]]
        test_subjects = [subject_map[subject_id] for subject_id in fold_plan["test_subject_ids"]]
        fold_summaries.append(
            _run_fold(
                fold_index=fold_index,
                total_folds=num_folds,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
                config=config,
                output_dir=output_dir,
            )
        )

    test_metric_rows = [fold_summary["test_metrics"] for fold_summary in fold_summaries if fold_summary.get("test_metrics")]
    best_val_dice_rows = [{"best_val_dice": value} for value in [fold["best_val_dice"] for fold in fold_summaries] if value is not None]
    summary = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "model_name": config["model"]["name"],
        "cross_validation": True,
        "num_folds": num_folds,
        "selected_fold_index": selected_fold_index,
        "dataset_format": dataset_format,
        "split_file": str(split_file_path) if split_file_path is not None else None,
        "input_mode": config.get("input_mode", "t1_flair"),
        "num_subjects": len(subjects),
        "folds": fold_summaries,
        "aggregate_best_val_dice": _aggregate_metric_dicts(best_val_dice_rows).get("best_val_dice"),
        "aggregate_test_metrics": _aggregate_metric_dicts(test_metric_rows),
    }

    (output_dir / "config.snapshot.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output_dir
