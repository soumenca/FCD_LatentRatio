from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SubjectSample:
    subject_dir: Path
    dataset: str
    subject_id: str
    cohort_root: str
    cohort_role: str
    has_label: bool
    t1_path: Path | None = None
    flair_path: Path | None = None
    t1_flair_ratio_path: Path | None = None
    flair_t1_ratio_path: Path | None = None
    label_path: Path | None = None
    brain_mask_path: Path | None = None


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _infer_subject_sample(subject_dir: Path, cohort_root: str, dataset: str) -> SubjectSample:
    label_path = subject_dir / "label.nii.gz"
    return SubjectSample(
        subject_dir=subject_dir,
        dataset=dataset,
        subject_id=subject_dir.name,
        cohort_root=cohort_root,
        cohort_role="patient" if cohort_root == "subjects" else "control",
        has_label=label_path.exists(),
        t1_path=subject_dir / "T1w.nii.gz",
        flair_path=subject_dir / "FLAIR.nii.gz",
        t1_flair_ratio_path=subject_dir / "T1w_div_FLAIR.nii.gz",
        flair_t1_ratio_path=subject_dir / "FLAIR_div_T1w.nii.gz",
        label_path=label_path,
        brain_mask_path=subject_dir / "brain_mask.nii.gz",
    )


def _infer_nnunetv2_cohort(subject_id: str) -> tuple[str, str]:
    if subject_id.startswith("FCD_"):
        return "nnunetv2", "patient"
    if subject_id.startswith("CON_"):
        return "nnunetv2", "control"
    return "nnunetv2", "case"


def _channel_suffix(channel_index: int) -> str:
    return f"_{channel_index:04d}.nii.gz"


def _strip_channel_suffix(name: str) -> str:
    if len(name) >= 12 and name[-12] == "_" and name[-11:-7].isdigit() and name.endswith(".nii.gz"):
        return name[:-12]
    return name[:-7] if name.endswith(".nii.gz") else name


def _resolve_channel_index(dataset_json: dict, desired_names: tuple[str, ...], default_index: int) -> int:
    channel_names = dataset_json.get("channel_names", {})
    for raw_index, raw_name in channel_names.items():
        normalized = str(raw_name).strip().lower()
        if normalized in desired_names:
            return int(raw_index)
    return default_index


def _resolve_optional_channel_index(dataset_json: dict, desired_names: tuple[str, ...]) -> int | None:
    channel_names = dataset_json.get("channel_names", {})
    for raw_index, raw_name in channel_names.items():
        normalized = str(raw_name).strip().lower().replace(" ", "").replace("-", "").replace("/", "").replace("|", "")
        if normalized in desired_names:
            return int(raw_index)
    return None


def _build_subject_index_from_subject_dirs(root: Path, include_controls: bool = False) -> list[SubjectSample]:
    samples: list[SubjectSample] = []
    cohort_roots = ["subjects"]
    if include_controls:
        cohort_roots.append("controls")

    for cohort_root in cohort_roots:
        cohort_dir = root / cohort_root
        if not cohort_dir.exists():
            continue
        for dataset_dir in sorted(path for path in cohort_dir.iterdir() if path.is_dir()):
            for subject_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
                manifest = subject_dir / "subject_manifest.json"
                if manifest.exists():
                    meta = json.loads(manifest.read_text())
                    samples.append(
                        SubjectSample(
                            subject_dir=subject_dir,
                            dataset=meta.get("dataset", dataset_dir.name),
                            subject_id=meta.get("subject_id", subject_dir.name),
                            cohort_root=meta.get("cohort_root", cohort_root),
                            cohort_role=meta.get("cohort_role", "patient" if cohort_root == "subjects" else "control"),
                            has_label=bool(meta.get("files", {}).get("label")) or (subject_dir / "label.nii.gz").exists(),
                            t1_path=subject_dir / meta.get("files", {}).get("t1", "T1w.nii.gz"),
                            flair_path=subject_dir / meta.get("files", {}).get("flair", "FLAIR.nii.gz"),
                            t1_flair_ratio_path=subject_dir / meta.get("files", {}).get("t1_flair_ratio", "T1w_div_FLAIR.nii.gz"),
                            flair_t1_ratio_path=subject_dir / meta.get("files", {}).get("flair_t1_ratio", "FLAIR_div_T1w.nii.gz"),
                            label_path=subject_dir / meta.get("files", {}).get("label", "label.nii.gz"),
                            brain_mask_path=subject_dir / meta.get("files", {}).get("brain_mask", "brain_mask.nii.gz"),
                        )
                    )
                else:
                    samples.append(_infer_subject_sample(subject_dir, cohort_root=cohort_root, dataset=dataset_dir.name))
    return samples


def _build_subject_index_from_nnunetv2(
    root: Path,
    include_controls: bool = False,
    t1_channel_index: int = 0,
    flair_channel_index: int = 1,
    t1_flair_ratio_channel_index: int | None = None,
    flair_t1_ratio_channel_index: int | None = None,
) -> list[SubjectSample]:
    dataset_json_path = root / "dataset.json"
    if dataset_json_path.exists():
        dataset_json = json.loads(dataset_json_path.read_text())
        t1_channel_index = _resolve_channel_index(dataset_json, ("t1w", "t1", "t1-weighted"), t1_channel_index)
        flair_channel_index = _resolve_channel_index(dataset_json, ("flair", "t2-flair", "t2 flair"), flair_channel_index)
        if t1_flair_ratio_channel_index is None:
            t1_flair_ratio_channel_index = _resolve_optional_channel_index(
                dataset_json,
                ("t1wflair", "t1divflair", "t1wdivflair", "t1flairratio"),
            )
        if flair_t1_ratio_channel_index is None:
            flair_t1_ratio_channel_index = _resolve_optional_channel_index(
                dataset_json,
                ("flairt1w", "flairdivt1", "flairdivt1w", "flairt1ratio"),
            )

    images_tr_dir = root / "imagesTr"
    labels_tr_dir = root / "labelsTr"
    if not images_tr_dir.exists():
        return []

    t1_suffix = _channel_suffix(t1_channel_index)
    flair_suffix = _channel_suffix(flair_channel_index)
    t1_flair_ratio_suffix = _channel_suffix(t1_flair_ratio_channel_index) if t1_flair_ratio_channel_index is not None else None
    flair_t1_ratio_suffix = _channel_suffix(flair_t1_ratio_channel_index) if flair_t1_ratio_channel_index is not None else None
    samples: list[SubjectSample] = []
    for t1_path in sorted(images_tr_dir.glob(f"*{t1_suffix}")):
        subject_id = _strip_channel_suffix(t1_path.name)
        cohort_root, cohort_role = _infer_nnunetv2_cohort(subject_id)
        if cohort_role == "control" and not include_controls:
            continue
        flair_path = images_tr_dir / f"{subject_id}{flair_suffix}"
        if not flair_path.exists():
            raise FileNotFoundError(
                f"Missing FLAIR channel for subject '{subject_id}'. Expected {flair_path.name} under {images_tr_dir}"
            )
        label_path = labels_tr_dir / f"{subject_id}.nii.gz"
        t1_flair_ratio_path = (
            images_tr_dir / f"{subject_id}{t1_flair_ratio_suffix}"
            if t1_flair_ratio_suffix is not None and (images_tr_dir / f"{subject_id}{t1_flair_ratio_suffix}").exists()
            else None
        )
        flair_t1_ratio_path = (
            images_tr_dir / f"{subject_id}{flair_t1_ratio_suffix}"
            if flair_t1_ratio_suffix is not None and (images_tr_dir / f"{subject_id}{flair_t1_ratio_suffix}").exists()
            else None
        )
        samples.append(
            SubjectSample(
                subject_dir=root / subject_id,
                dataset=root.name,
                subject_id=subject_id,
                cohort_root=cohort_root,
                cohort_role=cohort_role,
                has_label=label_path.exists(),
                t1_path=t1_path,
                flair_path=flair_path,
                t1_flair_ratio_path=t1_flair_ratio_path,
                flair_t1_ratio_path=flair_t1_ratio_path,
                label_path=label_path,
                brain_mask_path=None,
            )
        )
    return samples


def build_subject_index(
    root: Path,
    include_controls: bool = False,
    dataset_format: str = "subject_dirs",
    t1_channel_index: int = 0,
    flair_channel_index: int = 1,
    t1_flair_ratio_channel_index: int | None = None,
    flair_t1_ratio_channel_index: int | None = None,
) -> list[SubjectSample]:
    if dataset_format == "subject_dirs":
        return _build_subject_index_from_subject_dirs(root, include_controls=include_controls)
    if dataset_format == "nnunetv2":
        return _build_subject_index_from_nnunetv2(
            root,
            include_controls=include_controls,
            t1_channel_index=t1_channel_index,
            flair_channel_index=flair_channel_index,
            t1_flair_ratio_channel_index=t1_flair_ratio_channel_index,
            flair_t1_ratio_channel_index=flair_t1_ratio_channel_index,
        )
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def split_subjects(
    subjects: list[SubjectSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[SubjectSample], list[SubjectSample], list[SubjectSample]]:
    by_group: dict[tuple[str, str], list[SubjectSample]] = {}
    for subject in subjects:
        key = (subject.dataset, subject.cohort_role)
        by_group.setdefault(key, []).append(subject)

    rng = random.Random(seed)
    train: list[SubjectSample] = []
    val: list[SubjectSample] = []
    test: list[SubjectSample] = []
    for group_subjects in by_group.values():
        items = list(group_subjects)
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            train.extend(items)
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio)) if n >= 3 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return train, val, test


def build_cross_validation_folds(
    subjects: list[SubjectSample],
    num_folds: int = 5,
    seed: int = 42,
) -> list[list[SubjectSample]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    by_group: dict[tuple[str, str], list[SubjectSample]] = {}
    for subject in subjects:
        key = (subject.dataset, subject.cohort_role)
        by_group.setdefault(key, []).append(subject)

    rng = random.Random(seed)
    folds: list[list[SubjectSample]] = [[] for _ in range(num_folds)]
    for group_subjects in by_group.values():
        items = list(group_subjects)
        rng.shuffle(items)
        for index, subject in enumerate(items):
            folds[index % num_folds].append(subject)

    return folds


def load_subject_arrays(subject: SubjectSample) -> dict[str, np.ndarray]:
    t1_path = subject.t1_path or (subject.subject_dir / "T1w.nii.gz")
    flair_path = subject.flair_path or (subject.subject_dir / "FLAIR.nii.gz")
    t1_flair_ratio_path = subject.t1_flair_ratio_path or (subject.subject_dir / "T1w_div_FLAIR.nii.gz")
    flair_t1_ratio_path = subject.flair_t1_ratio_path or (subject.subject_dir / "FLAIR_div_T1w.nii.gz")
    label_path = subject.label_path or (subject.subject_dir / "label.nii.gz")
    brain_mask_path = subject.brain_mask_path or (subject.subject_dir / "brain_mask.nii.gz")
    t1 = _load_nifti(t1_path)
    flair = _load_nifti(flair_path)
    t1_flair_ratio = _load_nifti(t1_flair_ratio_path) if t1_flair_ratio_path.exists() else None
    flair_t1_ratio = _load_nifti(flair_t1_ratio_path) if flair_t1_ratio_path.exists() else None
    label = _load_nifti(label_path) if label_path.exists() else np.zeros_like(t1, dtype=np.float32)
    brain_mask = _load_nifti(brain_mask_path) if brain_mask_path.exists() else np.ones_like(t1, dtype=np.float32)
    return {
        "t1": t1,
        "flair": flair,
        "t1_flair_ratio": t1_flair_ratio,
        "flair_t1_ratio": flair_t1_ratio,
        "label": label,
        "brain_mask": brain_mask,
    }


def zscore_inside_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = image[mask > 0]
    if masked.size == 0:
        return image.astype(np.float32)
    mean = float(masked.mean())
    std = float(masked.std())
    std = std if std > 1e-6 else 1.0
    out = (image - mean) / std
    out[mask <= 0] = 0.0
    return out.astype(np.float32)


def pad_if_needed(array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    pad_width = []
    for current_dim, target_dim in zip(array.shape, target_shape):
        total = max(0, target_dim - current_dim)
        before = total // 2
        after = total - before
        pad_width.append((before, after))
    if any(before or after for before, after in pad_width):
        array = np.pad(array, pad_width, mode="constant")
    return array


def crop_patch(array: np.ndarray, start: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    z, y, x = start
    dz, dy, dx = patch_size
    return array[z:z + dz, y:y + dy, x:x + dx]


def choose_patch_start(
    shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    center: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    starts: list[int] = []
    for dim, patch, c in zip(shape, patch_size, center or (None, None, None)):
        max_start = max(0, dim - patch)
        if c is None:
            starts.append(random.randint(0, max_start) if max_start > 0 else 0)
        else:
            start = int(round(c - patch / 2))
            starts.append(min(max(start, 0), max_start))
    return tuple(starts)


def build_input_channels(
    t1: np.ndarray,
    flair: np.ndarray,
    input_mode: str,
    t1_flair_ratio: np.ndarray | None = None,
    flair_t1_ratio: np.ndarray | None = None,
) -> np.ndarray:
    if input_mode == "t1_flair":
        return np.stack([t1, flair], axis=0)
    if input_mode == "t1_flair_ratio":
        ratio_t1_flair = (
            t1_flair_ratio.astype(np.float32)
            if t1_flair_ratio is not None
            else (t1 / (np.abs(flair) + 1e-6)).astype(np.float32)
        )
        ratio_flair_t1 = (
            flair_t1_ratio.astype(np.float32)
            if flair_t1_ratio is not None
            else (flair / (np.abs(t1) + 1e-6)).astype(np.float32)
        )
        return np.stack([t1, flair, ratio_t1_flair, ratio_flair_t1], axis=0)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


class Patch3DSegmentationDataset(Dataset):
    def __init__(
        self,
        subjects: list[SubjectSample],
        patch_size: tuple[int, int, int],
        input_mode: str = "t1_flair",
        positive_patch_prob: float = 0.7,
    ) -> None:
        self.subjects = subjects
        self.patch_size = patch_size
        self.input_mode = input_mode
        self.positive_patch_prob = positive_patch_prob

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        subject = self.subjects[index]
        arrays = load_subject_arrays(subject)
        t1 = zscore_inside_mask(arrays["t1"], arrays["brain_mask"])
        flair = zscore_inside_mask(arrays["flair"], arrays["brain_mask"])
        t1_flair_ratio = arrays["t1_flair_ratio"]
        flair_t1_ratio = arrays["flair_t1_ratio"]
        label = (arrays["label"] > 0).astype(np.float32)

        target_shape = tuple(max(size, patch) for size, patch in zip(t1.shape, self.patch_size))
        t1 = pad_if_needed(t1, target_shape)
        flair = pad_if_needed(flair, target_shape)
        if t1_flair_ratio is not None:
            t1_flair_ratio = pad_if_needed(t1_flair_ratio.astype(np.float32), target_shape)
        if flair_t1_ratio is not None:
            flair_t1_ratio = pad_if_needed(flair_t1_ratio.astype(np.float32), target_shape)
        label = pad_if_needed(label, target_shape)

        center = None
        lesion_voxels = np.argwhere(label > 0.5)
        if lesion_voxels.size > 0 and random.random() < self.positive_patch_prob:
            center = tuple(int(value) for value in lesion_voxels[random.randrange(len(lesion_voxels))])
        start = choose_patch_start(t1.shape, self.patch_size, center=center)

        t1_patch = crop_patch(t1, start, self.patch_size)
        flair_patch = crop_patch(flair, start, self.patch_size)
        t1_flair_ratio_patch = (
            crop_patch(t1_flair_ratio, start, self.patch_size) if t1_flair_ratio is not None else None
        )
        flair_t1_ratio_patch = (
            crop_patch(flair_t1_ratio, start, self.patch_size) if flair_t1_ratio is not None else None
        )
        label_patch = crop_patch(label, start, self.patch_size)
        image = build_input_channels(
            t1_patch,
            flair_patch,
            self.input_mode,
            t1_flair_ratio=t1_flair_ratio_patch,
            flair_t1_ratio=flair_t1_ratio_patch,
        )

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label_patch[None, ...]),
        }
