# FCD Latent Ratio Experiment

This repository contains a minimal experiment scaffold for the 2-week ablation with 5-fold cross-validation:

- `U-Net-E5`: plain 3D U-Net with `T1w + FLAIR`
- `CRIL-U-Net`: 3D U-Net with a Compact Ratio-Interaction Learning module
- `U-Net + ratios`: plain 3D U-Net with manual ratio-style input channels

## Core idea

The proposed model does not emit handcrafted ratio images. Instead, it learns a compact latent MRI representation from conventional `T1w` and `FLAIR` inputs before segmentation:

```text
T1w + FLAIR
    -> Compact Ratio-Interaction Learning Module
    -> compressed latent representation
    -> 3D U-Net
    -> FCD mask
```

## Expected dataset layout

The training code now supports two dataset formats.

### nnU-Net v2 format

This is the recommended path if your data is already prepared for nnU-Net:

```text
data/
  dataset.json
  imagesTr/
    sub-001_0000.nii.gz   # T1w
    sub-001_0001.nii.gz   # FLAIR
    sub-002_0000.nii.gz
    sub-002_0001.nii.gz
  labelsTr/
    sub-001.nii.gz
    sub-002.nii.gz
```

Default channel mapping is `_0000 -> T1w` and `_0001 -> FLAIR`. If `dataset.json` contains `channel_names`, the loader will use that mapping automatically when possible.

### Subject-folder format

The original subject-space layout is still supported:

```text
data/
  subjects/
    FCDII/
      sub-001/
        T1w.nii.gz
        FLAIR.nii.gz
        label.nii.gz
        brain_mask.nii.gz          # optional
        subject_manifest.json      # optional
  controls/
    FCDII/
      sub-101/
        T1w.nii.gz
        FLAIR.nii.gz
        brain_mask.nii.gz          # optional
        subject_manifest.json      # optional
```

If `subject_manifest.json` is missing, the loader infers metadata from the folder names.

## Experiment presets

- [code/configs/exp_a_unet_e5.json](/Users/soumen/wkdir/CodexApp/FCD_LatentRatio/code/configs/exp_a_unet_e5.json): baseline 2-channel 3D U-Net
- [code/configs/exp_b_cril_unet.json](/Users/soumen/wkdir/CodexApp/FCD_LatentRatio/code/configs/exp_b_cril_unet.json): proposed CRIL-U-Net
- [code/configs/exp_c_unet_with_ratios.json](/Users/soumen/wkdir/CodexApp/FCD_LatentRatio/code/configs/exp_c_unet_with_ratios.json): manual ratio-style benchmark

## Quick start

Bootstrap the environment with:

```bash
bash code/scripts/setup_env.sh
```

Then run:

```bash
python code/scripts/train_experiment.py --config code/configs/exp_a_unet_e5.json
python code/scripts/train_experiment.py --config code/configs/exp_b_cril_unet.json
python code/scripts/train_experiment.py --config code/configs/exp_c_unet_with_ratios.json
```

You can override dataset and output locations at launch time:

```bash
python code/scripts/train_experiment.py \
  --config code/configs/exp_a_unet_e5.json \
  --data-root /path/to/nnunet_dataset \
  --output-root /path/to/outputs
```

You can also override epochs from the command line:

```bash
python code/scripts/train_experiment.py \
  --config code/configs/exp_a_unet_e5.json \
  --epochs 120
```

All outputs are written under `data/outputs/<experiment_name>/`.

For nnU-Net v2 datasets, set `data_root` to the dataset root containing `imagesTr/`, `labelsTr/`, and optionally `dataset.json`. The included configs already default to:

```json
{
  "dataset_format": "nnunetv2",
  "num_folds": 5,
  "epochs": 300,
  "t1_channel_index": 0,
  "flair_channel_index": 1
}
```

If you want to use the older subject-folder layout instead, change `dataset_format` to `"subject_dirs"`.

Each run now trains one model per fold under `data/outputs/<experiment_name>/fold_01/` through `fold_05/`. Every fold writes its own `best_model.pt`, `history.json`, `split.json`, and `summary.json`, while the top-level `data/outputs/<experiment_name>/summary.json` stores the aggregated cross-validation metrics.

Project layout now keeps executable code under `code/` and reserves `data/` for datasets.

## HPC Usage

For SLURM-based clusters, use [code/scripts/slurm_train.sh](/Users/soumen/wkdir/CodexApp/FCD_LatentRatio/code/scripts/slurm_train.sh) as the batch entrypoint.

Typical flow:

```bash
bash code/scripts/setup_env.sh
sbatch code/scripts/slurm_train.sh
```

Useful overrides:

```bash
MODULES="python/3.11 cuda/12.1" \
VENV_DIR=/path/to/venv \
DATA_ROOT_OVERRIDE=/path/to/nnunet_dataset \
OUTPUT_ROOT_OVERRIDE=/path/to/scratch/outputs \
CONFIG_PATH=/path/to/config.json \
sbatch code/scripts/slurm_train.sh
```

Notes for full-dataset HPC runs:

- Keep `patch_size` large enough for the 5-level U-Net. Very small patches such as `16x16x16` can collapse the bottleneck to `1x1x1` and fail during training.
- Prefer writing outputs to node-local or scratch storage when available, then copy final artifacts back to shared storage.
- Tune `batch_size`, `patch_size`, `num_workers`, memory, and walltime in the SLURM script for your cluster.

## Notes

- The CRIL module uses three branches:
  - local intensity features via `3x3x3` convolution
  - cross-modal interaction via `1x1x1` convolution
  - nonlinear ratio-like interaction from internal `T1w * FLAIR` and `|T1w - FLAIR|`
- The bottleneck compresses features to `4` latent channels by default.
- The U-Net backbone is shared across all three experiments for a clean ablation.
