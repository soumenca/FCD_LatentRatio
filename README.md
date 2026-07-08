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

For manual-ratio experiments, a 4-channel nnU-Net dataset is also supported:

```text
data/
  dataset.json
  imagesTr/
    sub-001_0000.nii.gz   # T1w
    sub-001_0001.nii.gz   # FLAIR
    sub-001_0002.nii.gz   # T1w / FLAIR
    sub-001_0003.nii.gz   # FLAIR / T1w
  labelsTr/
    sub-001.nii.gz
```

`exp_c` can now work with either:
- a 2-channel dataset containing only `T1w` and `FLAIR`, in which case the code computes `T1w / FLAIR` and `FLAIR / T1w` internally
- a 4-channel dataset containing precomputed ratio channels, in which case the loader reads `_0002` and `_0003` directly

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
- [code/configs/exp_c_unet_with_ratios.json](/Users/soumen/wkdir/CodexApp/FCD_LatentRatio/code/configs/exp_c_unet_with_ratios.json): manual bidirectional-ratio benchmark

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

Each run now trains one model per fold under `data/outputs/<experiment_name>/fold_01/` through `fold_05/`. Every fold writes its own `best_model.pt`, `history.json`, `split.json`, and `summary.json`, plus predicted validation masks under `fold_xx/validation/` as `.nii.gz` files, while the top-level `data/outputs/<experiment_name>/summary.json` stores the aggregated cross-validation metrics.

All bundled experiments now use the same shared 5-fold split file by default:

```bash
data/splits/shared_5fold_split.json
```

The first experiment run will create that file if it does not exist. Later runs of `exp_a`, `exp_b`, and `exp_c` will reuse it so all experiments evaluate on the exact same folds.

Reported segmentation metrics now include Dice, HD95, IoU, precision, recall, sensitivity, and specificity.

Project layout now keeps executable code under `code/` and reserves `data/` for datasets.

## HPC Usage

Use [code/scripts/slurm_train.sh](/Users/soumen/wkdir/CodexApp/FCD_LatentRatio/code/scripts/slurm_train.sh) on M3.

### 1. Clone on M3

Clone the repo:

```bash
cd /path/where/you/want/the/repo
git clone https://github.com/soumenca/FCD_LatentRatio.git
cd FCD_LatentRatio
```

If the repo is already there and you just want the latest updates:

```bash
cd /path/to/FCD_LatentRatio
git pull
```

If M3 requires SSH instead of HTTPS:

```bash
git clone git@github.com:soumenca/FCD_LatentRatio.git
```

### 2. Setup once

```bash
cd /path/to/FCD_LatentRatio
module load python/3.11
module load cuda/12.1
bash code/scripts/setup_env.sh
```

### 3. Submit a job

`exp_a` with `T1w + FLAIR`:

```bash
cd /path/to/FCD_LatentRatio
sbatch code/scripts/slurm_train.sh \
  --modules "python/3.11 cuda/12.1" \
  --exp a \
  --data-root /path/to/Dataset1 \
  --output-root /path/to/scratch/FCD_LatentRatio_outputs
```

`exp_b` with `T1w + FLAIR`:

```bash
cd /path/to/FCD_LatentRatio
sbatch code/scripts/slurm_train.sh \
  --modules "python/3.11 cuda/12.1" \
  --exp b \
  --data-root /path/to/Dataset1 \
  --output-root /path/to/scratch/FCD_LatentRatio_outputs
```

`exp_c`:
- with `Dataset1`, ratios are computed internally
- with `Dataset2`, `_0002` and `_0003` are read directly

```bash
cd /path/to/FCD_LatentRatio
sbatch code/scripts/slurm_train.sh \
  --modules "python/3.11 cuda/12.1" \
  --exp c \
  --data-root /path/to/Dataset2 \
  --output-root /path/to/scratch/FCD_LatentRatio_outputs
```

You can also pass other parameters from the `sbatch` command line:

```bash
sbatch code/scripts/slurm_train.sh \
  --exp a \
  --data-root /path/to/Dataset1 \
  --output-root /path/to/scratch/FCD_LatentRatio_outputs \
  --epochs 120
```

### 4. Check logs

```bash
tail -f logs/UNet_CV_fold0_<jobid>.out
```

### 5. Find outputs

```bash
data/outputs/<experiment_name>/
```

If you use `OUTPUT_ROOT_OVERRIDE`, outputs go there instead.

### Notes

- The SLURM script uses your M3 settings: `gpu`, `1 GPU`, `8 CPUs`, `96G`, `2 days`, array `0-4`
- Each array task runs one CV fold
- Supported SLURM script flags: `--exp`, `--config`, `--data-root`, `--output-root`, `--venv-dir`, `--modules`, `--epochs`, `--fold-index`
- Keep `patch_size` reasonably large; very small patches like `16x16x16` can fail

## Notes

- The CRIL module uses three branches:
  - local intensity features via `3x3x3` convolution
  - cross-modal interaction via `1x1x1` convolution
  - bidirectional ratio interaction from internal `T1w / FLAIR` and `FLAIR / T1w`
- The bottleneck compresses features to `4` latent channels by default.
- The U-Net backbone is shared across all three experiments for a clean ablation.
