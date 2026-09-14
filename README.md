# CRIL-U-Net: MICAD 2026 Experiments

This repository contains the training and evaluation code for the three architectures reported in the MICAD 2026 paper:

> CRIL-U-Net: Compact Ratio-Interaction Learning for Focal Cortical Dysplasia Segmentation from T1w and FLAIR MRI

The comparison uses the same five-level 3D U-Net backbone for all models:

- `exp_a`: 3D U-Net with concatenated T1w and FLAIR input
- `exp_b`: CRIL-U-Net with a learned four-channel CRIL representation
- `exp_d`: Self-Attention U-Net with input-level pooled attention

Later residual U-Net, SegResNet, CRIL-plus-attention, fixed-ratio-input, and legacy analysis experiments are intentionally excluded.

## Architectures

All models use encoder widths `[32, 64, 128, 256, 512]`, two convolutions per stage, instance normalisation, LeakyReLU, max-pooling, transposed-convolution upsampling, and skip concatenation.

| Experiment | Model | Input to U-Net | Trainable parameters |
|---|---|---:|---:|
| `exp_a` | 3D U-Net | 2 channels | 22,573,249 |
| `exp_d` | Self-Attention U-Net | 2 attended channels | 22,575,537 |
| `exp_b` | CRIL-U-Net | 4 latent channels | 22,577,569 |

### CRIL-U-Net

CRIL receives independently normalised T1w and FLAIR volumes and learns three feature streams:

- local spatial features using a `3x3x3` convolution
- voxel-wise cross-modal mixing using a `1x1x1` convolution
- bidirectional ratio interactions using `T1w / FLAIR` and `FLAIR / T1w`

Denominators with absolute value below `1e-3` are replaced by `1e-3`, and ratios are clipped to `[-10, 10]`. Each stream produces 16 channels. Two `1x1x1` fusion convolutions compress the concatenated features from 48 to 16 and then to four latent channels.

### Self-Attention U-Net

The attention comparator projects the two-channel input to 16 dimensions, adaptively pools it to a `6x6x6` grid, applies four-head self-attention and a feed-forward layer, trilinearly upsamples the result, projects it back to two channels, and adds it residually to the original input.

## Data Layout

The experiment configs use nnU-Net v2-style input:

```text
data/
  dataset.json
  imagesTr/
    FCD_001_0000.nii.gz   # T1w
    FCD_001_0001.nii.gz   # FLAIR
    CON_001_0000.nii.gz
    CON_001_0001.nii.gz
  labelsTr/
    FCD_001.nii.gz
    CON_001.nii.gz        # empty control mask
```

The default channel mapping is `_0000` for T1w and `_0001` for FLAIR. When present, `dataset.json` channel names are used to resolve the mapping. Subject IDs beginning with `FCD_` are treated as patients and IDs beginning with `CON_` as controls.

The original subject-folder loader is also retained for datasets organised as:

```text
data/
  subjects/<dataset>/<subject>/
    T1w.nii.gz
    FLAIR.nii.gz
    label.nii.gz
    brain_mask.nii.gz
  controls/<dataset>/<subject>/
    T1w.nii.gz
    FLAIR.nii.gz
    brain_mask.nii.gz
```

For each modality, the loader calculates the mean and standard deviation inside `brain_mask.nii.gz`, applies z-score normalisation, and sets voxels outside the mask to zero. If no brain mask is available, the loader uses the full volume.

## Experiment Settings

The bundled configs reproduce the common settings:

- five-fold stratified cross-validation with shared partitions and seed 42
- 15% of each development set reserved for validation
- `96x96x96` patches and batch size 4
- 500 epochs
- learning rate `1e-4` and weight decay `1e-5`
- mixed-precision training when CUDA is available
- lesion-centred sampling probability 0.7 for subjects with non-empty masks
- no data augmentation

The implementation uses `torch.optim.AdamW`. The accepted paper refers to this as Adam while also reporting weight decay; the optimizer implementation is left unchanged to preserve the experiment code path.

## Losses

The paper evaluates every architecture with two objectives:

- `dice_bce`: equally weighted soft Dice and binary cross-entropy
- `focal_tversky_focal`: equal-weight Focal Tversky and sigmoid focal components

The FTF defaults are `alpha=0.7`, `beta=0.3`, Tversky `gamma=1.33`, focal `alpha=0.25`, focal `gamma=2.0`, and smoothing `1e-5`.

## Setup

Create the environment and install dependencies:

```bash
bash code/scripts/setup_env.sh
source .venv/bin/activate
```

## Training

Run one architecture with the default Dice-BCE loss:

```bash
python code/scripts/train_experiment.py --config code/configs/exp_a_unet_e5.json
python code/scripts/train_experiment.py --config code/configs/exp_b_cril_unet.json
python code/scripts/train_experiment.py --config code/configs/exp_d_attn_unet.json
```

Run the reported FTF condition by overriding the loss:

```bash
python code/scripts/train_experiment.py \
  --config code/configs/exp_b_cril_unet.json \
  --loss-name focal_tversky_focal
```

Useful runtime overrides include:

```bash
python code/scripts/train_experiment.py \
  --config code/configs/exp_a_unet_e5.json \
  --data-root /path/to/dataset \
  --output-root /path/to/outputs \
  --fold-index 0
```

The shared split is stored at `data/splits/shared_5fold_split.json`. Each fold writes its checkpoint, history, split, summary, training curve, and held-out prediction masks beneath:

```text
data/outputs/<experiment>_<loss>_e500/fold_XX/
```

## SLURM

The included script launches one cross-validation fold per array task:

```bash
sbatch code/scripts/slurm_train.sh \
  --exp b \
  --data-root /path/to/dataset \
  --output-root /path/to/outputs \
  --loss-name focal_tversky_focal
```

Supported experiment shortcuts are `a`, `b`, and `d`.

## Inference and Analysis

Held-out predictions use sliding-window inference with 50% overlap. Overlapping logits are averaged, sigmoid probabilities are calculated, and masks are thresholded at `0.5`. No connected-component or other post-processing is applied.

Recalculate the paper metrics from exported predictions with:

```bash
python code/scripts/analyze_predictions.py \
  --run-dir data/outputs/exp_b_ftf_e500
```

The analyzer reports Dice, sensitivity, precision, lesion misses, and non-empty control predictions. Metrics are written separately for all subjects, FCD subjects, and healthy controls, with the fold-wise summary calculated from FCD subjects.

## Repository Layout

```text
code/
  configs/              # exp_a, exp_b, and exp_d
  scripts/              # setup, training, SLURM, and analysis entry points
  src/fcd_latent_ratio/ # data, losses, metrics, models, and training
data/                   # local datasets, splits, outputs, and logs (ignored)
```
