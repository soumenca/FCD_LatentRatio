#!/usr/bin/env bash
#SBATCH --job-name=UNet_CV
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=./logs/%x_fold%a_%j.out
#SBATCH --error=./logs/%x_fold%a_%j.err
#SBATCH --array=0-4

set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${SCRIPT_REPO_ROOT}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/code" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
mkdir -p "$LOG_DIR"

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/code/configs/exp_a_unet_e5.json}"
DATA_ROOT_OVERRIDE="${DATA_ROOT_OVERRIDE:-}"
OUTPUT_ROOT_OVERRIDE="${OUTPUT_ROOT_OVERRIDE:-$REPO_ROOT/data/outputs}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
FOLD_INDEX_OVERRIDE="${FOLD_INDEX_OVERRIDE:-${SLURM_ARRAY_TASK_ID:-}}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
LOSS_NAME_OVERRIDE="${LOSS_NAME_OVERRIDE:-}"
LOSS_BCE_WEIGHT_OVERRIDE="${LOSS_BCE_WEIGHT_OVERRIDE:-}"
LOSS_ALPHA_OVERRIDE="${LOSS_ALPHA_OVERRIDE:-}"
LOSS_BETA_OVERRIDE="${LOSS_BETA_OVERRIDE:-}"
LOSS_GAMMA_OVERRIDE="${LOSS_GAMMA_OVERRIDE:-}"
LOSS_SMOOTH_OVERRIDE="${LOSS_SMOOTH_OVERRIDE:-}"
LOSS_FOCAL_ALPHA_OVERRIDE="${LOSS_FOCAL_ALPHA_OVERRIDE:-}"
LOSS_FOCAL_GAMMA_OVERRIDE="${LOSS_FOCAL_GAMMA_OVERRIDE:-}"
LOSS_FOCAL_WEIGHT_OVERRIDE="${LOSS_FOCAL_WEIGHT_OVERRIDE:-}"
LOSS_TVERSKY_WEIGHT_OVERRIDE="${LOSS_TVERSKY_WEIGHT_OVERRIDE:-}"

usage() {
  cat <<EOF
Usage:
  sbatch code/scripts/slurm_train.sh [options]

Options:
  --exp {a|b|c|d}       Shortcut for bundled configs:
                        a -> exp_a_unet_e5.json
                        b -> exp_b_cril_unet.json
                        c -> exp_c_unet_with_ratios.json
                        d -> exp_d_cril_attn_unet.json
  --config PATH         Explicit config path
  --data-root PATH      Override dataset root
  --output-root PATH    Override output root
  --venv-dir PATH       Override virtual environment path
  --modules "A B C"     Modules to load before running
  --epochs N            Override epochs
  --fold-index N        Override fold index; defaults to SLURM_ARRAY_TASK_ID
  --loss-name NAME      Override loss: dice_bce, focal_tversky, focal_tversky_focal
  --loss-bce-weight X   Override BCE weight for dice_bce
  --loss-alpha X        Override Tversky alpha
  --loss-beta X         Override Tversky beta
  --loss-gamma X        Override Focal Tversky gamma
  --loss-smooth X       Override Tversky smoothing term
  --loss-focal-alpha X  Override focal alpha for focal_tversky_focal
  --loss-focal-gamma X  Override focal gamma for focal_tversky_focal
  --loss-focal-weight X Override focal component weight for focal_tversky_focal
  --loss-tversky-weight X Override Tversky component weight for focal_tversky_focal
  --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)
      case "${2:-}" in
        a) CONFIG_PATH="$REPO_ROOT/code/configs/exp_a_unet_e5.json" ;;
        b) CONFIG_PATH="$REPO_ROOT/code/configs/exp_b_cril_unet.json" ;;
        c) CONFIG_PATH="$REPO_ROOT/code/configs/exp_c_unet_with_ratios.json" ;;
        d) CONFIG_PATH="$REPO_ROOT/code/configs/exp_d_cril_attn_unet.json" ;;
        *)
          echo "Unknown experiment for --exp: ${2:-<missing>}" >&2
          usage
          exit 1
          ;;
      esac
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --data-root)
      DATA_ROOT_OVERRIDE="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT_OVERRIDE="${2:-}"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --modules)
      MODULES="${2:-}"
      shift 2
      ;;
    --epochs)
      EPOCHS_OVERRIDE="${2:-}"
      shift 2
      ;;
    --fold-index)
      FOLD_INDEX_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-name)
      LOSS_NAME_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-bce-weight)
      LOSS_BCE_WEIGHT_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-alpha)
      LOSS_ALPHA_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-beta)
      LOSS_BETA_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-gamma)
      LOSS_GAMMA_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-smooth)
      LOSS_SMOOTH_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-focal-alpha)
      LOSS_FOCAL_ALPHA_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-focal-gamma)
      LOSS_FOCAL_GAMMA_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-focal-weight)
      LOSS_FOCAL_WEIGHT_OVERRIDE="${2:-}"
      shift 2
      ;;
    --loss-tversky-weight)
      LOSS_TVERSKY_WEIGHT_OVERRIDE="${2:-}"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Optional module init for HPC systems that use environment modules.
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

# Optional module list, for example:
#   MODULES="python/3.11 cuda/12.1"
if [[ -n "${MODULES:-}" ]]; then
  for module_name in $MODULES; do
    module load "$module_name"
  done
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found at $VENV_DIR" >&2
  echo "Create it first with: bash $REPO_ROOT/code/scripts/setup_env.sh" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

CMD=(
  python "$REPO_ROOT/code/scripts/train_experiment.py"
  --config "$CONFIG_PATH"
  --output-root "$OUTPUT_ROOT_OVERRIDE"
)

if [[ -n "$DATA_ROOT_OVERRIDE" ]]; then
  CMD+=(--data-root "$DATA_ROOT_OVERRIDE")
fi
if [[ -n "$EPOCHS_OVERRIDE" ]]; then
  CMD+=(--epochs "$EPOCHS_OVERRIDE")
fi
if [[ -n "$FOLD_INDEX_OVERRIDE" ]]; then
  CMD+=(--fold-index "$FOLD_INDEX_OVERRIDE")
fi
if [[ -n "$LOSS_NAME_OVERRIDE" ]]; then
  CMD+=(--loss-name "$LOSS_NAME_OVERRIDE")
fi
if [[ -n "$LOSS_BCE_WEIGHT_OVERRIDE" ]]; then
  CMD+=(--loss-bce-weight "$LOSS_BCE_WEIGHT_OVERRIDE")
fi
if [[ -n "$LOSS_ALPHA_OVERRIDE" ]]; then
  CMD+=(--loss-alpha "$LOSS_ALPHA_OVERRIDE")
fi
if [[ -n "$LOSS_BETA_OVERRIDE" ]]; then
  CMD+=(--loss-beta "$LOSS_BETA_OVERRIDE")
fi
if [[ -n "$LOSS_GAMMA_OVERRIDE" ]]; then
  CMD+=(--loss-gamma "$LOSS_GAMMA_OVERRIDE")
fi
if [[ -n "$LOSS_SMOOTH_OVERRIDE" ]]; then
  CMD+=(--loss-smooth "$LOSS_SMOOTH_OVERRIDE")
fi
if [[ -n "$LOSS_FOCAL_ALPHA_OVERRIDE" ]]; then
  CMD+=(--loss-focal-alpha "$LOSS_FOCAL_ALPHA_OVERRIDE")
fi
if [[ -n "$LOSS_FOCAL_GAMMA_OVERRIDE" ]]; then
  CMD+=(--loss-focal-gamma "$LOSS_FOCAL_GAMMA_OVERRIDE")
fi
if [[ -n "$LOSS_FOCAL_WEIGHT_OVERRIDE" ]]; then
  CMD+=(--loss-focal-weight "$LOSS_FOCAL_WEIGHT_OVERRIDE")
fi
if [[ -n "$LOSS_TVERSKY_WEIGHT_OVERRIDE" ]]; then
  CMD+=(--loss-tversky-weight "$LOSS_TVERSKY_WEIGHT_OVERRIDE")
fi

echo "Running on host: $(hostname)"
echo "Repo root: $REPO_ROOT"
echo "Config: $CONFIG_PATH"
echo "Data root override: ${DATA_ROOT_OVERRIDE:-<config default>}"
echo "Output root: $OUTPUT_ROOT_OVERRIDE"
echo "Epochs override: ${EPOCHS_OVERRIDE:-<config default>}"
echo "Fold index override: ${FOLD_INDEX_OVERRIDE:-<all folds>}"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
