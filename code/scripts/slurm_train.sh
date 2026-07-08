#!/usr/bin/env bash
#SBATCH --job-name=fcd_latent_ratio
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
mkdir -p "$LOG_DIR"

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/code/configs/exp_a_unet_e5.json}"
DATA_ROOT_OVERRIDE="${DATA_ROOT_OVERRIDE:-}"
OUTPUT_ROOT_OVERRIDE="${OUTPUT_ROOT_OVERRIDE:-$REPO_ROOT/data/outputs}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

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

echo "Running on host: $(hostname)"
echo "Config: $CONFIG_PATH"
echo "Data root override: ${DATA_ROOT_OVERRIDE:-<config default>}"
echo "Output root: $OUTPUT_ROOT_OVERRIDE"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
