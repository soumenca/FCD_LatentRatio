#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"

echo "Repo root: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Virtual environment: $VENV_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python executable '$PYTHON_BIN' was not found." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

PIP_ARGS=()
if [[ "$PIP_NO_CACHE_DIR" == "1" ]]; then
  PIP_ARGS+=(--no-cache-dir)
fi
if [[ -n "$PIP_EXTRA_INDEX_URL" ]]; then
  PIP_ARGS+=(--extra-index-url "$PIP_EXTRA_INDEX_URL")
fi

python -m pip install "${PIP_ARGS[@]}" --upgrade pip setuptools wheel
python -m pip install "${PIP_ARGS[@]}" -r "$REPO_ROOT/requirements.txt"

python - <<'PY'
import matplotlib  # noqa: F401
import nibabel  # noqa: F401
import numpy  # noqa: F401
import torch  # noqa: F401

print("Environment setup complete. Core imports succeeded.")
PY

cat <<EOF

Environment is ready.
Activate it with:
  source "$VENV_DIR/bin/activate"

Train with:
  python "$REPO_ROOT/code/scripts/train_experiment.py" --config "$REPO_ROOT/code/configs/exp_a_unet_e5.json"

Useful overrides:
  PYTHON_BIN=python3.11 bash "$REPO_ROOT/code/scripts/setup_env.sh"
  VENV_DIR=/path/to/venv bash "$REPO_ROOT/code/scripts/setup_env.sh"
  PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 bash "$REPO_ROOT/code/scripts/setup_env.sh"
EOF
