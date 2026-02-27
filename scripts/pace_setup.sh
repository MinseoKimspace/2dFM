#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_BASE="${SCRATCH:-/scratch/$USER}"
PROJECT_NAME="${PROJECT_NAME:-2dFM}"
SCRATCH_ROOT="${SCRATCH_BASE}/${PROJECT_NAME}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

echo "Repository root: $REPO_ROOT"
echo "Scratch root: $SCRATCH_ROOT"

mkdir -p "$SCRATCH_ROOT/data" "$SCRATCH_ROOT/runs"

if [ -L "$REPO_ROOT/data" ] || [ ! -e "$REPO_ROOT/data" ]; then
  rm -f "$REPO_ROOT/data"
  ln -s "$SCRATCH_ROOT/data" "$REPO_ROOT/data"
else
  echo "Skipping data symlink because $REPO_ROOT/data already exists and is not a symlink."
fi

if [ -L "$REPO_ROOT/runs" ] || [ ! -e "$REPO_ROOT/runs" ]; then
  rm -f "$REPO_ROOT/runs"
  ln -s "$SCRATCH_ROOT/runs" "$REPO_ROOT/runs"
else
  echo "Skipping runs symlink because $REPO_ROOT/runs already exists and is not a symlink."
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"

echo "PACE setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"
