#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
CONFIG_PATH="${1:-configs/mnist_imf_mlp.yaml}"

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

python src/train.py --config "$CONFIG_PATH"
