#!/bin/bash
#SBATCH --job-name=2dfm-train
# Adjust this partition if your PACE allocation uses a different GPU queue.
#SBATCH --partition=A100-40GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/2dFM}"
CONFIG_PATH="${CONFIG_PATH:-configs/mnist_imf_mlp.yaml}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

cd "$REPO_ROOT"

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

module purge
module load anaconda3
module load cuda

source "$VENV_DIR/bin/activate"

python src/train.py --config "$CONFIG_PATH"
