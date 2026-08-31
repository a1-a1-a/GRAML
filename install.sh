#!/usr/bin/env bash
set -euo pipefail

# One-click environment setup for GRAML
# Usage: bash install.sh [cu121|cu124|cu128]
#   cu121 -> CUDA 12.1   (recommended for most GPUs)
#   cu124 -> CUDA 12.4
#   cu128 -> CUDA 12.8   (latest, e.g. RTX 5090)
# Default: cu121

CUDA_TAG="${1:-cu124}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-graml}"

echo "==> Setting up GRAML environment: ${CONDA_ENV} (${CUDA_TAG})"

# 1. Create conda env if it does not exist
if ! conda env list | grep -q "${CONDA_ENV}"; then
    echo "==> Creating conda environment: ${CONDA_ENV}"
    conda create -y -n "${CONDA_ENV}" python=3.12
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# 2. Install PyTorch with the requested CUDA tag
echo "==> Installing PyTorch (${CUDA_TAG})"
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# 3. Install Unsloth (built against the installed PyTorch)
echo "==> Installing Unsloth"
pip install "unsloth[${CUDA_TAG}]" unsloth_zoo

# 4. Install remaining dependencies
echo "==> Installing remaining dependencies"
pip install -r requirements.txt

echo ""
echo "==> Done. Activate with: conda activate ${CONDA_ENV}"
