#!/usr/bin/env bash
set -euo pipefail


CUDA_TAG="${1:-cu124}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-graml}"

echo "==> Setting up GRAML environment: ${CONDA_ENV} (${CUDA_TAG})"

if ! conda env list | grep -q "${CONDA_ENV}"; then
    echo "==> Creating conda environment: ${CONDA_ENV}"
    conda create -y -n "${CONDA_ENV}" python=3.12
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "==> Installing PyTorch (${CUDA_TAG})"
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Installing Unsloth"
pip install "unsloth[${CUDA_TAG}]" unsloth_zoo

echo "==> Installing remaining dependencies"
pip install -r requirements.txt

echo ""
echo "==> Done. Activate with: conda activate ${CONDA_ENV}"
