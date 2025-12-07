#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="3dgs_statue"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating conda env ${ENV_NAME}..."
  conda create -y -n "${ENV_NAME}" python=3.10
fi

echo "Activating ${ENV_NAME}..."
conda activate "${ENV_NAME}"

conda install -y -c conda-forge -c pytorch -c nvidia \
  numpy opencv imageio tqdm scikit-image huggingface_hub transformers \
  accelerate typer open3d ffmpeg rembg colmap pytorch-cuda=12.1

pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.1+cu121" \
  "torchvision==0.18.1+cu121" \
  "torchaudio==2.3.1+cu121"

pip install "gsplat==1.4.0" nerfstudio sam2

pip install -e .

echo "Environment ${ENV_NAME} ready. Activate it with 'conda activate ${ENV_NAME}'."
