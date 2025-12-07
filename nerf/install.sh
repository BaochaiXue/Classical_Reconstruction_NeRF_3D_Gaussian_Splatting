#!/usr/bin/env bash
set -euo pipefail

# Nerf (Nerfstudio) environment setup
# Adjust CUDA wheel index to match your driver/toolkit (cu121/cu124/cu118).

ENV_NAME=nerf
PY_VER=3.12
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

echo "[info] Creating conda env '${ENV_NAME}' (python=${PY_VER})..."
conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
echo "[info] Activate with: conda activate ${ENV_NAME}"
echo "[info] Installing PyTorch from ${TORCH_INDEX} (edit TORCH_INDEX if needed)..."
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install torch torchvision --index-url "${TORCH_INDEX}"

echo "[info] Installing Nerfstudio + helpers..."
conda run -n "${ENV_NAME}" pip install nerfstudio rich pyyaml

echo "[info] Installing COLMAP + FFmpeg (from conda-forge)..."
conda run -n "${ENV_NAME}" conda install -y -c conda-forge colmap ffmpeg

echo "[info] Install this project in editable mode..."
conda run -n "${ENV_NAME}" pip install -e .

cat <<'DONE'
[done] Environment ready.
To use:
  conda activate nerf
  ns-process-data --help
  ns-train --help
If PyTorch or CUDA versions mismatch, edit TORCH_INDEX / pip torch line above.
DONE
