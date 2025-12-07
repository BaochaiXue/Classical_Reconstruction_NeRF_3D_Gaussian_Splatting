#!/usr/bin/env bash
set -euo pipefail

# Classical SfM/MVS environment setup using conda + COLMAP.

ENV_NAME=classical
PY_VER=3.12

echo "[info] Creating conda env '${ENV_NAME}' (python=${PY_VER})..."
conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
echo "[info] Activate with: conda activate ${ENV_NAME}"

echo "[info] Installing COLMAP + FFmpeg from conda-forge..."
conda run -n "${ENV_NAME}" conda install -y -c conda-forge colmap ffmpeg

echo "[info] Installing Python deps..."
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install pyyaml rich

echo "[info] Installing this project in editable mode..."
conda run -n "${ENV_NAME}" pip install -e .

cat <<'DONE'
[done] Environment ready.
To use:
  conda activate classical
  colmap --help
  python -m classical_recon.cli --help
DONE
