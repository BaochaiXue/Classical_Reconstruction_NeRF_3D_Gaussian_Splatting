#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[warn] No conda environment active. Activate 3dgs_statue before running this script." >&2
fi

python -m gs_statue.cli --config "${PROJECT_ROOT}/configs/pipeline.yaml" "$@"
