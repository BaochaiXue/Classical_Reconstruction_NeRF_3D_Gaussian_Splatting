#!/usr/bin/env bash
set -euo pipefail

# Translate deprecated COLMAP flags used by nerfstudio into the 3.13 syntax.
COLMAP_BIN="${COLMAP_BIN:-$(command -v colmap)}"
if [[ -z "${COLMAP_BIN}" ]]; then
  echo "[colmap-wrapper] colmap binary not found; set COLMAP_BIN." >&2
  exit 1
fi

args=()
for arg in "$@"; do
  if [[ "${arg}" == "--SiftExtraction.use_gpu" ]]; then
    args+=(--FeatureExtraction.use_gpu)
  else
    args+=("${arg}")
  fi
done

exec "${COLMAP_BIN}" "${args[@]}"
