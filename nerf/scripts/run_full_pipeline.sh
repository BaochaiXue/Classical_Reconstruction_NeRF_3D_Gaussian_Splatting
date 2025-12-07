#!/usr/bin/env bash
set -euo pipefail

# Run the full Nerf pipeline with a single command.
# Usage: scripts/run_full_pipeline.sh [--skip-processing] [--skip-training] [--skip-exports]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! conda env list | grep -q "^nerf "; then
  echo "[warn] Conda env 'nerf' not found. Run: bash install.sh" >&2
fi

python -m ns_nerf.cli "$@"
