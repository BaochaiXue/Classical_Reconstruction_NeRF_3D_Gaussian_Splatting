#!/usr/bin/env bash
set -euo pipefail

# Run the full classical reconstruction pipeline.
# Usage: scripts/run_full_pipeline.sh [--skip-processing] [--skip-colmap]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! conda env list | grep -q "^classical "; then
  echo "[warn] Conda env 'classical' not found. Run: bash install.sh" >&2
fi

python -m classical_recon.cli "$@"
