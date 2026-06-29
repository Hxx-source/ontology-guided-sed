#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

python src/main_htsat_hier.py \
  --config_dir configs/train_ocl_caw.yaml \
  --save_folder runs/ocl_caw \
  --gpu "${GPU_ID}"
