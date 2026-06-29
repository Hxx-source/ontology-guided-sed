#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

python src/run_val_pred_psds_quantile_mf.py \
  --config configs/eval_camf.yaml \
  --save_folder runs/camf \
  --model_path weights/ocl_caw_best_model.pt \
  --gpu "${GPU_ID}" \
  --threshold 0.5 \
  --psds_num_jobs 1 \
  --psds_heartbeat_sec 30
