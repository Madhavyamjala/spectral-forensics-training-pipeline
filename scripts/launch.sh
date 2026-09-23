#!/usr/bin/env bash
# Linux launcher: runs main.py with the given config on all visible GPUs.
# 1 GPU -> python main.py ... | N GPUs -> torchrun --standalone --nproc_per_node=N (NCCL); torchrun's per-rank
# stdout/stderr are also captured under <repo>/runs/torchrun_logs for post-mortem debugging.
# Usage: bash scripts/launch.sh configs/full.yaml [main.py args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="$1"; shift
PY="$ROOT/.venv/bin/python"; [[ -x "$PY" ]] || PY=python3

export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

GPUS=$("$PY" -c "import torch; print(torch.cuda.device_count())")
echo "==> Config $CONFIG | GPUs detected: $GPUS"

# The tool-dependency benchmark is intentionally single-GPU: it is an exhaustive fixed-subset
# ablation, not a training job. Avoid starting unused distributed workers when it is requested
# as the only stage. Latency remains multi-GPU when launched normally.
STAGE_ARG=""
for arg in "$@"; do
  if [[ "$arg" == "--stage" ]]; then
    STAGE_ARG="__NEXT__"
  elif [[ "$STAGE_ARG" == "__NEXT__" ]]; then
    STAGE_ARG="$arg"
  fi
done
TOOLDEP_ONLY=0
if [[ "$STAGE_ARG" == "tooldependency" && "$*" != *"latency"* ]]; then
  TOOLDEP_ONLY=1
fi

if [[ "$TOOLDEP_ONLY" -eq 1 ]]; then
  echo "==> tooldependency stage: single-GPU mode"
  exec "$PY" main.py --config "$CONFIG" "$@"
fi

if [[ "$GPUS" -gt 1 ]]; then
  mkdir -p runs/torchrun_logs
  exec "$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
       --log-dir runs/torchrun_logs --tee 3 main.py --config "$CONFIG" "$@"
else
  exec "$PY" main.py --config "$CONFIG" "$@"
fi
