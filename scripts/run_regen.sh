#!/usr/bin/env bash
# AI-Edited regeneration + training, for tfgpu.cs.fiu.edu (6x H200 NVL; GPUs 0-1 are held by
# vLLM, so only 2/3/4 are used).
#
# Phases - run them in order, each is resumable, re-running skips what is done:
#   bash scripts/run_regen.sh --envs      build the per-model virtual environments (do this first)
#   bash scripts/run_regen.sh --kinetics  download + qualify the Kinetics-400 source clips
#   bash scripts/run_regen.sh --generate  render the 33,333 AI-Edited videos on GPUs 2/3/4
#   bash scripts/run_regen.sh --manifest  rebuild manifest.csv around what was produced
#   bash scripts/run_regen.sh --train     run the training pipeline (torchrun on GPUs 2/3/4)
#   bash scripts/run_regen.sh --push      replace the AI-Edited class on the Hub (destructive)
#   bash scripts/run_regen.sh --all       kinetics + generate + manifest + train
#
# Any extra arguments are passed through to main.py, e.g.
#   bash scripts/run_regen.sh --generate --set generation.only_models='[inswapper]'
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${CSF_CONFIG:-configs/regen.yaml}"
PY="$ROOT/.venv/bin/python"; [[ -x "$PY" ]] || PY=python3

export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# GPUs 0 and 1 run the vLLM workers - never touch them.
export CSF_TRAIN_GPUS="${CSF_TRAIN_GPUS:-2,3,4}"

PHASE="${1:---all}"; shift || true

gen_stage() {   # generation stages are single-process: they schedule their own per-GPU workers
  echo "==> [$1] config=$CONFIG"
  exec "$PY" main.py --config "$CONFIG" --stage "$1" "$@"
}

case "$PHASE" in
  --envs)
    echo "==> Building per-model environments (this pulls torch + repos + weights; allow ~1-2 h)"
    exec "$PY" -m csf.generation.envs --build all \
         --envs-root "$("$PY" -c "from csf.config import load_config; print(load_config('$CONFIG').generation.envs_root)")"
    ;;
  --status)
    exec "$PY" -m csf.generation.envs --status \
         --envs-root "$("$PY" -c "from csf.config import load_config; print(load_config('$CONFIG').generation.envs_root)")"
    ;;
  --plan)     exec "$PY" -m csf.generation.spec ;;
  --adapters) exec "$PY" -m csf.generation.adapters ;;
  --kinetics) gen_stage kinetics "$@" ;;
  --generate) gen_stage generate "$@" ;;
  --manifest) gen_stage regen_manifest "$@" ;;
  --push)     gen_stage push_dataset --set generation.push.enabled=true "$@" ;;
  --train)
    export CUDA_VISIBLE_DEVICES="$CSF_TRAIN_GPUS"
    NGPU=$(awk -F, '{print NF}' <<< "$CSF_TRAIN_GPUS")
    echo "==> Training on GPUs $CSF_TRAIN_GPUS ($NGPU process(es))"
    mkdir -p runs/torchrun_logs
    if [[ "$NGPU" -gt 1 ]]; then
      exec "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" \
           --log-dir runs/torchrun_logs --tee 3 main.py --config "$CONFIG" \
           --stage prepare,features,train_qwen,train_llama,predict_scanner,outcomes,train_dispatcher,evaluate,export,latency "$@"
    else
      exec "$PY" main.py --config "$CONFIG" \
           --stage prepare,features,train_qwen,train_llama,predict_scanner,outcomes,train_dispatcher,evaluate,export,latency "$@"
    fi
    ;;
  --all)
    "$PY" main.py --config "$CONFIG" --stage kinetics "$@"
    "$PY" main.py --config "$CONFIG" --stage generate "$@"
    "$PY" main.py --config "$CONFIG" --stage regen_manifest "$@"
    exec bash "$0" --train "$@"
    ;;
  *)
    echo "Unknown phase '$PHASE'." >&2
    sed -n '2,18p' "$0" >&2
    exit 2
    ;;
esac
