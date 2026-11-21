#!/bin/bash
#SBATCH --job-name=vae-outcomes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%A_%a.out
source "${SAFER_CODE:?set SAFER_CODE}/scripts/cluster/vae_common.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
NS=$([ "$MODE" = smoke ] && echo 1 || echo 2)
[ "${SLURM_ARRAY_TASK_ID:-0}" -lt "$NS" ] || exit 0
python benchmark_external_outcomes.py --steps csf_outcomes --shard "${SLURM_ARRAY_TASK_ID:-0}/$NS" \
  --out "$V/bench" --model-dir "$P/export" $DS
