#!/bin/bash
## JOB 3 (A1, full rerun): all four external benchmarks (15,016 videos) through the outcome run,
## with the 23 tool features saved. 4 array tasks = 4 disjoint shards, 1 GPU each. Resumable:
## a task that hits the time limit can be resubmitted and skips finished videos.
#SBATCH --job-name=safer-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
cd "$SAFER_CODE"
load_modules
export PATH=$P/bin:$PATH
source "$P/.venv/bin/activate"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python benchmark_external_outcomes.py --steps csf_outcomes --shard "${SLURM_ARRAY_TASK_ID}/4" \
  --out "$P/ext" --model-dir "$P/export" \
  --dataset vifbench=folders:"$P/data/vifbench/source_videos" \
  --dataset genbuster=csv:"$P/lists/genbuster.csv" \
  --dataset fakeparts=csv:"$P/lists/fakeparts.csv" \
  --dataset genvidbench=csv:"$P/lists/genvidbench.csv"
echo "SHARD_DONE ${SLURM_ARRAY_TASK_ID}"
