#!/bin/bash
## D3 scores on the normalized videos of the four benchmarks: 4 processes (one per benchmark) share one GPU.
#SBATCH --job-name=d3-scores
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%j.out
source "${SAFER_CODE:?set SAFER_CODE}/scripts/cluster/d3_common.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p "$D"
for b in $BENCHES; do
  python -m safer.eval.d3_scores --videos "$P/ext/manifests/$b.csv" --out "$D/$b.csv" --limit $LIMIT > "$D/$b.log" 2>&1 &
done
wait
tail -n 3 "$D"/*.log
