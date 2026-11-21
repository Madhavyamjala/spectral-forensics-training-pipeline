#!/bin/bash
## CPU: D3 vs SAFER zero-shot + fusion upper bound with D3 (same splits as rebuttal_fusion).
#SBATCH --job-name=d3-analyse
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source "${SAFER_CODE:?set SAFER_CODE}/scripts/cluster/d3_common.sh"
python -m safer.eval.rebuttal_d3 --dir "$P/ext/outcomes" --d3 "$D/{dataset}.csv" --out "$P/rebuttal_out/d3_$MODE"
