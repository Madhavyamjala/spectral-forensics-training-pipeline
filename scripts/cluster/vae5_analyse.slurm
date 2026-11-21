#!/bin/bash
#SBATCH --job-name=vae-analyse
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
source "${SAFER_CODE:?set SAFER_CODE}/scripts/cluster/vae_common.sh"
python benchmark_external_outcomes.py --steps outcomes_merge --out "$V/bench" $DS
mkdir -p "$P/rebuttal_out/vae_$MODE"
python -m safer.eval.rebuttal_vae --dir "$V/bench/outcomes" --manifests "$V/bench/manifests" \
  --ext "$P/ext/outcomes" --out "$P/rebuttal_out/vae_$MODE"
