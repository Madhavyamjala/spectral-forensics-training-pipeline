#!/bin/bash
#SBATCH --job-name=vae-prepare
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source "${SAFER_CODE:?set SAFER_CODE}/scripts/cluster/vae_common.sh"
python benchmark_external_outcomes.py --steps prepare --out "$V/bench" $DS
