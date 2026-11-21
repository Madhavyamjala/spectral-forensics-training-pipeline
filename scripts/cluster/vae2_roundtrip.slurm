#!/bin/bash
#SBATCH --job-name=vae-roundtrip
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%A_%a.out
source "${SAFER_CODE:?set SAFER_CODE}/scripts/cluster/vae_common.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m safer.eval.vae_roundtrip --list "$P/lists/vae_clips.txt" --out "$V" --limit $LIMIT --stage vae --vae $([ "${SLURM_ARRAY_TASK_ID:-0}" = 0 ] && echo wan || echo cog)
