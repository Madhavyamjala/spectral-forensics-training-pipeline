#!/bin/bash
# Submit the external rerun as one dependency chain: prepare (CPU) -> smoke (GPU) -> full (4 GPU shards) -> merge (CPU).
# usage: bash submit_rerun.sh            (after download_public.sh, build_datasets.py fakeparts and download_models.sh)
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
C=$SAFER_CODE/scripts/cluster; cd "$SAFER_ROOT"; mkdir -p logs
for f in vifbench_source_videos.zip GenBuster-Bench.zip genvidbench_ms.rar genvidbench_t2vz.rar genvidbench_vript.rar; do
  [ -s "$SAFER_ROOT/raw/$f" ] || { echo "missing raw/$f - run download_public.sh first"; exit 1; }
done
j1=$(sbatch --parsable "$C/job_prepare.slurm")
j2=$(sbatch --parsable --dependency=afterok:$j1 "$C/job_smoke.slurm")
j3=$(sbatch --parsable --dependency=afterok:$j2 "$C/job_full.slurm")
j4=$(sbatch --parsable --dependency=afterok:$j3 "$C/job_merge.slurm")
echo "prepare=$j1 smoke=$j2 full=$j3 merge=$j4"
