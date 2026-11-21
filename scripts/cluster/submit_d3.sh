#!/bin/bash
set -euo pipefail
MODE=$1
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
C=$SAFER_CODE/scripts/cluster; cd "$SAFER_ROOT"; mkdir -p logs
j1=$(sbatch --parsable --export=ALL,MODE=$MODE $C/d3_1_scores.slurm)
j2=$(sbatch --parsable --export=ALL,MODE=$MODE --dependency=afterok:$j1 $C/d3_2_analyse.slurm)
echo "d3 $MODE: scores $j1, analyse $j2"
