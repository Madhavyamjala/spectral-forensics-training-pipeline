#!/bin/bash
# usage: bash submit_vae.sh smoke|full     (submits the 5-job chain with afterok dependencies)
set -euo pipefail
MODE=$1
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
C=$SAFER_CODE/scripts/cluster; cd "$SAFER_ROOT"; mkdir -p logs
sub() { local out; out=$(sbatch --parsable --export=ALL,MODE=$MODE "$@") || true
        [ -n "$out" ] || out=$(squeue -u $USER -h -o "%i %j" --sort=-i | awk -v n="vae-$(basename ${@: -1} .slurm | cut -d_ -f2)" '$2==n{print $1; exit}')
        echo "${out%%;*}"; }
j1=$(sub $C/vae1_make.slurm)
j2=$(sub --dependency=afterok:$j1 --array=0-1 $C/vae2_roundtrip.slurm)
j3=$(sub --dependency=afterok:$j2 $C/vae3_prepare.slurm)
A=$([ "$MODE" = smoke ] && echo 0 || echo 0-1)
j4=$(sub --dependency=afterok:$j3 --array=$A $C/vae4_outcomes.slurm)
j5=$(sub --dependency=afterok:$j4 $C/vae5_analyse.slurm)
echo "MODE=$MODE make=$j1 roundtrip=$j2 prepare=$j3 outcomes=$j4 analyse=$j5"
