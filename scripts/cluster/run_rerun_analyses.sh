#!/bin/bash
# CPU only (login node or a CPU job): fusion bound, feature direction, reproduction check and the external
# analyses on the rerun tables written by job_merge.slurm. Outputs: $SAFER_ROOT/rebuttal_out/.
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
load_modules
source "$P/.venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$SAFER_CODE"
mkdir -p "$P/rebuttal_out"
# --old: the original external outcome tables (runs/extbench_outcomes in the paper run), for the reproduction check
python -m safer.eval.rebuttal_fusion --dir "$P/ext/outcomes" --out "$P/rebuttal_out" --old "$P/old_tables"
python -m safer.eval.rebuttal_external --dir "$P/ext/outcomes" --out "$P/rebuttal_out"
echo ANALYSES_DONE
