#!/bin/bash
## JOB 4 (CPU): gather the 4 shards into one outcome table per benchmark (submitted afterok on JOB 3).
#SBATCH --job-name=safer-merge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
cd "$SAFER_CODE"
load_modules
export PATH=$P/bin:$PATH
source "$P/.venv/bin/activate"
python benchmark_external_outcomes.py --steps outcomes_merge --out "$P/ext" \
  --dataset vifbench=folders:"$P/data/vifbench/source_videos" \
  --dataset genbuster=csv:"$P/lists/genbuster.csv" \
  --dataset fakeparts=csv:"$P/lists/fakeparts.csv" \
  --dataset genvidbench=csv:"$P/lists/genvidbench.csv"
ls -la "$P/ext/outcomes"
echo MERGE_DONE
