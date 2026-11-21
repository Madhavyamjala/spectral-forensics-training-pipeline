#!/bin/bash
## JOB 1 (CPU): unpack archives, write lists, normalize all external videos (ffmpeg). No GPU.
#SBATCH --job-name=safer-prepare
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
cd "$SAFER_CODE"
load_modules
export PATH=$P/bin:$PATH
source "$P/.venv/bin/activate"
python build_datasets.py extract
python build_datasets.py csv
python benchmark_external_outcomes.py --steps prepare --out "$P/ext" --workers 16 \
  --dataset vifbench=folders:"$P/data/vifbench/source_videos" \
  --dataset genbuster=csv:"$P/lists/genbuster.csv" \
  --dataset fakeparts=csv:"$P/lists/fakeparts.csv" \
  --dataset genvidbench=csv:"$P/lists/genvidbench.csv"
python - <<'EOF'
import os, glob, pandas as pd
for f in sorted(glob.glob(os.environ["SAFER_ROOT"] + "/ext/manifests/*.csv")):
    m = pd.read_csv(f); print(f, len(m), m.groupby("label").size().to_dict(), "usable", int(m.get("usable", pd.Series([True]*len(m))).sum()))
EOF
echo PREPARE_DONE
