#!/bin/bash
## JOB 2 (GPU smoke test): 50 ViF-Bench videos through the full outcome run, with features saved.
## Checks: models load (Llama-11B, Qwen-3B, SVD VAE - not the fallback), s/video, features present,
## and the 50 videos' scores agree with the original outcome table (same video_ids).
#SBATCH --job-name=safer-smoke
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:a30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
cd "$SAFER_CODE"
load_modules
export PATH=$P/bin:$PATH
source "$P/.venv/bin/activate"
export HF_HUB_OFFLINE=1          # models were pre-downloaded on the login node
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p "$P/smoke/manifests"
python - <<'EOF'
import os
import pandas as pd
m = pd.read_csv(os.environ["SAFER_ROOT"] + "/ext/manifests/vifbench.csv", dtype={"video_id": str})
m = m[m["usable"].astype(bool)]
pd.concat([m[m.label == "real"].head(10), m[m.label != "real"].head(40)]).to_csv(
    os.environ["SAFER_ROOT"] + "/smoke/manifests/vifbench.csv", index=False)
EOF
cp "$P/ext/manifests/_prepare.json" "$P/smoke/manifests/" 2>/dev/null || true
python benchmark_external_outcomes.py --steps csf_outcomes,outcomes_merge --out "$P/smoke" \
  --model-dir "$P/export" --dataset vifbench=folders:"$P/data/vifbench/source_videos"
python - <<'EOF'
import os
import numpy as np
new = np.load(os.environ["SAFER_ROOT"] + "/smoke/outcomes/vifbench.npz", allow_pickle=True)
old = np.load(os.environ["SAFER_ROOT"] + "/old_tables/vifbench.npz", allow_pickle=True)
pos = {v: i for i, v in enumerate(old["video_id"])}
idx = [pos[v] for v in new["video_id"]]
for k in ("scanner_probs", "notool_probs", "static_probs"):
    d = np.abs(new[k][:, 0] - old[k][idx, 0]); print(k, "max |diff| p_real", float(d.max()), "share>0.02", float((d > 0.02).mean()))
print("features_raw", new["features_raw"].shape, "nan share", float(np.isnan(new["features_raw"]).mean()))
print("mean s/video (tools+passes)", float(new["t_tools"].sum(1).mean() + new["t_state"].mean() + new["t_static"].mean() + new["t_actions"].sum(1).mean()))
EOF
echo SMOKE_DONE
