# sourced by the d3 jobs. MODE=smoke (20 videos per benchmark) or full, passed with sbatch --export.
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
D=$G/d3_${MODE}
LIMIT=$([ "$MODE" = smoke ] && echo 20 || echo 0)
cd "$SAFER_CODE"
load_modules
export PATH=$P/bin:$PATH CSF_FFMPEG=$P/bin/ffmpeg
source "$P/.venv/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONPATH=$SAFER_CODE
BENCHES="vifbench fakeparts genbuster genvidbench"
