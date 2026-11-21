# sourced by every vae_* job. MODE=smoke (5 clips) or full (346 clips) is passed with sbatch --export.
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
V=$G/vae_${MODE}
LIMIT=$([ "$MODE" = smoke ] && echo 5 || echo 0)
cd "$SAFER_CODE"
load_modules
export PATH=$P/bin:$PATH CSF_FFMPEG=$P/bin/ffmpeg
source "$P/.venv/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONPATH=$SAFER_CODE
VARIANTS="identity h264_crf18 h264_crf28 h264_crf35 h265_crf28 wan cog"
DS=""; for v in $VARIANTS; do DS="$DS --dataset vae_$v=csv:$V/lists/vae_$v.csv"; done
