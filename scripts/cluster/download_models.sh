#!/usr/bin/env bash
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
source $P/.venv/bin/activate
export HF_TOKEN_PATH=$HOME/.huggingface/token
export HF_HUB_ENABLE_HF_TRANSFER=0
python $SAFER_CODE/download_models.py
du -sh $HF_HOME
