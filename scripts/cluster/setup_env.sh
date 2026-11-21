#!/usr/bin/env bash
# Build the SAFER env on a login node (CPU only, no GPU time). Creates $SAFER_ROOT/.venv.
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
cd "$SAFER_CODE"
load_modules
[[ -d "$P/.venv" ]] || python -m venv "$P/.venv"
source "$P/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install "huggingface_hub[cli]" scikit-learn
python -c "import torch, transformers, diffusers; print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| transformers', transformers.__version__, '| diffusers', diffusers.__version__)"
python -c "import safer, csf.inference; print('safer import ok')"
echo SETUP_DONE
