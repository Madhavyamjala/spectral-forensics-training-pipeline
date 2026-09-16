#!/usr/bin/env bash
# Linux environment setup for Chrono-Spectral Forensics.
# Creates .venv, installs a CUDA build of PyTorch (default cu128; use cu126 for older drivers, cpu for CPU-only),
# installs requirements.txt, optionally flash-attn, and runs csf.env_check.
#
# Usage:  bash setup_env.sh [--cuda cu128] [--python python3.11] [--flash-attn]
set -euo pipefail
cd "$(dirname "$0")"

CUDA="cu128"
PYTHON=""
FLASH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --flash-attn) FLASH=1; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

if [[ -z "$PYTHON" ]]; then
  for c in python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then PYTHON="$c"; break; fi
  done
fi
echo "==> Using interpreter: $PYTHON ($($PYTHON --version))"

[[ -d .venv ]] || "$PYTHON" -m venv .venv
VPY=".venv/bin/python"
"$VPY" -m pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch ($CUDA)"
"$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"

echo "==> Installing requirements"
"$VPY" -m pip install -r requirements.txt

if [[ "$FLASH" == "1" ]]; then
  echo "==> Installing flash-attn (compiles if no wheel matches; can take a while)"
  "$VPY" -m pip install flash-attn --no-build-isolation || echo "flash-attn install failed - sdpa attention will be used"
fi

echo "==> Verifying environment"
"$VPY" -m csf.env_check || true
cat <<'EOF'

Next steps:
  1. source .venv/bin/activate
  2. huggingface-cli login     (accept the Llama-3.2-11B-Vision licence on huggingface.co first)
  3. bash scripts/run_full.sh  (or scripts/run_test.sh)
EOF
