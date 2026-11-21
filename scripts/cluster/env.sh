# Sourced by every script in scripts/cluster. Set the paths once, e.g. in ~/.bashrc:
#   export SAFER_ROOT=/path/to/work/safer        # work dir: raw/, data/, lists/, ext/, export/, logs/, .venv/
#   export SAFER_CODE=$SAFER_ROOT/code           # this repository (default)
#   export SAFER_DATA=/path/to/big/storage/safer # large outputs of the D3 and VAE runs (default: $SAFER_ROOT)
# Optional: HF_HOME (model cache), PIP_CACHE_DIR, SAFER_MODULES (environment modules to load).
: "${SAFER_ROOT:?set SAFER_ROOT, see docs/REPRODUCE.md}"
export SAFER_ROOT
export SAFER_CODE=${SAFER_CODE:-$SAFER_ROOT/code}
export SAFER_DATA=${SAFER_DATA:-$SAFER_ROOT}
export HF_HOME=${HF_HOME:-$SAFER_ROOT/hf}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SAFER_ROOT/pip_cache}
SAFER_MODULES=${SAFER_MODULES-"GCCcore/13.2.0 Python/3.11.5"}
load_modules() {
  if [ -n "$SAFER_MODULES" ] && command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    module load $SAFER_MODULES
  fi
}
P=$SAFER_ROOT
G=$SAFER_DATA
