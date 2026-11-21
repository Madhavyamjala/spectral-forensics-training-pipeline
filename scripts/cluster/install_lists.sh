#!/usr/bin/env bash
# The exact video selection of the paper's external rerun, kept in reproduce/lists/ with relative paths.
#   bash install_lists.sh install   before build_datasets.py fakeparts: copies fakeparts_picks.json (so the same
#                                   FakeParts subset is downloaded even if the upstream manifests change) and writes
#                                   lists/vae_clips.txt from $CHRONO_RAW (the folder holding chrono_triclass/).
#   bash install_lists.sh check     after job_prepare.slurm: compares the regenerated lists/<bench>.csv with
#                                   reproduce/lists/<bench>.csv (build_datasets.py csv is deterministic: sorted, seed 0).
set -euo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
L=$SAFER_CODE/reproduce/lists
case "${1:-}" in
  install)
    mkdir -p "$P/lists"
    cp "$L/fakeparts_picks.json" "$P/fakeparts_picks.json"
    if [ -n "${CHRONO_RAW:-}" ]; then
      sed "s|^|${CHRONO_RAW%/}/|" "$L/vae_clips.txt" > "$P/lists/vae_clips.txt"
      echo "wrote lists/vae_clips.txt ($(wc -l < "$P/lists/vae_clips.txt") clips)"
    else
      echo "CHRONO_RAW not set: skipped lists/vae_clips.txt (needed only for the VAE test)"
    fi ;;
  check)
    ok=1
    for b in genbuster fakeparts genvidbench; do
      if awk -v r="$P/" 'NR == 1 { print; next } { if (index($0, r) == 1) $0 = substr($0, length(r) + 1); print }' "$P/lists/$b.csv" | cmp -s - "$L/$b.csv"; then
        echo "same as the paper run: $b"
      else
        echo "DIFFERS from the paper run: $b"; ok=0
      fi
    done
    [ $ok = 1 ] ;;
  *) echo "usage: bash install_lists.sh install|check"; exit 2 ;;
esac
