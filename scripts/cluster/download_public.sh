#!/usr/bin/env bash
# Public benchmark archives -> $P/raw (login node, network only). Resumable (curl -C -).
set -uo pipefail
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
R="$P/raw"; mkdir -p "$R"; cd "$R"
HF=https://huggingface.co/datasets
get() { echo "$(date +%T) start $2"; curl -fL --retry 5 -C - -o "$2" "$1" && echo "$(date +%T) done  $2 $(du -h "$2" | cut -f1)" || echo "$(date +%T) FAIL $2"; }
get "$HF/JoeLeelyf/ViF-Bench/resolve/main/source_videos.zip" vifbench_source_videos.zip
get "$HF/l8cv/GenBuster-Bench/resolve/main/GenBuster-Bench.zip" GenBuster-Bench.zip
get "$HF/bitmind/GenVidBench/resolve/main/Pair1/ms.rar" genvidbench_ms.rar
get "$HF/bitmind/GenVidBench/resolve/main/Pair1/t2vz.rar" genvidbench_t2vz.rar
get "$HF/bitmind/GenVidBench/resolve/main/Pair1/vript.rar" genvidbench_vript.rar
mkdir -p fakeparts_manifests
for m in extrapolation__cosmos_predict2 faceswap__insightface inpainting__diffueraser inpainting__propainter \
         inpainting__rovi interpolation__framer it2v__opensora it2v__wan21 outpainting__akira real__tenk_youtube \
         style__anyv2v style__rave t2v__allegroai t2v__cogvideox t2v__hunyuanvideo t2v__ltx_video t2v__mochi1 \
         t2v__opensora t2v__sora_1080p t2v__sora_720p t2v__veo2 t2v__wan21; do
  get "$HF/hi-paris/FakeParts/resolve/main/manifests/$m.jsonl" "fakeparts_manifests/$m.jsonl"
done
echo DOWNLOADS_DONE
