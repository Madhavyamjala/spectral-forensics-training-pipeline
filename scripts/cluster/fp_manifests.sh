#!/usr/bin/env bash
source "${SAFER_CODE:?set SAFER_CODE, see docs/REPRODUCE.md}/scripts/cluster/env.sh"
R=$SAFER_ROOT/raw; mkdir -p $R/fakeparts_manifests; cd $R
for m in extrapolation__cosmos_predict2 faceswap__insightface inpainting__diffueraser inpainting__propainter inpainting__rovi interpolation__framer it2v__opensora it2v__wan21 outpainting__akira real__tenk_youtube style__anyv2v style__rave t2v__allegroai t2v__cogvideox t2v__hunyuanvideo t2v__ltx_video t2v__mochi1 t2v__opensora t2v__sora_1080p t2v__sora_720p t2v__veo2 t2v__wan21; do
  curl -fsSL --retry 5 -o fakeparts_manifests/$m.jsonl https://huggingface.co/datasets/hi-paris/FakeParts/resolve/main/manifests/$m.jsonl || echo "FAIL $m"
done
cd "$SAFER_CODE" && load_modules; python3 build_datasets.py fakeparts-dry
