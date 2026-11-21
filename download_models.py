import os, sys
from huggingface_hub import snapshot_download
jobs = [("Qwen/Qwen2.5-VL-3B-Instruct", None, None),
        ("meta-llama/Llama-3.2-11B-Vision-Instruct", None, ["original/*"]),
        ("stabilityai/stable-video-diffusion-img2vid-xt", ["vae/*", "model_index.json"], None),
        ("stabilityai/sd-vae-ft-mse", None, None),
        # rebuttal analyses: D3 encoder and the two video VAEs of the VAE round-trip test
        ("microsoft/xclip-base-patch16", None, None),
        ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", ["vae/*", "model_index.json"], None),
        ("zai-org/CogVideoX-2b", ["vae/*", "model_index.json"], None)]
ok = True
for repo, allow, ignore in jobs:
    try:
        snapshot_download(repo, allow_patterns=allow, ignore_patterns=ignore)
        print("done", repo, flush=True)
    except Exception as e:
        ok = False
        print("FAILED", repo, type(e).__name__, str(e)[:300], flush=True)
print("MODELS_DONE" if ok else "MODELS_INCOMPLETE", flush=True)
