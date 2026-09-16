"""
Adapter registry: every manipulation model in the spec, mapped to an environment and a worker.

Coverage is deliberately tiered, because these 32 models are not equally available. Tier 1 is
wired end to end against a public repo or pip package and is expected to render video. Tier 2 is
registered with the environment and options it *would* need, but has no working public release
we can drive unattended; its worker refuses loudly rather than emitting a silently-wrong video,
so a tier-2 model can never contaminate the dataset with placeholder content.

`python -m csf.generation.adapters` prints the table, including which environments each tier
needs and how many videos the spec assigns to it.

Adding a model later means: implement `workers/worker_<x>.py`, flip `implemented=True`, and
point it at an `EnvSpec`. Nothing else in the pipeline changes.
"""

from __future__ import annotations

from typing import Dict, List

from csf.generation import spec as S
from csf.generation.adapters.base import Adapter, AdapterError, NotImplementedAdapter, WorkerPool
from csf.generation.envs import EnvSpec, GitRepo, WeightFile

__all__ = ["ADAPTERS", "ENVS", "Adapter", "AdapterError", "NotImplementedAdapter", "WorkerPool",
           "env_specs", "adapter_for", "coverage"]

# --------------------------------------------------------------------------------------
# environments
# --------------------------------------------------------------------------------------

TORCH_CU121 = "torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"
TORCH_LEGACY = "torch==1.13.1 torchvision==0.14.1"
LEGACY_INDEX = "https://download.pytorch.org/whl/cu117"

ENVS: Dict[str, EnvSpec] = {
    # --- face swap: InsightFace / ONNX Runtime -----------------------------------------
    "insightface": EnvSpec(
        name="insightface",
        torch="",                                   # onnxruntime does the work; no torch needed
        requirements=("insightface==0.7.3", "onnxruntime-gpu==1.18.1", "opencv-python-headless",
                      "numpy<2", "imageio[ffmpeg]", "tqdm"),
        weights=(WeightFile(dest="weights/inswapper_128.onnx",
                            hf_repo="ezioruan/inswapper_128.onnx", hf_file="inswapper_128.onnx"),),
        note="INSwapper 128 + buffalo_l detection/recognition.",
    ),

    # --- background: SAM2 segmentation + diffusers generators ---------------------------
    "sam2_diffusers": EnvSpec(
        name="sam2_diffusers",
        torch=TORCH_CU121,
        requirements=("diffusers>=0.31", "transformers>=4.44", "accelerate", "safetensors",
                      "sentencepiece", "protobuf", "opencv-python-headless", "numpy<2",
                      "imageio[ffmpeg]", "pillow", "tqdm",
                      "git+https://github.com/facebookresearch/sam2.git"),
        weights=(WeightFile(dest="weights/sam2.1_hiera_base_plus.pt",
                            hf_repo="facebook/sam2.1-hiera-base-plus",
                            hf_file="sam2.1_hiera_base_plus.pt"),),
        note="SAM2 video segmentation, FLUX.1-schnell stills, Stable Video Diffusion clips.",
    ),

    # --- ProPainter: flow-guided video inpainting ---------------------------------------
    "propainter": EnvSpec(
        name="propainter",
        torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "pillow", "scipy", "imageio[ffmpeg]",
                      "av", "einops", "timm", "tqdm", "matplotlib", "scikit-image"),
        repos=(GitRepo("https://github.com/sczhou/ProPainter.git", name="ProPainter"),),
        weights=(
            WeightFile(dest="repos/ProPainter/weights/ProPainter.pth",
                       url="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth"),
            WeightFile(dest="repos/ProPainter/weights/raft-things.pth",
                       url="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth"),
            WeightFile(dest="repos/ProPainter/weights/recurrent_flow_completion.pth",
                       url="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth"),
        ),
    ),

    # --- E2FGVI / STTN / FuseFormer: transformer video inpainting -----------------------
    "videoinpaint": EnvSpec(
        name="videoinpaint",
        torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "pillow", "scipy", "imageio[ffmpeg]",
                      "av", "einops", "tqdm", "scikit-image"),
        repos=(GitRepo("https://github.com/MCG-NKU/E2FGVI.git", name="E2FGVI"),
               GitRepo("https://github.com/researchmm/STTN.git", name="STTN"),
               GitRepo("https://github.com/ruiliu-ai/FuseFormer.git", name="FuseFormer")),
        note="Checkpoints for these three are Google-Drive hosted upstream; stage them into "
             "repos/<name>/release_model/ before running (see the runbook).",
    ),

    # --- Wav2Lip ------------------------------------------------------------------------
    "wav2lip": EnvSpec(
        name="wav2lip",
        torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "librosa==0.10.2", "numba",
                      "imageio[ffmpeg]", "tqdm", "scipy"),
        repos=(GitRepo("https://github.com/Rudrabha/Wav2Lip.git", name="Wav2Lip"),),
        weights=(WeightFile(dest="repos/Wav2Lip/checkpoints/wav2lip_gan.pth",
                            hf_repo="camenduru/Wav2Lip", hf_file="checkpoints/wav2lip_gan.pth"),
                 WeightFile(dest="repos/Wav2Lip/face_detection/detection/sfd/s3fd.pth",
                            hf_repo="camenduru/Wav2Lip", hf_file="checkpoints/s3fd.pth")),
    ),

    # --- First Order Motion Model (reenactment) ------------------------------------------
    "fomm": EnvSpec(
        name="fomm",
        torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "scikit-image", "imageio[ffmpeg]",
                      "pyyaml", "tqdm", "scipy", "cffi", "matplotlib"),
        repos=(GitRepo("https://github.com/AliaksandrSiarohin/first-order-model.git", name="fomm"),),
        weights=(WeightFile(dest="weights/vox-adv-cpk.pth.tar",
                            hf_repo="Ubaidbhat/first_order_motion_model",
                            hf_file="vox-adv-cpk.pth.tar"),),
    ),

    # --- TokenFlow / InsV2V: diffusion video editing -------------------------------------
    "tokenflow": EnvSpec(
        name="tokenflow",
        torch=TORCH_CU121,
        requirements=("diffusers>=0.31", "transformers>=4.44", "accelerate", "safetensors",
                      "opencv-python-headless", "numpy<2", "imageio[ffmpeg]", "einops", "tqdm",
                      "av", "pillow"),
        repos=(GitRepo("https://github.com/omerbt/TokenFlow.git", name="TokenFlow"),),
    ),

    # --- tier-2 environments (declared, adapters not wired) -------------------------------
    "simswap": EnvSpec(
        name="simswap", torch=TORCH_LEGACY, torch_index=LEGACY_INDEX,
        requirements=("insightface==0.2.1", "onnxruntime-gpu", "opencv-python-headless", "numpy<2",
                      "imageio[ffmpeg]"),
        repos=(GitRepo("https://github.com/neuralchen/SimSwap.git", name="SimSwap"),),
        note="Weights (arcface + 512 checkpoint) are hosted on Google Drive / OneDrive upstream."),
    "sadtalker": EnvSpec(
        name="sadtalker", torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "librosa==0.10.2", "imageio[ffmpeg]",
                      "scipy", "yacs", "pydub", "kornia", "face-alignment", "tqdm"),
        repos=(GitRepo("https://github.com/OpenTalker/SadTalker.git", name="SadTalker"),
               GitRepo("https://github.com/OpenTalker/video-retalking.git", name="VideoReTalking"))),
    "musetalk": EnvSpec(
        name="musetalk", torch=TORCH_CU121,
        requirements=("diffusers>=0.30", "transformers>=4.44", "accelerate", "opencv-python-headless",
                      "numpy<2", "librosa==0.10.2", "imageio[ffmpeg]", "einops", "omegaconf"),
        repos=(GitRepo("https://github.com/TMElyralab/MuseTalk.git", name="MuseTalk"),)),
    "stylegan": EnvSpec(
        name="stylegan", torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "scipy", "ninja", "imageio[ffmpeg]",
                      "dlib", "tqdm"),
        repos=(GitRepo("https://github.com/williamyang1991/StyleGANEX.git", name="StyleGANEX"),)),
    "ganimation": EnvSpec(
        name="ganimation", torch=TORCH_LEGACY, torch_index=LEGACY_INDEX,
        requirements=("opencv-python-headless", "numpy<2", "scipy", "imageio[ffmpeg]"),
        repos=(GitRepo("https://github.com/albertpumarola/GANimation.git", name="GANimation"),)),
    "faceshifter": EnvSpec(
        name="faceshifter", torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "insightface==0.7.3", "onnxruntime-gpu",
                      "imageio[ffmpeg]"),
        repos=(GitRepo("https://github.com/mindslab-ai/faceshifter.git", name="faceshifter"),),
        note="Upstream ships training code only; no released inference checkpoint."),
    "pirenderer": EnvSpec(
        name="pirenderer", torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "scipy", "imageio[ffmpeg]", "lmdb",
                      "av", "tqdm"),
        repos=(GitRepo("https://github.com/RenYurui/PIRender.git", name="PIRender"),)),
    "anyv2v": EnvSpec(
        name="anyv2v", torch=TORCH_CU121,
        requirements=("diffusers>=0.31", "transformers>=4.44", "accelerate", "opencv-python-headless",
                      "numpy<2", "imageio[ffmpeg]", "einops"),
        repos=(GitRepo("https://github.com/TIGER-AI-Lab/AnyV2V.git", name="AnyV2V"),)),
    "videocomposer": EnvSpec(
        name="videocomposer", torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "imageio[ffmpeg]", "einops", "open-clip-torch"),
        repos=(GitRepo("https://github.com/ali-vilab/videocomposer.git", name="videocomposer"),)),
    "vid2vid": EnvSpec(
        name="vid2vid", torch=TORCH_LEGACY, torch_index=LEGACY_INDEX,
        requirements=("opencv-python-headless", "numpy<2", "dominate", "imageio[ffmpeg]"),
        repos=(GitRepo("https://github.com/NVIDIA/vid2vid.git", name="vid2vid"),)),
}


# --------------------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------------------

def _a(key: str, family: str, env: str, worker: str, *, tier: int = 1, implemented: bool = True,
       note: str = "", **options) -> Adapter:
    return Adapter(key=key, family=family, env_name=env, worker=worker, implemented=implemented,
                   tier=tier, options=options, note=note)


ADAPTERS: Dict[str, Adapter] = {a.key: a for a in [
    # ---- background manipulation (3,000) ----
    _a("bg_real_composite", "background_manipulation", "sam2_diffusers", "worker_background.py",
       mode="real_composite"),
    _a("bg_flux_image", "background_manipulation", "sam2_diffusers", "worker_background.py",
       mode="flux_image", flux_id="black-forest-labs/FLUX.1-schnell", steps=4),
    _a("bg_svd_video", "background_manipulation", "sam2_diffusers", "worker_background.py",
       mode="svd_video", svd_id="stabilityai/stable-video-diffusion-img2vid-xt", steps=25),
    _a("bg_propainter_recon", "background_manipulation", "propainter", "worker_propainter.py",
       mode="background_reconstruct"),

    # ---- face swap (6,250) ----
    _a("inswapper", "face_swap", "insightface", "worker_inswapper.py"),
    _a("simswap", "face_swap", "simswap", "worker_unimplemented.py", tier=2, implemented=False,
       note="Upstream weights are Google-Drive hosted; no unattended download path."),
    _a("faceshifter", "face_swap", "faceshifter", "worker_unimplemented.py", tier=2, implemented=False,
       note="Public repo ships training code only - no released inference checkpoint."),
    _a("face_transformer", "face_swap", "faceshifter", "worker_unimplemented.py", tier=2,
       implemented=False, note="No maintained public inference release."),

    # ---- facial reenactment (5,250) ----
    _a("fomm", "facial_reenactment", "fomm", "worker_fomm.py"),
    _a("face2face", "facial_reenactment", "pirenderer", "worker_unimplemented.py", tier=2,
       implemented=False, note="Original Face2Face is not publicly released."),
    _a("pirenderer", "facial_reenactment", "pirenderer", "worker_unimplemented.py", tier=2,
       implemented=False, note="Checkpoints are Google-Drive hosted upstream."),
    _a("face2face_rho", "facial_reenactment", "pirenderer", "worker_unimplemented.py", tier=2,
       implemented=False, note="No unattended weight download path."),

    # ---- lip-sync (5,250) ----
    _a("wav2lip", "lip_sync", "wav2lip", "worker_wav2lip.py"),
    _a("musetalk", "lip_sync", "musetalk", "worker_unimplemented.py", tier=2, implemented=False,
       note="Needs MuseTalk + whisper + dwpose weights staged manually."),
    _a("videoretalking", "lip_sync", "sadtalker", "worker_unimplemented.py", tier=2, implemented=False,
       note="Checkpoint bundle is Google-Drive hosted upstream."),
    _a("sadtalker", "lip_sync", "sadtalker", "worker_unimplemented.py", tier=2, implemented=False,
       note="Checkpoint bundle is Google-Drive hosted upstream."),

    # ---- expression / attribute editing (4,750) ----
    _a("ganimation", "expression_attribute_editing", "ganimation", "worker_unimplemented.py",
       tier=2, implemented=False, note="2018 repo, unmaintained; weights not published."),
    _a("styleganex", "expression_attribute_editing", "stylegan", "worker_unimplemented.py",
       tier=2, implemented=False, note="Needs StyleGANEX + pSp weights staged manually."),
    _a("latent_transformer", "expression_attribute_editing", "stylegan", "worker_unimplemented.py",
       tier=2, implemented=False),
    _a("vq_facial_editing", "expression_attribute_editing", "stylegan", "worker_unimplemented.py",
       tier=2, implemented=False, note="No public release."),

    # ---- object insertion / removal (3,500) ----
    _a("propainter_object", "object_insertion_removal", "propainter", "worker_propainter.py",
       mode="object"),
    _a("object_wiper", "object_insertion_removal", "videocomposer", "worker_unimplemented.py",
       tier=2, implemented=False, note="No public code release."),
    _a("anyv2v_object", "object_insertion_removal", "anyv2v", "worker_unimplemented.py",
       tier=2, implemented=False, note="Needs an I2V backbone + per-frame editor staged."),
    _a("videocomposer", "object_insertion_removal", "videocomposer", "worker_unimplemented.py",
       tier=2, implemented=False, note="Weights require a manual request upstream."),

    # ---- video inpainting (3,000) ----
    _a("propainter_inpaint", "video_inpainting", "propainter", "worker_propainter.py",
       mode="inpaint"),
    _a("e2fgvi_hq", "video_inpainting", "videoinpaint", "worker_videoinpaint.py", model="e2fgvi_hq"),
    _a("sttn", "video_inpainting", "videoinpaint", "worker_videoinpaint.py", model="sttn"),
    _a("fuseformer", "video_inpainting", "videoinpaint", "worker_videoinpaint.py", model="fuseformer"),

    # ---- video-to-video (2,333) ----
    _a("tokenflow", "video_to_video", "tokenflow", "worker_tokenflow.py",
       sd_id="stabilityai/stable-diffusion-2-1-base", steps=50),
    _a("insv2v", "video_to_video", "tokenflow", "worker_unimplemented.py", tier=2, implemented=False,
       note="InsV2V weights are Google-Drive hosted upstream."),
    _a("anyv2v_style", "video_to_video", "anyv2v", "worker_unimplemented.py", tier=2,
       implemented=False),
    _a("vid2vid", "video_to_video", "vid2vid", "worker_unimplemented.py", tier=2, implemented=False,
       note="NVIDIA vid2vid needs per-dataset training; no general pretrained release."),
]}


def adapter_for(model_key: str) -> Adapter:
    try:
        return ADAPTERS[model_key]
    except KeyError:
        raise AdapterError(f"No adapter registered for model {model_key!r}. "
                           f"Known: {sorted(ADAPTERS)}") from None


def env_specs() -> Dict[str, EnvSpec]:
    """Only the environments some registered adapter actually uses."""
    used = {a.env_name for a in ADAPTERS.values()}
    missing = used - set(ENVS)
    if missing:
        raise AdapterError(f"Adapters reference undefined environments: {sorted(missing)}")
    return {name: ENVS[name] for name in sorted(used)}


def coverage(targets: Dict[str, int] | None = None) -> Dict[str, object]:
    """How many of the planned videos are covered by tier-1 (wired) adapters."""
    targets = targets or S.FAMILY_TARGETS
    per_model: Dict[str, int] = {}
    for family in S.FAMILY_LIST:
        _, cols, _ = S.family_matrix(family, targets[family.key])
        for pipe, n in zip(family.pipelines, cols):
            per_model[pipe.key] = n
    wired = {k: v for k, v in per_model.items() if ADAPTERS[k].implemented}
    pending = {k: v for k, v in per_model.items() if not ADAPTERS[k].implemented}
    total = sum(per_model.values())
    return {"total_videos": total, "videos_wired": sum(wired.values()),
            "videos_pending": sum(pending.values()),
            "fraction_wired": round(sum(wired.values()) / total, 4) if total else 0.0,
            "wired_models": dict(sorted(wired.items(), key=lambda kv: -kv[1])),
            "pending_models": dict(sorted(pending.items(), key=lambda kv: -kv[1]))}


def _main() -> None:
    cov = coverage()
    print(f"{'model':<24}{'family':<32}{'env':<16}{'tier':>5}{'videos':>9}  worker")
    for family in S.FAMILY_LIST:
        _, cols, _ = S.family_matrix(family, S.FAMILY_TARGETS[family.key])
        for pipe, n in zip(family.pipelines, cols):
            a = ADAPTERS[pipe.key]
            mark = " " if a.implemented else "*"
            print(f"{mark}{a.key:<23}{family.key:<32}{a.env_name:<16}{a.tier:>5}{n:>9}  {a.worker}")
    print(f"\n* = registered but not wired to a runnable public release")
    print(f"tier-1 coverage: {cov['videos_wired']:,}/{cov['total_videos']:,} videos "
          f"({cov['fraction_wired']:.1%}) across "
          f"{sum(1 for a in ADAPTERS.values() if a.implemented)} of {len(ADAPTERS)} models")
    print(f"environments: {len(env_specs())}")


if __name__ == "__main__":
    _main()
