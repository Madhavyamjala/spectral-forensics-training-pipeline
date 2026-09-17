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
           "env_specs", "adapter_for", "coverage", "cost_estimate", "videos_per_model"]

# --------------------------------------------------------------------------------------
# environments
# --------------------------------------------------------------------------------------

TORCH_CU121 = "torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"
#: SAM2 declares torch>=2.5.1 / torchvision>=0.20.1. Installing it against TORCH_CU121 makes pip
#: pull the newest torch from PyPI on top of the pinned build - the env then flip-flops between
#: the two on every rebuild. Meet upstream's floor instead of fighting it.
TORCH_SAM2 = "torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
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
        torch=TORCH_SAM2,
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
        weights=(
            WeightFile(dest="repos/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth",
                       staged_name="e2fgvi_hq.pth",
                       where="https://github.com/MCG-NKU/E2FGVI (Google Drive / OneDrive link "
                             "in the README)"),
            WeightFile(dest="repos/STTN/checkpoints/sttn.pth", staged_name="sttn.pth",
                       where="https://github.com/researchmm/STTN (Drive link in the README)"),
            WeightFile(dest="repos/FuseFormer/checkpoints/fuseformer.pth",
                       staged_name="fuseformer.pth",
                       where="https://github.com/ruiliu-ai/FuseFormer (Drive link in README)"),
        ),
        note="All three checkpoints are Drive-hosted upstream, so they are staged by hand into "
             "generation.staged_weights_dir and copied into place by the build.",
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

    # --- LatentSync 1.6: audio-conditioned latent diffusion lip-sync ---------------------
    "latentsync": EnvSpec(
        name="latentsync",
        torch=TORCH_CU121,
        requirements=("diffusers>=0.32", "transformers>=4.44", "accelerate", "safetensors",
                      "opencv-python-headless", "numpy<2", "imageio[ffmpeg]", "einops",
                      "omegaconf", "librosa==0.10.2", "face-alignment", "python-speech-features",
                      "decord", "mediapipe", "tqdm"),
        repos=(GitRepo("https://github.com/bytedance/LatentSync.git", name="LatentSync"),),
        weights=(WeightFile(dest="repos/LatentSync/checkpoints/latentsync_unet.pt",
                            hf_repo="ByteDance/LatentSync-1.6", hf_file="latentsync_unet.pt"),
                 WeightFile(dest="repos/LatentSync/checkpoints/whisper/tiny.pt",
                            hf_repo="ByteDance/LatentSync-1.6", hf_file="whisper/tiny.pt")),
        note="Apache-2.0. Stands in for VideoReTalking, which it beats on every reported metric.",
    ),

    # --- MuseTalk 1.5: latent-space audio-conditioned inpainting -------------------------
    "musetalk": EnvSpec(
        name="musetalk",
        torch=TORCH_CU121,
        requirements=("diffusers>=0.30", "transformers>=4.44", "accelerate",
                      "opencv-python-headless", "numpy<2", "librosa==0.10.2",
                      "imageio[ffmpeg]", "einops", "omegaconf", "soundfile", "tqdm"),
        repos=(GitRepo("https://github.com/TMElyralab/MuseTalk.git", name="MuseTalk"),),
        weights=(
            WeightFile(dest="repos/MuseTalk/models/musetalkV15/unet.pth",
                       hf_repo="TMElyralab/MuseTalk", hf_file="musetalkV15/unet.pth"),
            WeightFile(dest="repos/MuseTalk/models/musetalkV15/musetalk.json",
                       hf_repo="TMElyralab/MuseTalk", hf_file="musetalkV15/musetalk.json"),
            WeightFile(dest="repos/MuseTalk/models/sd-vae/diffusion_pytorch_model.bin",
                       hf_repo="stabilityai/sd-vae-ft-mse",
                       hf_file="diffusion_pytorch_model.bin"),
            WeightFile(dest="repos/MuseTalk/models/sd-vae/config.json",
                       hf_repo="stabilityai/sd-vae-ft-mse", hf_file="config.json"),
            WeightFile(dest="repos/MuseTalk/models/whisper/tiny.pt",
                       hf_repo="openai/whisper-tiny", hf_file="pytorch_model.bin"),
            WeightFile(dest="repos/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth",
                       hf_repo="yzd-v/DWPose", hf_file="dw-ll_ucoco_384.pth"),
        ),
        note="MIT, commercial use allowed. face-parse-bisent is Drive-hosted upstream; the "
             "worker degrades to MuseTalk's own bbox path when it is absent.",
    ),

    # --- SadTalker: audio -> 3DMM coefficients -> neural rendering -----------------------
    "sadtalker": EnvSpec(
        name="sadtalker",
        torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "librosa==0.10.2", "imageio[ffmpeg]",
                      "scipy", "yacs", "pydub", "kornia", "face-alignment", "safetensors",
                      "basicsr", "facexlib", "gfpgan", "tqdm"),
        repos=(GitRepo("https://github.com/OpenTalker/SadTalker.git", name="SadTalker"),),
        post_install=(("-c", "import subprocess,os;subprocess.run(['bash','scripts/download_models.sh'],"
                             "cwd=os.path.join(os.environ['CSF_ENV_ROOT'],'repos','SadTalker'),check=True)"),),
        hub_repos=(),   # its downloader pulls from GitHub Releases, not the Hub
        note="Apache-2.0 (non-commercial restriction was removed upstream). Its own "
             "download_models.sh pulls every checkpoint from GitHub Releases.",
    ),

    # --- LivePortrait: implicit-keypoint animation + stitching/retargeting ---------------
    "liveportrait": EnvSpec(
        name="liveportrait",
        torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "imageio[ffmpeg]", "scipy", "tyro",
                      "onnxruntime-gpu==1.18.1", "rich", "pyyaml", "albumentations", "tqdm"),
        repos=(GitRepo("https://github.com/KwaiVGI/LivePortrait.git", name="LivePortrait"),),
        post_install=(("-c", "import subprocess,os;r=os.path.join(os.environ['CSF_ENV_ROOT'],'repos','LivePortrait');"
                             "subprocess.run(['huggingface-cli','download','KlingTeam/LivePortrait',"
                             "'--local-dir','pretrained_weights'],cwd=r,check=True)"),),
        hub_repos=("KlingTeam/LivePortrait",),
        note="Drives both reenactment (v2v) and expression editing (retargeting ratios).",
    ),

    # --- VACE (Wan2.1): all-in-one masked video editing ---------------------------------
    "vace": EnvSpec(
        name="vace",
        torch=TORCH_CU121,
        requirements=("diffusers>=0.31", "transformers>=4.49", "accelerate", "safetensors",
                      "opencv-python-headless", "numpy<2", "imageio[ffmpeg]", "einops",
                      "easydict", "ftfy", "regex", "omegaconf", "decord", "tqdm"),
        repos=(GitRepo("https://github.com/ali-vilab/VACE.git", name="VACE"),),
        post_install=(("-c", "import subprocess,os;subprocess.run(['huggingface-cli','download',"
                             "'Wan-AI/Wan2.1-VACE-1.3B','--local-dir',"
                             "os.path.join(os.environ['CSF_ENV_ROOT'],'weights','Wan2.1-VACE-1.3B')],check=True)"),),
        hub_repos=("Wan-AI/Wan2.1-VACE-1.3B",),
        note="Apache-2.0, ICCV 2025. One model covers masked object insertion/removal and "
             "prompt-driven V2V, so it fills several slots the document's models cannot.",
    ),

    # --- DiffuEraser: diffusion video object removal --------------------------------------
    "diffueraser": EnvSpec(
        name="diffueraser",
        torch=TORCH_CU121,
        requirements=("diffusers>=0.31", "transformers>=4.44", "accelerate", "safetensors",
                      "opencv-python-headless", "numpy<2", "imageio[ffmpeg]", "einops", "av",
                      "scipy", "tqdm"),
        repos=(GitRepo("https://github.com/lixiaowen-xw/DiffuEraser.git", name="DiffuEraser"),),
        post_install=(("-c", "import subprocess,os;subprocess.run(['huggingface-cli','download',"
                             "'lixiaowen/diffuEraser','--local-dir',"
                             "os.path.join(os.environ['CSF_ENV_ROOT'],'repos','DiffuEraser','weights','diffuEraser')],check=True)"),),
        hub_repos=("lixiaowen/diffuEraser",),
        note="Apache-2.0. Diffusion removal - a different artifact class from ProPainter's "
             "flow propagation, which is why it is worth a slot of its own.",
    ),

    # --- DreamID-V: DiT video face swapping ----------------------------------------------
    "dreamid": EnvSpec(
        name="dreamid",
        torch=TORCH_CU121,
        requirements=("diffusers>=0.31", "transformers>=4.49", "accelerate", "safetensors",
                      "opencv-python-headless", "numpy<2", "imageio[ffmpeg]", "einops",
                      "insightface==0.7.3", "onnxruntime-gpu==1.18.1", "easydict", "ftfy", "tqdm"),
        repos=(GitRepo("https://github.com/bytedance/DreamID-V.git", name="DreamID-V"),),
        post_install=(("-c", "import subprocess,os;subprocess.run(['huggingface-cli','download',"
                             "'XuGuo699/DreamID-V','--local-dir',"
                             "os.path.join(os.environ['CSF_ENV_ROOT'],'weights','DreamID-V')],check=True)"),),
        hub_repos=("XuGuo699/DreamID-V",),
        note="Apache-2.0, Wan2.1-1.3B DiT. Reported 99.9% ID retrieval vs SimSwap's 95.24%, but "
             "it is a diffusion transformer, so roughly 20x SimSwap's cost per video.",
    ),

    # --- REFace: diffusion face swapping --------------------------------------------------
    "reface": EnvSpec(
        name="reface",
        torch=TORCH_CU121,
        requirements=("diffusers>=0.31", "transformers>=4.44", "accelerate", "safetensors",
                      "opencv-python-headless", "numpy<2", "imageio[ffmpeg]", "einops",
                      "omegaconf", "pytorch-lightning", "kornia", "insightface==0.7.3",
                      "onnxruntime-gpu==1.18.1", "tqdm"),
        repos=(GitRepo("https://github.com/Sanoojan/REFace.git", name="REFace"),),
        weights=(WeightFile(dest="repos/REFace/checkpoints/last.ckpt",
                            hf_repo="Sanoojan/REFace", hf_file="last.ckpt"),),
        note="LICENCE WARNING: MIT code, but trained on CelebAMask-HQ, which restricts use to "
             "NON-COMMERCIAL RESEARCH. Skip this adapter if the dataset may ever ship "
             "commercially: --set generation.skip_models='[faceshifter]'",
    ),

    # --- Thin-Plate-Spline Motion Model: reenactment ---------------------------------------
    "tpsmm": EnvSpec(
        name="tpsmm",
        torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "scikit-image", "imageio[ffmpeg]",
                      "pyyaml", "scipy", "matplotlib", "tqdm"),
        repos=(GitRepo("https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git",
                       name="TPSMM"),),
        weights=(WeightFile(dest="repos/TPSMM/checkpoints/vox.pth.tar",
                            staged_name="vox.pth.tar",
                            where="https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model "
                                  "(Tsinghua Cloud / Google Drive link in the README)"),),
        note="MIT. The vox checkpoint is Tsinghua-Cloud/Drive hosted, so it is staged by hand.",
    ),

    # --- tier-2 environments (declared, adapters not wired) -------------------------------
    "simswap": EnvSpec(
        name="simswap", torch=TORCH_LEGACY, torch_index=LEGACY_INDEX,
        requirements=("insightface==0.2.1", "onnxruntime-gpu", "opencv-python-headless", "numpy<2",
                      "imageio[ffmpeg]"),
        repos=(GitRepo("https://github.com/neuralchen/SimSwap.git", name="SimSwap"),),
        note="Weights (arcface + 512 checkpoint) are hosted on Google Drive / OneDrive upstream."),
    "stylegan": EnvSpec(
        name="stylegan", torch=TORCH_CU121,
        requirements=("opencv-python-headless", "numpy<2", "scipy", "ninja", "imageio[ffmpeg]",
                      "dlib", "tqdm"),
        repos=(GitRepo("https://github.com/williamyang1991/StyleGANEX.git", name="StyleGANEX"),),
        weights=(
            # Upstream releases one checkpoint PER EDITING DIRECTION, not one "editing" model:
            # styleganex_edit_age.pt and styleganex_edit_hair.pt are the only two video-editing
            # directions published, which is what the worker may claim to produce.
            WeightFile(dest="repos/StyleGANEX/pretrained_models/styleganex_edit_age.pt",
                       staged_name="styleganex_edit_age.pt",
                       where="https://github.com/williamyang1991/StyleGANEX (Drive link in the "
                             "README, 'Video editing' section)"),
            WeightFile(dest="repos/StyleGANEX/pretrained_models/styleganex_edit_hair.pt",
                       staged_name="styleganex_edit_hair.pt",
                       where="https://github.com/williamyang1991/StyleGANEX (Drive link in the "
                             "README, 'Video editing' section)"),
        ),
        # video_editing.py needs dlib's 68-point landmark predictor and fetches it with the
        # `wget` package if absent. Place it during the build instead, so a job never blocks on
        # a download - it is a plain URL, just bz2-compressed.
        post_install=(("-c",
                       "import bz2,os,urllib.request,pathlib;"
                       "d=pathlib.Path(os.environ['CSF_ENV_ROOT'])/'repos'/'StyleGANEX'/"
                       "'pretrained_models';d.mkdir(parents=True,exist_ok=True);"
                       "t=d/'shape_predictor_68_face_landmarks.dat';"
                       "raw=urllib.request.urlopen('http://dlib.net/files/"
                       "shape_predictor_68_face_landmarks.dat.bz2',timeout=300).read() "
                       "if not t.exists() else b'';"
                       "t.write_bytes(bz2.decompress(raw)) if raw else None"),),
        note="Drive-hosted; staged by hand. Upstream publishes ONE checkpoint PER DIRECTION, "
             "and the only video-editing directions released are age and hair colour - so this "
             "renderer can honestly produce those two variants and no others."),
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
       actual_model: str = "", cost_s: float = 60.0, vram_gb: float = 8.0, note: str = "",
       variants: tuple = (), **options) -> Adapter:
    """Construct an adapter declaration with concise registry syntax."""
    return Adapter(key=key, family=family, env_name=env, worker=worker, implemented=implemented,
                   tier=tier, actual_model=actual_model, cost_s=cost_s, vram_gb=vram_gb,
                   options=options, variants=tuple(variants), note=note)


#: Slots the specification names but that have no runnable public release. Each is filled by a
#: substitute recorded in the manifest under its own name (see `Adapter.actual_model`), or left
#: unwired where every candidate would merely duplicate a mechanism already present - a second
#: slot rendered by an identical model adds videos but no new artifact class, and the per-method
#: breakdown would be reporting one fingerprint under two names.
ADAPTERS: Dict[str, Adapter] = {a.key: a for a in [
    # ---- background manipulation (3,000) ----
    _a("bg_real_composite", "background_manipulation", "sam2_diffusers", "worker_background.py",
       cost_s=25.0, mode="real_composite", vram_gb=6.0),
    _a("bg_flux_image", "background_manipulation", "sam2_diffusers", "worker_background.py",
       cost_s=45.0, mode="flux_image", flux_id="black-forest-labs/FLUX.1-schnell", steps=4, vram_gb=26.0),
    _a("bg_svd_video", "background_manipulation", "sam2_diffusers", "worker_background.py",
       cost_s=120.0, mode="svd_video", svd_id="stabilityai/stable-video-diffusion-img2vid-xt",
       steps=25, vram_gb=24.0),
    _a("bg_propainter_recon", "background_manipulation", "propainter", "worker_propainter.py",
       cost_s=70.0, mode="background_reconstruct", vram_gb=12.0),

    # ---- face swap (6,250) ----
    _a("inswapper", "face_swap", "insightface", "worker_inswapper.py", cost_s=8.0, vram_gb=3.0),
    _a("simswap", "face_swap", "dreamid", "worker_dreamid.py", actual_model="dreamid_v",
       cost_s=150.0,
       note="SimSwap's weights are Drive-only. DreamID-V stands in: same identity-injection "
            "role, DiT instead of GAN, 99.9% vs 95.24% ID retrieval.", vram_gb=22.0),
    _a("faceshifter", "face_swap", "reface", "worker_reface.py", actual_model="reface",
       cost_s=20.0,
       note="FaceShifter released no inference checkpoint. REFace stands in (diffusion "
            "swap). NON-COMMERCIAL training data - see the env note.", vram_gb=12.0),
    _a("face_transformer", "face_swap", "reface", "worker_unimplemented.py", tier=2,
       implemented=False,
       note="No maintained public release, and every candidate substitute duplicates a "
            "mechanism already covered by inswapper/DreamID-V/REFace.", vram_gb=12.0),

    # ---- facial reenactment (5,250) ----
    _a("fomm", "facial_reenactment", "fomm", "worker_fomm.py", cost_s=25.0, vram_gb=5.0),
    _a("face2face", "facial_reenactment", "liveportrait", "worker_liveportrait.py",
       actual_model="liveportrait", cost_s=20.0, mode="reenact",
       note="Face2Face was never publicly released. LivePortrait stands in: implicit-keypoint "
            "animation with stitching/retargeting, distinct from FOMM's local affine warping.", vram_gb=6.0),
    _a("pirenderer", "facial_reenactment", "tpsmm", "worker_tpsmm.py", actual_model="tpsmm",
       cost_s=20.0,
       note="PIRenderer's checkpoints are Drive-hosted. TPSMM stands in - thin-plate-spline "
            "warping, a different basis from FOMM. Its checkpoint also needs staging.", vram_gb=5.0),
    _a("face2face_rho", "facial_reenactment", "tpsmm", "worker_unimplemented.py", tier=2,
       implemented=False,
       note="No unattended weight path, and the available substitutes are already used by the "
            "face2face and pirenderer slots.", vram_gb=5.0),

    # ---- lip-sync (5,250) - fully covered ----
    _a("wav2lip", "lip_sync", "wav2lip", "worker_wav2lip.py", cost_s=30.0, vram_gb=5.0),
    _a("musetalk", "lip_sync", "musetalk", "worker_musetalk.py", cost_s=25.0,
       note="Genuinely available: download script pulls every component from the Hub.", vram_gb=11.0),
    _a("videoretalking", "lip_sync", "latentsync", "worker_latentsync.py",
       actual_model="latentsync", cost_s=60.0,
       note="VideoReTalking's bundle is Drive-hosted. LatentSync 1.6 stands in and beats it on "
            "every reported metric (HDTF FID 7.03 vs 9.5, SyncConf 8.9 vs 7.5, FVD 193 vs 271).", vram_gb=15.0),
    _a("sadtalker", "lip_sync", "sadtalker", "worker_sadtalker.py", cost_s=90.0,
       note="Genuinely available: Apache-2.0, download_models.sh pulls from GitHub Releases.", vram_gb=9.0),

    # ---- expression / attribute editing (4,750) ----
    _a("ganimation", "expression_attribute_editing", "liveportrait", "worker_liveportrait.py",
       actual_model="liveportrait_expr", cost_s=20.0, mode="expression",
       # LivePortrait deforms an existing face: it drives expression, gaze and mouth shape. It
       # has no notion of age or hair colour, so those two variants must not be routed here -
       # it would emit a video labelled "age" that contains no ageing at all.
       variants=("smile_happiness", "sadness_crying", "anger", "surprise",
                 "eye_gaze_modification", "mouth_expression_modification"),
       note="GANimation published no weights. LivePortrait's retargeting ratios give the same "
            "continuous expression-magnitude control the slot calls for.", vram_gb=6.0),
    _a("styleganex", "expression_attribute_editing", "stylegan", "worker_styleganex.py",
       cost_s=35.0,
       # The opposite half: each released checkpoint carries one editing direction, and only
       # age and hair colour are published for video.
       variants=("age", "hair_color"),
       note="Weights are Drive-hosted and must be staged manually (see the env note), but it is "
            "the family's only second mechanism, so it is worth the manual step.", vram_gb=10.0),
    _a("latent_transformer", "expression_attribute_editing", "stylegan", "worker_unimplemented.py",
       tier=2, implemented=False, note="No unattended weight path.", vram_gb=10.0),
    _a("vq_facial_editing", "expression_attribute_editing", "stylegan", "worker_unimplemented.py",
       tier=2, implemented=False, note="Never publicly released.", vram_gb=10.0),

    # ---- object insertion / removal (3,500) - fully covered ----
    _a("propainter_object", "object_insertion_removal", "propainter", "worker_propainter.py",
       cost_s=70.0, mode="object", vram_gb=12.0),
    _a("object_wiper", "object_insertion_removal", "diffueraser", "worker_diffueraser.py",
       actual_model="diffueraser", cost_s=120.0,
       note="Object-WIPER has no public code. DiffuEraser stands in: diffusion removal, a "
            "different artifact class from ProPainter's flow propagation.", vram_gb=20.0),
    _a("anyv2v_object", "object_insertion_removal", "vace", "worker_vace.py",
       actual_model="vace", cost_s=180.0, task="inpainting",
       note="AnyV2V needs an I2V backbone staged. VACE stands in for mask-guided insertion.", vram_gb=22.0),
    _a("videocomposer", "object_insertion_removal", "vace", "worker_vace.py",
       actual_model="vace", cost_s=180.0, task="inpainting",
       note="VideoComposer's weights need a manual request. VACE's reference-guided masked "
            "editing covers the same conditioning.", vram_gb=22.0),

    # ---- video inpainting (3,000) - fully covered ----
    _a("propainter_inpaint", "video_inpainting", "propainter", "worker_propainter.py",
       cost_s=70.0, mode="inpaint", vram_gb=12.0),
    _a("e2fgvi_hq", "video_inpainting", "videoinpaint", "worker_videoinpaint.py", cost_s=45.0,
       model="e2fgvi_hq", vram_gb=9.0),
    _a("sttn", "video_inpainting", "videoinpaint", "worker_videoinpaint.py", cost_s=35.0,
       model="sttn", vram_gb=7.0),
    _a("fuseformer", "video_inpainting", "videoinpaint", "worker_videoinpaint.py", cost_s=40.0,
       model="fuseformer", vram_gb=8.0),

    # ---- video-to-video (2,333) ----
    _a("tokenflow", "video_to_video", "tokenflow", "worker_tokenflow.py", cost_s=240.0,
       sd_id="stabilityai/stable-diffusion-2-1-base", steps=50, vram_gb=16.0),
    _a("insv2v", "video_to_video", "vace", "worker_vace.py", actual_model="vace", cost_s=180.0,
       task="depth",
       note="InsV2V's weights are Drive-hosted. VACE stands in for prompt-driven whole-frame "
            "transformation.", vram_gb=22.0),
    _a("anyv2v_style", "video_to_video", "vace", "worker_vace.py", actual_model="vace",
       cost_s=180.0, task="depth",
       note="Same substitution as the object slot; recorded as vace in the manifest.", vram_gb=22.0),
    _a("vid2vid", "video_to_video", "vid2vid", "worker_unimplemented.py", tier=2,
       implemented=False,
       note="NVIDIA vid2vid needs per-dataset training and has no general pretrained release. "
            "Left as an honest gap: every substitute would be another diffusion model, and the "
            "slot exists precisely to contribute a non-diffusion GAN artifact class.", vram_gb=12.0),
]}


def adapter_for(model_key: str) -> Adapter:
    """Return the adapter registered for a specification model."""
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


def videos_per_model(targets: Dict[str, int] | None = None) -> Dict[str, int]:
    """Planned video count per specification slot."""
    targets = targets or S.FAMILY_TARGETS
    out: Dict[str, int] = {}
    for family in S.FAMILY_LIST:
        _, cols, _ = S.family_matrix(family, targets[family.key])
        for pipe, n in zip(family.pipelines, cols):
            out[pipe.key] = n
    return out


def coverage(targets: Dict[str, int] | None = None) -> Dict[str, object]:
    """How many of the planned videos are covered by wired adapters."""
    per_model = videos_per_model(targets)
    wired = {k: v for k, v in per_model.items() if ADAPTERS[k].implemented}
    pending = {k: v for k, v in per_model.items() if not ADAPTERS[k].implemented}
    substituted = {k: v for k, v in wired.items() if ADAPTERS[k].substituted}
    total = sum(per_model.values())

    # what the manifest will actually report, grouped by the model that renders
    by_actual: Dict[str, int] = {}
    for key, n in wired.items():
        by_actual[ADAPTERS[key].runs] = by_actual.get(ADAPTERS[key].runs, 0) + n

    return {"total_videos": total, "videos_wired": sum(wired.values()),
            "videos_pending": sum(pending.values()),
            "videos_substituted": sum(substituted.values()),
            "fraction_wired": round(sum(wired.values()) / total, 4) if total else 0.0,
            "distinct_actual_models": len(by_actual),
            "videos_by_actual_model": dict(sorted(by_actual.items(), key=lambda kv: -kv[1])),
            "wired_models": dict(sorted(wired.items(), key=lambda kv: -kv[1])),
            "pending_models": dict(sorted(pending.items(), key=lambda kv: -kv[1]))}


def cost_estimate(targets: Dict[str, int] | None = None, gpus: int = 3) -> Dict[str, object]:
    """Estimated GPU-hours for the wired adapters, and wall clock on `gpus` GPUs."""
    per_model = videos_per_model(targets)
    per: Dict[str, float] = {}
    for key, n in per_model.items():
        a = ADAPTERS[key]
        if a.implemented:
            per[key] = n * a.cost_s / 3600.0
    total = sum(per.values())
    return {"gpu_hours_total": round(total, 1),
            "wall_clock_hours": round(total / max(1, gpus), 1),
            "per_model_gpu_hours": dict(sorted(per.items(), key=lambda kv: -kv[1]))}


def _main() -> None:
    """Print the adapter coverage and estimated generation cost report."""
    cov = coverage()
    per_model = videos_per_model()
    print(f"{'slot':<22}{'runs as':<18}{'family':<30}{'env':<15}{'videos':>8}{'GPU-h':>8}")
    print("-" * 101)
    for family in S.FAMILY_LIST:
        for pipe in family.pipelines:
            a = ADAPTERS[pipe.key]
            n = per_model[pipe.key]
            mark = " " if a.implemented else "*"
            runs = "-" if not a.implemented else (a.runs if a.substituted else "(itself)")
            hours = n * a.cost_s / 3600.0 if a.implemented else 0.0
            print(f"{mark}{a.key:<21}{runs:<18}{family.key:<30}{a.env_name:<15}"
                  f"{n:>8}{hours:>8.1f}")
    print("-" * 101)
    print("* = no runnable public release and no non-duplicating substitute\n")
    print(f"coverage      : {cov['videos_wired']:,}/{cov['total_videos']:,} videos "
          f"({cov['fraction_wired']:.1%}) | {cov['videos_substituted']:,} via substitution")
    print(f"actual models : {cov['distinct_actual_models']} distinct renderers")
    for name, n in cov["videos_by_actual_model"].items():
        print(f"                  {name:<22}{n:>8}")
    cost = cost_estimate(gpus=3)
    print(f"\nestimated cost: {cost['gpu_hours_total']:,} GPU-hours "
          f"({cost['wall_clock_hours']:,} h wall clock on 3 GPUs)")
    for name, h in list(cost["per_model_gpu_hours"].items())[:6]:
        print(f"                  {name:<22}{h:>8.1f} h")
    print(f"\nenvironments  : {len(env_specs())}")


if __name__ == "__main__":
    _main()
