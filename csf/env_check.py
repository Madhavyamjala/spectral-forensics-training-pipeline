"""
Environment verification, run by the setup scripts (python -m csf.env_check).

Checks: Python / torch / CUDA build, GPU name, VRAM, compute capability and bf16 support, a real CUDA
matmul, bitsandbytes 4-bit quantisation on the GPU, transformers / peft / diffusers imports, OpenCV video
decoding support, torch.distributed backend availability, and Hugging Face authentication (needed for the
gated Llama-3.2-Vision weights).

Input : none.
Output: human-readable report on stdout; exit code 1 if a hard requirement fails.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    ok = True

    def report(name, status, detail=""):
        nonlocal ok
        mark = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}[status]
        ok &= status != "fail"
        print(f"{mark} {name:<28} {detail}")

    report("python", "ok" if sys.version_info >= (3, 10) else "fail", sys.version.split()[0])
    try:
        import torch
        report("torch", "ok", f"{torch.__version__} (CUDA build {torch.version.cuda})")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            dtype = "bfloat16" if (p.major >= 8 and torch.cuda.is_bf16_supported()) else "float16 + GradScaler"
            report("gpu", "ok", f"{torch.cuda.device_count()}x {p.name} | {p.total_memory / 2**30:.1f} GiB | "
                                f"sm_{p.major}{p.minor} | training dtype: {dtype}")
            x = torch.randn(512, 512, device="cuda")
            torch.cuda.synchronize()
            report("cuda matmul", "ok", f"{float((x @ x).abs().mean()):.3f}")
            arch = f"sm_{p.major}{p.minor}"
            if arch not in torch.cuda.get_arch_list() and f"compute_{p.major}{p.minor}" not in torch.cuda.get_arch_list():
                report("arch support", "warn", f"{arch} not in {torch.cuda.get_arch_list()} - install a newer CUDA "
                                               "wheel (e.g. -Cuda cu128 for RTX 50xx / Blackwell)")
        else:
            report("gpu", "warn", "CUDA not available - only the CPU smoke test will work")
    except Exception as exc:
        report("torch", "fail", repr(exc))
        return 1

    for mod in ("torchvision", "transformers", "peft", "accelerate", "diffusers", "huggingface_hub", "sklearn",
                "cv2", "networkx"):
        try:
            m = __import__(mod)
            report(mod, "ok", getattr(m, "__version__", ""))
        except Exception as exc:
            report(mod, "fail", repr(exc))

    try:
        import bitsandbytes as bnb
        detail = bnb.__version__
        if torch.cuda.is_available():
            lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
            lin(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
            detail += " (4-bit NF4 forward OK)"
        report("bitsandbytes", "ok", detail)
    except Exception as exc:
        report("bitsandbytes", "fail" if torch.cuda.is_available() else "warn", repr(exc)[:200])

    try:
        import cv2
        report("opencv ffmpeg", "ok" if "FFMPEG" in cv2.getBuildInformation() else "warn", "video decode backend")
    except Exception as exc:
        report("opencv ffmpeg", "fail", repr(exc))

    import torch.distributed as dist
    backend = "nccl" if (dist.is_nccl_available() and platform.system() != "Windows") else "gloo"
    report("distributed backend", "ok" if dist.is_available() else "warn", backend)

    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami()
        report("huggingface login", "ok", who.get("name", "?"))
    except Exception:
        report("huggingface login", "warn", "not logged in - run `huggingface-cli login` (Llama 3.2 Vision is gated)")

    print("\nEnvironment", "READY" if ok else "has FAILURES (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
