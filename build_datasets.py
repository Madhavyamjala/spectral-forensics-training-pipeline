"""Unpack the public archives and write one CSV list per benchmark (path,label,category).

Run on a login node after scripts/cluster/download_public.sh (stdlib only; needs 7z on PATH).
Set SAFER_ROOT first (see docs/REPRODUCE.md).
    python build_datasets.py extract      # unzip / un-rar
    python build_datasets.py fakeparts    # pick and download the FakeParts subset (per-file, resumable)
    python build_datasets.py csv          # write lists/<bench>.csv
Category counts copy the original run (runs/extbench_outcomes), seed 0.
"""
import csv, json, os, random, subprocess, sys, urllib.request, zipfile
from pathlib import Path

P = Path(os.environ.get("SAFER_ROOT") or sys.exit("set SAFER_ROOT (see docs/REPRODUCE.md)"))
R, D, L = P / "raw", P / "data", P / "lists"
HF = "https://huggingface.co/datasets/hi-paris/FakeParts/resolve/main/"
VIDEO = (".mp4", ".avi", ".mov", ".webm", ".mkv")

# FakeParts: HF manifest -> (label, category as in the original run, count in the original run)
FAKEPARTS = {
    "style__anyv2v": ("ai_edited", "Change_of_style/AnyV2V", 300),
    "style__rave": ("ai_edited", "Change_of_style/RAVE", 300),
    "extrapolation__cosmos_predict2": ("ai_generated", "Extrapolation/Cosmos-Predict2", 300),
    "faceswap__insightface": ("ai_edited", "Faceswap/Insightface", 300),
    "inpainting__diffueraser": ("ai_edited", "Inpainting/DiffuEraser", 300),
    "inpainting__propainter": ("ai_edited", "Inpainting/Propainter", 300),
    "inpainting__rovi": ("ai_edited", "Inpainting/ROVI/Inpainting_MP4", 300),     # fake rows
    "interpolation__framer": ("ai_edited", "Interpolation/Framer", 300),
    "outpainting__akira": ("ai_edited", "Outpainting/akira", 600),                # akira + akira2 (300 each)
    "real__tenk_youtube": ("real", "Real/TenKReal", 2000),
    "t2v__cogvideox": ("ai_generated", "T2V/CogVideoX", 300),
    "t2v__hunyuanvideo": ("ai_generated", "T2V/HunyuanVideo", 300),
    "t2v__ltx_video": ("ai_generated", "T2V/LTXVideo", 300),
    "t2v__mochi1": ("ai_generated", "T2V/Mochi1", 300),
    "t2v__opensora": ("ai_generated", "T2V/Open-Sora", 300),
    "t2v__wan21": ("ai_generated", "T2V/Wan_21", 129),
    "t2v__allegroai": ("ai_generated", "T2V/allegroai", 128),
    "t2v__sora_1080p": ("ai_generated", "T2V/sora-1080p", 128),
    "t2v__sora_720p": ("ai_generated", "T2V/sora-720p", 72),
    "t2v__veo2": ("ai_generated", "T2V/veo2-720p", 300),
    "it2v__opensora": ("ai_generated", "TI2V/Open-Sora-768px", 300),
    "it2v__wan21": ("ai_generated", "TI2V/WAN", 93),
}


def extract():
    D.mkdir(parents=True, exist_ok=True)
    for z, dest in (("vifbench_source_videos.zip", "vifbench"), ("GenBuster-Bench.zip", "genbuster")):
        if not (D / dest).exists():
            print("unzip", z, flush=True)
            zipfile.ZipFile(R / z).extractall(D / dest)
    for r in ("ms", "t2vz", "vript"):
        out = D / "genvidbench" / r
        if not out.exists():
            out.mkdir(parents=True)
            print("7z x", r, flush=True)
            subprocess.run(["7z", "x", "-y", f"-o{out}", str(R / f"genvidbench_{r}.rar")], check=True,
                           stdout=subprocess.DEVNULL)


def _rows(manifest: str):
    for line in open(R / "fakeparts_manifests" / f"{manifest}.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            yield json.loads(line)


def fakeparts(dry: bool = False):
    if not dry and (P / "fakeparts_picks.json").exists():
        # reuse the paper run's subset (scripts/cluster/install_lists.sh install) instead of drawing again
        picks = [tuple(p) for p in json.load(open(P / "fakeparts_picks.json"))]
        print("using existing fakeparts_picks.json:", len(picks), "videos", flush=True)
        return _download_fakeparts(picks)
    rng = random.Random(0)
    picks = []
    for key, (label, cat, n) in FAKEPARTS.items():
        man, _, want_real = key.partition("#")
        rows = [r for r in _rows(man) if r.get("file_name", "").lower().endswith(VIDEO)
                and r.get("storage_status", "local") == "local"
                and ((r.get("label") == "real") == (want_real == "real" or label == "real"))]
        rows.sort(key=lambda r: r["file_name"])
        sel = rng.sample(rows, min(n, len(rows)))
        print(f"{key:34s} available {len(rows):6d} picked {len(sel):5d} -> {cat}", flush=True)
        picks += [(r["file_name"], label, cat) for r in sel]
    if dry:
        return
    _download_fakeparts(picks)


def _download_fakeparts(picks):
    root = D / "fakeparts"
    for i, (fn, _, _) in enumerate(picks):
        dst = root / fn
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(HF + fn, dst.with_suffix(".part"))
                os.replace(dst.with_suffix(".part"), dst)
                break
            except Exception as e:
                print("retry", fn, e, flush=True)
        if (i + 1) % 500 == 0:
            print(f"fakeparts {i + 1}/{len(picks)}", flush=True)
    with open(P / "fakeparts_picks.json", "w") as fh:
        json.dump(picks, fh)
    print("FAKEPARTS_DONE", len(picks))


def write_csv():
    L.mkdir(parents=True, exist_ok=True)
    # GenBuster: id/real, id/fake, ood/fake, wild/fake; category = split/label as in the original run
    gb = D / "genbuster" / "GenBuster-Bench"
    with open(L / "genbuster.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "category"]); n = 0
        for p in sorted(gb.rglob("*")):
            if p.suffix.lower() in VIDEO:
                rel = p.relative_to(gb).parts
                w.writerow([str(p), "real" if rel[1] == "real" else "ai_generated", f"{rel[0]}/{rel[1]}"]); n += 1
    print("genbuster", n)
    # GenVidBench: 500 per category, seed 0 (pika / vc2 overlap the training generators and are skipped)
    rng = random.Random(0)
    with open(L / "genvidbench.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "category"])
        for cat, label in (("ms", "ai_generated"), ("t2vz", "ai_generated"), ("vript", "real")):
            files = sorted(p for p in (D / "genvidbench" / cat).rglob("*") if p.suffix.lower() in VIDEO)
            sel = rng.sample(files, min(500, len(files)))
            print("genvidbench", cat, len(files), "->", len(sel))
            for p in sel:
                w.writerow([str(p), label, cat])
    # FakeParts
    picks = json.load(open(P / "fakeparts_picks.json"))
    with open(L / "fakeparts.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "category"])
        for fn, label, cat in picks:
            w.writerow([str(D / "fakeparts" / fn), label, cat])
    print("fakeparts", len(picks))
    print("ViF-Bench uses the folders adapter on", D / "vifbench" / "source_videos", "(ids match the original run)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"extract": extract, "fakeparts": fakeparts, "fakeparts-dry": lambda: fakeparts(True), "csv": write_csv}[cmd]()
