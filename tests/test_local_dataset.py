"""
Dependency-free checks for reading the dataset from a local, extracted copy (`data.local_root`).

Reviewers get the dataset as one archive, so the pipeline must run from the unzipped folder alone:
manifest and videos found there (even one folder down, as `unzip -d` leaves them), no call to the
Hub, and no video ever deleted - `delete_videos_after_cache` is meant for a download cache, and
here the files are the user's only copy. The Hub is replaced by a stub that fails on any call.

Run: python tests/test_local_dataset.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES = []
HUB_CALLS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion, printing its outcome without stopping the suite."""
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f": {detail}" if not condition else ""))
    if not condition:
        FAILURES.append(name)


def _stub_hub() -> None:
    """Any Hub access is recorded and fails, so a passing run proves the local path never needs it."""
    def refuse(*args, **kwargs):
        HUB_CALLS.append(args)
        raise RuntimeError("Hub access attempted in local-dataset mode")

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = refuse
    hub.HfApi = lambda *a, **k: types.SimpleNamespace(list_repo_files=refuse)
    sys.modules["huggingface_hub"] = hub


def _make_copy(base: Path) -> Path:
    """An extracted archive one folder down: <base>/anonymous_repo/Chrono-66k/{manifest.csv, videos}."""
    data = base / "anonymous_repo" / "Chrono-66k"
    rows = ["class,generator_edit_method,video_id,split"]
    for split, n0 in (("train", 0), ("valid", 100), ("test", 200)):
        for k in range(3):
            real, gen = f"real-{n0 + k}", f"aigen-{n0 + k}"
            rows += [f"real,unknown,{real},{split}", f"ai_generated,pika,{gen},{split}"]
            # standard Hub layout for most files ...
            for folder, vid in (("Real", real), ("AI_Generated", gen)):
                p = data / folder / "part-1" / f"{vid}.mp4"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x00" * 16)
    # ... one real clip in a non-standard place, one non-standard id, one row with no file at all
    moved = data / "Real" / "part-1" / "real-0.mp4"
    moved.rename(data / "Real" / "real-0.mp4")
    rows.append("real,unknown,kinetics_abc,train")
    (data / "Real" / "extra").mkdir(parents=True)
    (data / "Real" / "extra" / "kinetics_abc.mp4").write_bytes(b"\x00" * 16)
    rows.append("ai_generated,pika,aigen-999,train")
    (data / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return data


def test_local_dataset() -> None:
    _stub_hub()
    from safer.config import DataConfig
    from safer.data import manifest as M
    from safer.data.video_io import download_video

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data = _make_copy(base)
        cfg = DataConfig(repo_id="unused/never-called", local_root=str(base / "anonymous_repo"),
                         classes=["real", "ai_generated"], max_rows=None)
        check("dataset root found one folder below local_root", M.local_dataset_root(cfg) == data,
              str(M.local_dataset_root(cfg)))
        check("manifest resolved inside the local copy",
              M.resolve_manifest(cfg, base / "cache") == data / "manifest.csv")

        df = M.load_run_manifest(cfg, base / "cache", seed=0)
        paths = dict(zip(df["video_id"], df["repo_path"]))
        check("every kept row points at an existing file", all((data / p).is_file() for p in paths.values()))
        check("standard layout used as is", paths.get("real-1") == "Real/part-1/real-1.mp4", str(paths.get("real-1")))
        check("moved file found by folder scan", paths.get("real-0") == "Real/real-0.mp4", str(paths.get("real-0")))
        check("non-standard id found by folder scan", paths.get("kinetics_abc") == "Real/extra/kinetics_abc.mp4",
              str(paths.get("kinetics_abc")))
        check("row without a file is dropped", "aigen-999" not in paths)
        check("all three splits keep both classes",
              df.groupby("split")["class"].nunique().to_dict() == {"test": 2, "train": 2, "valid": 2})

        cfg.delete_videos_after_cache = True
        vdir, repo, may_delete = M.video_source(cfg, base / "videos")
        check("videos read in place from the local copy", vdir == data and repo is None, f"{vdir}, {repo}")
        check("local videos are never deleted, even with delete_videos_after_cache", may_delete is False)

        got = download_video(repo, paths["real-1"], vdir)
        check("download_video returns the local file", got == data / "Real" / "part-1" / "real-1.mp4")
        try:
            download_video(repo, "Real/part-1/missing.mp4", vdir)
            check("missing local file raises", False, "no exception")
        except FileNotFoundError:
            check("missing local file raises FileNotFoundError", True)
        check("no Hub call was made", not HUB_CALLS, str(HUB_CALLS[:2]))

        explicit = data / "manifest_explicit.csv"
        explicit.write_text((data / "manifest.csv").read_text(encoding="utf-8"), encoding="utf-8")
        cfg_explicit = DataConfig(local_root=str(base / "anonymous_repo"), manifest=str(explicit))
        check("an explicit manifest path still wins", M.resolve_manifest(cfg_explicit, base / "cache") == explicit)

        hub_cfg = DataConfig(repo_id="some/repo", delete_videos_after_cache=True)
        check("Hub mode unchanged", M.video_source(hub_cfg, base / "videos") == (base / "videos", "some/repo", True))

        nested = DataConfig(local_root=str(base), manifest="manifest.csv")
        (base / "other").mkdir()
        (base / "other" / "manifest.csv").write_text("class,video_id,split\n", encoding="utf-8")
        try:
            M.local_dataset_root(nested)
            check("two manifests under local_root is an error", False, "no exception")
        except ValueError:
            check("two manifests under local_root is an error", True)


def main() -> int:
    test_local_dataset()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all local-dataset checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
