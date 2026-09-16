"""
Multi-process launcher (torchrun replacement for Windows, works on Linux too).

torchrun's c10d rendezvous is unreliable on Windows builds of PyTorch (libuv TCPStore errors), so this
spawns one process per GPU itself, sets RANK / LOCAL_RANK / WORLD_SIZE and a file-based rendezvous
(CSF_INIT_METHOD=file://...) that `csf.distributed.init_distributed` picks up with the Gloo backend.
Rank 0 keeps the terminal's stdin (for the Hugging Face push prompt); other ranks get no stdin.
If any rank exits with an error, the remaining ranks are terminated and the launcher returns that code.

Input : --nproc N, then the script and its arguments, e.g.
        python -m csf.launch --nproc 2 main.py --config configs/full.yaml
Output: exit code of the first failing rank (0 if all succeed).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Spawn one training process per GPU")
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("script")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    store = Path(tempfile.gettempdir()) / f"csf_dist_{uuid.uuid4().hex}"
    procs = []
    for rank in range(a.nproc):
        env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(a.nproc),
                   LOCAL_WORLD_SIZE=str(a.nproc), CSF_INIT_METHOD="file:///" + str(store).replace(os.sep, "/"),
                   PYTHONUNBUFFERED="1")
        procs.append(subprocess.Popen([sys.executable, a.script, *a.args], env=env,
                                      stdin=None if rank == 0 else subprocess.DEVNULL))
        print(f"[csf.launch] started rank {rank} (pid {procs[-1].pid})", flush=True)

    code = 0
    try:
        while procs:
            for p in list(procs):
                rc = p.poll()
                if rc is None:
                    continue
                procs.remove(p)
                if rc != 0 and code == 0:
                    code = rc
                    print(f"[csf.launch] a rank exited with code {rc}; terminating the others. "
                          f"See runs/<run>/logs/rank*.log and crash_rank*.json", flush=True)
                    for q in procs:
                        q.terminate()
            time.sleep(0.5)
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        code = 130
    finally:
        try:
            store.unlink(missing_ok=True)
        except OSError:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
