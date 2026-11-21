# Reproducing the revision's analyses

This covers the analyses added in the ICLR 2027 revision: the latency-weight sweep, calibration, label budget,
routing under shift, the fusion upper bound, the VAE round-trip test, D3 and the one-harness latency. The
submitted paper's own tables are covered in the README ("Reproducing the paper's tables").

Only code and small lists live in this repository. Videos, model weights and run outputs are not committed;
each step below says how to get or rebuild them.

## What is needed and where it comes from

| Input | Size | Source |
|---|---|---|
| This repository | small | `git clone` into `$SAFER_CODE` |
| Python environment | ~8 GB | `scripts/cluster/setup_env.sh` |
| Base models (Qwen2.5-VL-3B, Llama-3.2-11B-Vision, SVD VAE, XCLIP-B/16, Wan2.1 and CogVideoX VAEs) | ~30 GB | `scripts/cluster/download_models.sh` (Llama is gated: accept its license and log in with `huggingface-cli login` first) |
| Trained SAFER checkpoints (`export/`: scanner and arbiter adapters, heads, dispatchers, `tool_costs.json`) | ~0.65 GB | the training pipeline (README, `scripts/run_full_2class.sh`); copy the run's `export/` to `$SAFER_ROOT/export` |
| In-domain outcome tables (`outcomes_valid.npz`, `outcomes_test.npz`) | ~0.1 GB | the same training run (`runs/full_2class/predictions/`) |
| Original external outcome tables (the paper's per-action run) | ~0.15 GB | `benchmark_external_outcomes.py` with the paper's run (`runs/extbench_outcomes/`); used by the in-domain-vs-shift analyses and by the reproduction check |
| Public benchmark archives (ViF-Bench, GenBuster, GenVidBench, FakeParts subset) | ~65 GB | `scripts/cluster/download_public.sh`, then `build_datasets.py` |
| Exact video selection of the paper's rerun | 2 MB | `reproduce/lists/` (relative paths) |
| Chrono-66k raw real clips (VAE test only) | — | the Chrono-66k release; set `CHRONO_RAW` to the folder holding `chrono_triclass/` |

## 1. Paths and environment

The scripts in `scripts/cluster/` run on a Slurm cluster with environment modules. Set three paths, for
example in `~/.bashrc`:

```bash
export SAFER_ROOT=/path/to/work/safer          # work dir: raw/, data/, lists/, ext/, export/, logs/, .venv/
export SAFER_CODE=$SAFER_ROOT/code             # this repository
export SAFER_DATA=/path/to/large/storage/safer # outputs of the D3 and VAE runs (defaults to $SAFER_ROOT)
# optional: HF_HOME (model cache, default $SAFER_ROOT/hf), PIP_CACHE_DIR,
#           SAFER_MODULES (modules to load; default "GCCcore/13.2.0 Python/3.11.5", empty to skip)
git clone <this repository> "$SAFER_CODE"
bash "$SAFER_CODE/scripts/cluster/setup_env.sh"      # login node, CPU only
```

Partition and GPU names in the `#SBATCH` lines (`cpu`, `gpu`, `gpu_debug`, `gpu:h100:1`, `gpu:a30:1`) are those
of the cluster we used; edit them for yours. Submit scripts write Slurm logs to `$SAFER_ROOT/logs/`.

## 2. Models, data and lists (login node, network only)

```bash
C=$SAFER_CODE/scripts/cluster
bash $C/download_models.sh                       # base models into $HF_HOME (jobs run with HF_HUB_OFFLINE=1)
bash $C/download_public.sh                       # benchmark archives and FakeParts manifests into $SAFER_ROOT/raw
CHRONO_RAW=/path/to/chrono/raw bash $C/install_lists.sh install   # paper's FakeParts subset and VAE clip list
cd $SAFER_CODE && python build_datasets.py fakeparts              # downloads that FakeParts subset (~8 GB)
```

`build_datasets.py extract` and `build_datasets.py csv` (unpack the archives, write `lists/<bench>.csv`) run inside
`job_prepare.slurm`. The CSV lists are deterministic (sorted files, seed 0); after the prepare job,
`bash $C/install_lists.sh check` confirms they match the paper run's lists in `reproduce/lists/`.

Copy the trained checkpoints to `$SAFER_ROOT/export` and the original external outcome tables to
`$SAFER_ROOT/old_tables/` (`<bench>.npz`).

## 3. GPU runs

| Run | Command | Jobs | Writes |
|---|---|---|---|
| External rerun with the 23 features saved | `bash $C/submit_rerun.sh` | prepare (CPU, ≤3 h, normalizes all videos) → smoke (1 GPU, ≤30 min, checks scores against `old_tables/`) → full (4 GPU shards, ≤2.5 h each, resumable) → merge (CPU) | `$SAFER_ROOT/ext/outcomes/<bench>.npz` |
| D3 on the normalized videos | `bash $C/submit_d3.sh smoke`, then `full` | scores (1 GPU, ~25 min for all four benchmarks) → analysis (CPU) | `$SAFER_DATA/d3_full/<bench>.csv`, `$SAFER_ROOT/rebuttal_out/d3_full/` |
| VAE round-trip vs re-encoding | `bash $C/submit_vae.sh smoke`, then `full` | make variants (CPU) → VAE round-trip (2 GPU jobs) → prepare (CPU) → outcomes (2 GPU shards) → analysis (CPU) | `$SAFER_DATA/vae_full/`, `$SAFER_ROOT/rebuttal_out/vae_full/` |

Run the smoke variant of D3 and VAE first; it processes 20 videos per benchmark and 5 clips respectively.
D3 and the VAE test need the rerun's `ext/manifests/` and `ext/outcomes/`.

After the rerun's merge job (CPU, login node or a CPU job):

```bash
bash $C/run_rerun_analyses.sh      # fusion bound, feature direction, reproduction check, external analyses
```

## 4. CPU analyses on the outcome tables (any machine)

Run from the repository root, with the in-domain tables in `runs/full_2class/predictions/` and the original
external tables in `runs/extbench_outcomes/`:

```bash
python -m safer.eval.rebuttal all                                      # in-domain: λ sweep (3 seeds), McNemar, calibration, abstention, low-FPR
python -m safer.eval.rebuttal_external --dir runs/extbench_outcomes    # under shift: label budget, calibration, recalibrate-then-route, routing
python -m safer.eval.rebuttal_latency --dir runs/extbench_outcomes --out runs/rebuttal_latency
```

## 5. Where each revision result comes from

| Revision (appendix paragraph or section) | Script | Output |
|---|---|---|
| Sec. 4.3, "Latency-weight sweep", "Abstention", "Probability calibration in-domain" | `safer.eval.rebuttal` | `runs/rebuttal/rebuttal_sweep.md`, `rebuttal_fast.md` |
| "Label budget under shift", "Calibration under shift", "Recalibrate, then route", "Routing under shift" | `safer.eval.rebuttal_external` | `runs/rebuttal/rebuttal_external.md` |
| "Fusion upper bound and feature direction under shift" | `safer.eval.rebuttal_fusion` (via `run_rerun_analyses.sh`) | `rebuttal_out/rebuttal_fusion.md` |
| "VAE round-trip against re-encoding" | `safer.eval.vae_roundtrip`, `safer.eval.rebuttal_vae` (via `submit_vae.sh`) | `rebuttal_out/vae_full/rebuttal_vae.md` |
| "A training-free external detector" (D3), Table 17 | `safer.eval.d3_scores`, `safer.eval.rebuttal_d3` (via `submit_d3.sh`) | `rebuttal_out/d3_full/rebuttal_d3.md` |
| "Latency in one harness", Sec. 4.5 | `safer.eval.rebuttal_latency` | `runs/rebuttal_latency/rebuttal_latency.md` |

`safer.eval.rebuttal` reimplements the supervised router and the confidence gate from the paper's text; its
gate makes 17 in-domain errors where the paper's original code (Table 3) makes 18.
