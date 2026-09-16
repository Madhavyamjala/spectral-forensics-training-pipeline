# Regenerating the AI-Edited class

This rebuilds the **AI-Edited** third of Chrono-TriClass-100k from scratch, following
*AI Edited Data Source and Pipeline*: every video is produced from a Kinetics-400 source clip by
a named manipulation model, and the manifest records which one.

The old class had 31,128 rows, 89% of them `generator_edit_method = "unknown"`. The new one has
33,333 rows and every single one knows its family, its model, its source clip and its edit
parameters — which is what makes per-method forensic analysis possible at all.

---

## 1. What gets built

33,333 videos over eight families and 32 models. The document specifies 28,333; the extra 5,000
are spread evenly over the four face-centric families to reach class balance against the
33,334 `real` and 33,312 `ai_generated` rows.

| Family | Spec | Built | Models |
|---|---:|---:|---|
| Face swap | 5,000 | **6,250** | SimSwap, FaceShifter, Face Transformer, INSwapper |
| Facial reenactment | 4,000 | **5,250** | Face2Face, FOMM, PIRenderer, Face2Face-ρ |
| Lip-sync | 4,000 | **5,250** | Wav2Lip, MuseTalk 1.5, VideoReTalking, SadTalker |
| Expression / attribute | 3,500 | **4,750** | GANimation, StyleGANEX, LatentTransformer, VQ editing |
| Object insertion / removal | 3,500 | 3,500 | ProPainter, Object-WIPER, AnyV2V, VideoComposer |
| Background manipulation | 3,000 | 3,000 | SAM2 composite, FLUX.1-schnell, SVD, ProPainter |
| Video inpainting | 3,000 | 3,000 | ProPainter, E²FGVI-HQ, STTN, FuseFormer |
| Video-to-video | 2,333 | 2,333 | TokenFlow+SD, InsV2V, AnyV2V, NVIDIA vid2vid |
| **Total** | 28,333 | **33,333** | |

Every (source group × model) cell count is derived by controlled rounding from the document's own
margins, so rows and columns both add up exactly. Inspect the full plan with:

```bash
python -m csf.generation.spec
```

The document's hand-computed tables are reproduced exactly at the base targets — including the
awkward ones (video-to-video 116/116/112/106, inpainting 180/150/130/140, and the object family's
700/200, 900/0, 0/850, 150/700 removal-insertion split).

## 2. Model coverage — read this before planning the week

The 32 models are **not** equally available. Some have no public inference release at all, and
several publish weights only through Google Drive, which cannot be fetched unattended.

```bash
python -m csf.generation.adapters      # per-model status, env and video count
```

Currently **13 of 32 models are wired end to end, covering 11,688 of the 33,333 videos (35%)**:

| Wired (tier 1) | Registered but not runnable (tier 2) |
|---|---|
| INSwapper, FOMM, Wav2Lip, TokenFlow, ProPainter (×3 roles), E²FGVI-HQ, STTN, FuseFormer, SAM2 composite, FLUX.1-schnell, SVD | SimSwap, FaceShifter, Face Transformer, Face2Face, PIRenderer, Face2Face-ρ, MuseTalk, VideoReTalking, SadTalker, GANimation, StyleGANEX, LatentTransformer, VQ editing, Object-WIPER, AnyV2V (×2), VideoComposer, InsV2V, vid2vid |

A tier-2 model **refuses its jobs** rather than emitting a copied or lightly-perturbed source
clip. That is deliberate: a placeholder video labelled AI-Edited would teach the detector that
"AI-Edited" means "unchanged video", which is worse than having fewer rows. Failed jobs are
recorded in the ledger and simply left out of the new manifest.

Three tier-1 models (E²FGVI-HQ, STTN, FuseFormer) clone cleanly but need their checkpoints staged
by hand — see §6.

**To promote a tier-2 model:** implement `csf/generation/adapters/workers/worker_<x>.py` against
the protocol in `adapters/base.py`, point its `EnvSpec` at the repo and weights, and set
`implemented=True` in the registry. Nothing else in the pipeline changes.

## 3. Hardware assumptions

Written for `tfgpu.cs.fiu.edu`: 6 × H200 NVL (143 GB). **GPUs 0 and 1 are held by vLLM workers**,
so everything here uses **GPUs 2, 3 and 4** and never touches the others.

- Generation pins its own workers via `generation.gpus: [2, 3, 4]`.
- Training uses `CUDA_VISIBLE_DEVICES=2,3,4` (`CSF_TRAIN_GPUS` overrides it).

Disk: budget ~1.2 TB — Kinetics source clips (~250 GB after filtering), the 33k generated videos
(~350 GB), the per-model environments (~120 GB, mostly duplicated torch builds), and the feature
cache (~25 GB). Generation pauses rather than crashing if free space drops below
`generation.min_free_gb`.

## 4. Running it

Each phase is resumable; re-running skips completed work.

```bash
# 0. one-off: build the per-model virtual environments (~1-2 h, pulls torch, repos, weights)
bash scripts/run_regen.sh --envs
bash scripts/run_regen.sh --status          # what is ready / stale / missing

# 1. Kinetics-400 source clips + qualification scoring
bash scripts/run_regen.sh --kinetics

# 2. render the videos on GPUs 2/3/4
bash scripts/run_regen.sh --generate

# 3. rebuild the manifest around what was actually produced
bash scripts/run_regen.sh --manifest

# 4. train the CSF pipeline on the new dataset
bash scripts/run_regen.sh --train
```

Before committing days to it, run the wiring check — a 200-video plan on the two cheapest
adapters, about 30 minutes:

```bash
python main.py --config configs/regen_smoke.yaml --stage kinetics,generate,regen_manifest
```

### Useful overrides

```bash
# only one model (e.g. after staging its weights)
bash scripts/run_regen.sh --generate --set generation.only_models='[inswapper]'

# retry everything that failed last time
bash scripts/run_regen.sh --generate --set generation.retry_failed=true

# cap the wall-clock budget; the run stops scheduling new groups after this
bash scripts/run_regen.sh --generate --set generation.deadline_hours=60
```

## 5. Expected timing

Estimated from `scheduler.COST_HINTS`, which the scheduler replaces with measured throughput once
a model has produced a few videos:

| Phase | Estimate |
|---|---|
| Kinetics download + scoring | 8–20 h (dominated by shard transfer) |
| Environment builds | 1–2 h |
| Generation, tier-1 models, 3 GPUs | ~59 GPU-hours wall clock (~2.5 days) |
| Feature extraction (~100k videos) | 8–14 h |
| Training + eval + export | 1.5–2.5 days |

TokenFlow is the critical path at roughly 40 h on its own GPU (DDIM inversion of every frame);
the balancer gives it a GPU to itself. If the week gets tight, drop it first:
`--set generation.skip_models='[tokenflow]'`.

`generation.deadline_hours` defaults to 84 (3.5 days) so generation always ends with a usable,
manifest-able set of videos rather than being killed mid-write.

## 6. Staging the checkpoints that cannot be auto-fetched

E²FGVI-HQ, STTN and FuseFormer publish weights via Google Drive. Put them here, then re-run:

```
cache/regen/envs/videoinpaint/repos/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth
cache/regen/envs/videoinpaint/repos/STTN/checkpoints/sttn.pth
cache/regen/envs/videoinpaint/repos/FuseFormer/checkpoints/fuseformer.pth
```

Each worker checks for its checkpoint at load time and fails with the exact path it wants, so a
missing file costs one log line, not a burned model group.

## 7. Publishing (destructive)

Replacing the AI-Edited class on the Hub is opt-in and never runs from `--stage all`:

```bash
bash scripts/run_regen.sh --push        # asks for confirmation + a WRITE token
```

It uploads the new videos, then the new manifest, then deletes only the `AI Edited/...` paths the
new manifest does not reference. Deletion runs last, so an interrupted push leaves extra files
rather than missing ones. Dry-run first with `--set generation.push.dry_run=true`.

## 8. Where things go wrong

| Symptom | What it means |
|---|---|
| `Only N clips pass the 'face' filter` | No face detector in the driver env. `pip install insightface` or `mediapipe`, then re-run with `--set generation.kinetics.rescore=true`. |
| A whole model group fails instantly | Usually a missing weight. Read `runs/regen/logs/generation/worker_<model>_gpu<N>.log` — the worker names the file it wanted. |
| `group abandoned after N consecutive failures` | `fail_fast` tripped. The group is skipped, the run continues; fix the cause and re-run with `retry_failed=true`. |
| `NotImplemented[<model>]` | Tier-2 model. Expected — see §2. |
| Generation paused on disk | Free space fell under `min_free_gb`. |
| Features stage re-downloading videos | `generation.video_root` ≠ `paths.video_dir`. They must match. |

Per-job outcomes live in `cache/regen/ledger.jsonl`; failures are summarised in
`runs/regen/metrics/generation_failures.csv`, and the dataset composition that actually resulted
in `runs/regen/metrics/regeneration_report.json`.
