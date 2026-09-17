# Regenerating the AI-Edited class

This rebuilds the **AI-Edited** third of Chrono-TriClass-100k from scratch, following
*AI Edited Data Source and Pipeline*: every video is produced from a Kinetics-400 source clip by
a named manipulation model, and the manifest records which one.

The old class had 31,128 rows, 89% of them `generator_edit_method = "unknown"`. The new one has
33,333 rows and every single one knows its family, its model, its source clip and its edit
parameters — which is what makes per-method forensic analysis possible at all.

---

## 0. Before you start

```bash
python -m csf.generation.prefetch --list            # the ~21 Hub repos a run needs
python -m csf.generation.prefetch --stage generate  # pull them (~90 GB), resumable
```

Source clips come from the Hugging Face mirror
[`liuhuanjim013/kinetics400`](https://huggingface.co/datasets/liuhuanjim013/kinetics400).
Community mirrors are not consistent in layout, so `SourcePool.probe_hf_layout` lists the repo
once and picks the matching reader:

| layout detected | reader |
|---|---|
| `<split>/<label>/<clip>.mp4` | one `hf_hub_download` per clip, per-label quotas |
| `*.tar` / `*.tar.gz` / `*.zip` shards | stream a shard, keep the needed clips, delete it |
| a `datasets`-loadable table | stream rows, resolve the class, materialise the clips |

The default mirror is the third kind, and it has two properties worth knowing:

**There is no label column.** Its 241,181 rows are `video_id`, `video_path`, `metadata`,
`clips[]` and `frames[]` — the action class appears nowhere directly. `resolve_row_label` recovers
it, most trustworthy source first:

1. an explicit label field, if a mirror has one,
2. the same inside `metadata`,
3. the directory component of `video_path` / `clips[].clip_path`,
4. **the official Kinetics annotation CSVs, joined on `video_id`** — this mirror's ids are YouTube
   ids, which is what those CSVs key on. This is the path that actually works here, which is why
   the `kinetics` stage downloads the annotation CSVs even when the S3 mirror is not used,
5. a frame-level `annotation` string.

**Clips are paths, not bytes.** Rows reference files elsewhere in the repo, so each selected clip
is fetched with `hf_hub_download`. Rows also carry per-clip `quality_metrics` and per-frame
`aesthetic_score`, so `rank_clips` takes the best clip of each video rather than the first — free
quality, since the mirror already did the scoring.

If the mirror cannot satisfy some labels, the CVDF S3 shards
(`generation.kinetics.mirror_base`) fill the gap. Set `generation.kinetics.local_root` instead if
Kinetics is already extracted on the cluster, and nothing is downloaded at all.

The `kinetics` stage needs the `datasets` package (`pip install datasets`, already in
`requirements.txt`), and ffmpeg for encoding. ffmpeg is resolved from `CSF_FFMPEG`/`CSF_FFPROBE`,
then `PATH`, then the static binary that `imageio-ffmpeg` ships — so `pip install imageio-ffmpeg`
is enough on a machine where you cannot install system packages.

### Licence

Kinetics-400 is **CC BY 4.0**, and every AI-Edited video is a modified Kinetics clip, so anything
you redistribute has to credit the original authors, link the licence and say that changes were
made. `csf/generation/upload.py` writes all three into the dataset card and an `ATTRIBUTION.md`
on every push; see [ATTRIBUTION.md](../ATTRIBUTION.md) in the repo root.

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

## 2. Model coverage and substitutions

The 32 models the document names are not equally available. Several publish weights only through
Google Drive (unfetchable unattended), some released training code with no inference checkpoint,
and a few were never released at all.

Where a named model cannot be run, a **substitute** fills the slot. The slot preserves the
document's allocation - its source-content mix and per-family balance - while the manifest records
the model that *actually rendered* the video:

| column | meaning |
|---|---|
| `generator_edit_method`, `model` | the model that really ran (`latentsync`, `vace`, ...) |
| `spec_model` | the document's slot it filled (`videoretalking`, `insv2v`, ...) |
| `substituted` | whether those two differ |

This separation is not cosmetic. Recording the slot name would attribute LatentSync's diffusion
fingerprint to VideoReTalking, and the per-method accuracy breakdown - the thing this whole
regeneration exists to enable - would report a model that never ran.

```bash
python -m csf.generation.adapters      # full table: slot, renderer, videos, GPU-hours
```

**Coverage: 26,828 of 33,333 videos (80.5%) across 23 distinct renderers**, of which 12,515 are
produced by a substitute.

### The substitutions

| Slot | Why it cannot run | Substitute | Evidence |
|---|---|---|---|
| SimSwap | Drive-only weights | **DreamID-V** (Apache-2.0, Wan2.1 DiT) | 99.9% vs 95.24% ID retrieval |
| FaceShifter | training code only | **REFace** (WACV'25) | ⚠ non-commercial data, see below |
| VideoReTalking | Drive-only bundle | **LatentSync 1.6** (Apache-2.0) | HDTF FID 7.03 vs 9.5, SyncConf 8.9 vs 7.5, FVD 193 vs 271 |
| Face2Face | never released | **LivePortrait** | implicit keypoints, distinct from FOMM's affine warping |
| PIRenderer | Drive-only weights | **TPSMM** (MIT) | thin-plate-spline, a third motion basis |
| GANimation | no weights published | **LivePortrait retargeting** | continuous expression magnitude, the AU-control analogue |
| Object-WIPER | no public code | **DiffuEraser** (Apache-2.0) | diffusion removal vs ProPainter's flow propagation |
| AnyV2V ×2, VideoComposer, InsV2V | Drive-only / manual request | **VACE Wan2.1-1.3B** (Apache-2.0) | masked V2V covers all four |

Two slots turned out to be **available after all** and are now wired as themselves, not
substituted: **MuseTalk** (MIT, `download_weights.sh` pulls from the Hub) and **SadTalker**
(Apache-2.0 since the non-commercial clause was dropped, `download_models.sh` pulls from GitHub
Releases).

### Still unwired (6,505 videos)

`face_transformer`, `face2face_rho`, `styleganex`, `latent_transformer`, `vq_facial_editing`,
`vid2vid`. These are left as honest gaps rather than filled, because every remaining candidate
would duplicate a mechanism already in that family. A second slot rendered by an identical model
adds rows but no new artifact class, and makes the per-method breakdown report one fingerprint
under two names. `vid2vid` in particular exists to contribute a *non-diffusion GAN* artifact
class, so substituting another diffusion model would defeat the slot's purpose.

### ⚠ REFace licence

REFace's code is MIT but its checkpoint is trained on CelebAMask-HQ, which permits
**non-commercial research use only**, and generated videos inherit that. Its worker refuses to
start unless you acknowledge this:

```yaml
generation:
  accept_noncommercial: true      # enables REFace (1,563 videos)
```

Leave it `false` (the default) and the slot is simply skipped. Decide before generating, not
after.

## 2a. Reaching the full 33,333

Six slots cannot be filled at any compute budget. To still produce the number the document
specifies, `generation.reallocate_unfillable` (on by default) re-apportions each family's target
across the models in that family that *do* run. Family totals and source-group mixes are
preserved exactly; only the per-pipeline split changes.

```
face_swap                        6250   dreamid_v=2344, reface=1953, inswapper=1953
facial_reenactment               5250   liveportrait=1750, fomm=1750, tpsmm=1750
lip_sync                         5250   wav2lip=1313, musetalk=1313, latentsync=1312, sadtalker=1312
expression_attribute_editing     4750   liveportrait_expr=2500, styleganex=2250
object_insertion_removal         3500   propainter_object=900, diffueraser=900, vace=1700
background_manipulation          3000   bg_real_composite/flux/svd/propainter = 750 each
video_inpainting                 3000   propainter_inpaint=900, e2fgvi_hq=750, sttn=650, fuseformer=700
video_to_video                   2333   tokenflow=785, vace=1548
TOTAL                           33333   24 distinct renderers
```

Set it to `false` to leave the unfillable slots empty and produce 26,828 instead.

## 2b. Per-GPU concurrency

The scheduler runs several workers of the same model on one card. This is not a micro-
optimisation: an H200 has 143 GB and INSwapper needs ~3 GB, so one worker per card leaves the
GPU almost entirely idle. The small nets are latency-bound anyway — most of a job is video
decoding and face detection on the CPU — so they scale nearly linearly. Diffusion samplers
already saturate the SMs and gain far less, but their VRAM footprint caps the worker count, so
one formula covers both:

```
workers = clamp(gpu_vram_gb * 0.85 / adapter.vram_gb, 1, max_workers_per_gpu)
```

Measured on a stub workload: **3.81× throughput at 4 workers**, with no duplicated or lost jobs.

### Time to the full 33,333

```
workers/GPU   effective GPU-h   wall clock on 4 GPUs
          1             612             6.4 days
          2             419             4.4 days
          4             377             3.9 days      <- default
          6             377             3.9 days      (concurrency saturates)
```

```bash
python -m csf.generation.budget --reallocate --gpus 4 --workers-per-gpu 4
```

Raise `generation.max_workers_per_gpu` past 4 and nothing improves — the diffusion models are
compute-bound by then. Getting below ~3.9 days needs more GPUs, not more workers.

## 2c. Fitting a smaller window

If you would rather cap wall-clock time than produce everything, set
`generation.budget_wall_clock_hours`. The planner then picks the model set up front, keeping at
least `budget_diversity_floor` (default 2) *distinct renderers* per family — renderers, not
slots, since two VACE slots are one mechanism — and spending the rest on volume.

```bash
python -m csf.generation.budget --hours 84 --gpus 3      # 18 models, 18,941 videos
```

This is preferred over `deadline_hours`, which stops whichever group is in flight when it fires
and so shapes the dataset by scheduling order. `deadline_hours` remains as a hard backstop.

## 2d. Progress, timing and resuming

Every long stage reports progress. With a terminal attached you get a `tqdm` bar; redirected or
under `nohup` you get the same information as periodic log lines, because a carriage-return bar
in a log file is useless. `CSF_NO_PROGRESS=1` silences both.

The environment build is the one that used to look frozen: it spends most of its time inside a
single `pip install torch`, and the output was buffered until the command finished. It now
streams, announces each step, and shows a heartbeat with elapsed time and the last line pip
printed:

```
Building environment 'sam2_diffusers' | 7 step(s). The torch install alone usually takes
5-20 minutes; each step streams its output below.
[sam2_diffusers  step 3/7] installing torch==2.4.1 (several GB)
  [  4m12s] installing torch==2.4.1 (several GB) - Downloading torch-2.4.1-cp312...whl (797 MB)
```

### How long each phase takes

Per environment, on a reasonable connection:

| Step | Time |
|---|---|
| venv + pip bootstrap | under a minute |
| `pip install torch` (envs that need it) | 5–20 min, dominated by ~2.5 GB of wheels |
| other requirements | 1–5 min |
| git clones | seconds to a minute |
| weight downloads | 1–15 min depending on the checkpoint |

There are 18 environments, but only the ones your run touches are built, and they are built on
first use. `bash scripts/run_regen.sh --envs` does them all up front (2–4 h); `--status` shows
which are ready.

Whole-run figures:

| Phase | 200-video smoke | Full 33,333 on 4 GPUs |
|---|---|---|
| environments | 15–45 min (2 envs) | 2–4 h (all 18) |
| prefetch | 5–15 min | 1–2 h (~115 GB) |
| kinetics: download | minutes if cached | 8–20 h |
| kinetics: scoring | 1–3 min | 1–3 h |
| generate | 20–60 min | ~3.9 days |
| regen_manifest | under a minute | 20–40 min (probe + hash every file) |

### Stopping and restarting

Every stage resumes. Stop with Ctrl-C and re-run the same command:

| What | Resumes by |
|---|---|
| environment build | a readiness marker per env; an interrupted build re-runs its pip steps, but pip skips what is already installed, so it is fast the second time |
| Kinetics download | counting clips already on disk, plus a ledger of consumed shards |
| clip scoring | `clip_features.csv`, checkpointed every 500 clips |
| generation | `ledger.jsonl`, appended per video, so a run that dies at 20,000 resumes at 20,001 |
| manifest / metadata | rebuilt from the ledger, cheap to redo |

Completed stages are also recorded in `runs/<run>/state.json` and skipped on the next run; use
`--force <stage>` to redo one deliberately.

## 3. Hardware assumptions

Written for `tfgpu.cs.fiu.edu`: 6 × H200 NVL (143 GB). The default config uses **GPUs 1–4**,
leaving GPU 0 to the vLLM workers.

- Generation pins its own workers via `generation.gpus: [2, 3, 4, 5]`, with
  `max_workers_per_gpu: 4` processes on each.
- `generation.driver_gpu` is the card the driver process itself binds. Without it a
  single-process run always takes `cuda:0`, since there is no `LOCAL_RANK` to take a hint from -
  which is wrong on a shared box. It defaults to `gpus[0]`.
- Training uses `CUDA_VISIBLE_DEVICES=2,3,4,5` (`CSF_TRAIN_GPUS` overrides it).

**These are physical ids**, the same numbering `nvidia-smi` prints. Do not also export
`CUDA_VISIBLE_DEVICES` for the generation stages: the ids would then be indices into that list
rather than physical devices, and `gpus: [2, 3, 4, 5]` would silently mean something else. The
run warns if it finds the variable set, and refuses outright if `generation.gpus` names a card it
cannot see.
- Adding GPUs is the only way below ~3.9 days for the full set; more workers per GPU does not
  help once the diffusion models dominate.

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
| Environment builds | 2–4 h (18 envs, several with their own torch build) |
| Generation at the default 84 h budget | 240 GPU-h ≈ 80 h wall clock on 3 GPUs |
| Generation, everything wired | 523 GPU-h ≈ 174 h ≈ 7.3 days |
| Feature extraction (~100k videos) | 8–14 h |
| Training + eval + export | 1.5–2.5 days |

The expensive models are DreamID-V (78 GPU-h), the four VACE slots (144 GPU-h combined) and
TokenFlow (40 GPU-h) — all diffusion samplers. The budget planner drops them first; pin one back
with `generation.budget_pin_models: [tokenflow]` if you would rather trade volume for that
mechanism.

`generation.deadline_hours` (90 h) remains as a hard backstop so generation always ends with a
usable, manifest-able set of videos rather than being killed mid-write.

## 6. Staging the checkpoints that cannot be auto-fetched

Four wired models publish weights via Google Drive / Tsinghua Cloud. Put them here, then re-run:

```
cache/regen/envs/videoinpaint/repos/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth
cache/regen/envs/videoinpaint/repos/STTN/checkpoints/sttn.pth
cache/regen/envs/videoinpaint/repos/FuseFormer/checkpoints/fuseformer.pth
cache/regen/envs/tpsmm/repos/TPSMM/checkpoints/vox.pth.tar
cache/regen/envs/stylegan/repos/StyleGANEX/pretrained_models/styleganex_editing.pt
```

StyleGANEX is worth the manual step specifically: it is the expression family's only second
mechanism. Without it, reallocation sends all 4,750 of that family's videos through LivePortrait
alone, and the per-method breakdown for the family becomes meaningless.

Everything else fetches itself. Two adapters call an upstream downloader during the env build
(SadTalker's `download_models.sh`, LivePortrait's `huggingface-cli download`); if those fail, the
env build reports it and the worker names the exact missing path at load time rather than failing
per job.

Each worker checks for its checkpoint at load time and fails with the exact path it wants, so a
missing file costs one log line, not a burned model group.

## 6a. Per-video metadata

The `regen_manifest` stage writes two files from one pass over the produced videos:

| file | purpose |
|---|---|
| `manifest_regen.csv` | training-facing, read by `csf.data.manifest` on every run - class, split, method |
| `metadata.csv` | analysis-facing, **70 columns**, 43 of them the specification's per-video fields |

They share `video_id`. Splitting them keeps the manifest narrow (the ablation only needs class,
split and method) while making the attribution fields the document asks for - identity, face
quality, visibility, occlusion, pose, mask class, audio source - real queryable columns rather
than a truncated JSON blob.

Column groups, listed by `python -m csf.generation.metadata --columns`:

- **identity** - `video_id`, `family`, `model`, `spec_model`, `substituted`, `split`, `sha256`
- **container** - duration, resolution, fps, codec, bitrate, audio. Probed from the *produced*
  file, never copied from the source clip: those fingerprints are exactly what
  `baseline_metadata_shortcut` in the ablation exists to expose.
- **job** - source/driving/audio clip ids, variant, mask class, prompt, seed, render seconds
- **specification** - the union of every family's `metadata_fields`, derived from `spec.py`, so a
  field added there becomes a column automatically

Where a field is both planned and measured, **the measured value wins** - a requested edit
magnitude the renderer could not honour would otherwise be recorded as fact.

`metadata_schema.json` sits beside it and reports the fill rate of every specification field per
family. Fields no renderer reports show up as `0.0`, which is the honest signal that the
attribution analysis cannot use them.

### Replacing the old class

`metadata.csv` is **merged, not overwritten**. Regenerating drops the old `ai_edited` rows, which
describe files the new manifest no longer references, and keeps every `real` and `ai_generated`
row untouched - those classes are not regenerated, and a wholesale rewrite would silently discard
two thirds of the dataset's metadata. Columns an older file carried are preserved, the write is
atomic, and re-running is idempotent.

`generation.keep_old_edited: true` keeps both sets, matching the manifest's behaviour.

Rebuild it at any time without re-rendering anything - it is reconstructed from the job plan and
the generation ledger:

```bash
python -m csf.generation.metadata --config configs/regen.yaml
```

The push uploads `metadata.csv` and `metadata_schema.json` alongside the manifest, so the Hub
copy can never describe deleted videos.

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
| Runs on `cuda:0` when you asked for another card | Set `generation.driver_gpu` (or `CSF_DRIVER_GPU=2`). A single-process run has no `LOCAL_RANK`, so it defaults to device 0. |
| `peft is not installed` / `bitsandbytes is required` on a generation-only run | Fixed: preflight now only demands the training stack when a training stage is selected. If you still see it, you have a training stage in `--stage`. |
| `generation.gpus ... names GPU(s) [n]` | Those ids do not exist in this process. Usually `CUDA_VISIBLE_DEVICES` is set and has renumbered them. |
| `N clip(s) are present ... but none could be read` | ffprobe is missing and OpenCV cannot decode them either. Install ffmpeg (`conda install -c conda-forge ffmpeg`), or set `generation.kinetics.probe_clips=false` to build the pool without container metadata. |
| `NoBaseEnvironmentError` from conda | Do not fight conda: `pip install imageio-ffmpeg` gives a static ffmpeg with no root and no conda, and the pipeline finds it automatically. |
| `No ffmpeg binary could be found` | As above, or set `CSF_FFMPEG` / `CSF_FFPROBE` to existing binaries. |
| `ffprobe was not found` warning | The pool still builds via OpenCV, but codec/bitrate/audio are recorded as unknown **and the generation workers need ffmpeg to encode**. Install it before the `generate` stage. |
| `No clips were downloaded to ...` | The download genuinely produced nothing - check `hf_repo` / `local_root` / `mirror_base` and your Hub login. |
| `N label(s) the spec needs have no file in this mirror` | Those Kinetics classes are spelled differently (or absent) upstream. The sampler redistributes within each source group, so a few are harmless. |
| `Only N clips pass the 'face' filter` | No face detector in the driver env. `pip install insightface` or `mediapipe`, then re-run with `--set generation.kinetics.rescore=true`. |
| A whole model group fails instantly | Usually a missing weight. Read `runs/regen/logs/generation/worker_<model>_gpu<N>.log` — the worker names the file it wanted. |
| `group abandoned after N consecutive failures` | `fail_fast` tripped. The group is skipped, the run continues; fix the cause and re-run with `retry_failed=true`. |
| `NotImplemented[<model>]` | Tier-2 model. Expected — see §2. |
| Generation paused on disk | Free space fell under `min_free_gb`. |
| Features stage re-downloading videos | `generation.video_root` ≠ `paths.video_dir`. They must match. |

Per-job outcomes live in `cache/regen/ledger.jsonl`; failures are summarised in
`runs/regen/metrics/generation_failures.csv`, and the dataset composition that actually resulted
in `runs/regen/metrics/regeneration_report.json`.
