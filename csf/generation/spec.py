"""
The AI-Edited regeneration spec, encoded as data.

This module is the single source of truth for *what* has to be generated: the eight
manipulation families, the Kinetics-400 source groups each draws from, the manipulation
pipelines (models) inside each family, and the exact number of videos per
(source group x model) cell.

It is deliberately dependency-free (standard library only) so the allocation can be
unit-tested and inspected without torch, pandas or a GPU:

    python -m csf.generation.spec            # prints the full plan and self-checks it

Counts
------
The proposal document specifies 28,333 videos across the eight families. The run target is
33,333 (class balance against 33,334 real / 33,312 ai_generated rows), so 5,000 extra videos
are spread evenly over the four face-centric families (+1,250 each). `FAMILY_TARGETS` holds
the result; `BASE_TARGETS` keeps the document's original numbers for reference.

Cell counts are produced by `cross_split`, which does *controlled rounding*: every cell is an
integer, and the row sums and column sums both match their margins exactly. The document's own
hand-computed tables are reproduced by this procedure at the base targets.

Input : nothing (static data) or a scale factor per family.
Output: `FAMILIES` (ordered dict of `Family`), `build_plan()` -> list of `Cell` records
        covering every (family, source group, model) triple with its video count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------------------
# apportionment helpers
# --------------------------------------------------------------------------------------


def apportion(total: int, weights: Sequence[float]) -> List[int]:
    """Largest-remainder (Hamilton) apportionment of `total` over `weights`.

    Returns integers that sum exactly to `total` and follow the weights as closely as
    integer arithmetic allows. Zero-weight entries always get 0.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("weights must contain at least one positive value")
    exact = [total * w / wsum for w in weights]
    out = [int(x) for x in exact]
    remainder = total - sum(out)
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - out[i]), i))
    for k in range(remainder):
        out[order[k % len(order)]] += 1
    return out


def cross_split(row_totals: Sequence[int], col_totals: Sequence[int]) -> List[List[int]]:
    """Controlled rounding of the independence table row_i*col_j/N to integers.

    Guarantees: every cell >= 0, each row sums to `row_totals[i]`, each column sums to
    `col_totals[j]`. Raises if the margins themselves disagree.
    """
    r_sum, c_sum = sum(row_totals), sum(col_totals)
    if r_sum != c_sum:
        raise ValueError(f"margins disagree: rows sum to {r_sum}, columns sum to {c_sum}")
    n_r, n_c = len(row_totals), len(col_totals)
    if r_sum == 0:
        return [[0] * n_c for _ in range(n_r)]

    exact = [[row_totals[i] * col_totals[j] / r_sum for j in range(n_c)] for i in range(n_r)]
    grid = [[int(exact[i][j]) for j in range(n_c)] for i in range(n_r)]
    row_need = [row_totals[i] - sum(grid[i]) for i in range(n_r)]
    col_need = [col_totals[j] - sum(grid[i][j] for i in range(n_r)) for j in range(n_c)]

    # distribute the +1s by descending fractional remainder, respecting both margins
    cells = sorted(((exact[i][j] - grid[i][j], i, j) for i in range(n_r) for j in range(n_c)),
                   key=lambda t: (-t[0], t[1], t[2]))
    for _, i, j in cells:
        if row_need[i] > 0 and col_need[j] > 0:
            grid[i][j] += 1
            row_need[i] -= 1
            col_need[j] -= 1

    # repair pass: greedy can strand a unit when remainders tie; place the rest anywhere legal
    for i in range(n_r):
        while row_need[i] > 0:
            j = max(range(n_c), key=lambda c: (col_need[c], -grid[i][c]))
            if col_need[j] <= 0:
                raise RuntimeError("controlled rounding failed to converge")
            grid[i][j] += 1
            row_need[i] -= 1
            col_need[j] -= 1

    if any(row_need) or any(col_need):
        raise RuntimeError(f"controlled rounding left residuals rows={row_need} cols={col_need}")
    return grid


# --------------------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceGroup:
    """One row of a family's source table: a pool of Kinetics-400 clips to draw from."""
    key: str
    name: str
    weight: int                       # the document's video count for this group (a relative weight)
    labels: Tuple[str, ...] = ()      # Kinetics-400 class names that feed this group
    note: str = ""


@dataclass(frozen=True)
class Pipeline:
    """One column of a family's pipeline table: the model that performs the manipulation."""
    key: str                          # stable id, used in video ids and as the adapter name
    name: str                         # human-readable model name from the document
    weight: int                       # the document's video count for this pipeline
    mechanism: str = ""


@dataclass(frozen=True)
class Family:
    key: str
    name: str
    base_videos: int
    source_groups: Tuple[SourceGroup, ...]
    pipelines: Tuple[Pipeline, ...]
    # optional secondary breakdown (manipulation type / target / transformation family),
    # apportioned independently of the source x model grid and attached to jobs as metadata
    variants: Tuple[Tuple[str, int], ...] = ()
    metadata_fields: Tuple[str, ...] = ()
    source_filter: str = "any"        # which clip filter the source pool must satisfy
    note: str = ""

    def scaled(self, target: int) -> "Family":
        return Family(self.key, self.name, target, self.source_groups, self.pipelines,
                      self.variants, self.metadata_fields, self.source_filter, self.note)


@dataclass(frozen=True)
class Cell:
    """One (family, source group, model) bucket with the number of videos to produce."""
    family: str
    source_group: str
    model: str
    videos: int
    source_filter: str
    labels: Tuple[str, ...]


# --------------------------------------------------------------------------------------
# 1. Background manipulation
# --------------------------------------------------------------------------------------

BACKGROUND = Family(
    key="background_manipulation",
    name="Background manipulation",
    base_videos=3000,
    source_filter="any",
    source_groups=(
        SourceGroup("human_portrait", "Human / portrait / interaction", 600, (
            "answering questions", "applauding", "applying cream", "baby waking up", "crawling baby",
            "brushing hair", "brushing teeth", "getting a haircut", "laughing", "reading book",
            "sign language interpreting", "singing", "sitting up")),
        SourceGroup("sports_motion", "Sports / high-motion", 700, (
            "playing basketball", "catching or throwing baseball", "playing tennis", "playing volleyball",
            "playing kickball", "shooting goal (soccer)", "skateboarding", "skiing slalom",
            "skiing (not slalom or crosscountry)")),
        SourceGroup("animal", "Animal-centered", 350, (
            "grooming dog", "training dog", "walking the dog", "petting cat", "petting animal (not cat)",
            "feeding birds", "feeding fish")),
        SourceGroup("vehicle", "Vehicle / rider / mechanical", 400, (
            "driving car", "driving tractor", "motorcycling", "riding a bike", "riding mountain bike",
            "riding scooter", "riding unicycle")),
        SourceGroup("object_hand", "Object / hand-object interaction", 400, (
            "playing guitar", "playing piano", "playing violin", "playing cello", "playing drums",
            "playing keyboard", "playing cards")),
        SourceGroup("environmental", "Environmental / outdoor", 300, (
            "rock climbing", "biking through snow", "snowmobiling", "skydiving", "paragliding", "sailing")),
        SourceGroup("indoor", "Indoor / structured environments", 250, (
            "cleaning floor", "cleaning windows", "cleaning pool", "making bed", "moving furniture",
            "setting table", "washing dishes")),
    ),
    pipelines=(
        Pipeline("bg_real_composite", "SAM2 + alpha compositing", 750,
                 "Segment foreground, swap in a different real background composite (no generative model)"),
        Pipeline("bg_flux_image", "SAM2 + FLUX.1-schnell", 750,
                 "Static AI-generated background behind a moving foreground"),
        Pipeline("bg_svd_video", "SAM2 + Stable Video Diffusion", 750,
                 "Dynamic AI-generated background"),
        Pipeline("bg_propainter_recon", "SAM2 + ProPainter", 750,
                 "Foreground removed, background reconstructed via inpainting, foreground restored"),
    ),
    metadata_fields=("background_source", "segmentation_model", "foreground_area_frac", "composite_mode"),
)

# --------------------------------------------------------------------------------------
# 2. Face swap
# --------------------------------------------------------------------------------------

FACE_SWAP = Family(
    key="face_swap",
    name="Face swap",
    base_videos=5000,
    source_filter="face",
    source_groups=(
        SourceGroup("face_talking", "Face / talking / expression", 1250, (
            "answering questions", "testifying", "news anchoring", "sign language interpreting",
            "laughing", "crying", "yawning")),
        SourceGroup("beauty", "Beauty / personal activities", 600, (
            "applying cream", "brushing hair", "brushing teeth", "fixing hair", "getting a haircut",
            "shaving head", "shaving legs", "waxing eyebrows", "doing nails")),
        SourceGroup("eating", "Eating / drinking", 500, (
            "eating burger", "eating cake", "eating hotdog", "eating ice cream", "eating spaghetti",
            "eating carrots", "eating chips", "eating doughnuts", "eating watermelon",
            "drinking", "drinking beer", "drinking shots", "tasting beer", "tasting food")),
        SourceGroup("music_performance", "Music / performance", 600, (
            "singing", "beatboxing", "whistling", "playing guitar", "playing piano", "playing violin",
            "playing drums", "playing trumpet", "playing saxophone")),
        SourceGroup("everyday", "Everyday activities", 500, (
            "reading book", "texting", "using computer", "writing", "waiting in line", "unboxing")),
        SourceGroup("dance", "Dance", 500, (
            "dancing ballet", "dancing charleston", "dancing gangnam style", "dancing macarena",
            "belly dancing", "breakdancing", "salsa dancing", "tango dancing", "tap dancing",
            "robot dancing", "krumping", "zumba")),
        SourceGroup("sports_motion", "Sports / high motion", 650, (
            "playing basketball", "playing tennis", "playing volleyball", "catching or throwing baseball",
            "punching bag", "boxing", "skateboarding", "surfing water")),
        SourceGroup("multi_person", "Multi-person / interaction", 400, (
            "hugging", "kissing", "celebrating", "shaking hands", "tickling", "arm wrestling",
            "shaking head", "applauding")),
    ),
    pipelines=(
        Pipeline("simswap", "SimSwap", 1500,
                 "Identity Injection Module - arbitrary identity-to-identity swap, retains target expression/gaze"),
        Pipeline("faceshifter", "FaceShifter", 1250,
                 "Two-stage: AAD-based synthesis + HEAR-Net occlusion correction"),
        Pipeline("face_transformer", "Face Transformer", 1000,
                 "Transformer-based semantic correspondence between source and target"),
        Pipeline("inswapper", "INSwapper (InsightFace buffalo_l, ArcFace 128x128)", 1250,
                 "Identity-conditioned latent swap, high-throughput baseline"),
    ),
    metadata_fields=("source_identity_id", "target_identity_id", "face_swap_model", "source_face_quality",
                     "target_face_quality", "face_visibility", "occlusion_level", "yaw", "pitch", "roll"),
    note="Shared pipeline: face detect/align -> source identity embedding -> swap model -> blend -> "
         "temporal consistency -> audio restore -> MP4. Optional GFPGAN/CodeFormer restoration is "
         "post-processing, not a fifth architecture.",
)

# --------------------------------------------------------------------------------------
# 3. Facial reenactment
# --------------------------------------------------------------------------------------

FACIAL_REENACTMENT = Family(
    key="facial_reenactment",
    name="Facial reenactment",
    base_videos=4000,
    source_filter="face",
    source_groups=(
        SourceGroup("talking_expression", "Talking / expression", 1200, (
            "answering questions", "testifying", "news anchoring", "sign language interpreting",
            "laughing", "crying", "yawning", "shaking head")),
        SourceGroup("face_centered", "Face-centered activities", 700, (
            "applying cream", "brushing hair", "brushing teeth", "fixing hair", "getting a haircut",
            "shaving head", "waxing eyebrows", "sticking tongue out", "sneezing")),
        SourceGroup("eating", "Eating / drinking", 500, (
            "eating burger", "eating cake", "eating hotdog", "eating ice cream", "eating spaghetti",
            "drinking", "tasting food", "tasting beer")),
        SourceGroup("performance_dance", "Performance / dance", 700, (
            "singing", "beatboxing", "whistling", "dancing ballet", "belly dancing", "breakdancing",
            "salsa dancing", "tango dancing", "zumba")),
        SourceGroup("social", "Social interaction", 400, (
            "hugging", "kissing", "celebrating", "shaking hands", "tickling", "arm wrestling")),
        SourceGroup("everyday", "Everyday activities", 500, (
            "reading book", "texting", "using computer", "writing", "waiting in line", "unboxing")),
    ),
    pipelines=(
        Pipeline("face2face", "Face2Face", 1000,
                 "Explicit 3DMM + expression deformation transfer (classical, non-deep first-gen)"),
        Pipeline("fomm", "First Order Motion Model", 1000,
                 "Learned keypoints + local affine motion (implicit motion representation)"),
        Pipeline("pirenderer", "PIRenderer", 1000,
                 "3DMM parameters as control signals -> neural rendering (motion-driven preferred)"),
        Pipeline("face2face_rho", "Face2Face-rho", 1000,
                 "3DMM-assisted warping + hierarchical coarse-to-fine motion + U-shaped rendering"),
    ),
    metadata_fields=("target_video_id", "driving_video_id", "reenactment_model", "target_identity",
                     "driving_identity", "face_size", "yaw", "pitch", "roll", "expression_intensity",
                     "occlusion_level"),
    note="Kinetics-400 supplies target videos only; the driving motion must come from a separate source clip.",
)

# --------------------------------------------------------------------------------------
# 4. Lip-sync
# --------------------------------------------------------------------------------------

LIP_SYNC = Family(
    key="lip_sync",
    name="Lip-sync",
    base_videos=4000,
    source_filter="mouth",
    source_groups=(
        SourceGroup("talking_speech", "Talking / speech", 900, (
            "answering questions", "testifying", "news anchoring", "sign language interpreting")),
        SourceGroup("singing", "Singing / vocal performance", 600, (
            "singing", "beatboxing", "whistling")),
        SourceGroup("eating_mouth", "Eating / mouth motion", 600, (
            "eating burger", "eating cake", "eating hotdog", "eating ice cream", "eating spaghetti",
            "drinking", "tasting food")),
        SourceGroup("social", "Social interaction", 600, (
            "hugging", "kissing", "celebrating", "shaking hands", "laughing")),
        SourceGroup("performance", "Performance", 500, (
            "playing guitar", "playing piano", "playing violin", "playing drums", "dancing ballet",
            "belly dancing", "salsa dancing")),
        SourceGroup("everyday_face", "Everyday face-visible activity", 800, (
            "reading book", "texting", "using computer", "writing", "waiting in line")),
    ),
    pipelines=(
        Pipeline("wav2lip", "Wav2Lip", 1000,
                 "Audio-visual sync-expert-guided direct mouth synthesis (classic baseline)"),
        Pipeline("musetalk", "MuseTalk 1.5", 1000,
                 "Latent-space audio-conditioned inpainting (SD 1.4 U-Net backbone, single-step, not diffusion)"),
        Pipeline("videoretalking", "VideoReTalking", 1000,
                 "3-stage: canonical-expression normalization -> audio-driven lip-sync -> identity-aware enhancement"),
        Pipeline("sadtalker", "SadTalker", 1000,
                 "Audio -> 3D facial motion coefficients -> neural rendering"),
    ),
    metadata_fields=("lip_sync_model", "audio_source", "speaker_id", "language", "speech_duration",
                     "face_size", "mouth_visibility", "yaw", "pitch", "expression_intensity",
                     "occlusion_level"),
    note="Source filter priority order: mouth visibility -> face size -> pose -> temporal stability -> "
         "category balance.",
)

# --------------------------------------------------------------------------------------
# 5. Expression / attribute editing
# --------------------------------------------------------------------------------------

EXPRESSION_EDIT = Family(
    key="expression_attribute_editing",
    name="Expression / attribute editing",
    base_videos=3500,
    source_filter="face",
    source_groups=(
        SourceGroup("expression_heavy", "Expression-heavy", 700, (
            "laughing", "crying", "yawning", "sticking tongue out", "sneezing", "sniffing", "shaking head")),
        SourceGroup("beauty", "Face / beauty activities", 500, (
            "applying cream", "brushing hair", "brushing teeth", "fixing hair", "getting a haircut",
            "shaving head", "waxing eyebrows")),
        SourceGroup("eating", "Eating / drinking", 450, (
            "eating burger", "eating cake", "eating hotdog", "eating ice cream", "eating spaghetti",
            "drinking", "tasting food")),
        SourceGroup("talking_performance", "Talking / performance", 650, (
            "answering questions", "testifying", "news anchoring", "singing", "beatboxing", "whistling")),
        SourceGroup("everyday", "Everyday activities", 500, (
            "reading book", "texting", "using computer", "writing", "waiting in line", "unboxing")),
        SourceGroup("multi_social", "Multi-person / social", 700, (
            "hugging", "kissing", "celebrating", "shaking hands", "tickling", "arm wrestling")),
    ),
    pipelines=(
        Pipeline("ganimation", "GANimation", 1000,
                 "Facial Action Unit conditioned GAN, continuous expression magnitude"),
        Pipeline("styleganex", "StyleGANEX", 900,
                 "StyleGAN latent/feature manipulation, works on unaligned faces/video"),
        Pipeline("latent_transformer", "Latent Transformer", 800,
                 "Disentangled StyleGAN latent transformation, identity-preserving"),
        Pipeline("vq_facial_editing", "VQ Facial Performance Editing", 800,
                 "3D face model + vector-quantized StyleGAN local control (eyes/teeth) + coarse expression"),
    ),
    variants=(("smile_happiness", 500), ("sadness_crying", 350), ("anger", 350), ("surprise", 350),
              ("eye_gaze_modification", 300), ("mouth_expression_modification", 300), ("age", 350),
              ("hair_color", 350), ("facial_attributes", 350)),
    metadata_fields=("edit_model", "manipulation_type", "edit_magnitude", "identity_preserved",
                     "face_size", "yaw", "pitch", "occlusion_level"),
)

# --------------------------------------------------------------------------------------
# 6. Object insertion / removal
# --------------------------------------------------------------------------------------

OBJECT_EDIT = Family(
    key="object_insertion_removal",
    name="Object insertion / removal",
    base_videos=3500,
    source_filter="object",
    source_groups=(
        SourceGroup("hand_object", "Hand / object interaction", 700, (
            "playing guitar", "playing piano", "playing violin", "playing drums", "playing cards",
            "drawing", "painting", "knitting", "making jewelry", "folding paper")),
        SourceGroup("cooking", "Cooking / food objects", 550, (
            "cooking chicken", "cooking egg", "cooking on campfire", "baking cookies", "cutting watermelon",
            "cutting pineapple", "eating burger", "eating cake", "eating hotdog", "eating ice cream",
            "eating spaghetti")),
        SourceGroup("tools", "Tools / mechanical objects", 450, (
            "welding", "assembling computer", "using computer", "pushing car", "pushing cart",
            "repairing puncture", "changing oil", "changing wheel")),
        SourceGroup("sports_equipment", "Sports equipment", 600, (
            "playing basketball", "playing tennis", "playing golf", "playing volleyball", "punching bag",
            "catching or throwing baseball", "skateboarding", "surfing water")),
        SourceGroup("animals_objects", "Animals / objects", 450, (
            "feeding birds", "feeding fish", "feeding goats", "grooming horse", "milking cow",
            "petting cat", "walking the dog", "training dog")),
        SourceGroup("vehicles", "Vehicles / transportation", 400, (
            "driving car", "riding a bike", "motorcycling", "riding scooter", "riding mountain bike",
            "riding a segway", "sailing", "canoeing or kayaking")),
        SourceGroup("general_objects", "General indoor/outdoor objects", 350, (
            "cleaning floor", "cleaning windows", "making bed", "moving furniture", "setting table",
            "washing dishes", "unboxing")),
    ),
    pipelines=(
        Pipeline("propainter_object", "ProPainter", 900,
                 "Dual-domain flow/feature propagation + mask-guided sparse video Transformer"),
        Pipeline("object_wiper", "Object-WIPER", 900,
                 "Training-free removal via video-DiT inversion + visual-text attention + token replacement"),
        Pipeline("anyv2v_object", "AnyV2V", 850,
                 "First-frame image edit -> image-to-video model + temporal feature injection propagation"),
        Pipeline("videocomposer", "VideoComposer", 850,
                 "Multimodal conditional video diffusion (reference image + spatial/semantic conditioning)"),
    ),
    variants=(("object_removal", 1750), ("object_insertion", 1750)),
    metadata_fields=("edit_model", "operation", "object_class", "mask_area_frac", "track_length"),
    note="Exact operation allocation: ProPainter 700 removal / 200 insertion; Object-WIPER 900 removal / 0 "
         "insertion; AnyV2V 0 removal / 850 insertion; VideoComposer 150 removal / 700 insertion.",
)

# Per-model removal/insertion split from the document (model key -> {operation: videos}).
OBJECT_OPERATION_SPLIT: Dict[str, Dict[str, int]] = {
    "propainter_object": {"object_removal": 700, "object_insertion": 200},
    "object_wiper": {"object_removal": 900, "object_insertion": 0},
    "anyv2v_object": {"object_removal": 0, "object_insertion": 850},
    "videocomposer": {"object_removal": 150, "object_insertion": 700},
}

# --------------------------------------------------------------------------------------
# 7. Video inpainting
# --------------------------------------------------------------------------------------

VIDEO_INPAINTING = Family(
    key="video_inpainting",
    name="Video inpainting",
    base_videos=3000,
    source_filter="object",
    source_groups=(
        SourceGroup("people", "People / human activity", 600, (
            "walking the dog", "jogging", "running on treadmill", "dancing ballet", "hiking",
            "rock climbing", "reading book", "using computer", "cooking chicken", "cleaning floor")),
        SourceGroup("sports", "Sports", 500, (
            "playing basketball", "playing tennis", "shooting goal (soccer)", "playing volleyball",
            "catching or throwing baseball", "playing golf", "skiing slalom", "snowboarding",
            "surfing water")),
        SourceGroup("animals", "Animals", 400, (
            "walking the dog", "training dog", "petting cat", "feeding birds", "grooming horse",
            "riding or walking with horse", "riding elephant")),
        SourceGroup("vehicles_outdoor", "Vehicles / outdoor", 400, (
            "driving car", "motorcycling", "riding a bike", "riding scooter", "sailing",
            "canoeing or kayaking", "skiing slalom")),
        SourceGroup("indoor_scenes", "Indoor scenes", 350, (
            "cleaning windows", "making bed", "moving furniture", "setting table", "washing dishes",
            "bartending", "news anchoring")),
        SourceGroup("object_activity", "Object / activity scenes", 400, (
            "playing guitar", "playing piano", "painting", "drawing", "welding", "knitting",
            "making jewelry", "baking cookies")),
        SourceGroup("complex_outdoor", "Complex outdoor scenes", 350, (
            "rock climbing", "riding mountain bike", "skydiving", "paragliding", "scuba diving",
            "snorkeling", "water skiing")),
    ),
    pipelines=(
        Pipeline("propainter_inpaint", "ProPainter", 900,
                 "Dual-domain propagation + mask-guided sparse video Transformer"),
        Pipeline("e2fgvi_hq", "E2FGVI-HQ", 750,
                 "End-to-end flow completion + feature propagation + content hallucination, arbitrary resolution"),
        Pipeline("sttn", "STTN", 650, "Spatial-temporal Transformer attention"),
        Pipeline("fuseformer", "FuseFormer", 700, "Spatial-temporal feature fusion"),
    ),
    variants=(("person_removal", 600), ("large_foreground_removal", 500), ("small_medium_object_removal", 500),
              ("text_signage_overlay_removal", 400), ("background_region_reconstruction", 400),
              ("moving_object_removal", 350), ("complex_occlusion_mixed", 250)),
    metadata_fields=("inpaint_model", "inpaint_target", "mask_size_class", "mask_motion_pattern",
                     "mask_area_frac"),
    note="Mask control, independent of model: 20% small / 35% medium / 30% large / 15% very-large-irregular, "
         "with static/slow/fast/intermittent/partially-occluded motion patterns.",
)

# Mask size and motion distributions (independent of which model runs).
INPAINT_MASK_SIZES: Tuple[Tuple[str, float], ...] = (
    ("small", 0.20), ("medium", 0.35), ("large", 0.30), ("very_large_irregular", 0.15))
INPAINT_MASK_MOTION: Tuple[Tuple[str, float], ...] = (
    ("static", 0.20), ("slow", 0.25), ("fast", 0.20), ("intermittent", 0.20), ("partially_occluded", 0.15))

# --------------------------------------------------------------------------------------
# 8. Video-to-video transformation
# --------------------------------------------------------------------------------------

VIDEO_TO_VIDEO = Family(
    key="video_to_video",
    name="Video-to-video transformation",
    base_videos=2333,
    source_filter="any",
    source_groups=(
        SourceGroup("human_activities", "Human activities", 450, (
            "walking the dog", "jogging", "hiking", "reading book", "using computer", "cooking chicken",
            "cleaning floor", "dancing ballet")),
        SourceGroup("sports", "Sports", 400, (
            "playing basketball", "playing tennis", "shooting goal (soccer)", "playing volleyball",
            "skateboarding", "surfing water", "snowboarding")),
        SourceGroup("animals", "Animals", 350, (
            "walking the dog", "petting cat", "feeding birds", "grooming horse",
            "riding or walking with horse", "riding elephant")),
        SourceGroup("vehicles", "Vehicles / transportation", 300, (
            "driving car", "motorcycling", "riding a bike", "riding scooter", "sailing",
            "canoeing or kayaking")),
        SourceGroup("music_performance", "Music / performance", 300, (
            "playing guitar", "playing piano", "playing violin", "playing drums", "singing",
            "dancing ballet", "breakdancing")),
        SourceGroup("indoor_object", "Indoor / object activities", 283, (
            "painting", "drawing", "welding", "knitting", "making jewelry", "baking cookies",
            "assembling computer")),
        SourceGroup("outdoor_env", "Outdoor / environmental", 250, (
            "rock climbing", "skydiving", "paragliding", "scuba diving", "snorkeling", "snowmobiling")),
    ),
    pipelines=(
        Pipeline("tokenflow", "TokenFlow + Stable Diffusion", 600,
                 "Video inversion -> diffusion-feature propagation across frames, training-free"),
        Pipeline("insv2v", "InstructVid2Vid / InsV2V", 600,
                 "Instruction-conditioned video diffusion + optical-flow motion compensation"),
        Pipeline("anyv2v_style", "AnyV2V", 583,
                 "First-frame image edit -> image-to-video temporal propagation"),
        Pipeline("vid2vid", "NVIDIA vid2vid", 550,
                 "Conditional GAN video synthesis (non-diffusion baseline)"),
    ),
    variants=(("photorealistic_to_artistic", 600), ("photorealistic_to_stylized_cg", 600),
              ("appearance_material_transformation", 583), ("semantic_environmental_transformation", 550)),
    metadata_fields=("transform_model", "transformation_family", "prompt", "strength"),
    note="Applied to the whole video, not a localized region.",
)

# --------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------

FAMILY_LIST: Tuple[Family, ...] = (
    FACE_SWAP, FACIAL_REENACTMENT, LIP_SYNC, EXPRESSION_EDIT,
    OBJECT_EDIT, BACKGROUND, VIDEO_INPAINTING, VIDEO_TO_VIDEO,
)
FAMILIES: Dict[str, Family] = {f.key: f for f in FAMILY_LIST}

BASE_TARGETS: Dict[str, int] = {f.key: f.base_videos for f in FAMILY_LIST}

#: Families that receive a share of the +5,000 top-up needed for class balance.
FACE_FAMILIES: Tuple[str, ...] = ("face_swap", "facial_reenactment", "lip_sync",
                                  "expression_attribute_editing")
TOPUP_TOTAL = 5000
RUN_TOTAL = sum(BASE_TARGETS.values()) + TOPUP_TOTAL      # 33,333


def family_targets(total: int = RUN_TOTAL) -> Dict[str, int]:
    """Per-family video counts for a run total.

    At the default total the document's per-family numbers are kept and the surplus is split
    evenly across the four face-centric families. For any other total every family is scaled
    proportionally instead, so the function stays usable for smoke runs.
    """
    base = sum(BASE_TARGETS.values())
    if total == RUN_TOTAL:
        out = dict(BASE_TARGETS)
        for key, extra in zip(FACE_FAMILIES, apportion(TOPUP_TOTAL, [1] * len(FACE_FAMILIES))):
            out[key] += extra
        return out
    if total == base:
        return dict(BASE_TARGETS)
    keys = [f.key for f in FAMILY_LIST]
    return dict(zip(keys, apportion(total, [BASE_TARGETS[k] for k in keys])))


FAMILY_TARGETS: Dict[str, int] = family_targets()


# --------------------------------------------------------------------------------------
# plan construction
# --------------------------------------------------------------------------------------


def family_pipelines(family: Family, allowed: Optional[Set[str]] = None) -> List[Pipeline]:
    """The family's pipelines, optionally restricted to a set of runnable models.

    Six of the document's models have no runnable public release. Restricting to the rest and
    re-apportioning is what lets a family still hit its specified total: the family size and its
    source-content mix are preserved exactly, and only the per-pipeline split changes.
    """
    if allowed is None:
        return list(family.pipelines)
    kept = [p for p in family.pipelines if p.key in allowed]
    if not kept:
        raise ValueError(f"family {family.key!r} has no runnable pipeline in {sorted(allowed)}")
    return kept


def family_matrix(family: Family, target: int, allowed: Optional[Set[str]] = None
                  ) -> Tuple[List[int], List[int], List[List[int]]]:
    """(row totals, column totals, cells) for one family scaled to `target` videos.

    With `allowed`, the target is redistributed across only those pipelines, so the family still
    produces `target` videos even though some of its models cannot be run.
    """
    pipelines = family_pipelines(family, allowed)
    rows = apportion(target, [g.weight for g in family.source_groups])
    cols = apportion(target, [p.weight for p in pipelines])
    return rows, cols, cross_split(rows, cols)


def build_plan(targets: Dict[str, int] | None = None,
               allowed: Optional[Set[str]] = None) -> List[Cell]:
    """Flat list of every (family, source group, model) bucket with its video count."""
    targets = targets or FAMILY_TARGETS
    cells: List[Cell] = []
    for family in FAMILY_LIST:
        target = targets[family.key]
        pipelines = family_pipelines(family, allowed)
        _, _, grid = family_matrix(family, target, allowed)
        for i, group in enumerate(family.source_groups):
            for j, pipe in enumerate(pipelines):
                if grid[i][j] > 0:
                    cells.append(Cell(family.key, group.key, pipe.key, grid[i][j],
                                      family.source_filter, group.labels))
    return cells


def variant_allocation(family: Family, target: int) -> Dict[str, int]:
    """Secondary breakdown (manipulation type / target / transformation family) scaled to `target`."""
    if not family.variants:
        return {}
    names = [n for n, _ in family.variants]
    return dict(zip(names, apportion(target, [w for _, w in family.variants])))


def label_demand(targets: Dict[str, int] | None = None) -> Dict[str, int]:
    """How many source clips each Kinetics-400 label has to supply.

    A source group spreads its videos evenly over the labels it lists, so a label used by several
    groups accumulates demand from all of them. The caller multiplies this by a margin, because
    clips are lost to the qualification filters and some Kinetics classes are simply small.
    """
    targets = targets or FAMILY_TARGETS
    demand: Dict[str, float] = {}
    for family in FAMILY_LIST:
        rows, _, _ = family_matrix(family, targets[family.key])
        for group, row_total in zip(family.source_groups, rows):
            if not group.labels:
                continue
            share = row_total / len(group.labels)
            for label in group.labels:
                demand[label] = demand.get(label, 0.0) + share
    return {k: int(math.ceil(v)) for k, v in sorted(demand.items())}


def all_labels() -> List[str]:
    """Every distinct Kinetics-400 class name referenced by the spec."""
    seen: Dict[str, None] = {}
    for family in FAMILY_LIST:
        for group in family.source_groups:
            for label in group.labels:
                seen.setdefault(label, None)
    return sorted(seen)


def all_models() -> List[str]:
    return sorted({p.key for f in FAMILY_LIST for p in f.pipelines})


def validate() -> None:
    """Self-check: every family's grid matches its margins and the totals add up."""
    total = 0
    for family in FAMILY_LIST:
        target = FAMILY_TARGETS[family.key]
        rows, cols, grid = family_matrix(family, target)
        assert sum(rows) == target, f"{family.key}: row margin {sum(rows)} != {target}"
        assert sum(cols) == target, f"{family.key}: col margin {sum(cols)} != {target}"
        for i, r in enumerate(grid):
            assert sum(r) == rows[i], f"{family.key}: row {i} sums to {sum(r)}, want {rows[i]}"
        for j, c in enumerate(cols):
            assert sum(grid[i][j] for i in range(len(rows))) == c, f"{family.key}: col {j} mismatch"
        assert all(v >= 0 for r in grid for v in r), f"{family.key}: negative cell"
        if family.variants:
            assert sum(variant_allocation(family, target).values()) == target
        total += target
    assert total == sum(FAMILY_TARGETS.values()), "family targets do not add up"
    assert sum(BASE_TARGETS.values()) == 28333, "base spec total drifted from the document"
    assert sum(c.videos for c in build_plan()) == total, "plan does not cover the target"


def _print_plan() -> None:
    print(f"AI-Edited regeneration plan - {sum(FAMILY_TARGETS.values()):,} videos "
          f"(document base {sum(BASE_TARGETS.values()):,} + {TOPUP_TOTAL:,} top-up)\n")
    for family in FAMILY_LIST:
        target = FAMILY_TARGETS[family.key]
        base = BASE_TARGETS[family.key]
        rows, cols, grid = family_matrix(family, target)
        extra = f"  (+{target - base} over the document's {base})" if target != base else ""
        print(f"== {family.name}: {target:,} videos{extra}")
        width = max(len(g.name) for g in family.source_groups) + 2
        header = "".join(f"{p.key:>22}" for p in family.pipelines)
        print(f"{'source group':<{width}}{header}{'total':>10}")
        for i, group in enumerate(family.source_groups):
            body = "".join(f"{grid[i][j]:>22}" for j in range(len(cols)))
            print(f"{group.name:<{width}}{body}{rows[i]:>10}")
        print(f"{'total':<{width}}" + "".join(f"{c:>22}" for c in cols) + f"{target:>10}")
        if family.variants:
            alloc = variant_allocation(family, target)
            print("   variants: " + ", ".join(f"{k}={v}" for k, v in alloc.items()))
        print()
    print(f"distinct Kinetics-400 labels referenced: {len(all_labels())}")
    print(f"distinct manipulation models: {len(all_models())}")


if __name__ == "__main__":
    validate()
    _print_plan()
    print("\nspec self-check OK")
