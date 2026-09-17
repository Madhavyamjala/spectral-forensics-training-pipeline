"""
Fitting the generation run into a wall-clock budget.

Wiring every available substitute brings coverage to 80.5% of the plan, but costs roughly 523
GPU-hours - about 7.3 days on three GPUs, before any training happens. That does not fit a
one-week end-to-end target, so the set of models actually run has to be chosen, and chosen
deliberately rather than by whichever ones the scheduler reaches before the deadline fires.

`deadline_hours` alone is not enough: it stops the run mid-way through whatever group happens to
be in flight, which biases the dataset toward the models that sort earliest. This module decides
up front instead.

Selection is two-phase, because raw efficiency (videos per GPU-hour) picks badly on its own -
it would spend the whole budget on the cheap face models and leave the video-to-video family
with nothing at all:

    1. **diversity floor** - admit the cheapest model in every family, then the second cheapest,
       so each of the eight families keeps at least two distinct manipulation mechanisms if the
       budget allows it. A family rendered by one model teaches the detector one artifact class.
    2. **efficiency fill** - spend whatever is left on the remaining models, most videos per
       GPU-hour first.

Input : a wall-clock budget and a GPU count.
Output: the models to run, the models to skip, and the coverage that results.

    python -m csf.generation.budget --hours 84 --gpus 3
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from csf.generation import spec as S
from csf.generation.adapters import ADAPTERS, videos_per_model
from csf.logging_utils import get_logger

log = get_logger("generation.budget")

#: How many distinct mechanisms to guarantee per family before spending on volume.
DIVERSITY_FLOOR = 2


@dataclass
class ModelCost:
    key: str
    family: str
    runs_as: str
    videos: int
    gpu_hours: float

    @property
    def efficiency(self) -> float:
        """Return videos produced per GPU-hour for this model."""
        return self.videos / self.gpu_hours if self.gpu_hours > 0 else float("inf")


@dataclass
class BudgetPlan:
    budget_gpu_hours: float
    gpus: int
    selected: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    unwired: List[str] = field(default_factory=list)
    gpu_hours_used: float = 0.0
    videos: int = 0
    total_videos: int = 0

    @property
    def wall_clock_hours(self) -> float:
        """Return estimated wall-clock time at the configured GPU count."""
        return self.gpu_hours_used / max(1, self.gpus)

    @property
    def coverage(self) -> float:
        """Return the selected fraction of all planned videos."""
        return self.videos / self.total_videos if self.total_videos else 0.0

    def to_dict(self) -> Dict[str, object]:
        """Serialize the budget plan for reporting."""
        return {"budget_gpu_hours": round(self.budget_gpu_hours, 1), "gpus": self.gpus,
                "gpu_hours_used": round(self.gpu_hours_used, 1),
                "wall_clock_hours": round(self.wall_clock_hours, 1),
                "videos": self.videos, "total_videos": self.total_videos,
                "coverage": round(self.coverage, 4),
                "selected": self.selected, "skipped": self.skipped, "unwired": self.unwired}


#: Renderers whose jobs saturate the GPU's SMs - diffusion samplers. Extra concurrent workers
#: buy them little, whereas the small GAN/ONNX nets spend most of a job decoding video and
#: detecting faces on the CPU and scale nearly linearly.
SATURATING = {"dreamid_v", "vace", "bg_svd_video", "bg_flux_image", "tokenflow", "latentsync",
              "diffueraser", "reface", "musetalk"}
SPEEDUP_SATURATING = 1.3
SPEEDUP_LATENCY_BOUND = 3.5


def effective_speedup(model: str, workers_per_gpu: int) -> float:
    """Throughput multiplier from running `workers_per_gpu` workers of `model` on one card."""
    if workers_per_gpu <= 1:
        return 1.0
    a = ADAPTERS.get(model)
    runs = a.runs if a is not None else model
    ceiling = SPEEDUP_SATURATING if (runs in SATURATING or model in SATURATING) \
        else SPEEDUP_LATENCY_BOUND
    return min(float(workers_per_gpu), ceiling)


def model_costs(targets: Optional[Dict[str, int]] = None) -> List[ModelCost]:
    """Build per-model cost records from the planned video targets."""
    per_model = videos_per_model(targets)
    out = []
    for key, n in per_model.items():
        a = ADAPTERS[key]
        if not a.implemented:
            continue
        out.append(ModelCost(key=key, family=a.family, runs_as=a.runs, videos=n,
                             gpu_hours=n * a.cost_s / 3600.0))
    return out


def plan(budget_gpu_hours: float, gpus: int = 3, targets: Optional[Dict[str, int]] = None,
         diversity_floor: int = DIVERSITY_FLOOR,
         pinned: Sequence[str] = ()) -> BudgetPlan:
    """Choose which models to run inside `budget_gpu_hours`."""
    targets = targets or S.FAMILY_TARGETS
    per_model = videos_per_model(targets)
    costs = model_costs(targets)
    by_family: Dict[str, List[ModelCost]] = {}
    for c in costs:
        by_family.setdefault(c.family, []).append(c)
    for fam in by_family:
        by_family[fam].sort(key=lambda c: c.gpu_hours)       # cheapest first

    result = BudgetPlan(budget_gpu_hours=budget_gpu_hours, gpus=gpus,
                        total_videos=sum(per_model.values()))
    result.unwired = sorted(k for k in per_model if not ADAPTERS[k].implemented)
    chosen: Dict[str, ModelCost] = {}
    spent = 0.0

    def admit(c: ModelCost) -> bool:
        """Add a model to the plan and update its accumulated totals."""
        nonlocal spent
        if c.key in chosen:
            return True
        if spent + c.gpu_hours > budget_gpu_hours:
            return False
        chosen[c.key] = c
        spent += c.gpu_hours
        return True

    # operator-pinned models go in first, budget or not
    for key in pinned:
        c = next((c for c in costs if c.key == key), None)
        if c is not None and c.key not in chosen:
            chosen[c.key] = c
            spent += c.gpu_hours

    # phase 1: diversity floor - cheapest first within each family, but counting *renderers*,
    # not slots. Two slots that both run VACE are one mechanism, so admitting both would satisfy
    # the floor on paper while leaving the family with a single artifact class.
    for rank in range(max(1, diversity_floor)):
        for fam in sorted(by_family):
            seen = {chosen[k].runs_as for k in chosen if chosen[k].family == fam}
            if len(seen) > rank:
                continue
            for c in by_family[fam]:
                if c.key not in chosen and c.runs_as not in seen:
                    admit(c)
                    break

    # phase 2: spend the rest on volume
    for c in sorted(costs, key=lambda c: -c.efficiency):
        admit(c)

    result.selected = sorted(chosen)
    result.skipped = sorted(c.key for c in costs if c.key not in chosen)
    result.gpu_hours_used = round(spent, 2)
    result.videos = sum(chosen[k].videos for k in chosen)
    return result


def family_breakdown(p: BudgetPlan, targets: Optional[Dict[str, int]] = None
                     ) -> Dict[str, Dict[str, object]]:
    """Per family: videos kept, videos dropped, and how many distinct renderers survive."""
    targets = targets or S.FAMILY_TARGETS
    per_model = videos_per_model(targets)
    out: Dict[str, Dict[str, object]] = {}
    for family in S.FAMILY_LIST:
        keys = [pipe.key for pipe in family.pipelines]
        kept = [k for k in keys if k in p.selected]
        out[family.key] = {
            "planned": targets[family.key],
            "kept": sum(per_model[k] for k in kept),
            "mechanisms": len({ADAPTERS[k].runs for k in kept}),
            "models": sorted({ADAPTERS[k].runs for k in kept}),
        }
    return out


def wall_clock_estimate(gpus: int, workers_per_gpu: int, allowed: Optional[set] = None,
                        targets: Optional[Dict[str, int]] = None) -> Dict[str, object]:
    """Wall-clock hours for a model set, accounting for per-GPU concurrency.

    With `allowed`, each family's target is re-apportioned across the runnable models (the
    `reallocate_unfillable` behaviour), so the totals reflect what will actually be generated.
    """
    targets = targets or S.FAMILY_TARGETS
    counts: Dict[str, int] = {}
    if allowed is None:
        counts = {k: v for k, v in videos_per_model(targets).items() if ADAPTERS[k].implemented}
    else:
        for family in S.FAMILY_LIST:
            pipelines = S.family_pipelines(family, allowed)
            _, cols, _ = S.family_matrix(family, targets[family.key], allowed)
            for pipe, n in zip(pipelines, cols):
                counts[pipe.key] = n
    serial = sum(n * ADAPTERS[k].cost_s / 3600.0 for k, n in counts.items())
    effective = sum(n * ADAPTERS[k].cost_s / 3600.0 / effective_speedup(k, workers_per_gpu)
                    for k, n in counts.items())
    return {"videos": sum(counts.values()),
            "gpu_hours_serial": round(serial, 1),
            "gpu_hours_effective": round(effective, 1),
            "gpus": gpus, "workers_per_gpu": workers_per_gpu,
            "wall_clock_hours": round(effective / max(1, gpus), 1),
            "wall_clock_days": round(effective / max(1, gpus) / 24.0, 2)}


def _main() -> int:
    """Print the adapter coverage and estimated generation cost report."""
    import argparse

    ap = argparse.ArgumentParser(description="Fit the generation run into a wall-clock budget")
    ap.add_argument("--hours", type=float, default=84.0, help="wall-clock hours available")
    ap.add_argument("--gpus", type=int, default=3)
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--reallocate", action="store_true",
                    help="re-apportion unfillable slots onto runnable models (full 33,333)")
    ap.add_argument("--floor", type=int, default=DIVERSITY_FLOOR,
                    help="distinct mechanisms to guarantee per family")
    ap.add_argument("--pin", default="", help="comma-separated models to always include")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.reallocate:
        allowed = {k for k, a in ADAPTERS.items() if a.implemented}
        est = wall_clock_estimate(args.gpus, args.workers_per_gpu, allowed=allowed)
        print(f"Reallocated plan (every family reaches its specified size)\n")
        print(f"  videos                 : {est['videos']:,}")
        print(f"  cost, 1 worker/GPU     : {est['gpu_hours_serial']:,} GPU-hours")
        print(f"  cost, {args.workers_per_gpu} workers/GPU    : "
              f"{est['gpu_hours_effective']:,} GPU-hours effective")
        print(f"  wall clock on {args.gpus} GPUs   : {est['wall_clock_hours']:,} h "
              f"= {est['wall_clock_days']} days")
        return 0

    budget = args.hours * args.gpus
    pinned = [s.strip() for s in args.pin.split(",") if s.strip()]
    p = plan(budget, args.gpus, diversity_floor=args.floor, pinned=pinned)

    if args.json:
        print(json.dumps({"plan": p.to_dict(), "families": family_breakdown(p)}, indent=2))
        return 0

    print(f"Budget: {args.hours:.0f} h wall clock x {args.gpus} GPUs = {budget:.0f} GPU-hours\n")
    costs = {c.key: c for c in model_costs()}
    print(f"{'model':<22}{'runs as':<20}{'videos':>8}{'GPU-h':>9}  status")
    for key in sorted(costs, key=lambda k: -costs[k].efficiency):
        c = costs[key]
        status = "run" if key in p.selected else "SKIP (over budget)"
        print(f"{key:<22}{c.runs_as:<20}{c.videos:>8}{c.gpu_hours:>9.1f}  {status}")
    for key in p.unwired:
        print(f"{key:<22}{'-':<20}{videos_per_model()[key]:>8}{0.0:>9.1f}  not wired")

    print(f"\nselected : {len(p.selected)} models, {p.videos:,} videos "
          f"({p.coverage:.1%} of the plan)")
    print(f"cost     : {p.gpu_hours_used:,.0f} GPU-hours = {p.wall_clock_hours:.0f} h "
          f"wall clock on {p.gpus} GPUs")
    print(f"\n{'family':<32}{'kept':>8}{'planned':>9}{'mechanisms':>12}  renderers")
    for fam, info in family_breakdown(p).items():
        print(f"{fam:<32}{info['kept']:>8}{info['planned']:>9}{info['mechanisms']:>12}  "
              f"{', '.join(info['models']) or '-'}")
    fams = family_breakdown(p)
    empty = [f for f, i in fams.items() if i["mechanisms"] == 0]
    thin = [f for f, i in fams.items() if i["mechanisms"] == 1]
    if empty:
        print(f"\nWARNING: {empty} would produce no videos at all at this budget.")
    if thin:
        print(f"\nNOTE: {thin} would be rendered by a single model, so that family teaches one "
              f"artifact class. Raise the budget or pin a second renderer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
