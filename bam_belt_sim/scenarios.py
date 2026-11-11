"""Named worlds for the two ladders. Each varies one thing from its neighbour where it can.

The daily note's axes: number of items and their arrival pattern (density, burstiness, stream vs hopper), number of
arms, categories per arm (bins it can reach), bin placement, values, and whether the line recirculates.

Arm numbers are the facility model's UR5e (bam_facility_model/scenarios.py: 0.8 m/s, 1.2 m/s^2 default URScript
accel, grasp success 0.9, ~1 m window from 850 mm reach), so the picks/min land in the 10-20 cobots manage in MRFs.
"""

from __future__ import annotations

from .config import ArmSpec, BeltSpec, Bin, Category, Scenario, Supply

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
           "#7f7f7f"]

HEAD_M, PITCH_M, TAIL_M, WIDTH_M = 1.0, 1.5, 0.6, 0.6
BIN_OFFSET_Y = 0.15


def bin_slots(x: float, width: float, n: int) -> list[tuple[float, float]]:
    """Drop points around an arm, nearest first: both sides of the belt, fanning out along x."""
    slots = []
    for dx in (0.0, -0.3, 0.3, -0.6, 0.6, -0.9, 0.9):
        slots.append((x + dx, -BIN_OFFSET_Y))
        slots.append((x + dx, width + BIN_OFFSET_Y))
    return slots[:n]


def uniform_mix(k: int, value: float = 1.0, p: float = 0.9) -> tuple[Category, ...]:
    return tuple(Category(f"c{j}", 1.0 / k, value, p, PALETTE[j % len(PALETTE)]) for j in range(k))


def mrf_mix(gold: bool = False) -> tuple[Category, ...]:
    """A mixed-recycling-like stream by item count, with per-item values (USD, assumed: price/t x typical mass).

    Residue has no bin anywhere: it is on the belt, takes space, and nobody may pick it. The gold nugget (0.3 %) is
    worth 50x the best commodity: the daily note's "If a gold nugget is passing by, make sure to grab that".
    """
    cats = [
        Category("cardboard", 0.30, 0.010, 0.92, "#8c564b"),  # ~100 g at ~$100/t
        Category("PET", 0.16, 0.010, 0.90, "#1f77b4"),  # ~25 g at ~$400/t
        Category("HDPE", 0.08, 0.024, 0.90, "#17becf"),  # ~40 g at ~$600/t
        Category("aluminium", 0.06, 0.021, 0.85, "#7f7f7f"),  # ~14 g at ~$1500/t
        Category("film", 0.12, 0.0005, 0.70, "#bcbd22"),  # ~10 g at ~$50/t, hard to grasp
        Category("steel", 0.05, 0.010, 0.92, "#2ca02c"),  # ~50 g at ~$200/t
        Category("residue", 0.23, 0.0, 0.90, "#cccccc"),
    ]
    if gold:
        cats.append(Category("gold", 0.003, 1.0, 0.9, "#ffbf00"))
    return tuple(cats)


def line(
    name: str,
    n_arms: int,
    categories: tuple[Category, ...],
    supply: Supply,
    bins_per_arm: int = 2,
    copies: int = 1,
    segmented: bool = True,
    circulating: bool = False,
    targets: list[str] | None = None,
    note: str = "",
    **arm_kw,
) -> Scenario:
    """A straight line of ``n_arms`` at ``PITCH_M``, categories dealt round-robin onto arms' bins.

    ``copies`` > 1 gives each category a bin at that many arms (overlap: a downstream arm gets a second chance).
    ``segmented`` puts a belt boundary between neighbouring arms (and one after the head), so S4 has belts to run;
    the single-belt rungs command every segment the same speed, which is the same as one belt.
    """
    length = HEAD_M + n_arms * PITCH_M + TAIL_M
    xs = [HEAD_M + PITCH_M * (k + 0.5) for k in range(n_arms)]
    targets = targets or [c.name for c in categories if c.value > 0 or c.name.startswith("c")]
    assign: list[list[str]] = [[] for _ in range(n_arms)]
    slot = 0
    for _copy in range(copies):
        for cat in targets:
            for _ in range(n_arms):
                k = slot % n_arms
                slot += 1
                if cat not in assign[k] and len(assign[k]) < bins_per_arm:
                    assign[k].append(cat)
                    break
    arms = []
    for k, x in enumerate(xs):
        slots = bin_slots(x, WIDTH_M, max(len(assign[k]), 1))
        bins = tuple(Bin(cat, sx, sy) for cat, (sx, sy) in zip(assign[k], slots, strict=False))
        arms.append(ArmSpec(f"arm{k + 1}", x, WIDTH_M / 2, bins, **arm_kw))
    boundaries = ()
    if segmented and n_arms > 1:
        boundaries = (HEAD_M,) + tuple(HEAD_M + PITCH_M * k for k in range(1, n_arms))
    elif segmented:
        boundaries = (HEAD_M,)
    belt = BeltSpec(length_m=length, width_m=WIDTH_M, boundaries_m=boundaries, circulating=circulating)
    return Scenario(name, belt, tuple(arms), categories, supply, note=note)


def speed_benchmark() -> dict[str, Scenario]:
    """The belt-speed ladder's worlds (hopper-fed: running faster brings more items per second)."""
    six = uniform_mix(6)
    return {
        "A 1 arm, 2 cats": line("A 1 arm, 2 cats", 1, uniform_mix(2), Supply(density_per_m=4.0), bins_per_arm=2),
        "B 3 arms, 6 cats": line("B 3 arms, 6 cats", 3, six, Supply(density_per_m=4.0)),
        "C bursty": line("C bursty", 3, six, Supply(density_per_m=4.0, burstiness=0.6)),
        "D sparse 1/m": line("D sparse 1/m", 3, six, Supply(density_per_m=1.0)),
        "E 6 arms, overlap": line("E 6 arms, overlap", 6, six, Supply(density_per_m=8.0), copies=2),
        "F MRF mix": line("F MRF mix", 3, mrf_mix(), Supply(density_per_m=5.0)),
        "G stream 0.9/s": line("G stream 0.9/s", 3, six, Supply(mode="stream", rate_per_s=0.9)),
    }


def circulation_benchmark() -> dict[str, Scenario]:
    base = speed_benchmark()
    out = {}
    for key in ("B 3 arms, 6 cats", "D sparse 1/m"):
        sc = base[key]
        name = key + " + recirc"
        out[name] = sc.with_(name=name, belt=_replace_belt(sc.belt, circulating=True))
    return out


def _replace_belt(belt: BeltSpec, **kw) -> BeltSpec:
    from dataclasses import replace

    return replace(belt, **kw)


def pick_benchmark() -> dict[str, Scenario]:
    """The scheduling ladder's worlds: a constant-speed belt fed faster than the arms can clear (they must choose)."""
    six = uniform_mix(6)
    gold = mrf_mix(gold=True)
    return {
        "PA 3 arms, 6 equal cats": line("PA 3 arms, 6 equal cats", 3, six, Supply(mode="stream", rate_per_s=1.5),
                                        segmented=False),
        "PB MRF values + gold": line("PB MRF values + gold", 3, gold, Supply(mode="stream", rate_per_s=1.5),
                                     bins_per_arm=3, segmented=False),
        "PC overlap (2 arms/cat)": line("PC overlap (2 arms/cat)", 3, six, Supply(mode="stream", rate_per_s=1.5),
                                        bins_per_arm=4, copies=2, segmented=False),
        "PD 1 arm, 6 bins": line("PD 1 arm, 6 bins", 1, six, Supply(mode="stream", rate_per_s=0.8), bins_per_arm=6,
                                 segmented=False),
    }


def balanced_line(name: str, n_arms: int, categories: tuple[Category, ...], supply: Supply, bins_per_arm: int = 2,
                  duplicate_top: int = 0, **kw) -> Scenario:
    """Bins dealt by load, not by list order: longest-processing-time-first on share / p_grasp, each arm's heaviest
    category in its nearest bin slot. ``duplicate_top`` > 0 also gives the heaviest categories a second bin on the
    next arm with room (a downstream second chance where the load is)."""
    base = line(name, n_arms, categories, supply, bins_per_arm=bins_per_arm, **kw)
    targets = [c for c in categories if c.value > 0 or c.name.startswith("c")]
    load = [0.0] * n_arms
    assign: list[list[Category]] = [[] for _ in range(n_arms)]
    for c in sorted(targets, key=lambda c: -c.share / c.p_grasp):
        k = min((k for k in range(n_arms) if len(assign[k]) < bins_per_arm), key=lambda k: load[k])
        assign[k].append(c)
        load[k] += c.share / c.p_grasp
    for c in sorted(targets, key=lambda c: -c.share / c.p_grasp)[:duplicate_top]:
        home = next(k for k in range(n_arms) if c in assign[k])
        for k in list(range(home + 1, n_arms)) + list(range(0, home)):
            if c not in assign[k] and len(assign[k]) < bins_per_arm + 1:
                assign[k].append(c)
                break
    arms = []
    for k, arm in enumerate(base.arms):
        cats = sorted(assign[k], key=lambda c: -c.share)
        slots = bin_slots(arm.x, WIDTH_M, len(cats))
        bins = tuple(Bin(c.name, sx, sy) for c, (sx, sy) in zip(cats, slots, strict=True))
        arms.append(ArmSpec(arm.name, arm.x, arm.y, bins, **kw.get("arm_kw", {})))
    return base.with_(arms=tuple(arms))


def layout_benchmark() -> dict[str, Scenario]:
    """Layout, the daily note's first question in miniature: same MRF line, bins dealt three ways."""
    mix = mrf_mix()
    sup = Supply(density_per_m=5.0)
    return {
        "L0 MRF round-robin bins": line("L0 MRF round-robin bins", 3, mix, sup),
        "L1 MRF load-balanced bins": balanced_line("L1 MRF load-balanced bins", 3, mix, sup),
        "L2 balanced + top-2 doubled": balanced_line("L2 balanced + top-2 doubled", 3, mix, sup, duplicate_top=2),
    }
