"""The world a ladder rung runs in: belt, segments, arms with their bins, the item mix and how it is fed.

Everything here is a frozen dataclass so a scenario is one value that can be hashed, printed and diffed. Units are SI
(m, s, m/s); the report converts at the display edge.

WHY these simplifications (Daily Note, Nov 11 2026, "Conveyor belt speed control and FIFO object scheduling"):

* Items are points on a flat 2D belt (x along, y across). No stacking, no tumbling: the question is who can reach
  what, when, not contact physics (bam_pick_worlds owns that).
* "If an arm can pick an item it can place it": an arm can pick a category only if one of its bins holds it.
* Grasp success is a per-item probability the scheduler knows, and a hard belt-speed line above which the arm does
  not try at all. Zach: "there is likely quite a sharp line there, so probably simpler to just model as a max belt
  speed the arms can handle".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Category:
    name: str
    share: float  # share of items by count
    value: float = 1.0  # what one successful pick of it is worth (v1: all equal)
    p_grasp: float = 0.9  # per-attempt grasp success at a trackable belt speed
    color: str = "#888888"


@dataclass(frozen=True, slots=True)
class Bin:
    category: str
    x: float  # world position of the drop point
    y: float


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One pick arm. Reach is a disc of ``reach_m`` around the base (x, y) at belt height.

    Two timing models:

    * travel (default): trapezoidal point-to-point moves at ``tool_speed_m_s`` / ``tool_accel_m_s2``, plus a grasp
      dwell and a release dwell. The tool intercepts a moving item (it tracks it during the grasp dwell).
    * fixed (``fixed_cycle_s`` set): every attempt takes ``fixed_reach_s`` from commit to contact and the arm is
      busy for ``fixed_cycle_s``. It reproduces the facility model's F1 line_sim, which tests use as the oracle.
    """

    name: str
    x: float
    y: float
    bins: tuple[Bin, ...]
    reach_m: float = 0.55
    tool_speed_m_s: float = 0.8  # facility model UR5e (scenarios.py): 0.8 m/s, URScript default accel 1.2 m/s^2
    tool_accel_m_s2: float = 1.2
    grasp_dwell_s: float = 0.30
    release_dwell_s: float = 0.20
    v_track_max_m_s: float = 0.5  # belt speed above which the arm cannot grasp (hard line)
    # WHY a re-plan budget: the tool tracks its item (visual servoing). If the belt changed speed after the arm
    # committed, it re-aims from where it is rather than giving up; it gives up when the item leaves its reach or
    # this long after the first planned contact.
    replan_budget_s: float = 3.0
    # A tool this close to its item at the planned contact closes the rest during the grasp dwell (it is tracking).
    contact_tol_m: float = 0.06
    fixed_cycle_s: float | None = None
    fixed_reach_s: float | None = None
    # Optional rectangular window along x instead of the reach disc (fixed model / oracle only).
    window_x: tuple[float, float] | None = None

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(b.category for b in self.bins)

    def bin_for(self, category: str) -> Bin:
        for b in self.bins:
            if b.category == category:
                return b
        raise KeyError(category)

    def travel_s(self, distance_m: float) -> float:
        return travel_time(distance_m, self.tool_speed_m_s, self.tool_accel_m_s2)


def travel_time(distance_m, speed, accel):
    """Trapezoidal (or triangular) point-to-point move time; works on floats and numpy arrays."""
    if isinstance(distance_m, float | int):
        d = abs(distance_m)
        if d >= speed * speed / accel:
            return d / speed + speed / accel
        return 2.0 * math.sqrt(d / accel)
    import numpy as np

    d = np.abs(distance_m)
    d_switch = speed * speed / accel
    tri = 2.0 * np.sqrt(d / accel)
    trap = d / speed + speed / accel
    out = np.where(d >= d_switch, trap, tri)
    return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True, slots=True)
class BeltSpec:
    length_m: float
    width_m: float = 0.6
    # Segment boundaries strictly inside (0, length). One segment = one belt; N belts in a row each get a speed.
    boundaries_m: tuple[float, ...] = ()
    v_max_m_s: float = 1.0
    v_min_m_s: float = -0.3  # negative = the belt can run backwards
    accel_m_s2: float = 1.0
    circulating: bool = False  # items leaving the end go back to the head (Zach: "they can just teleport back")
    max_passes: int = 3
    loop_delay_s: float = 4.0

    @property
    def n_segments(self) -> int:
        return len(self.boundaries_m) + 1

    def segment_edges(self) -> tuple[float, ...]:
        return (0.0, *self.boundaries_m, self.length_m)


@dataclass(frozen=True, slots=True)
class Supply:
    """How items reach the head of the belt.

    * ``hopper``: an unlimited pile meters items onto the belt as it moves, at ``density_per_m`` items per metre of
      belt travel. Running faster brings more items per second: the throughput question.
    * ``stream``: an upstream line delivers ``rate_per_s`` items per second whatever the belt does. A belt that is
      too slow builds a backlog at the head: the recovery question.

    ``burstiness`` in [0, 1): the share of items that arrive in a clump right behind the previous one (spacing
    ``min_gap_m``); the rest are spread out so the mean density is unchanged. 0 = Poisson.
    """

    mode: str = "hopper"
    density_per_m: float = 4.0
    rate_per_s: float = 1.0
    burstiness: float = 0.0
    min_gap_m: float = 0.04
    y_margin_m: float = 0.05


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    belt: BeltSpec
    arms: tuple[ArmSpec, ...]
    categories: tuple[Category, ...]
    supply: Supply
    p_grasp_spread: float = 0.0  # per-item grasp probability ~ p_cat +/- U(spread): the scheduler knows it
    note: str = ""

    def category_index(self) -> dict[str, int]:
        return {c.name: i for i, c in enumerate(self.categories)}

    def with_(self, **kwargs) -> Scenario:
        return replace(self, **kwargs)

    def capacity_ppm(self) -> float:
        """Fluid ceiling on successful picks/min: every arm busy all the time on its shortest possible cycle.

        Shortest cycle = grasp dwell + release dwell + a round trip from the belt edge nearest its closest bin. It
        ignores the belt entirely, so no policy can beat it; it is loose by design (a ceiling, not a target).
        """
        total = 0.0
        for arm in self.arms:
            if arm.fixed_cycle_s is not None:
                cycle = arm.fixed_cycle_s
            else:
                d_bin = min(_bin_to_belt_m(b, self.belt.width_m) for b in arm.bins)
                cycle = arm.grasp_dwell_s + arm.release_dwell_s + 2.0 * arm.travel_s(max(d_bin, 1e-3))
            p = max(c.p_grasp for c in self.categories if c.name in arm.categories)
            total += 60.0 * p / cycle
        return total


def _bin_to_belt_m(b: Bin, width: float) -> float:
    if 0.0 <= b.y <= width:
        return 0.0
    return min(abs(b.y), abs(b.y - width))


def reach_chord_m(arm: ArmSpec, y: float) -> float:
    """Half-length of the reach disc's chord at belt lateral position y (0 if out of reach)."""
    dy = y - arm.y
    return math.sqrt(max(arm.reach_m**2 - dy * dy, 0.0))


__all__ = [
    "ArmSpec",
    "BeltSpec",
    "Bin",
    "Category",
    "Scenario",
    "Supply",
    "reach_chord_m",
    "travel_time",
]

