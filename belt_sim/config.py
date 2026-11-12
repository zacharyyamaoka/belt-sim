"""The world a ladder rung runs in: belt, segments, gantry arms with their bins, the item mix and how it is fed.

Everything here is a frozen dataclass so a scenario is one value that can be hashed, printed and diffed. Units are SI
(m, s, m/s); reports convert at the display edge.

WHY these simplifications (the design note this package answers: "conveyor belt speed control and FIFO object
scheduling"):

* Items are points on a flat 2D belt (x along, y across). No stacking, no tumbling: the question is who can reach
  what, when, not contact physics.
* If an arm can pick an item it can place it: an arm may pick a category only if one of its bins holds it.
* Grasp success is a per-item probability the scheduler knows, plus a hard belt-speed line above which the arm does
  not try at all ("there is likely quite a sharp line there, so model it as a max belt speed the arms can handle").
* The arm is a Cartesian gantry: an x carriage riding along the belt and a y carriage across it, each with its own
  max speed and acceleration and a trapezoidal velocity profile. A move takes as long as its slower axis.
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
    """One Cartesian (gantry) pick arm centred at ``x`` along the belt.

    Workspace: x in [x - stroke_x_m/2, x + stroke_x_m/2], y in [y_min_m, y_max_m] (the whole belt width plus the bins
    on both edges). The gripper drops (z) to grasp; that plunge, close and lift is the grasp dwell.

    Two timing models:

    * gantry (default): the x and y axes move at once, each on a trapezoidal profile (``vx_max`` / ``ax_max``,
      ``vy_max`` / ``ay_max``); a move lasts as long as the slower axis. The tool intercepts a moving item and tracks
      it through the grasp dwell.
    * fixed (``fixed_cycle_s`` set): every attempt takes ``fixed_reach_s`` from commit to contact and the arm is
      busy for ``fixed_cycle_s``. It reproduces the event-driven reference sim (tests/oracle_line_sim.py).
    """

    name: str
    x: float
    bins: tuple[Bin, ...]
    stroke_x_m: float = 1.0
    y_min_m: float = -0.3
    y_max_m: float = 0.9
    # WHY these defaults (assumed, not a datasheet): a belt-driven pick gantry - 1.5 m/s and 6 m/s^2 per axis puts a
    # 0.4 m move at ~0.5 s and the arm near 35-45 picks/min, the range gantry/delta sorters are sold at.
    vx_max: float = 1.5
    ax_max: float = 6.0
    vy_max: float = 1.5
    ay_max: float = 6.0
    grasp_dwell_s: float = 0.35  # z plunge, close, lift
    release_dwell_s: float = 0.15
    v_track_max_m_s: float = 0.5  # belt speed above which the arm cannot grasp (hard line)
    # WHY a re-plan budget: the tool tracks its item (visual servoing). If the belt changed speed after the arm
    # committed, it re-aims from where it is rather than giving up; it gives up when the item leaves its reach or
    # this long after the first planned contact.
    replan_budget_s: float = 3.0
    # A tool this close to its item at the planned contact closes the rest during the grasp dwell (it is tracking).
    contact_tol_m: float = 0.06
    fixed_cycle_s: float | None = None
    fixed_reach_s: float | None = None
    # Optional x window overriding the stroke (fixed model / oracle only).
    window_x: tuple[float, float] | None = None

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(b.category for b in self.bins)

    @property
    def x_range(self) -> tuple[float, float]:
        if self.window_x is not None:
            return self.window_x
        return (self.x - self.stroke_x_m / 2, self.x + self.stroke_x_m / 2)

    def bin_for(self, category: str) -> Bin:
        for b in self.bins:
            if b.category == category:
                return b
        raise KeyError(category)

    def move_time(self, dx, dy):
        """Time for a point-to-point move: both axes at once, each trapezoidal; floats or numpy arrays."""
        tx = travel_time(dx, self.vx_max, self.ax_max)
        ty = travel_time(dy, self.vy_max, self.ay_max)
        if isinstance(tx, float) and isinstance(ty, float):
            return max(tx, ty)
        import numpy as np

        return np.maximum(tx, ty)


def travel_time(distance_m, speed, accel):
    """Trapezoidal (or triangular) single-axis move time from rest to rest; floats and numpy arrays."""
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


def trapezoid_position(d: float, t: float, speed: float, accel: float) -> float:
    """Distance covered after t seconds of a rest-to-rest trapezoidal move of length |d| (signed like d)."""
    sign = 1.0 if d >= 0 else -1.0
    d = abs(d)
    if d <= 0.0 or t <= 0.0:
        return 0.0
    t_total = travel_time(d, speed, accel)
    if t >= t_total:
        return sign * d
    if d >= speed * speed / accel:
        t_acc = speed / accel
        v_peak = speed
    else:
        t_acc = math.sqrt(d / accel)
        v_peak = accel * t_acc
    if t < t_acc:
        s = 0.5 * accel * t * t
    elif t < t_total - t_acc:
        s = 0.5 * accel * t_acc * t_acc + v_peak * (t - t_acc)
    else:
        tr = t_total - t
        s = d - 0.5 * accel * tr * tr
    return sign * s


@dataclass(frozen=True, slots=True)
class BeltSpec:
    length_m: float
    width_m: float = 0.6
    # Segment boundaries strictly inside (0, length). One segment = one belt; N belts in a row each get a speed.
    boundaries_m: tuple[float, ...] = ()
    v_max_m_s: float = 1.0
    v_min_m_s: float = -0.3  # negative = the belt can run backwards
    accel_m_s2: float = 1.0
    circulating: bool = False  # items leaving the end go back to the head ("they can just teleport back")
    # WHY only targetable items loop: residue nobody may pick would otherwise circle forever and fill the belt.
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
                cycle = arm.grasp_dwell_s + arm.release_dwell_s + 2.0 * arm.move_time(0.0, max(d_bin, 1e-3))
            p = max(c.p_grasp for c in self.categories if c.name in arm.categories)
            total += 60.0 * p / cycle
        return total


def _bin_to_belt_m(b: Bin, width: float) -> float:
    if 0.0 <= b.y <= width:
        return 0.0
    return min(abs(b.y), abs(b.y - width))


__all__ = [
    "ArmSpec",
    "BeltSpec",
    "Bin",
    "Category",
    "Scenario",
    "Supply",
    "trapezoid_position",
    "travel_time",
]

