"""The belt-speed ladder: how fast each belt segment runs, decided every control period.

S0 constant (v1, the daily note's "just arbitrary constant speed"): 0.3 m/s whatever happens.
S1 best constant: the same policy, its speed tuned per scenario on tuning seeds (the honest open-loop baseline).
S2 rush when idle (Zach: "if none of the arms have anything in their view they can pick up then it should accelerate
   until at least one has one"): cruise while any arm has a pickable item in reach, otherwise run up to ``v_rush``,
   but never faster than lets the belt brake back to cruise before the next pickable item reaches an arm.
S3 deadline-aware: S2, plus slow down so the next ``protect_n`` items that only this arm can still take (no
   downstream arm has their bin) stay in reach until the arm gets to them, never below ``v_floor``. Items worth
   ``hv_value`` or more are protected down to ``v_min`` of the belt (it may stop or back up: the gold nugget).
S4 segmented: one belt per arm, each running S3 on its own arms (Zach: "a row of N conveyor belts with independent
   speed control"). Needs a scenario whose belt has segment boundaries.

All speeds are commands; the engine rate-limits them by the belt's acceleration.
"""

from __future__ import annotations

import math

import numpy as np

from .engine import APPROACH, FREE, GRASP, ON_BELT, Sim


class Constant:
    def __init__(self, v: float = 0.3, name: str = "S0 constant 0.3 m/s"):
        self.v = v
        self.name = name

    def command(self, sim: Sim) -> np.ndarray:
        return np.full(sim.n_seg, self.v)


def _reach_entry_exit(sim: Sim, k: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = sim.sc.arms[k]
    if spec.window_x is not None:
        return np.full(y.shape, spec.window_x[0]), np.full(y.shape, spec.window_x[1])
    half = np.sqrt(np.maximum(spec.reach_m**2 - (y - spec.y) ** 2, 0.0))
    return spec.x - half, spec.x + half


def _arm_view(sim: Sim, k: int):
    """Eligible items for arm k with their distance to reach entry (<=0: in reach) and to reach exit."""
    ids = sim.eligible(k)
    if ids.size == 0:
        return ids, np.zeros(0), np.zeros(0)
    y = sim.y[ids]
    entry, exit_ = _reach_entry_exit(sim, k, y)
    # Items the arm cannot reach at any x (outside its lateral reach) are not in its view.
    reachable = exit_ > entry
    ids, entry, exit_ = ids[reachable], entry[reachable], exit_[reachable]
    x = sim.x[ids]
    keep = x < exit_
    return ids[keep], (entry - x)[keep], (exit_ - x)[keep]


class RushWhenIdle:
    def __init__(self, v_cruise: float = 0.3, v_rush: float = 1.0, margin_m: float = 0.05, name: str | None = None):
        self.v_cruise, self.v_rush, self.margin = v_cruise, v_rush, margin_m
        self.name = name or f"S2 rush when idle (cruise {v_cruise:g})"

    def _rush_cap(self, sim: Sim, arms) -> float:
        """Cruise, or faster while no arm has anything to pick: as fast as still lets the belt brake back to cruise
        before the next pickable item reaches any of ``arms``. Never faster than an arm mid-pick can live with."""
        a = sim.sc.belt.accel_m_s2
        rush = self.v_rush
        hold = math.inf
        for k in arms:
            arm = sim.arms[k]
            if arm.phase in (APPROACH, GRASP):
                # WHY: an arm on its way to (or closing on) an item planned its intercept at this speed and can only
                # grasp below its tracking limit ...
                hold = min(hold, self.v_cruise, sim.sc.arms[k].v_track_max_m_s)
                if arm.phase == APPROACH and sim.status[arm.item] == ON_BELT:
                    # ... and the belt must not carry that item out of its reach before the tool gets there.
                    i = arm.item
                    exit_x = float(_reach_entry_exit(sim, k, np.array([sim.y[i]]))[1][0])
                    t_rem = max(arm.t_contact - sim.t, 0.0) + 0.15
                    hold = min(hold, max((exit_x - float(sim.x[i]) - 0.03) / t_rem, 0.0))
            ids, to_entry, _ = _arm_view(sim, k)
            if ids.size == 0:
                continue
            d = float(np.min(to_entry))
            if d <= 0.0:
                rush = self.v_cruise
                continue
            # Braking distance from v to cruise: (v^2 - vc^2) / 2a <= d - margin (plus one control period's travel).
            d_ok = max(d - self.margin - rush * sim.control_period_s, 0.0)
            rush = min(rush, math.sqrt(self.v_cruise**2 + 2.0 * a * d_ok))
        return min(max(rush, self.v_cruise), hold)

    def segment_speed(self, sim: Sim, arms) -> float:
        return self._rush_cap(sim, arms)

    def command(self, sim: Sim) -> np.ndarray:
        return np.full(sim.n_seg, self.segment_speed(sim, range(len(sim.sc.arms))))


class DeadlineAware(RushWhenIdle):
    def __init__(self, v_cruise: float = 0.3, v_rush: float = 1.0, v_floor: float = 0.05, protect_n: int = 1,
                 hv_value: float = math.inf, backlog_gain: float = 0.0, name: str | None = None):
        super().__init__(v_cruise, v_rush)
        self.v_floor, self.protect_n, self.hv_value, self.backlog_gain = v_floor, protect_n, hv_value, backlog_gain
        self.name = name or f"S3 deadline-aware (cruise {v_cruise:g}, floor {v_floor:g}, protect {protect_n})"

    def _protect_cap(self, sim: Sim, arms) -> float:
        cap = math.inf
        v_min = sim.sc.belt.v_min_m_s
        floor = self.v_floor
        if self.backlog_gain > 0 and sim.sc.supply.mode == "stream":
            floor = floor + self.backlog_gain * sim.queue_length()
        for k in arms:
            spec = sim.sc.arms[k]
            ids, to_entry, to_exit = _arm_view(sim, k)
            if ids.size == 0:
                continue
            last = ~sim.downstream_cat[k, sim.cat[ids]]
            hv = sim.value[ids] >= self.hv_value
            prot = last | hv
            if not prot.any():
                continue
            ids, to_exit, hv = ids[prot], to_exit[prot], hv[prot]
            order = np.argsort(to_exit)[: self.protect_n]
            t_free = sim.time_until_free(k)
            pos = sim.free_position(k)
            base_cycle = spec.fixed_cycle_s or (spec.grasp_dwell_s + spec.release_dwell_s + 2 * spec.travel_s(0.4))
            for rank, j in enumerate(order):
                i = int(ids[j])
                reach_s = spec.fixed_reach_s if spec.fixed_reach_s is not None else spec.travel_s(
                    float(math.hypot(sim.x[i] - pos[0], sim.y[i] - pos[1])))
                t_need = t_free + rank * base_cycle + reach_s + 0.1
                v_save = (float(to_exit[j]) - 0.02) / t_need
                lo = v_min if hv[j] else floor
                cap = min(cap, max(v_save, lo))
        return cap

    def segment_speed(self, sim: Sim, arms) -> float:
        arms = list(arms)
        v = self._rush_cap(sim, arms)
        # Starving arms want the belt fast; a protected item wants it slow. Saving an item already in reach wins.
        return min(v, self._protect_cap(sim, arms))


class Segmented:
    """One belt per arm; each segment runs ``inner`` on the arms whose base lies on it. Arm-less segments (the head
    infeed) follow the next segment downstream."""

    def __init__(self, inner: DeadlineAware | RushWhenIdle, buffer_m: float = 0.3, buffer_cap: int = 3,
                 name: str | None = None):
        self.inner = inner
        self.buffer_m, self.buffer_cap = buffer_m, buffer_cap
        self.name = name or f"S4 segmented ({inner.name.split(' ', 1)[1]})"
        self._seg_arms: list[list[int]] | None = None

    def command(self, sim: Sim) -> np.ndarray:
        if self._seg_arms is None:
            seg = np.searchsorted(sim.boundaries, [a.x for a in sim.sc.arms], side="right")
            self._seg_arms = [[k for k, s in enumerate(seg) if s == j] for j in range(sim.n_seg)]
        out = np.full(sim.n_seg, np.nan)
        for j in range(sim.n_seg - 1, -1, -1):
            arms = self._seg_arms[j]
            if arms:
                out[j] = self.inner.segment_speed(sim, arms)
            else:
                out[j] = out[j + 1] if j + 1 < sim.n_seg else self.inner.v_cruise
            # WHY zone control (accumulation conveyors): when items are already queuing at the hand-off to a slower
            # belt, running this one faster only heaps them there, so it may not outrun its downstream neighbour.
            if j + 1 < sim.n_seg and self._queued_at(sim, j) >= self.buffer_cap:
                out[j] = min(out[j], max(out[j + 1], 0.0))
        return out

    def _queued_at(self, sim: Sim, j: int) -> int:
        ids = sim.on_belt
        if ids.size == 0:
            return 0
        edge = sim.boundaries[j]
        x = sim.x[ids]
        return int(np.count_nonzero((x > edge - self.buffer_m) & (x < edge) & (sim.status[ids] == ON_BELT)))


def idle_arms(sim: Sim) -> int:
    return sum(1 for a in sim.arms if a.phase == FREE)
