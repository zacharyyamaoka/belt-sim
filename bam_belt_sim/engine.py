"""The headless engine: a fixed-step 2D belt with N arms, a speed policy and a pick policy.

One step (``dt``, default 50 ms) does, in order: speed control (every ``control_period_s``), belt motion with segment
hand-offs, infeed, arm phase changes, and pick decisions for free arms. Items are numpy arrays; the per-step cost is
a handful of vector ops plus a Python loop over arms, so a 20 min episode of a 3-arm line runs in well under a
second (measured in the report).

WHY a fixed step and not events (the facility model's F1 ``line_sim`` is event-driven): with a variable-speed belt an
item's position is the integral of a speed the controller chooses as it goes, so "when does item i enter arm k's
reach" has no closed form until the controller is known. A 50 ms step moves a 1 m/s belt 5 cm, under the 4 cm
minimum item spacing x 2, and the oracle test checks the constant-speed case agrees with the event-driven F1.

WHY common random numbers: every item's category, lateral position, spacing, grasp probability and its first eight
grasp-outcome draws come from the scenario seed alone, in arrival order. Two policies run on the same seed see the
same items and the same coin flips, so their difference is the policy, not the luck.

Segment hand-off rule: an item crossing into a neighbouring belt that is stopped or running the other way stays at
the boundary (it piles there) until that belt moves its way. The peak pile is reported (``peak_items_per_0_2m``).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import ArmSpec, Scenario, travel_time

QUEUED, ON_BELT, HELD, PICKED, LOST = 0, 1, 2, 3, 4
FREE, APPROACH, GRASP, DELIVER = 0, 1, 2, 3
N_DRAWS = 8
EPS = 1e-6


class SpeedPolicy(Protocol):
    name: str

    def command(self, sim: Sim) -> np.ndarray: ...


class PickPolicy(Protocol):
    name: str

    def choose(self, sim: Sim, arm: int, cand: Candidates) -> int | None: ...


@dataclass
class Candidates:
    """Items one free arm could commit to now, with the predicted intercept and the full cycle estimate."""

    idx: np.ndarray  # item ids
    tau: np.ndarray  # commit -> contact, s
    cx: np.ndarray  # contact point x
    cy: np.ndarray
    cycle: np.ndarray  # commit -> arm free again if the grasp succeeds, s
    p: np.ndarray
    value: np.ndarray
    seq: np.ndarray  # first-entry order (FIFO key)
    exit_s: np.ndarray  # time until the item leaves this arm's reach at the current belt speed (inf if never)

    def __len__(self) -> int:
        return int(self.idx.size)


@dataclass
class ArmState:
    phase: int = FREE
    item: int = -1
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t_contact: float = 0.0
    t_first_contact: float = 0.0
    t_phase_end: float = 0.0
    t_free: float = 0.0  # fixed-cycle model: busy until
    contact: np.ndarray = field(default_factory=lambda: np.zeros(2))
    busy_s: float = 0.0
    busy_since: float = 0.0


@dataclass
class Log:
    picks: list = field(default_factory=list)  # (t, item, arm, value, category)
    losses: list = field(default_factory=list)  # (t, item, value, category)
    attempts: int = 0
    failed_grasps: int = 0
    missed_intercepts: int = 0
    recirculated: int = 0
    speed_t: list = field(default_factory=list)
    speed_v: list = field(default_factory=list)
    queue_len: list = field(default_factory=list)
    peak_pile: int = 0
    frames: list = field(default_factory=list)


class ItemStream:
    """The scenario's items in arrival order, generated lazily in chunks from the seed alone (CRN)."""

    CHUNK = 2048

    def __init__(self, scenario: Scenario, seed: int):
        self.sc = scenario
        self.rng = np.random.default_rng([seed, 0xBE17])
        self.n = 0
        cap = self.CHUNK
        self.cat = np.zeros(cap, np.int16)
        self.y = np.zeros(cap)
        self.gap = np.zeros(cap)
        self.release = np.zeros(cap)
        self.p = np.zeros(cap)
        self.value = np.zeros(cap)
        self.draws = np.zeros((cap, N_DRAWS))
        self._t_last_release = 0.0
        shares = np.array([c.share for c in scenario.categories], float)
        self._cum = np.cumsum(shares / shares.sum())
        self._p_cat = np.array([c.p_grasp for c in scenario.categories])
        self._v_cat = np.array([c.value for c in scenario.categories])

    def ensure(self, n: int) -> None:
        while self.n < n:
            self._grow()

    def _grow(self) -> None:
        sc, rng, k = self.sc, self.rng, self.CHUNK
        sup = sc.supply
        cat = np.searchsorted(self._cum, rng.random(k), side="right").clip(0, len(self._cum) - 1)
        y = rng.uniform(sup.y_margin_m, sc.belt.width_m - sup.y_margin_m, k)
        u_burst = rng.random(k)
        e = rng.exponential(1.0, k)
        b = sup.burstiness
        if sup.mode == "hopper":
            mean_gap = 1.0 / sup.density_per_m
            # Clumped items sit at min_gap; the rest spread so the mean gap stays 1/density.
            spread = max((mean_gap - b * sup.min_gap_m) / max(1.0 - b, 1e-9) - sup.min_gap_m, 1e-6)
            gap = np.where(u_burst < b, sup.min_gap_m, sup.min_gap_m + spread * e)
            release = np.zeros(k)
        else:
            mean_dt = 1.0 / sup.rate_per_s
            small = 0.02 * mean_dt
            spread = max((mean_dt - b * small) / max(1.0 - b, 1e-9), 1e-9)
            dts = np.where(u_burst < b, small, spread * e)
            release = self._t_last_release + np.cumsum(dts)
            self._t_last_release = float(release[-1])
            gap = np.full(k, sup.min_gap_m)
        spread_p = sc.p_grasp_spread
        p = self._p_cat[cat] + (rng.uniform(-spread_p, spread_p, k) if spread_p > 0 else 0.0)
        draws = rng.random((k, N_DRAWS))
        sl = slice(self.n, self.n + k)
        if self.n + k > self.cat.size:
            new = self.cat.size * 2
            for name in ("cat", "y", "gap", "release", "p", "value"):
                arr = getattr(self, name)
                grown = np.zeros(new, arr.dtype)
                grown[: arr.size] = arr
                setattr(self, name, grown)
            grown = np.zeros((new, N_DRAWS))
            grown[: self.draws.shape[0]] = self.draws
            self.draws = grown
        self.cat[sl], self.y[sl], self.gap[sl], self.release[sl] = cat, y, gap, release
        self.p[sl] = np.clip(p, 0.0, 1.0)
        self.value[sl] = self._v_cat[cat]
        self.draws[sl] = draws
        self.n += k


class Sim:
    def __init__(
        self,
        scenario: Scenario,
        speed_policy: SpeedPolicy,
        pick_policy: PickPolicy,
        seed: int = 0,
        dt: float = 0.05,
        control_period_s: float = 0.1,
        record_every: int = 0,
    ):
        self.sc = scenario
        self.speed_policy = speed_policy
        self.pick_policy = pick_policy
        self.dt = dt
        self.control_period_s = control_period_s
        self.record_every = record_every
        self.t = 0.0
        self.steps = 0
        self.stream = ItemStream(scenario, seed)
        belt = scenario.belt
        self.boundaries = np.array(belt.boundaries_m, float)
        self.n_seg = belt.n_segments
        self._transfer_clear = np.full(max(self.n_seg - 1, 0), 1e3)
        self.v = np.zeros(self.n_seg)
        self.v_cmd = np.zeros(self.n_seg)
        # Item state, indexed by stream id. Grown alongside the stream.
        self.cap = 0
        self.x = np.zeros(0)
        self.status = np.zeros(0, np.int8)
        self.reserved = np.zeros(0, np.int16)
        self.passes = np.zeros(0, np.int16)
        self.attempts = np.zeros(0, np.int16)
        self.seq = np.zeros(0, np.int64)
        self.entered_t = np.zeros(0)
        self.entered_x = np.zeros(0)
        self.on_belt: np.ndarray = np.zeros(0, np.int64)  # ids currently on the belt
        self.next_fresh = 0
        self.entries = 0
        self.recirc: deque[tuple[float, int]] = deque()
        self.since_last_entry_m = 0.0  # set to the first gap below, so item 0 enters at x = 0
        self.next_gap_m = 0.0
        self.arms = [ArmState() for _ in scenario.arms]
        for a, spec in zip(self.arms, scenario.arms, strict=True):
            b = spec.bins[0]
            a.pos = np.array([b.x, b.y], float)
        cats = scenario.category_index()
        self.arm_cat_mask = np.zeros((len(scenario.arms), len(scenario.categories)), bool)
        self.arm_bins = []
        for k, spec in enumerate(scenario.arms):
            bins = np.zeros((len(scenario.categories), 2))
            for b in spec.bins:
                self.arm_cat_mask[k, cats[b.category]] = True
                bins[cats[b.category]] = (b.x, b.y)
            self.arm_bins.append(bins)
        # Which arms sit downstream of arm k (larger x) and share a category: used by "last chance" logic.
        self.downstream_cat = np.zeros((len(scenario.arms), len(scenario.categories)), bool)
        for k, spec in enumerate(scenario.arms):
            for j, other in enumerate(scenario.arms):
                if other.x > spec.x:
                    self.downstream_cat[k] |= self.arm_cat_mask[j]
        self.log = Log()
        self._next_control = 0.0
        self._ensure_items(1)
        self.since_last_entry_m = float(self.stream.gap[0])

    # ------------------------------------------------------------------ items
    def _ensure_items(self, n: int) -> None:
        self.stream.ensure(n)
        if n <= self.cap:
            return
        new = max(n, 2 * self.cap, 1024)

        def grow(arr, fill=0):
            out = np.full(new, fill, arr.dtype)
            out[: arr.size] = arr
            return out

        self.x = grow(self.x)
        self.status = grow(self.status, QUEUED)
        self.reserved = grow(self.reserved, -1)
        self.passes = grow(self.passes)
        self.attempts = grow(self.attempts)
        self.seq = grow(self.seq, -1)
        self.entered_t = grow(self.entered_t, np.nan)
        self.entered_x = grow(self.entered_x, np.nan)
        self.cap = new

    @property
    def cat(self) -> np.ndarray:
        return self.stream.cat

    @property
    def y(self) -> np.ndarray:
        return self.stream.y

    @property
    def p(self) -> np.ndarray:
        return self.stream.p

    @property
    def value(self) -> np.ndarray:
        return self.stream.value

    def eligible(self, k: int) -> np.ndarray:
        """Items on the belt arm k has a bin for and nobody has reserved."""
        ids = self.on_belt
        if ids.size == 0:
            return ids
        ids = ids[(self.status[ids] == ON_BELT) & (self.reserved[ids] < 0)]
        return ids[self.arm_cat_mask[k, self.cat[ids]]]

    def item_speed(self, ids: np.ndarray) -> np.ndarray:
        return self.v[np.searchsorted(self.boundaries, self.x[ids], side="right")]

    def queue_length(self) -> int:
        """Items waiting at the head (stream mode backlog + recirculated). Hopper's pile is unlimited, not a queue."""
        n = len(self.recirc)
        if self.sc.supply.mode == "stream":
            self._ensure_items(self.next_fresh + 1)
            # released but not yet on the belt
            rel = self.stream.release
            hi = self.next_fresh
            while rel[hi] <= self.t:
                hi += 1
                self._ensure_items(hi + 1)
            n += hi - self.next_fresh
        return n

    # ------------------------------------------------------------------ step
    def run(self, duration_s: float) -> Sim:
        n = int(round(duration_s / self.dt))
        for _ in range(n):
            self.step()
        return self

    def step(self) -> None:
        dt = self.dt
        if self.t + EPS >= self._next_control:
            cmd = np.asarray(self.speed_policy.command(self), float).reshape(-1)
            if cmd.size == 1 and self.n_seg > 1:
                cmd = np.full(self.n_seg, float(cmd[0]))
            belt = self.sc.belt
            self.v_cmd = np.clip(cmd, belt.v_min_m_s, belt.v_max_m_s)
            self._next_control += self.control_period_s
            self.log.speed_t.append(self.t)
            self.log.speed_v.append(self.v.copy())
            self.log.queue_len.append(self.queue_length())
            if self.on_belt.size:
                h = np.bincount((self.x[self.on_belt] / 0.2).astype(np.int64))
                self.log.peak_pile = max(self.log.peak_pile, int(h.max()))
        dv = np.clip(self.v_cmd - self.v, -self.sc.belt.accel_m_s2 * dt, self.sc.belt.accel_m_s2 * dt)
        self.v = self.v + dv
        self._move(dt)
        self._infeed(dt)
        self.t += dt
        self.steps += 1
        self._arms()
        if self.record_every and self.steps % self.record_every == 0:
            self._record()

    def _move(self, dt: float) -> None:
        ids = self.on_belt
        if ids.size == 0:
            return
        x = self.x[ids]
        seg = np.searchsorted(self.boundaries, x, side="right")
        vi = self.v[seg]
        nx = x + vi * dt
        if self.boundaries.size:
            # How far each downstream belt has carried the last item it took (every step, crossings or not).
            self._transfer_clear = self._transfer_clear + np.maximum(self.v[1:], 0.0) * dt
            nseg = np.searchsorted(self.boundaries, nx, side="right")
            cross = nseg != seg
            if cross.any():
                ci = np.nonzero(cross)[0]
                direction = np.sign(vi[ci]).astype(np.int64)
                neighbour = np.clip(seg[ci] + direction, 0, self.n_seg - 1)
                blocked = self.v[neighbour] * vi[ci] <= 0.0
                # WHY a transfer spacing rule: a faster upstream belt bunches items onto a slower downstream one; real
                # items cannot overlap, so an item may cross a forward hand-off only once the last one to cross has
                # moved min_gap clear of it (as at the infeed). The rest wait at the boundary on the upstream belt.
                fwd = np.nonzero(~blocked & (direction > 0))[0]
                if fwd.size:
                    order = fwd[np.argsort(-x[ci[fwd]])]
                    gap = self.sc.supply.min_gap_m
                    for j in order:
                        b = int(seg[ci[j]])
                        edge = self.boundaries[b]
                        overshoot = nx[ci[j]] - edge
                        if self._transfer_clear[b] - overshoot < gap:
                            blocked[j] = True
                        else:
                            self._transfer_clear[b] = overshoot
                if blocked.any():
                    bi = ci[blocked]
                    d = direction[blocked]
                    edge = np.where(d > 0, self.boundaries[np.clip(seg[bi], 0, self.boundaries.size - 1)],
                                    self.boundaries[np.clip(seg[bi] - 1, 0, self.boundaries.size - 1)])
                    nx[bi] = edge - d * 1e-4
        np.maximum(nx, 0.0, out=nx)
        self.x[ids] = nx
        L = self.sc.belt.length_m
        out = nx >= L
        if out.any():
            gone = ids[out]
            keep = ~out
            self.on_belt = ids[keep]
            for i in gone:
                i = int(i)
                if self.status[i] != ON_BELT:
                    continue
                self.reserved[i] = -1
                belt = self.sc.belt
                if belt.circulating and self.passes[i] + 1 < belt.max_passes:
                    self.passes[i] += 1
                    self.status[i] = QUEUED
                    self.recirc.append((self.t + belt.loop_delay_s, i))
                    self.log.recirculated += 1
                else:
                    self.status[i] = LOST
                    self.log.losses.append((self.t, i, float(self.value[i]), int(self.cat[i])))

    def _infeed(self, dt: float) -> None:
        v0 = self.v[0]
        self.since_last_entry_m = max(self.since_last_entry_m + v0 * dt, 0.0)
        sup = self.sc.supply
        t_now = self.t + dt
        new = []
        for _ in range(64):  # bounded: at most a few items per step
            if self.recirc and self.recirc[0][0] <= t_now:
                gap = sup.min_gap_m
                if self.since_last_entry_m < gap:
                    break
                ready, i = self.recirc[0]
                self.recirc.popleft()
            else:
                self._ensure_items(self.next_fresh + 1)
                i = self.next_fresh
                gap = float(self.stream.gap[i])
                ready = float(self.stream.release[i]) if sup.mode == "stream" else -math.inf
                if ready > t_now or self.since_last_entry_m < gap:
                    break
                self.next_fresh += 1
                self.seq[i] = self.entries
                self.entered_t[i] = t_now
            # Place it where it would be had it entered the moment space opened (or it arrived).
            x0 = self.since_last_entry_m - gap
            if math.isfinite(ready):
                x0 = min(x0, max(v0, 0.0) * (t_now - ready))
            self.x[i] = max(x0, 0.0)
            if self.passes[i] == 0:
                self.entered_x[i] = self.x[i]
            self.status[i] = ON_BELT
            self.reserved[i] = -1
            self.since_last_entry_m = self.x[i]
            self.entries += 1
            new.append(i)
        if new:
            self.on_belt = np.concatenate([self.on_belt, np.array(new, np.int64)])

    # ------------------------------------------------------------------ arms
    def in_reach(self, k: int, px, py):
        spec = self.sc.arms[k]
        if spec.window_x is not None:
            return (px >= spec.window_x[0]) & (px < spec.window_x[1])
        return (px - spec.x) ** 2 + (py - spec.y) ** 2 <= spec.reach_m**2

    def candidates(self, k: int, from_pos: np.ndarray | None = None, extra_delay_s: float = 0.0) -> Candidates:
        """Items arm k could commit to, predicted at the current belt speeds.

        ``from_pos`` / ``extra_delay_s`` let a speed policy ask "what could this busy arm do once it is free".
        """
        spec = self.sc.arms[k]
        ids = self.on_belt
        if ids.size:
            ids = ids[(self.status[ids] == ON_BELT) & (self.reserved[ids] < 0)]
            ids = ids[self.arm_cat_mask[k, self.cat[ids]]]
        if ids.size:
            x = self.x[ids]
            lo = (spec.window_x[0] if spec.window_x else spec.x - spec.reach_m) - 3.0
            hi = spec.window_x[1] if spec.window_x else spec.x + spec.reach_m
            m = (x >= lo) & (x <= hi)
            ids = ids[m]
        if ids.size == 0:
            z = np.zeros(0)
            return Candidates(np.zeros(0, np.int64), z, z, z, z, z, z, np.zeros(0, np.int64), z)
        pos = self.arms[k].pos if from_pos is None else from_pos
        x, y = self.x[ids], self.y[ids]
        u = self.item_speed(ids)
        xd = x + u * extra_delay_s
        if spec.fixed_reach_s is not None:
            tau = np.full(ids.size, spec.fixed_reach_s)
        else:
            tau = travel_time(np.hypot(xd - pos[0], y - pos[1]), spec.tool_speed_m_s, spec.tool_accel_m_s2)
            for _ in range(4):
                tau = travel_time(np.hypot(xd + u * tau - pos[0], y - pos[1]), spec.tool_speed_m_s,
                                  spec.tool_accel_m_s2)
        cx = xd + u * tau
        # WHY the dwell end too: the tool tracks the item while it closes, so the item must still be in reach when
        # the grasp completes, not only at first contact (caught by test_arms_grasp_only_inside_reach...).
        dwell = 0.0 if spec.fixed_cycle_s is not None else spec.grasp_dwell_s
        ok = (self.in_reach(k, cx, y) & self.in_reach(k, cx + u * dwell, y)
              & (np.abs(u) <= spec.v_track_max_m_s + 1e-9) & (cx < self.sc.belt.length_m))
        ids, tau, cx, y, u = ids[ok], tau[ok], cx[ok], y[ok], u[ok]
        cats = self.cat[ids]
        if spec.fixed_cycle_s is not None:
            cycle = np.full(ids.size, spec.fixed_cycle_s)
        else:
            bins = self.arm_bins[k][cats]
            cycle = (tau + spec.grasp_dwell_s
                     + travel_time(np.hypot(bins[:, 0] - cx, bins[:, 1] - y), spec.tool_speed_m_s,
                                   spec.tool_accel_m_s2)
                     + spec.release_dwell_s)
        exit_x = self._exit_x(k, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            exit_s = np.where(u > 1e-9, (exit_x - self.x[ids]) / u, np.inf)
        return Candidates(ids, tau, cx, y, cycle, self.p[ids], self.value[ids], self.seq[ids], exit_s)

    def _exit_x(self, k: int, y: np.ndarray) -> np.ndarray:
        spec = self.sc.arms[k]
        if spec.window_x is not None:
            return np.full(np.shape(y), spec.window_x[1])
        return spec.x + np.sqrt(np.maximum(spec.reach_m**2 - (y - spec.y) ** 2, 0.0))

    def time_until_free(self, k: int) -> float:
        a, spec = self.arms[k], self.sc.arms[k]
        if a.phase == FREE:
            return 0.0
        if spec.fixed_cycle_s is not None:
            return max(a.t_free - self.t, 0.0)
        if a.phase == APPROACH:
            i = a.item
            b = self.arm_bins[k][self.cat[i]]
            return (a.t_contact - self.t) + spec.grasp_dwell_s + spec.travel_s(
                float(np.hypot(b[0] - a.contact[0], b[1] - a.contact[1]))) + spec.release_dwell_s
        if a.phase == GRASP:
            i = a.item
            b = self.arm_bins[k][self.cat[i]]
            return (a.t_phase_end - self.t) + spec.travel_s(
                float(np.hypot(b[0] - a.contact[0], b[1] - a.contact[1]))) + spec.release_dwell_s
        return max(a.t_phase_end - self.t, 0.0)

    def free_position(self, k: int) -> np.ndarray:
        """Where arm k's tool will be when it is next free (its target bin on success)."""
        a = self.arms[k]
        if a.phase in (APPROACH, GRASP) and a.item >= 0:
            return self.arm_bins[k][self.cat[a.item]]
        return a.pos

    def _arms(self) -> None:
        t = self.t
        for k, (a, spec) in enumerate(zip(self.arms, self.sc.arms, strict=True)):
            fixed = spec.fixed_cycle_s is not None
            if a.phase == APPROACH and t + EPS >= a.t_contact:
                self._contact(k, a, spec, fixed)
            if a.phase == GRASP and t + EPS >= a.t_phase_end:
                self._grasp_outcome(k, a, spec)
            if a.phase == DELIVER and t + EPS >= a.t_phase_end:
                i = a.item
                self.status[i] = PICKED
                self.log.picks.append((t, i, k, float(self.value[i]), int(self.cat[i])))
                a.pos = self.arm_bins[k][self.cat[i]].copy()
                self._set_free(a)
            if fixed and a.phase == FREE and t + EPS < a.t_free:
                continue
            if a.phase == FREE:
                cand = self.candidates(k)
                if len(cand) == 0:
                    continue
                c = self.pick_policy.choose(self, k, cand)
                if c is None:
                    continue
                i = int(cand.idx[c])
                self.reserved[i] = k
                a.item = i
                a.phase = APPROACH
                a.t_contact = t + float(cand.tau[c])
                a.t_first_contact = a.t_contact
                a.contact = np.array([cand.cx[c], cand.cy[c]])
                a.busy_since = t
                if fixed:
                    a.t_free = t + spec.fixed_cycle_s
                self.log.attempts += 1

    def _set_free(self, a: ArmState) -> None:
        # WHY t_free for the fixed model: its arm is busy for the whole fixed cycle, whatever happened at contact.
        a.busy_s += max(self.t, a.t_free) - a.busy_since
        a.phase = FREE
        a.item = -1

    def _contact(self, k: int, a: ArmState, spec: ArmSpec, fixed: bool) -> None:
        i = a.item
        ok = self.status[i] == ON_BELT and self.reserved[i] == k
        if ok:
            xi, yi = float(self.x[i]), float(self.y[i])
            u = float(self.item_speed(np.array([i]))[0])
            here = abs(xi - a.contact[0]) <= spec.contact_tol_m or fixed
            dwell = 0.0 if fixed else spec.grasp_dwell_s
            ok = (bool(self.in_reach(k, xi, yi)) and bool(self.in_reach(k, xi + u * dwell, yi))
                  and abs(u) <= spec.v_track_max_m_s + 1e-9 and here)
            if not ok and not fixed and self.t - a.t_first_contact <= spec.replan_budget_s:
                if self._replan(k, a, spec, xi, yi, u):
                    return
        if not ok:
            self.log.missed_intercepts += 1
            if self.reserved[i] == k:
                self.reserved[i] = -1
            a.pos = a.contact.copy()
            self._set_free(a)
            return
        a.contact = np.array([self.x[i], self.y[i]])
        if fixed:
            self._grasp_outcome(k, a, spec)
            return
        a.phase = GRASP
        a.t_phase_end = self.t + spec.grasp_dwell_s

    def _replan(self, k: int, a: ArmState, spec: ArmSpec, xi: float, yi: float, u: float) -> bool:
        """The belt did not do what the arm assumed: re-aim from the tool's current point, or wait for the item."""
        exit_x = float(self._exit_x(k, np.array([yi]))[0])
        if xi >= exit_x:
            return False  # it has left this arm's reach for good (until the belt reverses: not worth waiting)
        if abs(u) > spec.v_track_max_m_s:
            a.t_contact = self.t + self.dt  # too fast to grasp right now: hover and wait for the belt to slow
            return True
        tau = spec.travel_s(float(np.hypot(xi - a.contact[0], yi - a.contact[1])))
        for _ in range(3):
            tau = spec.travel_s(float(np.hypot(xi + u * tau - a.contact[0], yi - a.contact[1])))
        cx = xi + u * tau
        if bool(self.in_reach(k, cx, yi)) and bool(self.in_reach(k, cx + u * spec.grasp_dwell_s, yi)):
            a.contact = np.array([cx, yi])
            a.t_contact = self.t + max(tau, self.dt)
            return True
        if cx < exit_x:
            a.t_contact = self.t + 0.1  # still upstream of its reach (belt slowed or reversed): wait
            return True
        return False

    def _grasp_outcome(self, k: int, a: ArmState, spec: ArmSpec) -> None:
        i = a.item
        n = int(self.attempts[i])
        self.attempts[i] += 1
        draw = self.stream.draws[i, n] if n < N_DRAWS else self._extra_draw(i, n)
        if self.status[i] != ON_BELT:  # should not happen: it was reserved
            self._set_free(a)
            return
        a.contact = np.array([self.x[i], self.y[i]])
        if draw < self.p[i]:
            self.status[i] = HELD
            self.reserved[i] = k
            self.on_belt = self.on_belt[self.on_belt != i]
            if spec.fixed_cycle_s is not None:
                self.status[i] = PICKED
                self.log.picks.append((self.t, i, k, float(self.value[i]), int(self.cat[i])))
                self._set_free(a)
                return
            b = self.arm_bins[k][self.cat[i]]
            a.phase = DELIVER
            a.t_phase_end = self.t + spec.travel_s(float(np.hypot(b[0] - a.contact[0], b[1] - a.contact[1]))) \
                + spec.release_dwell_s
        else:
            self.log.failed_grasps += 1
            self.reserved[i] = -1
            a.pos = a.contact.copy()
            self._set_free(a)

    def _extra_draw(self, i: int, n: int) -> float:
        return float(np.random.default_rng([i, n, 0xD2A]).random())

    def _record(self) -> None:
        ids = self.on_belt
        arms = []
        for k, a in enumerate(self.arms):
            if a.phase == FREE:
                pos = a.pos
            elif a.phase in (APPROACH,):
                pos = a.contact
            elif a.phase == GRASP:
                pos = np.array([self.x[a.item], self.y[a.item]])
            else:
                pos = self.arm_bins[k][self.cat[a.item]]
            arms.append((float(pos[0]), float(pos[1]), int(a.phase), int(a.item)))
        self.log.frames.append({
            "t": self.t,
            "ids": ids.copy(),
            "x": self.x[ids].copy(),
            "y": self.y[ids].copy(),
            "cat": self.cat[ids].astype(np.int16).copy(),
            "reserved": self.reserved[ids].copy(),
            "v": self.v.copy(),
            "arms": arms,
            "picked": len(self.log.picks),
            "lost": len(self.log.losses),
        })

    # ------------------------------------------------------------------ fixed-cycle bookkeeping
    def finish_busy(self) -> None:
        for a in self.arms:
            if a.phase != FREE:
                a.busy_s += self.t - a.busy_since
                a.busy_since = self.t


__all__ = ["Candidates", "PickPolicy", "Sim", "SpeedPolicy"]
