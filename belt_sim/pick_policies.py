"""The pick-scheduling ladder: which item a free arm commits to next.

P0 FIFO (v1, the daily note's "FIFO is good v1"): the reachable item that entered the belt first.
P1 SPT: the shortest cycle (commit -> grasp -> bin -> free). Classic shortest-processing-time: more picks per minute.
P2 WSPT / value rate: highest expected value per second, p * value / cycle (Smith's rule). Grabs the gold nugget.
P3 last chance: P2, but an item a downstream arm could still take counts at ``downstream_weight`` of its value, so
   an arm spends its time on what only it can save.
P4 lookahead: P3 over short sequences. For the top-``k`` first choices it plans ``depth`` picks ahead on the
   predicted belt and keeps the first pick of the best expected-value-per-second sequence.

``MostDownstream`` is not a rung: it is the facility model F1 line_sim's rule, kept for the oracle test.
"""

from __future__ import annotations


import numpy as np

from .engine import Candidates, Sim


class FIFO:
    name = "P0 FIFO"

    def choose(self, sim: Sim, arm: int, cand: Candidates) -> int | None:
        return int(np.argmin(cand.seq))


class MostDownstream:
    name = "most downstream (F1 rule)"

    def choose(self, sim: Sim, arm: int, cand: Candidates) -> int | None:
        return int(np.argmax(cand.cx))


class ShortestCycle:
    name = "P1 shortest cycle"

    def choose(self, sim: Sim, arm: int, cand: Candidates) -> int | None:
        return int(np.argmin(cand.cycle))


class ValueRate:
    name = "P2 value rate"

    def choose(self, sim: Sim, arm: int, cand: Candidates) -> int | None:
        return int(np.argmax(cand.p * cand.value / cand.cycle))


class LastChance:
    name = "P3 last chance"

    def __init__(self, downstream_weight: float = 0.5):
        self.w = downstream_weight

    def weights(self, sim: Sim, arm: int, ids: np.ndarray) -> np.ndarray:
        later = sim.downstream_cat[arm, sim.cat[ids]]
        return np.where(later, self.w, 1.0)

    def choose(self, sim: Sim, arm: int, cand: Candidates) -> int | None:
        score = cand.p * cand.value * self.weights(sim, arm, cand.idx) / cand.cycle
        return int(np.argmax(score))


class Lookahead:
    """Plan ``depth`` picks ahead for this arm on the predicted belt (current speeds held), first pick of the best."""

    name = "P4 lookahead"

    def __init__(self, depth: int = 3, k: int = 6, downstream_weight: float = 0.5):
        self.depth, self.k = depth, k
        self.base = LastChance(downstream_weight)

    def choose(self, sim: Sim, arm: int, cand: Candidates) -> int | None:
        if len(cand) == 1:
            return 0
        spec = sim.sc.arms[arm]
        w_c = self.base.weights(sim, arm, cand.idx)
        first_score = cand.p * cand.value * w_c / cand.cycle
        order = np.argsort(-first_score)[: self.k]
        # The pool for later picks: everything this arm could take, including items still upstream of its reach.
        pool = sim.eligible(arm)
        if pool.size == 0:
            return int(order[0])
        px, py = sim.x[pool], sim.y[pool]
        pu = sim.item_speed(pool)
        pc = sim.cat[pool]
        pev = sim.p[pool] * sim.value[pool] * self.base.weights(sim, arm, pool)
        bins = sim.arm_bins[arm]
        fixed = spec.fixed_cycle_s is not None
        id_to_pool = {int(i): j for j, i in enumerate(pool)}
        mean_cycle = float(np.mean(cand.cycle))

        def plan(pos, t0, used, depth):
            """Best (value, time) continuation from tool pos at time offset t0 (greedy-exhaustive, small)."""
            if depth == 0:
                return 0.0, 0.0
            mask = np.ones(pool.size, bool)
            if used:
                mask[list(used)] = False
            if not mask.any():
                return 0.0, mean_cycle
            xs = px + pu * t0
            if fixed:
                tau = np.full(pool.size, spec.fixed_reach_s)
            else:
                tau = spec.move_time(xs - pos[0], py - pos[1])
                for _ in range(3):
                    tau = spec.move_time(xs + pu * tau - pos[0], py - pos[1])
            cx = xs + pu * tau
            ok = mask & sim.in_reach(arm, cx, py) & (np.abs(pu) <= spec.v_track_max_m_s)
            if not ok.any():
                return 0.0, mean_cycle  # it would idle: charge a cycle of waiting
            j_ok = np.nonzero(ok)[0]
            if fixed:
                cyc = np.full(j_ok.size, spec.fixed_cycle_s)
            else:
                b = bins[pc[j_ok]]
                cyc = tau[j_ok] + spec.grasp_dwell_s + spec.move_time(b[:, 0] - cx[j_ok], b[:, 1] - py[j_ok]) \
                    + spec.release_dwell_s
            best = j_ok[np.argsort(-(pev[j_ok] / cyc))[:3]]
            best_val, best_t = 0.0, mean_cycle
            best_rate = -1.0
            for j in best:
                c = float(cyc[np.searchsorted(j_ok, j)])
                bpos = bins[pc[j]] if not fixed else pos
                v2, t2 = plan(bpos, t0 + c, used | {int(j)}, depth - 1)
                val, tt = pev[j] + v2, c + t2
                if val / tt > best_rate:
                    best_rate, best_val, best_t = val / tt, val, tt
            return best_val, best_t

        best_c, best_rate = int(order[0]), -1.0
        for c in order:
            i = int(cand.idx[c])
            j = id_to_pool.get(i)
            used = {j} if j is not None else set()
            pos = bins[sim.cat[i]] if not fixed else sim.arms[arm].pos
            v2, t2 = plan(pos, float(cand.cycle[c]), used, self.depth - 1)
            val = float(cand.p[c] * cand.value[c] * w_c[c]) + v2
            rate = val / (float(cand.cycle[c]) + t2)
            if rate > best_rate:
                best_rate, best_c = rate, int(c)
        return best_c


PICK_LADDER = {
    "P0": FIFO,
    "P1": ShortestCycle,
    "P2": ValueRate,
    "P3": LastChance,
    "P4": Lookahead,
}

