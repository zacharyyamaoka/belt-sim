"""An LP upper bound on the expected value any pick policy can extract from a constant-speed line (scheduling ladder).

Setting: one belt at constant speed v, stream supply, no recirculation. Then every item's entry time and position are
fixed by the seed alone (the infeed spaces items by belt odometry, not by what arms do), so its reach window at each
arm is known in advance: the clairvoyant view.

Time-indexed LP over slots of ``slot_s``. Variable y[k,i,s] = expected number of attempts arm k makes on item i with
contact in slot s; z[i] = probability item i is picked.

    max  sum_i value_i z_i
    s.t. z_i <= 1,   z_i <= p_i sum_{k,s} y[k,i,s]           (P(picked) <= min(1, p * E[attempts]))
         for every arm k and slot t:  sum of the expected occupancy of slot t <= 1
         y >= 0

Occupancy of one attempt: the grasp dwell after contact always, and with probability p the move to the bin and the
release. The move *to* the item is not charged (after a failed grasp the tool is already at the belt, so no positive
lower bound holds for every attempt), and durations are rounded down to whole slots and start one slot late. Every
relaxation only loosens the bound, so it stays a bound: the expected value of any policy's run is a feasible point.

``reference=True`` also charges the move from the arm's nearest bin to the item before contact. That holds whenever
the previous grasp succeeded (most of the time at p ~ 0.9) but not after a miss, so it is a tighter *reference*, not a
proof. The report shows both.

Arms start free at the window start and items entering before it are not counted, which only flatters the LP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Scenario
from .engine import Sim


class _NeverPick:
    name = "never pick"

    def choose(self, sim, arm, cand):
        return None


class _ConstantSpeed:
    def __init__(self, v):
        self.v = v
        self.name = f"constant {v}"

    def command(self, sim):
        return np.full(sim.n_seg, self.v)


@dataclass(frozen=True)
class BoundResult:
    value: float  # bound on expected value extracted from items entering in [t0, t1]
    n_items: int
    n_vars: int
    status: str
    item_ids: np.ndarray


def item_arrivals(scenario: Scenario, v: float, seed: int, t_end: float, dt: float = 0.05):
    """Entry time and position of every item (policy-independent at constant speed: run once with no picking)."""
    sim = Sim(scenario, _ConstantSpeed(v), _NeverPick(), seed=seed, dt=dt).run(t_end)
    n = sim.next_fresh
    return sim, sim.entered_t[:n].copy(), sim.entered_x[:n].copy()


def lp_bound(scenario: Scenario, v: float, seed: int, t0: float, t1: float, tail_s: float = 30.0,
             slot_s: float = 0.1, reference: bool = False, dt: float = 0.05) -> BoundResult:
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    sim, et, ex = item_arrivals(scenario, v, seed, t1 + 1.0, dt)
    ids = np.nonzero((et >= t0) & (et < t1))[0]
    cats = sim.cat[ids]
    ys = sim.y[ids]
    ps = sim.p[ids]
    vals = sim.value[ids]
    n_slots = int(np.ceil((t1 + tail_s - t0) / slot_s))
    var_item, var_arm, var_slot, cols_rows, cols_coef = [], [], [], [], []
    for k, arm in enumerate(scenario.arms):
        bins = sim.arm_bins[k]
        ok_cat = sim.arm_cat_mask[k, cats]
        for local in np.nonzero(ok_cat)[0]:
            y = ys[local]
            if not (arm.y_min_m <= y <= arm.y_max_m):
                continue
            lo, hi = arm.x_range
            i = ids[local]
            t_in = et[i] + (lo - ex[i]) / v
            # the grasp dwell must also end inside the stroke (as in the engine)
            t_out = et[i] + (hi - ex[i]) / v - arm.grasp_dwell_s
            t_in = max(t_in, et[i])
            if t_out <= t_in or v > arm.v_track_max_m_s:
                continue
            s_lo = max(int(np.floor((t_in - t0) / slot_s)), 0)
            s_hi = min(int(np.floor((t_out - t0) / slot_s)), n_slots - 1)
            b = bins[cats[local]]
            for s in range(s_lo, s_hi + 1):
                c = t0 + (s + 0.5) * slot_s
                cx = ex[i] + v * (c - et[i])
                t_bin = arm.move_time(float(b[0] - cx), float(b[1] - y))
                n_always = int(np.floor(arm.grasp_dwell_s / slot_s))
                n_succ = int(np.floor((arm.grasp_dwell_s + t_bin + arm.release_dwell_s) / slot_s)) - n_always
                rows, coef = [], []
                # one slot late: contact may fall anywhere in slot s
                for r in range(s + 1, min(s + n_always, n_slots)):
                    rows.append(r)
                    coef.append(1.0)
                for r in range(s + max(n_always, 1), min(s + n_always + n_succ, n_slots)):
                    rows.append(r)
                    coef.append(float(ps[local]))
                if reference:
                    own = bins[sim.arm_cat_mask[k]]
                    t_near = float(np.min(arm.move_time(own[:, 0] - cx, own[:, 1] - y)))
                    n_in = int(np.floor(t_near / slot_s))
                    for r in range(max(s - n_in + 1, 0), s + 1):
                        rows.append(r)
                        coef.append(1.0)
                var_item.append(local)
                var_arm.append(k)
                var_slot.append(s)
                cols_rows.append([k * n_slots + r for r in rows])
                cols_coef.append(coef)
    n_y = len(var_item)
    n_items = ids.size
    if n_y == 0:
        return BoundResult(0.0, int(n_items), 0, "no feasible attempts", ids)
    n_arm_rows = len(scenario.arms) * n_slots
    # Variables: [y (n_y), z (n_items)]. Rows: arm-slot capacity, then z_i - p_i sum y <= 0.
    r_idx, c_idx, data = [], [], []
    for j, (rows, coef) in enumerate(zip(cols_rows, cols_coef, strict=True)):
        r_idx.extend(rows)
        c_idx.extend([j] * len(rows))
        data.extend(coef)
    var_item = np.asarray(var_item)
    r_idx.extend((n_arm_rows + var_item).tolist())
    c_idx.extend(range(n_y))
    data.extend((-ps[var_item]).tolist())
    r_idx.extend((n_arm_rows + np.arange(n_items)).tolist())
    c_idx.extend((n_y + np.arange(n_items)).tolist())
    data.extend([1.0] * n_items)
    A = coo_matrix((data, (r_idx, c_idx)), shape=(n_arm_rows + n_items, n_y + n_items)).tocsr()
    b = np.concatenate([np.ones(n_arm_rows), np.zeros(n_items)])
    cost = np.concatenate([np.zeros(n_y), -vals])
    bounds = [(0, None)] * n_y + [(0, 1)] * n_items
    res = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return BoundResult(float(-res.fun) if res.status == 0 else float("nan"), int(n_items), n_y, res.message, ids)


def policy_value_on(sim: Sim, ids: np.ndarray) -> float:
    """Value a finished run extracted from exactly the bound's item set."""
    want = set(int(i) for i in ids)
    return float(sum(p[3] for p in sim.log.picks if p[1] in want))
