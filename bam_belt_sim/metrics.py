"""What a run scores. Everything is counted after the warm-up so the empty-belt start does not flatter a policy.

* ``sppm``: successful picks per minute, the line's headline (Daily Note Nov 11: "Successful picks per minute ...
  at a single top-level KPI we can assume successful means pure as well").
* ``value_per_min``: the same, weighted by each item's value (the scheduling ladder's headline: "maximize value
  extracted"; with all values 1 it equals SPPM).
* ``recovery``: picked / (picked + lost) over items whose fate was settled in the window: the share of targeted
  material that did not go to landfill. Items with no bin anywhere are not counted (nobody could have picked them).
* ``backlog``: items waiting at the head at the end (stream supply); ``backlog_growth_per_min`` says whether the belt
  keeps up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .engine import Sim


@dataclass(frozen=True)
class RunMetrics:
    sppm: float
    value_per_min: float
    recovery: float
    value_recovery: float
    attempts_per_min: float
    grasp_fail_share: float
    missed_intercept_share: float
    utilisation: float
    mean_speed_m_s: float
    backlog_end: int
    backlog_growth_per_min: float
    peak_items_per_0_2m: int
    entered_per_min: float
    window_min: float
    sim_s_per_wall_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(sim: Sim, warmup_s: float, wall_s: float = 0.0) -> RunMetrics:
    t_end = sim.t
    window = max(t_end - warmup_s, 1e-9)
    minutes = window / 60.0
    targetable = sim.arm_cat_mask.any(axis=0)
    picks = [p for p in sim.log.picks if p[0] >= warmup_s]
    losses = [lo for lo in sim.log.losses if lo[0] >= warmup_s and targetable[lo[3]]]
    n_pick, n_loss = len(picks), len(losses)
    v_pick = sum(p[3] for p in picks)
    v_loss = sum(lo[2] for lo in losses)
    sim.finish_busy()
    busy = sum(a.busy_s for a in sim.arms)
    st = np.asarray(sim.log.speed_t)
    sv = np.asarray(sim.log.speed_v)
    in_win = st >= warmup_s
    mean_speed = float(sv[in_win, 0].mean()) if in_win.any() else 0.0
    q = np.asarray(sim.log.queue_len, float)
    qt = st
    growth = 0.0
    if in_win.sum() > 2:
        growth = float(np.polyfit(qt[in_win] / 60.0, q[in_win], 1)[0])
    entered = int(np.sum(sim.entered_t[: sim.next_fresh] >= warmup_s))
    return RunMetrics(
        sppm=n_pick / minutes,
        value_per_min=v_pick / minutes,
        recovery=n_pick / max(n_pick + n_loss, 1),
        value_recovery=v_pick / max(v_pick + v_loss, 1e-12),
        attempts_per_min=sim.log.attempts / (t_end / 60.0),
        grasp_fail_share=sim.log.failed_grasps / max(sim.log.attempts, 1),
        missed_intercept_share=sim.log.missed_intercepts / max(sim.log.attempts, 1),
        utilisation=busy / (len(sim.arms) * t_end),
        mean_speed_m_s=mean_speed,
        backlog_end=int(q[-1]) if q.size else 0,
        backlog_growth_per_min=growth,
        peak_items_per_0_2m=sim.log.peak_pile,
        entered_per_min=entered / minutes,
        window_min=minutes,
        sim_s_per_wall_s=(t_end / wall_s) if wall_s > 0 else 0.0,
    )
