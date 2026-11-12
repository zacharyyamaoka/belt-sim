"""Climb both ladders: tune each rung's knobs on tuning seeds, score it on separate evaluation seeds.

WHY two seed sets: a rung whose parameters were picked on the seeds it is scored on would be graded on its own
homework. Tuning seeds 0-2, evaluation seeds 100-104, same items for every rung (common random numbers).

Speed ladder headline, hopper supply: **SPPM at recovery >= 90 %** - the most successful picks per minute a policy
reaches while sending at most a tenth of the targeted items to landfill. Without that gate "run at max speed" wins
SPPM and loses everything else ("if you just want maximum throughput then just run it at max speed and it all goes
to waste out the other side"). Stream supply: SPPM with the gate "the backlog does not grow". Recirculating belt:
plain SPPM - a miss comes round again, so throughput is the question.

Pick ladder headline: value per minute, and its share of the LP reference and bound (bound.py).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import pick_policies as P
from . import scenarios as SC
from . import speed_policies as S
from .engine import Sim
from .metrics import summarize

TUNE_SEEDS = (0, 1, 2)
EVAL_SEEDS = (100, 101, 102, 103, 104)
RECOVERY_GATE = 0.90
BACKLOG_GATE_PER_MIN = 0.5
WARMUP_S = 120.0


@dataclass(frozen=True)
class SpeedRung:
    rung: str
    title: str
    grid: tuple  # tuple of param dicts


V_GRID = (0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5)
S3_GRID = tuple(dict(v_cruise=c, v_floor=f, protect_n=n)
                for c, f, n in itertools.product((0.1, 0.2, 0.3, 0.45), (0.0, 0.1), (1, 2)))

SPEED_RUNGS = (
    SpeedRung("S0", "constant 0.3 m/s (v1)", (dict(v=0.3),)),
    SpeedRung("S1", "best constant speed", tuple(dict(v=v) for v in V_GRID)),
    SpeedRung("S2", "rush when idle", tuple(dict(v_cruise=v) for v in V_GRID)),
    SpeedRung("S3", "deadline-aware", S3_GRID),
    SpeedRung("S4", "segmented belts (one per arm)", S3_GRID),
)


def make_speed_policy(rung: str, params: dict):
    if rung in ("S0", "S1"):
        return S.Constant(params["v"], name=f"{rung} constant {params['v']:g} m/s")
    if rung == "S2":
        return S.RushWhenIdle(**params)
    if rung == "S3":
        return S.DeadlineAware(**params)
    if rung == "S4":
        return S.Segmented(S.DeadlineAware(**params))
    raise KeyError(rung)


def _all_scenarios() -> dict:
    out = {}
    out.update(SC.speed_benchmark())
    out.update(SC.circulation_benchmark())
    out.update(SC.pick_benchmark())
    out.update(SC.layout_benchmark())
    out.update(SC.pick_loop_benchmark())
    return out


def _run_speed(args):
    scen_name, rung, params, seed, duration = args
    sc = _all_scenarios()[scen_name]
    pol = make_speed_policy(rung, params)
    t0 = time.perf_counter()
    sim = Sim(sc, pol, P.FIFO(), seed=seed).run(WARMUP_S + duration)
    m = summarize(sim, WARMUP_S, time.perf_counter() - t0)
    return scen_name, rung, json.dumps(params, sort_keys=True), seed, m.as_dict()


def _passes(sc, m: dict) -> bool:
    if sc.belt.circulating:
        return True  # WHY no gate on a loop: a miss comes round again, so the question is plain picks/min
    if sc.supply.mode == "stream":
        return m["backlog_growth_per_min"] <= BACKLOG_GATE_PER_MIN
    return m["recovery"] >= RECOVERY_GATE


def _score(sc, m: dict) -> float:
    """Tuning objective: SPPM if the gate passes; otherwise below every passing config, ordered by how close."""
    if _passes(sc, m):
        return m["sppm"]
    if sc.supply.mode == "stream":
        return -m["backlog_growth_per_min"]
    return -1000.0 * (RECOVERY_GATE - m["recovery"]) + 1e-3 * m["sppm"]


FRONTIER_KEYS = ("sppm", "sppm_sd", "recovery", "recovery_sd", "value_per_min", "utilisation", "mean_speed_m_s",
                 "missed_intercept_share", "passes_per_pick", "backlog_growth_per_min", "window_value")


def _keep(m: dict) -> dict:
    """The frontier stores what the report and W&B read; every metric of every config made the file 700 KB."""
    return {k: m[k] for k in FRONTIER_KEYS if k in m}


def _mean(ms: list[dict]) -> dict:
    keys = ms[0].keys()
    return {k: float(np.mean([m[k] for m in ms])) for k in keys} | {
        "sppm_sd": float(np.std([m["sppm"] for m in ms], ddof=1)) if len(ms) > 1 else 0.0,
        "recovery_sd": float(np.std([m["recovery"] for m in ms], ddof=1)) if len(ms) > 1 else 0.0,
    }


def climb_speed(scenario_names, tune_s=600.0, eval_s=900.0, workers=None, log=print) -> dict:
    scs = _all_scenarios()
    workers = workers or max(1, (os.cpu_count() or 4) - 4)
    jobs = [(n, r.rung, p, s, tune_s) for n in scenario_names for r in SPEED_RUNGS for p in r.grid
            for s in TUNE_SEEDS]
    log(f"speed ladder: tuning {len(jobs)} runs on {workers} workers")
    t0 = time.time()
    tune: dict = {}
    with ProcessPoolExecutor(workers) as ex:
        for n, rung, pj, _seed, m in ex.map(_run_speed, jobs, chunksize=4):
            tune.setdefault((n, rung, pj), []).append(m)
    log(f"  tuned in {time.time() - t0:.0f} s")
    best: dict = {}
    frontier: dict = {}
    for (n, rung, pj), ms in tune.items():
        mm = _mean(ms)
        frontier.setdefault(n, {}).setdefault(rung, []).append({"params": json.loads(pj), **_keep(mm)})
        sc = scs[n]
        sc_val = _score(sc, mm)
        if (n, rung) not in best or sc_val > best[(n, rung)][0]:
            best[(n, rung)] = (sc_val, json.loads(pj))
    ejobs = [(n, rung, best[(n, rung)][1], s, eval_s) for n in scenario_names for rung in [r.rung for r in SPEED_RUNGS]
             for s in EVAL_SEEDS]
    t0 = time.time()
    evals: dict = {}
    with ProcessPoolExecutor(workers) as ex:
        for n, rung, _pj, _seed, m in ex.map(lambda_args_speed, ejobs, chunksize=2):
            evals.setdefault((n, rung), []).append(m)
    log(f"  evaluated {len(ejobs)} runs in {time.time() - t0:.0f} s")
    out = {"gate": {"recovery": RECOVERY_GATE, "backlog_growth_per_min": BACKLOG_GATE_PER_MIN},
           "tune_seeds": TUNE_SEEDS, "eval_seeds": EVAL_SEEDS, "tune_s": tune_s, "eval_s": eval_s,
           "rungs": {r.rung: r.title for r in SPEED_RUNGS}, "scenarios": {}}
    for n in scenario_names:
        sc = scs[n]
        rows = {}
        for r in SPEED_RUNGS:
            mm = _mean(evals[(n, r.rung)])
            # WHY no per-run dicts: mean and sd are what the report reads; the runs made the file 700 KB.
            rows[r.rung] = {"params": best[(n, r.rung)][1], "passes_gate": _passes(sc, mm), **mm}
        out["scenarios"][n] = {"capacity_ppm": sc.capacity_ppm(), "supply": sc.supply.mode, "rungs": rows,
                               "loop": sc.belt.circulating,
                               "frontier": frontier[n], "describe": describe(sc)}
    return out


def lambda_args_speed(args):
    return _run_speed(args)


def describe(sc) -> dict:
    return {
        "arms": len(sc.arms),
        "categories": [c.name for c in sc.categories],
        "bins": {a.name: [b.category for b in a.bins] for a in sc.arms},
        "supply": sc.supply.mode,
        "density_per_m": sc.supply.density_per_m,
        "rate_per_s": sc.supply.rate_per_s,
        "burstiness": sc.supply.burstiness,
        "belt_m": sc.belt.length_m,
        "segments": sc.belt.n_segments,
        "circulating": sc.belt.circulating,
    }


# ----------------------------------------------------------------------------------------------- pick ladder
PICK_SPEED_M_S = 0.25
PICK_RUNGS = {
    "P0": ("FIFO (v1)", (dict(),)),
    "P1": ("shortest cycle", (dict(),)),
    "P2": ("value rate p*v/cycle", (dict(),)),
    "P3": ("last chance", tuple(dict(downstream_weight=w) for w in (0.2, 0.5, 0.8))),
    "P4": ("lookahead", tuple(dict(depth=d, downstream_weight=w) for d in (2, 3) for w in (0.2, 0.5))),
}


def make_pick_policy(rung: str, params: dict):
    return P.PICK_LADDER[rung](**params)


def _run_pick(args):
    scen_name, rung, params, seed, duration = args
    sc = _all_scenarios()[scen_name]
    from .bound import policy_value_on

    t0 = time.perf_counter()
    sim = Sim(sc, S.Constant(PICK_SPEED_M_S), make_pick_policy(rung, params), seed=seed).run(WARMUP_S + duration + 30)
    wall = time.perf_counter() - t0
    m = summarize(sim, WARMUP_S, wall).as_dict()
    # Items entering in the bound's window: what the policy got from them (value), and the gold share if any.
    window = np.nonzero((sim.entered_t[: sim.next_fresh] >= WARMUP_S)
                        & (sim.entered_t[: sim.next_fresh] < WARMUP_S + duration))[0]
    m["window_value"] = policy_value_on(sim, window)
    names = [c.name for c in sc.categories]
    if "gold" in names:
        g = names.index("gold")
        gold = window[sim.cat[window] == g]
        got = sum(1 for p in sim.log.picks if p[4] == g and p[1] in set(gold.tolist()))
        m["gold_items"], m["gold_picked"] = int(gold.size), int(got)
    return scen_name, rung, json.dumps(params, sort_keys=True), seed, m


def _run_bound(args):
    scen_name, seed, duration, reference = args
    from .bound import lp_bound

    sc = _all_scenarios()[scen_name]
    b = lp_bound(sc, PICK_SPEED_M_S, seed, WARMUP_S, WARMUP_S + duration, reference=reference)
    return scen_name, seed, reference, b.value


def climb_pick(scenario_names, duration=300.0, workers=None, log=print) -> dict:
    workers = workers or max(1, (os.cpu_count() or 4) - 4)
    jobs = [(n, r, p, s, duration) for n in scenario_names for r, (_t, grid) in PICK_RUNGS.items() for p in grid
            for s in TUNE_SEEDS + EVAL_SEEDS]
    scs = _all_scenarios()
    # WHY no LP on a loop: an item's re-entry depends on what the arms did, so its windows are not known in advance
    bjobs = [(n, s, duration, ref) for n in scenario_names if not scs[n].belt.circulating for s in EVAL_SEEDS
             for ref in (False, True)]
    log(f"pick ladder: {len(jobs)} runs + {len(bjobs)} LPs on {workers} workers")
    t0 = time.time()
    res: dict = {}
    bounds: dict = {}
    with ProcessPoolExecutor(workers) as ex:
        fb = [ex.submit(_run_bound, a) for a in bjobs]
        for n, rung, pj, seed, m in ex.map(_run_pick, jobs, chunksize=2):
            res.setdefault((n, rung, pj), {})[seed] = m
        for f in fb:
            n, seed, ref, val = f.result()
            bounds.setdefault(n, {}).setdefault("reference" if ref else "bound", {})[seed] = val
    log(f"  done in {time.time() - t0:.0f} s")
    out = {"belt_speed_m_s": PICK_SPEED_M_S, "duration_s": duration, "tune_seeds": TUNE_SEEDS,
           "eval_seeds": EVAL_SEEDS, "rungs": {r: t for r, (t, _g) in PICK_RUNGS.items()}, "scenarios": {}}
    for n in scenario_names:
        rows = {}
        loop = scs[n].belt.circulating
        key = "value_per_min" if loop else "window_value"
        frontier = {}
        for r, (_t, grid) in PICK_RUNGS.items():
            # tune: best mean window value on tuning seeds
            best_p, best_v = None, -math.inf
            for p in grid:
                pj = json.dumps(p, sort_keys=True)
                v = np.mean([res[(n, r, pj)][s][key] for s in TUNE_SEEDS])
                frontier.setdefault(r, []).append({"params": p, **_keep(_mean([
                    {k: x for k, x in res[(n, r, pj)][s].items() if isinstance(x, int | float)} for s in TUNE_SEEDS]))})
                if v > best_v:
                    best_p, best_v = pj, v
            ev = [res[(n, r, best_p)][s] for s in EVAL_SEEDS]
            row = _mean([{k: v for k, v in m.items() if isinstance(v, int | float)} for m in ev])
            if not loop:
                ref = [bounds[n]["reference"][s] for s in EVAL_SEEDS]
                bnd = [bounds[n]["bound"][s] for s in EVAL_SEEDS]
                row["share_of_reference"] = float(np.mean([m["window_value"] / x for m, x in zip(ev, ref, strict=True)]))
                row["share_of_bound"] = float(np.mean([m["window_value"] / x for m, x in zip(ev, bnd, strict=True)]))
            row["params"] = json.loads(best_p)
            if "gold_items" in ev[0]:
                gi = sum(m["gold_items"] for m in ev)
                row["gold_recovery"] = sum(m["gold_picked"] for m in ev) / max(gi, 1)
                row["gold_items_total"] = gi
            rows[r] = row
        out["scenarios"][n] = {"rungs": rows, "bound": bounds.get(n), "describe": describe(scs[n]), "loop": loop,
                               "frontier": frontier,
                               "value_unit": "USD" if "gold" in [c.name for c in scs[n].categories] or
                               "PET" in [c.name for c in scs[n].categories] else "picks"}
    return out


def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
