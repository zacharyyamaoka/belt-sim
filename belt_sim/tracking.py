"""Weights & Biases pages for the two black boxes: belt speed control and pick scheduling.

Every configuration a ladder tried becomes one W&B run (job_type "tune": its knobs as config, the tuning-seed means as
summary), so W&B's parallel-coordinates and scatter panels read like a sweep. Each rung's tuned pick also gets a
"eval" run with time series over one evaluation episode (picks/min, recovery, belt speed, backlog, arms busy).

Runs are written offline first and uploaded in one ``wandb sync`` (hundreds of online inits are slow).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import ladder as L
from . import pick_policies as P
from . import speed_policies as S
from .engine import FREE, Sim

PROJECTS = {"speed": "belt-sim-speed-control", "pick": "belt-sim-pick-scheduling"}
WARMUP_S, EPISODE_S, SAMPLE_S = 120.0, 300.0, 5.0


def episode_series(args) -> tuple:
    """Run one evaluation episode and sample the live stats every SAMPLE_S of sim time."""
    kind, scen_name, rung, params, seed = args
    sc = L._all_scenarios()[scen_name]
    if kind == "speed":
        speed, pick = L.make_speed_policy(rung, params), P.FIFO()
    else:
        speed, pick = S.Constant(L.PICK_SPEED_M_S), L.make_pick_policy(rung, params)
    sim = Sim(sc, speed, pick, seed=seed)
    sim.run(WARMUP_S)
    rows = []
    busy_acc, n_acc = 0.0, 0
    p0, l0, v0 = len(sim.log.picks), len(sim.log.losses), 0.0
    for _ in range(int(EPISODE_S / SAMPLE_S)):
        for _ in range(int(SAMPLE_S / sim.dt)):
            sim.step()
            busy_acc += sum(a.phase != FREE for a in sim.arms) / len(sim.arms)
            n_acc += 1
        t = sim.t - WARMUP_S
        picks = sim.log.picks[p0:]
        recent = [p for p in picks if p[0] > sim.t - 60.0]
        n_loss = len(sim.log.losses) - l0
        rows.append({
            "sim_time_s": t,
            "picks_per_min/rolling_60s": len(recent) * 60.0 / min(60.0, t),
            "picks_per_min/since_start": len(picks) * 60.0 / t,
            "value_per_min/since_start": sum(p[3] for p in picks) * 60.0 / t,
            "recovery": len(picks) / max(len(picks) + n_loss, 1),
            "backlog_items": sim.queue_length(),
            "arms_busy": busy_acc / max(n_acc, 1),
            "items_on_belt": int(sim.on_belt.size),
            **{f"belt_speed/belt{j}": float(v) for j, v in enumerate(sim.v)},
        })
        busy_acc, n_acc = 0.0, 0
    _ = v0
    return kind, scen_name, rung, rows


def log_all(results_dir: Path, out_dir: Path, workers: int | None = None, upload: bool = True, log=print) -> dict:
    import wandb

    os.environ.setdefault("WANDB_SILENT", "true")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {k: json.loads((results_dir / f"{k}_ladder.json").read_text()) for k in ("speed", "pick")}
    jobs = []
    for kind, d in data.items():
        for n, scd in d["scenarios"].items():
            for r, row in scd["rungs"].items():
                jobs.append((kind, n, r, row["params"], L.EVAL_SEEDS[0]))
    workers = workers or max(1, (os.cpu_count() or 4) - 4)
    log(f"W&B: {len(jobs)} evaluation episodes for time series on {workers} workers")
    series = {}
    with ProcessPoolExecutor(workers) as ex:
        for kind, n, r, rows in ex.map(episode_series, jobs):
            series[(kind, n, r)] = rows
    n_runs = 0
    for kind, d in data.items():
        project = PROJECTS[kind]
        for n, scd in d["scenarios"].items():
            describe = scd.get("describe", {})
            base_cfg = {"scenario": n, "loop": scd.get("loop", False), "arms": describe.get("arms"),
                        "supply": describe.get("supply"), "segments": describe.get("segments")}
            for r, pts in scd.get("frontier", {}).items():
                for pt in pts:
                    params = pt["params"]
                    run = wandb.init(project=project, dir=str(out_dir), mode="offline", group=n, job_type="tune",
                                     name=f"{r} {json.dumps(params, sort_keys=True)}",
                                     config={**base_cfg, "rung": r, "rung_title": d["rungs"][r], **params},
                                     tags=[r, "tune"], reinit="create_new")
                    run.summary.update({k: v for k, v in pt.items() if k != "params"})
                    run.finish()
                    n_runs += 1
            for r, row in scd["rungs"].items():
                run = wandb.init(project=project, dir=str(out_dir), mode="offline", group=n, job_type="eval",
                                 name=f"{r} {d['rungs'][r]} · eval",
                                 config={**base_cfg, "rung": r, "rung_title": d["rungs"][r], **row["params"],
                                         "eval_seeds": list(L.EVAL_SEEDS)},
                                 tags=[r, "eval"], reinit="create_new")
                run.define_metric("sim_time_s")
                run.define_metric("*", step_metric="sim_time_s")
                for rec in series[(kind, n, r)]:
                    run.log(rec)
                run.summary.update({f"eval/{k}": v for k, v in row.items()
                                    if isinstance(v, int | float) and not isinstance(v, bool)})
                run.finish()
                n_runs += 1
    log(f"W&B: wrote {n_runs} offline runs under {out_dir}")
    if upload:
        runs = sorted(str(p) for p in (out_dir / "wandb").glob("offline-run-*"))
        for i in range(0, len(runs), 50):
            subprocess.run([str(Path(sys.executable).parent / "wandb"), "sync", *runs[i:i + 50]], check=False,
                           stdout=subprocess.DEVNULL)
        log("W&B: synced")
    api = wandb.Api()
    entity = api.default_entity
    return {kind: f"https://wandb.ai/{entity}/{p}" for kind, p in PROJECTS.items()} | {"runs": n_runs}


_ = np
