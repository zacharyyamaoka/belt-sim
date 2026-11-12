"""Drill into one episode in Rerun: the belt, items and gantries in 3D on a sim-time timeline, with the stats the
ladders score computed live beside them (rolling and cumulative picks/min, recovery, belt speed per segment, backlog,
arm busy share) and a text log of every pick, miss and loss.

One .rrd per policy, same world and seed: open several at once (``rerun a.rrd b.rrd``) and switch between them in the
recordings panel, or scrub them side by side.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

from . import ladder as L
from . import pick_policies as P
from . import scenarios as SC
from . import speed_policies as S
from .engine import APPROACH, DELIVER, FREE, GRASP, Sim

RAIL_Z, GRIP_UP, GRIP_DOWN = 0.55, 0.32, 0.04
PHASE_RGB = {FREE: (150, 150, 150), APPROACH: (214, 39, 40), GRASP: (255, 127, 14), DELIVER: (44, 160, 44)}
RESULTS = Path(__file__).resolve().parent.parent / "docs" / "results"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))


def tuned_policies(scenario_name: str) -> dict:
    """Each rung with the knobs the ladder tuned for this world (results JSON), or defaults if none is saved."""
    speed = json.loads((RESULTS / "speed_ladder.json").read_text()) if (RESULTS / "speed_ladder.json").exists() else {}
    pick = json.loads((RESULTS / "pick_ladder.json").read_text()) if (RESULTS / "pick_ladder.json").exists() else {}
    if scenario_name in speed.get("scenarios", {}):
        rungs = speed["scenarios"][scenario_name]["rungs"]
        return {f"{r} {speed['rungs'][r]}": (lambda r=r: (L.make_speed_policy(r, rungs[r]["params"]), P.FIFO()))
                for r in rungs}
    if scenario_name in pick.get("scenarios", {}):
        rungs = pick["scenarios"][scenario_name]["rungs"]
        v = pick.get("belt_speed_m_s", L.PICK_SPEED_M_S)
        return {f"{r} {pick['rungs'][r]}": (lambda r=r: (S.Constant(v), L.make_pick_policy(r, rungs[r]["params"])))
                for r in rungs}
    return {
        "S0 constant 0.3 m/s": lambda: (S.Constant(0.3), P.FIFO()),
        "S3 deadline-aware": lambda: (S.DeadlineAware(0.3, v_floor=0.1, protect_n=2), P.FIFO()),
        "S4 one belt per arm": lambda: (S.Segmented(S.DeadlineAware(0.2, v_floor=0.0, protect_n=2)), P.FIFO()),
    }


def blueprint(sc):
    import rerun.blueprint as rrb

    centre = [sc.belt.length_m / 2, sc.belt.width_m / 2, 0.0]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world", name="Line",
                eye_controls=rrb.EyeControls3D(position=[centre[0], -sc.belt.length_m * 0.45, sc.belt.length_m * 0.55],
                                               look_target=centre, eye_up=[0, 0, 1]),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/stats/throughput", name="Picks / min"),
                rrb.TimeSeriesView(origin="/stats/belt_speed", name="Belt speed (m/s)"),
                rrb.TimeSeriesView(origin="/stats/line", name="Recovery · backlog · arms busy"),
                rrb.TextLogView(origin="/events", name="Events"),
                row_shares=[3, 3, 3, 2],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def log_episode(sc, sim: Sim, label: str, path: Path, t0: float) -> Path:
    """Write one recorded episode (``sim.log.frames``) to ``path`` as a Rerun recording."""
    import rerun as rr

    rec = rr.RecordingStream("belt_sim", recording_id=f"{sc.name} · {label}")
    rec.save(str(path))
    rec.send_blueprint(blueprint(sc))
    cat_rgb = np.array([_rgb(c.color) for c in sc.categories], np.uint8)
    names = [c.name for c in sc.categories]
    belt = sc.belt
    edges = belt.segment_edges()
    # static scene: belt outline, rails, legs, bins
    for k, arm in enumerate(sc.arms):
        lo, hi = arm.x_range
        strips = [[[lo, arm.y_min_m, RAIL_Z], [hi, arm.y_min_m, RAIL_Z]], [[lo, arm.y_max_m, RAIL_Z], [hi, arm.y_max_m, RAIL_Z]]]
        strips += [[[x, y, -0.3], [x, y, RAIL_Z]] for x in (lo, hi) for y in (arm.y_min_m, arm.y_max_m)]
        rec.log(f"world/arm{k}/frame", rr.LineStrips3D(strips, colors=[(80, 80, 85)], radii=0.012), static=True)
        bins = [(b.x, b.y, -0.11) for b in arm.bins]
        rec.log(f"world/arm{k}/bins", rr.Boxes3D(centers=bins, half_sizes=[(0.08, 0.08, 0.11)] * len(bins),
                                                 colors=[cat_rgb[sc.category_index()[b.category]] for b in arm.bins],
                                                 labels=[b.category for b in arm.bins], fill_mode="solid"),
                static=True)
    # pick / loss timelines for the live stats
    picks = sorted(sim.log.picks, key=lambda p: p[0])
    losses = sorted(sim.log.losses, key=lambda x: x[0])
    pick_t = [p[0] for p in picks]
    loss_t = [x[0] for x in losses]
    st, sv = np.asarray(sim.log.speed_t), np.asarray(sim.log.speed_v)
    q = np.asarray(sim.log.queue_len)
    busy_hist = []
    logged_pick = base_pick = bisect.bisect_left(pick_t, sim.log.frames[0]["t"])
    logged_loss = base_loss = bisect.bisect_left(loss_t, sim.log.frames[0]["t"])
    for f in sim.log.frames:
        t = f["t"]
        rec.set_time("sim_time", duration=t - t0)
        n = f["x"].size
        if n:
            rec.log("world/items", rr.Points3D(np.stack([f["x"], f["y"], np.full(n, 0.025)], 1),
                                               colors=cat_rgb[f["cat"]], radii=0.022))
        else:
            rec.log("world/items", rr.Clear(recursive=False))
        vmax = max(belt.v_max_m_s, 1e-6)
        centers, half, cols = [], [], []
        for j in range(belt.n_segments):
            s = min(abs(float(f["v"][j])) / vmax, 1.0)
            centers.append(((edges[j] + edges[j + 1]) / 2, belt.width_m / 2, -0.02))
            half.append(((edges[j + 1] - edges[j]) / 2 - 0.005, belt.width_m / 2, 0.02))
            cols.append((int(215 - 120 * s), int(215 - 60 * s), 225))
        rec.log("world/belt", rr.Boxes3D(centers=centers, half_sizes=half, colors=cols, fill_mode="solid"))
        busy = 0
        for k, (arm, state) in enumerate(zip(sc.arms, f["arms"], strict=True)):
            tx, ty, phase, _item, z, grip, held = state
            busy += phase != FREE
            tip = GRIP_DOWN + (GRIP_UP - GRIP_DOWN) * z
            gap = 0.014 + 0.03 * grip
            ymid = (arm.y_min_m + arm.y_max_m) / 2
            yhalf = (arm.y_max_m - arm.y_min_m) / 2 + 0.03
            rod_mid = (RAIL_Z + tip + 0.07) / 2
            rod_half = max((RAIL_Z - tip - 0.07) / 2, 0.005)
            rec.log(f"world/arm{k}/gantry", rr.Boxes3D(
                centers=[(tx, ymid, RAIL_Z + 0.03), (tx, ty, RAIL_Z + 0.03), (tx, ty, rod_mid),
                         (tx - gap, ty, tip + 0.035), (tx + gap, ty, tip + 0.035)],
                half_sizes=[(0.03, yhalf, 0.025), (0.05, 0.05, 0.04), (0.012, 0.012, rod_half),
                            (0.006, 0.025, 0.035), (0.006, 0.025, 0.035)],
                colors=[(150, 150, 155), PHASE_RGB[phase], (200, 200, 205), (30, 30, 30), (30, 30, 30)],
                fill_mode="solid"))
            if held >= 0:
                rec.log(f"world/arm{k}/held", rr.Points3D([(tx, ty, tip + 0.02)], colors=[cat_rgb[held]], radii=0.024))
            else:
                rec.log(f"world/arm{k}/held", rr.Clear(recursive=False))
        busy_hist.append((t, busy / max(len(sc.arms), 1)))
        # live stats, computed from what has happened up to t
        n_win = bisect.bisect_right(pick_t, t) - bisect.bisect_right(pick_t, t - 60.0)
        # WHY over the past 60 s of sim time, not since recording started: the episode was warmed up before t0
        window = min(60.0, t)
        rec.log("stats/throughput/rolling_60s", rr.Scalars(n_win * 60.0 / window))
        n_pick = bisect.bisect_right(pick_t, t) - base_pick
        n_loss = bisect.bisect_right(loss_t, t) - base_loss
        rec.log("stats/throughput/since_start", rr.Scalars(n_pick * 60.0 / max(t - t0, 1.0)))
        rec.log("stats/line/recovery", rr.Scalars(n_pick / max(n_pick + n_loss, 1)))
        i = min(int(np.searchsorted(st, t, side="right")) - 1, len(st) - 1)
        if i >= 0:
            for j in range(belt.n_segments):
                rec.log(f"stats/belt_speed/belt{j}", rr.Scalars(float(sv[i, j])))
            rec.log("stats/line/backlog_items_div10", rr.Scalars(float(q[i]) / 10.0))
        recent = [b for tt, b in busy_hist if tt > t - 10.0]
        rec.log("stats/line/arms_busy_10s", rr.Scalars(float(np.mean(recent))))
        # events since the last frame
        while logged_pick < len(picks) and picks[logged_pick][0] <= t:
            p = picks[logged_pick]
            lap = f", lap {p[5] + 1}" if len(p) > 5 and p[5] else ""
            rec.log("events", rr.TextLog(f"arm{p[2] + 1} picked {names[p[4]]} (value {p[3]:g}{lap})", level="INFO"))
            logged_pick += 1
        while logged_loss < len(losses) and losses[logged_loss][0] <= t:
            x = losses[logged_loss]
            rec.log("events", rr.TextLog(f"lost {names[x[3]]} off the end", level="WARN"))
            logged_loss += 1
    if hasattr(rec, "flush"):
        rec.flush()
    return path


def record(sc, speed, pick, seed: int, t_start: float, t_len: float, every: int = 2) -> Sim:
    sim = Sim(sc, speed, pick, seed=seed)
    sim.run(t_start)
    sim.record_every = every
    sim.run(t_len)
    return sim


def write_replays(scenario_name: str, out_dir: Path, seconds: float = 60.0, seed: int = 100, t_start: float = 120.0,
                  only: list[str] | None = None) -> list[Path]:
    all_sc = {**SC.speed_benchmark(), **SC.circulation_benchmark(), **SC.layout_benchmark(), **SC.pick_benchmark(),
              **SC.pick_loop_benchmark()}
    sc = all_sc[scenario_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for label, make in tuned_policies(scenario_name).items():
        rung = label.split(" ")[0]
        if only and rung not in only:
            continue
        speed, pick = make()
        sim = record(sc, speed, pick, seed, t_start, seconds)
        slug = "".join(ch if ch.isalnum() else "_" for ch in f"{scenario_name}_{rung}").strip("_")
        paths.append(log_episode(sc, sim, label, out_dir / f"{slug}.rrd", sim.log.frames[0]["t"]))
    return paths
