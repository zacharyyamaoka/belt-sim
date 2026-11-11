"""2D replays (matplotlib, headless): top-down belt, items as dots by category, arms' reach discs, tool and bins.

Each replay is several policies on the SAME seed and time window, stacked, so the only difference on screen is the
policy. Written as MP4 (H.264, for the report's hero video), GIF (fallback) and a poster PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from . import pick_policies as P  # noqa: E402
from . import scenarios as SC  # noqa: E402
from . import speed_policies as S  # noqa: E402
from .engine import APPROACH, DELIVER, GRASP, Sim  # noqa: E402

PHASE_COLOR = {0: "#444444", APPROACH: "#d62728", GRASP: "#ff7f0e", DELIVER: "#2ca02c"}


def record(scenario, speed, pick, seed=100, t_start=150.0, t_len=20.0, every=2):
    sim = Sim(scenario, speed, pick, seed=seed)
    sim.run(t_start)
    picks0, lost0 = len(sim.log.picks), len(sim.log.losses)
    sim.record_every = every
    sim.run(t_len)
    for f in sim.log.frames:
        f["picked"] -= picks0
        f["lost"] -= lost0
    return sim


def _draw_panel(ax, sim: Sim, frame: dict, title: str, gold_idx: int | None):
    sc = sim.sc
    belt = sc.belt
    ax.clear()
    ax.set_xlim(-0.2, belt.length_m + 0.2)
    ax.set_ylim(-0.45, belt.width_m + 0.62)
    ax.set_aspect("equal")
    ax.axis("off")
    edges = belt.segment_edges()
    vmax = max(belt.v_max_m_s, 1e-6)
    for j in range(belt.n_segments):
        v = frame["v"][j]
        shade = 0.93 - 0.25 * min(abs(v) / vmax, 1.0)
        col = (shade, shade, shade + (0.05 if v >= 0 else -0.05))
        ax.add_patch(Rectangle((edges[j], 0), edges[j + 1] - edges[j], belt.width_m, facecolor=col,
                               edgecolor="#999999", lw=0.6))
        if belt.n_segments > 1:
            ax.text((edges[j] + edges[j + 1]) / 2, belt.width_m + 0.36, f"{v:+.2f}", ha="center", va="center",
                    fontsize=6.5, color="#333333")
    for k, arm in enumerate(sc.arms):
        ax.add_patch(Circle((arm.x, arm.y), arm.reach_m, fill=False, ls="--", lw=0.6, color="#777777"))
        for b in arm.bins:
            ci = sc.category_index()[b.category]
            ax.add_patch(Rectangle((b.x - 0.07, b.y - 0.07), 0.14, 0.14, facecolor=sc.categories[ci].color,
                                   edgecolor="black", lw=0.5, alpha=0.85))
        tx, ty, phase, _item = frame["arms"][k]
        ax.plot([arm.x, tx], [arm.y, ty], color="#555555", lw=1.2, solid_capstyle="round")
        ax.add_patch(Circle((arm.x, arm.y), 0.035, color="#222222"))
        ax.add_patch(Circle((tx, ty), 0.04, facecolor="white", edgecolor=PHASE_COLOR[phase], lw=1.8))
    cols = [sc.categories[c].color for c in frame["cat"]]
    sizes = np.where(frame["cat"] == gold_idx, 70, 22) if gold_idx is not None else 22
    edge = np.where(frame["reserved"] >= 0, "black", "none")
    ax.scatter(frame["x"], frame["y"], s=sizes, c=cols, edgecolors=edge, linewidths=0.8, zorder=3)
    if belt.n_segments == 1 or not belt.boundaries_m:
        ax.text(belt.length_m / 2, belt.width_m + 0.36, f"belt {frame['v'][0]:+.2f} m/s", ha="center",
                fontsize=7, color="#333333")
    ax.text(-0.15, belt.width_m + 0.56, title, ha="left", va="center", fontsize=8.5, weight="bold")
    ax.text(belt.length_m + 0.15, -0.36, f"picked {frame['picked']}   lost {frame['lost']}", ha="right",
            va="center", fontsize=8)


def render_compare(runs: list[tuple[str, Sim]], out_stem: Path, fps: int = 15, dpi: int = 110) -> dict:
    """Stack the runs' frames into one video. Returns paths written."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    sc = runs[0][1].sc
    names = [c.name for c in sc.categories]
    gold_idx = names.index("gold") if "gold" in names else None
    n = min(len(sim.log.frames) for _t, sim in runs)
    width = 1.6 + sc.belt.length_m * 1.25
    fig, axes = plt.subplots(len(runs), 1, figsize=(width, 1.55 * len(runs) + 0.3), dpi=dpi)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, hspace=0.05)
    frames = []
    for fi in range(n):
        for ax, (title, sim) in zip(axes, runs, strict=True):
            _draw_panel(ax, sim, sim.log.frames[fi], title, gold_idx)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        h, w = img.shape[:2]
        frames.append(img[: h - h % 2, : w - w % 2])
    plt.close(fig)
    import imageio.v2 as imageio

    mp4 = out_stem.with_suffix(".mp4")
    imageio.mimwrite(mp4, frames, fps=fps, codec="libx264", quality=5, pixelformat="yuv420p", macro_block_size=2)
    gif = out_stem.with_suffix(".gif")
    imageio.mimwrite(gif, frames[::2], duration=2.0 / fps, loop=0)
    png = out_stem.with_suffix(".png")
    imageio.imwrite(png, frames[len(frames) // 2])
    return {"mp4": mp4, "gif": gif, "png": png, "frames": n}


def speed_trace_png(runs: list[tuple[str, Sim]], out: Path, dpi=120):
    fig, ax = plt.subplots(figsize=(7.5, 2.7), dpi=dpi)
    for title, sim in runs:
        t = np.array([f["t"] for f in sim.log.frames])
        v = np.array([f["v"] for f in sim.log.frames])
        if not isinstance(sim.speed_policy, S.Segmented):
            ax.plot(t - t[0], v[:, -1], label=title, lw=1.4)
        else:
            for j in range(1, v.shape[1]):
                ax.plot(t - t[0], v[:, j], lw=1.0, label=f"{title} · belt {j}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("belt speed (m/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def render_all(out: Path, results: dict | None = None) -> dict:
    """The report's replays. Speed: S0 vs S3 vs S4 on world B. Pick: FIFO vs value rate on the gold world."""
    import json

    out.mkdir(parents=True, exist_ok=True)
    res_path = Path(__file__).resolve().parent.parent / "docs" / "results" / "speed_ladder.json"
    tuned = {}
    if res_path.exists():
        tuned = json.loads(res_path.read_text())["scenarios"]
    written = {}
    name = "B 3 arms, 6 cats"
    sc = SC.speed_benchmark()[name]
    p3 = tuned.get(name, {}).get("rungs", {}).get("S3", {}).get("params", dict(v_cruise=0.3, v_floor=0.0, protect_n=1))
    p4 = tuned.get(name, {}).get("rungs", {}).get("S4", {}).get("params", p3)
    p1 = tuned.get(name, {}).get("rungs", {}).get("S1", {}).get("params", dict(v=0.3))
    runs = [
        ("S0 constant 0.3 m/s", record(sc, S.Constant(0.3), P.FIFO())),
        (f"S1 best constant {p1['v']:g} m/s", record(sc, S.Constant(p1["v"]), P.FIFO())),
        ("S3 deadline-aware", record(sc, S.DeadlineAware(**p3), P.FIFO())),
        ("S4 segmented belts", record(sc, S.Segmented(S.DeadlineAware(**p4)), P.FIFO())),
    ]
    written["speed"] = render_compare(runs, out / "speed_ladder_B")
    speed_trace_png(runs, out / "speed_trace_B.png")
    gold = SC.pick_benchmark()["PB MRF values + gold"]
    seed, t_gold = find_gold_window(gold)
    t0 = max(t_gold - 3.0, 5.0)
    runs = [
        ("P0 FIFO", record(gold, S.Constant(0.25), P.FIFO(), seed=seed, t_start=t0, t_len=20.0)),
        ("P2 value rate (p·value/cycle)", record(gold, S.Constant(0.25), P.ValueRate(), seed=seed, t_start=t0,
                                                  t_len=20.0)),
    ]
    written["gold_window"] = {"seed": seed, "t_start": t0}
    written["pick"] = render_compare(runs, out / "pick_ladder_gold")
    return written


def find_gold_window(sc, seeds=range(100, 120), t_max=600.0):
    """First (seed, entry time) where a gold item enters, FIFO loses it and value-rate picks it: the window that shows
    the difference. Found by search, not hand-picked, and the report says which."""
    names = [c.name for c in sc.categories]
    g = names.index("gold")
    for seed in seeds:
        a = Sim(sc, S.Constant(0.25), P.FIFO(), seed=seed).run(t_max)
        b = Sim(sc, S.Constant(0.25), P.ValueRate(), seed=seed).run(t_max)
        got_a = {p[1] for p in a.log.picks if p[4] == g}
        got_b = {p[1] for p in b.log.picks if p[4] == g}
        for i in sorted(got_b - got_a):
            if a.entered_t[i] > 20.0:
                return seed, float(a.entered_t[i])
    return 100, 60.0
