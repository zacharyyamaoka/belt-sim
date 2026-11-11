"""3D replay in the browser on the house viewer (viser-3d-viewer: no logo, Fusion ViewCube, asserted port, Playback).

It replays recorded runs; it never simulates live. The engine stays headless and fast, the viewer only reads its
frames. A dropdown switches between speed policies run on the same seed, so the only thing that changes is the policy.
"""

from __future__ import annotations

import numpy as np

from . import pick_policies as P
from . import scenarios as SC
from . import speed_policies as S
from .engine import APPROACH, DELIVER, GRASP, Sim

PHASE_RGB = {0: (90, 90, 90), APPROACH: (214, 39, 40), GRASP: (255, 127, 14), DELIVER: (44, 160, 44)}
ARM_HEIGHT = 0.7


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))


def policies_for(name: str) -> dict:
    return {
        "S0 constant 0.3 m/s": lambda: S.Constant(0.3),
        "S2 rush when idle": lambda: S.RushWhenIdle(0.3),
        "S3 deadline-aware": lambda: S.DeadlineAware(0.3, v_floor=0.0, protect_n=1),
        "S4 segmented belts": lambda: S.Segmented(S.DeadlineAware(0.3, v_floor=0.0, protect_n=1)),
    }


def record_runs(scenario, seed: int = 100, t_start: float = 120.0, t_len: float = 60.0) -> dict[str, Sim]:
    from .viz_mpl import record

    return {label: record(scenario, make(), P.FIFO(), seed=seed, t_start=t_start, t_len=t_len, every=2)
            for label, make in policies_for(scenario.name).items()}


def build(viewer, scenario, runs: dict[str, Sim]):
    from viser_3d_viewer import Playback

    server = viewer.server
    belt = scenario.belt
    edges = belt.segment_edges()
    seg_handles = []
    for j in range(belt.n_segments):
        length = edges[j + 1] - edges[j]
        h = server.scene.add_box(f"/belt/seg{j}", color=(200, 200, 205),
                                 dimensions=(length - 0.01, belt.width_m, 0.04),
                                 position=((edges[j] + edges[j + 1]) / 2, belt.width_m / 2, -0.02))
        seg_handles.append(h)
    cat_rgb = np.array([_rgb(c.color) for c in scenario.categories], np.uint8)
    for k, arm in enumerate(scenario.arms):
        th = np.linspace(0, 2 * np.pi, 65)
        pts = np.stack([arm.x + arm.reach_m * np.cos(th), arm.y + arm.reach_m * np.sin(th), np.full_like(th, 0.003)], 1)
        server.scene.add_line_segments(f"/arms/{k}/reach", np.stack([pts[:-1], pts[1:]], 1), (120, 120, 120),
                                       thickness=0.006)
        server.scene.add_cylinder(f"/arms/{k}/post", radius=0.03, height=ARM_HEIGHT, color=(60, 60, 60),
                                  position=(arm.x, arm.y, ARM_HEIGHT / 2 + 0.25))
        for b in arm.bins:
            ci = scenario.category_index()[b.category]
            server.scene.add_box(f"/arms/{k}/bin_{b.category}", color=tuple(int(v) for v in cat_rgb[ci]),
                                 dimensions=(0.16, 0.16, 0.22), position=(b.x, b.y, -0.11))
    items = server.scene.add_point_cloud("/items", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
                                         point_size=0.05, point_shape="circle")
    tools = [server.scene.add_icosphere(f"/arms/{k}/tool", radius=0.035, color=(90, 90, 90))
             for k in range(len(scenario.arms))]
    links = server.scene.add_line_segments("/arms/links", np.zeros((len(scenario.arms), 2, 3), np.float32),
                                           (70, 70, 70), thickness=0.012)
    gui = server.gui
    choice = gui.add_dropdown("Policy", list(runs))
    status = gui.add_markdown("")
    state = {"sim": runs[choice.value]}

    def show(i: int) -> None:
        sim = state["sim"]
        f = sim.log.frames[min(i, len(sim.log.frames) - 1)]
        n = f["x"].size
        if n:
            pts = np.stack([f["x"], f["y"], np.full(n, 0.03)], 1).astype(np.float32)
            cols = cat_rgb[f["cat"]]
            items.points, items.colors = pts, cols
        else:
            items.points, items.colors = np.zeros((1, 3), np.float32) - 10, np.zeros((1, 3), np.uint8)
        seg_pts = []
        for k, (arm, (tx, ty, phase, _item)) in enumerate(zip(scenario.arms, f["arms"], strict=True)):
            tools[k].position = (tx, ty, 0.06)
            tools[k].color = PHASE_RGB[phase]
            seg_pts.append([[arm.x, arm.y, 0.25], [tx, ty, 0.06]])
        links.points = np.asarray(seg_pts, np.float32)
        vmax = max(belt.v_max_m_s, 1e-6)
        for j, h in enumerate(seg_handles):
            s = min(abs(float(f["v"][j])) / vmax, 1.0)
            h.color = (int(215 - 120 * s), int(215 - 60 * s), 225)
        speeds = "  ".join(f"{v:+.2f}" for v in f["v"])
        status.content = (f"**t {f['t'] - sim.log.frames[0]['t']:5.1f} s** · picked **{f['picked']}** · "
                          f"lost **{f['lost']}**  \nbelt m/s: `{speeds}`")

    n_frames = min(len(s.log.frames) for s in runs.values())
    playback = Playback(viewer, n_frames=n_frames, fps=10.0, on_frame=show, label="Frame", autoplay=True)

    @choice.on_update
    def _(_event) -> None:
        state["sim"] = runs[choice.value]
        playback.refresh()

    show(0)
    return playback


def serve(scenario_name: str, port: int) -> None:
    from viser_3d_viewer import Viewer, ViewerConfig

    sc = {**SC.speed_benchmark(), **SC.circulation_benchmark(), **SC.pick_benchmark()}[scenario_name]
    runs = record_runs(sc)
    centre = (sc.belt.length_m / 2, sc.belt.width_m / 2, 0.0)
    # WHY from above and in front: the belt reads as dots on a strip (the daily note's mental picture); no grid, the
    # belt itself is the ground reference.
    viewer = Viewer(ViewerConfig(title=f"Belt sim · {sc.name}", port=port, home_target=centre,
                                 home_direction=(0.0, -0.75, 1.0), home_distance=sc.belt.length_m * 0.72,
                                 grid=False))
    build(viewer, sc, runs)
    viewer.print_ready()
    viewer.serve_forever()
