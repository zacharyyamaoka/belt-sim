"""3D replay in the browser (Viser): the belt, items, and each Cartesian gantry - legs, rails, the bridge sliding along
the belt, the carriage running across it, a z rod and a two-finger gripper that drops, closes and carries its item.

It replays recorded runs; it never simulates live. The engine stays headless and fast, the viewer only reads its
frames. A dropdown switches between speed policies run on the same seed, so the only thing that changes is the policy.

Uses the BAM house viewer (``viser_3d_viewer``: ViewCube, asserted port, playback) when it is installed, and plain
Viser with its logo hidden otherwise.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import pick_policies as P
from . import scenarios as SC
from . import speed_policies as S
from .engine import APPROACH, DELIVER, GRASP, Sim

PHASE_RGB = {0: (120, 120, 120), APPROACH: (214, 39, 40), GRASP: (255, 127, 14), DELIVER: (44, 160, 44)}
RAIL_Z = 0.55  # rail and bridge height above the belt
GRIP_UP, GRIP_DOWN = 0.32, 0.04  # gripper tip height, up and at the belt


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))


def policies_for() -> dict:
    return {
        "S0 constant 0.3 m/s": lambda: S.Constant(0.3),
        "S2 rush when idle": lambda: S.RushWhenIdle(0.3),
        "S3 deadline-aware": lambda: S.DeadlineAware(0.3, v_floor=0.0, protect_n=1),
        "S4 one belt per arm": lambda: S.Segmented(S.DeadlineAware(0.2, v_floor=0.0, protect_n=1)),
    }


def record_runs(scenario, seed: int = 100, t_start: float = 120.0, t_len: float = 60.0) -> dict[str, Sim]:
    from .viz_mpl import record

    return {label: record(scenario, make(), P.FIFO(), seed=seed, t_start=t_start, t_len=t_len, every=1)
            for label, make in policies_for().items()}


class _Gantry:
    def __init__(self, server, k, arm, cat_rgb):
        sc_ = server.scene
        lo, hi = arm.x_range
        self.arm = arm
        span_y = arm.y_max_m - arm.y_min_m
        for name, yr in (("near", arm.y_min_m), ("far", arm.y_max_m)):
            sc_.add_box(f"/arm{k}/rail_{name}", color=(70, 70, 75), dimensions=(hi - lo, 0.04, 0.04),
                        position=((lo + hi) / 2, yr, RAIL_Z))
            for xl in (lo, hi):
                sc_.add_box(f"/arm{k}/leg_{name}_{xl:.2f}", color=(90, 90, 95), dimensions=(0.04, 0.04, RAIL_Z + 0.3),
                            position=(xl, yr, (RAIL_Z - 0.3) / 2))
        self.bridge = sc_.add_box(f"/arm{k}/bridge", color=(150, 150, 155), dimensions=(0.06, span_y + 0.06, 0.05),
                                  position=(arm.x, (arm.y_min_m + arm.y_max_m) / 2, RAIL_Z + 0.03))
        self.carriage = sc_.add_box(f"/arm{k}/carriage", color=(40, 40, 45), dimensions=(0.10, 0.10, 0.08),
                                    position=(arm.x, 0.3, RAIL_Z + 0.03))
        self.rod = sc_.add_box(f"/arm{k}/rod", color=(200, 200, 205), dimensions=(0.025, 0.025, 0.3),
                               position=(arm.x, 0.3, 0.4))
        self.fingers = [sc_.add_box(f"/arm{k}/finger{j}", color=(30, 30, 30), dimensions=(0.012, 0.05, 0.07),
                                    position=(arm.x, 0.3, 0.1)) for j in range(2)]
        self.held = sc_.add_icosphere(f"/arm{k}/held", radius=0.022, color=(255, 255, 255), visible=False)
        self.cat_rgb = cat_rgb

    def show(self, state):
        tx, ty, phase, _item, z, grip, held = state
        tip = GRIP_DOWN + (GRIP_UP - GRIP_DOWN) * z
        self.bridge.position = (tx, (self.arm.y_min_m + self.arm.y_max_m) / 2, RAIL_Z + 0.03)
        self.carriage.position = (tx, ty, RAIL_Z + 0.03)
        self.carriage.color = PHASE_RGB[phase]
        rod_len = RAIL_Z - (tip + 0.07)
        self.rod.position = (tx, ty, tip + 0.07 + rod_len / 2)
        self.rod.scale = (1.0, 1.0, max(rod_len / 0.3, 0.05))
        gap = 0.014 + 0.03 * grip
        for sgn, f in zip((-1, 1), self.fingers, strict=True):
            f.position = (tx + sgn * gap, ty, tip + 0.035)
        if held >= 0:
            self.held.position = (tx, ty, tip + 0.02)
            self.held.color = tuple(int(v) for v in self.cat_rgb[held])
            self.held.visible = True
        else:
            self.held.visible = False


def build(server, scenario, runs: dict[str, Sim], viewer=None):
    belt = scenario.belt
    edges = belt.segment_edges()
    seg_handles = []
    for j in range(belt.n_segments):
        length = edges[j + 1] - edges[j]
        seg_handles.append(server.scene.add_box(
            f"/belt/seg{j}", color=(200, 200, 205), dimensions=(length - 0.01, belt.width_m, 0.04),
            position=((edges[j] + edges[j + 1]) / 2, belt.width_m / 2, -0.02)))
    cat_rgb = np.array([_rgb(c.color) for c in scenario.categories], np.uint8)
    for k, arm in enumerate(scenario.arms):
        for b in arm.bins:
            ci = scenario.category_index()[b.category]
            server.scene.add_box(f"/arm{k}/bin_{b.category}", color=tuple(int(v) for v in cat_rgb[ci]),
                                 dimensions=(0.16, 0.16, 0.22), position=(b.x, b.y, -0.11))
    gantries = [_Gantry(server, k, arm, cat_rgb) for k, arm in enumerate(scenario.arms)]
    items = server.scene.add_point_cloud("/items", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
                                         point_size=0.045, point_shape="circle")
    gui = server.gui
    choice = gui.add_dropdown("Policy", list(runs))
    status = gui.add_markdown("")
    state = {"sim": runs[choice.value]}

    def show(i: int) -> None:
        sim = state["sim"]
        f = sim.log.frames[min(i, len(sim.log.frames) - 1)]
        n = f["x"].size
        if n:
            items.points = np.stack([f["x"], f["y"], np.full(n, 0.025)], 1).astype(np.float32)
            items.colors = cat_rgb[f["cat"]]
        else:
            items.points, items.colors = np.zeros((1, 3), np.float32) - 10, np.zeros((1, 3), np.uint8)
        for g, st in zip(gantries, f["arms"], strict=True):
            g.show(st)
        vmax = max(belt.v_max_m_s, 1e-6)
        for j, h in enumerate(seg_handles):
            s = min(abs(float(f["v"][j])) / vmax, 1.0)
            h.color = (int(215 - 120 * s), int(215 - 60 * s), 225)
        speeds = "  ".join(f"{v:+.2f}" for v in f["v"])
        status.content = (f"**t {f['t'] - sim.log.frames[0]['t']:5.1f} s** · picked **{f['picked']}** · "
                          f"lost **{f['lost']}**  \nbelt m/s: `{speeds}`")

    n_frames = min(len(s.log.frames) for s in runs.values())
    if viewer is not None:
        from viser_3d_viewer import Playback

        playback = Playback(viewer, n_frames=n_frames, fps=1.0 / runs[choice.value].dt, on_frame=show, label="Frame",
                            autoplay=True)
        refresh = playback.refresh
    else:
        slider = gui.add_slider("Frame", 0, n_frames - 1, 1, 0)
        playing = gui.add_checkbox("Play", True)
        slider.on_update(lambda _e: show(int(slider.value)))

        def loop():
            dt = runs[choice.value].dt
            while True:
                time.sleep(dt)
                if playing.value:
                    slider.value = (int(slider.value) + 1) % n_frames

        threading.Thread(target=loop, daemon=True).start()
        refresh = lambda: show(int(slider.value))  # noqa: E731

    @choice.on_update
    def _(_event) -> None:
        state["sim"] = runs[choice.value]
        refresh()

    show(0)


def serve(scenario_name: str, port: int) -> None:
    sc = {**SC.speed_benchmark(), **SC.circulation_benchmark(), **SC.pick_benchmark()}[scenario_name]
    runs = record_runs(sc)
    centre = (sc.belt.length_m / 2, sc.belt.width_m / 2, 0.0)
    try:
        from viser_3d_viewer import Viewer, ViewerConfig
    except ImportError:
        Viewer = None
    if Viewer is not None:
        # WHY from above and in front: the belt reads as dots on a strip; no grid, the belt is the ground reference.
        viewer = Viewer(ViewerConfig(title=f"Belt sim · {sc.name}", port=port, home_target=centre,
                                     home_direction=(0.0, -0.9, 1.0), home_distance=sc.belt.length_m * 0.8,
                                     grid=False))
        build(viewer.server, sc, runs, viewer)
        viewer.print_ready()
        viewer.serve_forever()
        return
    import viser

    server = viser.ViserServer(host="127.0.0.1", port=port)
    server.gui.configure_theme(show_logo=False, show_share_button=False, dark_mode=True)
    build(server, sc, runs)

    @server.on_client_connect
    def _(client) -> None:
        client.camera.position = (centre[0], centre[1] - sc.belt.length_m * 0.55, sc.belt.length_m * 0.6)
        client.camera.look_at = centre

    print(f"READY http://127.0.0.1:{server.get_port()}/", flush=True)
    while True:
        time.sleep(3600)
