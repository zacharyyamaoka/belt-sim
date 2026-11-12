"""belt-sim: climb the ladders, render the 2D replay, serve the 3D replay.

    cd ~/bam_ws/src/dev/belt_sim
    uv run belt-sim ladder all          # both ladders -> docs/results/*.json
    uv run belt-sim replay              # matplotlib MP4/GIF + PNG stills -> docs/media
    uv run belt-sim viser --port 8091   # live 3D replay in the browser
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import ladder
from . import scenarios as SC

PKG = Path(__file__).resolve().parent.parent
RESULTS = PKG / "docs" / "results"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="belt-sim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("ladder")
    lp.add_argument("which", choices=["speed", "pick", "all"])
    lp.add_argument("--workers", type=int, default=None)
    lp.add_argument("--quick", action="store_true", help="short runs, for smoke testing")
    rp = sub.add_parser("replay")
    rp.add_argument("--out", type=Path, default=PKG / "docs" / "media")
    vp = sub.add_parser("viser")
    vp.add_argument("--port", type=int, default=8091)
    vp.add_argument("--scenario", default="B 3 arms, 6 cats")
    a = ap.parse_args(argv)
    if a.cmd == "ladder":
        tune_s, eval_s, pick_s = (120.0, 180.0, 120.0) if a.quick else (480.0, 720.0, 300.0)
        if a.which in ("speed", "all"):
            names = list(SC.speed_benchmark()) + list(SC.circulation_benchmark()) + list(SC.layout_benchmark())
            t0 = time.time()
            out = ladder.climb_speed(names, tune_s=tune_s, eval_s=eval_s, workers=a.workers)
            out["wall_s"] = time.time() - t0
            ladder.save(out, RESULTS / "speed_ladder.json")
        if a.which in ("pick", "all"):
            t0 = time.time()
            names = list(SC.pick_benchmark()) + list(SC.pick_loop_benchmark())
            out = ladder.climb_pick(names, duration=pick_s, workers=a.workers)
            out["wall_s"] = time.time() - t0
            ladder.save(out, RESULTS / "pick_ladder.json")
    elif a.cmd == "replay":
        from .viz_mpl import render_all

        render_all(a.out)
    elif a.cmd == "viser":
        from .viz_viser import serve

        serve(a.scenario, a.port)


if __name__ == "__main__":
    main()
