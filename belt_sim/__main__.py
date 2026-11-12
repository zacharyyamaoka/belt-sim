"""belt-sim: climb the ladders, render the 2D replay, serve the 3D replay.

    cd ~/belt-sim
    uv run belt-sim ladder all          # both ladders -> docs/results/*.json
    uv run belt-sim replay              # matplotlib MP4/GIF + PNG stills -> docs/media
    uv run belt-sim viser --port 8091   # live 3D replay in the browser
    uv run belt-sim rerun --scenario "B 3 arms, 6 cats"   # drill-down replays with live stats in Rerun
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
    rr_ = sub.add_parser("rerun", help="write Rerun replays (one .rrd per rung) for a world and open them")
    rr_.add_argument("--scenario", default="B 3 arms, 6 cats")
    rr_.add_argument("--seconds", type=float, default=60.0)
    rr_.add_argument("--rungs", default="", help="comma-separated rung ids to keep, e.g. S0,S4 (default: all)")
    rr_.add_argument("--out", type=Path, default=PKG / "docs" / "media" / "rerun")
    rr_.add_argument("--no-open", action="store_true", help="write the .rrd files only")
    wb = sub.add_parser("wandb", help="log both ladders to Weights & Biases (one run per tried config)")
    wb.add_argument("--no-upload", action="store_true")
    wb.add_argument("--workers", type=int, default=None)
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
    elif a.cmd == "rerun":
        import shutil
        import subprocess

        from .rerun_replay import write_replays

        paths = write_replays(a.scenario, a.out, seconds=a.seconds, only=[r for r in a.rungs.split(",") if r] or None)
        for p in paths:
            print(p)
        if not a.no_open:
            viewer = shutil.which("rerun") or str(Path(__import__("sys").executable).parent / "rerun")
            subprocess.Popen([viewer, *map(str, paths)], start_new_session=True)
    elif a.cmd == "wandb":
        from .tracking import log_all

        print(log_all(RESULTS, PKG / "wandb_runs", workers=a.workers, upload=not a.no_upload))
    elif a.cmd == "viser":
        from .viz_viser import serve

        serve(a.scenario, a.port)


if __name__ == "__main__":
    main()
