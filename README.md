# bam_belt_sim

Headless, fast 2D sim of a conveyor line with N pick arms, for two questions from the Daily Note of Nov 11 2026
("Conveyor belt speed control and FIFO object scheduling"):

1. **Belt speed control**: how fast should each belt run when several arms share it? (constant → rush when idle →
   deadline-aware → one belt per arm, with and without recirculation)
2. **Pick scheduling**: which item should a free arm take next? (FIFO → shortest cycle → value rate → last chance →
   lookahead, against an LP upper bound)

Both run on one engine; the two ladders are two policy axes over the same world, so each rung changes one thing.

```bash
cd ~/bam_ws/src/dev/bam_belt_sim
uv run bam-belt-sim ladder all          # tune on seeds 0-2, score on 100-104 -> docs/results/*.json
uv run python docs/build_report.py      # charts, replays, report -> ~/bam_ws/reports/media/belt-sim-<date>.html
uv run bam-belt-sim viser --port 8091   # 3D replay (house viewer), policy dropdown
uv run pytest -q                        # invariants, F1 oracle + its mutation, LP bound
```

The report's **Facility KPIs** section reads each ladder rung as the line seen from outside: items in and items
picked, diversion rate (picked ÷ in) and capture rate (recovery), with the rate switch every pick-and-place page
shares (`src/dev/bam_cell_kpis`). That is where "high throughput, low diversion" shows: the MRF mix at v1's 0.3 m/s
takes in 91 items a minute and diverts 46 % of them. The sim's items have no mass, so the mass columns say so. To check
only that section, in seconds:

```bash
cd ~/bam_ws/src/dev/bam_belt_sim
uv run python docs/build_report.py --kpis-only --out /tmp/belt-sim-kpis.html
```

## Model

| Piece | What it is | Where |
|---|---|---|
| Belt | x along, y across; N segments with independent speeds (reverse allowed, accel-limited); hand-off holds items when the next belt is stopped or reversing and keeps `min_gap` spacing onto a slower one | `engine.py` `_move` |
| Supply | `hopper`: items per metre of belt travel (faster belt, more items/s) · `stream`: items per second whatever the belt does (backlog at the head) · `burstiness` clumps · optional recirculation (end → head, max passes) | `engine.py` `ItemStream`, `_infeed` |
| Arm | reach disc, a bin per category it may pick, trapezoidal tool moves (facility model UR5e: 0.8 m/s, 1.2 m/s²), grasp/release dwell, intercept of a moving item with re-plan, hard belt-speed line for grasping | `config.py` `ArmSpec`, `engine.py` |
| Items | category, value, per-item grasp probability known to the scheduler, 8 pre-drawn outcomes (common random numbers) | `engine.py` `ItemStream` |
| Metrics | SPPM, value/min, recovery, backlog growth, utilisation, missed intercepts, peak pile | `metrics.py` |
| Speed policies | S0-S4 | `speed_policies.py` |
| Pick policies | P0-P4 | `pick_policies.py` |
| Bound | time-indexed LP (HiGHS) on expected value, clairvoyant; strict bound + tighter planning reference | `bound.py` |
| Worlds | speed, recirculation, layout and pick benchmarks | `scenarios.py` |

## Stock parts

| Element | Today | Off-the-shelf part | Behaviour that must survive |
|---|---|---|---|
| Time stepping | hand-rolled fixed-step loop | SimPy (stock DES) was considered | variable belt speed makes positions an integral of the controller's choice; the fixed step is the simple honest form. The F1 oracle test pins agreement with an exact event loop at constant speed |
| LP | `scipy.optimize.linprog(method="highs")` | HiGHS via SciPy | unchanged (stock seam) |
| 3D view | `viser_3d_viewer.Viewer` + `Playback` | house viewer on viser 1.1.1 | no logo, ViewCube, asserted port |
| 2D replay | matplotlib + imageio (H.264) | stock | unchanged (stock seam) |
| Constant-speed oracle | `bam_facility_model.line_sim` (F1) | — | recovery agrees within 3 SE + 1.5 % on 4 cases |

## Traps

- With reach time × belt speed equal to the window, F1 grasps at exactly the window's far edge (closed interval); a
  fixed-step engine never lands there. Keep oracle cases away from that equality.
- An item must stay in reach through the grasp dwell, not only at contact (the tool tracks it while closing).
- A speed policy must not speed the belt up under an arm that has committed to an intercept: that alone caused
  15-50 % missed intercepts before the hold cap.
