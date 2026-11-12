# belt-sim

A fast, headless simulator of a conveyor line with Cartesian pick robots, built to optimise two black boxes:

1. **Belt speed control** — how fast should each belt segment run when several robots share it?
   (constant → rush when idle → deadline-aware → one belt per robot, on straight and recirculating belts)
2. **Pick scheduling** — which item should a free robot take next?
   (FIFO → shortest cycle → value rate → last chance → lookahead, against an LP upper bound)

Both ladders run on one engine, so every rung changes exactly one thing. Results, replays and charts:
**https://zacharyyamaoka.github.io/belt-sim**

```bash
git clone https://github.com/zacharyyamaoka/belt-sim && cd belt-sim
uv sync
uv run belt-sim ladder all                          # tune on seeds 0-2, score on 100-104 -> docs/results/*.json
uv run belt-sim rerun --scenario "B 3 arms, 6 cats" # drill-down replays with live stats in Rerun
uv run belt-sim viser --port 8091                   # 3D gantries in the browser, policy dropdown
uv run belt-sim wandb                               # one Weights & Biases run per tried configuration
uv run python docs/build_site.py                    # the landing page -> _site/
uv run pytest -q
```

## Model

| Piece | What it is | Where |
|---|---|---|
| Belt | x along, y across; N segments with independent speeds (reverse allowed, accel-limited); a hand-off holds items when the next belt is stopped or reversing and keeps `min_gap` spacing onto a slower one | `engine.py` `_move` |
| Supply | `hopper`: items per metre of belt travel (faster belt, more items/s) · `stream`: items per second whatever the belt does (backlog at the head) · `burstiness` clumps · recirculation (end → head; residue nobody may pick leaves) | `engine.py` `ItemStream`, `_infeed` |
| Robot | Cartesian gantry: x and y axes with their own max speed and acceleration, trapezoidal profiles, move time = slower axis; 1 m stroke along the belt, full width across; up to 6 bins; intercepts a moving item and tracks it through the grasp dwell; re-plans if the belt changes speed; hard belt-speed line for grasping | `config.py` `ArmSpec`, `engine.py` |
| Items | category, value, per-item grasp probability known to the scheduler, pre-drawn outcomes (common random numbers across policies) | `engine.py` `ItemStream` |
| Metrics | picks/min, value/min, capture (recovery), diversion, backlog growth, robots busy, missed intercepts, laps per pick | `metrics.py` |
| Speed policies | S0-S4 | `speed_policies.py` |
| Pick policies | P0-P4 | `pick_policies.py` |
| Bound | time-indexed LP on expected value (HiGHS), clairvoyant; a strict bound and a tighter planning reference | `bound.py` |
| Worlds | straight, recirculating, layout and scheduling benchmarks | `scenarios.py` |
| Views | matplotlib top-down replays, Viser 3D, Rerun recordings with live stats, W&B projects | `viz_mpl.py`, `viz_viser.py`, `rerun_replay.py`, `tracking.py` |

## Scoring

* **Straight belt, hopper supply:** best picks/min with capture ≥ 90 %. Without the gate, "run flat out" wins
  picks/min and sends everything else to landfill.
* **Straight belt, stream supply:** picks/min with the head backlog not growing.
* **Recirculating belt:** plain picks/min — a miss comes round again.
* **Scheduling:** value per minute, and its share of the LP bound / reference on straight belts.

Every knob is tuned on seeds 0-2 and scored on seeds 100-104.

## Stock parts

| Element | Today | Off-the-shelf part | Behaviour that must survive |
|---|---|---|---|
| Time stepping | hand-rolled fixed 50 ms step | SimPy (stock DES) considered | belt positions are the integral of a speed the controller picks as it goes; the oracle test pins agreement with an exact event loop at constant speed |
| LP | `scipy.optimize.linprog(method="highs")` | HiGHS via SciPy | unchanged (stock seam) |
| Replays | Rerun recordings + app.rerun.io viewer | Rerun | unchanged (stock seam) |
| Experiment tracking | Weights & Biases | W&B | unchanged (stock seam) |
| 3D view | Viser (logo hidden) | Viser | unchanged (stock seam) |
| Constant-speed oracle | `tests/oracle_line_sim.py` | an independent event-driven line sim | capture agrees within 3 SE + 1.5 % on 4 cases; a 15 % cycle mutation fails |

## Traps

- With reach time × belt speed exactly equal to the window, the reference line sim grasps at the window's far edge
  (its interval is closed there); a fixed-step engine never lands on that edge. Keep oracle cases off that equality.
- A grasp must END inside the stroke, allowing for the belt accelerating while the gripper closes.
- A speed policy must not speed the belt up under a robot that has committed to an intercept, and must not hold the
  belt still for an item already too close to the stroke end to grasp (it stalled the line once).

MIT licence.
