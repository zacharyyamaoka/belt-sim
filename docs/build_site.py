"""Build the landing page (GitHub Pages) from docs/results/*.json: charts, replay videos, Rerun replays, links.

    cd ~/belt-sim
    uv run python docs/build_site.py            # -> _site/ (index.html + media + replays/*.rrd)

The page is static. Rerun replays open in Rerun's hosted viewer (app.rerun.io) straight from this site's files;
GitHub Pages serves them with CORS, which is what that viewer needs.
"""

from __future__ import annotations

import html
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
from belt_sim import pick_policies as P  # noqa: E402
from belt_sim import scenarios as SC  # noqa: E402
from belt_sim import speed_policies as S  # noqa: E402
from belt_sim.engine import Sim  # noqa: E402
from belt_sim.metrics import summarize  # noqa: E402

RES = PKG / "docs" / "results"
OUT = PKG / "_site"
REPO = "https://github.com/zacharyyamaoka/belt-sim"
PAGES = "https://zacharyyamaoka.github.io/belt-sim"
RERUN_VIEWER = "https://app.rerun.io/version/0.43.1/index.html?url="
RUNG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)



def speed_dot_plot(speed: dict, names: list[str], out: Path, title: str):
    rungs = list(speed["rungs"])
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(names) + 1.1), dpi=130)
    for row, n in enumerate(names):
        sc = speed["scenarios"][n]
        y = len(names) - 1 - row
        vals = [sc["rungs"][r]["sppm"] for r in rungs]
        ax.plot([min(vals), max(vals)], [y, y], color="#d6d5d0", lw=1.2, zorder=1)
        for j, r in enumerate(rungs):
            d = sc["rungs"][r]
            off = (j - 2) * 0.09
            face = RUNG_COLORS[j] if d["passes_gate"] else "white"
            ax.scatter(d["sppm"], y + off, s=46, facecolor=face, edgecolor=RUNG_COLORS[j], linewidth=1.8, zorder=3)
        best = max((r for r in rungs if sc["rungs"][r]["passes_gate"]), key=lambda r: sc["rungs"][r]["sppm"],
                   default=None)
        if best:
            ax.text(sc["rungs"][best]["sppm"] + 1.2, y, f"{best} {sc['rungs'][best]['sppm']:.0f}", va="center",
                    fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=8, color=INK)
    ax.set_xlabel("successful picks per minute (eval seeds, mean)", fontsize=8, color=INK2)
    style_ax(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=c, markeredgecolor=c, label=f"{r} {speed['rungs'][r]}")
               for r, c in zip(rungs, RUNG_COLORS, strict=True)]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2,
                              label="hollow = fails the gate"))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.5 / len(names)),
              ncol=3)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)



def frontier_plot(speed: dict, name: str, out: Path):
    sc = speed["scenarios"][name]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    for j, (r, pts) in enumerate(sc["frontier"].items()):
        x = [p["recovery"] for p in pts]
        y = [p["sppm"] for p in pts]
        ax.scatter(x, y, s=16, color=RUNG_COLORS[j], alpha=0.55, linewidth=0, label=f"{r} (tuning grid)")
        e = sc["rungs"][r]
        ax.scatter([e["recovery"]], [e["sppm"]], s=70, facecolor=RUNG_COLORS[j], edgecolor="white", linewidth=1.5,
                   zorder=4)
    ax.axvline(speed["gate"]["recovery"], color=INK2, lw=1, ls="--")
    ax.text(speed["gate"]["recovery"] + 0.004, ax.get_ylim()[0] + 1, "gate: recovery 90 %", fontsize=7, color=INK2)
    ax.set_xlabel("recovery (share of targeted items picked, not landfilled)", fontsize=8, color=INK2)
    ax.set_ylabel("successful picks / min", fontsize=8, color=INK2)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(fontsize=7, frameon=False, ncol=1, loc="lower left")
    ax.set_title(f"{name}: every tuned config (small) and each rung's pick on eval seeds (large)", fontsize=8.5,
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)



def pick_plot(pick: dict, out: Path):
    """Value per minute of every rung, as a share of FIFO's on the same world (FIFO = 100 %)."""
    names = list(pick["scenarios"])
    rungs = list(pick["rungs"])
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(names) + 1.3), dpi=130)
    for row, n in enumerate(names):
        y = len(names) - 1 - row
        rr = pick["scenarios"][n]["rungs"]
        base = rr["P0"]["value_per_min"] or 1e-12
        vals = [100 * rr[r]["value_per_min"] / base for r in rungs]
        ax.plot([min(vals), max(vals)], [y, y], color="#d6d5d0", lw=1.2, zorder=1)
        for j, r in enumerate(rungs):
            ax.scatter(vals[j], y + (j - 2) * 0.09, s=46, color=RUNG_COLORS[j], zorder=3)
    ax.axvline(100, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=8)
    ax.set_xlabel("value extracted per minute, % of FIFO on the same world (evaluation seeds)", fontsize=8, color=INK2)
    style_ax(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"{r} {pick['rungs'][r]}")
               for r, c in zip(rungs, RUNG_COLORS, strict=True)]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def table(headers, rows):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def fmt_params(p: dict) -> str:
    return ", ".join(f"{k.replace('v_', 'v ').replace('_', ' ')} {v:g}" for k, v in p.items()) or "-"


def speed_rows(speed, names):
    rows = []
    for n in names:
        sc = speed["scenarios"][n]
        for r, d in sc["rungs"].items():
            gate = "loop" if sc.get("loop") else ("✓" if d["passes_gate"] else "✗")
            div = d["sppm"] / d["entered_per_min"] if d.get("entered_per_min") else 0.0
            rows.append([html.escape(n) if r == "S0" else "", f"{r} {speed['rungs'][r]}",
                         f"<b>{d['sppm']:.1f}</b> ± {d['sppm_sd']:.1f}", f"{100 * d['recovery']:.0f} %",
                         f"{100 * min(div, 1):.0f} %", gate, f"{d['mean_speed_m_s']:.2f}",
                         f"{100 * d['utilisation']:.0f} %",
                         f"{d.get('passes_per_pick', 0):.2f}" if sc.get("loop") else "", fmt_params(d["params"])])
    return table(["world", "rung", "picks/min", "capture", "diversion", "gate", "belt m/s", "arms busy",
                  "laps/pick", "tuned knobs"], rows)


def pick_rows(pick):
    rows = []
    for n, scd in pick["scenarios"].items():
        usd = scd.get("value_unit") == "USD"
        for r, d in scd["rungs"].items():
            val = f"{d['value_per_min']:.3f} $" if usd else f"{d['value_per_min']:.1f}"
            gold = f"{100 * d['gold_recovery']:.0f} % of {d['gold_items_total']}" if "gold_recovery" in d else ""
            ref = f"{100 * d['share_of_reference']:.0f} %" if "share_of_reference" in d else "n/a (loop)"
            rows.append([html.escape(n) if r == "P0" else "", f"{r} {pick['rungs'][r]}", f"<b>{val}</b>",
                         f"{d['sppm']:.1f}", ref, gold, fmt_params(d["params"])])
    return table(["world", "rung", "value / min", "picks/min", "of LP reference", "gold picked", "tuned knobs"], rows)


def oracle_rows(seeds=3, duration=600.0):
    sys.path.insert(0, str(PKG / "tests"))
    from oracle_line_sim import simulate_line
    from test_oracle import CASES

    from belt_sim.oracle import line_sim_scenario

    rows = []
    for case in CASES:
        rate, v, arms, w, cycle, reach, p = case
        f1 = [simulate_line(items_per_s=rate, belt_speed_m_s=v, arms=arms, window_m=w, cycle_time_s=cycle,
                            reach_time_s=reach, grasp_success=p, duration_s=duration, warmup_s=120, seed=s).recovery
              for s in range(seeds)]
        sc = line_sim_scenario(rate, v, arms, w, cycle, reach, p)
        ours = [summarize(Sim(sc, S.Constant(v), P.MostDownstream(), seed=s, dt=0.01).run(duration + 120),
                          120).recovery for s in range(seeds)]
        rows.append([", ".join(f"{x:g}" for x in case), f"{statistics.mean(f1):.3f}", f"{statistics.mean(ours):.3f}"])
    return table(["items/s, belt m/s, arms, window m, cycle s, reach s, p", "reference sim", "this engine"], rows)


def realtime_factor():
    sc = SC.speed_benchmark()["B 3 arms, 6 cats"]
    t0 = time.perf_counter()
    Sim(sc, S.Constant(0.3), P.FIFO(), seed=0).run(600)
    return 600 / (time.perf_counter() - t0)


REPLAYS = [("B 3 arms, 6 cats", ["S0", "S1", "S3", "S4"]), ("B 3 arms, 6 cats · loop", ["S0", "S4"]),
           ("PB MRF values + gold", ["P0", "P2"]), ("PB MRF values + gold · loop", ["P0", "P2"])]


def main():
    from belt_sim.rerun_replay import write_replays
    from belt_sim.viz_mpl import render_all

    speed = json.loads((RES / "speed_ladder.json").read_text())
    pick = json.loads((RES / "pick_ladder.json").read_text())
    links = json.loads((RES / "links.json").read_text()) if (RES / "links.json").exists() else {}
    if OUT.exists():
        shutil.rmtree(OUT)
    media = OUT / "media"
    media.mkdir(parents=True)
    written = render_all(media)
    lines = list(SC.speed_benchmark())
    loops = list(SC.circulation_benchmark())
    speed_dot_plot(speed, lines, media / "speed_ladder.png",
                   "Straight belt: picks/min per rung (filled = capture ≥ 90 %; stream: backlog does not grow)")
    speed_dot_plot(speed, loops, media / "speed_loops.png", "Recirculating belt: picks/min per rung (misses come round)")
    speed_dot_plot(speed, list(SC.layout_benchmark()), media / "layout.png", "Layout: one MRF line, bins dealt three ways")
    frontier_plot(speed, "B 3 arms, 6 cats", media / "frontier_B.png")
    frontier_plot(speed, "B 3 arms, 6 cats · loop", media / "frontier_B_loop.png")
    pick_plot(pick, media / "pick_ladder.png")
    replay_links = []
    for world, rungs in REPLAYS:
        for path in write_replays(world, OUT / "replays", seconds=40.0, only=rungs):
            replay_links.append((world, path.stem.rsplit("_", 1)[-1], path.name))
    oracle = oracle_rows()
    rtf = realtime_factor()

    def btn(label, href, cls=""):
        return f'<a class="btn {cls}" href="{href}" target="_blank" rel="noopener">{label}</a>'

    def video(key, caption):
        w = written[key]
        return (f'<figure><video controls autoplay muted loop playsinline poster="media/{w["png"].name}" '
                f'src="media/{w["mp4"].name}"></video><figcaption>{caption} · <a href="media/{w["gif"].name}">GIF</a>'
                f"</figcaption></figure>")

    B = speed["scenarios"]["B 3 arms, 6 cats"]["rungs"]
    BL = speed["scenarios"]["B 3 arms, 6 cats · loop"]["rungs"]
    D = speed["scenarios"]["D sparse 2.5/m"]["rungs"]
    PL = pick["scenarios"]["PA 3 arms, 6 equal cats · loop"]["rungs"]
    eq = [n for n in pick["scenarios"] if "gold" not in n and "loop" not in n]
    gains = [pick["scenarios"][n]["rungs"]["P1"]["value_per_min"] / pick["scenarios"][n]["rungs"]["P0"]["value_per_min"] - 1
             for n in eq]
    eq_txt = f"{100 * min(gains):.0f}-{100 * max(gains):.0f} %"
    pb = pick["scenarios"]["PB MRF values + gold"]["rungs"]

    def best_gate(n):
        sc = speed["scenarios"][n]["rungs"]
        return max((r for r in sc if sc[r]["passes_gate"]), key=lambda r: sc[r]["sppm"])

    winners = {n: best_gate(n) for n in lines + loops}
    win_txt = " · ".join(f"{html.escape(n.split(' ')[0])}{' loop' if 'loop' in n else ''} → <b>{w}</b>"
                         for n, w in winners.items())
    replay_html = "".join(
        f'<li><span class="w">{html.escape(w)}</span> {btn(r, RERUN_VIEWER + quote(f"{PAGES}/replays/{f}", safe=""))}'
        f' <a class="dl" href="replays/{f}">.rrd</a></li>' for w, r, f in replay_links)
    wandb_html = ""
    if links:
        wandb_html = (btn("W&amp;B · Speed control", links["speed"], "wb") + btn("W&amp;B · Pick scheduling",
                                                                               links["pick"], "wb"))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>belt-sim</title>
<meta name="description" content="Fast headless conveyor-line simulator: belt speed control and pick scheduling for Cartesian pick robots, with ladders, an LP bound, Rerun replays and W&B pages.">
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--card:#f4f3f0}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151514;--ink:#f0efec;--ink2:#c3c2b7;--line:#33332f;--card:#1f1f1d}}}}
body{{background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}}
main{{max-width:1080px;margin:0 auto;padding:32px 16px 90px}}
h1{{font-size:40px;margin:0;letter-spacing:-.02em}} .tag{{font-size:19px;color:var(--ink2);margin:6px 0 18px}}
h2{{font-size:22px;margin:44px 0 10px;border-top:1px solid var(--line);padding-top:20px}} h3{{font-size:16px;margin:22px 0 6px}}
.btn{{display:inline-block;padding:7px 13px;margin:3px 6px 3px 0;border-radius:7px;background:var(--accent);color:#fff;
text-decoration:none;font-size:14px;font-weight:600}} .btn.wb{{background:#e6a817;color:#111}} .btn.gh{{background:#24292f}}
.verdict ul{{margin:6px 0 0;padding-left:20px}} .verdict{{background:var(--card);border-left:4px solid var(--accent);padding:12px 16px;border-radius:6px}}
video,img{{width:100%;border-radius:6px;background:#fff}} figure{{margin:14px 0}} figcaption{{font-size:13px;color:var(--ink2)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;width:100%;margin:8px 0}}
th,td{{border-bottom:1px solid var(--line);padding:4px 7px;text-align:left;white-space:nowrap}} th{{color:var(--ink2)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media (max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
ul.rr{{list-style:none;padding:0}} ul.rr li{{margin:4px 0}} .w{{display:inline-block;min-width:230px;color:var(--ink2)}}
.dl{{font-size:13px;color:var(--ink2)}} code,pre{{font-size:13.5px}} pre{{background:var(--card);padding:10px 12px;border-radius:6px;overflow-x:auto}}
.muted{{color:var(--ink2)}}
</style></head><body><main>
<h1>belt-sim</h1>
<p class="tag">A fast, headless conveyor-line simulator for two black boxes: <b>how fast should the belt run</b>, and
<b>which item should each robot pick next</b>.</p>
<p>{btn("GitHub", REPO, "gh")}{wandb_html}{btn("Open a replay in Rerun", "#replays")}</p>
{video("speed", "Same items, same seed, four belt-speed policies: constant 0.3 m/s · best constant · deadline-aware · one belt per gantry. Each Cartesian gantry: rails along the belt, the bridge slides in x, the carriage runs in y, the two-finger gripper drops, closes and carries the item to its bin (red approach, orange grasp, green deliver).")}
<div class="verdict"><b>Best rung per world</b> (straight belts at their gate, loops on plain picks/min): {win_txt}.
<ul>
<li><b>Straight, dense belts:</b> a well-chosen constant speed is hard to beat once the robots are fast; deadline-aware speed wins where arrivals are uneven, and one belt per gantry wins where the belt is sparse ({D['S4']['sppm']:.0f} vs {D['S1']['sppm']:.0f} picks/min for the best constant speed).</li>
<li><b>Recirculating belts:</b> run as fast as the robots can still grasp - a miss just comes round; holding the belt for items costs throughput (world B: {BL['S1']['sppm']:.0f} picks/min at {BL['S1']['mean_speed_m_s']:.1f} m/s vs {BL['S3']['sppm']:.0f} deadline-aware).</li>
<li><b>Pick order:</b> shortest cycle first beats FIFO by {eq_txt} on equal-value worlds; value rate (p · value ÷ cycle) takes {100 * pb['P2'].get('gold_recovery', 0):.0f} % of rare high-value items vs FIFO's {100 * pb['P0'].get('gold_recovery', 0):.0f} % (+{100 * (pb['P2']['value_per_min'] / pb['P0']['value_per_min'] - 1):.0f} % value/min). On a loop FIFO makes items circle {PL['P0']['passes_per_pick']:.1f} laps before a pick vs {PL['P1']['passes_per_pick']:.1f} for shortest cycle.</li>
</ul></div>

<h2>The model</h2>
<ul>
<li><b>Belt</b>: a flat 2D strip (x along, y across) split into N segments with independent speeds (forward or reverse, acceleration-limited), fed by a hopper (items per metre of travel: faster belt, more items) or a stream (items per second whatever the belt does), optionally recirculating (items reaching the end go back to the head).</li>
<li><b>Robots</b>: Cartesian gantries. x and y axes each have a max speed and acceleration (default 1.5 m/s, 6 m/s²); every move is a pair of trapezoidal profiles and lasts as long as the slower axis. The gripper intercepts the moving item, tracks it through a 0.35 s grasp (plunge, close, lift) and drops it in the bin for its category (up to 6 bins per gantry).</li>
<li><b>Items</b>: category, value, a per-item grasp probability the scheduler knows, and a hard belt-speed line (0.5 m/s) above which no grasp is tried. All random draws come from the seed, so every policy sees the same items and the same coin flips.</li>
<li><b>Scores</b>: successful picks/min; <b>capture</b> = picked ÷ (picked + lost) of items a robot could take; <b>diversion</b> = picked ÷ entered; value/min for scheduling. Straight belt: best picks/min with capture ≥ 90 % (else "run flat out" wins and everything goes to landfill). Recirculating belt: plain picks/min, because a miss comes round again.</li>
<li><b>Speed</b>: ≈ {rtf:.0f}× real time for a 3-gantry line on one core of a busy machine. Knobs are tuned on seeds 0-2 and scored on seeds 100-104.</li>
</ul>

<h2>Belt speed control</h2>
<p>S0 constant 0.3 m/s → S1 best constant → S2 rush when no robot has anything pickable (and brake back in time) → S3 also slow down so an item only this robot can take is still in reach when it is free → S4 one belt per gantry, each running S3 for its own robot, with zone control at the hand-offs.</p>
<img src="media/speed_ladder.png" alt="speed ladder">
<div class="grid2"><img src="media/frontier_B.png" alt="frontier B"><img src="media/frontier_B_loop.png" alt="frontier B loop"></div>
<img src="media/speed_loops.png" alt="loops">
<img src="media/layout.png" alt="layout">
<img src="media/speed_trace_B.png" alt="speed trace" style="max-width:780px">
<details><summary>All numbers</summary>{speed_rows(speed, lines + loops + list(SC.layout_benchmark()))}</details>

<h2>Pick scheduling</h2>
<p>Constant belt, more items than the robots can clear, so they must choose. P0 FIFO → P1 shortest cycle → P2 value rate (p · value ÷ cycle) → P3 last chance (an item a downstream robot could still take counts less) → P4 lookahead (plan 2-3 picks ahead, keep the first). Straight belts are scored against a time-indexed LP upper bound (clairvoyant, HiGHS) and a tighter planning reference; recirculating belts have no such bound (re-entry depends on what the robots did).</p>
{video("pick", "Recycling-like stream with a rare item worth 50× the best commodity (big yellow dot, yellow bin at gantry 1): FIFO lets it pass, value rate takes it.")}
<img src="media/pick_ladder.png" alt="pick ladder">
<details><summary>All numbers</summary>{pick_rows(pick)}</details>

<h2 id="replays">Drill into a replay</h2>
<p>Each button opens one 40 s episode in Rerun's web viewer: the line in 3D on a sim-time timeline, with the live stats beside it (rolling and cumulative picks/min, capture, belt speed per segment, backlog, robots busy) and a log of every pick and loss. Or download the <code>.rrd</code> and open several in the native viewer to compare.</p>
<ul class="rr">{replay_html}</ul>
<p class="muted">Locally: <code>uv run belt-sim rerun --scenario "B 3 arms, 6 cats"</code> (Rerun) or <code>uv run belt-sim viser</code> (3D gantries in the browser, policy dropdown).</p>

<h2>Checked against</h2>
<p>An independent event-driven constant-speed line sim (vendored in <code>tests/oracle_line_sim.py</code>): with the same rules the two must agree on capture, and a 15 % longer cycle must fail that comparison.</p>
{oracle}
<p>Tests also pin item conservation, common random numbers, no policy beating the capacity ceiling, grasps only inside the gantry's stroke and below the tracking speed (through the whole dwell), hopper and stream rates, belt hand-offs, recirculation, the LP bound dominating every policy, and that no speed policy stalls the line.</p>

<h2>Run it</h2>
<pre>git clone {REPO} && cd belt-sim
uv sync
uv run belt-sim ladder all        # tune + score both ladders -> docs/results/*.json
uv run belt-sim rerun             # replays with live stats in Rerun
uv run belt-sim wandb             # one W&amp;B run per tried config
uv run python docs/build_site.py  # this page</pre>
<p class="muted">MIT licence. Built from a design note on conveyor speed control and FIFO scheduling for robotic waste sorting.</p>
</main></body></html>"""
    (OUT / "index.html").write_text(page)
    (OUT / ".nojekyll").write_text("")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(OUT / "index.html", f"{size / 1e6:.1f} MB total")


if __name__ == "__main__":
    main()
