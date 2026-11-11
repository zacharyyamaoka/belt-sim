"""Build the review report from docs/results/*.json + docs/media/* into the main checkout's reports/media/.

    cd ~/bam_ws/src/dev/bam_belt_sim
    uv run python docs/build_report.py

Charts are matplotlib PNGs (palette: the dataviz skill's validated categorical order, slots 1-5 for rungs S0-S4 /
P0-P4); every chart has its numbers in a table beside it. Videos are embedded (H.264) with the GIF as a sibling
file fallback.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
from bam_belt_sim import pick_policies as P  # noqa: E402
from bam_belt_sim import scenarios as SC  # noqa: E402
from bam_belt_sim import speed_policies as S  # noqa: E402
from bam_belt_sim.engine import Sim  # noqa: E402
from bam_belt_sim.metrics import summarize  # noqa: E402

# WHY a path import: bam_cell_kpis is the stdlib sibling every pick-and-place page shares (src/dev/bam_cell_kpis)
sys.path.insert(0, str(PKG.parent / "bam_cell_kpis"))
from bam_cell_kpis import (  # noqa: E402
    CellWindow,
    Period,
    assets_html,
    capture_rate,
    diversion_rate,
    flow,
    flow_display,
    share_text,
    tooltip,
)

RES = PKG / "docs" / "results"
MEDIA = PKG / "docs" / "media"
DATE = dt.date.today().isoformat()
OUT_DIR = Path("/home/bam/bam_ws/reports/media")
OUT = OUT_DIR / f"belt-sim-{DATE}.html"
RUNG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def img(path: Path, alt: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{html.escape(alt)}" style="width:{width}">'


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
    names = list(pick["scenarios"])
    rungs = list(pick["rungs"])
    fig, ax = plt.subplots(figsize=(8.2, 0.6 * len(names) + 1.2), dpi=130)
    for row, n in enumerate(names):
        y = len(names) - 1 - row
        rr = pick["scenarios"][n]["rungs"]
        vals = [100 * rr[r]["share_of_reference"] for r in rungs]
        ax.plot([min(vals), max(vals)], [y, y], color="#d6d5d0", lw=1.2, zorder=1)
        for j, r in enumerate(rungs):
            ax.scatter(100 * rr[r]["share_of_reference"], y + (j - 2) * 0.09, s=46, color=RUNG_COLORS[j], zorder=3)
    ax.axvline(100, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=8)
    ax.set_xlabel("value extracted, % of the LP planning reference (eval seeds)", fontsize=8, color=INK2)
    style_ax(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"{r} {pick['rungs'][r]}")
               for r, c in zip(rungs, RUNG_COLORS, strict=True)]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def oracle_rows(seeds=4, duration=900.0):
    from bam_facility_model.line_sim import simulate_line

    from bam_belt_sim.oracle import line_sim_scenario

    sys.path.insert(0, str(PKG / "tests"))
    from test_oracle import CASES

    rows = []
    for case in CASES:
        rate, v, arms, w, cycle, reach, p = case
        f1 = [simulate_line(items_per_s=rate, belt_speed_m_s=v, arms=arms, window_m=w, cycle_time_s=cycle,
                            reach_time_s=reach, grasp_success=p, duration_s=duration, warmup_s=120, seed=s).recovery
              for s in range(seeds)]
        sc = line_sim_scenario(rate, v, arms, w, cycle, reach, p)
        ours = [summarize(Sim(sc, S.Constant(v), P.MostDownstream(), seed=s, dt=0.01).run(duration + 120),
                          120).recovery for s in range(seeds)]
        rows.append((case, statistics.mean(f1), statistics.mean(ours)))
    return rows


def count_tests() -> int:
    import subprocess

    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(PKG / "tests")],
                         capture_output=True, text=True, cwd=PKG).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


def realtime_factor():
    sc = SC.speed_benchmark()["B 3 arms, 6 cats"]
    out = {}
    for label, pol in (("S0 constant", S.Constant(0.3)), ("S4 segmented", S.Segmented(S.DeadlineAware(0.2)))):
        t0 = time.perf_counter()
        Sim(sc, pol, P.FIFO(), seed=0).run(600)
        out[label] = 600 / (time.perf_counter() - t0)
    return out


def table(headers, rows):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    # WHY in a scroll box: a phone (390 px) scrolled the whole report sideways for the wide ladder tables (524 px)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def fmt_params(p: dict) -> str:
    return ", ".join(f"{k.replace('v_', 'v ').replace('_', ' ')} {v:g}" for k, v in p.items()) or "-"


def speed_table(speed, names):
    rows = []
    for n in names:
        sc = speed["scenarios"][n]
        for r, d in sc["rungs"].items():
            gate = "✓" if d["passes_gate"] else "✗"
            rows.append([html.escape(n) if r == "S0" else "", f"{r} {speed['rungs'][r]}",
                         f"<b>{d['sppm']:.1f}</b> ± {d['sppm_sd']:.1f}", f"{100 * d['recovery']:.1f} %", gate,
                         f"{d['mean_speed_m_s']:.2f}", f"{100 * d['utilisation']:.0f} %",
                         f"{100 * d['missed_intercept_share']:.1f} %", fmt_params(d["params"])])
    return table(["world", "rung", "SPPM", "recovery", "gate", "mean belt m/s", "arm busy", "missed intercepts",
                  "tuned knobs"], rows)


# ---------------------------------------------------------------------------------------------- facility KPIs
NO_MASS = "the belt sim's items carry a category and a value, no mass (engine.py ItemStream)"
ATTEMPTS_NOTE = ("attempts are counted over the whole run, the 2 min warm-up included (metrics.attempts_per_min); every "
                 "other number over the window after it")


def kpi_window(name: str, d: dict) -> CellWindow:
    """One rung of one world as a black box: items spawned onto the line, items picked into their bins.

    WHY "open" accounting: the counts are flows over a stretch of a running line (mean of the eval seeds), so an item
    picked in the window may have entered before it; bam_cell_kpis reports such a window's ratio as not measured
    instead of clipping it.
    """

    minutes = d["window_min"]
    picked = d["sppm"] * minutes
    return CellWindow(
        label=name, source="docs/results", attempts=d["attempts_per_min"] * minutes, successful_picks=picked,
        items_entering=d["entered_per_min"] * minutes, window_s=60.0 * minutes, entering_kg=None, diverted_kg=None,
        # capture: recovery = picked / (picked + lost) over items some arm had a bin for, so picked / recovery of them
        target_items=picked / d["recovery"] if d["recovery"] > 0 else None, diverted_target_items=picked,
        accounting="open",
        missing={"entering_kg": NO_MASS, "diverted_kg": NO_MASS,
                 "in_bins_kg": "the sim assumes a successful pick is pure (Daily Note Nov 11: successful means pure)"},
    )


def _flow_td(window: CellWindow, kpi: str, field: str, kind: str, note: str = "") -> str:
    reading = flow(window, kpi, Period.MINUTE)
    text, unit = flow_display(kpi, reading, Period.MINUTE)
    base = {"attempts": round(window.attempts, 6), "window_s": window.window_s, field: round(getattr(window, field), 6)}
    # WHY only the reasons this cell can need: 85 rows x 2 cells, and the page is already past the desktop preview's size
    missing = {name: window.missing[name] for name in ("window_s", field) if name in window.missing}
    if missing:
        base["missing"] = missing
    spec = html.escape(json.dumps({"base": base, "field": field, "kind": kind, "note": note}, separators=(",", ":")),
                       quote=True)
    title = html.escape(reading.why if not reading.measured else f"{text} {unit}" + (f" · {note}" if note else ""))
    return f'<td class="num" data-kpi-flow="{spec}" title="{title}">{html.escape(text)}</td>'


def _share_td(reading) -> str:
    title = html.escape(reading.why if not reading.measured else reading.note)
    css = "num" if reading.measured else "num kpi-nm"
    return f'<td class="{css}" title="{title}">{html.escape(share_text(reading))}</td>'


def facility_kpis_section(speed: dict, pick: dict) -> str:
    """Zach, 2026-11-11: "you can have very high throughput but low diversion rate, than not helpful". The two
    ladders' rungs read as facility KPIs, with the one rate switch every pick page shares (bam_cell_kpis)."""

    rows = []
    for ladder, rung_names, label in ((speed, speed["rungs"], "speed"), (pick, pick["rungs"], "pick")):
        for world, sc in ladder["scenarios"].items():
            first = True
            for rung, d in sc["rungs"].items():
                w = kpi_window(f"{world} {rung}", d)
                rate, capture = diversion_rate(w), capture_rate(w)
                speed_m_s = d.get("mean_speed_m_s")
                rows.append(
                    "<tr>"
                    + (f'<td rowspan="{len(sc["rungs"])}">{html.escape(world)}<div class="muted">{label} ladder</div></td>'
                       if first else "")
                    + f"<td>{html.escape(rung)} {html.escape(rung_names[rung])}</td>"
                    + f'<td class="num">{speed_m_s:.2f}</td>'
                    + _flow_td(w, "throughput_items", "items_entering", "items", "items spawned onto the line")
                    + _flow_td(w, "successful_picks", "successful_picks", "picks", "items picked into their bins")
                    + _share_td(rate) + _share_td(capture)
                    + f'<td class="num kpi-nm" title="{html.escape(NO_MASS)}">not measured</td>' * 2
                    + "</tr>")
                first = False
    # the story in two lines, from the numbers: a busy world at v1's speed, and the same world's best gated rung
    f_world = speed["scenarios"].get("F MRF mix", {}).get("rungs", {})
    story = ""
    if {"S0", "S1"} <= set(f_world):
        fast, slow = kpi_window("F S0", f_world["S0"]), kpi_window("F S1", f_world["S1"])
        story = (f"<p>On the MRF mix at v1's 0.3 m/s the line takes in <b>{f_world['S0']['entered_per_min']:.0f} items a "
                 f"minute</b> and diverts <b>{share_text(diversion_rate(fast))}</b> of them; slowed to the best gated "
                 f"constant speed ({f_world['S1']['mean_speed_m_s']:.2f} m/s) it takes in "
                 f"{f_world['S1']['entered_per_min']:.0f} and diverts {share_text(diversion_rate(slow))}, capturing "
                 f"{share_text(capture_rate(slow))} of the items an arm has a bin for. High throughput, low diversion.</p>")
    head = ("<tr><th>world</th><th>rung</th><th>belt m/s</th>"
            f'<th title="{html.escape(tooltip("throughput_items"))}">items in <span data-kpi-unit="items">/min</span></th>'
            f'<th title="{html.escape(tooltip("successful_picks"))}">successful picks <span data-kpi-unit="picks">/min</span></th>'
            f'<th title="{html.escape(tooltip("diversion_rate"))}">diversion rate (by count)</th>'
            f'<th title="{html.escape(tooltip("capture_rate"))}">capture rate</th>'
            f'<th title="{html.escape(tooltip("throughput"))}">throughput <span data-kpi-unit="mass">kg/min</span></th>'
            f'<th title="{html.escape(tooltip("diverted"))}">diverted <span data-kpi-unit="mass">kg/min</span></th></tr>')
    return f"""<h2 id="facility-kpis">Facility KPIs: the line as a black box</h2>
<div class="kpi-head"><p class="muted">Items are spawned onto the belt and some are diverted into bins. Throughput is what
goes through the belt; diversion rate is the share of it that ends in a bin; capture rate is the share of the items an arm
has a bin for. Mean of the eval seeds over the 12 min window after the warm-up. Definitions:
<code>src/dev/bam_cell_kpis</code>.</p><bam-rate-switch></bam-rate-switch></div>
{story}
<div class="wrap"><table class="kpis"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">Diversion and capture are by count: the sim's items have no mass, so every mass column says so instead of a
number. A ratio a window cannot state (picked more than entered, an edge effect of a running line) says so too, never
clipped. Per pick = per pick attempt, all arms together; {ATTEMPTS_NOTE}.</p>"""


def main(out: Path = OUT):
    side = out.with_suffix("")  # the GIF fallbacks' folder beside the page
    speed = json.loads((RES / "speed_ladder.json").read_text())
    pick = json.loads((RES / "pick_ladder.json").read_text())
    from bam_belt_sim.viz_mpl import render_all

    MEDIA.mkdir(parents=True, exist_ok=True)
    written = render_all(MEDIA)
    hopper = [n for n in SC.speed_benchmark() if speed["scenarios"][n]["supply"] == "hopper"]
    speed_dot_plot(speed, hopper + ["G stream 0.9/s"], MEDIA / "speed_ladder.png",
                   "Speed ladder: SPPM per rung (filled = recovery ≥ 90 %; stream: backlog does not grow)")
    speed_dot_plot(speed, list(SC.circulation_benchmark()) + ["B 3 arms, 6 cats", "D sparse 1/m"],
                   MEDIA / "recirc.png", "Recirculation: the same worlds with and without the loop back")
    speed_dot_plot(speed, list(SC.layout_benchmark()), MEDIA / "layout.png",
                   "Layout: the MRF line with its bins dealt three ways")
    frontier_plot(speed, "B 3 arms, 6 cats", MEDIA / "frontier_B.png")
    frontier_plot(speed, "D sparse 1/m", MEDIA / "frontier_D.png")
    pick_plot(pick, MEDIA / "pick_ladder.png")
    oracle = oracle_rows()
    n_tests = count_tests()
    rtf = realtime_factor()

    side.mkdir(parents=True, exist_ok=True)
    for key in ("speed", "pick"):
        (side / written[key]["gif"].name).write_bytes(written[key]["gif"].read_bytes())

    B = speed["scenarios"]["B 3 arms, 6 cats"]["rungs"]
    D = speed["scenarios"]["D sparse 1/m"]["rungs"]
    pb = pick["scenarios"]["PB MRF values + gold"]["rungs"]

    def best_gate(n):
        sc = speed["scenarios"][n]["rungs"]
        return max((r for r in sc if sc[r]["passes_gate"]), key=lambda r: sc[r]["sppm"])

    winners = {n: best_gate(n) for n in hopper}

    def short(n):
        return n.split(" ")[0]

    winners_txt = ", ".join(f"{short(n)} → <b>{w}</b>" for n, w in winners.items())
    s4_gain = {n: speed["scenarios"][n]["rungs"]["S4"]["sppm"] / speed["scenarios"][n]["rungs"]["S1"]["sppm"] - 1
               for n, w in winners.items() if w == "S4"}
    s4_txt = ", ".join(f"{short(n)} +{100 * g:.0f} % over the best constant speed" for n, g in s4_gain.items())
    nots = []
    for n, w in winners.items():
        if w == "S4":
            continue
        r4 = speed["scenarios"][n]["rungs"]["S4"]
        if not r4["passes_gate"]:
            nots.append(f"on {short(n)} it misses the gate on the evaluation seeds ({100 * r4['recovery']:.0f} %)")
        else:
            rw = speed["scenarios"][n]["rungs"][w]
            nots.append(f"on {short(n)} {w} beats it ({rw['sppm']:.0f} vs {r4['sppm']:.0f} picks/min)")
    s4_not_txt = "; ".join(nots)

    eq = [n for n in pick["scenarios"] if "gold" not in n]
    eq_g = [pick["scenarios"][n]["rungs"]["P1"]["value_per_min"] / pick["scenarios"][n]["rungs"]["P0"]["value_per_min"] - 1
            for n in eq]
    eq_gain_txt = f"{100 * min(eq_g):.0f}-{100 * max(eq_g):.0f} %"
    pick_rows = []
    for n, scd in pick["scenarios"].items():
        for r, d in scd["rungs"].items():
            unit = scd["value_unit"]
            val = f"{d['value_per_min']:.3f} $" if unit == "USD" else f"{d['value_per_min']:.1f}"
            gold = f"{100 * d['gold_recovery']:.0f} % of {d['gold_items_total']}" if "gold_recovery" in d else ""
            pick_rows.append([html.escape(n) if r == "P0" else "", f"{r} {pick['rungs'][r]}", f"<b>{val}</b>",
                              f"{d['sppm']:.1f}", f"{100 * d['share_of_reference']:.0f} %",
                              f"{100 * d['share_of_bound']:.0f} %", gold, fmt_params(d["params"])])
    pick_tbl = table(["world", "rung", "value / min", "SPPM", "of LP reference", "of LP bound", "gold picked",
                      "tuned knobs"], pick_rows)
    oracle_tbl = table(["items/s, belt m/s, arms, window m, cycle s, reach s, p", "F1 line_sim recovery",
                        "this engine", "Δ"],
                       [[", ".join(f"{x:g}" for x in c), f"{a:.3f}", f"{b:.3f}", f"{b - a:+.3f}"]
                        for c, a, b in oracle])

    def video(key, caption):
        mp4 = written[key]["mp4"]
        poster = written[key]["png"]
        gif = side.name + "/" + written[key]["gif"].name
        return (f'<figure><video controls autoplay muted loop playsinline poster="data:image/png;base64,{b64(poster)}" '
                f'src="data:video/mp4;base64,{b64(mp4)}" style="width:100%"></video>'
                f'<figcaption>{caption} · <a href="{gif}">GIF fallback</a></figcaption></figure>')

    kpi_assets = assets_html()
    kpis_html = facility_kpis_section(speed, pick)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Belt sim ladders</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--card:#f4f3f0}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}}
main{{max-width:1080px;margin:0 auto;padding:28px 18px 80px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:19px;margin:38px 0 8px;border-top:1px solid var(--line);padding-top:18px}}
h3{{font-size:15px;margin:22px 0 6px}}
.verdict{{font-size:17px;background:var(--card);border-left:4px solid var(--accent);padding:12px 16px;border-radius:6px}}
.muted{{color:var(--ink2)}} code{{font-size:13px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin:8px 0}}
th,td{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}}
th{{color:var(--ink2);font-weight:600}}
figure{{margin:14px 0}} figcaption{{font-size:12.5px;color:var(--ink2)}}
img{{background:#fff;border-radius:4px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media (max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
ul{{padding-left:20px}} li{{margin:3px 0}}
details{{margin:8px 0}} summary{{cursor:pointer;color:var(--ink2)}}
.wrap{{overflow-x:auto;max-width:100%}} table.kpis td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.kpi-head{{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;justify-content:space-between}} .kpi-head p{{margin:0;max-width:640px}}
bam-rate-switch{{--brs-accent:var(--accent);--brs-border:var(--line);--brs-muted:var(--ink2);--brs-bg:var(--bg);--brs-fg:var(--ink)}}
</style>{kpi_assets}</head><body><main>
<h1>Conveyor belt sim: speed control and pick scheduling</h1>
<p class="muted">bam_belt_sim · {DATE} · from Daily Note Nov 11 2026, "Conveyor belt speed control and FIFO object scheduling"</p>
<p class="verdict"><b>Two ladders climbed on one fast headless sim.</b> Speed (best rung at recovery ≥ 90 %, per hopper world): {winners_txt}.
One belt per arm (S4) is the biggest single step where it wins: {s4_txt}; on world B it keeps {100 * B['S4']['sppm'] / B['S0']['sppm']:.0f} % of v1's picks/min at
{100 * B['S4']['recovery']:.0f} % recovery instead of {100 * B['S0']['recovery']:.0f} %. It is not universal: {s4_not_txt}.
Scheduling: value-rate picking (P2) extracts <b>+{100 * (pb['P2']['value_per_min'] / pb['P0']['value_per_min'] - 1):.0f} % value/min</b> on the MRF + gold world and takes {100 * pb['P2'].get('gold_recovery', 0):.0f} % of gold nuggets vs FIFO's {100 * pb['P0'].get('gold_recovery', 0):.0f} %; with equal values, shortest-cycle beats FIFO by {eq_gain_txt} and the cleverer rungs add under 1 %.
Simulation only: points on a 2D belt, UR5e-class arms from the facility model.</p>

{video("speed", "World B (3 arms, 6 categories, hopper 4 items/m), same seed and 20 s of sim time (played at 1.5×) for all four: S0 constant 0.3 m/s · S1 best constant · S3 deadline-aware · S4 one belt per arm (segment speeds printed above each belt). Dots = items coloured by category; squares = bins; dashed circle = reach; tool ring: red approach, orange grasp, green delivering.")}

<h2>What is measured</h2>
<ul>
<li><b>SPPM</b>: successful picks per minute, counted after a 2 min warm-up (the daily note's top KPI; "successful means pure").</li>
<li><b>Recovery</b>: picked ÷ (picked + lost) for items some arm had a bin for. Lost = left the end of the line (or ran out of passes when recirculating).</li>
<li><b>Speed ladder headline: SPPM at recovery ≥ 90 %.</b> Hopper supply brings more items the faster the belt runs, so without a recovery gate "run flat out" wins SPPM and landfills everything. With stream supply (fixed items/s), the gate is "the head backlog does not grow".</li>
<li><b>Pick ladder headline: value per minute</b>, and its share of an LP bound (no policy can beat it) and a tighter LP planning reference (see the method notes).</li>
<li>Each rung's knobs are tuned on seeds {speed['tune_seeds']} and scored on different seeds {speed['eval_seeds']}: same items for every rung (common random numbers).</li>
</ul>

{kpis_html}

<h2>Speed ladder</h2>
<p>S0 constant 0.3 m/s (v1) → S1 best constant speed → S2 rush when idle (Zach's idea: speed up while no arm has anything pickable; brake back in time) →
S3 deadline-aware (also slow down so an item only this arm can take is still in reach when the arm frees up) → S4 one belt per arm, each running S3 on its own arm, with zone control at the hand-offs.</p>
{img(MEDIA / "speed_ladder.png", "speed ladder dot plot")}
<div class="grid2">{img(MEDIA / "frontier_B.png", "frontier B")}{img(MEDIA / "frontier_D.png", "frontier D")}</div>
<p class="muted">Frontier: each small dot is one tuned configuration (mean of the tuning seeds). Up and to the right is better. Past the 90 % gate S4's points sit highest on both worlds: separate belts let a busy arm hold its own items while the next belt keeps feeding the next arm.</p>
{img(MEDIA / "speed_trace_B.png", "belt speed trace", "80%")}
<details open><summary>All numbers (eval seeds, mean ± sd over 5 seeds)</summary>{speed_table(speed, list(SC.speed_benchmark()))}</details>

<h3>What the speed ladder says</h3>
<ul>
<li><b>Constant speed is a recovery-vs-throughput dial.</b> v1 (0.3 m/s) gets the most picks/min on dense worlds but loses 30-75 % of targets; the best gated constant speed is 0.08-0.2 m/s and gives up 4-35 % of v1's picks.</li>
<li><b>Rush-when-idle (S2) pays where the belt is sparse</b> (D: {D['S2']['sppm']:.0f} vs {D['S1']['sppm']:.0f} picks/min) and is neutral where arms are always busy: there is never "nothing in view" to rush through.</li>
<li><b>One belt per arm (S4) is the biggest step where arms want different speeds</b> (one is busy with an item only it can take, the next is starved): a single belt has one speed for both, a belt each serves both. Where every arm is busy all the time (dense E with every category on two arms) there is nothing to arbitrate and constant speed wins; on bursty C the tuned S4 lands just under the gate on new seeds - its knobs over-fit the tuning seeds there.</li>
<li><b>Stream supply flips the answer:</b> with a fixed infeed rate, crawling (0.05 m/s) is best: items cannot arrive faster than upstream sends them, so slowing only makes the belt denser. There the smart policies tie or lose against the best constant speed.</li>
</ul>

<h3>Recirculation and layout</h3>
<div class="grid2">{img(MEDIA / "recirc.png", "recirculation")}{img(MEDIA / "layout.png", "layout")}</div>
<ul>
<li><b>Recirculating</b> (items reaching the end go back to the head, max 3 passes): even v1's constant 0.3 m/s clears the gate (B: {speed['scenarios']['B 3 arms, 6 cats + recirc']['rungs']['S0']['sppm']:.0f} picks/min at {100 * speed['scenarios']['B 3 arms, 6 cats + recirc']['rungs']['S0']['recovery']:.0f} %). The loop buys the second chance that speed control otherwise buys by slowing down.</li>
<li><b>Layout matters as much as speed control.</b> Same MRF line: dealing bins by load and giving the two heaviest categories a second bin downstream (L2) lifts the best constant speed from {speed['scenarios']['L0 MRF round-robin bins']['rungs']['S1']['sppm']:.0f} to {speed['scenarios']['L2 balanced + top-2 doubled']['rungs']['S1']['sppm']:.0f} picks/min before any speed control.</li>
</ul>

<h2>Pick-scheduling ladder</h2>
<p>Constant belt 0.25 m/s, stream supply 1.5 items/s (more than the arms can clear, so they must choose). P0 FIFO (v1) → P1 shortest cycle → P2 value rate (p·value ÷ cycle: the gold nugget rule) → P3 last chance (an item a downstream arm could still take counts less) → P4 lookahead (plan 2-3 picks ahead, keep the first).</p>
{video("pick", f"MRF stream with a 0.3 % gold-nugget category worth 50× the best commodity (big yellow dot; yellow bin at arm 1). Same seed/window, found by search (seed {written['gold_window']['seed']}): FIFO lets the gold pass, value rate takes it.")}
{img(MEDIA / "pick_ladder.png", "pick ladder")}
{pick_tbl}
<ul>
<li><b>Gold:</b> FIFO's arm is usually busy with whatever came first when the nugget passes; value rate re-ranks on every free moment. Count-based SPPM barely moves, the money does.</li>
<li><b>Equal values:</b> shortest cycle first (nearest item, nearest bin) is the whole win over FIFO ({eq_gain_txt}); value rate equals it when values and p are equal. Last chance and 2-3 pick lookahead add under 1 %, even with every category on two arms (PC): at 0.25 m/s the arm cycle, not the order, is the bottleneck. The best rungs reach 85-91 % of the LP planning reference.</li>
<li>LP bound and reference are clairvoyant (they know every future item) and let arms start free; the reference also assumes every pick starts from the nearest bin.</li>
</ul>

<h2>The simulator</h2>
<ul>
<li><b>Headless engine</b> (<code>engine.py</code>, numpy only): fixed 50 ms step, belt of N segments with independent speeds (forward or reverse, accel-limited), hopper or stream infeed with min spacing, optional recirculation; arms with a reach disc, bins per category, trapezoidal tool moves (UR5e: 0.8 m/s, 1.2 m/s², grasp 0.3 s, release 0.2 s), moving-target intercept with re-planning, per-item grasp probability, and a hard belt-speed line (0.5 m/s) above which an arm does not try.</li>
<li><b>Speed:</b> {', '.join(f'{k} ≈ {v:.0f}× real time' for k, v in rtf.items())} for a 3-arm line on this machine while it was loaded by other jobs (one core each). The full two-ladder sweep (≈ 3,000 episodes) took {speed.get('wall_s', 0) / 60:.0f} + {pick.get('wall_s', 0) / 60:.0f} min on 18 workers.</li>
<li><b>Views:</b> matplotlib 2D replays (the videos above) and a Viser 3D replay on the house viewer (policy dropdown, playback, ViewCube): <code>bam-belt-sim viser</code>.</li>
</ul>
<h3>Independent oracle: the facility model's F1 line_sim</h3>
<p>At constant speed the facility model already had an event-driven line sim. Restated in this engine (same windows, fixed cycle, most-downstream rule, Poisson stream) recovery must agree; a test that lengthens our cycle by 15 % must fail that comparison (it does).</p>
{oracle_tbl}
<p class="muted">Found on the way: F1's window test is closed at the far edge, so when reach time × belt speed equals the window, its first arm grasps items at exactly x = w the instant they arrive (388 attempts in 10 min where a fixed-step engine gets 0). That case is swapped for a 0.5 s reach in the test; F1 itself is untouched.</p>
<h3>Tests</h3>
<p>{n_tests} tests: item conservation, common random numbers, capacity ceiling never beaten, grasps only inside reach and below tracking speed (this caught a real bug: items drifted out of reach during the grasp dwell), hopper density and stream rate, hand-off holds and spacing, recirculation helps, LP bound dominates policies, the F1 oracle and its mutation.</p>

<h2>Decided for you</h2>
<ul>
<li><b>One engine, two ladders</b> instead of two engines: speed and pick order are two policy axes over the same belt, so each rung changes one thing. Undo: split <code>speed_policies.py</code>/<code>pick_policies.py</code> into their own worlds.</li>
<li><b>Success line = SPPM at recovery ≥ 90 %</b> (hopper) / backlog-stable (stream). The 90 % is a knob: <code>ladder.RECOVERY_GATE</code>.</li>
<li><b>Grasp speed limit as a hard line at 0.5 m/s</b>, per the note. Per-item success p is known to the scheduler.</li>
<li><b>No robot_msgs types</b>: positions are scalars on a 2D belt; nothing pose-shaped crosses the package boundary.</li>
<li><b>Landed on main</b> with this report.</li>
</ul>
<h2>Left / next rungs</h2>
<ul>
<li>Speed S5: model-predictive speed control (roll the sim forward per candidate speed) and the joint speed × pick ladder (only one axis moved at a time here).</li>
<li>Pick P5: assignment across arms at once (Hungarian / auction), and executing the LP plan itself.</li>
<li>Layout search proper: arm pitch, bins per arm (6 reachable bins), and where the second bin of a category should go.</li>
<li>Real numbers: the BAM arm's cycle and grasp success from the pick worlds, real mixes from bam-waste-objects, belt speed / grasp-success curve instead of the hard line.</li>
</ul>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(out, f"{out.stat().st_size / 1e6:.2f} MB")


def kpis_only(out: Path) -> None:
    """Only the facility-KPI section on a bare page: seconds instead of the full build's minutes (charts, videos,
    oracle runs), for checking the section and its rate switch."""

    speed = json.loads((RES / "speed_ladder.json").read_text())
    pick = json.loads((RES / "pick_ladder.json").read_text())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Belt sim facility KPIs</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--card:#f4f3f0}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 80px}} h2{{font-size:19px;margin:0 0 8px}} .muted{{color:var(--ink2)}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin:8px 0}} th,td{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}}
th{{color:var(--ink2);font-weight:600}} code{{font-size:13px}}
.wrap{{overflow-x:auto;max-width:100%}} table.kpis td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.kpi-head{{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;justify-content:space-between}} .kpi-head p{{margin:0;max-width:640px}}
bam-rate-switch{{--brs-accent:var(--accent);--brs-border:var(--line);--brs-muted:var(--ink2);--brs-bg:var(--bg);--brs-fg:var(--ink)}}
</style>{assets_html()}</head><body><main>{facility_kpis_section(speed, pick)}</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT, help=f"the page (default {OUT})")
    parser.add_argument("--kpis-only", action="store_true", help="only the facility-KPI section, in seconds")
    args = parser.parse_args()
    kpis_only(args.out) if args.kpis_only else main(args.out)
