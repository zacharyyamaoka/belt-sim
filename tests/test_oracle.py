"""Constant speed: this engine must agree with the facility model's event-driven F1 line_sim (independent oracle)."""

import statistics

import pytest
from bam_facility_model.line_sim import simulate_line

from bam_belt_sim import pick_policies as P
from bam_belt_sim import speed_policies as S
from bam_belt_sim.engine import Sim
from bam_belt_sim.metrics import summarize
from bam_belt_sim.oracle import line_sim_scenario

CASES = [
    # items/s, belt m/s, arms, window m, cycle s, reach s, grasp success  (from F1's own fluid-bound test cases + 2)
    # WHY not F1's (5.0, 1.0, 3, 0.8, 1.5, 0.8, 0.7) as is: there reach x speed = 0.8 m = the window, and F1's window
    # test is closed at the far edge, so its first arm grasps items at exactly x = w the instant they arrive (388
    # attempts in 10 min) while a fixed-step engine never lands on that edge (0). Same case with a 0.5 s reach:
    (5.0, 1.0, 3, 0.8, 1.5, 0.5, 0.7),
    (2.0, 0.2, 2, 1.0, 2.98, 1.54, 0.95),
    (3.0, 0.4, 4, 1.0, 1.6, 0.6, 0.9),
    (1.0, 0.3, 1, 1.0, 2.0, 0.5, 0.9),
]


@pytest.mark.parametrize("case", CASES)
def test_constant_speed_recovery_matches_f1(case):
    rate, v, arms, w, cycle, reach, p = case
    f1 = [simulate_line(items_per_s=rate, belt_speed_m_s=v, arms=arms, window_m=w, cycle_time_s=cycle,
                        reach_time_s=reach, grasp_success=p, duration_s=1200, warmup_s=120, seed=s).recovery
          for s in range(6)]
    sc = line_sim_scenario(rate, v, arms, w, cycle, reach, p)
    ours = []
    for s in range(6):
        sim = Sim(sc, S.Constant(v), P.MostDownstream(), seed=s, dt=0.01).run(1320)
        ours.append(summarize(sim, 120).recovery)
    m1, m2 = statistics.mean(f1), statistics.mean(ours)
    se = (statistics.variance(f1) / len(f1) + statistics.variance(ours) / len(ours)) ** 0.5
    # WHY 3 SE + 1.5 %: different random streams (the means are compared, not runs), and a 10 ms step adds up to
    # one step of idle per cycle that the exact event loop does not have.
    assert abs(m1 - m2) <= 3 * se + 0.015, (m1, m2, se, f1, ours)


def test_the_oracle_gate_can_fail():
    """Mutation: a 15 % longer cycle in our engine only. The same comparison must now reject it."""
    rate, v, arms, w, cycle, reach, p = (3.0, 0.4, 4, 1.0, 1.6, 0.6, 0.9)
    f1 = [simulate_line(items_per_s=rate, belt_speed_m_s=v, arms=arms, window_m=w, cycle_time_s=cycle,
                        reach_time_s=reach, grasp_success=p, duration_s=1200, warmup_s=120, seed=s).recovery
          for s in range(6)]
    sc = line_sim_scenario(rate, v, arms, w, cycle * 1.15, reach, p)
    ours = [summarize(Sim(sc, S.Constant(v), P.MostDownstream(), seed=s, dt=0.01).run(1320), 120).recovery
            for s in range(6)]
    m1, m2 = statistics.mean(f1), statistics.mean(ours)
    se = (statistics.variance(f1) / len(f1) + statistics.variance(ours) / len(ours)) ** 0.5
    assert abs(m1 - m2) > 3 * se + 0.015, (m1, m2)
