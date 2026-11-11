"""The LP bound must sit above every policy (on the same items), and the tighter reference below the bound."""

import pytest

from bam_belt_sim import pick_policies as P
from bam_belt_sim import scenarios as SC
from bam_belt_sim import speed_policies as S
from bam_belt_sim.bound import lp_bound, policy_value_on
from bam_belt_sim.engine import Sim


@pytest.mark.parametrize("name", ["PA 3 arms, 6 equal cats", "PD 1 arm, 6 bins"])
def test_bound_dominates_policies(name):
    sc = SC.pick_benchmark()[name]
    b = lp_bound(sc, 0.25, seed=5, t0=60.0, t1=180.0)
    ref = lp_bound(sc, 0.25, seed=5, t0=60.0, t1=180.0, reference=True)
    assert ref.value <= b.value + 1e-6
    for pol in (P.FIFO(), P.ShortestCycle(), P.Lookahead()):
        sim = Sim(sc, S.Constant(0.25), pol, seed=5).run(220)
        got = policy_value_on(sim, b.item_ids)
        # a single run's realized value is random; the bound is on the expectation. Allow 2 sd of a binomial.
        slack = 2.0 * (got * 0.1) ** 0.5 + 1
        assert got <= b.value + slack, (pol.name, got, b.value)
