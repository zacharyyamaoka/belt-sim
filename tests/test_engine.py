"""Engine invariants: each is a check that a cheating or broken engine would fail."""

import numpy as np
import pytest

from bam_belt_sim import pick_policies as P
from bam_belt_sim import scenarios as SC
from bam_belt_sim import speed_policies as S
from bam_belt_sim.engine import HELD, LOST, ON_BELT, PICKED, QUEUED, Sim
from bam_belt_sim.metrics import summarize


def _b():
    return SC.speed_benchmark()["B 3 arms, 6 cats"]


@pytest.mark.parametrize("pol", [S.Constant(0.3), S.DeadlineAware(0.3), S.Segmented(S.DeadlineAware(0.3))],
                         ids=lambda p: p.name[:12])
def test_every_item_that_entered_is_accounted_for(pol):
    sim = Sim(_b(), pol, P.FIFO(), seed=3).run(240)
    n = sim.next_fresh
    st = sim.status[:n]
    assert np.all(np.isin(st, [ON_BELT, HELD, PICKED, LOST, QUEUED]))
    assert int(np.sum(st == PICKED)) == len(sim.log.picks)
    assert int(np.sum(st == LOST)) == len(sim.log.losses)
    assert set(sim.on_belt.tolist()) == set(np.nonzero(st == ON_BELT)[0].tolist())


def test_common_random_numbers_same_items_for_every_policy():
    a = Sim(_b(), S.Constant(0.3), P.FIFO(), seed=7).run(60)
    b = Sim(_b(), S.Segmented(S.DeadlineAware(0.2)), P.ShortestCycle(), seed=7).run(60)
    n = min(a.next_fresh, b.next_fresh)
    assert n > 20
    assert np.array_equal(a.cat[:n], b.cat[:n]) and np.array_equal(a.y[:n], b.y[:n])
    assert np.array_equal(a.stream.draws[:n], b.stream.draws[:n])


@pytest.mark.parametrize("name", ["A 1 arm, 2 cats", "B 3 arms, 6 cats", "D sparse 1/m"])
def test_no_policy_beats_the_capacity_ceiling(name):
    sc = SC.speed_benchmark()[name]
    for pol in (S.Constant(0.5), S.Segmented(S.DeadlineAware(0.3))):
        m = summarize(Sim(sc, pol, P.ShortestCycle(), seed=1).run(400), 60)
        assert m.sppm <= sc.capacity_ppm()


def test_arms_grasp_only_inside_reach_and_below_tracking_speed(monkeypatch):
    seen = []
    orig = Sim._grasp_outcome

    def spy(self, k, a, spec):
        i = a.item
        u = float(self.item_speed(np.array([i]))[0])
        seen.append((bool(self.in_reach(k, self.x[i], self.y[i])), abs(u) <= spec.v_track_max_m_s + 1e-9))
        return orig(self, k, a, spec)

    monkeypatch.setattr(Sim, "_grasp_outcome", spy)
    Sim(_b(), S.RushWhenIdle(0.3), P.FIFO(), seed=2).run(300)
    assert len(seen) > 50
    # checked at the END of the grasp dwell: the tracked item must still be inside the reach disc
    assert all(r for r, _ in seen)
    assert all(t for _, t in seen)


def test_hopper_feeds_density_per_metre_of_belt_travel():
    sc = SC.speed_benchmark()["B 3 arms, 6 cats"]
    for v in (0.1, 0.4):
        sim = Sim(sc, S.Constant(v), P.FIFO(), seed=0).run(300)
        travelled = v * (300 - v / sc.belt.accel_m_s2 / 2)
        assert sim.entries / travelled == pytest.approx(sc.supply.density_per_m, rel=0.12)


def test_stream_feeds_its_rate_whatever_the_belt_speed():
    sc = SC.speed_benchmark()["G stream 0.9/s"]
    for v in (0.1, 0.4):
        sim = Sim(sc, S.Constant(v), P.FIFO(), seed=0).run(600)
        assert sim.entries / 600 == pytest.approx(0.9, rel=0.1)


def test_a_stopped_downstream_belt_holds_items_at_the_handoff():
    sc = SC.speed_benchmark()["B 3 arms, 6 cats"]

    class Stop2:
        name = "stop belt 2"

        def command(self, sim):
            v = np.full(sim.n_seg, 0.4)
            v[2] = 0.0
            return v

    class Never:
        name = "never pick"

        def choose(self, sim, arm, cand):
            return None

    sim = Sim(sc, Stop2(), Never(), seed=0).run(60)
    edge = sc.belt.boundaries_m[1]
    x = sim.x[sim.on_belt]
    assert not np.any((x > edge) & (x < sc.belt.boundaries_m[2]))  # nothing crossed onto the stopped belt
    assert np.sum(np.abs(x - edge) < 1e-3) > 3  # they wait at the boundary


def test_transfer_spacing_never_packs_tighter_than_min_gap_downstream():
    sc = SC.speed_benchmark()["B 3 arms, 6 cats"]

    class FastSlow:
        name = "fast then slow"

        def command(self, sim):
            v = np.full(sim.n_seg, 0.6)
            v[2:] = 0.05
            return v

    sim = Sim(sc, FastSlow(), P.FIFO(), seed=0, record_every=10).run(120)
    lo, hi = sc.belt.boundaries_m[1], sc.belt.boundaries_m[2]
    for f in sim.log.frames[-20:]:
        xs = np.sort(f["x"][(f["x"] > lo + 0.01) & (f["x"] < hi)])
        if xs.size > 1:
            assert np.min(np.diff(xs)) >= sc.supply.min_gap_m - 1e-6


def test_recirculation_gives_second_chances():
    base = SC.speed_benchmark()["B 3 arms, 6 cats"]
    circ = SC.circulation_benchmark()["B 3 arms, 6 cats + recirc"]
    a = summarize(Sim(base, S.Constant(0.3), P.FIFO(), seed=0).run(400), 60)
    b = summarize(Sim(circ, S.Constant(0.3), P.FIFO(), seed=0).run(400), 60)
    assert b.recovery > a.recovery + 0.15


def test_value_rate_takes_the_gold_fifo_lets_pass():
    sc = SC.pick_benchmark()["PB MRF values + gold"]
    g = [c.name for c in sc.categories].index("gold")
    got = {}
    for pol in (P.FIFO(), P.ValueRate()):
        sim = Sim(sc, S.Constant(0.25), pol, seed=100).run(600)
        n = sim.next_fresh
        gold = np.nonzero(sim.cat[:n] == g)[0]
        got[pol.name] = (sum(1 for p in sim.log.picks if p[4] == g), gold.size)
    assert got["P2 value rate"][0] > got["P0 FIFO"][0]
