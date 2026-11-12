"""Vendored, unchanged in behaviour, as this repo's independent oracle: the event-driven constant-speed line sim
("F1") from BAM Robotics' facility model, written separately from this engine.

F1: an event-driven belt with N arms in a row. It checks F0's arm count against geometry.

Items arrive as a Poisson stream at the head of the belt and ride it at constant speed.
Arm k can grasp anything inside its window [k*w, (k+1)*w). A free arm commits to the most
downstream item the head camera has already seen and that it can still reach by the time
its gripper gets there, reserves it, and spends one cycle on it. The grasp succeeds with
probability ``grasp_success``, drawn when the gripper closes; a miss releases the item for
this arm or a downstream one to retry. Whatever leaves the last window is missed.

WHY hand-rolled and not SimPy: the whole model here is belt geometry (who can reach what,
when), which SimPy would not simplify. When F1 grows shifts, breakdowns and several
lines, move it onto SimPy, the stock DES engine, instead of growing this loop.
"""

from __future__ import annotations

import bisect
import heapq
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class LineSimResult:
    arms: int
    arrivals: int
    picked: int
    attempts: int
    recovery: float
    utilisation: float


class _NextAvailable:
    """Find the first not-yet-picked item at or after an index, in near O(1) (union-find)."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count + 1))  # index ``count`` is the end sentinel

    def find(self, index: int) -> int:
        parent = self.parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def remove(self, index: int) -> None:
        self.parent[index] = index + 1


def simulate_line(
    items_per_s: float,
    belt_speed_m_s: float,
    arms: int,
    window_m: float,
    cycle_time_s: float,
    reach_time_s: float,
    grasp_success: float,
    duration_s: float = 3_600.0,
    warmup_s: float = 300.0,
    seed: int = 0,
    trace: list[tuple[int, int, float, float, bool]] | None = None,
) -> LineSimResult:
    """Recovery of one line, measured on items that arrive after the warm-up.

    ``reach_time_s`` is the time from an arm being free to its gripper closing on an item
    (one move plus the grasp dwell); the item must still be in the window at that moment.
    ``trace``, if given, collects every grasp as (arm, item, arrival time, grasp time, success)
    for tests.
    """
    rng = random.Random(seed)
    horizon = warmup_s + duration_s
    arrivals: list[float] = []
    clock = 0.0
    while True:
        clock += rng.expovariate(items_per_s)
        if clock > horizon:
            break
        arrivals.append(clock)
    count = len(arrivals)
    picked = [False] * count
    available = _NextAvailable(count)
    reserved: set[int] = set()

    line_end_s = horizon + arms * window_m / belt_speed_m_s
    # Events: (time, sequence, kind, arm, item). FREE: an arm looks for work. GRASP: the
    # gripper closes and the outcome is drawn. WHY two events (Codex r1 #3): drawing the
    # outcome at commitment let a downstream arm retry an item before its first grasp had
    # even happened. Now a committed item is reserved until contact, and a miss releases it.
    FREE, GRASP = 0, 1
    events: list[tuple[float, int, int, int, int]] = []
    sequence = 0

    def push(time: float, kind: int, arm: int, item: int = -1) -> None:
        nonlocal sequence
        sequence += 1
        heapq.heappush(events, (time, sequence, kind, arm, item))

    busy_until = [0.0] * arms
    for arm in range(arms):
        push(0.0, FREE, arm)
    attempts = 0
    busy_s = 0.0
    while events:
        now, _, kind, arm, item = heapq.heappop(events)
        if now > line_end_s:
            continue
        if kind == GRASP:
            reserved.discard(item)
            success = rng.random() < grasp_success
            if success:
                picked[item] = True
                available.remove(item)
            if trace is not None:
                trace.append((arm, item, arrivals[item], now, success))
            if not success:
                # The item is back on offer: wake every idle arm to reconsider.
                for other in range(arms):
                    if busy_until[other] <= now:
                        push(now, FREE, other)
            continue
        if busy_until[arm] > now:
            continue  # a stale wake-up for an arm that is mid-cycle
        grasp_at = now + reach_time_s
        window_start = arm * window_m
        window_end = window_start + window_m
        # An item at x(t) = (t - arrival) * v is in this window at grasp time when its
        # arrival lies in [grasp_at - window_end / v, grasp_at - window_start / v].
        earliest_arrival = grasp_at - window_end / belt_speed_m_s
        # WHY min(now): an arm may only commit to items the camera at the head of the belt
        # has already seen. Without it, a slow arm "plans" for items that do not exist yet
        # (caught by test_a_reach_slower_than_the_window_picks_nothing).
        latest_arrival = min(grasp_at - window_start / belt_speed_m_s, now)
        index = available.find(bisect.bisect_left(arrivals, earliest_arrival))
        while index < count and index in reserved:
            index = available.find(index + 1)
        if index < count and arrivals[index] <= latest_arrival:
            attempts += 1
            busy_s += cycle_time_s
            reserved.add(index)
            busy_until[arm] = now + cycle_time_s
            push(grasp_at, GRASP, arm, index)
            push(now + cycle_time_s, FREE, arm)
        elif index < count:
            # Nothing reachable yet: wake when the next item has been seen and will be inside
            # the window at grasp time.
            enters_reach_at = arrivals[index] + window_start / belt_speed_m_s - reach_time_s
            push(max(enters_reach_at, arrivals[index], now + 1e-6), FREE, arm)

    measured = [i for i, arrival in enumerate(arrivals) if arrival >= warmup_s]
    picked_count = sum(1 for i in measured if picked[i])
    return LineSimResult(
        arms=arms,
        arrivals=len(measured),
        picked=picked_count,
        attempts=attempts,
        recovery=picked_count / max(len(measured), 1),
        utilisation=busy_s / (arms * line_end_s),
    )


def arms_for_recovery(
    target_recovery: float,
    start_arms: int,
    max_arms: int,
    **line_kwargs,
) -> tuple[int | None, list[LineSimResult]]:
    """Smallest arm count whose simulated recovery reaches the target, and every run tried."""
    runs = []
    for arms in range(max(1, start_arms), max_arms + 1):
        result = simulate_line(arms=arms, **line_kwargs)
        runs.append(result)
        if result.recovery >= target_recovery:
            return arms, runs
    return None, runs
