"""
Checks for the conflict-free planner and the FCFS / Priority dispatchers.

Run:  python -m pytest tests/
"""

import pytest

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env, usable_seeds
from railsim.planner import OFF_MAP, plan_all
from railsim.policies import FCFSDispatcher, PriorityDispatcher
from railsim.train_status import find_deadlocked_trains, train_position

DISPATCHERS = [FCFSDispatcher, PriorityDispatcher]
SEEDS = usable_seeds(EnvConfig(num_trains=8), count=12)


def _play(env, policy):
    policy.reset(env)
    deadlock_seen = False
    done = False
    while not done:
        _, _, dones, _ = env.step(policy.act(env))
        deadlock_seen |= bool(find_deadlocked_trains(env))
        done = dones["__all__"]
    return deadlock_seen


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("policy_class", DISPATCHERS)
def test_booked_routes_never_conflict(seed, policy_class):
    """No two trains are booked in the same cell at the same time, and no
    two trains swap cells head-on in one step."""
    env = make_env(EnvConfig(num_trains=8, seed=seed))[0]
    routes, parked = plan_all(env, policy_class().priority_order(env, 0), 0)
    assert not parked

    cell_at = {
        handle: {t: c[0] for t, (c, _a) in route.items() if c is not OFF_MAP}
        for handle, route in routes.items()
    }

    occupied = {}
    for handle, cells in cell_at.items():
        for t, cell in cells.items():
            assert (cell, t) not in occupied, f"trains {handle} and {occupied[(cell, t)]} share {cell} at {t}"
            occupied[(cell, t)] = handle

    for a, cells_a in cell_at.items():
        for t, cell in cells_a.items():
            before = cells_a.get(t - 1)
            if before is None or before == cell:
                continue
            b = occupied.get((before, t))
            if b is not None and b != a:
                assert cell_at[b].get(t - 1) != cell, f"trains {a} and {b} swap {before} <-> {cell} at {t}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("policy_class", DISPATCHERS)
def test_plans_are_followed_exactly_without_disruptions(seed, policy_class):
    env = make_env(EnvConfig(num_trains=8, seed=seed))[0]
    policy = policy_class()
    assert not _play(env, policy)
    assert policy.divergences == 0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("policy_class", DISPATCHERS)
@pytest.mark.parametrize("spec", [
    DisruptionSpec(count=2, duration=20),
    DisruptionSpec(count=3, duration=40, simultaneous=True),
])
def test_no_deadlocks_with_disruptions(seed, policy_class, spec):
    env = make_env(EnvConfig(num_trains=8, seed=seed), spec)[0]
    policy = policy_class()
    assert not _play(env, policy)
    # Every breakdown that hit a running train forces a re-plan.
    hits = list(env.disruption_status.applied.values()).count("on_map")
    assert policy.divergences >= min(hits, 1)


def test_dispatched_trains_arrive_without_disruptions():
    """Sanity check on overall quality: most trains arrive when nothing breaks."""
    arrived = total = 0
    for seed in SEEDS:
        env = make_env(EnvConfig(num_trains=8, seed=seed))[0]
        _play(env, PriorityDispatcher())
        arrived += sum(a.state.name == "DONE" for a in env.agents)
        total += len(env.agents)
    assert arrived / total > 0.9
