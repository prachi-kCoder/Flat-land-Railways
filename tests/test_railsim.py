"""
Checks for the shared railsim helpers.

Run:  python -m pytest tests/
"""

import pytest
from flatland.envs.rail_env_action import RailEnvActions

from railsim import EnvConfig, find_deadlocked_trains, make_env, train_position
from railsim.train_status import _possible_next_cells


def _timetable(env):
    return [
        (a.initial_configuration, sorted(a.targets), a.earliest_departure, a.latest_arrival)
        for a in env.agents
    ]


def test_same_seed_gives_same_scenario():
    env_a, _, _ = make_env(EnvConfig(num_trains=5, seed=7))
    env_b, _, _ = make_env(EnvConfig(num_trains=5, seed=7))
    assert _timetable(env_a) == _timetable(env_b)
    assert (env_a.rail.grid == env_b.rail.grid).all()


def test_different_seeds_give_different_scenarios():
    env_a, _, _ = make_env(EnvConfig(num_trains=5, seed=1))
    env_b, _, _ = make_env(EnvConfig(num_trains=5, seed=2))
    assert _timetable(env_a) != _timetable(env_b)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("num_trains", [5, 8])
def test_deadlock_detector(seed, num_trains):
    """
    Run the always-forward policy and check two things every step:
      1. No false alarms: a train reported deadlocked never moves again.
      2. No misses on head-on deadlocks: when Flatland reports two
         trains trying to swap cells and each one's only way out is the
         other's cell, both must be reported as deadlocked.

    Flatland's own `motion_check.deadlocked` is a per-step flag ("these
    trains tried to swap cells this step"). It also fires when a train
    on a switch could still turn away, so it is not a subset check.
    """
    env, _, _ = make_env(EnvConfig(num_trains=num_trains, seed=seed))
    frozen_at = {}  # handle -> cell where it was reported deadlocked

    for _ in range(env._max_episode_steps):
        actions = {h: RailEnvActions.MOVE_FORWARD for h in env.get_agent_handles()}
        _, _, dones, _ = env.step(actions)

        for handle, cell in frozen_at.items():
            assert train_position(env.agents[handle]) == cell, (
                f"train {handle} was reported deadlocked but moved"
            )

        ours = find_deadlocked_trains(env)
        exits = {
            h: set(_possible_next_cells(env, env.agents[h]))
            for h in env.motion_check.deadlocked
            if env.agents[h].current_configuration is not None
        }
        cells = {h: train_position(env.agents[h]) for h in exits}
        for a, b in [(a, b) for a in exits for b in exits if a < b]:
            if exits[a] == {cells[b]} and exits[b] == {cells[a]}:
                assert {a, b} <= ours, f"trains {a} and {b} are trapped head-on but not reported"

        for handle in ours:
            frozen_at.setdefault(handle, train_position(env.agents[handle]))

        if dones["__all__"]:
            break
