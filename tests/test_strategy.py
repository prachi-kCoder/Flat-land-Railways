"""
Checks for rescheduling-strategy selection (railsim/strategy.py).

Run:  python -m pytest tests/
"""

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env, usable_seeds
from railsim.metrics import network_delay
from railsim.policies import FCFSDispatcher, PriorityDispatcher
from railsim.runner import run_episode
from railsim.strategy import (
    NAMES, EDFDispatcher, GreedySelect, SPFDispatcher, StrategyEnv, greedy_action, plan_lateness,
)
from railsim.train_status import find_deadlocked_trains

SPEC = DisruptionSpec(count=1, duration=20)
SEEDS = usable_seeds(EnvConfig(num_trains=8), count=6, start=1000)
FIXED = [FCFSDispatcher, PriorityDispatcher, EDFDispatcher, SPFDispatcher]


def _play_env(env, scenario, choose):
    obs, _ = env.reset(options={"scenario": scenario})
    total, done = 0.0, False
    while not done:
        assert env.observation_space.contains(obs)
        obs, reward, done, _, info = env.step(choose(obs))
        total += reward
    return total, info


def test_rule_order_matches_names():
    """Action i in the RL env must mean the rule FIXED[i] uses."""
    assert [p.name for p in FIXED] == NAMES


def test_env_passes_sb3_checker():
    check_env(StrategyEnv(SEEDS), warn=True)


@pytest.mark.parametrize("scenario", SEEDS)
def test_rewards_sum_to_minus_delay(scenario):
    env = StrategyEnv(SEEDS)
    rng = np.random.default_rng(scenario)
    total, info = _play_env(env, scenario, lambda obs: int(rng.integers(len(NAMES))))
    assert info["final_delay"] == network_delay(env.env)
    assert total == pytest.approx(-(info["final_delay"] - env.delay_at_first_decision) / 8)


@pytest.mark.parametrize("scenario", SEEDS)
@pytest.mark.parametrize("index", range(len(FIXED)))
def test_constant_choice_equals_fixed_rule(scenario, index):
    """Always choosing rule i in the RL env reproduces that rule's dispatcher."""
    env = StrategyEnv(SEEDS, num_trains=8, disruptions=SPEC)
    _, info = _play_env(env, scenario, lambda obs: index)
    result = run_episode(EnvConfig(num_trains=8, seed=scenario), FIXED[index], SPEC)
    assert (info["final_delay"], sum(a.state.name == "DONE" for a in env.env.agents)) == \
           (result.total_delay, result.arrived)


@pytest.mark.parametrize("scenario", SEEDS)
def test_greedy_action_equals_greedy_select(scenario):
    """greedy_action (read from the observation) behaves exactly like GreedySelect."""
    env = StrategyEnv(SEEDS, num_trains=8, disruptions=SPEC)
    _, info = _play_env(env, scenario, greedy_action)
    result = run_episode(EnvConfig(num_trains=8, seed=scenario), GreedySelect, SPEC)
    assert (info["final_delay"], sum(a.state.name == "DONE" for a in env.env.agents)) == \
           (result.total_delay, result.arrived)


@pytest.mark.parametrize("scenario", SEEDS[:3])
def test_greedy_select_never_deadlocks(scenario):
    env = make_env(EnvConfig(num_trains=12, seed=scenario), DisruptionSpec(count=3, duration=40, simultaneous=True))[0]
    policy = GreedySelect()
    policy.reset(env)
    done = False
    while not done:
        _, _, dones, _ = env.step(policy.act(env))
        assert not find_deadlocked_trains(env)
        done = dones["__all__"]


def test_plan_lateness_is_zero_for_an_on_time_plan():
    """At step 0 with a relaxed timetable, a plan where every train is on
    time must have zero lateness, and lateness is never negative."""
    for scenario in SEEDS:
        env = make_env(EnvConfig(num_trains=3, seed=scenario))[0]
        dispatcher = GreedySelect()
        dispatcher.reset(env)
        for rule, (routes, _parked, lateness) in dispatcher.candidate_plans(env, 0).items():
            assert lateness >= 0
            on_time = all(max(r) <= env.agents[h].latest_arrival for h, r in routes.items())
            if on_time and len(routes) == len(env.agents):
                assert lateness == 0 == plan_lateness(env, routes, 0)
