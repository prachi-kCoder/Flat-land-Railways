"""
Checks for the RL dispatcher, its training environment and the
departure-control baselines.

Run:  python -m pytest tests/
"""

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env, usable_seeds
from railsim.metrics import network_delay
from railsim.rl_dispatch import (
    DISPATCH, DispatchEnv, HoldIfBlocked, ReleaseAll, hold_if_blocked_action,
)
from railsim.runner import run_episode
from railsim.train_status import find_deadlocked_trains

SPEC = DisruptionSpec(count=1, duration=20)
SEEDS = usable_seeds(EnvConfig(num_trains=8), count=6, start=1000)


def _play_env(env, scenario, choose):
    """Run one training-env episode with `choose(obs) -> action`."""
    obs, _ = env.reset(options={"scenario": scenario})
    total, done = 0.0, False
    while not done:
        assert env.observation_space.contains(obs)
        obs, reward, done, _, info = env.step(choose(obs))
        total += reward
    return total, info


def test_env_passes_sb3_checker():
    check_env(DispatchEnv(SEEDS), warn=True)


@pytest.mark.parametrize("scenario", SEEDS)
def test_rewards_sum_to_minus_delay(scenario):
    env = DispatchEnv(SEEDS)
    rng = np.random.default_rng(scenario)
    total, info = _play_env(env, scenario, lambda obs: int(rng.integers(2)))
    assert info["final_delay"] == network_delay(env.env)
    assert total == pytest.approx(-(info["final_delay"] - env.delay_at_first_decision) / 8)


@pytest.mark.parametrize("scenario", SEEDS)
@pytest.mark.parametrize("policy_class, choose", [
    (ReleaseAll, lambda obs: DISPATCH),
    (HoldIfBlocked, hold_if_blocked_action),
])
def test_training_env_matches_deployed_policy(scenario, policy_class, choose):
    """The same decisions give the same result in the training environment
    and when the policy is run by run_experiment.py."""
    env = DispatchEnv(SEEDS, num_trains=8, disruptions=SPEC)
    _, info = _play_env(env, scenario, choose)
    trained = (info["final_delay"], sum(a.state.name == "DONE" for a in env.env.agents))

    result = run_episode(EnvConfig(num_trains=8, seed=scenario), policy_class, SPEC)
    assert trained == (result.total_delay, result.arrived)


@pytest.mark.parametrize("scenario", SEEDS[:3])
@pytest.mark.parametrize("policy_class", [ReleaseAll, HoldIfBlocked])
def test_departure_control_never_deadlocks(scenario, policy_class):
    env = make_env(EnvConfig(num_trains=8, seed=scenario), DisruptionSpec(count=3, duration=30))[0]
    policy = policy_class()
    policy.reset(env)
    done = False
    while not done:
        _, _, dones, _ = env.step(policy.act(env))
        assert not find_deadlocked_trains(env)
        done = dones["__all__"]


def test_hold_if_blocked_actually_waits():
    """Guard against the rule silently behaving like release_all."""
    env = DispatchEnv(SEEDS)
    waits = 0
    for scenario in SEEDS:
        obs, _ = env.reset(options={"scenario": scenario})
        done = False
        while not done:
            action = hold_if_blocked_action(obs)
            waits += action == 0
            obs, _, done, _, _ = env.step(action)
    assert waits > 0


def test_training_validation_and_test_seeds_do_not_overlap():
    import train_rl
    train = usable_seeds(EnvConfig(num_trains=8), 300, start=train_rl.TRAIN_SEED_START)
    val = usable_seeds(EnvConfig(num_trains=8), 30, start=train_rl.VALIDATION_SEED_START)
    test = usable_seeds(EnvConfig(num_trains=8), 100, start=0)
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
