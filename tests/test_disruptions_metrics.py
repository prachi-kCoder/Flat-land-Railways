"""
Checks for disruption injection, metrics and the episode runner.

Run:  python -m pytest tests/
"""

import pytest
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from railsim.disruptions import Disruption, DisruptionSpec
from railsim.env_factory import EnvConfig, make_env, usable_seeds
from railsim.metrics import network_delay, recovery_time
from railsim.policies import AlwaysForward, ShortestPathNoDispatch
from railsim.runner import run_episode


# ---------------------------------------------------------------- planning

def test_plan_is_reproducible():
    config = EnvConfig(num_trains=8, seed=3)
    spec = DisruptionSpec(count=3, duration=20)
    plan_a = make_env(config, spec)[0].disruption_plan
    plan_b = make_env(config, spec)[0].disruption_plan
    assert plan_a == plan_b
    assert len(plan_a) == 3


def test_disruptions_do_not_change_the_scenario():
    """Adding disruptions must not change the network or timetable."""
    config = EnvConfig(num_trains=8, seed=3)
    plain = make_env(config)[0]
    disrupted = make_env(config, DisruptionSpec(count=3, duration=20))[0]
    assert (plain.rail.grid == disrupted.rail.grid).all()
    assert [(a.initial_configuration, a.earliest_departure, a.latest_arrival) for a in plain.agents] == \
           [(a.initial_configuration, a.earliest_departure, a.latest_arrival) for a in disrupted.agents]


@pytest.mark.parametrize("seed", range(10))
def test_spread_plan_uses_distinct_trains_inside_their_windows(seed):
    env = make_env(EnvConfig(num_trains=8, seed=seed), DisruptionSpec(count=4, duration=10))[0]
    trains = [d.train for d in env.disruption_plan]
    assert len(set(trains)) == 4
    for d in env.disruption_plan:
        agent = env.agents[d.train]
        assert agent.earliest_departure < d.start_step
        assert d.start_step < max(agent.earliest_departure + 2, agent.latest_arrival)


@pytest.mark.parametrize("seed", range(10))
def test_simultaneous_plan_starts_together(seed):
    env = make_env(EnvConfig(num_trains=8, seed=seed), DisruptionSpec(count=3, duration=10, simultaneous=True))[0]
    plan = env.disruption_plan
    assert len({d.start_step for d in plan}) == 1
    assert len({d.train for d in plan}) == 3
    for d in plan:
        agent = env.agents[d.train]
        assert agent.earliest_departure < d.start_step < max(agent.earliest_departure + 2, agent.latest_arrival)


def test_too_many_disruptions_is_an_error():
    with pytest.raises(ValueError):
        make_env(EnvConfig(num_trains=3, seed=1), DisruptionSpec(count=4))


# ---------------------------------------------------------------- injection

@pytest.mark.parametrize("seed", range(8))
def test_breakdown_lasts_exactly_duration(seed):
    """A train broken on the track stays in MALFUNCTION for exactly `duration` steps."""
    spec = DisruptionSpec(count=3, duration=7)
    env = make_env(EnvConfig(num_trains=6, seed=seed), spec)[0]
    states = {d: [] for d in env.disruption_plan}

    done = False
    while not done:
        actions = {h: RailEnvActions.MOVE_FORWARD for h in env.get_agent_handles()}
        _, _, dones, _ = env.step(actions)
        for d in env.disruption_plan:
            states[d].append(env.agents[d.train].state)
        done = dones["__all__"]

    for d, history in states.items():
        if env.disruption_status.applied.get(d) != "on_map":
            continue
        broken = history[d.start_step - 1: d.start_step - 1 + d.duration]
        assert all(s == TrainState.MALFUNCTION for s in broken), (d, broken)
        if len(history) > d.end_step:
            assert history[d.end_step] != TrainState.MALFUNCTION


# ---------------------------------------------------------------- metrics

@pytest.mark.parametrize("seed", usable_seeds(EnvConfig(num_trains=5), count=10))
@pytest.mark.parametrize("policy_class", [AlwaysForward, ShortestPathNoDispatch])
def test_delay_matches_flatland_reward(seed, policy_class):
    """
    Total delay must equal minus Flatland's own episode reward. This holds
    when every train departed: Flatland scores a train that never departed
    with a separate cancellation penalty. No intermediate stops or collision
    penalties are used in this setup.
    """
    config = EnvConfig(num_trains=5, seed=seed)
    result = run_episode(config, policy_class)
    assert result.delay_curve[-1] == result.total_delay

    env = make_env(config)[0]
    policy = policy_class()
    policy.reset(env)
    done = False
    while not done:
        _, _, dones, _ = env.step(policy.act(env))
        done = dones["__all__"]
    never_departed = [a for a in env.agents if a.state.is_off_map_state()]

    # Comparison only holds when no train was cancelled; in these
    # scenarios every train departs.
    assert not never_departed
    assert result.total_delay == pytest.approx(-result.flatland_reward)


def test_network_delay_is_zero_at_start():
    env = make_env(EnvConfig(num_trains=5, seed=4))[0]
    assert network_delay(env) == 0


def test_recovery_time():
    plan = [Disruption(train=0, start_step=3, duration=2)]  # broken on steps 3-4
    reference = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    # No extra delay at all.
    assert recovery_time([0, 0, 0, 0, 1, 1, 1, 1, 1], reference, plan) == 0
    # Extra delay grows until step 6, then stays at +3: recovered at step 6.
    assert recovery_time([0, 0, 1, 2, 3, 4, 4, 4, 4], reference, plan) == 3
    # Extra delay still changing on the last step: not recovered.
    assert recovery_time([0, 0, 1, 2, 3, 4, 5, 6, 7], reference, plan) is None
    # Extra delay stops changing on step 4, but the train is still broken
    # then; recovery cannot be earlier than step 5 (the step after the end).
    assert recovery_time([0, 0, 1, 1, 2, 2, 2, 2, 2], reference, plan) == 2
    # Curves of different lengths: the shorter is extended with its last value.
    assert recovery_time([0, 0, 1, 2, 3, 3], reference, plan) == 2


# ---------------------------------------------------------------- seeds

def test_usable_seeds_skip_fallback_layouts():
    seeds = usable_seeds(EnvConfig(num_trains=5), count=30)
    assert 10 not in seeds and 17 not in seeds
    assert len(seeds) == 30
    for seed in seeds[:5]:
        assert not make_env(EnvConfig(num_trains=5, seed=seed))[0].layout_changed


def test_runner_same_plan_for_every_policy():
    config = EnvConfig(num_trains=6, seed=5)
    spec = DisruptionSpec(count=2, duration=10)
    a = run_episode(config, AlwaysForward, spec)
    b = run_episode(config, ShortestPathNoDispatch, spec)
    assert a.disruption_count == b.disruption_count == 2
    assert make_env(config, spec)[0].disruption_plan == make_env(config, spec)[0].disruption_plan


@pytest.mark.parametrize("seed", usable_seeds(EnvConfig(num_trains=6), count=4))
def test_extra_delay_equals_difference_with_undisrupted_run(seed):
    """extra_delay = total delay with breakdowns - total delay of the same
    policy on the same scenario without breakdowns (run independently here)."""
    config = EnvConfig(num_trains=6, seed=seed)
    disrupted = run_episode(config, ShortestPathNoDispatch, DisruptionSpec(count=2, duration=15))
    undisrupted = run_episode(config, ShortestPathNoDispatch)
    assert disrupted.reference_delay == undisrupted.total_delay
    assert disrupted.extra_delay == disrupted.total_delay - undisrupted.total_delay
    assert undisrupted.extra_delay is None and undisrupted.reference_delay is None
