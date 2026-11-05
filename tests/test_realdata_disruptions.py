"""
Checks for breakdown lengths and counts drawn from real data
(DisruptionSpec.durations and DisruptionSpec.rate_per_train).

Run:  python -m pytest tests/
"""

import pytest

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env
from railsim.runner import run_episode
from railsim.policies import FCFSDispatcher

# Plans produced by the planner BEFORE these options were added
# (seed, (count, duration, simultaneous), [[train, start_step, duration], ...]).
BEFORE = [
    [0, [1, 20, False], [[4, 53, 20]]],
    [0, [3, 15, False], [[3, 21, 15], [7, 42, 15], [6, 71, 15]]],
    [0, [3, 30, True], [[4, 43, 30], [5, 43, 30], [6, 43, 30]]],
    [3, [1, 20, False], [[7, 9, 20]]],
    [3, [3, 15, False], [[4, 15, 15], [5, 19, 15], [1, 33, 15]]],
    [3, [3, 30, True], [[0, 61, 30], [2, 61, 30], [4, 61, 30]]],
    [7, [1, 20, False], [[6, 48, 20]]],
    [7, [3, 15, False], [[7, 18, 15], [5, 22, 15], [6, 26, 15]]],
    [7, [3, 30, True], [[0, 58, 30], [3, 58, 30], [5, 58, 30]]],
]
SAMPLE = (1, 1, 2, 3, 5, 8, 20, 45)


def _plan(seed, spec, trains=8):
    return make_env(EnvConfig(num_trains=trains, seed=seed), spec)[0].disruption_plan


@pytest.mark.parametrize("seed, spec, expected", BEFORE)
def test_existing_plans_are_unchanged(seed, spec, expected):
    plan = _plan(seed, DisruptionSpec(*spec))
    assert [[d.train, d.start_step, d.duration] for d in plan] == expected


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_empirical_lengths_keep_trains_and_times(seed):
    """Drawing lengths from a sample changes only the lengths."""
    fixed = _plan(seed, DisruptionSpec(count=3, duration=15))
    sampled = _plan(seed, DisruptionSpec(count=3, duration=15, durations=SAMPLE))
    assert [(d.train, d.start_step) for d in fixed] == [(d.train, d.start_step) for d in sampled]
    assert all(d.duration in SAMPLE for d in sampled)


def test_empirical_lengths_are_reproducible_and_vary():
    spec = DisruptionSpec(count=3, durations=SAMPLE)
    assert _plan(5, spec) == _plan(5, spec)
    lengths = {d.duration for seed in range(30) for d in _plan(seed, spec)}
    assert len(lengths) > 1


def test_rate_per_train_gives_reproducible_counts_in_range():
    spec = DisruptionSpec(durations=SAMPLE, rate_per_train=0.3)
    counts = [len(_plan(seed, spec)) for seed in range(30)]
    assert counts == [len(_plan(seed, spec)) for seed in range(30)]
    assert all(0 <= c <= 8 for c in counts) and len(set(counts)) > 1
    assert all(len(_plan(seed, DisruptionSpec(rate_per_train=0.0))) == 0 for seed in range(5))


def test_bad_options_are_rejected():
    with pytest.raises(ValueError):
        _plan(1, DisruptionSpec(durations=()))
    with pytest.raises(ValueError):
        _plan(1, DisruptionSpec(durations=(0, 3)))
    with pytest.raises(ValueError):
        _plan(1, DisruptionSpec(rate_per_train=1.5))


def test_runner_reports_mean_planned_length():
    spec = DisruptionSpec(count=3, durations=SAMPLE)
    config = EnvConfig(num_trains=8, seed=3)
    result = run_episode(config, FCFSDispatcher, spec)
    plan = make_env(config, spec)[0].disruption_plan
    assert result.disruption_duration == round(sum(d.duration for d in plan) / len(plan))
    fixed = run_episode(config, FCFSDispatcher, DisruptionSpec(count=1, duration=20))
    assert fixed.disruption_duration == 20
