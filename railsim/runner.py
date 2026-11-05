"""
Run one policy on one scenario and measure it.
"""

import time

from flatland.envs.step_utils.states import TrainState

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env
from railsim.metrics import EpisodeResult, network_delay, recovery_time, train_delay
from railsim.train_status import find_deadlocked_trains


def _run(config, policy, disruptions):
    """Play one episode. Returns (env, delay_curve, stats dict)."""
    env, _, _ = make_env(config, disruptions)
    policy.reset(env)

    delay_curve = []
    decision_times = []
    total_reward = 0.0
    first_deadlock_step = None

    done = False
    while not done:
        started = time.perf_counter()
        actions = policy.act(env)
        decision_times.append(time.perf_counter() - started)

        _, rewards, dones, _ = env.step(actions)
        total_reward += sum(rewards.values())
        delay_curve.append(network_delay(env))

        if first_deadlock_step is None and find_deadlocked_trains(env):
            first_deadlock_step = env._elapsed_steps
        done = dones["__all__"]

    stats = {
        "replans": getattr(policy, "replans", 0),
        "decision_times": decision_times,
        "total_reward": total_reward,
        "first_deadlock_step": first_deadlock_step,
    }
    return env, delay_curve, stats


def run_episode(config: EnvConfig, policy_class, disruptions: DisruptionSpec = None):
    """
    Run `policy_class` on the scenario defined by `config` and `disruptions`.

    If there are disruptions, the same scenario is also run without them
    (same policy) to measure recovery time against an undisrupted baseline.
    """
    env, curve, stats = _run(config, policy_class(), disruptions)

    reference_curve = None
    if disruptions is not None and env.disruption_plan:
        _, reference_curve, _ = _run(config, policy_class(), None)

    delays = [train_delay(agent, env) for agent in env.agents]
    arrived = [a for a in env.agents if a.state == TrainState.DONE]
    applied = list(env.disruption_status.applied.values())
    spec = disruptions or DisruptionSpec(count=0, duration=0)
    times_ms = [t * 1000 for t in stats["decision_times"]]

    return EpisodeResult(
        policy=policy_class.name,
        seed=config.seed,
        num_trains=config.num_trains,
        disruption_count=len(env.disruption_plan),
        disruption_duration=spec.duration if env.disruption_plan else 0,
        simultaneous=spec.simultaneous if env.disruption_plan else False,
        steps=env._elapsed_steps,
        arrived=len(arrived),
        arrived_on_time=sum(a.arrival_time <= a.latest_arrival for a in arrived),
        deadlocked=len(find_deadlocked_trains(env)),
        first_deadlock_step=stats["first_deadlock_step"],
        total_delay=sum(delays),
        mean_delay=round(sum(delays) / len(delays), 2),
        max_delay=max(delays),
        flatland_reward=round(stats["total_reward"], 2),
        recovery_time=(
            recovery_time(curve, reference_curve, env.disruption_plan)
            if reference_curve is not None else None
        ),
        disruptions_on_map=applied.count("on_map"),
        disruptions_off_map=applied.count("off_map"),
        disruptions_skipped=applied.count("skipped"),
        decision_ms_mean=round(sum(times_ms) / len(times_ms), 3),
        decision_ms_max=round(max(times_ms), 3),
        replans=stats["replans"],
        delay_curve=curve,
    )
