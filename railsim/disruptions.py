"""
Controlled disruption injection.

A disruption is a train breakdown: the train stops where it is for a fixed
number of steps and blocks its cell (and so the track) for everyone behind
it. If the train has not departed yet, its departure is delayed instead.

Disruptions are planned in advance from the timetable only, never from what
happens during the run. That way every dispatching method (rule-based or RL)
faces exactly the same disruptions for a given scenario seed.

Usage:
    spec = DisruptionSpec(count=2, duration=20)
    env, obs, info = make_env(config, spec)
    env.disruption_plan   # list of Disruption that will be applied
"""

from dataclasses import dataclass

import numpy as np
from flatland.core.effects_generator import EffectsGenerator
from flatland.envs.step_utils.states import TrainState


@dataclass(frozen=True)
class Disruption:
    """Train `train` is broken from step `start_step` for `duration` steps."""

    train: int
    start_step: int
    duration: int

    @property
    def end_step(self):
        """Last step on which the train is still broken."""
        return self.start_step + self.duration - 1


@dataclass(frozen=True)
class DisruptionSpec:
    """
    What kind of disruptions a scenario should contain.

    count        : number of trains that break down (each at most once)
    duration     : steps each breakdown lasts
    simultaneous : if True, all breakdowns start on the same step;
                   otherwise each starts at its own random time
    """

    count: int = 1
    duration: int = 10
    simultaneous: bool = False


def plan_disruptions(env, spec, seed):
    """
    Choose which trains break down and when, using only the timetable.

    Each breakdown starts while the chosen train is scheduled to be running:
    after its earliest departure and before its latest arrival. A breakdown
    timed before departure would end before the train leaves and have no
    effect.

    Uses its own random generator (seeded by `seed`), so planning never
    changes the environment's own random numbers.
    """
    if spec.count == 0:
        return []
    if spec.count > len(env.agents):
        raise ValueError(f"cannot disrupt {spec.count} trains: scenario has only {len(env.agents)}")
    if spec.duration < 1:
        raise ValueError("disruption duration must be at least 1 step")

    rng = np.random.default_rng(seed)
    windows = {
        agent.handle: (agent.earliest_departure + 1, max(agent.earliest_departure + 2, agent.latest_arrival))
        for agent in env.agents
    }  # start step range [low, high) for each train

    if not spec.simultaneous:
        trains = rng.choice(len(env.agents), size=spec.count, replace=False)
        return sorted(
            (Disruption(int(t), int(rng.integers(*windows[int(t)])), spec.duration) for t in trains),
            key=lambda d: (d.start_step, d.train),
        )

    # Simultaneous: pick a step where at least `count` trains are scheduled
    # to be running, then pick that many of those trains.
    first = min(low for low, _ in windows.values())
    last = max(high for _, high in windows.values())
    candidate_steps = [
        step for step in range(first, last)
        if sum(low <= step < high for low, high in windows.values()) >= spec.count
    ]
    if not candidate_steps:
        raise ValueError(f"no step has {spec.count} trains scheduled to run at the same time")

    step = int(rng.choice(candidate_steps))
    running = [handle for handle, (low, high) in windows.items() if low <= step < high]
    trains = rng.choice(running, size=spec.count, replace=False)
    return sorted((Disruption(int(t), step, spec.duration) for t in trains), key=lambda d: d.train)


class ScheduledDisruptions(EffectsGenerator):
    """
    Flatland hook that applies a fixed disruption plan during the run.

    Flatland calls `on_episode_step_start` at the start of each step, after
    increasing `env._elapsed_steps` to the new step number. On a
    disruption's start step, the train's malfunction counter is
    set to the disruption's duration; Flatland then keeps the train in the
    MALFUNCTION state for exactly that many steps.

    `applied` records what actually happened to each planned disruption:
    "on_map"  - the train broke down on the track,
    "off_map" - the train had not departed yet; its departure is delayed,
    "skipped" - the train had already arrived, so nothing happened.
    """

    def __init__(self):
        super().__init__()
        self.plan = []
        self.applied = {}

    def on_episode_start(self, env, *args, **kwargs):
        self.applied = {}
        return env

    def on_episode_step_start(self, env, *args, **kwargs):
        for disruption in self.plan:
            if disruption.start_step != env._elapsed_steps:
                continue
            agent = env.agents[disruption.train]
            if agent.state == TrainState.DONE:
                self.applied[disruption] = "skipped"
                continue
            self.applied[disruption] = "on_map" if agent.state.is_on_map_state() else "off_map"
            agent.malfunction_handler.malfunction_down_counter = disruption.duration
        return env
