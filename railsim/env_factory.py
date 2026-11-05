"""
Single place where the Flatland environment is built.

Every script should create its environment through `make_env` so that
all experiments use exactly the same network, trains, timetable and
disruptions for a given configuration and seed.
"""

import warnings
from dataclasses import dataclass

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from railsim.disruptions import ScheduledDisruptions, plan_disruptions


@dataclass
class EnvConfig:
    """All settings needed to rebuild the same railway scenario."""

    width: int = 30
    height: int = 30
    num_trains: int = 3
    seed: int = 42

    # Railway network layout
    max_cities: int = 3
    grid_mode: bool = False
    max_rails_between_cities: int = 2
    max_rail_pairs_in_city: int = 2


# Flatland warnings that mean the requested city layout could not be built
# and a different network was generated instead.
_LAYOUT_WARNINGS = ("could not set all required cities", "changing to grid mode")


def make_env(config: EnvConfig, disruptions=None):
    """
    Build and reset a Flatland RailEnv from `config`.

    disruptions : optional DisruptionSpec. The plan is drawn from the
                  timetable using `config.seed`, so the same config always
                  gives the same disruptions.

    Returns (env, obs, info) after the first reset. Extra attributes:
      env.disruption_plan    list of Disruption (empty if none)
      env.disruption_status  ScheduledDisruptions hook; its `.applied`
                             says what happened to each disruption
      env.layout_changed     True if Flatland could not build the requested
                             city layout for this seed and fell back to a
                             different one

    The seed is passed only to the environment, not to the rail generator:
    Flatland 4.x draws all randomness from the environment's seed and warns
    if the generator is given its own.
    """
    rail_generator = sparse_rail_generator(
        max_num_cities=config.max_cities,
        grid_mode=config.grid_mode,
        max_rails_between_cities=config.max_rails_between_cities,
        max_rail_pairs_in_city=config.max_rail_pairs_in_city,
    )

    disruption_hook = ScheduledDisruptions()

    env = RailEnv(
        width=config.width,
        height=config.height,
        rail_generator=rail_generator,
        line_generator=sparse_line_generator(),
        number_of_agents=config.num_trains,
        random_seed=config.seed,
        effects_generator=disruption_hook,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obs, info = env.reset(random_seed=config.seed)

    env.layout_changed = any(
        pattern in str(w.message).lower() for w in caught for pattern in _LAYOUT_WARNINGS
    )
    for w in caught:
        if not any(pattern in str(w.message).lower() for pattern in _LAYOUT_WARNINGS):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    if disruptions is not None:
        # Separate stream from the env's own randomness: [seed, 1].
        disruption_hook.plan = plan_disruptions(env, disruptions, seed=[config.seed, 1])

    env.disruption_plan = disruption_hook.plan
    env.disruption_status = disruption_hook
    return env, obs, info


def usable_seeds(config: EnvConfig, count: int, start: int = 0):
    """
    The first `count` seeds from `start` upwards that build the requested
    city layout. Seeds where Flatland falls back to a different layout are
    skipped so all scenarios in an experiment share the same kind of network.
    """
    seeds = []
    seed = start
    while len(seeds) < count:
        probe = EnvConfig(**{**config.__dict__, "seed": seed})
        env, _, _ = make_env(probe)
        if not env.layout_changed:
            seeds.append(seed)
        seed += 1
    return seeds
