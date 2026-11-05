"""
Single place where the Flatland environment is built.

Every script should create its environment through `make_env` so that
all experiments use exactly the same network, trains and timetable for a
given configuration and seed.
"""

from dataclasses import dataclass

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator


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


def make_env(config: EnvConfig):
    """
    Build and reset a Flatland RailEnv from `config`.

    Returns (env, obs, info) after the first reset.

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

    env = RailEnv(
        width=config.width,
        height=config.height,
        rail_generator=rail_generator,
        line_generator=sparse_line_generator(),
        number_of_agents=config.num_trains,
        random_seed=config.seed,
    )

    obs, info = env.reset(random_seed=config.seed)
    return env, obs, info
