# ============================================================
# Flatland-RL: Initial Railway Environment Setup
# ============================================================

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator


# ------------------------------------------------------------
# 1. Environment configuration
# ------------------------------------------------------------

WIDTH = 30
HEIGHT = 30
NUMBER_OF_TRAINS = 3
SEED = 42


# ------------------------------------------------------------
# 2. Railway network generator
# ------------------------------------------------------------

rail_generator = sparse_rail_generator(
    max_num_cities=3,
    grid_mode=False,
    max_rails_between_cities=2,
    max_rail_pairs_in_city=2,
    seed=SEED
)


# ------------------------------------------------------------
# 3. Train start/target generator
# ------------------------------------------------------------

line_generator = sparse_line_generator()


# ------------------------------------------------------------
# 4. Create Flatland environment
# ------------------------------------------------------------

env = RailEnv(
    width=WIDTH,
    height=HEIGHT,
    rail_generator=rail_generator,
    line_generator=line_generator,
    number_of_agents=NUMBER_OF_TRAINS,
    random_seed=SEED
)


# ------------------------------------------------------------
# 5. Generate the railway scenario
# ------------------------------------------------------------

obs, info = env.reset(random_seed=SEED)


# ------------------------------------------------------------
# 6. Display basic information
# ------------------------------------------------------------

print("=" * 60)
print("FLATLAND RAILWAY ENVIRONMENT")
print("=" * 60)

print(f"Grid size        : {WIDTH} x {HEIGHT}")
print(f"Number of trains : {NUMBER_OF_TRAINS}")
print(f"Random seed      : {SEED}")

print("\nEnvironment successfully created!")


# ------------------------------------------------------------
# 7. Display train information
# ------------------------------------------------------------

print("\nTRAIN INFORMATION")
print("-" * 60)

for agent in env.agents:

    # Flatland versions may expose initial position/direction
    # through the agent's current position/direction instead.
    start_position = getattr(
        agent, "initial_position",
        getattr(agent, "position", None)
    )

    start_direction = getattr(
        agent, "initial_direction",
        getattr(agent, "direction", None)
    )

    print(
        f"Train {agent.handle}: "
        f"start={start_position}, "
        f"target={agent.targets}, "
        f"direction={start_direction}"
    )


# ------------------------------------------------------------
# 8. Basic observation information
# ------------------------------------------------------------

print("\nOBSERVATION INFORMATION")
print("-" * 60)

print("Observation agents:", list(obs.keys()))

print("\nInitial Flatland setup completed successfully!")