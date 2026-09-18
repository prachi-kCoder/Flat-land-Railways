

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

WIDTH = 30
HEIGHT = 30
NUMBER_OF_TRAINS = 3
SEED = 42


# ------------------------------------------------------------
# 2. Railway Network Generator
# ------------------------------------------------------------

rail_generator = sparse_rail_generator(
    max_num_cities=3,
    grid_mode=False,
    max_rails_between_cities=2,
    max_rail_pairs_in_city=2,
    seed=SEED
)


# ------------------------------------------------------------
# 3. Train Start/Target Generator
# ------------------------------------------------------------

line_generator = sparse_line_generator()


# ------------------------------------------------------------
# 4. Create Flatland Environment
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
# 5. Generate Railway Scenario
# ------------------------------------------------------------

obs, info = env.reset(random_seed=SEED)


# ------------------------------------------------------------
# 6. Display Environment Information
# ------------------------------------------------------------

print("=" * 65)
print("        FLATLAND-RL INITIAL RAILWAY SIMULATION")
print("=" * 65)

print(f"Grid Size        : {WIDTH} x {HEIGHT}")
print(f"Number of Trains : {NUMBER_OF_TRAINS}")
print(f"Random Seed      : {SEED}")

print("\nEnvironment successfully created!")


# ------------------------------------------------------------
# 7. Display Train Information
# ------------------------------------------------------------

print("\nTRAIN INFORMATION")
print("-" * 65)

for agent in env.agents:

    # Compatible with Flatland versions where
    # initial_position / initial_direction are unavailable.
    start_position = getattr(
        agent,
        "initial_position",
        getattr(agent, "position", None)
    )

    start_direction = getattr(
        agent,
        "initial_direction",
        getattr(agent, "direction", None)
    )

    print(
        f"Train {agent.handle}: "
        f"Start={start_position}, "
        f"Target={agent.targets}, "
        f"Direction={start_direction}"
    )


# ------------------------------------------------------------
# 8. Observation Information
# ------------------------------------------------------------

print("\nOBSERVATION INFORMATION")
print("-" * 65)

print("Active observation agents:", list(obs.keys()))


# ------------------------------------------------------------
# 9. Render Railway Network
# ------------------------------------------------------------

print("\nRendering railway environment...")

renderer = RenderTool(env)

renderer.render_env(
    show=True,
    show_observations=False,
    show_predictions=False
)

print("\nRailway visualization displayed successfully.")
print("Initial Flatland-RL environment setup completed.")


# ------------------------------------------------------------
# 10. Keep Visualization Open
# ------------------------------------------------------------

input("\nPress ENTER to close the simulation...")

env.close()