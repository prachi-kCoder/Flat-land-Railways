# ============================================================
# Flatland-RL: Initial Railway Environment Setup
# ============================================================
#
# Builds the base 30x30 railway scenario and prints each
# train's start, target and timetable.
#
# Run:  python initial-rail.py
# ============================================================

from railsim import EnvConfig, direction_name, make_env, train_target


CONFIG = EnvConfig(width=30, height=30, num_trains=3, seed=42)

env, obs, info = make_env(CONFIG)


print("=" * 60)
print("FLATLAND RAILWAY ENVIRONMENT")
print("=" * 60)
print(f"Grid size        : {CONFIG.width} x {CONFIG.height}")
print(f"Number of trains : {CONFIG.num_trains}")
print(f"Random seed      : {CONFIG.seed}")
print(f"Episode length   : {env._max_episode_steps} steps")
print("\nEnvironment successfully created!")


print("\nTRAIN INFORMATION")
print("-" * 60)

for agent in env.agents:
    start_cell, start_direction = agent.initial_configuration

    print(
        f"Train {agent.handle}: "
        f"start={start_cell} facing {direction_name(start_direction)}, "
        f"target={train_target(agent)}, "
        f"departs at step {agent.earliest_departure}, "
        f"due by step {agent.latest_arrival}"
    )


print("\nOBSERVATION INFORMATION")
print("-" * 60)
print("Observation agents:", list(obs.keys()))

print("\nInitial Flatland setup completed successfully!")
