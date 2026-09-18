# ============================================================
# Flatland-RL: Initial Study of Train Movement
#               and Resource Conflicts
# ============================================================

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

WIDTH = 30
HEIGHT = 30
NUMBER_OF_TRAINS = 5
SEED = 42
MAX_STEPS = 100


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
# 3. Train Generator
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
# 5. Reset Environment
# ------------------------------------------------------------

obs, info = env.reset(random_seed=SEED)


# ------------------------------------------------------------
# 6. Display Initial Configuration
# ------------------------------------------------------------

print("=" * 70)
print("     INITIAL STUDY OF TRAIN MOVEMENT AND RESOURCE CONFLICTS")
print("=" * 70)

print(f"Grid Size        : {WIDTH} x {HEIGHT}")
print(f"Number of Trains : {NUMBER_OF_TRAINS}")
print(f"Maximum Steps    : {MAX_STEPS}")
print(f"Random Seed      : {SEED}")


print("\nINITIAL TRAIN CONFIGURATION")
print("-" * 70)

for agent in env.agents:

    start_position = getattr(
        agent,
        "initial_position",
        getattr(agent, "position", None)
    )

    print(
        f"Train {agent.handle}: "
        f"Start={start_position}, "
        f"Target={agent.targets}"
    )


# ------------------------------------------------------------
# 7. Simulation Variables
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("MULTI-TRAIN MOVEMENT OBSERVATION")
print("=" * 70)

movement_count = 0
conflict_count = 0
completed_trains = set()

previous_positions = {
    agent.handle: getattr(agent, "position", None)
    for agent in env.agents
}


# ------------------------------------------------------------
# 8. Run Multi-Train Simulation
# ------------------------------------------------------------

for step in range(1, MAX_STEPS + 1):

    # --------------------------------------------------------
    # Select actions
    # --------------------------------------------------------
    # Flatland actions:
    # 0 = DO_NOTHING
    # 1 = MOVE_LEFT
    # 2 = MOVE_FORWARD
    # 3 = MOVE_RIGHT
    # 4 = STOP_MOVING

    actions = {}

    for agent in env.agents:
        actions[agent.handle] = 2


    # --------------------------------------------------------
    # Execute simulation step
    # --------------------------------------------------------

    obs, rewards, dones, info = env.step(actions)


    # --------------------------------------------------------
    # IMPORTANT:
    # Check whether Flatland has ended the episode.
    # This prevents:
    # "Episode is done, cannot call step()"
    # --------------------------------------------------------

    episode_done = dones.get("__all__", False)


    # --------------------------------------------------------
    # Get current train positions
    # --------------------------------------------------------

    current_positions = {}

    for agent in env.agents:

        current_positions[agent.handle] = getattr(
            agent,
            "position",
            None
        )


    # --------------------------------------------------------
    # Detect train movement
    # --------------------------------------------------------

    moved_this_step = []

    for handle, current_position in current_positions.items():

        previous_position = previous_positions.get(handle)

        if (
            current_position is not None
            and current_position != previous_position
        ):

            moved_this_step.append(handle)
            movement_count += 1


    # --------------------------------------------------------
    # Detect shared railway resources
    # --------------------------------------------------------

    position_to_trains = {}

    for handle, position in current_positions.items():

        if position is not None:

            if position not in position_to_trains:
                position_to_trains[position] = []

            position_to_trains[position].append(handle)


    # --------------------------------------------------------
    # Identify potential conflicts
    # --------------------------------------------------------

    step_conflicts = []

    for position, trains in position_to_trains.items():

        if len(trains) > 1:

            step_conflicts.append(
                (position, trains)
            )


    # --------------------------------------------------------
    # Display movement
    # --------------------------------------------------------

    if moved_this_step:

        print(
            f"\nStep {step}: "
            f"Moving trains = {moved_this_step}"
        )


    # --------------------------------------------------------
    # Display resource conflicts
    # --------------------------------------------------------

    if step_conflicts:

        conflict_count += len(step_conflicts)

        print(
            f"  RESOURCE CONFLICTS DETECTED: "
            f"{len(step_conflicts)}"
        )

        for position, trains in step_conflicts:

            print(
                f"    Shared cell {position} "
                f"occupied by trains {trains}"
            )


    # --------------------------------------------------------
    # Detect trains reaching their targets
    # --------------------------------------------------------

    for agent in env.agents:

        current_position = getattr(
            agent,
            "position",
            None
        )

        target_position = getattr(
            agent,
            "target",
            None
        )

        if (
            current_position is not None
            and target_position is not None
            and current_position == target_position
            and agent.handle not in completed_trains
        ):

            completed_trains.add(agent.handle)

            print(
                f"  Train {agent.handle} "
                f"reached its target."
            )


    # --------------------------------------------------------
    # Update previous positions
    # --------------------------------------------------------

    previous_positions = current_positions.copy()


    # --------------------------------------------------------
    # STOP CONDITION
    # --------------------------------------------------------
    # Flatland tells us the episode is finished through
    # dones["__all__"]. We must break BEFORE calling
    # env.step() again.

    if episode_done:

        print(
            f"\nEpisode completed by Flatland "
            f"at simulation step {step}."
        )

        break


# ------------------------------------------------------------
# 9. Study Summary
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("INITIAL RESOURCE-CONFLICT STUDY SUMMARY")
print("=" * 70)

print(f"Simulation steps executed : {step}")
print(f"Train movements observed  : {movement_count}")
print(f"Resource conflicts found  : {conflict_count}")
print(f"Completed trains           : {len(completed_trains)}")


# ------------------------------------------------------------
# 10. Research Observations
# ------------------------------------------------------------

print("\nOBSERVATIONS")
print("-" * 70)

print("1. Multiple trains were simulated simultaneously.")
print("2. Train positions were monitored at every time step.")
print("3. Railway cells were treated as shared resources.")
print("4. Potential shared-resource conflicts were monitored.")
print("5. Train movement and target completion were recorded.")
print("6. These interactions motivate coordinated train")
print("   movement and dynamic rescheduling.")


# ------------------------------------------------------------
# 11. Close Environment
# ------------------------------------------------------------

# env.close()

print("\nInitial train movement and resource-conflict study completed.")