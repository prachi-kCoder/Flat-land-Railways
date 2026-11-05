# ============================================================
# Flatland-RL: Initial Study of Train Movement
#               and Resource Conflicts
# ============================================================
#
# Every train is always told MOVE_FORWARD (no dispatcher logic).
# The script records, step by step:
#   - departures and arrivals,
#   - trains that wanted to move but were blocked by another train,
#   - trains that became permanently deadlocked.
#
# Two trains can never occupy the same cell in Flatland, so
# conflicts show up as blocking and deadlocks, not as overlaps.
#
# Run:  python train-movement.py
# ============================================================

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from railsim import EnvConfig, find_deadlocked_trains, make_env, train_position


CONFIG = EnvConfig(width=30, height=30, num_trains=5, seed=42)

env, obs, info = make_env(CONFIG)
max_steps = env._max_episode_steps


print("=" * 70)
print("     INITIAL STUDY OF TRAIN MOVEMENT AND RESOURCE CONFLICTS")
print("=" * 70)
print(f"Grid Size        : {CONFIG.width} x {CONFIG.height}")
print(f"Number of Trains : {CONFIG.num_trains}")
print(f"Episode Length   : {max_steps} steps (set by Flatland)")
print(f"Random Seed      : {CONFIG.seed}")

print("\nTIMETABLE")
print("-" * 70)
for agent in env.agents:
    print(
        f"Train {agent.handle}: departs at step {agent.earliest_departure}, "
        f"due by step {agent.latest_arrival}"
    )


print("\n" + "=" * 70)
print("MULTI-TRAIN MOVEMENT OBSERVATION  (policy: always MOVE_FORWARD)")
print("=" * 70)

movement_count = 0
blocked_count = 0              # train-steps spent blocked by another train
departed_trains = set()
arrived_trains = {}            # handle -> arrival step
deadlocked_trains = {}         # handle -> step the deadlock was detected

previous_positions = {agent.handle: train_position(agent) for agent in env.agents}


for step in range(1, max_steps + 1):

    actions = {handle: RailEnvActions.MOVE_FORWARD for handle in env.get_agent_handles()}
    obs, rewards, dones, info = env.step(actions)

    events = []
    occupied_cells = {
        train_position(agent): agent.handle
        for agent in env.agents
        if train_position(agent) is not None
    }

    for agent in env.agents:
        handle = agent.handle
        position = train_position(agent)
        previous = previous_positions[handle]

        if previous is None and position is not None and handle not in departed_trains:
            departed_trains.add(handle)
            events.append(f"Train {handle} departed from {position}")

        if position is not None and previous is not None and position != previous:
            movement_count += 1

        if agent.state == TrainState.DONE and handle not in arrived_trains:
            arrived_trains[handle] = step
            late = step - agent.latest_arrival
            timing = f"{late} steps late" if late > 0 else "on time"
            events.append(f"Train {handle} reached its target ({timing})")

        # Blocked: on the map, not broken down, but did not move this step.
        if (
            position is not None
            and position == previous
            and agent.state != TrainState.MALFUNCTION
            and handle not in deadlocked_trains
        ):
            blocked_count += 1

        previous_positions[handle] = position

    for handle in sorted(find_deadlocked_trains(env)):
        if handle not in deadlocked_trains:
            deadlocked_trains[handle] = step
            events.append(
                f"DEADLOCK: Train {handle} can no longer move "
                f"(at {train_position(env.agents[handle])})"
            )

    if events:
        print(f"\nStep {step}:")
        for event in events:
            print(f"  {event}")

    if dones["__all__"]:
        print(f"\nEpisode ended by Flatland at step {step}.")
        break


never_departed = [a.handle for a in env.agents if a.handle not in departed_trains]

print("\n" + "=" * 70)
print("RESOURCE-CONFLICT STUDY SUMMARY")
print("=" * 70)
print(f"Simulation steps executed   : {step}")
print(f"Cell-to-cell movements      : {movement_count}")
print(f"Train-steps spent blocked   : {blocked_count}")
print(f"Trains departed             : {len(departed_trains)} / {CONFIG.num_trains}")
print(f"Trains reached target       : {len(arrived_trains)} / {CONFIG.num_trains}")
print(f"Trains deadlocked           : {len(deadlocked_trains)} / {CONFIG.num_trains}")
if deadlocked_trains:
    print(f"First deadlock at step      : {min(deadlocked_trains.values())}")
if never_departed:
    print(f"Trains that never departed  : {never_departed}")

print("\nOBSERVATIONS")
print("-" * 70)
print("1. Two trains can never share a cell; conflicts appear as")
print("   blocking (waiting for a cell) and deadlocks (permanent).")
print("2. With no dispatcher logic, trains facing each other on a")
print("   single track deadlock and every train behind them is stuck.")
print("3. This motivates coordinated decisions: hold, proceed,")
print("   reroute and reorder.")
