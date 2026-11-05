"""
Read train positions and detect deadlocks.

Flatland 4.x stores a train's location as a "configuration":
    ((row, col), direction)
and it is None while the train is off the map (before departure or
after arrival). These helpers hide that detail from the rest of the code.
"""

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState


def train_position(agent):
    """Current (row, col) of the train, or None if it is off the map."""
    configuration = agent.current_configuration
    return None if configuration is None else configuration[0]


def train_direction(agent):
    """Current facing direction (0=N, 1=E, 2=S, 3=W), or None if off the map."""
    configuration = agent.current_configuration
    return None if configuration is None else configuration[1]


def train_target(agent):
    """
    Target (row, col) of the train.

    Flatland 4.x stores the target as a set of (cell, direction) pairs:
    the same cell, entered from any allowed direction.
    """
    cells = {cell for cell, _direction in agent.targets}
    return next(iter(cells)) if len(cells) == 1 else cells


def direction_name(direction):
    """Readable name for a direction number, e.g. 1 -> 'EAST'."""
    if direction is None:
        return None
    return Grid4TransitionsEnum(direction).name


def _possible_next_cells(env, agent):
    """Cells the train could legally move into from where it stands now."""
    position, direction = agent.current_configuration
    transitions = env.rail.get_transitions((position, direction))
    return [
        get_new_position(position, exit_direction)
        for exit_direction, allowed in enumerate(transitions)
        if allowed
    ]


def find_deadlocked_trains(env):
    """
    Return the set of train handles that can never move again.

    A train is deadlocked when every cell it could move into is occupied
    by another deadlocked train. This catches:
      - head-on deadlocks (two trains facing each other on one track),
      - chains (a train stuck behind a deadlocked train),
      - cycles (A waits for B, B waits for C, C waits for A).

    Method: start by assuming every train on the map is stuck, then
    repeatedly release any train that has at least one next cell that is
    free or held by a train that is not stuck. Whatever remains when
    nothing more can be released is deadlocked.

    Trains that are malfunctioning are treated like any other occupant:
    they block cells, but a malfunction ends, so a train is only reported
    as deadlocked if it is stuck behind other deadlocked trains.

    Limitation: this checks blocking by cells only. It ignores timing
    (speeds) and Flatland's level-free crossings, which are not used in
    the current configuration.
    """
    on_map = {
        agent.handle: agent
        for agent in env.agents
        if agent.state.is_on_map_state() and agent.current_configuration is not None
    }
    cell_to_train = {train_position(agent): handle for handle, agent in on_map.items()}

    stuck = {
        handle for handle, agent in on_map.items()
        if agent.state != TrainState.MALFUNCTION
    }

    changed = True
    while changed:
        changed = False
        for handle in list(stuck):
            for cell in _possible_next_cells(env, on_map[handle]):
                occupant = cell_to_train.get(cell)
                if occupant is None or occupant not in stuck:
                    stuck.discard(handle)
                    changed = True
                    break

    return stuck
