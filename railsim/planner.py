"""
Conflict-free route planning in space and time (prioritized planning).

Trains are planned one at a time, in priority order. Each train searches
for the earliest arrival at its target, moving one cell per step or
waiting, while avoiding every cell already booked by trains planned before
it. Its own route is then booked for the trains after it. Because no two
booked routes ever overlap, the resulting plan has no conflicts and so
cannot deadlock.

Time convention (matches Flatland, speed 1):
    "position at time t" = where the train is after env step t.
    The action sent in step t moves the train from its position at t-1 to
    its position at t.
    A train ready to depart appears on its start cell at the end of the
    step in which it is told to move.

Booking rule: a train may enter a cell at time t only if no train already
booked is there at t-1 or at t. As a result:
  - no two trains are ever booked in the same cell at the same time;
  - two trains never swap cells head-on in one step;
  - a train planned later may still leave a cell in the same step that a
    train planned earlier enters it (nose-to-tail following). Flatland
    allows this; if the leading train is stopped by a breakdown at that
    moment, Flatland holds the follower back, the dispatcher sees the
    train is not where planned, and re-plans.

If a train already on the track cannot find any route, it is "parked": it
books its current cell for the rest of the episode, and planning restarts
with that cell blocked for everyone. A train that has not departed and has
no route simply does not depart yet.
"""

import heapq
import itertools

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

MOVE_ACTIONS = (RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT)
OFF_MAP = None
ARRIVED = "ARRIVED"   # search marker: the train has reached its target


class Bookings:
    """Which train holds which cell at which time."""

    def __init__(self):
        self._at = {}       # (cell, t) -> handle
        self._from = {}     # cell -> (handle, first t) for cells held to the end

    def holder(self, cell, t):
        handle = self._at.get((cell, t))
        if handle is not None:
            return handle
        held = self._from.get(cell)
        if held is not None and t >= held[1]:
            return held[0]
        return None

    def free_for(self, cell, t, handle):
        holder = self.holder(cell, t)
        return holder is None or holder == handle

    def book(self, cell, t, handle):
        self._at[(cell, t)] = handle

    def book_until_end(self, cell, t, handle):
        self._from[cell] = (handle, t)


def _next_moves(env, configuration):
    """{next configuration: action} for every legal move from `configuration`."""
    moves = {}
    for action in MOVE_ACTIONS:
        result = env.rail.apply_action_independent(action, configuration)
        if result is not None:
            next_configuration, _ = result
            moves.setdefault(next_configuration, action)
    return moves


def _target_cell(agent):
    return next(iter(agent.targets))[0]


def earliest_move_time(agent, now):
    """First time step at which this train can be somewhere new."""
    broken_for = agent.malfunction_handler.malfunction_down_counter
    earliest = now + broken_for + 1
    if agent.state.is_off_map_state():
        # Flatland marks a train ready to depart during step
        # max(earliest_departure, 1) (there is no step 0), and it can appear
        # on the track from the following step.
        earliest = max(earliest, max(agent.earliest_departure, 1) + 1)
    return earliest


def plan_route(env, agent, now, bookings, horizon):
    """
    Earliest-arrival route for one train that avoids `bookings`.

    Returns {t: (configuration or OFF_MAP, action)} for t = now+1 .. arrival,
    where action is what to send in step t. Returns None if the train cannot
    reach its target by `horizon`.
    """
    handle = agent.handle
    target = _target_cell(agent)
    distance = env.distance_map.get()[handle]
    start_move = earliest_move_time(agent, now)

    def estimate(configuration):
        if configuration is OFF_MAP:
            (r, c), d = agent.initial_configuration
            return distance[r, c, d] + 1
        (r, c), d = configuration
        return distance[r, c, d]

    start = agent.current_configuration if agent.state.is_on_map_state() else OFF_MAP
    if estimate(start) == float("inf"):
        return None

    tie = itertools.count()
    queue = [(now + estimate(start), next(tie), now, start)]
    came_from = {(now, start): None}

    while queue:
        _, _, t, configuration = heapq.heappop(queue)
        if configuration == ARRIVED:
            return _unwind(came_from, (t, ARRIVED))
        if t >= horizon:
            continue
        step = t + 1

        options = []  # (next configuration, action, reached target)

        # Wait where we are.
        if configuration is OFF_MAP:
            options.append((OFF_MAP, RailEnvActions.STOP_MOVING, False))
        elif bookings.free_for(configuration[0], step, handle):
            options.append((configuration, RailEnvActions.STOP_MOVING, False))

        # Move (or depart), if allowed by malfunction / timetable.
        if step >= start_move:
            if configuration is OFF_MAP:
                moves = {agent.initial_configuration: RailEnvActions.MOVE_FORWARD}
            else:
                moves = _next_moves(env, configuration)
            for next_configuration, action in moves.items():
                cell = next_configuration[0]
                if bookings.free_for(cell, t, handle) and bookings.free_for(cell, step, handle):
                    options.append((next_configuration, action, cell == target))

        for next_configuration, action, arrived in options:
            key = (step, ARRIVED if arrived else next_configuration)
            if key in came_from:
                continue
            came_from[key] = ((t, configuration), action, next_configuration)
            remaining = 0 if arrived else estimate(next_configuration)
            if remaining == float("inf"):
                continue
            heapq.heappush(queue, (step + remaining, next(tie), key[0], key[1]))

    return None


def _unwind(came_from, key):
    route = {}
    while came_from[key] is not None:
        (previous_t, previous_configuration), action, configuration = came_from[key]
        route[key[0]] = (configuration, action)
        key = (previous_t, previous_configuration)
    return route


def plan_all(env, order, now):
    """
    Plan every unfinished train in `order` (list of handles, first = highest
    priority).

    Returns (routes, parked):
      routes  {handle: route} for trains that found a route
      parked  set of on-track trains with no route; they stay where they are
    Trains that have not departed and found no route are in neither.
    """
    horizon = env._max_episode_steps
    active = [h for h in order if env.agents[h].state != TrainState.DONE]
    parked = set()

    while True:
        bookings = Bookings()

        # Facts that hold whatever the plan: every train on the track holds
        # its cell now and for as long as it is broken down.
        for handle in active:
            agent = env.agents[handle]
            if agent.state.is_on_map_state():
                cell = agent.current_configuration[0]
                for t in range(now, earliest_move_time(agent, now)):
                    bookings.book(cell, t, handle)
        for handle in parked:
            bookings.book_until_end(env.agents[handle].current_configuration[0], now, handle)

        routes = {}
        newly_parked = None
        for handle in active:
            if handle in parked:
                continue
            agent = env.agents[handle]
            route = plan_route(env, agent, now, bookings, horizon)
            if route is None:
                if agent.state.is_on_map_state():
                    newly_parked = handle
                    break
                continue
            routes[handle] = route
            for t, (configuration, _action) in route.items():
                if configuration is not OFF_MAP:
                    bookings.book(configuration[0], t, handle)
            last_t = max(route)
            last_configuration = route[last_t][0]
            target = _target_cell(agent)
            if last_configuration is not OFF_MAP and last_configuration[0] == target:
                bookings.book(target, last_t, handle)

        if newly_parked is None:
            return routes, parked
        parked.add(newly_parked)
