"""
Dispatching policies.

A policy is any object with:
    name             short label used in results tables
    reset(env)       called once after the environment is built
    act(env) -> dict {train handle: action} for the next step

Reference policies with no conflict handling:
    always_forward, shortest_path
Rule-based dispatchers (conflict-free, re-plan after disruptions):
    fcfs, priority
"""

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.step_utils.states import TrainState

from railsim.planner import OFF_MAP, plan_all


class AlwaysForward:
    """Every train always tries to move forward. Ignores routes and conflicts."""

    name = "always_forward"

    def reset(self, env):
        pass

    def act(self, env):
        return {handle: RailEnvActions.MOVE_FORWARD for handle in env.get_agent_handles()}


class ShortestPathNoDispatch:
    """
    Every train follows its own shortest route to its target and departs as
    soon as its timetable allows. Nobody ever waits for anyone, so this is
    "no dispatcher": it shows what happens when conflicts are not managed.

    Wraps Flatland's built-in ShortestPathPolicy.
    """

    name = "shortest_path"

    def reset(self, env):
        self._policy = ShortestPathPolicy()

    def act(self, env):
        handles = env.get_agent_handles()
        return self._policy.act_many(handles, [env] * len(handles))


def train_slack(env, handle, now):
    """
    Spare steps before the train misses its latest arrival, if it ran its
    shortest path from now (or from its departure, if it has not left yet):

        slack = latest_arrival - (start + shortest remaining travel time)

    Negative = the train will be late even with a clear track.
    """
    agent = env.agents[handle]
    distance = env.distance_map.get()
    if agent.state.is_on_map_state():
        (r, c), d = agent.current_configuration
        return agent.latest_arrival - (now + distance[handle, r, c, d])
    (r, c), d = agent.initial_configuration
    start = max(now, agent.earliest_departure)
    return agent.latest_arrival - (start + distance[handle, r, c, d] + 1)


class PlanningDispatcher:
    """
    Rule-based dispatcher: books conflict-free routes for all trains in a
    priority order (see railsim/planner.py) and follows them.

    It re-plans, from the current state, whenever:
      - any train is not where the plan says it should be (for example a
        breakdown just stopped it), or
      - some trains had no route last time (retried every `retry_every`
        steps, since the network may have cleared).

    The dispatcher learns about a breakdown only when it happens, and is
    then told how long it will last (Flatland's malfunction counter).

    Subclasses only decide the priority order.
    """

    name = None
    retry_every = 5

    def reset(self, env):
        self.routes = {}
        self.parked = set()
        self.unplanned = set()
        self.last_plan_step = None
        self.replans = 0
        self.divergences = 0   # times a train was not where the plan said

    def priority_order(self, env, now):
        raise NotImplementedError

    def _where(self, agent):
        if agent.state.is_on_map_state():
            return agent.current_configuration
        return OFF_MAP

    def _plan_is_valid(self, env, now):
        for agent in env.agents:
            if agent.state == TrainState.DONE:
                continue
            route = self.routes.get(agent.handle)
            if route is None:
                continue
            expected = route.get(now)
            if expected is None or expected[0] != self._where(agent):
                return False
        return True

    def _replan(self, env, now):
        order = self.priority_order(env, now)
        self.routes, self.parked = plan_all(env, order, now)
        # Trains we tried to plan but could not (not departed, no route yet).
        self.unplanned = {
            h for h in order
            if env.agents[h].state != TrainState.DONE and h not in self.routes and h not in self.parked
        }
        self.last_plan_step = now
        self.replans += 1

    def needs_replan(self, env, now):
        """True if the plan must be redone now (first step, a train is off
        its plan, or a retry for trains without a route is due)."""
        diverged = self.last_plan_step is not None and not self._plan_is_valid(env, now)
        self.divergences += diverged
        return (
            self.last_plan_step is None
            or diverged
            or bool((self.parked or self.unplanned) and now - self.last_plan_step >= self.retry_every)
        )

    def follow_plan(self, env, now):
        """Actions that move every train as its booked route says."""
        actions = {}
        for agent in env.agents:
            step = self.routes.get(agent.handle, {}).get(now + 1)
            actions[agent.handle] = step[1] if step else RailEnvActions.STOP_MOVING
        return actions

    def act(self, env):
        now = env._elapsed_steps
        if self.needs_replan(env, now):
            self._replan(env, now)
        return self.follow_plan(env, now)


class FCFSDispatcher(PlanningDispatcher):
    """
    First-Come-First-Served: trains scheduled to depart earlier book their
    routes first. The order never changes during the episode.
    """

    name = "fcfs"

    def priority_order(self, env, now):
        return sorted(env.get_agent_handles(), key=lambda h: (env.agents[h].earliest_departure, h))


class PriorityDispatcher(PlanningDispatcher):
    """
    Priority (least slack first): the train closest to missing its latest
    arrival books its route first. Slack is recomputed at every re-plan:

        slack = latest_arrival - (now + shortest remaining travel time)

    A negative slack means the train will be late even with a clear track.
    """

    name = "priority"

    def priority_order(self, env, now):
        return sorted(env.get_agent_handles(), key=lambda h: (train_slack(env, h, now), h))


POLICIES = {
    AlwaysForward.name: AlwaysForward,
    ShortestPathNoDispatch.name: ShortestPathNoDispatch,
    FCFSDispatcher.name: FCFSDispatcher,
    PriorityDispatcher.name: PriorityDispatcher,
}
