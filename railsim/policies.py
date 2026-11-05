"""
Dispatching policies.

A policy is any object with:
    name             short label used in results tables
    reset(env)       called once after the environment is built
    act(env) -> dict {train handle: action} for the next step

Phase 1 provides two reference policies with no conflict handling at all.
The rule-based dispatchers (FCFS, Priority) and the RL policy will follow
the same interface.
"""

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policies import ShortestPathPolicy


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


POLICIES = {
    AlwaysForward.name: AlwaysForward,
    ShortestPathNoDispatch.name: ShortestPathNoDispatch,
}
