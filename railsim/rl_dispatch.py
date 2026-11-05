"""
Reinforcement-learning dispatcher.

Division of work (inspired by the dispatching/routing split in Castagna et
al. 2026, arXiv:2605.10257):

  - RL decides WHEN each train is released from its station: for every
    train that is ready to depart, at every step, the agent chooses
    Dispatch or Wait.
  - The same conflict-free planner used by FCFS and Priority does the
    routing. Released trains book routes in the order they were released
    (ties: least slack first). Trains still waiting are not planned and
    stay at their station.

So the RL agent controls exactly what the rule-based dispatchers control
(who goes, and in which order), and like them it cannot cause a deadlock.
If the agent dispatches every train as soon as it is ready, this becomes
"release_all", which is kept as a sanity-check policy.

This module holds:
  ReleaseDispatcher   planner + release decisions (used by all below)
  observation()       the numbers the agent sees for one decision
  ReleaseAll          dispatch every ready train immediately (no learning)
  RLDispatcher        uses a trained Stable-Baselines3 model
  DispatchEnv         Gymnasium environment used to train the model
"""

import gymnasium as gym
import numpy as np
from flatland.envs.step_utils.states import TrainState

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env
from railsim.metrics import network_delay
from railsim.planner import OFF_MAP, bookings_for, plan_route
from railsim.policies import PlanningDispatcher, train_slack

WAIT, DISPATCH = 0, 1
OBS_SIZE = 13


def ready_to_release(env, now):
    """
    Trains that could appear on the track in the next step if released:
    not departed, not finished, and past their earliest departure.
    (Flatland lets a train appear from step max(earliest_departure, 1) + 1.)
    """
    return [
        agent.handle for agent in env.agents
        if agent.state.is_off_map_state()
        and now >= max(agent.earliest_departure, 1)
    ]


class ReleaseDispatcher(PlanningDispatcher):
    """Planner that only plans trains on the track or already released."""

    def reset(self, env):
        super().reset(env)
        self.released = {}   # handle -> step it was released

    def pending(self, env, now):
        """Ready trains not yet released, most urgent (least slack) first."""
        waiting = [h for h in ready_to_release(env, now) if h not in self.released]
        return sorted(waiting, key=lambda h: (train_slack(env, h, now), h))

    def priority_order(self, env, now):
        planned = [
            a.handle for a in env.agents
            if a.state != TrainState.DONE
            and (a.state.is_on_map_state() or a.handle in self.released)
        ]
        return sorted(planned, key=lambda h: (self.released.get(h, -1), train_slack(env, h, now), h))

    def release(self, env, handle, now):
        """Release one train and re-plan at once, so later decisions in the
        same step see its booked route."""
        self.released[handle] = now
        self._replan(env, now)

    def decide(self, env, now, handle):
        """Return DISPATCH or WAIT for one ready train. Subclasses decide."""
        raise NotImplementedError

    def act(self, env):
        now = env._elapsed_steps
        for handle in self.pending(env, now):
            if self.decide(env, now, handle) == DISPATCH:
                self.release(env, handle, now)
        return super().act(env)


def observation(env, dispatcher, handle, now):
    """
    What the agent sees when deciding about train `handle` at step `now`.
    Every value is scaled by the episode length T or the number of trains N,
    so the same policy can be run on scenarios of other sizes.

     0  can_go          1 if the planner finds a route if released now, else 0
     1  planned_late    lateness at arrival if released now (T if no route) / T
     2  planned_hold    steps the planner would keep it at the station / T
     3  slack           spare steps before its latest arrival / T, in [-1, 1]
     4  distance        shortest travel time to its target / T
     5  time_left       steps left in the episode / T
     6  on_track        trains currently on the track / N
     7  arrived         trains already arrived / N
     8  others_waiting  other ready trains not yet released / N
     9  broken          trains broken down right now / N
    10  late_in_plan    released trains the current plan makes late / N
    11  urgency_rank    position among waiting trains by slack (0 = most urgent) / N
    12  free_now        1 if released now it could leave at once (route found,
                        no hold at the station), else 0. Same information as
                        0 and 2 combined, but as a clear on/off signal: a hold
                        of 1 step is only ~0.01 in feature 2, which a small
                        network barely separates from 0.
    """
    agent = env.agents[handle]
    T = env._max_episode_steps
    N = len(env.agents)
    (r, c), d = agent.initial_configuration
    distance = env.distance_map.get()[handle, r, c, d]

    bookings = bookings_for(env, dispatcher.routes, dispatcher.parked, now)
    route = plan_route(env, agent, now, bookings, T)
    if route is None:
        can_go, planned_late, planned_hold = 0.0, 1.0, 1.0
    else:
        arrival = max(route)
        entered = min(t for t, (cfg, _a) in route.items() if cfg is not OFF_MAP)
        can_go = 1.0
        planned_late = max(0, arrival - agent.latest_arrival) / T
        planned_hold = (entered - (now + 1)) / T

    late_in_plan = 0
    for h, plan in dispatcher.routes.items():
        if max(plan) > env.agents[h].latest_arrival:
            late_in_plan += 1

    waiting = dispatcher.pending(env, now)
    others_waiting = len([h for h in waiting if h != handle])
    rank = waiting.index(handle) if handle in waiting else 0

    values = np.array([
        can_go,
        planned_late,
        planned_hold,
        float(np.clip(train_slack(env, handle, now) / T, -1, 1)),
        distance / T,
        (T - now) / T,
        sum(a.state.is_on_map_state() for a in env.agents) / N,
        sum(a.state == TrainState.DONE for a in env.agents) / N,
        others_waiting / N,
        sum(a.state == TrainState.MALFUNCTION for a in env.agents) / N,
        late_in_plan / N,
        rank / N,
        1.0 if route is not None and planned_hold == 0 else 0.0,
    ], dtype=np.float32)
    # Values are already in range by construction; clipping only guards
    # against unreachable targets (infinite distance).
    return np.clip(np.nan_to_num(values, posinf=1.0, neginf=-1.0), -1.0, 1.0)


class ReleaseAll(ReleaseDispatcher):
    """Dispatch every ready train immediately (the RL agent's 'always
    Dispatch' policy). Used to check what the learned waiting adds."""

    name = "release_all"

    def decide(self, env, now, handle):
        return DISPATCH


def hold_if_blocked_action(obs):
    """Dispatch only if the train could leave at once (obs[12], free_now);
    otherwise Wait."""
    return DISPATCH if obs[12] == 1.0 else WAIT


class HoldIfBlocked(ReleaseDispatcher):
    """
    Simple departure-control rule (no learning): release a ready train only
    if the planner can let it leave right away, i.e. its route is free now.
    Otherwise keep it at the station and ask again next step.

    Departure rules like this are what Laurent et al. (2021, arXiv:2103.16511,
    Sec. 6) found "greatly improve the performance" in the Flatland 2020
    challenge, so the RL agent must beat this rule, not only release_all,
    before any gain can be credited to learning.
    """

    name = "hold_if_blocked"

    def decide(self, env, now, handle):
        return hold_if_blocked_action(observation(env, self, handle, now))


class RLDispatcher(ReleaseDispatcher):
    """Dispatcher driven by a trained Stable-Baselines3 model.

    Set the model file with `RLDispatcher.model_path = "..."` before use;
    the model is loaded once and shared.
    """

    name = "rl"
    model_path = "models/rl_dispatcher.zip"
    _model = None
    _loaded_from = None

    @classmethod
    def model(cls):
        if cls._model is None or cls._loaded_from != cls.model_path:
            from stable_baselines3 import PPO
            cls._model = PPO.load(cls.model_path, device="cpu")
            cls._loaded_from = cls.model_path
        return cls._model

    def decide(self, env, now, handle):
        obs = observation(env, self, handle, now)
        action, _ = self.model().predict(obs, deterministic=True)
        return int(action)


class DispatchEnv(gym.Env):
    """
    Gymnasium environment for training: one RL step = one Dispatch/Wait
    decision for one ready train.

    After the last pending decision of a simulation step, the simulator
    advances (the planner moves released trains) until some train needs a
    decision again or the episode ends.

    Reward: minus the increase in network delay since the previous
    decision, divided by the number of trains. Network delay is the metric
    from railsim/metrics.py (equal to minus Flatland's own reward), so over
    an episode the rewards add up to
        -(final total delay - delay at the first decision) / N
    (delay that builds up before any train can be released cannot be
    influenced, so it is not part of the reward).

    Each episode uses a scenario seed drawn from `seeds`. Keep these
    separate from the evaluation seeds.
    """

    metadata = {"render_modes": []}

    def __init__(self, seeds, num_trains=8, disruptions=DisruptionSpec(count=1, duration=20)):
        super().__init__()
        self.seeds = list(seeds)
        self.num_trains = num_trains
        self.disruptions = disruptions
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = int(self.np_random.choice(self.seeds))
        if options and "scenario" in options:
            scenario = int(options["scenario"])
        self.scenario = scenario
        self.env, _, _ = make_env(EnvConfig(num_trains=self.num_trains, seed=scenario), self.disruptions)
        self.dispatcher = _ExternalDecisions()
        self.dispatcher.reset(self.env)
        self.queue = []
        self.done = False
        self.decisions = 0
        self._delay = network_delay(self.env)
        self._advance()   # no decision possible yet, so this delay is not rewarded
        self.delay_at_first_decision = self._delay
        return self._obs(), {"scenario": scenario}

    def step(self, action):
        handle = self.queue.pop(0)
        now = self.env._elapsed_steps
        self.decisions += 1
        if int(action) == DISPATCH:
            self.dispatcher.release(self.env, handle, now)

        # The remaining trains of this step are asked in turn (a train that
        # was told to Wait is asked again in the next step, not this one).
        reward = 0.0
        if not self.queue:
            reward += self._sim_step()
            reward += self._advance()

        info = {"scenario": self.scenario}
        if self.done:
            info["final_delay"] = self._delay
        return self._obs(), reward, self.done, False, info

    # -- helpers ---------------------------------------------------------

    def _sim_step(self):
        """Advance the simulator one step; return the reward for it."""
        before = self._delay   # network delay after the previous step
        _, _, dones, _ = self.env.step(self.dispatcher.act(self.env))
        self.done = dones["__all__"]
        self._delay = network_delay(self.env)
        return -(self._delay - before) / self.num_trains

    def _advance(self):
        """Step the simulator until a train needs a decision or the episode ends."""
        reward = 0.0
        while not self.done:
            self.queue = self.dispatcher.pending(self.env, self.env._elapsed_steps)
            if self.queue:
                return reward
            reward += self._sim_step()
        self.queue = []
        return reward

    def _obs(self):
        if self.done or not self.queue:
            return np.zeros(OBS_SIZE, dtype=np.float32)
        return observation(self.env, self.dispatcher, self.queue[0], self.env._elapsed_steps)


class _ExternalDecisions(ReleaseDispatcher):
    """Release decisions are made by the training loop, not by `decide`."""

    def act(self, env):
        return PlanningDispatcher.act(self, env)
