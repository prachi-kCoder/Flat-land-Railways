"""
Rescheduling-strategy selection.

The rule-based dispatchers differ only in the ORDER in which trains book
their routes. No single order is best everywhere: on our validation
scenarios FCFS beat Priority in 6 of 30 and Priority beat FCFS in 9, and
choosing the better of the two per scenario cut mean delay by about a
third. Li et al. (ICAPS 2021) likewise ran several orders and kept the best.

So here the decision at every re-plan (the start, and every time a
breakdown or other event knocks a train off its plan) is: which ordering
rule should the whole network be re-planned with?

  ORDERINGS          the candidate rules
  OrderedDispatcher  always the same rule (fcfs, priority, edf, spf)
  plan_lateness()    how late a plan would make the trains
  GreedySelect       try every rule, keep the plan with least lateness
                     (strong non-learning baseline)
  strategy_observation()  what the RL agent sees at a re-plan
"""

import gymnasium as gym
import numpy as np
from flatland.envs.step_utils.states import TrainState

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env
from railsim.metrics import network_delay
from railsim.planner import plan_all
from railsim.policies import PlanningDispatcher, train_slack


def _remaining_distance(env, handle):
    agent = env.agents[handle]
    distance = env.distance_map.get()
    if agent.state.is_on_map_state():
        (r, c), d = agent.current_configuration
        return distance[handle, r, c, d]
    (r, c), d = agent.initial_configuration
    return distance[handle, r, c, d] + 1


# Each rule maps (env, handle, now) to a sort key; smaller books first.
ORDERINGS = {
    # First come, first served: scheduled departure.
    "fcfs": lambda env, h, now: (env.agents[h].earliest_departure, h),
    # Least slack first (Chen et al. 2023, arXiv:2306.06455).
    "priority": lambda env, h, now: (train_slack(env, h, now), h),
    # Earliest deadline first: latest arrival time.
    "edf": lambda env, h, now: (env.agents[h].latest_arrival, h),
    # Shortest remaining trip first.
    "spf": lambda env, h, now: (_remaining_distance(env, h), h),
}
NAMES = list(ORDERINGS)


def order_by(env, rule, now):
    return sorted(env.get_agent_handles(), key=lambda h: ORDERINGS[rule](env, h, now))


def plan_lateness(env, routes, now):
    """
    Total lateness (steps) the trains would have if the plan ran as booked:
      - a train with a route: max(0, planned arrival - latest arrival)
      - a train without one (held or parked): treated as never arriving,
        max(0, episode end + remaining travel - latest arrival), the same
        projection the delay metric uses at the end of an episode.
    """
    horizon = env._max_episode_steps
    total = 0
    for agent in env.agents:
        if agent.state == TrainState.DONE:
            continue
        route = routes.get(agent.handle)
        if route is not None:
            total += max(0, max(route) - agent.latest_arrival)
        else:
            total += max(0, horizon + _remaining_distance(env, agent.handle) - agent.latest_arrival)
    return total


class OrderedDispatcher(PlanningDispatcher):
    """Prioritized planning with one fixed ordering rule."""

    rule = None

    def priority_order(self, env, now):
        return order_by(env, self.rule, now)


class EDFDispatcher(OrderedDispatcher):
    """Earliest deadline (latest arrival) first."""
    name = rule = "edf"


class SPFDispatcher(OrderedDispatcher):
    """Shortest remaining trip first."""
    name = rule = "spf"


class SelectingDispatcher(PlanningDispatcher):
    """
    Re-plans with a rule chosen at each re-plan by `choose`.
    `self.choices` records (step, rule) for analysis.
    """

    def reset(self, env):
        super().reset(env)
        self.choices = []
        self.current_rule = None

    def choose(self, env, now, candidates):
        """candidates: {rule: (routes, parked, lateness)}. Return a rule."""
        raise NotImplementedError

    def candidate_plans(self, env, now):
        plans = {}
        for rule in NAMES:
            routes, parked = plan_all(env, order_by(env, rule, now), now)
            plans[rule] = (routes, parked, plan_lateness(env, routes, now))
        return plans

    def apply(self, env, now, rule, plans):
        routes, parked, _ = plans[rule]
        self.routes, self.parked = routes, parked
        order = order_by(env, rule, now)
        self.unplanned = {
            h for h in order
            if env.agents[h].state != TrainState.DONE and h not in routes and h not in parked
        }
        self.last_plan_step = now
        self.replans += 1
        self.current_rule = rule
        self.choices.append((now, rule))

    def _replan(self, env, now):
        plans = self.candidate_plans(env, now)
        self.apply(env, now, self.choose(env, now, plans), plans)


class GreedySelect(SelectingDispatcher):
    """At each re-plan, keep the rule whose plan has the least lateness.
    Ties keep the current rule if it is among the best, else the first in
    NAMES. This looks only at the plan, not at future breakdowns."""

    name = "greedy_select"

    def choose(self, env, now, plans):
        best = min(p[2] for p in plans.values())
        tied = [r for r in NAMES if plans[r][2] == best]
        return self.current_rule if self.current_rule in tied else tied[0]


def strategy_observation(env, dispatcher, now, plans):
    """
    What the RL agent sees at a re-plan (all values in [-1, 1]):

      0          time used so far / T
      1          trains on the track / N
      2          trains arrived / N
      3          trains not yet departed / N
      4          trains broken down now / N
      5          remaining breakdown steps of broken trains / T (capped at 1)
      6          mean slack of unfinished trains / T
      7          minimum slack / T
      8          unfinished trains with negative slack / N
      9          least plan lateness among the rules / (N * 10)  (capped at 1)
      10..13     per rule: (its plan lateness - least) / (least + 1), capped at 1
      14..17     per rule: 1 if its plan has the least lateness
      18..21     per rule: 1 if it is the rule currently in use
    """
    T = env._max_episode_steps
    N = len(env.agents)
    unfinished = [a for a in env.agents if a.state != TrainState.DONE]
    slacks = [train_slack(env, a.handle, now) for a in unfinished] or [0]
    broken = [a for a in env.agents if a.state.is_malfunction_state()]
    lateness = {rule: plans[rule][2] for rule in NAMES}
    least = min(lateness.values())

    values = [
        now / T,
        sum(a.state.is_on_map_state() for a in env.agents) / N,
        sum(a.state == TrainState.DONE for a in env.agents) / N,
        sum(a.state.is_off_map_state() for a in env.agents) / N,
        len(broken) / N,
        min(1.0, sum(a.malfunction_handler.malfunction_down_counter for a in broken) / T),
        float(np.mean(slacks)) / T,
        float(np.min(slacks)) / T,
        sum(s < 0 for s in slacks) / N,
        min(1.0, least / (N * 10)),
    ]
    values += [min(1.0, (lateness[r] - least) / (least + 1)) for r in NAMES]
    values += [1.0 if lateness[r] == least else 0.0 for r in NAMES]
    values += [1.0 if dispatcher.current_rule == r else 0.0 for r in NAMES]
    values = np.array(values, dtype=np.float32)
    return np.clip(np.nan_to_num(values, posinf=1.0, neginf=-1.0), -1.0, 1.0)


STRATEGY_OBS_SIZE = 10 + 3 * len(NAMES)


# ---------------------------------------------------------------- RL

def greedy_action(obs):
    """GreedySelect's choice, read from a strategy observation: the rule
    in use if its plan is among the least late, else the first such rule."""
    best = obs[10 + len(NAMES): 10 + 2 * len(NAMES)]
    current = obs[10 + 2 * len(NAMES): 10 + 3 * len(NAMES)]
    for i in range(len(NAMES)):
        if best[i] == 1.0 and current[i] == 1.0:
            return i
    return int(np.argmax(best == 1.0))


class RLStrategy(SelectingDispatcher):
    """Chooses the re-planning rule with a trained Stable-Baselines3 model.
    Set `RLStrategy.model_path` before use."""

    name = "rl_strategy"
    model_path = "models/rl_strategy.zip"
    _model = None
    _loaded_from = None

    @classmethod
    def model(cls):
        if cls._model is None or cls._loaded_from != cls.model_path:
            from stable_baselines3 import PPO
            cls._model = PPO.load(cls.model_path, device="cpu")
            cls._loaded_from = cls.model_path
        return cls._model

    def choose(self, env, now, plans):
        obs = strategy_observation(env, self, now, plans)
        action, _ = self.model().predict(obs, deterministic=True)
        return NAMES[int(action)]


class StrategyEnv(gym.Env):
    """
    Gymnasium environment: one RL step = choose the ordering rule for
    one re-plan. The simulator then runs the plan until the next
    re-plan is needed (a breakdown, a train off its plan, or a retry)
    or the episode ends.

    Reward: minus the increase in network delay since the previous
    decision, divided by the number of trains. Over an episode the
    rewards add up to -(final delay - delay at the first decision) / N.
    """

    metadata = {"render_modes": []}

    # Conditions sampled per episode when mixed=True (the "mixed-condition"
    # control suggested by Yue et al. 2024: does broader training reduce
    # degradation?). Still narrower than the robustness test levels.
    MIXED_TRAINS = [6, 8, 10, 12]
    MIXED_DURATIONS = [10, 20, 30, 40]
    MIXED_COUNTS = [1, 2]

    def __init__(self, seeds, num_trains=8, disruptions=DisruptionSpec(count=1, duration=20), mixed=False):
        super().__init__()
        self.seeds = list(seeds)
        self.base_num_trains = num_trains
        self.base_disruptions = disruptions
        self.mixed = mixed
        self.num_trains = num_trains
        self.disruptions = disruptions
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(STRATEGY_OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(NAMES))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = int(self.np_random.choice(self.seeds))
        if options and "scenario" in options:
            scenario = int(options["scenario"])
        self.scenario = scenario
        if self.mixed and not (options and options.get("fixed_conditions")):
            count = int(self.np_random.choice(self.MIXED_COUNTS))
            self.num_trains = int(self.np_random.choice(self.MIXED_TRAINS))
            self.disruptions = DisruptionSpec(
                count=count, duration=int(self.np_random.choice(self.MIXED_DURATIONS)),
                simultaneous=bool(count > 1 and self.np_random.random() < 0.5),
            )
        else:
            self.num_trains, self.disruptions = self.base_num_trains, self.base_disruptions
        self.env, _, _ = make_env(EnvConfig(num_trains=self.num_trains, seed=scenario), self.disruptions)
        self.dispatcher = SelectingDispatcher()
        self.dispatcher.reset(self.env)
        self.done = False
        self._delay = network_delay(self.env)
        self.delay_at_first_decision = self._delay
        self.decisions = 0
        self._prepare_decision()   # step 0: the first plan is always needed
        return self._obs(), {"scenario": scenario}

    def step(self, action):
        now = self.env._elapsed_steps
        self.dispatcher.apply(self.env, now, NAMES[int(action)], self.plans)
        self.decisions += 1
        before = self._delay
        self._run_until_next_decision()
        reward = -(self._delay - before) / self.num_trains
        info = {"scenario": self.scenario}
        if self.done:
            info["final_delay"] = self._delay
        return self._obs(), reward, self.done, False, info

    def _prepare_decision(self):
        self.plans = self.dispatcher.candidate_plans(self.env, self.env._elapsed_steps)

    def _run_until_next_decision(self):
        while True:
            now = self.env._elapsed_steps
            _, _, dones, _ = self.env.step(self.dispatcher.follow_plan(self.env, now))
            self.done = dones["__all__"]
            self._delay = network_delay(self.env)
            if self.done:
                self.plans = None
                return
            if self.dispatcher.needs_replan(self.env, self.env._elapsed_steps):
                self._prepare_decision()
                return

    def _obs(self):
        if self.done:
            return np.zeros(STRATEGY_OBS_SIZE, dtype=np.float32)
        return strategy_observation(self.env, self.dispatcher, self.env._elapsed_steps, self.plans)
