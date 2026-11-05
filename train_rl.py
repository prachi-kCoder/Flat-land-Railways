# ============================================================
# Train an RL policy (PPO, Stable-Baselines3)
# ============================================================
#
# Two tasks:
#   strategy   (default) at every re-plan, choose which ordering rule
#              (fcfs, priority, edf, spf) the network is re-planned with.
#              Warm start: imitate greedy_select.
#   departure  for every train ready to leave, choose Dispatch or Wait.
#              Warm start: imitate hold_if_blocked. Kept for the record:
#              it could not improve on its warm start (see README).
#
# Training conditions are held fixed (the "training distribution"):
#   8 trains, 1 breakdown of 20 steps per episode.
# Everything else is varied only at test time (run_robustness.py).
#
# Scenario seeds are kept apart:
#   training    seeds from 1000 upwards
#   validation  seeds from 5000 upwards (used to keep the best checkpoint)
#   testing     seeds from 0 upwards   (run_experiment.py / run_robustness.py)
#
# Example:
#   python train_rl.py --seed 1
# Writes:
#   models/rl_strategy_seed1.zip        best checkpoint on validation
#   models/rl_strategy_seed1_log.csv    validation score during training
# ============================================================

import argparse
import csv
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig
from railsim.env_factory import usable_seeds
from railsim.rl_dispatch import DispatchEnv, hold_if_blocked_action
from railsim.strategy import NAMES, StrategyEnv, greedy_action

TRAIN_TRAINS = 8
TRAIN_DISRUPTIONS = DisruptionSpec(count=1, duration=20)
TRAIN_SEED_START = 1000
VALIDATION_SEED_START = 5000

TASKS = {
    "strategy": {
        "env": StrategyEnv, "teacher": greedy_action, "teacher_name": "greedy_select",
        "prefix": "rl_strategy", "timesteps": 60_000, "eval_every": 5_000,
        "warm_start_scenarios": 300, "action_names": NAMES,
        # About 4-5 decisions per episode, so no discounting is needed:
        # with gamma = 1 the return is exactly -(total delay)/N.
        "ppo": dict(n_steps=128, batch_size=128, n_epochs=10, gamma=1.0, gae_lambda=0.95,
                    learning_rate=1e-4, ent_coef=0.01, clip_range=0.1),
    },
    "strategy_mixed": {
        # Same as "strategy", but each training episode samples its
        # conditions (6-12 trains, 10-40 step breakdowns, 1-2 breakdowns).
        # Validation stays on the standard conditions for comparability.
        "env": StrategyEnv, "env_kwargs": {"mixed": True},
        "teacher": greedy_action, "teacher_name": "greedy_select",
        "prefix": "rl_strategy_mixed", "timesteps": 60_000, "eval_every": 5_000,
        "warm_start_scenarios": 300, "action_names": NAMES,
        "ppo": dict(n_steps=128, batch_size=128, n_epochs=10, gamma=1.0, gae_lambda=0.95,
                    learning_rate=1e-4, ent_coef=0.01, clip_range=0.1),
    },
    "departure": {
        "env": DispatchEnv, "teacher": hold_if_blocked_action, "teacher_name": "hold_if_blocked",
        "prefix": "rl_dispatcher", "timesteps": 300_000, "eval_every": 20_000,
        "warm_start_scenarios": 60, "action_names": ["Wait", "Dispatch"],
        # gamma and gae_lambda apply per decision, and up to N decisions
        # happen per simulation step. With gamma=0.99 the agent looked only
        # ~12 steps ahead and collapsed to "always Wait". With
        # gae_lambda=0.95 credit faded within ~3 steps, so a Wait was judged
        # only by its immediate cost (the held train gets later) and PPO
        # drifted back to "always Dispatch". gamma=0.999 and gae_lambda=1.0
        # judge each decision by the rest of the episode.
        "ppo": dict(n_steps=512, batch_size=256, n_epochs=10, gamma=0.999, gae_lambda=1.0,
                    learning_rate=1e-4, ent_coef=0.01, clip_range=0.1),
    },
}


def validation_score(model, seeds, env_class):
    """Mean final total delay over `seeds` (lower is better), plus arrivals."""
    env = env_class(seeds, TRAIN_TRAINS, TRAIN_DISRUPTIONS)
    delays, arrived = [], []
    for scenario in seeds:
        obs, _ = env.reset(options={"scenario": scenario})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        delays.append(info["final_delay"])
        arrived.append(sum(a.state.name == "DONE" for a in env.env.agents) / TRAIN_TRAINS)
    return float(np.mean(delays)), float(np.mean(arrived))


def warm_start(model, seeds, task, target=0.9, max_updates=2000):
    """
    Start PPO from a policy that imitates the task's rule-based teacher
    (greedy_select for "strategy", hold_if_blocked for "departure").

    Why: learning from scratch did not work. In the departure task early
    training collapsed to "always Wait", because one random Wait only
    delays a train by a step and its benefit is rarely discovered by
    chance. Castagna et al. (2026) also start their dispatcher by imitating
    a heuristic; we do the same, then let PPO look for improvements.

    1. Policy: trained to copy the teacher on states the teacher visits,
       until its best action matches the teacher on >= 99% of decisions and
       the teacher's action has mean probability >= `target` (not 100%, so
       PPO can still explore other actions).
    2. Value function: trained on the returns the teacher actually got, so
       the first PPO updates use sensible advantage estimates.
    """
    import torch

    gamma = task["ppo"]["gamma"]
    teacher_action = task["teacher"]
    env = task["env"](seeds, TRAIN_TRAINS, TRAIN_DISRUPTIONS, **task.get("env_kwargs", {}))
    states, actions, returns = [], [], []
    for scenario in seeds:
        obs, _ = env.reset(options={"scenario": scenario})
        done, rewards = False, []
        while not done:
            action = teacher_action(obs)
            states.append(obs)
            actions.append(action)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
        g, episode_returns = 0.0, []
        for reward in reversed(rewards):
            g = reward + gamma * g
            episode_returns.append(g)
        returns.extend(reversed(episode_returns))

    states = torch.as_tensor(np.array(states))
    actions = torch.as_tensor(actions, dtype=torch.long)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    policy = model.policy

    def agreement():
        with torch.no_grad():
            probs = policy.get_distribution(states).distribution.probs
            matches = (probs.argmax(dim=1) == actions).float().mean().item()
            p_rule = probs.gather(1, actions[:, None]).mean().item()
        return matches, p_rule

    # 1. Policy: copy the rule until its best action matches the rule on
    #    >= 99% of decisions and the rule's action has mean probability >= target.
    actor = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor, lr=1e-3)
    for _ in range(max_updates):
        matches, p_rule = agreement()
        if matches >= 0.99 and p_rule >= target:
            break
        _, log_prob, _ = policy.evaluate_actions(states, actions)
        loss = -log_prob.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 2. Value function: fit the rule's discounted returns.
    critic = list(policy.mlp_extractor.value_net.parameters()) + list(policy.value_net.parameters())
    optimizer = torch.optim.Adam(critic, lr=1e-3)
    for _ in range(max_updates):
        value_loss = ((policy.predict_values(states).flatten() - returns) ** 2).mean()
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

    matches, p_rule = agreement()
    with torch.no_grad():
        mse = ((policy.predict_values(states).flatten() - returns) ** 2).mean().item()
    r2 = 1 - mse / returns.var().item()
    shares = ", ".join(f"{name} {(actions == i).float().mean().item():.0%}"
                       for i, name in enumerate(task["action_names"]))
    print(f"Warm start from {task['teacher_name']} on {len(states)} decisions ({shares}): "
          f"best action matches teacher {matches:.1%}, mean P(teacher action) {p_rule:.2f}, "
          f"value fit R^2 {r2:.2f}", flush=True)


class KeepBest(BaseCallback):
    """Every `every` steps, score the policy on the validation seeds and
    save it if it has the lowest mean delay so far."""

    def __init__(self, seeds, every, path, log_path, env_class):
        super().__init__()
        self.seeds, self.every, self.path, self.log_path = seeds, every, path, log_path
        self.env_class = env_class
        self.best = float("inf")
        self.next_check = 0
        self.started = time.time()

    def _on_training_start(self):
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "minutes", "val_mean_delay", "val_arrived", "saved"])
        self._check()

    def _on_step(self):
        if self.num_timesteps >= self.next_check:
            self._check()
        return True

    def _check(self):
        delay, arrived = validation_score(self.model, self.seeds, self.env_class)
        saved = delay < self.best
        if saved:
            self.best = delay
            self.model.save(self.path)
        minutes = (time.time() - self.started) / 60
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, round(minutes, 1), round(delay, 2), round(arrived, 4), saved])
        print(f"[{minutes:5.1f} min] steps {self.num_timesteps:>7}  validation delay {delay:7.2f}  "
              f"arrived {100 * arrived:5.1f}%  {'saved' if saved else ''}", flush=True)
        self.next_check = self.num_timesteps + self.every


def main():
    parser = argparse.ArgumentParser(description="Train an RL policy with PPO.")
    parser.add_argument("--task", choices=list(TASKS), default="strategy")
    parser.add_argument("--timesteps", type=int, help="decisions to train on (default depends on task)")
    parser.add_argument("--seed", type=int, default=1, help="random seed for PPO")
    parser.add_argument("--envs", type=int, default=4, help="parallel training environments (CPU cores)")
    parser.add_argument("--train-scenarios", type=int, default=300)
    parser.add_argument("--val-scenarios", type=int, default=30)
    parser.add_argument("--eval-every", type=int, help="decisions between validation checks")
    args = parser.parse_args()
    task = TASKS[args.task]

    base = EnvConfig(num_trains=TRAIN_TRAINS)
    train_seeds = usable_seeds(base, args.train_scenarios, start=TRAIN_SEED_START)
    val_seeds = usable_seeds(base, args.val_scenarios, start=VALIDATION_SEED_START)
    assert max(train_seeds) < VALIDATION_SEED_START, "training and validation seeds overlap"

    os.makedirs("models", exist_ok=True)
    path = f"models/{task['prefix']}_seed{args.seed}"
    print(f"Task: {args.task}")
    print(f"Training scenarios: {len(train_seeds)} (seeds {train_seeds[0]}..{train_seeds[-1]})")
    print(f"Validation scenarios: {len(val_seeds)} (seeds {val_seeds[0]}..{val_seeds[-1]})")
    if task.get("env_kwargs", {}).get("mixed"):
        print(f"Conditions: mixed per episode, trains {StrategyEnv.MIXED_TRAINS}, breakdown steps "
              f"{StrategyEnv.MIXED_DURATIONS}, breakdowns {StrategyEnv.MIXED_COUNTS}; "
              f"validation on {TRAIN_TRAINS} trains, {TRAIN_DISRUPTIONS}", flush=True)
    else:
        print(f"Conditions: {TRAIN_TRAINS} trains, {TRAIN_DISRUPTIONS}", flush=True)

    vec_env = make_vec_env(
        task["env"], n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
        env_kwargs={"seeds": train_seeds, "num_trains": TRAIN_TRAINS, "disruptions": TRAIN_DISRUPTIONS,
                    **task.get("env_kwargs", {})},
    )
    model = PPO(
        "MlpPolicy", vec_env, seed=args.seed, device="cpu", verbose=0,
        policy_kwargs={"net_arch": [64, 64]}, **task["ppo"],
    )
    warm_start(model, train_seeds[:task["warm_start_scenarios"]], task)
    callback = KeepBest(val_seeds, args.eval_every or task["eval_every"],
                        path + ".zip", path + "_log.csv", task["env"])
    model.learn(total_timesteps=args.timesteps or task["timesteps"], callback=callback)
    callback._check()
    vec_env.close()
    print(f"Best validation mean delay: {callback.best:.2f}  ->  {path}.zip")


if __name__ == "__main__":
    main()
