# ============================================================
# Train the RL dispatcher (PPO, Stable-Baselines3)
# ============================================================
#
# Training conditions are held fixed (the "training distribution"):
#   8 trains, 1 breakdown of 20 steps per episode.
# Everything else is varied only at test time (run_experiment.py).
#
# Scenario seeds are kept apart:
#   training    seeds from 1000 upwards
#   validation  seeds from 5000 upwards (used to keep the best checkpoint)
#   testing     seeds from 0 upwards   (run_experiment.py default)
#
# Example:
#   python train_rl.py --timesteps 200000 --seed 1
# Writes:
#   models/rl_dispatcher_seed1.zip        best checkpoint on validation
#   models/rl_dispatcher_seed1_log.csv    validation score during training
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
from railsim.rl_dispatch import DispatchEnv

TRAIN_TRAINS = 8
TRAIN_DISRUPTIONS = DisruptionSpec(count=1, duration=20)
TRAIN_SEED_START = 1000
VALIDATION_SEED_START = 5000


def validation_score(model, seeds):
    """Mean final total delay over `seeds` (lower is better), plus arrivals."""
    env = DispatchEnv(seeds, TRAIN_TRAINS, TRAIN_DISRUPTIONS)
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


def warm_start(model, seeds, target=0.9, max_updates=500):
    """
    Teach the untrained policy to mostly Dispatch before PPO starts.

    Without this, early random exploration quickly settles on "always
    Wait": the penalty for holding a train only appears many decisions
    later, when the train starts running late. Castagna et al. (2026)
    likewise start their dispatcher by imitating a simple heuristic.

    States come from running "always Dispatch" (the release_all policy) on
    the training scenarios. We stop as soon as the policy dispatches with
    probability >= `target` on those states, so it keeps some chance of
    choosing Wait and PPO can still learn when waiting helps.
    """
    import torch

    env = DispatchEnv(seeds, TRAIN_TRAINS, TRAIN_DISRUPTIONS)
    states = []
    for scenario in seeds:
        obs, _ = env.reset(options={"scenario": scenario})
        done = False
        while not done:
            states.append(obs)
            obs, _, done, _, _ = env.step(1)
    states = torch.as_tensor(np.array(states))
    dispatch = torch.ones(len(states), dtype=torch.long)

    policy = model.policy
    for update in range(max_updates):
        probability = policy.get_distribution(states).distribution.probs[:, 1].mean().item()
        if probability >= target:
            break
        _, log_prob, _ = policy.evaluate_actions(states, dispatch)
        loss = -log_prob.mean()
        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()
    print(f"Warm start: {len(states)} states, {update} updates, "
          f"mean P(Dispatch) = {probability:.3f}")


class KeepBest(BaseCallback):
    """Every `every` steps, score the policy on the validation seeds and
    save it if it has the lowest mean delay so far."""

    def __init__(self, seeds, every, path, log_path):
        super().__init__()
        self.seeds, self.every, self.path, self.log_path = seeds, every, path, log_path
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
        delay, arrived = validation_score(self.model, self.seeds)
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
    parser = argparse.ArgumentParser(description="Train the RL dispatcher with PPO.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="dispatch decisions to train on")
    parser.add_argument("--seed", type=int, default=1, help="random seed for PPO")
    parser.add_argument("--envs", type=int, default=4, help="parallel training environments (CPU cores)")
    parser.add_argument("--train-scenarios", type=int, default=300)
    parser.add_argument("--val-scenarios", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=10_000)
    args = parser.parse_args()

    base = EnvConfig(num_trains=TRAIN_TRAINS)
    train_seeds = usable_seeds(base, args.train_scenarios, start=TRAIN_SEED_START)
    val_seeds = usable_seeds(base, args.val_scenarios, start=VALIDATION_SEED_START)
    assert max(train_seeds) < VALIDATION_SEED_START, "training and validation seeds overlap"

    os.makedirs("models", exist_ok=True)
    path = f"models/rl_dispatcher_seed{args.seed}"
    print(f"Training scenarios: {len(train_seeds)} (seeds {train_seeds[0]}..{train_seeds[-1]})")
    print(f"Validation scenarios: {len(val_seeds)} (seeds {val_seeds[0]}..{val_seeds[-1]})")
    print(f"Conditions: {TRAIN_TRAINS} trains, {TRAIN_DISRUPTIONS}")

    vec_env = make_vec_env(
        DispatchEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
        env_kwargs={"seeds": train_seeds, "num_trains": TRAIN_TRAINS, "disruptions": TRAIN_DISRUPTIONS},
    )
    model = PPO(
        "MlpPolicy", vec_env, seed=args.seed, device="cpu", verbose=0,
        # gamma is per decision. Up to N decisions happen per simulation
        # step, so 0.99 would look only ~12 steps ahead; 0.999 covers an
        # episode (~100 steps).
        n_steps=512, batch_size=256, n_epochs=10, gamma=0.999, gae_lambda=0.95,
        learning_rate=3e-4, ent_coef=0.01, clip_range=0.2,
        policy_kwargs={"net_arch": [64, 64]},
    )
    warm_start(model, train_seeds[:50])
    callback = KeepBest(val_seeds, args.eval_every, path + ".zip", path + "_log.csv")
    model.learn(total_timesteps=args.timesteps, callback=callback)
    callback._check()
    vec_env.close()
    print(f"Best validation mean delay: {callback.best:.2f}  ->  {path}.zip")


if __name__ == "__main__":
    main()
