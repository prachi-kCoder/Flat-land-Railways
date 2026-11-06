"""Train a compact MAPPO dispatcher without touching the existing PPO scripts.

Quick smoke run:
    python train_mappo.py --episodes 2 --trains 4 --output models/mappo_smoke.pt

Small exploratory run:
    python train_mappo.py --episodes 20 --trains 8 --output models/mappo_seed1.pt
"""
import argparse
import random
import numpy as np
import torch
from railsim.disruptions import DisruptionSpec
from railsim.mappo import MAPPO, generalized_advantage, joint_state
from railsim.rl_dispatch import DispatchEnv


def collect_episode(env, learner, gamma=1.0, gae_lambda=0.95):
    obs, _ = env.reset()
    records = {key: [] for key in
               ("obs", "states", "actions", "log_probs", "values", "rewards", "dones")}
    done = False
    while not done:
        local_obs = obs.copy()
        state = joint_state(env)
        action, log_prob, value = learner.act(local_obs, state)
        next_obs, reward, done, _, info = env.step(action)
        records["obs"].append(local_obs)
        records["states"].append(state)
        records["actions"].append(action)
        records["log_probs"].append(log_prob)
        records["values"].append(value)
        records["rewards"].append(float(reward))
        records["dones"].append(bool(done))
        obs = next_obs
    advantages, returns = generalized_advantage(
        records["rewards"], records["values"], records["dones"],
        gamma=gamma, gae_lambda=gae_lambda,
    )
    records["advantages"] = advantages
    records["returns"] = returns
    return records, info.get("final_delay")


def merge_batches(batches):
    keys = ("obs", "states", "actions", "log_probs", "returns", "advantages")
    return {key: np.concatenate([np.asarray(batch[key]) for batch in batches], axis=0)
            for key in keys}


def main():
    parser = argparse.ArgumentParser(description="Train a shared-actor, centralized-critic MAPPO policy.")
    parser.add_argument("--episodes", type=int, default=8, help="number of training episodes")
    parser.add_argument("--trains", type=int, default=4, help="number of Flatland trains (small by default)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="models/mappo_seed1.pt")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.episodes < 1 or args.trains < 2:
        parser.error("--episodes must be >= 1 and --trains must be >= 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    seeds = list(range(1000 + args.seed * 100, 1000 + args.seed * 100 + args.episodes))
    env = DispatchEnv(seeds, num_trains=args.trains,
                      disruptions=DisruptionSpec(count=1, duration=20))
    learner = MAPPO(args.trains, learning_rate=args.learning_rate)
    episode_delays = []
    batch = []
    for episode in range(args.episodes):
        records, final_delay = collect_episode(env, learner)
        batch.append(records)
        if final_delay is not None:
            episode_delays.append(float(final_delay))
        if len(batch) >= min(4, args.episodes) or episode == args.episodes - 1:
            metrics = learner.update(merge_batches(batch), epochs=args.epochs,
                                     batch_size=args.batch_size)
            batch.clear()
            mean_delay = float(np.mean(episode_delays)) if episode_delays else float("nan")
            print(f"episode {episode + 1}/{args.episodes} | "
                  f"mean final delay {mean_delay:.2f} | "
                  f"actor loss {metrics['actor_loss']:.4f} | "
                  f"critic loss {metrics['critic_loss']:.4f}", flush=True)
    learner.save(args.output)
    print(f"Saved MAPPO actor + critic checkpoint: {args.output}")
    print("This is a prototype training run; evaluate on held-out seeds before drawing conclusions.")


if __name__ == "__main__":
    main()
