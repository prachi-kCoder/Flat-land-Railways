"""Evaluate a saved MAPPO actor on one Flatland disruption scenario."""
import argparse
from flatland.envs.step_utils.states import TrainState
from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env
from railsim.mappo import MAPPODispatcher
from railsim.metrics import network_delay
from railsim.train_status import find_deadlocked_trains


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MAPPO dispatcher.")
    parser.add_argument("--model", default="models/mappo_seed1.pt")
    parser.add_argument("--trains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disruptions", type=int, default=1)
    parser.add_argument("--duration", type=int, default=20)
    args = parser.parse_args()

    MAPPODispatcher.model_path = args.model
    MAPPODispatcher._actor = None
    MAPPODispatcher._loaded_from = None
    env, _, _ = make_env(EnvConfig(num_trains=args.trains, seed=args.seed),
                         DisruptionSpec(count=args.disruptions, duration=args.duration))
    policy = MAPPODispatcher()
    policy.reset(env)
    done = False
    while not done:
        _, _, dones, _ = env.step(policy.act(env))
        done = dones["__all__"]
    arrived = sum(agent.state == TrainState.DONE for agent in env.agents)
    deadlocked = len(find_deadlocked_trains(env))
    print(f"policy=mappo seed={args.seed} trains={args.trains}")
    print(f"final_network_delay={network_delay(env)}")
    print(f"arrived={arrived}/{args.trains} ({arrived / args.trains:.1%})")
    print(f"deadlocked={deadlocked}")


if __name__ == "__main__":
    main()
