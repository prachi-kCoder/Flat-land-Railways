# ============================================================
# Run policies on a batch of scenarios and save the results
# ============================================================
#
# Every policy sees exactly the same scenarios (network, timetable
# and disruptions) for each seed.
#
# Examples:
#   python run_experiment.py
#   python run_experiment.py --trains 8 --seeds 20 --disruptions 2 --duration 30
#   python run_experiment.py --disruptions 3 --simultaneous --out outputs/sim.csv
#
# Writes one CSV row per (policy, seed) and prints averages.
# ============================================================

import argparse
import csv
import os
import statistics

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, usable_seeds
from railsim.policies import POLICIES
from railsim.rl_dispatch import ReleaseAll, RLDispatcher

ALL_POLICIES = {**POLICIES, ReleaseAll.name: ReleaseAll, RLDispatcher.name: RLDispatcher}
DEFAULT_POLICIES = list(POLICIES) + [ReleaseAll.name]   # add "rl" with --policies ... rl
from railsim.runner import run_episode


parser = argparse.ArgumentParser(description="Compare dispatching policies on the same scenarios.")
parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES, choices=list(ALL_POLICIES))
parser.add_argument("--model", default=RLDispatcher.model_path, help="trained model for the 'rl' policy")
parser.add_argument("--trains", type=int, default=5, help="number of trains (default 5)")
parser.add_argument("--seeds", type=int, default=10, help="number of scenarios (default 10)")
parser.add_argument("--first-seed", type=int, default=0, help="search for usable seeds from here (default 0)")
parser.add_argument("--disruptions", type=int, default=1, help="breakdowns per scenario (default 1, 0 = none)")
parser.add_argument("--duration", type=int, default=10, help="steps each breakdown lasts (default 10)")
parser.add_argument("--simultaneous", action="store_true", help="all breakdowns start on the same step")
parser.add_argument("--out", default="outputs/results.csv", help="CSV file to write (default outputs/results.csv)")
args = parser.parse_args()
RLDispatcher.model_path = args.model


base = EnvConfig(num_trains=args.trains)
seeds = usable_seeds(base, args.seeds, start=args.first_seed)
spec = (
    DisruptionSpec(count=args.disruptions, duration=args.duration, simultaneous=args.simultaneous)
    if args.disruptions > 0 else None
)

print(f"Trains: {args.trains}   Scenarios: {len(seeds)} (seeds {seeds[0]}..{seeds[-1]})")
print(f"Disruptions: {spec if spec else 'none'}\n")

results = []
for policy_name in args.policies:
    for seed in seeds:
        config = EnvConfig(num_trains=args.trains, seed=seed)
        results.append(run_episode(config, ALL_POLICIES[policy_name], spec))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", newline="") as f:
    rows = [r.as_row() for r in results]
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def mean(values):
    return statistics.mean(values) if values else float("nan")


def spread(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


header = (
    f"{'policy':<16}{'arrived %':>12}{'total delay':>18}{'deadlocked':>12}{'cancelled':>11}"
    f"{'recovered':>12}{'recovery steps':>16}{'ms/step':>10}"
)
print(header)
print("-" * len(header))
for policy_name in args.policies:
    rs = [r for r in results if r.policy == policy_name]
    arrived = [100 * r.throughput for r in rs]
    delay = [r.total_delay for r in rs]
    deadlocked = [r.deadlocked for r in rs]
    recoveries = [r.recovery_time for r in rs if r.recovery_time is not None]
    recovered = f"{len(recoveries)}/{len(rs)}" if spec else "-"
    recovery = f"{mean(recoveries):.1f}" if recoveries else "-"
    print(
        f"{policy_name:<16}{mean(arrived):>11.1f}%"
        f"{mean(delay):>11.1f} ± {spread(delay):<5.1f}"
        f"{mean(deadlocked):>12.2f}{mean([r.cancelled for r in rs]):>11.2f}{recovered:>12}{recovery:>16}"
        f"{mean([r.decision_ms_mean for r in rs]):>10.2f}"
    )

print(f"\nPer-scenario results saved to {args.out}")
print("Averages are over scenarios; ± is the standard deviation.")
print("Recovered: runs where the extra delay caused by the breakdowns stopped growing")
print("before the episode ended. Recovery steps: average over those runs.")
