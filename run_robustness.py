# ============================================================
# Robustness study: how do the policies cope when test
# conditions move away from the training conditions?
# ============================================================
#
# Training conditions (what the RL policy was trained on):
#   8 trains, 1 breakdown of 20 steps.
#
# Each "shift" changes ONE factor and keeps the others at their
# training value:
#   density       number of trains: 8, 12, 16, 20, 24   (+0% .. +200%)
#   duration      breakdown length: 10, 20, 40, 60, 100 steps
#   simultaneous  breakdowns starting at the same step: 1, 2, 3, 4
#
# Every policy is run on the same test scenarios (seeds from 0) for
# every level, so results are paired and can be compared scenario by
# scenario. The RL policies are frozen: they are never retrained.
#
# Examples:
#   python run_robustness.py --seeds 30
#   python run_robustness.py --seeds 30 --models models/rl_strategy_seed1.zip models/rl_strategy_seed2.zip
#
# Writes:
#   outputs/robustness.csv          one row per (shift, level, policy, seed)
#   outputs/robustness_summary.csv  means, 95% intervals, paired tests
#   outputs/robustness_<metric>.png degradation curves
# ============================================================

import argparse
import csv
import os
from multiprocessing import Pool

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, usable_seeds
from railsim.policies import FCFSDispatcher, PriorityDispatcher
from railsim.rl_dispatch import HoldIfBlocked, ReleaseAll, RLDispatcher
from railsim.runner import run_episode
from railsim.strategy import EDFDispatcher, GreedySelect, RLStrategy, SPFDispatcher

TRAINING = {"trains": 8, "count": 1, "duration": 20}
SHIFTS = {
    "density": ("trains", [8, 12, 16, 20, 24]),
    "duration": ("duration", [10, 20, 40, 60, 100]),
    "simultaneous": ("count", [1, 2, 3, 4]),
}
BASELINES = [
    FCFSDispatcher, PriorityDispatcher, EDFDispatcher, SPFDispatcher,   # fixed ordering rules
    GreedySelect,                                                       # rule selection, no learning
    ReleaseAll, HoldIfBlocked,                                          # departure control, no learning
]


def rl_policy(path):
    """A frozen RL policy class bound to one model file. The file name says
    which task it was trained for: rl_strategy_* or rl_dispatcher_*."""
    stem = os.path.splitext(os.path.basename(path))[0]
    base = RLStrategy if stem.startswith("rl_strategy") else RLDispatcher
    return type(stem, (base,), {"name": stem, "model_path": path, "_model": None, "_loaded_from": None})


def conditions(shift, level):
    values = dict(TRAINING)
    values[SHIFTS[shift][0]] = level
    spec = DisruptionSpec(
        count=values["count"], duration=values["duration"],
        simultaneous=shift == "simultaneous" and values["count"] > 1,
    )
    return values["trains"], spec


def policy_from_key(key):
    """Jobs carry a policy name (baselines) or "model:<path>" (RL), because
    classes created at run time cannot be sent to worker processes."""
    if key.startswith("model:"):
        return rl_policy(key[len("model:"):])
    return {p.name: p for p in BASELINES}[key]


def run_one(job):
    shift, level, policy_key, seed = job
    policy_class = policy_from_key(policy_key)
    import torch
    torch.set_num_threads(1)   # one process per core; avoid oversubscription
    trains, spec = conditions(shift, level)
    result = run_episode(EnvConfig(num_trains=trains, seed=seed), policy_class, spec)
    row = {"shift": shift, "level": level}
    row.update(result.as_row())
    return row


def main():
    parser = argparse.ArgumentParser(description="Degradation of frozen policies under shifted conditions.")
    parser.add_argument("--seeds", type=int, default=30, help="test scenarios per level (default 30)")
    parser.add_argument("--models", nargs="*", default=[], help="trained RL models to include")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="outputs/robustness.csv")
    args = parser.parse_args()

    for path in args.models:
        if not os.path.exists(path):
            parser.error(f"model not found: {path}")
    policies = [p.name for p in BASELINES] + ["model:" + p for p in args.models]
    seeds = usable_seeds(EnvConfig(num_trains=TRAINING["trains"]), args.seeds, start=0)
    jobs = [
        (shift, level, policy, seed)
        for shift, (_factor, levels) in SHIFTS.items()
        for level in levels
        for policy in policies
        for seed in seeds
    ]
    print(f"{len(jobs)} runs: {len(SHIFTS)} shifts, {len(policies)} policies, {len(seeds)} test scenarios "
          f"(seeds {seeds[0]}..{seeds[-1]})", flush=True)

    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs done", flush=True)

    rows.sort(key=lambda r: (r["shift"], r["level"], r["policy"], r["seed"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out}")
    print("Next: python analyse_robustness.py")


if __name__ == "__main__":
    main()
