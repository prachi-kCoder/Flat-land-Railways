# ============================================================
# Test the frozen policies with breakdown lengths taken from
# real railway data (Fintraffic Digitraffic, Finland, CC BY 4.0)
# ============================================================
#
# Breakdown lengths are drawn from an empirical sample of delay
# episodes (results/realdata/minutes_*.csv, in minutes), converted
# to Flatland steps with --minutes-per-step. Flatland has no time
# unit, so this conversion is a modelling choice: see
# results/realdata/conversion_options.txt and the README.
#
# Scenario types (each run at every --trains value):
#   single     1 breakdown, length from the real sample
#   multi      --multi breakdowns at random times, lengths from the sample
#   realistic  number of breakdowns ~ Binomial(trains, rate) with the rate
#              measured from the same data (--rate-per-train), lengths
#              from the sample
#
# Example:
#   python run_realdata.py --minutes-per-step 3.77 --seeds 20 \
#       --models results/models/rl_strategy_seed1.zip results/models/rl_strategy_seed2.zip
#
# Writes outputs/realdata.csv (same columns as robustness.csv).
# ============================================================

import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, usable_seeds
from railsim.runner import run_episode
from run_robustness import BASELINES, policy_from_key


def load_steps(path, minutes_per_step):
    """Real delay episodes (minutes) -> breakdown lengths in steps (>= 1)."""
    minutes = np.loadtxt(path, skiprows=1, ndmin=1)
    return tuple(int(s) for s in np.maximum(1, np.round(minutes / minutes_per_step)))


def run_one(job):
    mode, trains, policy_key, seed, spec = job
    import torch
    torch.set_num_threads(1)
    result = run_episode(EnvConfig(num_trains=trains, seed=seed), policy_from_key(policy_key), spec)
    row = {"shift": f"realdata_{mode}", "level": trains}
    row.update(result.as_row())
    return row


def main():
    parser = argparse.ArgumentParser(description="Policies under breakdown lengths from real data.")
    parser.add_argument("--minutes-per-step", type=float, required=True,
                        help="conversion of real minutes to Flatland steps (no default: a modelling choice)")
    parser.add_argument("--minutes-file", default="results/realdata/minutes_failure_episodes_ge3.csv")
    parser.add_argument("--rate-per-train", type=float,
                        help="chance a train breaks down in an episode (default: measured, from "
                             "results/realdata/sample_summary.json)")
    parser.add_argument("--trains", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--modes", nargs="+", default=["single", "multi", "realistic"],
                        choices=["single", "multi", "realistic"])
    parser.add_argument("--multi", type=int, default=3, help="breakdowns in 'multi' scenarios")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--baselines", nargs="*", help="subset of baseline names (default: all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="outputs/realdata.csv")
    args = parser.parse_args()

    steps = load_steps(args.minutes_file, args.minutes_per_step)
    rate = args.rate_per_train
    if rate is None:
        summary = json.load(open("results/realdata/sample_summary.json"))
        rate = summary["failure_episodes_ge3_per_train_run"]
    q = np.percentile(steps, [50, 90, 99])
    print(f"Real sample: {len(steps)} episodes from {args.minutes_file}; at {args.minutes_per_step} min/step "
          f"lengths p50/p90/p99 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} steps, max {max(steps)}; "
          f"rate per train {rate:.4f}", flush=True)

    specs = {
        "single": DisruptionSpec(count=1, durations=steps),
        "multi": DisruptionSpec(count=args.multi, durations=steps),
        "realistic": DisruptionSpec(durations=steps, rate_per_train=rate),
    }
    names = args.baselines or [p.name for p in BASELINES]
    for path in args.models:
        if not os.path.exists(path):
            parser.error(f"model not found: {path}")
    policies = names + ["model:" + p for p in args.models]
    seeds = usable_seeds(EnvConfig(num_trains=8), args.seeds, start=0)
    jobs = [(mode, trains, policy, seed, specs[mode])
            for mode in args.modes for trains in args.trains for policy in policies for seed in seeds]
    print(f"{len(jobs)} runs: modes {args.modes}, trains {args.trains}, {len(policies)} policies, "
          f"{len(seeds)} test scenarios (seeds {seeds[0]}..{seeds[-1]})", flush=True)

    rows = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4), 1):
                rows.append(row)
                if i % 200 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} runs done", flush=True)
    else:
        rows = [run_one(job) for job in jobs]

    rows.sort(key=lambda r: (r["shift"], r["level"], r["policy"], r["seed"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out}\nNext: python analyse_realdata.py --csv {args.out}")


if __name__ == "__main__":
    main()
