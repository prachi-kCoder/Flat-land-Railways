# ============================================================
# Analyse the robustness study (output of run_robustness.py)
# ============================================================
#
#   python analyse_robustness.py [--csv outputs/robustness.csv]
#
# Writes:
#   outputs/robustness_summary.csv   mean and 95% interval per policy/level
#   outputs/robustness_paired.csv    each RL model vs each baseline, paired
#   outputs/robustness_total_delay.png, outputs/robustness_arrived.png
#
# Statistics:
#   - 95% interval of a mean: percentile bootstrap over test scenarios
#     (5000 resamples, fixed random seed, so results are repeatable).
#   - Paired test: every policy runs the same scenarios, so we compare
#     per scenario with the Wilcoxon signed-rank test (two-sided). Ties
#     (identical results) carry no information and are dropped, which is
#     scipy's default ("wilcox" zero handling). p < 0.05 is reported as
#     significant; with many comparisons, treat isolated p-values near
#     0.05 with care.
# ============================================================

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SHIFT_LABELS = {
    "density": "Number of trains (trained on 8)",
    "duration": "Breakdown length in steps (trained on 20)",
    "simultaneous": "Simultaneous breakdowns (trained on 1)",
}
BASELINES = ["fcfs", "priority", "edf", "spf", "greedy_select", "release_all", "hold_if_blocked"]


def bootstrap_ci(values, rng, resamples=5000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.mean(), values.mean()
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarise(df):
    rng = np.random.default_rng(0)
    rows = []
    for (shift, level, policy), g in df.groupby(["shift", "level", "policy"]):
        for metric in ["total_delay", "throughput", "deadlocked", "cancelled"]:
            low, high = bootstrap_ci(g[metric], rng)
            rows.append({"shift": shift, "level": level, "policy": policy, "metric": metric,
                         "mean": g[metric].mean(), "ci_low": low, "ci_high": high, "n": len(g)})
    return pd.DataFrame(rows)


def paired(df, rl_policies):
    rows = []
    for (shift, level), g in df.groupby(["shift", "level"]):
        by_policy = {p: s.set_index("seed")["total_delay"] for p, s in g.groupby("policy")}
        for rl in rl_policies:
            for base in BASELINES:
                a, b = by_policy[rl].align(by_policy[base], join="inner")
                diff = (a - b).to_numpy()   # negative = RL has less delay
                nonzero = diff[diff != 0]
                p = wilcoxon(nonzero).pvalue if len(nonzero) >= 1 else 1.0
                rows.append({
                    "shift": shift, "level": level, "rl": rl, "baseline": base, "n": len(diff),
                    "mean_delay_rl": a.mean(), "mean_delay_baseline": b.mean(),
                    "mean_difference": diff.mean(),
                    "rl_better": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
                    "rl_worse": int((diff > 0).sum()), "p_value": p,
                    "significant": p < 0.05,
                })
    return pd.DataFrame(rows)


def plot(summary, metric, ylabel, path):
    shifts = list(SHIFT_LABELS)
    fig, axes = plt.subplots(1, len(shifts), figsize=(5 * len(shifts), 4), sharey=False)
    for ax, shift in zip(axes, shifts):
        part = summary[(summary["shift"] == shift) & (summary["metric"] == metric)]
        for policy, g in part.groupby("policy"):
            g = g.sort_values("level")
            style = "-" if policy.startswith("rl") else "--"
            ax.plot(g["level"], g["mean"], style, marker="o", label=policy)
            ax.fill_between(g["level"], g["ci_low"], g["ci_high"], alpha=0.15)
        ax.set_xlabel(SHIFT_LABELS[shift])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{ylabel} under shifted conditions (mean and 95% interval over test scenarios)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/robustness.csv")
    args = parser.parse_args()
    out = os.path.dirname(args.csv) or "."

    df = pd.read_csv(args.csv)
    rl_policies = sorted(p for p in df["policy"].unique() if p.startswith("rl"))

    summary = summarise(df)
    summary.to_csv(os.path.join(out, "robustness_summary.csv"), index=False)
    plot(summary, "total_delay", "Total delay (steps)", os.path.join(out, "robustness_total_delay.png"))
    summary_pct = summary.copy()
    mask = summary_pct["metric"] == "throughput"
    summary_pct.loc[mask, ["mean", "ci_low", "ci_high"]] *= 100
    plot(summary_pct, "throughput", "Trains arrived (%)", os.path.join(out, "robustness_arrived.png"))

    delay = summary[summary["metric"] == "total_delay"].pivot_table(
        index=["shift", "level"], columns="policy", values="mean")
    pd.set_option("display.width", 200)
    print("Mean total delay per level (lower is better):")
    print(delay.round(1).to_string())

    if rl_policies:
        tests = paired(df, rl_policies)
        tests.to_csv(os.path.join(out, "robustness_paired.csv"), index=False)
        print("\nRL vs baselines, paired per scenario (difference < 0 means RL has less delay):")
        cols = ["shift", "level", "rl", "baseline", "mean_difference", "rl_better", "ties", "rl_worse", "p_value"]
        print(tests[cols].round(3).to_string(index=False))
    print(f"\nSaved summaries and plots in {out}/")


if __name__ == "__main__":
    main()
