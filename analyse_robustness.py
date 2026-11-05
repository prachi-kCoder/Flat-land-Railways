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
#   outputs/recovery_summary.csv     extra delay caused by the breakdowns,
#                                    how runs ended (no effect / recovered /
#                                    lasted to the end), median recovery time,
#                                    share of planned breakdowns that happened
#   outputs/recovery_paired.csv      extra delay, each RL model vs each baseline
#   outputs/robustness_recovery.png  recovery plots
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


# Shown in the plots (all policies stay in the CSV files). Interval bands
# only for RL and greedy_select, to keep the figure readable.
PLOTTED = ["fcfs", "priority", "hold_if_blocked", "greedy_select"]
STYLES = {"fcfs": "C1", "priority": "C4", "hold_if_blocked": "C3", "greedy_select": "C2"}


def plot(summary, metric, ylabel, path):
    shifts = list(SHIFT_LABELS)
    rl = sorted(p for p in summary["policy"].unique() if p.startswith("rl"))
    fig, axes = plt.subplots(1, len(shifts), figsize=(5 * len(shifts), 4.2))
    for ax, shift in zip(axes, shifts):
        part = summary[(summary["shift"] == shift) & (summary["metric"] == metric)]
        for policy in PLOTTED + rl:
            g = part[part["policy"] == policy].sort_values("level")
            if g.empty:
                continue
            is_rl = policy.startswith("rl")
            color = STYLES.get(policy, "C0" if policy == (rl[0] if rl else None) else "C7")
            ax.plot(g["level"], g["mean"], "-" if is_rl else "--", marker="o" if is_rl else "s",
                    color=color, linewidth=2 if is_rl else 1.3, markersize=5, label=policy)
            if is_rl or policy == "greedy_select":
                ax.fill_between(g["level"], g["ci_low"], g["ci_high"], color=color, alpha=0.12)
        ax.set_xticks(sorted(part["level"].unique()))
        ax.set_xlabel(SHIFT_LABELS[shift])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle(f"{ylabel} under shifted conditions (mean over test scenarios; bands = 95% interval)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def recovery_summary(df):
    """
    Per shift, level and policy:
      breakdowns_happened  share of planned breakdowns that actually hit a
                           train (a breakdown planned for a train that has
                           already arrived never happens)
      extra_delay          total delay minus the same policy's delay on the
                           same scenario without breakdowns (mean, 95% interval)
      no_effect_pct        runs where the breakdowns caused no extra delay
      recovered_pct        runs where the extra delay stopped changing
                           before the episode ended
      lasted_to_end_pct    runs where it was still changing at the end
      median_recovery      median recovery time (steps) of recovered runs
    """
    rng = np.random.default_rng(1)
    rows = []
    for (shift, level, policy), g in df.groupby(["shift", "level", "policy"]):
        low, high = bootstrap_ci(g["extra_delay"], rng)
        rec = g["recovery_time"]
        recovered = rec[rec > 0]
        rows.append({
            "shift": shift, "level": level, "policy": policy, "n": len(g),
            "breakdowns_happened": ((g["disruptions_on_map"] + g["disruptions_off_map"]).sum()
                                    / g["disruption_count"].sum()),
            "extra_delay": g["extra_delay"].mean(), "extra_ci_low": low, "extra_ci_high": high,
            "no_effect_pct": 100 * (rec == 0).mean(),
            "recovered_pct": 100 * (rec > 0).mean(),
            "lasted_to_end_pct": 100 * rec.isna().mean(),
            "median_recovery": recovered.median() if len(recovered) else np.nan,
        })
    return pd.DataFrame(rows)


def paired_extra_delay(df, rl_policies):
    rows = []
    for (shift, level), g in df.groupby(["shift", "level"]):
        by_policy = {p: s.set_index("seed")["extra_delay"] for p, s in g.groupby("policy")}
        for rl in rl_policies:
            for base in BASELINES:
                a, b = by_policy[rl].align(by_policy[base], join="inner")
                diff = (a - b).to_numpy()
                nonzero = diff[diff != 0]
                p = wilcoxon(nonzero).pvalue if len(nonzero) >= 1 else 1.0
                rows.append({"shift": shift, "level": level, "rl": rl, "baseline": base,
                             "mean_extra_rl": a.mean(), "mean_extra_baseline": b.mean(),
                             "mean_difference": diff.mean(), "rl_better": int((diff < 0).sum()),
                             "ties": int((diff == 0).sum()), "rl_worse": int((diff > 0).sum()),
                             "p_value": p, "significant": p < 0.05})
    return pd.DataFrame(rows)


def plot_recovery(rec, path):
    shifts = list(SHIFT_LABELS)
    rl = sorted(p for p in rec["policy"].unique() if p.startswith("rl"))
    fig, axes = plt.subplots(2, len(shifts), figsize=(5 * len(shifts), 7.5))
    for col, shift in enumerate(shifts):
        part = rec[rec["shift"] == shift]
        for policy in PLOTTED + rl:
            g = part[part["policy"] == policy].sort_values("level")
            if g.empty:
                continue
            is_rl = policy.startswith("rl")
            color = STYLES.get(policy, "C0" if policy == (rl[0] if rl else None) else "C7")
            kw = dict(color=color, marker="o" if is_rl else "s", markersize=5,
                      linewidth=2 if is_rl else 1.3, linestyle="-" if is_rl else "--", label=policy)
            axes[0, col].plot(g["level"], g["extra_delay"], **kw)
            if is_rl or policy == "greedy_select":
                axes[0, col].fill_between(g["level"], g["extra_ci_low"], g["extra_ci_high"], color=color, alpha=0.12)
            axes[1, col].plot(g["level"], g["lasted_to_end_pct"], **kw)
        for row in (0, 1):
            axes[row, col].set_xticks(sorted(part["level"].unique()))
            axes[row, col].grid(alpha=0.3)
        axes[0, col].axhline(0, color="black", linewidth=0.8)
        axes[0, col].set_ylabel("Extra delay from breakdowns (steps)")
        axes[1, col].set_ylabel("Runs where effect lasted to the end (%)")
        axes[1, col].set_xlabel(SHIFT_LABELS[shift])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle("Recovery from breakdowns: damage done (top) and how often it was still spreading "
                 "when the episode ended (bottom)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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
    if "extra_delay" in df.columns and df["extra_delay"].notna().any():
        rec = recovery_summary(df)
        rec.to_csv(os.path.join(out, "recovery_summary.csv"), index=False)
        plot_recovery(rec, os.path.join(out, "robustness_recovery.png"))
        print("\nRecovery: extra delay caused by the breakdowns (mean; lower is better):")
        print(rec.pivot_table(index=["shift", "level"], columns="policy", values="extra_delay").round(1).to_string())
        print("\nRecovery: runs where the effect lasted until the episode ended (%):")
        print(rec.pivot_table(index=["shift", "level"], columns="policy", values="lasted_to_end_pct").round(0).to_string())
        print("\nRecovery: median recovery time of recovered runs (steps):")
        print(rec.pivot_table(index=["shift", "level"], columns="policy", values="median_recovery").round(1).to_string())
        print("\nShare of planned breakdowns that actually happened:")
        print(rec.pivot_table(index=["shift", "level"], columns="policy", values="breakdowns_happened").round(2).to_string())
        if rl_policies:
            extra_tests = paired_extra_delay(df, rl_policies)
            extra_tests.to_csv(os.path.join(out, "recovery_paired.csv"), index=False)
    print(f"\nSaved summaries and plots in {out}/")


if __name__ == "__main__":
    main()
