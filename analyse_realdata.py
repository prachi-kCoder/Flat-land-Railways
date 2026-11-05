# ============================================================
# Analyse the real-data breakdown test (output of run_realdata.py)
# ============================================================
#
#   python analyse_realdata.py [--csv outputs/realdata.csv]
#
# Uses the same statistics as analyse_robustness.py (bootstrap 95%
# intervals, paired Wilcoxon tests per scenario, extra delay caused
# by the breakdowns). Writes next to the CSV:
#   realdata_summary.csv, realdata_paired.csv, realdata_recovery.csv,
#   realdata_recovery_paired.csv, realdata_total_delay.png
# ============================================================

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analyse_robustness import PLOTTED, RL_COLOURS, STYLES, paired, paired_extra_delay, recovery_summary, summarise

MODE_LABELS = {
    "realdata_single": "1 breakdown",
    "realdata_multi": "Several breakdowns",
    "realdata_realistic": "Real breakdown frequency",
}


def plot(summary, path):
    modes = [m for m in MODE_LABELS if m in summary["shift"].unique()]
    rl = sorted(p for p in summary["policy"].unique() if p.startswith("rl"))
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4.2), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        part = summary[(summary["shift"] == mode) & (summary["metric"] == "total_delay")]
        for policy in PLOTTED + rl:
            g = part[part["policy"] == policy].sort_values("level")
            if g.empty:
                continue
            is_rl = policy.startswith("rl")
            color = STYLES.get(policy) or RL_COLOURS[rl.index(policy) % len(RL_COLOURS)]
            ax.plot(g["level"], g["mean"], "-" if is_rl else "--", marker="o" if is_rl else "s",
                    color=color, linewidth=2 if is_rl else 1.3, markersize=5, label=policy)
            if is_rl or policy == "greedy_select":
                ax.fill_between(g["level"], g["ci_low"], g["ci_high"], color=color, alpha=0.12)
        ax.set_xticks(sorted(part["level"].unique()))
        ax.set_xlabel(f"Number of trains ({MODE_LABELS[mode]})")
        ax.set_ylabel("Total delay (steps)")
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle("Total delay with breakdown lengths from real data (mean; bands = 95% interval)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/realdata.csv")
    args = parser.parse_args()
    out = os.path.dirname(args.csv) or "."
    df = pd.read_csv(args.csv)
    rl = sorted(p for p in df["policy"].unique() if p.startswith("rl"))
    pd.set_option("display.width", 220)

    summary = summarise(df)
    summary.to_csv(os.path.join(out, "realdata_summary.csv"), index=False)
    plot(summary, os.path.join(out, "realdata_total_delay.png"))
    for metric, label in [("total_delay", "Mean total delay"), ("throughput", "Share of trains arrived")]:
        table = summary[summary["metric"] == metric].pivot_table(index=["shift", "level"], columns="policy", values="mean")
        print(f"\n{label}:")
        print(table.round(3 if metric == "throughput" else 1).to_string())

    rec = recovery_summary(df)
    rec.to_csv(os.path.join(out, "realdata_recovery.csv"), index=False)
    print("\nExtra delay caused by the breakdowns (mean):")
    print(rec.pivot_table(index=["shift", "level"], columns="policy", values="extra_delay").round(1).to_string())
    print("\nShare of planned breakdowns that actually happened:")
    print(rec.pivot_table(index=["shift", "level"], columns="policy", values="breakdowns_happened").round(2).to_string())

    if rl:
        tests = paired(df, rl)
        tests.to_csv(os.path.join(out, "realdata_paired.csv"), index=False)
        paired_extra_delay(df, rl).to_csv(os.path.join(out, "realdata_recovery_paired.csv"), index=False)
        cols = ["shift", "level", "rl", "baseline", "mean_difference", "rl_better", "ties", "rl_worse", "p_value"]
        print("\nRL vs baselines, paired per scenario (difference < 0 means RL has less delay):")
        print(tests[cols].round(3).to_string(index=False))
    print(f"\nSaved summaries and plot in {out}/")


if __name__ == "__main__":
    main()
