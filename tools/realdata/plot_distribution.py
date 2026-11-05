"""Histogram of real delay episodes (minutes) used for breakdown lengths.

Run from the repository root:
  python tools/realdata/plot_distribution.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fail = np.loadtxt("results/realdata/minutes_failure_episodes_ge3.csv", skiprows=1)
every = np.loadtxt("results/realdata/minutes_delay_episodes_ge3.csv", skiprows=1)
bins = np.arange(3, 185, 3)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(np.clip(every, None, 181), bins=bins, color="C7", alpha=0.6,
        label=f"All delay episodes >= 3 min (n = {len(every)})")
ax.hist(np.clip(fail, None, 181), bins=bins, color="C3", alpha=0.85,
        label=f"Rolling-stock or infrastructure cause (n = {len(fail)})")
for q, style in ((50, ":"), (90, "--"), (99, "-.")):
    v = np.percentile(fail, q)
    ax.axvline(v, color="C3", linestyle=style, linewidth=1)
    ax.text(v, 1, f" p{q} = {v:.0f}", color="C3", fontsize=8, rotation=90,
            va="bottom")
ax.set_yscale("log")
ax.set_xlabel("Delay added in one episode (minutes; 180+ grouped at the right)")
ax.set_ylabel("Number of episodes (log scale)")
ax.set_title("Delay episodes of Finnish long-distance trains, 48 days in 2024-2025 (Digitraffic)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/realdata/real_delay_distribution.png", dpi=150)
print("Saved results/realdata/real_delay_distribution.png")
