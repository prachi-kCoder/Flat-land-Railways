"""Build empirical distributions of delay increments from the downloaded
Digitraffic sample (long-distance trains, 48 days in 2024-2025).

Increment = differenceInMinutes at a timetable row minus differenceInMinutes
at the previous row of the same train run (both rows must have an actual
time). A positive increment is delay gained between two consecutive
timetable points (a running section or a stop).

Definitions compared (thresholds in minutes):
  all_increments        every positive increment >= threshold
  rolling_stock_cause   increments >= threshold on rows whose causes include
                        category K (Kalusto: rolling stock, multiple units and
                        coaches) or V (Vetokalusto: traction stock)
  infrastructure_cause  increments >= threshold on rows whose causes include
                        S (Sahkorata: electric traction line), T (Rata:
                        track structure) or P (Liikenteenhoitojarjestelmat:
                        traffic control systems)
  delay_episodes        runs of consecutive positive increments merged into one
                        episode (sum), episodes >= threshold
  failure_episodes      delay episodes with a rolling-stock OR infrastructure
                        cause (K, V, S, T, P), matched as below
  rolling_stock_episodes / infrastructure_episodes
                        delay episodes with a K/V (or S/T/P) cause on any row
                        of the episode or on the row just before or after it.
                        Causes are recorded on the row where the delay grows
                        only about 42% of the time in this sample (often 1-2
                        rows away), so per-increment cause matching misses
                        many cases.
Category names are from /api/v1/metadata/cause-category-codes (Finnish);
the English glosses above are our own translation.
"""
import csv
import glob
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
ROLLING, INFRA = {"K", "V"}, {"S", "T", "P"}

increments = []            # (minutes, set of cause categories)
episodes = []              # merged consecutive increases
journeys = []              # scheduled journey time per train run (minutes)
runs = rows_used = 0
for f in sorted(glob.glob(str(DATA / "*.json.gz"))):
    for train in json.load(gzip.open(f, "rt")):
        if train.get("cancelled"):
            continue
        rows = train["timeTableRows"]
        if len(rows) >= 2:
            t0 = datetime.fromisoformat(rows[0]["scheduledTime"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(rows[-1]["scheduledTime"].replace("Z", "+00:00"))
            journeys.append((t1 - t0).total_seconds() / 60)
        runs += 1
        usable = [r for r in rows if "differenceInMinutes" in r and "actualTime" in r]
        rows_used += len(usable)
        cats_at = [{c.get("categoryCode") for c in r.get("causes", [])} for r in usable]
        start = None   # index of first row of the current episode
        total = 0
        for i in range(1, len(usable) + 1):
            inc = usable[i]["differenceInMinutes"] - usable[i - 1]["differenceInMinutes"] if i < len(usable) else 0
            if inc > 0:
                increments.append((inc, cats_at[i]))
                if start is None:
                    start = i
                total += inc
            elif start is not None:
                window = set().union(*cats_at[max(0, start - 1): min(len(usable), i + 1)])
                episodes.append((total, window))
                start, total = None, 0

def describe(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    q = np.percentile(v, [50, 75, 90, 95, 99])
    return {"n": len(v), "mean": v.mean(), "p50": q[0], "p75": q[1], "p90": q[2], "p95": q[3],
            "p99": q[4], "max": v.max()}

sets = {}
for th in (1, 3, 5, 10):
    sets[("all_increments", th)] = [m for m, c in increments if m >= th]
    sets[("rolling_stock_cause", th)] = [m for m, c in increments if m >= th and c & ROLLING]
    sets[("infrastructure_cause", th)] = [m for m, c in increments if m >= th and c & INFRA]
    sets[("delay_episodes", th)] = [m for m, c in episodes if m >= th]
    sets[("rolling_stock_episodes", th)] = [m for m, c in episodes if m >= th and c & ROLLING]
    sets[("infrastructure_episodes", th)] = [m for m, c in episodes if m >= th and c & INFRA]
    sets[("failure_episodes", th)] = [m for m, c in episodes if m >= th and c & (ROLLING | INFRA)]

with open(OUT / "distributions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["definition", "threshold_min", "n", "mean", "p50", "p75", "p90", "p95", "p99", "max"])
    for (name, th), vals in sets.items():
        s = describe(vals)
        w.writerow([name, th] + [round(s.get(k, float("nan")), 2) for k in
                                 ("n", "mean", "p50", "p75", "p90", "p95", "p99", "max")])
for (name, th), vals in sets.items():
    if th == 3:
        np.savetxt(OUT / f"minutes_{name}_ge3.csv", np.asarray(vals, dtype=int), fmt="%d",
                   header="delay minutes", comments="")
np.savetxt(OUT / "journey_minutes.csv", np.round(journeys, 1), fmt="%.1f", header="scheduled journey minutes",
           comments="")
failures = sets[("failure_episodes", 3)]
summary = {"train_runs": runs, "rows_with_actual_time": rows_used, "positive_increments": len(increments),
           "episodes": len(episodes), "journey_minutes": describe(journeys),
           "failure_episodes_ge3": len(failures),
           "failure_episodes_ge3_per_train_run": round(len(failures) / runs, 4),
           "note": ("rate = failure episodes (>=3 min, rolling-stock or infrastructure cause) per long-distance "
                    "train run in the Digitraffic sample; used as the chance one train breaks down once in an episode")}
json.dump(summary, open(OUT / "sample_summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
