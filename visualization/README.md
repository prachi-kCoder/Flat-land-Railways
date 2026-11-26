# Flatland episode replay

The visualizer records a real Flatland episode and writes a **self-contained HTML
replay**. Open the file in any modern browser; no web server, React toolchain,
Pygame installation, or extra Python package is needed.

The railway geometry is extracted from Flatland's transition map. Train positions,
states, malfunction counters, arrivals, dispatch decisions, PPO ordering choices,
and delay values are captured from the actual environment while it runs. The
browser only replays the recorded data.

## Run

From the repository root with the project's virtual environment active:

```powershell
# MAPPO (default model: results/models/mappo_seed1.pt)
python visualize.py --policy mappo --trains 8 --seed 42 --disruptions 1 --duration 20

# PPO strategy selector (default model: results/models/rl_strategy_seed1.zip)
python visualize.py --policy ppo --trains 8 --seed 42 --disruptions 1 --duration 20

# Rule-based baseline
python visualize.py --policy greedy_select --trains 8 --seed 42 --disruptions 1 --duration 20
```

## Compare approaches side by side

`--compare` runs 2 to 4 approaches on the **same scenario** (same network,
timetable and breakdowns) and writes one page with all of them on screen:

```powershell
python visualize.py --compare fcfs priority ppo mappo --trains 8 --seed 28 --disruptions 1 --duration 20
```

The page shows one shared timeline (play, step, scrub), one network view per
approach with its live arrived / delay / on-track / broken-down counts, one
delay-over-time chart with a line per approach (the breakdown is shaded), and
a final results table: arrived, total delay, extra delay caused by the
breakdowns, deadlocked trains and steps to finish. The best total delay is in
bold. `--ppo-model` and `--mappo-model` choose other model files. Seed 28 is a
good demo: PPO finishes with total delay 4, FCFS 107, MAPPO 117, Priority 130.

MAPPO samples its actions, so the visualizer seeds PyTorch with the scenario
seed before each run, as the experiment scripts do. A replay therefore shows
exactly the run that was evaluated (same numbers as `results/robustness/`).

By default the replay is saved under `outputs/` and opened in the browser.
Use `--output path/to/replay.html` to choose a different location or
`--no-open` to only write the file.

## Playback

- Play/pause, previous/next step, timeline scrubbing, and playback speed.
- Train positions, origin markers for trains waiting to depart, destination markers colour-matched to each train, malfunction highlighting, and readable train states.
- One-based display labels (Train 1, Train 2, ...) while preserving Flatland's zero-based IDs internally.
- Live per-step arrivals, delay, on-track and malfunction counts, shown separately from final episode results.
- Breakdown, recovery, arrival, planner-replan, and policy-decision event logs, with a compact per-step WAIT/DISPATCH summary.
- Network-delay curve for the disrupted run.
- Final arrival, delay, deadlock, and recovery metrics.

The recovery metric compares the disrupted run with a second run of the same
policy, seed, timetable, and railway layout **without** disruptions. This
reference run is enabled by default and roughly doubles evaluation time.
Pass `--skip-reference` to omit it; recovery will then be shown as unavailable.

## Policy names

Supported values for `--policy`:

- `mappo`: shared actor, WAIT/DISPATCH decisions (requires a MAPPO checkpoint).
- `ppo`: PPO chooses the route-booking ordering at each re-plan (default model
  `results/models/rl_strategy_seed1.zip`).
- `rl_strategy`: the same, with the model given by `--model`.
- `fcfs`, `priority`, `edf`, `spf`, `greedy_select`,
  `release_all`, `hold_if_blocked`: existing baselines.

## Interpretation

A train breakdown is represented as a malfunctioning train. If it is already
on the network, it remains in its cell and can obstruct trains behind it. If
it has not departed, its departure is delayed. The visualizer does not invent
track closures or alternate routes.

The MAPPO and PPO policies solve different decision problems in this repository:
MAPPO selects per-train WAIT/DISPATCH actions, while the strategy PPO selects
one of four route-booking orderings at a re-plan. Their displayed decisions
are labelled differently. Their outcomes (delay, arrivals, deadlocks) are
directly comparable, because every approach runs on the same scenario.

This first version is a **post-run replay**, not live rendering during training.
It does not alter observations, rewards, actions, route planning, or experiment
results. Large robustness sweeps should continue to use the existing experiment
scripts rather than generating a replay for every scenario.
