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
# MAPPO checkpoint
python visualize.py --policy mappo --model models/mappo_demo.pt --trains 8 --seed 42 --disruptions 1 --duration 20

# Existing PPO strategy selector
python visualize.py --policy rl_strategy --model results/models/rl_strategy_seed1.zip --trains 8 --seed 42 --disruptions 1 --duration 20

# Rule-based baseline
python visualize.py --policy greedy_select --trains 8 --seed 42 --disruptions 1 --duration 20
```

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
- `rl_strategy`: PPO chooses the route-booking ordering at each re-plan (requires
  the Stable-Baselines3 strategy checkpoint).
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
are intentionally labelled differently; this replay is not a claim that their
performance is directly comparable.

This first version is a **post-run replay**, not live rendering during training.
It does not alter observations, rewards, actions, route planning, or experiment
results. Large robustness sweeps should continue to use the existing experiment
scripts rather than generating a replay for every scenario.
