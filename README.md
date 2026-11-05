# Flatland Railways

**Simulation Based Evaluation of Reinforcement Learning for Robust Railway Traffic Rescheduling Under Dynamic Disruptions**

Final-year project, B.Tech CSE, Maharaja Surajmal Institute of Technology.
Anamika Aggarwal, Drishya Garg, Prachi Kumari. Supervised by Ms. Amita Yadav.

We use the [Flatland-RL](https://github.com/flatland-association/flatland-rl) railway simulator to compare a reinforcement-learning dispatcher with simple rule-based dispatchers (First-Come-First-Served, Priority), and to measure how a trained policy holds up when traffic and disruptions differ from what it was trained on.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The code is written for **flatland-rl 4.3.0**. Older versions (3.x) use a different agent API and the scripts will not work with them.

## Scripts

| Script | What it does |
|---|---|
| `python initial-rail.py` | Builds the base 30x30 scenario and prints each train's start, target and timetable. |
| `python railway.py` | Draws the railway network in a window. Use `--save outputs/network.png` to save an image instead (works without a screen). |
| `python train-movement.py` | Runs 5 trains with no dispatcher (always move forward) and reports departures, arrivals, blocking and deadlocks step by step. |
| `python run_experiment.py` | Runs every policy on the same batch of disrupted scenarios, prints averages and writes one CSV row per run to `outputs/results.csv`. See `--help` for options (trains, number of scenarios, breakdown count/duration, simultaneous). |

## Code layout

```
railsim/
  env_factory.py   EnvConfig + make_env(): the one place the environment is built;
                   usable_seeds() skips seeds where Flatland changes the city layout
  train_status.py  train position/direction/target helpers, deadlock detection
  disruptions.py   DisruptionSpec, planned breakdowns, Flatland hook that applies them
  policies.py      dispatching policies: always_forward, shortest_path (no dispatcher),
                   fcfs, priority (rule-based, conflict-free)
  planner.py       conflict-free route booking in space and time, used by fcfs/priority
  metrics.py       delay, recovery time, EpisodeResult
  runner.py        run_episode(): play one policy on one scenario and measure it
tests/
  test_railsim.py               reproducibility and deadlock-detector checks
  test_disruptions_metrics.py   disruption, metric and runner checks
  test_dispatchers.py           planner and FCFS/Priority checks
```

All scripts build their environment through `make_env(EnvConfig(...))`, so the same config and seed always give the same network, trains and timetable.

### How deadlocks are detected

`find_deadlocked_trains(env)` returns trains that can never move again, whatever actions they are given: every cell a train could enter is held by another deadlocked train. This covers head-on deadlocks, trains stuck behind them, and circular waits.

Flatland also has `env.motion_check.deadlocked`, but that is a per-step flag meaning "these trains tried to swap cells this step". It fires even when one of the trains could still turn away at a switch, so it is not used as the deadlock measure here.

### Disruptions

A disruption is a train breakdown: the train stops where it is for a fixed number of steps and blocks the track behind it. If it has not departed yet, its departure is delayed instead.

```python
from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env

env, obs, info = make_env(EnvConfig(num_trains=8, seed=3),
                          DisruptionSpec(count=2, duration=20, simultaneous=False))
env.disruption_plan   # [Disruption(train=..., start_step=..., duration=20), ...]
```

The plan is drawn from the timetable only (each breakdown starts while that train is scheduled to be running), never from what happens during the run. So every policy faces exactly the same disruptions for a given seed. Adding disruptions does not change the network or timetable.

Not implemented yet: speed restrictions and track closures without a train on them.

### Policies

| Policy | What it does |
|---|---|
| `always_forward` | Every train always moves forward. Ignores routes and other trains. |
| `shortest_path` | Every train follows its shortest route and departs as soon as allowed; nobody ever waits for anyone (Flatland's `ShortestPathPolicy`). This is "no dispatcher". |
| `fcfs` | Rule-based dispatcher. Trains book conflict-free routes in order of scheduled departure (first come, first served). |
| `priority` | Rule-based dispatcher. Trains book routes in order of least slack: the train closest to missing its latest arrival goes first. Recomputed at every re-plan. |

**How `fcfs` and `priority` work** (`railsim/planner.py`): trains are planned one at a time in priority order. Each one searches for its earliest possible arrival, moving one cell per step or waiting (holding at its station or on the track), while avoiding every cell already booked by the trains before it. Its route is then booked. Booked routes never share a cell at the same time and never swap cells head-on, so these dispatchers cannot deadlock (checked in the tests). The two rules differ only in the order trains get to book.

They re-plan from the current state whenever a train is not where its plan says (for example, a breakdown has just stopped it), and every 5 steps while some train has no route. They learn about a breakdown only when it happens, and are then told how long it will last.

A train that has not departed and cannot reach its target before the episode ends is held at its station (it counts as not arrived).

### Metrics

| Metric | Definition |
|---|---|
| Arrived % (throughput) | Trains that reached their target / all trains |
| Arrived on time | Arrived no later than `latest_arrival` |
| Total delay | Sum over trains. Arrived: `max(0, arrival - latest_arrival)`. Not arrived: how late it would be if it ran its shortest path from where it is now, `max(0, now + remaining travel time - latest_arrival)`. Same convention as Flatland's reward; equals minus Flatland's episode reward when every train departs (checked in the tests). |
| Deadlocked | Trains that can never move again at the end of the episode |
| Recovery time | Steps from the first breakdown until the breakdowns stop adding delay. The same policy is also run on the same scenario without breakdowns; the extra delay is the difference between the two runs. Recovered = the first step (no earlier than the step after the last breakdown ends) from which the extra delay never changes again. Delay already caused stays; recovery means it stops growing. `None` = still changing when the episode ended. **Only meaningful for policies that do not deadlock**: a deadlocked network freezes, so its extra delay also stops changing. |
| Replans | How many times a rule-based dispatcher re-planned |
| Decision time | Wall-clock time the policy takes to choose actions per step |

The delay of a train that is not arrived is optimistic for deadlocked trains (they will never arrive), so always read delay together with arrived % and deadlocked.

## Tests

```bash
python -m pytest tests/
```

## Notes

- Flatland sets the episode length itself from the timetable (for example 98 steps for 5 trains, seed 42).
- For some seeds Flatland cannot fit 3 cities on a 30x30 grid and warns that it is switching to grid mode or placing fewer cities. The scenario still runs, but its layout differs from the others.
