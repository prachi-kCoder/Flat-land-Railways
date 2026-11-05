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

## Code layout

```
railsim/
  env_factory.py   EnvConfig + make_env(): the one place the environment is built
  train_status.py  train position/direction/target helpers, deadlock detection
tests/
  test_railsim.py  reproducibility and deadlock-detector checks
```

All scripts build their environment through `make_env(EnvConfig(...))`, so the same config and seed always give the same network, trains and timetable.

### How deadlocks are detected

`find_deadlocked_trains(env)` returns trains that can never move again, whatever actions they are given: every cell a train could enter is held by another deadlocked train. This covers head-on deadlocks, trains stuck behind them, and circular waits.

Flatland also has `env.motion_check.deadlocked`, but that is a per-step flag meaning "these trains tried to swap cells this step". It fires even when one of the trains could still turn away at a switch, so it is not used as the deadlock measure here.

## Tests

```bash
python -m pytest tests/
```

## Notes

- Flatland sets the episode length itself from the timetable (for example 98 steps for 5 trains, seed 42).
- For some seeds Flatland cannot fit 3 cities on a 30x30 grid and warns that it is switching to grid mode or placing fewer cities. The scenario still runs, but its layout differs from the others.
