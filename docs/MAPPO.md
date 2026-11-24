# MAPPO prototype

This is an additional experimental implementation. It does not replace or
modify the existing Stable-Baselines3 PPO strategy selector or departure PPO.

## Design

- Each ready train is treated as an agent making a WAIT or DISPATCH decision.
- A single actor network is shared by all train agents. At execution time, it
  receives that train's 13-element per-train observation. The observation
  contains train-specific features plus selected network summary features.
- During training, a centralized critic receives the concatenated per-train
  observations for all trains plus a one-hot indicator for the train currently
  making a decision. This is the centralized-training/decentralized-execution
  (CTDE) structure.
- All agents receive the network-delay reward already used by DispatchEnv.
- The existing prioritized, conflict-free route planner remains unchanged and
  continues to allocate routes after dispatch decisions.

The environment asks ready trains for decisions sequentially within a Flatland
step. The rollout therefore records one agent decision at a time and assigns
the resulting shared reward to the transition on which the simulator advances.
This is a lightweight shared-policy MAPPO prototype, not a claim that the full
robustness study has been rerun.

## Install

No new package is required: PyTorch is already pinned in requirements.txt.

## Run

On a small machine, begin with a smoke run:

    python train_mappo.py --episodes 2 --trains 4 --output models/mappo_smoke.pt
    python evaluate_mappo.py --model models/mappo_smoke.pt --trains 4 --seed 42

A slightly longer exploratory run:

    python train_mappo.py --episodes 20 --trains 8 --output models/mappo_seed1.pt
    python evaluate_mappo.py --model models/mappo_seed1.pt --trains 8 --seed 42

The checkpoint is written locally under models/ and is not committed. The
network is small (64-unit MLPs); no replay dataset, pretrained model, GPU,
parallel workers, or extra runtime package is needed.

## Interpretation and limitations

A two-episode smoke run checks that the simulator, rollout, optimizer, save,
and load paths execute; it is not enough to claim a trained or competitive
policy. Report PPO and MAPPO separately, use held-out scenario seeds, and
compare delay alongside arrival rate and deadlocks. The MAPPO prototype has
not yet been evaluated across the repository's full robustness grid.
