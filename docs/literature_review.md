# Literature review

How each source shapes this project. Every citation below was checked against the paper or page itself (arXiv API/PDF, publisher metadata or the dataset page). "Read" says how much of the source was actually read. Where only an abstract was available, only abstract-level claims are used.

## Core papers

### Castagna et al. 2026: semi-hierarchical RL in Flatland
A. Castagna, S. Zahlner, A. Egli, C. Eichenberger, D. Boos, M. Meyer, A. Fuxjäger. "Towards Autonomous Railway Operations: A Semi-Hierarchical Deep Reinforcement Learning Approach to the Vehicle Rescheduling Problem." arXiv:2605.10257, May 2026. Code: github.com/enlite-ai/maze-flatland. **Read: full text.**

- **Approach:** splits control into a dispatching policy (Dispatch/Wait for trains not yet on the map) and a routing policy (follow path / deviate / stop at junctions).
- **Why they split it:** with one policy, fewer than 15% of actions were departures, so departure timing was hardly learned (Sec. IV).
- **Training:** imitation learning from tree search (MCTS) plus behavioural cloning, with the dispatcher first trained to imitate a simple "delay if conflict" heuristic (Sec. VI). Not plain RL from rewards.
- **Setup:** Flatland, speed 1, breakdowns of 20–50 steps, 7–80 trains, 50 seeds.
- **Result:** Prioritized Planning had the highest success rate with 7–10 trains; their method was better from 20 trains on (Sec. VII).
- **Used here:**
  - the idea of RL controlling departures with the planner routing (our departure-control experiment);
  - starting RL by imitating a heuristic (our warm start);
  - reporting cancelled trains separately.

### Li et al. 2021: winners of the Flatland 2020 challenge
J. Li, Z. Chen, Y. Zheng, S.-H. Chan, D. Harabor, P. J. Stuckey, H. Ma, S. Koenig. "Scalable Rail Planning and Replanning: Winning the 2020 Flatland Challenge." ICAPS 31(1):477–485, 2021. DOI 10.1609/icaps.v31i1.15994. No arXiv version found. **Read: full text** (by the review agent; summarised from its notes).

- **Planning:** Prioritized Planning. Trains are planned one by one in a priority order, each with space-time A* avoiding already-planned trains; this is the technique our FCFS/Priority dispatchers use.
- **Several orders in parallel:** they ran 4 priority orders and kept the best, noting that "no single priority ordering ... dominates". This motivates our strategy selection.
- **After breakdowns:** partial replanning of affected trains, plus "lazy planning" (not pushing all trains in at once).
- **Differences from ours:**
  - We re-plan all trains rather than only the affected ones.
  - We have no LNS (large neighbourhood search) improvement step.
  - Our planner, like theirs, knows a breakdown's duration once it happens.

### Chen et al. 2023: slack-based priority
Z. Chen, J. Li, D. Harabor, P. J. Stuckey. "Scalable Rail Planning and Replanning with Soft Deadlines." arXiv:2306.06455, 2023. **Read: relevant sections; "Slack Based Priority" section checked directly.**

- Defines slack-based priority for prioritized planning in Flatland 3. This is the citation for our `priority` (least slack first) rule.

### Laurent et al. 2021: Flatland 2020 competition report
F. Laurent et al. "Flatland Competition 2020: MAPF and MARL for Efficient Train Coordination on a Grid World." PMLR 133:275–301, 2021. arXiv:2103.16511. **Read: full text; key quote checked directly.**

- **Competition breakdowns:** durations uniform in [20, 50] steps. Our 20-step training breakdown is at the low end of this range.
- **Simple rules matter:** "heuristics customised to the specifics of Flatland ... such as departure schedules and prioritisation, greatly improve the performance of solutions" (Sec. 6). That is why we added `hold_if_blocked` and `greedy_select` as non-learning baselines that RL has to beat.
- **RL vs OR:** best RL score 214 vs best operations-research (OR) score 298; "RL solutions are still a considerable distance away from OR based solutions".

### Yue et al. 2024: RL generalisation under disruption severity
P. Yue, Y. Jin, X. Dai, Z. Feng, D. Cui. "Reinforcement Learning for Scalable Train Timetable Rescheduling With Graph Representation." IEEE T-ITS, 2024. arXiv:2401.06952. **Read: relevant sections; curriculum passage checked directly.**

- **Supports our hypothesis:** a model trained on small delays did worse when applied to large disruptions (Sec. VII-C).
- **But:** a learning curriculum (small, then large delays) largely fixed it. Reviewers may raise this, so we discuss it and, where time allows, compare with mixed-condition training.

### Cobbe et al. 2019 and Kirk et al. 2023: generalisation in RL
- K. Cobbe, O. Klimov, C. Hesse, T. Kim, J. Schulman. "Quantifying Generalization in Reinforcement Learning." arXiv:1812.02341. **Read: abstract.** Separate training and test environments are needed to measure overfitting. We use separate seed ranges for training, validation and testing.
- R. Kirk, A. Zhang, E. Grefenstette, T. Rocktäschel. "A Survey of Zero-shot Generalisation in Deep Reinforcement Learning." JAIR 76:201–264, 2023. arXiv:2111.09794. **Read: abstract.** This is the framework for "train once, test on unseen conditions".

## Decentralised and recent railway rescheduling

### Naldini et al. 2026: self-organising railway traffic management
arXiv:2601.17017 (v1 Jan 2026, v2 Apr 2026). **Read: full text** (by the review agent).

- Trains coordinate by consensus with neighbours.
- **Results:** they beat FCFS by about 31%, and centralised optimisation by about 2–5%.
- **Disruption model:** their delays are entrance delays drawn from historical distributions (Table I), not breakdowns. FCFS is described as a "myopic strategy".

### Oddi et al. 2026: effect of the look-ahead horizon
arXiv:2606.13068. **Read: full text** (by the review agent).

- **Finding:** the total delay improvement stayed about 5% for every horizon tested (5–30 min); longer horizons only cost computation (Sec. V–VI).
- **Used here:** supports short-horizon observations; our agent sees the current plan and network state, not long forecasts.
- **Caveat:** the study used only 10 scenarios.

### Fouladi, Okhrin, Bešinović: Micro-RaiL (GNN-based RL)
SSRN 6604465, 2026. **Read: abstract only** (SSRN blocked).

- **Claims:** 61% less delay than FCFS, and generalisation to 200 unseen disruption scenarios.
- **Difference:** it models switch failures, while our disruptions are train breakdowns.

## Data for realistic disruptions

**FI-TW**, a Finnish train–weather dataset: arXiv:2601.16592 and its Kaggle page.
- **Contents:** about 48 million timetable events, 2018–2025, GPL-3.0.
- **Limitation:** it records **delays at stations, not breakdown durations**. Using it to set breakdown lengths would need our own processing and a minutes-to-steps assumption, so we did not use it for the current experiments.

Other Kaggle datasets examined (pages only, no files downloaded): Toronto subway delays, Indian Railways delays, Deutsche Bahn delays, PKP Intercity delays and NJ Transit/Amtrak. None publishes breakdown-duration distributions directly.

## Gap this project addresses

To our knowledge (searching arXiv for 2022–2026 Flatland RL work), no study systematically measures how a frozen RL policy degrades against strong prioritized-planning baselines as traffic density, breakdown length and the number of simultaneous breakdowns move away from training conditions.
