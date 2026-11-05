"""
Evaluation metrics.

Delay of one train (in steps), following Flatland's own conventions:
  - arrived:      how late it arrived, max(0, arrival_step - latest_arrival)
  - not arrived:  how late it would be if it could run its shortest path
                  from where it is now without stopping,
                  max(0, now + remaining_travel_time - latest_arrival)

The second case is optimistic for trains that are deadlocked (they will
never arrive). That is why "trains arrived" and "trains deadlocked" are
always reported next to delay, never delay alone.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from flatland.envs.step_utils.states import TrainState


def train_delay(agent, env):
    """Delay of one train at the current step (see module docstring)."""
    if agent.state == TrainState.DONE:
        return max(0, agent.arrival_time - agent.latest_arrival)
    return max(0, -agent.get_current_delay(env._elapsed_steps, env.distance_map))


def network_delay(env):
    """Sum of all trains' delays at the current step."""
    return sum(train_delay(agent, env) for agent in env.agents)


def recovery_time(delay_curve, reference_curve, disruption_plan):
    """
    Steps from the first breakdown until the disruption stops adding delay.

    delay_curve      network delay after each step of the disrupted run
    reference_curve  same, for the same scenario and policy, no breakdowns
    Both are indexed by step number - 1. The shorter curve is extended with
    its last value (once an episode ends, delays no longer change).

    The extra delay caused by the disruption at step t is
        excess(t) = delay_curve(t) - reference_curve(t).
    The network has recovered at the first step s, no earlier than the step
    after the last breakdown ends, from which excess never changes again:
    the disruption is no longer spreading. Delay already caused (trains
    that arrived late) stays in `excess`; recovery means it stops growing,
    not that it disappears.

    Returns 0 if the disruption never caused any extra delay, and None if
    the extra delay was still changing when the episode ended.
    """
    if not disruption_plan:
        return None
    start = min(d.start_step for d in disruption_plan)
    earliest_check = max(d.end_step for d in disruption_plan) + 1

    length = max(len(delay_curve), len(reference_curve))

    def extend(curve):
        return list(curve) + [curve[-1]] * (length - len(curve))

    excess = [a - b for a, b in zip(extend(delay_curve), extend(reference_curve))]
    if not any(excess):
        return 0

    # Last step at which the excess changed (step numbers are index + 1).
    last_change = max(
        (i + 1 for i in range(1, length) if excess[i] != excess[i - 1]),
        default=1,
    )
    if last_change >= length:
        return None
    recovered_at = max(last_change, earliest_check)
    return recovered_at - start


@dataclass
class EpisodeResult:
    """Everything measured in one run of one policy on one scenario."""

    policy: str
    seed: int
    num_trains: int
    disruption_count: int
    disruption_duration: int
    simultaneous: bool

    steps: int
    arrived: int
    arrived_on_time: int
    deadlocked: int
    cancelled: int   # never left their station (Castagna et al. 2026 report this separately)
    first_deadlock_step: Optional[int]

    total_delay: int
    mean_delay: float
    max_delay: int
    flatland_reward: float

    recovery_time: Optional[int]
    disruptions_on_map: int
    disruptions_off_map: int
    disruptions_skipped: int

    decision_ms_mean: float
    decision_ms_max: float
    replans: int   # rule-based dispatchers only; 0 for the reference policies

    delay_curve: List[int] = field(default_factory=list, repr=False)

    @property
    def throughput(self):
        """Fraction of trains that reached their target."""
        return self.arrived / self.num_trains

    def as_row(self):
        """Flat dict for a CSV row (without the per-step curve)."""
        row = asdict(self)
        row.pop("delay_curve")
        row["throughput"] = round(self.throughput, 4)
        return row
