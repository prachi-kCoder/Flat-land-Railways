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
    Steps from the first disruption starting until the network is back to
    the delay it would have had with no disruption at all.

    delay_curve      network delay at each step of the disrupted run
    reference_curve  same, for the same scenario and policy, no disruptions
    Both curves are indexed by step number - 1.

    Returns 0 if the disruptions never pushed delay above the reference,
    and None if the network never recovered before the episode ended.
    The network is only checked from the step after the last breakdown
    ends, so a train that is still broken cannot count as recovered.
    """
    if not disruption_plan:
        return None
    start = min(d.start_step for d in disruption_plan)
    earliest_check = max(d.end_step for d in disruption_plan) + 1

    if all(delay <= ref for delay, ref in zip(delay_curve, reference_curve)):
        return 0

    for step in range(earliest_check, len(delay_curve) + 1):
        reference = reference_curve[min(step, len(reference_curve)) - 1]
        if delay_curve[step - 1] <= reference:
            return step - start
    return None


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
