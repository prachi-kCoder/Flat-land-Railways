"""Fast numerical tests for MAPPO networks, GAE, and the PPO update."""
import numpy as np
import torch
from railsim.mappo import Actor, CentralizedCritic, MAPPO, generalized_advantage


def test_actor_returns_two_action_logits():
    actor = Actor(obs_size=13, hidden_size=16)
    logits = actor(torch.zeros(5, 13))
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_centralized_critic_returns_one_value_per_joint_state():
    critic = CentralizedCritic(state_size=56, hidden_size=16)
    values = critic(torch.zeros(3, 56))
    assert values.shape == (3,)
    assert torch.isfinite(values).all()


def test_gae_is_finite_and_terminal_reward_propagates_back():
    advantages, returns = generalized_advantage(
        rewards=[0.0, 1.0], values=[0.0, 0.0], dones=[False, True],
        gamma=1.0, gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages, [1.0, 1.0])
    np.testing.assert_allclose(returns, [1.0, 1.0])


def test_mappo_update_produces_finite_losses():
    learner = MAPPO(num_trains=2, hidden_size=16)
    batch = {
        "obs": np.zeros((4, 13), dtype=np.float32),
        "states": np.zeros((4, 28), dtype=np.float32),
        "actions": np.array([0, 1, 0, 1]),
        "log_probs": np.log(np.full(4, 0.5, dtype=np.float32)),
        "returns": np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32),
        "advantages": np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32),
    }
    metrics = learner.update(batch, epochs=1, batch_size=4)
    assert all(np.isfinite(value) for value in metrics.values())
