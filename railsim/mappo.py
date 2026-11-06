"""
A lightweight MAPPO prototype for Flatland railway dispatching.

Actors are shared across trains and see the active train's 13-D per-train
observation (which includes selected network summary features). During training,
a centralized critic sees all train observations plus an active-train indicator
(CTDE). At execution, only the shared actor is needed. The existing
conflict-free route planner remains the safety layer.
This module does not modify the existing PPO implementation.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from flatland.envs.step_utils.states import TrainState
from railsim.rl_dispatch import OBS_SIZE, ReleaseDispatcher, observation


class Actor(nn.Module):
    """Shared decentralized policy: per-train observation -> action logits."""
    def __init__(self, obs_size=OBS_SIZE, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, obs):
        return self.net(obs)


class CentralizedCritic(nn.Module):
    """Training-only value function over the joint railway state."""
    def __init__(self, state_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def joint_state(dispatch_env):
    """Flatten per-train observations and append a one-hot active-train mask."""
    env = dispatch_env.env
    now = env._elapsed_steps
    n = len(env.agents)
    rows = []
    for handle, agent in enumerate(env.agents):
        if agent.state == TrainState.DONE:
            rows.append(np.zeros(OBS_SIZE, dtype=np.float32))
        else:
            rows.append(observation(env, dispatch_env.dispatcher, handle, now))
    active = np.zeros(n, dtype=np.float32)
    if dispatch_env.queue:
        active[int(dispatch_env.queue[0])] = 1.0
    return np.concatenate([np.asarray(rows, dtype=np.float32).reshape(-1), active])


def generalized_advantage(rewards, values, dones, gamma=1.0, gae_lambda=0.95):
    """GAE computed over the sequential per-train decisions in one episode."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        alive = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * alive - values[t]
        gae = delta + gamma * gae_lambda * alive * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


class MAPPO:
    """Shared-actor MAPPO learner with a centralized training critic."""
    def __init__(self, num_trains, hidden_size=64, learning_rate=3e-4,
                 clip_range=0.2, entropy_coef=0.01, value_coef=0.5, device="cpu"):
        self.num_trains = int(num_trains)
        self.obs_size = OBS_SIZE
        self.state_size = self.num_trains * (OBS_SIZE + 1)
        self.device = torch.device(device)
        self.actor = Actor(OBS_SIZE, hidden_size).to(self.device)
        self.critic = CentralizedCritic(self.state_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )
        self.clip_range = float(clip_range)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)

    def act(self, per_train_obs, global_state, deterministic=False):
        obs = torch.as_tensor(per_train_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = Categorical(logits=self.actor(obs))
            action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, batch, epochs=4, batch_size=128):
        """PPO clipped update over collected per-agent decision records."""
        obs = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32, device=self.device)
        states = torch.as_tensor(np.asarray(batch["states"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        if len(actions) < 2:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        totals = np.zeros(3, dtype=np.float64)
        updates = 0
        for _ in range(int(epochs)):
            indices = torch.randperm(len(actions), device=self.device)
            for start in range(0, len(actions), int(batch_size)):
                idx = indices[start:start + int(batch_size)]
                dist = Categorical(logits=self.actor(obs[idx]))
                new_logp = dist.log_prob(actions[idx])
                ratio = torch.exp(new_logp - old_logp[idx])
                unclipped = ratio * advantages[idx]
                clipped = torch.clamp(ratio, 1.0 - self.clip_range,
                                      1.0 + self.clip_range) * advantages[idx]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                values = self.critic(states[idx])
                critic_loss = torch.mean((values - returns[idx]) ** 2)
                entropy = dist.entropy().mean()
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.optimizer.step()
                totals += [actor_loss.item(), critic_loss.item(), entropy.item()]
                updates += 1
        totals /= max(updates, 1)
        return {"actor_loss": float(totals[0]), "critic_loss": float(totals[1]),
                "entropy": float(totals[2])}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "num_trains": self.num_trains,
            "obs_size": self.obs_size,
            "state_size": self.state_size,
        }, path)

    @classmethod
    def load_actor(cls, path, num_trains=None, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        n = int(num_trains or checkpoint["num_trains"])
        model = cls(n, device=device)
        model.actor.load_state_dict(checkpoint["actor"])
        model.actor.eval()
        return model.actor


class MAPPODispatcher(ReleaseDispatcher):
    """Flatland execution adapter: one shared decentralized actor per train."""
    name = "mappo"
    model_path = "models/mappo_seed1.pt"
    _actor = None
    _loaded_from = None

    @classmethod
    def actor(cls):
        if cls._actor is None or cls._loaded_from != cls.model_path:
            cls._actor = MAPPO.load_actor(cls.model_path, device="cpu")
            cls._loaded_from = cls.model_path
        return cls._actor

    def decide(self, env, now, handle):
        per_train_obs = observation(env, self, handle, now)
        with torch.no_grad():
            logits = self.actor()(torch.as_tensor(per_train_obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits, dim=-1).item())
