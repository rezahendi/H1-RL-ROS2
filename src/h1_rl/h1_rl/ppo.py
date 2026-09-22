"""Compact PPO (in the style of rsl_rl) with an asymmetric actor-critic.

The actor sees only what the real robot can measure (IMU + joint encoders,
with history). The critic additionally gets privileged simulator state (base
velocity, contacts, randomized friction/mass) which makes value estimates
much better and training faster; the critic is not needed at deployment.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


def mlp(in_dim: int, hidden: list[int], out_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), ACTIVATIONS[activation]()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class EmpiricalNormalization(nn.Module):
    """Running mean/std normalizer (updated only in train mode)."""

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(1, dim))
        self.register_buffer("var", torch.ones(1, dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(x)
        return (x - self.mean) / (self.std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        n = x.shape[0]
        self.count += n
        rate = n / self.count.item()
        mean_x = x.mean(dim=0, keepdim=True)
        var_x = x.var(dim=0, unbiased=False, keepdim=True)
        delta = mean_x - self.mean
        self.mean += rate * delta
        self.var += rate * (var_x - self.var + delta * (mean_x - self.mean))


class ActorCritic(nn.Module):
    def __init__(self, num_obs: int, num_critic_obs: int, num_actions: int,
                 actor_hidden: list[int], critic_hidden: list[int], activation: str = "elu",
                 init_noise_std: float = 1.0):
        super().__init__()
        self.actor = mlp(num_obs, actor_hidden, num_actions, activation)
        self.critic = mlp(num_critic_obs, critic_hidden, 1, activation)
        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(init_noise_std)))
        # small last layer -> actions start close to the default pose
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.std.expand_as(mean))

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)


class RolloutStorage:
    def __init__(self, num_steps: int, num_envs: int, num_obs: int, num_critic_obs: int,
                 num_actions: int, device: torch.device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.obs = z(num_steps, num_envs, num_obs)
        self.critic_obs = z(num_steps, num_envs, num_critic_obs)
        self.actions = z(num_steps, num_envs, num_actions)
        self.log_probs = z(num_steps, num_envs)
        self.values = z(num_steps, num_envs)
        self.rewards = z(num_steps, num_envs)
        self.dones = z(num_steps, num_envs)
        self.mu = z(num_steps, num_envs, num_actions)
        self.sigma = z(num_steps, num_envs, num_actions)
        self.returns = z(num_steps, num_envs)
        self.advantages = z(num_steps, num_envs)
        self.num_steps, self.num_envs = num_steps, num_envs
        self.step = 0

    def add(self, obs, critic_obs, actions, log_probs, values, rewards, dones, mu, sigma):
        t = self.step
        self.obs[t], self.critic_obs[t], self.actions[t] = obs, critic_obs, actions
        self.log_probs[t], self.values[t], self.rewards[t], self.dones[t] = log_probs, values, rewards, dones
        self.mu[t], self.sigma[t] = mu, sigma
        self.step += 1

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        adv = torch.zeros_like(last_values)
        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + not_done * gamma * next_values - self.values[t]
            adv = delta + not_done * gamma * lam * adv
            self.returns[t] = adv + self.values[t]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        self.step = 0

    def minibatches(self, num_mini_batches: int, num_epochs: int):
        total = self.num_steps * self.num_envs
        size = total // num_mini_batches
        flat = lambda x: x.reshape(total, *x.shape[2:])  # noqa: E731
        data = [flat(x) for x in (self.obs, self.critic_obs, self.actions, self.values, self.advantages,
                                   self.returns, self.log_probs, self.mu, self.sigma)]
        for _ in range(num_epochs):
            perm = torch.randperm(total, device=self.obs.device)
            for i in range(num_mini_batches):
                idx = perm[i * size:(i + 1) * size]
                yield [x[idx] for x in data]


class PPO:
    def __init__(self, policy: ActorCritic, cfg: dict, device: torch.device):
        self.policy = policy
        self.device = device
        self.lr = float(cfg["learning_rate"])
        self.schedule = cfg.get("schedule", "adaptive")
        self.desired_kl = float(cfg.get("desired_kl", 0.01))
        self.clip = float(cfg["clip_param"])
        self.gamma, self.lam = float(cfg["gamma"]), float(cfg["lam"])
        self.entropy_coef = float(cfg["entropy_coef"])
        self.value_coef = float(cfg["value_loss_coef"])
        self.epochs = int(cfg["num_learning_epochs"])
        self.mini_batches = int(cfg["num_mini_batches"])
        self.max_grad_norm = float(cfg["max_grad_norm"])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def update(self, storage: RolloutStorage) -> dict:
        stats = {"value_loss": 0.0, "surrogate_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n = 0
        for obs, cobs, act, old_v, adv, ret, old_logp, old_mu, old_sigma in storage.minibatches(
                self.mini_batches, self.epochs):
            dist = self.policy.distribution(obs)
            logp = dist.log_prob(act).sum(-1)
            entropy = dist.entropy().sum(-1)
            value = self.policy.value(cobs)
            mu, sigma = dist.mean, dist.stddev

            with torch.no_grad():  # KL(old || new) for the adaptive learning rate
                kl = torch.sum(torch.log(sigma / old_sigma + 1e-5)
                               + (old_sigma ** 2 + (old_mu - mu) ** 2) / (2.0 * sigma ** 2) - 0.5, dim=-1).mean()
            if self.schedule == "adaptive":
                if kl > 2.0 * self.desired_kl:
                    self.lr = max(1e-5, self.lr / 1.5)
                elif 0.0 < kl < 0.5 * self.desired_kl:
                    self.lr = min(1e-2, self.lr * 1.5)
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr

            ratio = torch.exp(logp - old_logp)
            surr = -adv * ratio
            surr_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
            surrogate_loss = torch.max(surr, surr_clipped).mean()
            v_clipped = old_v + (value - old_v).clamp(-self.clip, self.clip)
            value_loss = torch.max((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
            loss = surrogate_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats["value_loss"] += value_loss.item()
            stats["surrogate_loss"] += surrogate_loss.item()
            stats["entropy"] += entropy.mean().item()
            stats["kl"] += kl.item()
            n += 1
        return {k: v / max(n, 1) for k, v in stats.items()}
