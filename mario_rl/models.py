"""Mạng Actor-Critic dùng chung thân CNN (kiến trúc "Nature DQN" của DeepMind)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Khởi tạo trực giao (orthogonal) - mặc định thực hành tốt của PPO."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """π(a|s) và V(s) chia sẻ phần trích đặc trưng.

    Đầu vào : (B, 4, 84, 84) uint8  -> chia 255 thành [0, 1]
    Thân    : Conv(32,8x8,s4) - Conv(64,4x4,s2) - Conv(64,3x3,s1) - FC(512)
    Đầu ra  : logits (B, n_actions) cho actor, giá trị (B,) cho critic
    """

    def __init__(self, n_actions: int, in_channels: int = 4) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.actor = _init(nn.Linear(512, n_actions), gain=0.01)  # chính sách ban đầu ~ đồng đều
        self.critic = _init(nn.Linear(512, 1), gain=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs.float() / 255.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(obs)).squeeze(-1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        logits, _ = self(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()
