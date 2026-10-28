"""No-vision actor inputs and network; usable without Isaac Lab or the SDK."""

import math

import torch
from torch import nn


class JointHistory:
    """Oldest-to-newest joint feedback and applied position targets, in radians.

    Each frame has 13 normalized measured angles, then 13 normalized commands.
    Repeating the first frame supplies startup history without a fake previous
    episode. Call advance once per 20 Hz control step, after reading feedback.
    """

    def __init__(self, num_envs, steps, lower, upper):
        if num_envs < 1 or steps < 1 or lower.shape != (13,) or upper.shape != (13,):
            raise ValueError("Expected positive batch/history sizes and 13 joint limits")
        if not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all() & (upper > lower).all()):
            raise ValueError("Joint limits must be finite and ordered")
        self.lower, self.upper = lower.clone(), upper.clone()
        self.data = lower.new_zeros((num_envs, steps, 26))
        self.initialized = False

    def _frame(self, measured, commanded):
        expected = (self.data.shape[0], 13)
        if measured.shape != expected or commanded.shape != expected:
            raise ValueError("Expected batched 13-joint measurements and commands")
        if not bool(torch.isfinite(measured).all() & torch.isfinite(commanded).all()):
            raise ValueError("Nonfinite joint feedback or target")
        q = 2.0 * (measured - self.lower) / (self.upper - self.lower) - 1.0
        target = 2.0 * (commanded - self.lower) / (self.upper - self.lower) - 1.0
        return torch.cat((q, target), dim=-1).clamp(-5.0, 5.0)

    def reset(self, measured, commanded):
        self.data[:] = self._frame(measured, commanded).unsqueeze(1)
        self.initialized = True

    def advance(self, measured, commanded, reset_mask):
        if not self.initialized:
            raise ValueError("Initialize history before advancing")
        if reset_mask.shape != (self.data.shape[0],) or reset_mask.dtype != torch.bool:
            raise ValueError("Expected one boolean reset flag per environment")
        frame = self._frame(measured, commanded)
        self.data[:, :-1] = self.data[:, 1:].clone()
        self.data[:, -1] = frame
        self.data[reset_mask] = frame[reset_mask, None, :]

    def observation(self, speed):
        if not self.initialized or not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("History must be initialized and speed finite and positive")
        command = self.data.new_full((self.data.shape[0], 1), speed)
        return torch.cat((self.data.flatten(1), command), dim=-1)


class StudentPolicy(nn.Module):
    """Bounded 13-joint incremental action, with no simulator state input."""

    def __init__(self, history_steps=30):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(history_steps * 26 + 1, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 13),
            nn.Tanh(),
        )

    def forward(self, observation):
        return self.actor(observation)
