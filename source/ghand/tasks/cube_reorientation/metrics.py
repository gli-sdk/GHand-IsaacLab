# SPDX-License-Identifier: BSD-3-Clause
"""Simulator-independent episode accounting and checkpoint configuration restoration."""

from __future__ import annotations

from collections import Counter
import math
import statistics


def apply_reset_mode(config, saved, override="saved"):
    """Use checkpoint reset semantics independently of the selected task variant.

    Historical checkpoints predate correct_joint_reset and therefore use legacy
    behavior. Explicit overrides allow controlled comparisons and are reported.
    """
    saved_corrected = saved.get("correct_joint_reset", False)
    effective_corrected = saved_corrected if override == "saved" else override == "corrected"
    config.correct_joint_reset = effective_corrected
    return {
        "saved_reset_mode": "corrected" if saved_corrected else "legacy",
        "effective_reset_mode": "corrected" if effective_corrected else "legacy",
        "reset_mode_override": None if override == "saved" else override,
    }


def capture_terminal_successes(env):
    """Capture the exact post-reward counter before DirectRLEnv auto-resets episodes."""
    capture = {}
    original = env._get_rewards

    def reward_with_capture():
        reward = original()
        capture["successes"] = env.successes.clone()
        return reward

    env._get_rewards = reward_with_capture
    return capture


def _mean(values):
    return statistics.mean(values) if values else None


def _se(values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


class EpisodeAccumulator:
    """Equal quotas per environment; successes supplied AFTER reward, BEFORE reset."""

    def __init__(self, num_envs, episodes_per_env, step_dt):
        if num_envs < 1 or episodes_per_env < 1 or step_dt <= 0:
            raise ValueError("Environment count, episode quota, and step duration must be positive")
        self.num_envs = num_envs
        self.quota = episodes_per_env
        self.step_dt = step_dt
        self.counts = [0] * num_envs
        self.steps = [0] * num_envs
        self.last_successes = [0] * num_envs
        self.success_times = [[] for _ in range(num_envs)]
        self.records = []

    @property
    def complete(self):
        return all(count == self.quota for count in self.counts)

    def step(self, successes, terminated, truncated, numerical_failures=None):
        numerical_failures = numerical_failures or [False] * self.num_envs
        if any(len(values) != self.num_envs for values in (successes, terminated, truncated, numerical_failures)):
            raise ValueError("Step arrays must contain one value per environment")
        for i in range(self.num_envs):
            if self.counts[i] >= self.quota:
                continue
            self.steps[i] += 1
            bad = bool(numerical_failures[i]) or not math.isfinite(successes[i])
            if not bad:
                current = int(successes[i])
                increment = max(0, current - self.last_successes[i])
                self.success_times[i].extend([self.steps[i] * self.step_dt] * increment)
                self.last_successes[i] = current
            if not (terminated[i] or truncated[i] or bad):
                continue
            value = None if bad else int(successes[i])
            self.records.append(
                {
                    "env_id": i,
                    "episode_index": self.counts[i],
                    "successes": value,
                    "success_times_seconds": self.success_times[i],
                    "steps": self.steps[i],
                    "duration_seconds": self.steps[i] * self.step_dt,
                    "fall": bool(terminated[i]),
                    "timeout": bool(truncated[i]),
                    "numerical_failure": bad,
                    "termination": "numerical_failure" if bad else "fall" if terminated[i] else "timeout",
                }
            )
            self.counts[i] += 1
            self.steps[i] = 0
            self.last_successes[i] = 0
            self.success_times[i] = []

    def summary(self):
        valid = [r for r in self.records if not r["numerical_failure"]]
        successes = [r["successes"] for r in valid]
        reached = [int(v > 0) for v in successes]
        # Strictly after one second; include zero late successes in the valid-episode denominator.
        late_successes = [sum(t > 1.0 for t in r["success_times_seconds"]) for r in valid]
        durations = [r["duration_seconds"] for r in self.records]
        env_means = [_mean([r["successes"] for r in valid if r["env_id"] == i]) for i in range(self.num_envs)]
        return {
            "complete": self.complete,
            "episode_count": len(self.records),
            "expected_episode_count": self.num_envs * self.quota,
            "episodes_per_env": self.counts,
            "valid_episode_count": len(valid),
            "mean_successes": _mean(successes),
            "successes_standard_error": _se(successes),
            "successes_env_cluster_standard_error": _se([v for v in env_means if v is not None]),
            "at_least_one_success_rate": _mean(reached),
            "at_least_one_success_standard_error": _se(reached),
            "at_least_two_success_rate": _mean([int(v >= 2) for v in successes]),
            "mean_successes_after_1s": _mean(late_successes),
            "episodes_with_success_after_1s_rate": _mean([int(v > 0) for v in late_successes]),
            "success_distribution": dict(sorted(Counter(str(v) for v in successes).items())),
            "mean_duration_seconds": _mean(durations),
            "duration_standard_error": _se(durations),
            "fall_count": sum(r["fall"] for r in self.records),
            "fall_rate": _mean([int(r["fall"]) for r in self.records]),
            "timeout_count": sum(r["timeout"] for r in self.records),
            "timeout_rate": _mean([int(r["timeout"]) for r in self.records]),
            "numerical_failure_count": sum(r["numerical_failure"] for r in self.records),
            "standard_error_method": "sample standard deviation / sqrt(n); null when n < 2; numerical failures excluded from success metrics",
        }
