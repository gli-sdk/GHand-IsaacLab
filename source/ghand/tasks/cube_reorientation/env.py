# SPDX-License-Identifier: BSD-3-Clause
"""Native-mimic GHand cube reorientation using the updated vendor model."""

import torch
from ghand.common.inhand_env import rotation_distance
from ghand.common.cube_env import GHandCubeEnv
from ghand.common.kinematics import (
    sample_relative_goal_rotations,
)


class GHandReorientationEnv(GHandCubeEnv):
    """Reorientation with native follower-joint constraints and optional goal curriculum."""

    def _reset_target_pose(self, env_ids):
        probability = self.cfg.relative_goal_probability
        if probability == 0.0:
            return super()._reset_target_pose(env_ids)
        if self._defer_goal_reset:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(env_ids):
            return
        if probability < 1.0:
            relative = torch.rand(len(env_ids), device=self.device) < probability
            full_random_ids = env_ids[~relative]
            if len(full_random_ids):
                super()._reset_target_pose(full_random_ids)
            env_ids = env_ids[relative]
        if not len(env_ids):
            return
        current = self.object.data.root_quat_w[env_ids]
        goals = sample_relative_goal_rotations(
            current,
            self.cfg.relative_goal_angle_min,
            self.cfg.relative_goal_angle_max,
        )
        error = rotation_distance(current, goals)
        valid = (
            (error >= self.cfg.relative_goal_angle_min - 1e-5)
            & (error <= self.cfg.relative_goal_angle_max + 1e-5)
            & (error > self.cfg.success_tolerance)
        )
        if not bool(valid.all()):
            raise RuntimeError("Relative goal violates its angle interval or is already within success tolerance")
        self.goal_rot[env_ids] = goals
        self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)
        self.reset_goal_buf[env_ids] = 0

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._rotation_error_before = rotation_distance(self.object_rot, self.goal_rot)

    def _get_rewards(self):
        # Capture the same goal before the base reward resamples successful targets.
        rotation_error = rotation_distance(self.object_rot, self.goal_rot)
        reward = super()._get_rewards()
        if self.cfg.rotation_progress_reward_scale:
            progress = self._rotation_error_before - rotation_error
            reward = reward + self.cfg.rotation_progress_reward_scale * progress
        # Fail loudly on solver explosions instead of learning from invalid physics.
        if self.common_step_counter % 64 == 0:
            q = self.hand.data.joint_pos
            v = self.hand.data.joint_vel
            valid = (
                torch.isfinite(q).all() & torch.isfinite(v).all() & (q.abs().max() < 10.0) & (v.abs().max() < 1000.0)
            )
            if not bool(valid):
                raise RuntimeError("GHand physics diverged: inspect joint states and native mimic constraints")
            residuals = torch.stack([torch.abs(q[:, c] - a * q[:, p] - b) for c, p, a, b in self._mimic_indices])
            self.extras["log"]["rotation_error_mean_rad"] = rotation_error.mean()
            self.extras["log"]["rotation_progress_mean_rad"] = (self._rotation_error_before - rotation_error).mean()
            self.extras["log"]["mimic_error_mean_rad"] = residuals.mean()
            self.extras["log"]["mimic_error_max_rad"] = residuals.max()
        return reward
