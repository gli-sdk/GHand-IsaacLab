"""Shared GHand cube reset and native follower-joint handling."""

import torch
from .inhand_env import InHandManipulationEnv
from .kinematics import apply_mimic_state, feasible_parent_limits, validate_relative_goal_config


class GHandCubeEnv(InHandManipulationEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        validate_relative_goal_config(
            cfg.relative_goal_probability,
            cfg.relative_goal_angle_min,
            cfg.relative_goal_angle_max,
            cfg.relative_goal_margin,
            cfg.success_tolerance,
        )
        self._defer_goal_reset = False
        super().__init__(cfg, render_mode, **kwargs)
        self._mimic_indices = []
        for child_name, (parent_name, multiplier, offset) in cfg.mimic_joints.items():
            child = self.hand.joint_names.index(child_name)
            parent = self.hand.joint_names.index(parent_name)
            self._mimic_indices.append((child, parent, multiplier, offset))
            lower, upper = feasible_parent_limits(
                self.hand_dof_lower_limits[:, parent],
                self.hand_dof_upper_limits[:, parent],
                self.hand_dof_lower_limits[:, child],
                self.hand_dof_upper_limits[:, child],
                multiplier,
                offset,
            )
            self.hand_dof_lower_limits[:, parent] = lower
            self.hand_dof_upper_limits[:, parent] = upper
        self.goal_pos[:] = torch.tensor([-0.16, -0.14, 0.72], device=self.device)

    def _reset_idx(self, env_ids):
        # The parent resets goals before objects. Relative targets need the NEW pose.
        self._defer_goal_reset = self.cfg.relative_goal_probability > 0.0
        try:
            super()._reset_idx(env_ids)
        finally:
            self._defer_goal_reset = False
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        position, velocity = apply_mimic_state(
            self.hand.data.joint_pos[env_ids],
            self.hand.data.joint_vel[env_ids],
            self._mimic_indices,
        )
        self.prev_targets[env_ids] = position
        self.cur_targets[env_ids] = position
        self.hand_dof_targets[env_ids] = position
        self.hand.set_joint_position_target(position, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
        self._compute_intermediate_values()
        if self.cfg.relative_goal_probability > 0.0:
            self._reset_target_pose(env_ids)
