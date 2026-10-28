"""Continuous spin teacher. Privileged observations are NOT deployment inputs."""

import json
import math
from pathlib import Path

import torch

from ghand.common.cube_env import GHandCubeEnv
from ghand.common.inhand_env import unscale
from ghand.tasks.cube_spin.randomization import sample_range, cube_inertia, friction_buckets
from ghand.tasks.cube_spin.core import (
    command_limits,
    rotation_increment,
    rotation_vector,
    sdk_joint_contract,
    spin_reward,
)


class GHandSpinEnv(GHandCubeEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.target_speed <= 0 or cfg.action_step_rad <= 0:
            raise ValueError("target_speed and action_step_rad must be positive")
        if not math.isfinite(cfg.spin_position_penalty_scale) or cfg.spin_position_penalty_scale < 0:
            raise ValueError("Position penalty must be finite and nonnegative")
        if (cfg.cube_size_range_m is None) != (cfg.cube_mass_range_kg is None):
            raise ValueError("Set both cube size and mass ranges, or neither")
        if cfg.cube_size_range_m is not None:
            sample_range(cfg.cube_size_range_m, 0)
            sample_range(cfg.cube_mass_range_kg, 0)
            cfg.scene.replicate_physics = False
            cfg.scene.clone_in_fabric = False
        if cfg.cube_static_friction_range is not None:
            self._friction_buckets = friction_buckets(
                cfg.cube_static_friction_range,
                cfg.cube_dynamic_friction_ratio_range,
                cfg.friction_bucket_count,
                seed=int(cfg.seed) + 200003,
            )
            cfg.scene.replicate_physics = False
            cfg.scene.clone_in_fabric = False
        super().__init__(cfg, render_mode, **kwargs)
        if cfg.cube_static_friction_range is not None:
            hand_material = self.hand.root_physx_view.get_material_properties()
            if not torch.allclose(hand_material[..., :2], torch.ones_like(hand_material[..., :2])):
                raise ValueError("Cube pair friction assumes hand static/dynamic friction = 1")
            self._cube_material_cpu = self.object.root_physx_view.get_material_properties().clone()
            self.cube_friction = torch.zeros((self.num_envs, 2), device=self.device)
        if cfg.cube_size_range_m is not None:
            self.cube_edge_m = self._cube_edges_cpu.to(self.device)
            self.cube_mass_kg = torch.zeros(self.num_envs, device=self.device)
            self._cube_mass_cpu = self.object.root_physx_view.get_masses().clone()
            self._cube_inertia_cpu = self.object.root_physx_view.get_inertias().clone()
        manifest = json.loads(Path(cfg.robot_cfg.spawn.usd_path).with_suffix(".manifest.json").read_text())
        self.joint_contract = sdk_joint_contract(manifest, json.loads(Path(cfg.hardware_config).read_text()))
        # Fixed semantic order, independent of PhysX's internal joint ordering.
        self.actuated_dof_indices = [self.hand.joint_names.index(j["name"]) for j in self.joint_contract]
        for index, joint in zip(self.actuated_dof_indices, self.joint_contract):
            self.hand_dof_lower_limits[:, index] = joint["lower"]
            self.hand_dof_upper_limits[:, index] = joint["upper"]
        command_lower, command_upper = command_limits(
            self.joint_contract,
            cfg.command_upper_deg,
            cfg.command_margin_fraction,
        )
        self.command_lower_limits = torch.tensor(command_lower, device=self.device)
        self.command_upper_limits = torch.tensor(command_upper, device=self.device)
        self.axis = torch.tensor(cfg.spin_axis, device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(self.axis).all()) or float(self.axis.norm()) < 1e-6:
            raise ValueError("spin_axis must be finite and nonzero")
        self.axis /= self.axis.norm()
        self.goal_markers.set_visibility(False)
        self.actions = torch.zeros((self.num_envs, 13), device=self.device)
        self.action_delta = torch.zeros_like(self.actions)
        self.previous_object_quat = self.object.data.root_quat_w.clone()
        self.net_angle = torch.zeros(self.num_envs, device=self.device)
        self.reverse_angle = torch.zeros_like(self.net_angle)
        self.off_axis_angle = torch.zeros_like(self.net_angle)
        self._spin_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()
        if self.cfg.cube_size_range_m is not None:
            from pxr import Gf, UsdGeom

            # Separate size stream: reset(seed) must not repeat the size draws as masses.
            self._cube_edges_cpu = sample_range(
                self.cfg.cube_size_range_m,
                self.num_envs,
                seed=int(self.cfg.seed) + 100003,
            )
            # The pinned DexCube asset is 60 mm. Author each clone before physics parsing.
            for index, edge in enumerate(self._cube_edges_cpu.tolist()):
                prim = self.sim.stage.GetPrimAtPath(f"/World/envs/env_{index}/object")
                scale = next(
                    op
                    for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale
                )
                scale.Set(Gf.Vec3d(*([edge / 0.06] * 3)))
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        if self.cfg.cube_static_friction_range is not None:
            from pxr import UsdShade
            import isaaclab.sim as sim_utils

            # Parent binding also reaches instanceable collider descendants.
            material_path = "/World/SpinCubeMaterial"
            material_cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="multiply",
            )
            material_cfg.func(material_path, material_cfg)
            material = UsdShade.Material(self.sim.stage.GetPrimAtPath(material_path))
            for index in range(self.num_envs):
                prim = self.sim.stage.GetPrimAtPath(f"/World/envs/env_{index}/object")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _reset_target_pose(self, env_ids):
        # No target orientation: rotation continues past every full revolution.
        self.reset_goal_buf[env_ids] = False

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        if self.cfg.cube_size_range_m is not None:
            cpu_ids = env_ids.to(device="cpu", dtype=torch.int32)
            mass = sample_range(self.cfg.cube_mass_range_kg, len(cpu_ids))
            self._cube_mass_cpu[cpu_ids, 0] = mass
            self._cube_inertia_cpu[cpu_ids] = cube_inertia(self._cube_edges_cpu[cpu_ids], mass)
            self.object.root_physx_view.set_masses(self._cube_mass_cpu, cpu_ids)
            self.object.root_physx_view.set_inertias(self._cube_inertia_cpu, cpu_ids)
            self.cube_mass_kg[env_ids] = mass.to(self.device)
        if self.cfg.cube_static_friction_range is not None:
            cpu_ids = env_ids.to(device="cpu", dtype=torch.int32)
            bucket_ids = torch.randint(len(self._friction_buckets), (len(cpu_ids),), device="cpu")
            material = self._friction_buckets[bucket_ids]
            self._cube_material_cpu[cpu_ids] = material[:, None, :]
            self.object.root_physx_view.set_material_properties(self._cube_material_cpu, cpu_ids)
            self.cube_friction[env_ids] = material[:, :2].to(self.device)
        # Reset may sample a physical pose outside the narrower command interval.
        # Leave that measured pose intact, but replace targets before physics resumes.
        ids = self.actuated_dof_indices
        targets = self.cur_targets[env_ids].clone()
        targets[:, ids] = torch.clamp(targets[:, ids], self.command_lower_limits, self.command_upper_limits)
        self.cur_targets[env_ids] = targets
        self.prev_targets[env_ids] = targets
        self.hand_dof_targets[env_ids] = targets
        self.hand.set_joint_position_target(targets, env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        self.previous_object_quat[env_ids] = self.object.data.root_quat_w[env_ids]
        self.net_angle[env_ids] = 0.0
        self.reverse_angle[env_ids] = 0.0
        self.off_axis_angle[env_ids] = 0.0
        self._spin_dropped[env_ids] = False

    def _pre_physics_step(self, actions):
        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError("Nonfinite policy action")
        next_actions = actions.clamp(-1.0, 1.0)
        self.action_delta = next_actions - self.actions
        self.actions = next_actions.clone()
        ids = self.actuated_dof_indices
        self.cur_targets[:, ids] = torch.clamp(
            self.cur_targets[:, ids] + self.cfg.action_step_rad * self.actions,
            self.command_lower_limits,
            self.command_upper_limits,
        )
        self.prev_targets.copy_(self.cur_targets)

    def _apply_action(self):
        # Compute increments ONCE per control step, not once per physics substep.
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    def _get_dones(self):
        self._compute_intermediate_values()
        for state in (self.hand_dof_pos, self.hand_dof_vel, self.object_pos, self.object_rot, self.object_velocities):
            if not bool(torch.isfinite(state).all()):
                raise RuntimeError("Nonfinite physics state in spin task")
        if bool((self.hand_dof_pos.abs() > 10.0).any()) or bool((self.hand_dof_vel.abs() > 1000.0).any()):
            raise RuntimeError("GHand spin physics diverged")
        self._spin_dropped = self.object_pos[:, 2] < self.in_hand_pos[:, 2] - 0.055
        if self.cfg.spin_terminate_on_distance:
            self._spin_dropped |= (
                torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1) > self.cfg.fall_dist
            )
        return self._spin_dropped, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        axial, off_axis = rotation_increment(self.previous_object_quat, self.object_rot, self.axis)
        # Contact-solver velocities may be nonzero for an almost stationary cube.
        # Reward the same actual orientation progress that evaluation measures.
        pose_angvel = rotation_vector(self.previous_object_quat, self.object_rot) / self.step_dt
        self.previous_object_quat.copy_(self.object_rot)
        active = (self.episode_length_buf * self.step_dt) > self.cfg.spin_warmup_s
        valid_motion = active & ~self._spin_dropped
        axial = torch.where(valid_motion, axial, 0.0)
        self.net_angle += axial
        self.reverse_angle += torch.clamp(-axial, min=0.0)
        self.off_axis_angle += torch.where(valid_motion, off_axis, 0.0)
        terms = spin_reward(
            pose_angvel,
            self.object_linvel,
            self.object_pos - self.in_hand_pos,
            self.action_delta,
            self.axis,
            self.cfg.target_speed,
            position_penalty_scale=self.cfg.spin_position_penalty_scale,
        )
        reward = terms["total"] - terms["rotation"] + torch.where(valid_motion, terms["rotation"], 0.0)
        reward += self.cfg.spin_survival_reward
        reward += self._spin_dropped * self.cfg.spin_drop_penalty
        self.extras.setdefault("log", {})
        for name, value in terms.items():
            self.extras["log"]["spin/reward_" + name] = value.mean()
        self.extras["log"]["spin/net_turns"] = (self.net_angle / (2.0 * math.pi)).mean()
        self.extras["log"]["spin/reverse_turns"] = (self.reverse_angle / (2.0 * math.pi)).mean()
        self.extras["log"]["spin/angular_speed"] = (pose_angvel @ self.axis).mean()
        self.extras["log"]["spin/reported_angular_speed"] = (self.object_angvel @ self.axis).mean()
        self.extras["log"]["spin/dropped_step_fraction"] = self._spin_dropped.float().mean()
        self.extras["log"]["spin/height"] = self.object_pos[:, 2].mean()
        # Snapshot BEFORE DirectRLEnv automatically resets terminal environments.
        self.extras["spin_metrics"] = {
            "net_turns": self.net_angle.clone() / (2.0 * math.pi),
            "reverse_turns": self.reverse_angle.clone() / (2.0 * math.pi),
            "off_axis_radians": self.off_axis_angle.clone(),
            "seconds": self.episode_length_buf.clone() * self.step_dt,
            "dropped": self._spin_dropped.clone(),
        }
        if self.cfg.cube_size_range_m is not None:
            self.extras["spin_metrics"]["cube_edge_m"] = self.cube_edge_m.clone()
            self.extras["spin_metrics"]["cube_mass_kg"] = self.cube_mass_kg.clone()
        if self.cfg.cube_static_friction_range is not None:
            self.extras["spin_metrics"]["cube_static_friction"] = self.cube_friction[:, 0].clone()
            self.extras["spin_metrics"]["cube_dynamic_friction"] = self.cube_friction[:, 1].clone()
        return reward

    def _get_observations(self):
        ids = self.actuated_dof_indices
        lower, upper = self.hand_dof_lower_limits[:, ids], self.hand_dof_upper_limits[:, ids]
        obs = torch.cat(
            (
                unscale(self.hand_dof_pos[:, ids], lower, upper),
                self.hand_dof_vel[:, ids] * 0.2,
                unscale(self.cur_targets[:, ids], lower, upper),
                self.object_pos - self.in_hand_pos,
                self.object_rot,
                self.object_linvel,
                self.object_angvel * 0.2,
                (self.fingertip_pos - self.in_hand_pos[:, None, :]).reshape(self.num_envs, -1),
                torch.full((self.num_envs, 1), self.cfg.target_speed, device=self.device),
            ),
            dim=-1,
        )
        return {"policy": obs}
