"""Two freely contacting balls in a fixed, palm-up GHand. No hardware access."""

import json
import math
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from ghand.tasks.dual_ball_rotation.env_cfg import DualBallRotationEnvCfg
from ghand.common.hand_asset import MIMIC_JOINTS, USD_PATH
from ghand.tasks.dual_ball_rotation.joint_contract import sdk_joint_contract
from ghand.tasks.dual_ball_rotation.orbit import (
    orbit_delta,
    novel_progress,
    orbit_goal_offsets,
    stall_fraction,
    shared_lift_fraction,
)
from ghand.tasks.dual_ball_rotation.perturbations import sample_domains
from ghand.tasks.dual_ball_rotation.palm_frame import rotate_palm, tilted_quaternion, sample_tilts

ROOT = Path(__file__).resolve().parents[4]


class DualBallRotationEnv(DirectRLEnv):
    cfg: DualBallRotationEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.domain_randomization:
            cfg.scene.replicate_physics = False
        super().__init__(cfg, render_mode, **kwargs)
        contract = sdk_joint_contract(
            json.loads(USD_PATH.with_suffix(".manifest.json").read_text()),
            json.loads((ROOT / "config/dual_ball_rotation/hand.json").read_text()),
        )
        self.ids = [self.hand.joint_names.index(j["name"]) for j in contract]
        self.lo = torch.tensor([j["lower"] for j in contract], device=self.device)
        self.hi = torch.tensor([j["upper"] for j in contract], device=self.device)
        self.mimics = [
            (self.hand.joint_names.index(c), self.hand.joint_names.index(p), a, b)
            for c, (p, a, b) in MIMIC_JOINTS.items()
        ]
        self.initial_xy_offsets = None
        if cfg.initial_xy_offsets_m is not None:
            self.initial_xy_offsets = torch.tensor(cfg.initial_xy_offsets_m, device=self.device)
            if self.initial_xy_offsets.shape != (self.num_envs, 2) or not torch.isfinite(self.initial_xy_offsets).all():
                raise ValueError("Initial offsets must be finite with shape (num_envs, 2)")
        if cfg.rotation_direction not in (-1, 1):
            raise ValueError("Rotation direction must be -1 or +1")
        self.tilt_generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 104729)
        sample_tilts(1, cfg.palm_tilt_range_deg, torch.Generator())  # Validate before reset.
        if not math.isfinite(cfg.palm_side_tilt_deg) or abs(cfg.palm_side_tilt_deg) > 90:
            raise ValueError("Side tilt must be finite and within +/-90 degrees")
        self._tilted = tuple(cfg.palm_tilt_range_deg) != (0.0, 0.0) or cfg.palm_side_tilt_deg != 0.0
        self.tilt_deg = torch.zeros(self.num_envs, device=self.device)
        self.side_tilt_deg = (
            torch.full_like(self.tilt_deg, float(cfg.palm_side_tilt_deg)) if cfg.palm_side_tilt_deg != 0.0 else None
        )
        self.palm_pivot = torch.tensor(cfg.robot_cfg.init_state.pos, device=self.device)
        self.domain_generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        self.episode_offsets = torch.zeros((self.num_envs, 2), device=self.device)
        self.friction_scales = torch.ones(self.num_envs)
        self.domain_scales = torch.ones(self.num_envs, device=self.device)
        self._material_buffers = [ball.root_physx_view.get_material_properties().clone() for ball in self.balls]
        self._domain_mode_at_creation = cfg.domain_randomization
        self.domain_is_nominal = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if cfg.domain_randomization:
            initial_shift, scales = sample_domains(
                self.num_envs,
                cfg.placement_jitter_mm,
                cfg.friction_scale_range,
                cfg.nominal_fraction,
                self.domain_generator,
            )
            self.friction_scales = scales
            self.domain_scales = scales.to(self.device)
            self.domain_is_nominal = ((initial_shift == 0).all(-1) & (scales == 1)).to(self.device)
            cpu_ids = torch.arange(self.num_envs)
            for ball, props in zip(self.balls, self._material_buffers):
                props[:, :, 0] = cfg.ball_static_friction * scales[:, None]
                props[:, :, 1] = cfg.ball_dynamic_friction * scales[:, None]
                ball.root_physx_view.set_material_properties(props, cpu_ids)
        self.center = torch.tensor(cfg.orbit_center, device=self.device)
        self.radii = torch.tensor(cfg.orbit_radii, device=self.device)
        self.targets = torch.zeros((self.num_envs, 13), device=self.device)
        self.actions = torch.zeros_like(self.targets)
        self.change = torch.zeros_like(self.targets)
        self.net = torch.zeros(self.num_envs, device=self.device)
        self.high = torch.zeros_like(self.net)
        self.reverse = torch.zeros_like(self.net)
        self.stall_history = None
        if cfg.stall_penalty_scale:
            if (
                not all(
                    math.isfinite(v)
                    for v in (cfg.stall_penalty_scale, cfg.stall_window_s, cfg.stall_speed_rad_s, cfg.stall_grace_s)
                )
                or min(cfg.stall_penalty_scale, cfg.stall_window_s, cfg.stall_speed_rad_s) <= 0
                or cfg.stall_grace_s < 0
            ):
                raise ValueError("Require positive finite stall weight/window/speed and nonnegative grace")
            self.stall_history = torch.zeros(
                (self.num_envs, max(1, round(cfg.stall_window_s / self.step_dt))), device=self.device
            )
            self.stall_sum = torch.zeros_like(self.net)
            self.stall_pointer = 0
        lift_values = (cfg.shared_lift_penalty_scale, cfg.shared_lift_tolerance_m, cfg.shared_lift_transition_m)
        if not all(math.isfinite(v) for v in lift_values) or min(lift_values[:2]) < 0 or lift_values[2] <= 0:
            raise ValueError("Require finite nonnegative shared-lift weight/tolerance and positive transition")
        self.initial_world_z = torch.zeros((self.num_envs, 2), device=self.device)
        self.shared_lift_steps = torch.zeros_like(self.net)
        self.max_shared_lift = torch.zeros_like(self.net)
        self.phase = torch.zeros_like(self.net)
        self.goals = torch.zeros((self.num_envs, 2, 2), device=self.device)
        self.individual = torch.zeros((self.num_envs, 2), device=self.device)
        self.valid_steps = torch.zeros_like(self.net)
        self._refresh()

    def seed(self, seed=-1):
        actual = super().seed(seed)
        if hasattr(self, "domain_generator"):
            self.domain_generator.manual_seed(actual)
        if hasattr(self, "tilt_generator"):
            self.tilt_generator.manual_seed(actual + 104729)
        return actual

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.contact_pair_friction = None
        self.balls = []
        for i, color in enumerate(((0.85, 0.12, 0.10), (0.1, 0.35, 0.9))):
            b = RigidObject(
                RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Ball{i}",
                    spawn=sim_utils.SphereCfg(
                        radius=self.cfg.ball_radius,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            max_depenetration_velocity=1.0,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=2,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                        mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.ball_mass),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            static_friction=self.cfg.ball_static_friction,
                            dynamic_friction=self.cfg.ball_dynamic_friction,
                            restitution=0.0,
                            friction_combine_mode="multiply",
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.016 if i == 0 else -0.016, -0.077, 0.54)),
                )
            )
            self.balls.append(b)
        # Local primitive floor avoids network downloads; it never supports balls during a valid episode.
        floor = sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        )
        floor.func("/World/Floor", floor, translation=(0.0, 0.0, -0.025))
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.domain_randomization:
            from pxr import UsdShade

            # Each collider needs its own material handle: mutating shared clone materials
            # would couple otherwise independent episodes after reset/reassignment.
            for index in range(self.num_envs):
                for ball_id in range(2):
                    path = f"/World/DualBallMaterials/env_{index}_ball_{ball_id}"
                    material_cfg = sim_utils.RigidBodyMaterialCfg(
                        static_friction=self.cfg.ball_static_friction,
                        dynamic_friction=self.cfg.ball_dynamic_friction,
                        restitution=0.0,
                        friction_combine_mode="multiply",
                    )
                    material_cfg.func(path, material_cfg)
                    prim = self.sim.stage.GetPrimAtPath(f"/World/envs/env_{index}/Ball{ball_id}")
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        UsdShade.Material(self.sim.stage.GetPrimAtPath(path)),
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        materialPurpose="physics",
                    )
        self.scene.articulations["hand"] = self.hand
        for i, b in enumerate(self.balls):
            self.scene.rigid_objects[f"ball{i}"] = b
        self.scene.filter_collisions(global_prim_paths=["/World/Floor"])
        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.9, 0.9, 0.9))
        light.func("/World/Light", light)

    def _refresh(self):
        self.pos = torch.stack([b.data.root_pos_w - self.scene.env_origins for b in self.balls], 1)
        self.world_z = self.pos[..., 2]
        self.vel = torch.stack([b.data.root_lin_vel_w for b in self.balls], 1)
        self.omega = torch.stack([b.data.root_ang_vel_w for b in self.balls], 1)
        if self._tilted:
            self.pos = (
                rotate_palm(self.pos - self.palm_pivot, self.tilt_deg, self.side_tilt_deg, inverse=True)
                + self.palm_pivot
            )
            self.vel = rotate_palm(self.vel, self.tilt_deg, self.side_tilt_deg, inverse=True)
            self.omega = rotate_palm(self.omega, self.tilt_deg, self.side_tilt_deg, inverse=True)

    def _update_goals(self, ids):
        offset = orbit_goal_offsets(self.phase[ids], self.radii, self.cfg.rotation_direction)
        self.goals[ids, 0] = self.center[:2] + offset
        self.goals[ids, 1] = self.center[:2] - offset

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError("Nonfinite action")
        self._refresh()
        self.previous_pair = (self.pos[:, 0] - self.pos[:, 1]).clone()
        self.previous_relative = (self.pos - self.center).clone()
        self.old_error = (self.pos[..., :2] - self.goals).norm(dim=-1).mean(-1)
        self.change = actions.clamp(-1, 1) - self.actions
        self.actions = actions.clamp(-1, 1)
        self.targets = (self.targets + self.cfg.action_step_rad * self.actions).clamp(self.lo, self.hi)

    def _apply_action(self):
        applied = self.targets
        self.hand.set_joint_position_target(applied, joint_ids=self.ids)

    def _get_dones(self):
        self._refresh()
        if not (torch.isfinite(self.pos).all() and torch.isfinite(self.hand.data.joint_pos).all()):
            raise RuntimeError("Nonfinite physics state")
        # A generous hand workspace, not a penalty for changing the midpoint.
        dropped = (self.pos[..., 2] < 0.485).any(-1)
        escaped = ((self.pos[..., 0].abs() > 0.095) | (self.pos[..., 1] < -0.205) | (self.pos[..., 1] > 0.015)).any(-1)
        return dropped | escaped, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        diameter = 2 * self.cfg.ball_radius
        d, valid = orbit_delta(
            self.previous_pair, self.pos[:, 0] - self.pos[:, 1], diameter, self.cfg.rotation_direction
        )
        valid &= ~self.reset_terminated
        d = torch.where(valid, d, 0.0)
        self.net += d
        self.reverse += (-d).clamp_min(0)
        self.valid_steps += valid.float()
        stall_cost = 0.0
        if self.stall_history is not None:
            self.stall_sum += d - self.stall_history[:, self.stall_pointer]
            self.stall_history[:, self.stall_pointer] = d
            self.stall_pointer = (self.stall_pointer + 1) % self.stall_history.shape[1]
            speed = self.stall_sum / (self.stall_history.shape[1] * self.step_dt)
            cost_fraction = stall_fraction(speed, self.cfg.stall_speed_rad_s)
            cost_fraction *= self.episode_length_buf * self.step_dt >= self.cfg.stall_grace_s
            stall_cost = self.cfg.stall_penalty_scale * cost_fraction
        common_rise = (self.world_z - self.initial_world_z).amin(-1)
        self.shared_lift_steps += ((common_rise > self.cfg.shared_lift_tolerance_m) & ~self.reset_terminated).float()
        self.max_shared_lift = torch.maximum(self.max_shared_lift, common_rise.clamp_min(0.0))
        lift_fraction = shared_lift_fraction(
            self.world_z, self.initial_world_z, self.cfg.shared_lift_tolerance_m, self.cfg.shared_lift_transition_m
        )
        lift_cost = self.cfg.shared_lift_penalty_scale * lift_fraction
        cap = self.cfg.progress_speed_cap_rad_s
        credit, self.high = novel_progress(self.net, self.high, cap * self.step_dt if cap > 0 else None)
        # Each ball must also travel around the palm center in independent evaluation.
        old = self.previous_relative.clone()
        new = self.pos - self.center
        old[..., 2] = 0.0
        new = new.clone()
        new[..., 2] = 0.0
        single, _ = orbit_delta(old, new, diameter * 0.4, self.cfg.rotation_direction)
        self.individual += torch.where(valid[:, None], single, 0.0)
        errors = (self.pos[..., :2] - self.goals).norm(dim=-1)
        error = errors.mean(-1)
        hit = (errors.max(-1).values < self.cfg.waypoint_tolerance) & (self.net >= self.phase - 0.15) & valid
        reward = (
            self.cfg.progress_scale * credit
            + self.cfg.waypoint_progress_scale * ((self.old_error - error) / 0.01).clamp(-2, 2)
            + self.cfg.waypoint_bonus * hit.float()
            - self.cfg.action_change_scale * self.change.square().sum(-1)
            - self.cfg.drop_penalty * self.reset_terminated.float()
            - stall_cost
            - lift_cost
        )
        self.phase += hit.float() * self.cfg.waypoint_step
        self._update_goals(hit.nonzero(as_tuple=False).flatten())
        metrics = {
            "seconds": self.episode_length_buf * self.step_dt,
            "net_turns": self.net / (2 * math.pi),
            "ball0_turns": self.individual[:, 0] / (2 * math.pi),
            "ball1_turns": self.individual[:, 1] / (2 * math.pi),
            "reverse_turns": self.reverse / (2 * math.pi),
            "dropped": self.reset_terminated,
            "valid_fraction": self.valid_steps / self.episode_length_buf.clamp_min(1),
            "world_signed_net_turns": self.cfg.rotation_direction * self.net / (2 * math.pi),
            "initial_offset_x_m": self.episode_offsets[:, 0],
            "initial_offset_y_m": self.episode_offsets[:, 1],
            "friction_scale": self.domain_scales,
            "nominal_domain": self.domain_is_nominal,
            "palm_tilt_deg": self.tilt_deg,
            "shared_lift_fraction": self.shared_lift_steps / self.episode_length_buf.clamp_min(1),
            "max_shared_lift_m": self.max_shared_lift,
        }
        self.extras["rotation_metrics"] = {k: v.clone() for k, v in metrics.items()}
        self.extras["terminal_positions"] = self.pos.clone()
        self.extras["terminal_q"] = self.hand.data.joint_pos.clone()
        self.extras["terminal_targets"] = self.targets.clone()
        self.extras["terminal_actions"] = self.actions.clone()
        self.extras["log"] = {
            "net_turns": metrics["net_turns"].mean(),
            "goal_error_m": error.mean(),
            "valid_fraction": metrics["valid_fraction"].mean(),
        }
        if self.stall_history is not None:
            self.extras["log"]["stall_fraction"] = cost_fraction.mean()
        self.extras["log"]["shared_lift_cost"] = lift_cost.mean()
        return reward

    def _get_observations(self):
        self._refresh()
        q = self.hand.data.joint_pos[:, self.ids]

        def normalize(value):
            return 2 * (value - self.lo) / (self.hi - self.lo) - 1

        obs = torch.cat(
            (
                normalize(q),
                self.hand.data.joint_vel[:, self.ids] * 0.2,
                normalize(self.targets),
                ((self.pos - self.center) / 0.05).flatten(1),
                self.vel.flatten(1),
                self.omega.flatten(1) * 0.1,
                ((self.goals - self.pos[..., :2]) / 0.05).flatten(1),
                ((self.high - self.net) / math.pi)[:, None],
                ((self.phase - self.net) / math.pi)[:, None],
            ),
            -1,
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        if self._tilted:
            self.tilt_deg[env_ids] = sample_tilts(n, self.cfg.palm_tilt_range_deg, self.tilt_generator).to(self.device)
            root = self.hand.data.default_root_state[env_ids, :7].clone()
            root[:, :3] += self.scene.env_origins[env_ids]
            root[:, 3:7] = tilted_quaternion(
                root[:, 3:7],
                self.tilt_deg[env_ids],
                self.side_tilt_deg[env_ids] if self.side_tilt_deg is not None else None,
            )
            self.hand.write_root_pose_to_sim(root, env_ids=env_ids)
        if self.cfg.domain_randomization != self._domain_mode_at_creation:
            raise ValueError("Domain mode is fixed at construction; use a separate nominal environment")
        if self.cfg.domain_randomization:
            shift, _ = sample_domains(
                n, self.cfg.placement_jitter_mm, self.cfg.friction_scale_range, 0.0, self.domain_generator
            )
            shift[self.domain_is_nominal[env_ids].cpu()] = 0.0
        else:
            shift = torch.zeros((n, 2))
        self.episode_offsets[env_ids] = shift.to(self.device)
        if self.initial_xy_offsets is not None:
            self.episode_offsets[env_ids] += self.initial_xy_offsets[env_ids]
        q = self.hand.data.default_joint_pos[env_ids].clone()
        q[:, self.ids] = q[:, self.ids].clamp(self.lo, self.hi)
        for c, p, a, b in self.mimics:
            q[:, c] = a * q[:, p] + b
        self.hand.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.targets[env_ids] = q[:, self.ids]
        self.hand.set_joint_position_target(self.targets[env_ids], joint_ids=self.ids, env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.change[env_ids] = 0.0
        self.net[env_ids] = 0.0
        self.high[env_ids] = 0.0
        self.reverse[env_ids] = 0.0
        if self.stall_history is not None:
            self.stall_history[env_ids] = 0.0
            self.stall_sum[env_ids] = 0.0
        self.individual[env_ids] = 0.0
        self.valid_steps[env_ids] = 0.0
        self.shared_lift_steps[env_ids] = 0.0
        self.max_shared_lift[env_ids] = 0.0
        self.phase[env_ids] = self.cfg.waypoint_step
        self._update_goals(env_ids)
        for i, b in enumerate(self.balls):
            state = b.data.default_root_state[env_ids].clone()
            state[:, :3] = self.center + self.scene.env_origins[env_ids]
            state[:, 0] += (1 if i == 0 else -1) * self.cfg.orbit_radii[0]
            state[:, 2] = self.cfg.initial_ball_heights[i] + self.scene.env_origins[env_ids, 2]
            state[:, :2] += self.episode_offsets[env_ids]
            if self._tilted:
                relative = state[:, :3] - self.scene.env_origins[env_ids] - self.palm_pivot
                state[:, :3] = (
                    rotate_palm(
                        relative,
                        self.tilt_deg[env_ids],
                        self.side_tilt_deg[env_ids] if self.side_tilt_deg is not None else None,
                    )
                    + self.palm_pivot
                    + self.scene.env_origins[env_ids]
                )
            self.initial_world_z[env_ids, i] = state[:, 2] - self.scene.env_origins[env_ids, 2]
            state[:, 7:] = 0.0
            b.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
            b.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
