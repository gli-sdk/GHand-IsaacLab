# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Cube reorientation configuration shared with the continuous-spin task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from ghand.common.local_assets import bundled_asset

from ghand.common.hand_asset import ACTIVE_JOINTS, GHAND_CFG, MIMIC_JOINTS


@configclass
class GHandCubeEnvCfg(DirectRLEnvCfg):
    """13 active joints and 5 followers manipulating a 36 mm, 26.45 g cube."""

    # Policy interface and timing.
    decimation = 2
    episode_length_s = 10.0
    action_space = 13
    observation_space = 138
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = GHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    actuated_joint_names = ACTIVE_JOINTS
    fingertip_body_names = ["LF_DIP_link", "RF_DIP_link", "MF_DIP_link", "FF_DIP_link", "Thumb_IP_link"]
    mimic_joints = MIMIC_JOINTS

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=bundled_asset("Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
            scale=(0.6, 0.6, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.09, 0.60), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=bundled_asset("Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
                scale=(0.6, 0.6, 0.6),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )

    # Reset noise is applied within feasible joint limits.
    correct_joint_reset: bool = True
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

    # Optional curriculum controls; zero uses fully random target orientations.
    rotation_progress_reward_scale: float = 0.0
    relative_goal_probability: float = 0.0
    relative_goal_angle_min: float = 0.4
    relative_goal_angle_max: float = 0.8
    relative_goal_margin: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        # Explicit mass takes precedence over the value authored in the cube USD.
        self.object_cfg.spawn.mass_props.mass = 567.0 * 0.036**3
