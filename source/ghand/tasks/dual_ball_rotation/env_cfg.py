from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from ghand.common.hand_asset import GHAND_CFG


@configclass
class DualBallRotationEnvCfg(DirectRLEnvCfg):
    decimation = 6
    episode_length_s = 30.0
    action_space = 13
    observation_space = 63
    state_space = 0
    action_step_rad = 0.075
    rotation_direction = 1
    palm_tilt_range_deg = (0.0, 0.0)  # Positive lowers fingertips; sampled once per reset.
    palm_side_tilt_deg = 0.0  # Negative lowers the right-hand thumb side.
    domain_randomization = False
    placement_jitter_mm = 2.0
    friction_scale_range = (0.75, 1.25)
    nominal_fraction = 0.25
    ball_radius = 0.015
    ball_mass = 0.015
    ball_static_friction = 0.8
    ball_dynamic_friction = 0.6
    orbit_center = (0.0, -0.077, 0.537)
    initial_ball_heights = (0.53575, 0.5375)
    orbit_radii = (0.016, 0.023)
    initial_xy_offsets_m = None  # Optional fixed per-environment common shift for paired evaluation.
    waypoint_step = 0.35
    waypoint_tolerance = 0.009
    progress_scale = 10.0
    progress_speed_cap_rad_s = 0.0  # Zero preserves legacy reward; positive caps reward, not motion.
    waypoint_progress_scale = 0.4
    waypoint_bonus = 1.0
    action_change_scale = 0.005
    drop_penalty = 10.0
    stall_penalty_scale = 0.0  # Opt-in; legacy policies keep their original reward.
    stall_window_s = 0.5
    stall_speed_rad_s = 0.5
    stall_grace_s = 2.0
    shared_lift_penalty_scale = 0.0  # Opt-in; preserve all saved legacy rewards.
    shared_lift_tolerance_m = 0.005
    shared_lift_transition_m = 0.010
    sim = SimulationCfg(
        dt=1 / 120,
        render_interval=6,
        physx=PhysxCfg(gpu_max_rigid_contact_count=2**23, gpu_max_rigid_patch_count=2**18),
    )
    scene = InteractiveSceneCfg(num_envs=1024, env_spacing=0.45, replicate_physics=True, clone_in_fabric=False)
    robot_cfg = GHAND_CFG.replace(prim_path="/World/envs/env_.*/Hand")


def env_config():
    cfg = DualBallRotationEnvCfg()
    import os

    baseline_path = os.environ.get("GHAND_DUAL_BALL_BASE_CONFIG")
    if baseline_path:
        from ghand.common.config_io import load_config_yaml, restore_config

        restore_config(cfg, load_config_yaml(baseline_path))
    # Saved training paths belong to the original workstation; use this checkout's asset.
    cfg.robot_cfg.spawn.usd_path = GHAND_CFG.spawn.usd_path
    cfg.viewer.eye = (0.22, -0.35, 0.84)
    cfg.viewer.lookat = (0.0, -0.085, 0.52)
    return cfg
