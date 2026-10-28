"""Independent, SDK-bounded teacher task for continuous cube rotation."""

from pathlib import Path

from isaaclab.utils import configclass

from ghand.common.cube_env_cfg import GHandCubeEnvCfg
from ghand.common.config_io import load_config_yaml, restore_config
from ghand.common.hand_asset import USD_PATH

PROJECT = Path(__file__).resolve().parents[4]


@configclass
class GHandSpinEnvCfg(GHandCubeEnvCfg):
    observation_space = 68
    state_space = 0
    spin_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    target_speed: float = 1.0
    action_step_rad: float = 0.075
    command_upper_deg: dict[str, float] = {}  # Target ceilings; physical/observation ranges stay unchanged.
    command_margin_fraction: float = 0.0  # Fraction of each physical interval inset at both target endpoints.
    # None preserves the fixed-asset baseline. Dimensions are sampled before physics starts.
    cube_size_range_m: tuple[float, float] | None = None
    cube_mass_range_kg: tuple[float, float] | None = None
    # Cube material uses multiply; hand material stays at 1, so these are pair coefficients.
    cube_static_friction_range: tuple[float, float] | None = None
    cube_dynamic_friction_ratio_range: tuple[float, float] = (0.7, 1.0)
    friction_bucket_count: int = 128
    spin_warmup_s: float = 0.5
    spin_drop_penalty: float = -10.0
    spin_survival_reward: float = 0.1
    spin_position_penalty_scale: float = 0.5  # Set 0 to allow free in-hand translation.
    spin_terminate_on_distance: bool = True  # Legacy initial-position radius; height drop stays active.
    hardware_config: str = str(PROJECT / "config/cube_spin/hand.json")


def env_config():
    """Reuse validated physics; apply spin-specific settings after restoration."""
    cfg = GHandSpinEnvCfg()
    restore_config(cfg, load_config_yaml(PROJECT / "config/cube_spin/env.yaml"))
    cfg.robot_cfg.spawn.usd_path = str(USD_PATH)
    cfg.hardware_config = str(PROJECT / "config/cube_spin/hand.json")
    cfg.action_space = 13
    cfg.observation_space = 68
    cfg.state_space = 0
    cfg.asymmetric_obs = False
    cfg.decimation = 6  # 120 Hz physics, 20 Hz policy. Hardware rate still needs measurement.
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = 30.0
    cfg.scene.num_envs = 1024
    cfg.reset_position_noise = 0.002
    cfg.reset_dof_pos_noise = 0.1
    cfg.reset_dof_vel_noise = 0.0
    cfg.correct_joint_reset = True
    cfg.relative_goal_probability = 0.0
    cfg.max_consecutive_success = 0
    cfg.rotation_progress_reward_scale = 0.0
    cfg.fall_dist = 0.12
    cfg.viewer.eye = (0.35, -0.40, 0.90)
    cfg.viewer.lookat = (0.0, -0.06, 0.56)
    return cfg
