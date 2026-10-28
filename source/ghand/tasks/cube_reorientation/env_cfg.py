"""Cube reorientation training configuration."""

from pathlib import Path
from ghand.common.cube_env_cfg import GHandCubeEnvCfg
from ghand.common.config_io import load_config_yaml, restore_config
from ghand.common.hand_asset import USD_PATH

PROJECT = Path(__file__).resolve().parents[4]


def env_config():
    """Load training settings and bind the hand to this project's validated asset.

    Evaluation restores the selected checkpoint's params/env.yaml afterwards.
    """
    cfg = GHandCubeEnvCfg()
    restore_config(cfg, load_config_yaml(PROJECT / "config/cube_reorientation/env.yaml"))
    cfg.robot_cfg.spawn.usd_path = str(USD_PATH)
    return cfg
