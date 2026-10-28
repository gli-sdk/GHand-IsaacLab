"""Register GHand cube reorientation without importing simulator modules."""

from pathlib import Path

import gymnasium as gym

PROJECT = Path(__file__).resolve().parents[4]
TASK = "Isaac-Reorient-Cube-GHand-v0"

if TASK not in gym.registry:
    gym.register(
        id=TASK,
        entry_point="ghand.tasks.cube_reorientation.env:GHandReorientationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "ghand.tasks.cube_reorientation.env_cfg:env_config",
            "rl_games_cfg_entry_point": str(PROJECT / "config/cube_reorientation/ppo.yaml"),
        },
    )
