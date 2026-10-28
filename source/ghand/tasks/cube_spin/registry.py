"""Register continuous cube rotation independently of reorientation."""

from pathlib import Path
import gymnasium as gym

TASK = "Isaac-Spin-Cube-GHand-v0"
PROJECT = Path(__file__).resolve().parents[4]
if TASK not in gym.registry:
    gym.register(
        id=TASK,
        entry_point="ghand.tasks.cube_spin.env:GHandSpinEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "ghand.tasks.cube_spin.env_cfg:env_config",
            "rl_games_cfg_entry_point": str(PROJECT / "config/cube_spin/ppo.yaml"),
        },
    )
