from pathlib import Path
import gymnasium as gym

TASK = "Isaac-Dual-Ball-Rotation-GHand-v0"
ROOT = Path(__file__).resolve().parents[4]
if TASK not in gym.registry:
    gym.register(
        id=TASK,
        entry_point="ghand.tasks.dual_ball_rotation.env:DualBallRotationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "ghand.tasks.dual_ball_rotation.env_cfg:env_config",
            "rl_games_cfg_entry_point": str(ROOT / "config/dual_ball_rotation/ppo.yaml"),
        },
    )
