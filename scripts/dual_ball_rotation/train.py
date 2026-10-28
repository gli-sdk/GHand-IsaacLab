"""Train or resume Dual-ball rotation PPO through the external Isaac Lab runtime."""

import os
import runpy
import sys
from pathlib import Path

from ghand.tasks.dual_ball_rotation.registry import TASK, ROOT


def main():
    if not any(arg == "--task" or arg.startswith("--task=") for arg in sys.argv):
        sys.argv.extend(["--task", TASK])
    # The trainer writes below runs/, so resolve user checkpoint paths first.
    for index, argument in enumerate(sys.argv):
        if argument == "--checkpoint" and index + 1 < len(sys.argv):
            sys.argv[index + 1] = str(Path(sys.argv[index + 1]).expanduser().resolve(strict=True))
        elif argument.startswith("--checkpoint="):
            checkpoint = Path(argument.split("=", 1)[1]).expanduser().resolve(strict=True)
            sys.argv[index] = f"--checkpoint={checkpoint}"
    (ROOT / "runs").mkdir(exist_ok=True)
    os.chdir(ROOT / "runs")
    trainer = Path(os.environ["ISAACLAB_PATH"]) / "scripts/reinforcement_learning/rl_games/train.py"
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
