"""Train or resume cube reorientation with Isaac Lab's RL-Games trainer."""

from pathlib import Path
import os
import runpy
import sys

import ghand.tasks.cube_reorientation

PROJECT = Path(__file__).resolve().parents[2]


def main():
    runtime = Path(os.environ["ISAACLAB_PATH"])
    if "--task" not in sys.argv and not any(arg.startswith("--task=") for arg in sys.argv):
        sys.argv.extend(["--task", ghand.tasks.cube_reorientation.TASK])
    if "--num_envs" not in sys.argv and not any(arg.startswith("--num_envs=") for arg in sys.argv):
        sys.argv.extend(["--num_envs", "2048"])

    # Resolve user paths before moving into the training output directory.
    for index, argument in enumerate(sys.argv):
        if argument == "--checkpoint" and index + 1 < len(sys.argv):
            sys.argv[index + 1] = str(Path(sys.argv[index + 1]).expanduser().resolve(strict=True))
        elif argument.startswith("--checkpoint="):
            checkpoint = Path(argument.split("=", 1)[1]).expanduser().resolve(strict=True)
            sys.argv[index] = f"--checkpoint={checkpoint}"

    (PROJECT / "runs").mkdir(exist_ok=True)
    os.chdir(PROJECT / "runs")
    runpy.run_path(str(runtime / "scripts/reinforcement_learning/rl_games/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
