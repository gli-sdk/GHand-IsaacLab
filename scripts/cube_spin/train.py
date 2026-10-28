"""Train the privileged continuous-spin teacher; no hardware is connected."""

from pathlib import Path
import os
import runpy
import sys

from ghand.tasks.cube_spin.registry import TASK

PROJECT = Path(__file__).resolve().parents[2]
if not any(a == "--task" or a.startswith("--task=") for a in sys.argv):
    sys.argv.extend(["--task", TASK])
for index, argument in enumerate(sys.argv):
    if argument == "--checkpoint" and index + 1 < len(sys.argv):
        sys.argv[index + 1] = str(Path(sys.argv[index + 1]).expanduser().resolve(strict=True))
    elif argument.startswith("--checkpoint="):
        sys.argv[index] = "--checkpoint=" + str(Path(argument.split("=", 1)[1]).expanduser().resolve(strict=True))
(PROJECT / "runs").mkdir(exist_ok=True)
os.chdir(PROJECT / "runs")
runpy.run_path(
    str(Path(os.environ["ISAACLAB_PATH"]) / "scripts/reinforcement_learning/rl_games/train.py"), run_name="__main__"
)
