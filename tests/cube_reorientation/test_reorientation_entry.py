"""Public reorientation entry points work without starting Isaac Sim."""

import json
from pathlib import Path
import subprocess
import unittest

import gymnasium as gym
import ghand.tasks.cube_reorientation

PROJECT = Path(__file__).resolve().parents[2]


class ReorientationEntryTests(unittest.TestCase):
    def test_single_reorientation_task_matches_model_bundle(self):
        task_ids = [
            name
            for name, spec in gym.registry.items()
            if spec.entry_point and str(spec.entry_point).startswith("ghand.tasks.cube_reorientation.env:")
        ]
        self.assertEqual(task_ids, ["Isaac-Reorient-Cube-GHand-v0"])
        selection = json.loads((PROJECT / "models/cube_reorientation/teacher/selection.json").read_text())
        self.assertEqual(selection["task"], ghand.tasks.cube_reorientation.TASK)
        spec = gym.spec(ghand.tasks.cube_reorientation.TASK)
        self.assertEqual(spec.kwargs["env_cfg_entry_point"], "ghand.tasks.cube_reorientation.env_cfg:env_config")

    def test_launcher_accepts_environment_count_without_model_variant(self):
        result = subprocess.run(
            ["bash", str(PROJECT / "scripts/cube_reorientation/play.sh"), "4", "--dry-run"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--num_envs 4", result.stdout)
        self.assertIn("models/cube_reorientation/teacher/nn/policy.pth", result.stdout)
        self.assertIn("--task Isaac-Reorient-Cube-GHand-v0", result.stdout)

    def test_launcher_accepts_options_without_environment_count(self):
        result = subprocess.run(
            ["bash", str(PROJECT / "scripts/cube_reorientation/play.sh"), "--dry-run"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--num_envs 16", result.stdout)


if __name__ == "__main__":
    unittest.main()
