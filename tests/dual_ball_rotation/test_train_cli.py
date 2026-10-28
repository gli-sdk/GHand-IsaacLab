"""Check trainer argument forwarding without starting Isaac Sim."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from ghand.tasks.dual_ball_rotation import registry

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dual_ball_rotation/train.py"


class TrainingCliTests(unittest.TestCase):
    def invoke(self, arguments, root):
        captured = {}

        def trainer(*args, **kwargs):
            captured["argv"] = list(sys.argv)
            captured["cwd"] = Path.cwd()

        previous = Path.cwd()
        try:
            os.chdir(root)
            with (
                patch.object(registry, "ROOT", root),
                patch.object(sys, "argv", ["train.py", *arguments]),
                patch.dict(os.environ, {"ISAACLAB_PATH": "/unused/runtime"}),
                patch("runpy.run_path", side_effect=trainer),
            ):
                exec(
                    compile(SCRIPT.read_text(), str(SCRIPT), "exec"), {"__name__": "__main__", "__file__": str(SCRIPT)}
                )
        finally:
            os.chdir(previous)
        return captured

    def test_relative_resume_path_survives_output_directory_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resume.pth").touch()
            result = self.invoke(["--checkpoint", "resume.pth"], root)
            self.assertEqual(result["argv"][2], str(root / "resume.pth"))
            self.assertEqual(result["cwd"], root / "runs")

    def test_equals_checkpoint_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resume.pth").touch()
            result = self.invoke(["--checkpoint=resume.pth"], root)
            self.assertEqual(result["argv"][1], "--checkpoint=" + str(root / "resume.pth"))

    def test_equals_task_argument_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(["--task=" + registry.TASK], Path(directory))
            self.assertNotIn("--task", result["argv"])
