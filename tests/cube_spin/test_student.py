import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

MODULE = Path(__file__).parents[2] / "source/ghand/tasks/cube_spin/student.py"


class StudentTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), "No-vision student module is missing")
        spec = importlib.util.spec_from_file_location("spin_student", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.lower = torch.full((13,), -1.0)
        self.upper = torch.full((13,), 1.0)

    def history(self, steps=3):
        return self.module.JointHistory(2, steps, self.lower, self.upper)

    def test_history_orders_old_to_new_and_reset_does_not_leak_previous_episode(self):
        h = self.history()
        q = torch.zeros(2, 13)
        t = torch.full_like(q, 0.2)
        h.reset(q, t)
        h.advance(q + 0.3, t + 0.2, torch.tensor([False, False]))
        h.advance(q + 0.6, t + 0.4, torch.tensor([False, True]))
        obs = h.observation(1.0)
        self.assertEqual(obs.shape, (2, 79))
        torch.testing.assert_close(obs[0, :13], q[0])
        torch.testing.assert_close(obs[0, 26:39], q[0] + 0.3)
        torch.testing.assert_close(obs[0, 52:65], q[0] + 0.6)
        expected = torch.cat((q[1] + 0.6, t[1] + 0.4)).repeat(3)
        torch.testing.assert_close(obs[1, :-1], expected)
        torch.testing.assert_close(obs[:, -1], torch.ones(2))
        snapshot = obs.clone()
        h.advance(q, t, torch.tensor([False, False]))
        torch.testing.assert_close(obs, snapshot)

    def test_feedback_and_command_use_joint_specific_radian_limits(self):
        lower = torch.arange(13).float() * 0.01
        upper = lower + 2.0
        h = self.module.JointHistory(2, 1, lower, upper)
        q = torch.stack((lower, upper))
        h.reset(q, (lower + upper).expand(2, -1) / 2.0)
        obs = h.observation(0.5)
        torch.testing.assert_close(obs[0, :13], -torch.ones(13))
        torch.testing.assert_close(obs[1, :13], torch.ones(13))
        torch.testing.assert_close(obs[:, 13:26], torch.zeros(2, 13))

    def test_rejects_nonfinite_feedback_bad_dimensions_and_invalid_limits(self):
        h = self.history()
        q = torch.zeros(2, 13)
        with self.assertRaises(ValueError):
            h.observation(1.0)
        for bad in (torch.full_like(q, float("nan")), torch.zeros(2, 18)):
            with self.assertRaises(ValueError):
                h.reset(bad, q)
        with self.assertRaises(ValueError):
            self.module.JointHistory(2, 3, self.lower, self.lower)
        with self.assertRaises(ValueError):
            self.history(0)

    def test_cpu_export_reproduces_policy_and_actions_are_bounded(self):
        torch.manual_seed(42)
        model = self.module.StudentPolicy(3).eval()
        x = torch.randn(8, 79)
        expected = model(x)
        self.assertEqual(expected.shape, (8, 13))
        self.assertTrue(torch.isfinite(expected).all())
        self.assertLessEqual(expected.abs().max().item(), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "actor.ts")
            torch.jit.script(model).save(path)
            restored = torch.jit.load(path)
            torch.testing.assert_close(restored(x), expected)


if __name__ == "__main__":
    unittest.main()
