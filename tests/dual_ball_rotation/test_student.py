import unittest
import torch
from ghand.tasks.dual_ball_rotation.student import JointHistory, StudentPolicy, two_relative_turns


class StudentTests(unittest.TestCase):
    def test_history_contains_only_measured_angles_and_applied_commands(self):
        h = JointHistory(2, 3, torch.zeros(13), torch.ones(13))
        q = torch.full((2, 13), 0.25)
        u = torch.full((2, 13), 0.75)
        h.reset(q, u)
        expected = torch.cat((torch.full((2, 13), -0.5), torch.full((2, 13), 0.5)), 1)
        self.assertTrue(torch.equal(h.observation(), expected.repeat(1, 3)))

    def test_reset_cannot_leak_previous_episode(self):
        h = JointHistory(2, 3, torch.zeros(13), torch.ones(13))
        h.reset(torch.zeros(2, 13), torch.zeros(2, 13))
        h.advance(torch.ones(2, 13), torch.ones(2, 13), torch.tensor([True, False]))
        self.assertTrue(torch.equal(h.observation()[0], torch.ones(78)))
        self.assertTrue(torch.equal(h.observation()[1], torch.cat((-torch.ones(52), torch.ones(26)))))

    def test_actor_export_bounded_and_identical(self):
        net = StudentPolicy(30).eval()
        x = torch.randn(4, 780)
        exported = torch.jit.script(net)
        self.assertTrue(torch.allclose(net(x), exported(x)))
        self.assertEqual(tuple(net(x).shape), (4, 13))
        self.assertLessEqual(float(net(x).abs().max().detach()), 1.0)

    def test_relative_turn_screen_requires_duration_and_no_drop(self):
        row = dict(seconds=29.95, net_turns=3.0, dropped=False, ball0_turns=0.0, ball1_turns=0.0)
        self.assertTrue(two_relative_turns(row))
        self.assertFalse(two_relative_turns(dict(row, seconds=10.0)))
        self.assertFalse(two_relative_turns(dict(row, dropped=True)))
        self.assertFalse(two_relative_turns(dict(row, net_turns=1.9)))


if __name__ == "__main__":
    unittest.main()
