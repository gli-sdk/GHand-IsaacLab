import math
import unittest
import torch
from ghand.tasks.dual_ball_rotation.orbit import orbit_delta, novel_progress, stall_fraction


class OrbitTests(unittest.TestCase):
    def test_full_circle_and_branch_cut(self):
        a = torch.linspace(0, 2 * math.pi, 101)
        v = 0.032 * torch.stack((a.cos(), a.sin(), torch.zeros_like(a)), -1)
        d, valid = orbit_delta(v[:-1], v[1:], 0.030)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(d.sum().item(), 2 * math.pi, places=5)

    def test_translation_does_not_count(self):
        p = torch.tensor([[0.016, -0.077, 0.54], [-0.016, -0.077, 0.54]])
        q = p + torch.tensor([0.02, 0.01, 0.003])
        d, _ = orbit_delta((p[0] - p[1])[None], (q[0] - q[1])[None], 0.030)
        self.assertAlmostEqual(d.item(), 0.0, places=6)

    def test_stacking_not_orbit(self):
        a = torch.tensor([[0.001, 0, 0.035]])
        b = torch.tensor([[0.0, 0.001, 0.035]])
        d, valid = orbit_delta(a, b, 0.030)
        self.assertFalse(valid.item())
        self.assertEqual(d.item(), 0.0)

    def test_reversal_and_repeated_arc_no_new_credit(self):
        net = torch.zeros(1)
        high = torch.zeros(1)
        paid = 0.0
        for step in [1.0, -1.0, 1.0, -1.0, 1.0]:
            net += step
            credit, high = novel_progress(net, high)
            paid += credit.item()
        self.assertAlmostEqual(paid, 1.0)

    def test_capped_credit_prefers_even_progress_without_banking_bursts(self):
        def total(steps):
            net = torch.zeros(1)
            high = torch.zeros(1)
            paid = 0.0
            for step in steps:
                net += step
                credit, high = novel_progress(net, high, max_credit=0.15)
                paid += credit.item()
            return paid

        self.assertAlmostEqual(total([0.15] * 4), 0.6, places=6)
        self.assertAlmostEqual(total([0.6, 0.0, 0.0, 0.0]), 0.15, places=6)
        self.assertAlmostEqual(total([0.15, -0.15, 0.15, 0.0]), 0.15, places=6)

    def test_stall_fraction_is_soft_and_requires_forward_relative_motion(self):
        speeds = torch.tensor([0.0, 0.25, 0.5, 2.0, -0.5])
        torch.testing.assert_close(stall_fraction(speeds, 0.5), torch.tensor([1.0, 0.5, 0.0, 0.0, 1.0]))

    def test_shared_lift_penalty_requires_both_balls_to_rise(self):
        from ghand.tasks.dual_ball_rotation.orbit import shared_lift_fraction

        initial = torch.tensor([[0.45, 0.451]]).expand(5, -1)
        heights = initial + torch.tensor(
            [[0.010, 0.010], [0.020, 0.0], [0.020, 0.003], [-0.005, -0.002], [0.030, 0.030]]
        )
        cost = shared_lift_fraction(heights, initial, 0.005, 0.010)
        torch.testing.assert_close(cost, torch.tensor([0.5, 0.0, 0.0, 0.0, 1.0]), atol=1e-5, rtol=1e-5)

    def test_shared_lift_uses_relative_world_heights_not_scene_origin(self):
        from ghand.tasks.dual_ball_rotation.orbit import shared_lift_fraction

        initial = torch.tensor([[0.45, 0.451], [1.45, 1.451]])
        heights = initial + torch.tensor([[0.010, 0.010], [0.010, 0.010]])
        torch.testing.assert_close(
            shared_lift_fraction(heights, initial, 0.005, 0.010), torch.tensor([0.5, 0.5]), atol=1e-5, rtol=1e-5
        )

    def test_reverse_negative(self):
        a = torch.tensor([[0.032, 0.0, 0.0]])
        b = torch.tensor([[0.032 * math.cos(0.2), -0.032 * math.sin(0.2), 0.0]])
        d, _ = orbit_delta(a, b, 0.030)
        self.assertAlmostEqual(d.item(), -0.2, places=6)


if __name__ == "__main__":
    unittest.main()
