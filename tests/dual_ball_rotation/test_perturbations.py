import unittest
import torch
from ghand.tasks.dual_ball_rotation.perturbations import placement_offsets


class PerturbationTests(unittest.TestCase):
    def test_zero_is_exact_original_start(self):
        self.assertTrue(torch.equal(placement_offsets(64, 0.0, 17), torch.zeros(64, 2)))

    def test_seed_pairs_conditions_without_changing_global_rng(self):
        torch.manual_seed(10)
        before = torch.random.get_rng_state()
        a = placement_offsets(64, 2.0, 123)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.manual_seed(98)
        self.assertTrue(torch.equal(a, placement_offsets(64, 2.0, 123)))
        self.assertLessEqual(float(a.abs().max()), 0.002)
        self.assertGreater(float(a.std()), 0.0005)
        self.assertTrue(torch.allclose(placement_offsets(64, 4.0, 123), a * 2))

    def test_common_translation_preserves_pair_separation(self):
        a = placement_offsets(64, 4.0, 123)
        balls = torch.tensor([[0.016, -0.077], [-0.016, -0.077]])
        moved = balls[None, :, :] + a[:, None, :]
        self.assertTrue(torch.allclose(moved[:, 0] - moved[:, 1], (balls[0] - balls[1]).expand(64, 2)))

    def test_reject_invalid_bounds(self):
        for count, bound in [(0, 2), (64, -1), (64, float("nan")), (64, float("inf"))]:
            with self.assertRaises(ValueError):
                placement_offsets(count, bound, 1)


class BallSizeTests(unittest.TestCase):
    def resize(self, diameter):
        import ghand.tasks.dual_ball_rotation.perturbations as module

        self.assertTrue(hasattr(module, "resized_ball_geometry"), "Optional ball-size geometry is not implemented")
        return module.resized_ball_geometry(0.015, (0.016, 0.023), (0.53575, 0.5375), diameter)

    def test_default_preserves_original_geometry_exactly(self):
        self.assertEqual(self.resize(None), (0.015, (0.016, 0.023), (0.53575, 0.5375)))
        self.assertEqual(self.resize(30.0), self.resize(None))

    def test_larger_balls_keep_clearance_through_full_goal_orbit(self):
        import math

        for diameter in (35.0, 40.0):
            radius, radii, heights = self.resize(diameter)
            self.assertAlmostEqual(radius * 2000.0, diameter)
            for angle in torch.linspace(0.0, 2 * math.pi, 101):
                separation = 2 * math.hypot(radii[0] * math.cos(angle), radii[1] * math.sin(angle))
                self.assertGreaterEqual(separation - 2 * radius, 0.002 - 1e-8)
            for old, new in zip((0.53575, 0.5375), heights):
                self.assertAlmostEqual(old - 0.015, new - radius)

    def test_invalid_size_fails_before_physics(self):
        for diameter in (0.0, -5.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.resize(diameter)


if __name__ == "__main__":
    unittest.main()
