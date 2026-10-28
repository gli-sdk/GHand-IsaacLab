import math
import unittest
import torch
from ghand.tasks.dual_ball_rotation.orbit import orbit_delta, orbit_goal_offsets
from ghand.tasks.dual_ball_rotation.perturbations import sample_domains


class ReverseDomainTests(unittest.TestCase):
    def test_clockwise_has_positive_progress_only_for_reverse_task(self):
        a = torch.tensor([[0.032, 0.0, 0.0]])
        b = torch.tensor([[0.032 * math.cos(0.2), -0.032 * math.sin(0.2), 0.0]])
        d, v = orbit_delta(a, b, 0.030, direction=-1)
        self.assertTrue(v.item())
        self.assertAlmostEqual(d.item(), 0.2, places=6)
        forward, _ = orbit_delta(a, b, 0.030, direction=1)
        self.assertAlmostEqual(forward.item(), -0.2, places=6)

    def test_goals_reverse_without_changing_phase_progress(self):
        phase = torch.tensor([0.35, 0.7])
        radii = torch.tensor([0.016, 0.023])
        forward = orbit_goal_offsets(phase, radii, 1)
        reverse = orbit_goal_offsets(phase, radii, -1)
        torch.testing.assert_close(forward[:, 0], reverse[:, 0])
        torch.testing.assert_close(forward[:, 1], -reverse[:, 1])
        for bad in [0, 2, float("nan")]:
            with self.assertRaises(ValueError):
                orbit_goal_offsets(phase, radii, bad)

    def test_domains_preserve_nominal_and_reproducible_bounded_variants(self):
        kwargs = dict(num_envs=2048, bound_mm=2.0, friction_scale_range=(0.75, 1.25), nominal_fraction=0.25)

        def draw():
            return sample_domains(**kwargs, generator=torch.Generator().manual_seed(19))

        shift, scale = draw()
        other, other_scale = draw()
        torch.testing.assert_close(shift, other)
        torch.testing.assert_close(scale, other_scale)
        self.assertLessEqual(float(shift.abs().max()), 0.002)
        self.assertTrue(((scale >= 0.75) & (scale <= 1.25)).all())
        nominal = (shift == 0).all(-1)
        self.assertGreater(float(nominal.float().mean()), 0.2)
        self.assertLess(float(nominal.float().mean()), 0.3)
        self.assertTrue((scale[nominal] == 1.0).all())
        self.assertLessEqual(len(scale.unique()), 65)
        self.assertTrue((0.6 * scale <= 0.8 * scale).all())

    def test_fully_nominal_is_exact_baseline(self):
        shift, scale = sample_domains(64, 2.0, (0.75, 1.25), 1.0, torch.Generator().manual_seed(19))
        self.assertTrue((shift == 0).all())
        self.assertTrue((scale == 1).all())


if __name__ == "__main__":
    unittest.main()
