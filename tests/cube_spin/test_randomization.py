import math
import unittest
import torch
from ghand.tasks.cube_spin.randomization import sample_range, cube_inertia
from ghand.tasks.cube_spin import randomization as randomization


class CubeRandomizationTests(unittest.TestCase):
    def test_bounds_seed_and_independent_samples(self):
        torch.manual_seed(123)
        size = sample_range((0.035, 0.040), 4096)
        mass = sample_range((0.030, 0.050), 4096)
        self.assertTrue(bool(((size >= 0.035) & (size <= 0.040)).all()))
        self.assertTrue(bool(((mass >= 0.030) & (mass <= 0.050)).all()))
        self.assertLess(float(size.min()), 0.0351)
        self.assertGreater(float(size.max()), 0.0399)
        self.assertLess(abs(float(torch.corrcoef(torch.stack((size, mass)))[0, 1])), 0.06)
        torch.manual_seed(123)
        torch.testing.assert_close(size, sample_range((0.035, 0.040), 4096))

    def test_size_does_not_consume_mass_rng_or_repeat_after_reset(self):
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        size = sample_range((0.035, 0.040), 4096, seed=100126)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.manual_seed(123)
        mass = sample_range((0.030, 0.050), 4096)
        correlation = float(torch.corrcoef(torch.stack((size, mass)))[0, 1])
        self.assertLess(abs(correlation), 0.06)
        torch.testing.assert_close(size, sample_range((0.035, 0.040), 4096, seed=100126))

    def test_fixed_endpoints_and_invalid_ranges(self):
        torch.testing.assert_close(sample_range((0.04, 0.04), 3), torch.full((3,), 0.04))
        for bounds in ((0.0, 0.04), (0.04, 0.035), (math.nan, 0.04), (0.03, math.inf), (0.03,), (-0.03, 0.05)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                sample_range(bounds, 3)

    def test_uniform_cube_inertia_tracks_both_size_and_mass(self):
        sizes = torch.tensor([0.035, 0.04, 0.04])
        masses = torch.tensor([0.03, 0.03, 0.05])
        inertia = cube_inertia(sizes, masses)
        expected = torch.zeros(3, 9)
        expected[:, [0, 4, 8]] = (masses * sizes.square() / 6)[:, None]
        torch.testing.assert_close(inertia, expected)
        self.assertGreater(float(inertia[1, 0]), float(inertia[0, 0]))
        self.assertAlmostEqual(float(inertia[2, 0] / inertia[1, 0]), 5 / 3, places=5)
        self.assertTrue(bool((torch.linalg.eigvalsh(inertia.reshape(-1, 3, 3)) > 0).all()))


class FrictionRandomizationTests(unittest.TestCase):
    def test_consistent_bounded_reproducible_material_buckets(self):
        self.assertTrue(hasattr(randomization, "friction_buckets"), "Friction sampling is not implemented")
        before = torch.random.get_rng_state().clone()
        values = randomization.friction_buckets((0.5, 1.5), (0.7, 1.0), 128, seed=123)
        self.assertEqual(values.shape, (128, 3))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(bool(((values[:, 0] >= 0.5) & (values[:, 0] <= 1.5)).all()))
        ratio = values[:, 1] / values[:, 0]
        self.assertTrue(bool(((ratio >= 0.7 - 1e-6) & (ratio <= 1.0)).all()))
        self.assertTrue(bool((values[:, 2] == 0.0).all()))
        self.assertLess(abs(float(torch.corrcoef(torch.stack((values[:, 0], ratio)))[0, 1])), 0.25)
        torch.testing.assert_close(values, randomization.friction_buckets((0.5, 1.5), (0.7, 1.0), 128, seed=123))

    def test_fixed_material_and_invalid_friction_settings(self):
        self.assertTrue(hasattr(randomization, "friction_buckets"), "Friction sampling is not implemented")
        torch.testing.assert_close(
            randomization.friction_buckets((1.5, 1.5), (0.8, 0.8), 2, seed=1),
            torch.tensor([[1.5, 1.2, 0.0], [1.5, 1.2, 0.0]]),
        )
        for static, ratio, count in [
            ((1.0, 0.5), (0.7, 1.0), 64),
            ((0.5, 1.0), (0.7, 1.1), 64),
            ((0.5, math.nan), (0.7, 1.0), 64),
            ((0.5, 1.0), (0.0, 1.0), 64),
            ((0.5, 1.0), (0.7, 1.0), 0),
            ((0.5, 1.0), (0.7, 1.0), 65536),
        ]:
            with self.subTest(static=static, ratio=ratio, count=count), self.assertRaises(ValueError):
                randomization.friction_buckets(static, ratio, count, seed=1)


if __name__ == "__main__":
    unittest.main()
