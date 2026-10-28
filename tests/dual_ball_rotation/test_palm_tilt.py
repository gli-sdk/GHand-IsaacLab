import unittest
import torch
import math
from ghand.tasks.dual_ball_rotation.palm_frame import rotate_x, tilted_quaternion, sample_tilts, rotate_palm


class PalmTiltTests(unittest.TestCase):
    def test_forward_and_inverse_with_batched_ball_positions(self):
        v = torch.tensor([[[0.0, -0.1, 0.0], [0.02, -0.08, 0.03]], [[0.0, -0.1, 0.0], [-0.02, -0.08, 0.03]]])
        a = torch.tensor([5.0, 15.0])
        w = rotate_x(v, a)
        self.assertTrue((w[:, 0, 2] < 0).all())
        self.assertTrue(torch.allclose(rotate_x(w, -a), v, atol=1e-7))
        self.assertTrue(torch.allclose(w.norm(dim=-1), v.norm(dim=-1), atol=1e-7))

    def test_zero_angle_legacy_identity_and_root_quaternion(self):
        v = torch.randn(3, 2, 3)
        a = torch.zeros(3)
        self.assertTrue(torch.equal(rotate_x(v, a), v))
        q = torch.tensor([[0.0, 0.0, -(2**-0.5), 2**-0.5]])
        self.assertTrue(torch.equal(tilted_quaternion(q, torch.zeros(1)), q))
        tilted = tilted_quaternion(q, torch.tensor([10.0]))
        self.assertAlmostEqual(float(tilted.norm()), 1.0, places=6)
        self.assertAlmostEqual(float(tilted[0, 2]), -math.sin(math.radians(50)), places=6)

    def test_seeded_bounded_episode_sampling(self):
        a = sample_tilts(1024, (5.0, 15.0), torch.Generator().manual_seed(42))
        b = sample_tilts(1024, (5.0, 15.0), torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(((a >= 5) & (a <= 15)).all())
        self.assertGreater(float(a.std()), 2.0)
        self.assertTrue(torch.equal(sample_tilts(3, (10.0, 10.0), torch.Generator()), torch.full((3,), 10.0)))
        with self.assertRaises(ValueError):
            sample_tilts(1, (15.0, 5.0), torch.Generator())

    def test_steep_tilt_keeps_fingertips_down_and_roundtrips(self):
        a = sample_tilts(2, (50.0, 50.0), torch.Generator())
        v = torch.tensor([[[0.0, -0.1, 0.0]], [[0.02, -0.08, 0.03]]])
        w = rotate_x(v, a)
        self.assertAlmostEqual(float(w[0, 0, 2]), -0.1 * math.sin(math.radians(50)), places=6)
        torch.testing.assert_close(rotate_x(w, -a), v, atol=1e-7, rtol=1e-6)
        q = tilted_quaternion(torch.tensor([[0.0, 0.0, -(2**-0.5), 2**-0.5]]), a)
        torch.testing.assert_close(q.norm(dim=-1), torch.ones(2))
        for bounds in ((-91.0, 0.0), (0.0, 91.0), (float("nan"), 50.0)):
            with self.assertRaises(ValueError):
                sample_tilts(1, bounds, torch.Generator())

    def test_side_tilt_lowers_thumb_side_and_roundtrips(self):
        forward = torch.tensor([50.0, 50.0])
        side = torch.tensor([-5.0, -5.0])
        # Thumb is on negative X in the untilted task frame (checked against asset FK).
        points = torch.tensor([[[-0.05, -0.1, 0.0], [0.05, -0.1, 0.0]], [[0.02, -0.08, 0.03], [-0.03, -0.09, 0.04]]])
        moved = rotate_palm(points, forward, side)
        self.assertLess(float(moved[0, 0, 2]), float(moved[0, 1, 2]))
        torch.testing.assert_close(rotate_palm(moved, forward, side, inverse=True), points, atol=1e-7, rtol=1e-6)

    def test_combined_root_and_ball_rotations_agree(self):
        def apply(q, v):
            xyz = q[:, 1:]
            return v + 2 * torch.cross(xyz, torch.cross(xyz, v, dim=-1) + q[:, :1] * v, dim=-1)

        base = torch.tensor([[0.0, 0.0, -(2**-0.5), 2**-0.5]])
        v = torch.tensor([[0.02, 0.03, -0.1]])
        forward = torch.tensor([50.0])
        side = torch.tensor([-5.0])
        q = tilted_quaternion(base, forward, side)
        torch.testing.assert_close(apply(q, v), rotate_palm(apply(base, v), forward, side), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(q.norm(dim=-1), torch.ones(1))
        torch.testing.assert_close(rotate_palm(v, forward, None), rotate_x(v, forward), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
