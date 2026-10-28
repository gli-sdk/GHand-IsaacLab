import copy
import importlib.util
import math
from pathlib import Path
import unittest

import torch


MODULE = Path(__file__).parents[2] / "source/ghand/tasks/cube_spin/core.py"
ACTIVE = [
    "LF_MCP",
    "LF_PIP",
    "RF_MCP",
    "RF_PIP",
    "MF_MCP",
    "MF_PIP",
    "FF_MCP_AA",
    "FF_MCP",
    "FF_PIP",
    "Thumb_TMC_PS",
    "Thumb_TMC_AA",
    "Thumb_TMC_FE",
    "Thumb_MCP",
]


def fixture():
    manifest = {
        "active_joints": ACTIVE.copy(),
        "joints": {name: {"lower": -2.0, "upper": 2.0, "mimic": None} for name in ACTIVE},
    }
    sdk = {"joints": [{"id": name.upper(), "min": -90.0, "max": 90.0} for name in ACTIVE]}
    return manifest, sdk


def quat(angles, axis=(0.0, 0.0, 1.0)):
    angles = torch.as_tensor(angles, dtype=torch.float64)
    axis = torch.tensor(axis, dtype=torch.float64)
    return torch.cat((torch.cos(angles / 2).unsqueeze(-1), torch.sin(angles / 2).unsqueeze(-1) * axis), dim=-1)


class SpinCoreTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), "spin_core implementation is missing")
        spec = importlib.util.spec_from_file_location("spin_core", MODULE)
        self.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.core)

    def test_contract_order_ids_and_limit_intersection(self):
        manifest, sdk = fixture()
        manifest["joints"]["LF_MCP"]["upper"] = 0.8
        manifest["joints"]["LF_DIP"] = {
            "lower": 0.1,
            "upper": 0.5,
            "mimic": {"joint": "LF_PIP", "multiplier": 2.0, "offset": -0.1},
        }
        rows = self.core.sdk_joint_contract(manifest, sdk)
        self.assertEqual([row["name"] for row in rows], ACTIVE)
        self.assertEqual([row["sdk_id"] for row in rows], [17, 16, 14, 13, 11, 10, 8, 7, 6, 4, 3, 2, 1])
        self.assertAlmostEqual(rows[0]["lower"], -math.pi / 2)
        self.assertEqual(rows[0]["upper"], 0.8)
        self.assertAlmostEqual(rows[1]["lower"], 0.1)
        self.assertAlmostEqual(rows[1]["upper"], 0.3)

    def test_negative_and_zero_mimic_multiplier(self):
        manifest, sdk = fixture()
        manifest["joints"]["LF_DIP"] = {
            "lower": 0.2,
            "upper": 0.6,
            "mimic": {"joint": "LF_PIP", "multiplier": -2.0, "offset": 0.8},
        }
        row = self.core.sdk_joint_contract(manifest, sdk)[1]
        self.assertAlmostEqual(row["lower"], 0.1)
        self.assertAlmostEqual(row["upper"], 0.3)
        manifest["joints"]["LF_DIP"]["mimic"]["multiplier"] = 0.0
        with self.assertRaises(ValueError):
            self.core.sdk_joint_contract(manifest, sdk)

    def test_rejects_incomplete_duplicate_unknown_and_empty_contracts(self):
        manifest, sdk = fixture()
        for names in [ACTIVE[:-1], ACTIVE[:-1] + [ACTIVE[0]], ACTIVE[:-1] + ["unknown"]]:
            bad = copy.deepcopy(manifest)
            bad["active_joints"] = names
            with self.assertRaises(ValueError):
                self.core.sdk_joint_contract(bad, sdk)
        manifest["joints"]["LF_MCP"]["lower"] = 1.8
        with self.assertRaises(ValueError):
            self.core.sdk_joint_contract(manifest, sdk)
        manifest, sdk = fixture()
        sdk["joints"].append(sdk["joints"][0].copy())
        with self.assertRaises(ValueError):
            self.core.sdk_joint_contract(manifest, sdk)

    def test_rejects_nonfinite_sdk_limits(self):
        manifest, sdk = fixture()
        sdk["joints"][0]["min"] = float("nan")
        with self.assertRaises(ValueError):
            self.core.sdk_joint_contract(manifest, sdk)

    def test_command_caps_preserve_contract_and_uncapped_joints(self):
        manifest, sdk = fixture()
        contract = self.core.sdk_joint_contract(manifest, sdk)
        original = copy.deepcopy(contract)
        limits = self.core.command_upper_limits(contract, {"RF_PIP": 80.0, "LF_PIP": 64.0})
        self.assertEqual(contract, original)
        for row, limit in zip(contract, limits):
            expected = (
                math.radians({"RF_PIP": 80.0, "LF_PIP": 64.0}[row["name"]])
                if row["name"] in ("RF_PIP", "LF_PIP")
                else row["upper"]
            )
            self.assertAlmostEqual(limit, expected)
        self.assertEqual(self.core.command_upper_limits(contract, {}), [r["upper"] for r in contract])

    def test_command_caps_reject_unknown_nonfinite_and_outside_physical_range(self):
        contract = self.core.sdk_joint_contract(*fixture())
        for caps in ({"RF_DIP": 80.0}, {"RF_PIP": float("nan")}, {"RF_PIP": 91.0}, {"RF_PIP": -90.0}):
            with self.subTest(caps=caps), self.assertRaises(ValueError):
                self.core.command_upper_limits(contract, caps)

    def test_margin_insets_both_endpoints_and_preserves_physical_contract(self):
        contract = self.core.sdk_joint_contract(*fixture())
        contract[6]["lower"], contract[6]["upper"] = math.radians(-8.0), math.radians(8.0)
        original = copy.deepcopy(contract)
        lower, upper = self.core.command_limits(contract, {}, 0.05)
        self.assertEqual(contract, original)
        self.assertAlmostEqual(math.degrees(lower[6]), -7.2)
        self.assertAlmostEqual(math.degrees(upper[6]), 7.2)
        for row, lo, hi in zip(contract, lower, upper):
            self.assertGreater(lo, row["lower"])
            self.assertLess(hi, row["upper"])
            self.assertAlmostEqual(hi - lo, 0.9 * (row["upper"] - row["lower"]))
        lo0, hi0 = self.core.command_limits(contract, {})
        self.assertEqual(lo0, [r["lower"] for r in contract])
        self.assertEqual(hi0, [r["upper"] for r in contract])

    def test_margin_rejects_invalid_and_intersects_legacy_caps(self):
        contract = self.core.sdk_joint_contract(*fixture())
        for margin in (-0.01, 0.5, 1.0, float("nan"), float("inf")):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                self.core.command_limits(contract, {}, margin)
        lower, upper = self.core.command_limits(contract, {"RF_PIP": 70.0}, 0.05)
        self.assertAlmostEqual(math.degrees(upper[3]), 70.0)
        with self.assertRaises(ValueError):
            self.core.command_limits(contract, {"RF_PIP": -85.0}, 0.05)

    def test_two_turns_survive_quaternion_wrap_and_sign_changes(self):
        angles = torch.linspace(0.0, 4 * math.pi, 161, dtype=torch.float64)
        quats = quat(angles)
        quats[::3] *= -1
        axial, off = self.core.rotation_increment(quats[:-1], quats[1:], torch.tensor([0.0, 0.0, 2.0]))
        self.assertAlmostEqual(axial.sum().item(), 4 * math.pi, places=10)
        self.assertLess(off.abs().max().item(), 1e-12)
        reverse, _ = self.core.rotation_increment(quats[1:], quats[:-1], torch.tensor([0.0, 0.0, 1.0]))
        self.assertAlmostEqual((axial + reverse).sum().item(), 0.0, places=10)

    def test_all_six_world_directions_preserve_signed_turns_and_reward(self):
        angles = torch.linspace(0.0, 4 * math.pi, 161, dtype=torch.float64)
        for dimension in range(3):
            for sign in (-1.0, 1.0):
                axis = torch.zeros(3, dtype=torch.float64)
                axis[dimension] = sign
                rotations = quat(angles, axis.tolist())
                axial, off = self.core.rotation_increment(rotations[:-1], rotations[1:], axis)
                self.assertAlmostEqual(axial.sum().item(), 4 * math.pi, places=10)
                self.assertLess(off.max().item(), 1e-12)
                backward, _ = self.core.rotation_increment(rotations[:-1], rotations[1:], -axis)
                self.assertAlmostEqual(backward.sum().item(), -4 * math.pi, places=10)
                zero = torch.zeros(1, 3, dtype=torch.float64)
                terms = self.core.spin_reward(axis[None], zero, zero, torch.zeros(1, 13), axis, 1.0)
                self.assertAlmostEqual(terms["rotation"].item(), 2.0)
                self.assertAlmostEqual(terms["off_axis"].item(), 0.0)

    def test_off_axis_and_equivalent_quaternions(self):
        initial = quat([0.0, 0.0])
        current = quat([0.4, -0.7], (1.0, 0.0, 0.0)) * 3
        axial, off = self.core.rotation_increment(initial, current, torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(axial, torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(off, torch.tensor([0.4, 0.7], dtype=torch.float64))
        axial, off = self.core.rotation_increment(current, -current, torch.tensor([0.0, 0.0, 1.0]))
        self.assertTrue(torch.isfinite(axial).all() and torch.isfinite(off).all())
        self.assertLess((axial.abs() + off).max().item(), 1e-12)

    def test_delta_is_world_frame(self):
        # World z rotation following an existing x rotation must score along z.
        previous = quat([math.pi / 2], (1.0, 0.0, 0.0))
        current = torch.tensor(
            [
                [
                    math.cos(0.2) / math.sqrt(2),
                    math.cos(0.2) / math.sqrt(2),
                    math.sin(0.2) / math.sqrt(2),
                    math.sin(0.2) / math.sqrt(2),
                ]
            ],
            dtype=torch.float64,
        )
        axial, off = self.core.rotation_increment(previous, current, torch.tensor([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(axial.item(), 0.4, places=10)
        self.assertLess(off.item(), 1e-12)

    def test_reward_stationary_reverse_cap_and_off_axis(self):
        omega = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 20.0], [2.0, 0.0, 2.0]])
        zero = torch.zeros(5, 3)
        rewards = self.core.spin_reward(omega, zero, zero, torch.zeros(5, 13), torch.tensor([0.0, 0.0, 2.0]), 2.0)
        torch.testing.assert_close(rewards["total"], torch.tensor([0.0, -2.0, 2.0, 2.0, 1.4]))
        self.assertTrue(all(value.shape == (5,) for value in rewards.values()))
        self.assertEqual(set(rewards), {"rotation", "off_axis", "position", "linear_velocity", "action_rate", "total"})

    def test_reward_penalties_and_invalid_target_speed(self):
        zero = torch.zeros(1, 3)
        rewards = self.core.spin_reward(
            zero,
            torch.tensor([[2.0, 0.0, 0.0]]),
            torch.tensor([[0.05, 0.0, 0.0]]),
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([0.0, 0.0, 1.0]),
            1.0,
        )
        self.assertAlmostEqual(rewards["total"].item(), -0.72, places=6)
        for speed in [0.0, -1.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                self.core.spin_reward(zero, zero, zero, zero, torch.tensor([0.0, 0.0, 1.0]), speed)

    def test_zero_net_oscillation_cannot_earn_rotation_reward(self):
        # Four slow forward steps and one fast reverse step cancel displacement.
        omega = torch.tensor([[0.0, 0.0, 1.0]] * 4 + [[0.0, 0.0, -4.0]])
        zero = torch.zeros(5, 3)
        rewards = self.core.spin_reward(omega, zero, zero, torch.zeros(5, 13), torch.tensor([0.0, 0.0, 1.0]), 1.0)
        self.assertLessEqual(rewards["rotation"].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
