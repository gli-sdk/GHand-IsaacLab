import copy
import json
from pathlib import Path
import unittest
import shutil
from tempfile import TemporaryDirectory
import torch
from ghand.tasks.dual_ball_rotation.deployment import PolicyAdapter, load_policy, validate_contract
from ghand.tasks.dual_ball_rotation.student import JointHistory

ROOT = Path(__file__).resolve().parents[2]


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.contract, cls.actor, cls.meta = load_policy(ROOT / "models/dual_ball_rotation/student")
        cls.low = torch.tensor(cls.contract["joint_contract"]["lower"])
        cls.high = torch.tensor(cls.contract["joint_contract"]["upper"])

    def test_export_and_contract_match(self):
        joints = validate_contract(self.contract)
        self.assertEqual(len(joints), 13)
        self.assertEqual([j["sdk_id"] for j in joints], [17, 16, 14, 13, 11, 10, 8, 7, 6, 4, 3, 2, 1])
        self.assertEqual(self.actor(torch.zeros(1, 780)).shape, (1, 13))

    def test_hardware_adapter_matches_simulation_history_and_actions(self):
        q = self.low[None].clone()
        target = q.clone()
        history = JointHistory(1, 30, self.low, self.high)
        history.reset(q, target)
        adapter = PolicyAdapter(self.contract, self.actor, q)
        generator = torch.Generator().manual_seed(123)
        for step in range(100):
            with torch.inference_mode():
                expected_action = self.actor(history.observation())
            expected = (target + 0.075 * expected_action).clamp(self.low, self.high)
            proposed, action = adapter.propose(q, step * 0.05)
            torch.testing.assert_close(action, expected_action)
            torch.testing.assert_close(proposed, expected)
            adapter.acknowledge(proposed)
            target = proposed.clone()
            q = (q + 0.3 * (target - q) + 0.001 * torch.randn(q.shape, generator=generator)).clamp(self.low, self.high)
            history.advance(q, target, torch.zeros(1, dtype=torch.bool))

    def test_accumulates_acknowledged_commands_not_feedback(self):
        q = ((self.low + self.high) * 0.5)[None]
        adapter = PolicyAdapter(self.contract, lambda _: torch.ones(1, 13), q)
        first, _ = adapter.propose(q, 0.0)
        torch.testing.assert_close(adapter.command, q)
        accepted = first - 0.01
        adapter.acknowledge(accepted)
        second, _ = adapter.propose(q, 0.05)
        torch.testing.assert_close(second, (accepted + 0.075).clamp(self.low, self.high))

    def test_all_targets_use_training_limits(self):
        for sign, q in [(1.0, self.high[None]), (-1.0, self.low[None])]:
            adapter = PolicyAdapter(self.contract, lambda _: torch.full((1, 13), sign), q)
            target, _ = adapter.propose(q, 0.0)
            torch.testing.assert_close(target, q)

    def test_rejects_wrong_contract(self):
        for key, value in [
            ("observation_size", 781),
            ("step_dt", 0.1),
            ("action_step_rad", 0.1),
            ("history_steps", 29),
        ]:
            c = copy.deepcopy(self.contract)
            c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(c)
        c = copy.deepcopy(self.contract)
        c["joint_contract"]["names"].reverse()
        with self.assertRaises(ValueError):
            validate_contract(c)
        c = copy.deepcopy(self.contract)
        c["joint_contract"]["upper"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_contract(c)

    def test_direction_metadata_must_match_checkpoint(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ["student.pt", "actor.ts", "contract.json"]:
                shutil.copy2(ROOT / "models/dual_ball_rotation/student" / name, d / name)
            c = json.loads((d / "contract.json").read_text())
            c["rotation_direction"] = -1
            (d / "contract.json").write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                load_policy(d)

    def test_rejects_stale_time_or_bad_action(self):
        q = self.low[None]
        adapter = PolicyAdapter(self.contract, self.actor, q)
        target, _ = adapter.propose(q, 1.0)
        adapter.acknowledge(target)
        with self.assertRaises(ValueError):
            adapter.propose(q, 1.0)
        for bad in [torch.full((1, 13), float("nan")), torch.ones(1, 14), torch.full((1, 13), 2.0)]:
            adapter = PolicyAdapter(self.contract, lambda _: bad, q)
            with self.assertRaises(ValueError):
                adapter.propose(q, 0.0)


if __name__ == "__main__":
    unittest.main()
