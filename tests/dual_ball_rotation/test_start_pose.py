import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
import unittest
import torch
import yaml
from ghand.tasks.dual_ball_rotation.start_pose import load_start_pose, PoseAdapter

ROOT = Path(__file__).resolve().parents[2]


class StartPoseTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT / "models/dual_ball_rotation/student/contract.json").read_text())
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.p = Path(self.folder.name)
        (self.p / "params").mkdir()
        self.q = (torch.tensor(self.c["joint_contract"]["lower"]) + torch.tensor(self.c["joint_contract"]["upper"])) / 2
        self.cfg = {
            "robot_cfg": {"init_state": {"joint_pos": dict(zip(self.c["joint_contract"]["names"], self.q.tolist()))}},
            "palm_tilt_range_deg": [25.0, 25.0],
            "ball_radius": 0.0175,
            "ball_mass": 0.03,
        }
        self.save()

    def save(self):
        (self.p / "params/env.yaml").write_text(yaml.safe_dump(self.cfg))
        (self.p / "contract.json").write_text(json.dumps(self.c))

    def test_pose_uses_named_saved_joints_and_excludes_passive_joints(self):
        self.cfg["robot_cfg"]["init_state"]["joint_pos"]["Thumb_IP"] = 0.7
        self.save()
        q, setup = load_start_pose(self.p, self.c)
        torch.testing.assert_close(q, self.q[None])
        self.assertEqual(setup["palm_tilt_range_deg"], [25.0, 25.0])
        self.assertEqual(setup["ball_diameter_mm"], 35)
        self.assertEqual(setup["mass_g_each"], 30)

    def test_missing_nonfinite_or_out_of_range_pose_is_rejected(self):
        name = self.c["joint_contract"]["names"][0]
        for value in (None, float("nan"), 10.0):
            if value is None:
                self.cfg["robot_cfg"]["init_state"]["joint_pos"].pop(name)
            else:
                self.cfg["robot_cfg"]["init_state"]["joint_pos"][name] = value
            self.save()
            with self.assertRaises(ValueError):
                load_start_pose(self.p, self.c)

    def test_ramp_reaches_pose_and_holds_without_actor_or_return_to_open(self):
        start = torch.tensor([self.c["joint_contract"]["lower"]])
        target = self.q[None]
        a = PoseAdapter(self.c, start, target, ramp_seconds=4.0, hold_seconds=2.0)
        rows = []
        for i in range(120):
            q, act = a.propose(start, i * 0.05)
            rows.append(q)
            a.acknowledge(q)
            self.assertTrue(torch.equal(act, torch.zeros(1, 13)))
        torch.testing.assert_close(rows[-1], target)
        torch.testing.assert_close(rows[79], target)
        for q in rows[80:]:
            torch.testing.assert_close(q, target)
        self.assertTrue(all(((q >= start - 1e-6) & (q <= target + 1e-6)).all() for q in rows))
        self.assertLess(float((rows[0] - start).abs().max()), 0.01)

    def test_targets_require_acknowledgement_and_increasing_timestamps(self):
        a = PoseAdapter(self.c, self.q[None], self.q[None])
        q, _ = a.propose(self.q[None], 0.0)
        with self.assertRaises(ValueError):
            a.propose(self.q[None], 0.05)
        a.acknowledge(q)
        with self.assertRaises(ValueError):
            a.propose(self.q[None], 0.0)

    def test_cli_offline_never_opens_sdk_and_execute_needs_empty_hand(self):
        spec = importlib.util.spec_from_file_location(
            "prepare_test", ROOT / "scripts/dual_ball_rotation/prepare_hand.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        output = self.p / "latest.json"
        with (
            patch.object(m, "execute", side_effect=AssertionError("SDK should not open")),
            redirect_stdout(io.StringIO()),
        ):
            m.main(["--policy-dir", str(self.p), "--output", str(output)])
        d = json.loads(output.read_text())
        self.assertFalse(d["hardware_motion_executed"])
        self.assertEqual(len(d["trajectory"]["targets_deg"]), 120)
        with patch.object(m, "execute") as run, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            m.main(["--policy-dir", str(self.p), "--execute", "--onsite-cutoff-ready"])
        run.assert_not_called()

    def test_deployment_offline_uses_and_reports_saved_start_pose(self):
        spec = importlib.util.spec_from_file_location(
            "deployment_pose_test", ROOT / "scripts/dual_ball_rotation/deploy.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out = self.p / "deploy.json"
        with patch.object(m, "execute", side_effect=AssertionError("No device access")), redirect_stdout(io.StringIO()):
            m.main(["--duration", ".1", "--output", str(out)])
        d = json.loads(out.read_text())
        c = json.loads((ROOT / "models/dual_ball_rotation/student/contract.json").read_text())
        expected, setup = load_start_pose(ROOT / "models/dual_ball_rotation/student", c)
        self.assertEqual(d["trained_setup"], setup)
        torch.testing.assert_close(torch.tensor(d["offline_initial_deg"]), torch.rad2deg(expected[0]))


if __name__ == "__main__":
    unittest.main()
