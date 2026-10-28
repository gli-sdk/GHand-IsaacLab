import importlib.util
from contextlib import redirect_stdout, redirect_stderr
from enum import IntEnum
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).parents[2] / "scripts/cube_spin/reset_hand.py"


class ResetTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "Reset entry is missing")
        spec = importlib.util.spec_from_file_location("reset_spin_hand", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.limits = {f"J{i}": (0.0, 90.0) for i in range(13)}

    def test_left_cli_uses_neutral_sdk_pose_without_reading_right_reference(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "reset.json"
            argv = [str(SCRIPT), "--hand-side", "left", "--output", str(output)]
            with (
                patch.object(self.module.sys, "argv", argv),
                patch.object(self.module, "REFERENCE", Path(temporary) / "no_right_record.json"),
                patch.object(self.module, "execute") as execute,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                try:
                    self.module.main()
                except SystemExit as error:
                    self.fail(f"Left reset CLI missing: {error}")
            execute.assert_not_called()
            result = json.loads(output.read_text())
            self.assertEqual(result["requested_hand_side"], "left")
            self.assertEqual(result["reference"], "sdk_neutral_left")
            self.assertEqual(len(result["target_deg"]), 13)
            self.assertEqual(result["target_deg"]["THUMB_TMC_AA"], 20.0)
            self.assertTrue(all(v == 0.0 for k, v in result["target_deg"].items() if k != "THUMB_TMC_AA"))

    def test_left_reset_accepts_304_and_rejects_wrong_side_or_unknown_firmware(self):
        for actual, firmware, expected in [
            ("left_hand", "V3.0.4", True),
            ("right_hand", "V3.0.6", False),
            ("left_hand", "unknown", False),
        ]:
            with self.subTest(actual=actual, firmware=firmware):
                result, _, packets = self.run_fake_reset(
                    reach_after=0.0, requested_side="left", actual_side=actual, firmware=firmware
                )
                self.assertEqual(
                    result["complete"],
                    expected,
                    {k: v for k, v in result.items() if k in ("error", "identity", "complete", "motion_commands")},
                )
                self.assertEqual(bool(packets), expected)
                if expected:
                    self.assertEqual(result["identity"]["hand_type"], "left_hand")
                    self.assertEqual(result["target_deg"]["THUMB_TMC_AA"], 20.0)
                    self.assertTrue(result["returned_near_reference"])

    def test_default_output_is_one_reusable_reset_file(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "reset_latest.json"
            with (
                patch.object(self.module, "DEFAULT_OUTPUT", output, create=True),
                patch.object(self.module.sys, "argv", [str(SCRIPT)]),
                patch.object(self.module, "execute") as execute,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                for _ in range(2):
                    try:
                        self.module.main()
                    except SystemExit as error:
                        self.fail(f"Reset should reuse the default output without --output: {error}")
                execute.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [output])
            result = json.loads(output.read_text())
            self.assertEqual(result["mode"], "restore_pre_policy_open_pose")
            self.assertFalse(result["hardware_executed"])
            self.assertIsNone(result["feedback_tolerance_deg"])
            self.assertIsNone(result["host_temperature_limit_c"])

    def test_explicit_output_replaces_previous_reset_record(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "reset.json"
            output.write_text(json.dumps({"mode": "restore_pre_policy_open_pose", "old_marker": "x" * 10000}))
            argv = [str(SCRIPT), "--output", str(output)]
            with (
                patch.object(self.module.sys, "argv", argv),
                patch.object(self.module, "execute") as execute,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                try:
                    self.module.main()
                except SystemExit as error:
                    self.fail(f"Reset should overwrite the previous output: {error}")
                execute.assert_not_called()
            result = json.loads(output.read_text())
            self.assertNotIn("old_marker", result)
            self.assertEqual(len(result["target_deg"]), 13)

    def test_unwritable_output_fails_before_any_hardware_execution(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "directory"
            output.mkdir()
            argv = [str(SCRIPT), "--execute", "--output", str(output)]
            with (
                patch.object(self.module.sys, "argv", argv),
                patch.object(self.module, "execute") as execute,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises((OSError, SystemExit)):
                    self.module.main()
                execute.assert_not_called()

    def test_reference_accepts_sensor_roundoff_but_rejects_bad_data(self):
        ref = dict.fromkeys(self.limits, 0.0)
        ref["J0"] = -0.00001
        self.assertEqual(self.module.reference_targets(ref, self.limits)["J0"], 0.0)
        ref["J0"] = 92.5
        self.assertEqual(self.module.reference_targets(ref, self.limits)["J0"], 90.0)
        for value in (-3.01, 93.01, math.nan):
            ref["J0"] = value
            with self.assertRaises(ValueError):
                self.module.reference_targets(ref, self.limits)

    def test_ramp_returns_to_reference_without_overshoot(self):
        initial = dict.fromkeys(self.limits, 90.0)
        target = dict.fromkeys(self.limits, 0.0)
        previous = initial
        for step in range(81):
            row = self.module.ramp_target(initial, target, step * 0.05)
            for name, value in row.items():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, previous[name])
                self.assertLessEqual(previous[name] - value, 1.500001)
            previous = row
        self.assertEqual(self.module.ramp_target(initial, target, 0.0), initial)
        self.assertEqual(self.module.ramp_target(initial, target, 3.0), target)
        with self.assertRaises(ValueError):
            self.module.ramp_target(initial, target, math.nan)

    def test_final_check_requires_every_sample_within_one_degree(self):
        target = dict.fromkeys(self.limits, 0.0)
        samples = [{"feedback_time": i * 0.01, "angles_deg": dict.fromkeys(self.limits, 1.0)} for i in range(51)]
        self.assertTrue(self.module.returned_to_reference(samples, target))
        samples[0]["angles_deg"]["J1"] = 1.1
        self.assertFalse(self.module.returned_to_reference(samples, target))
        self.assertFalse(self.module.returned_to_reference([], target))

    def test_final_check_rejects_stable_but_incomplete_return(self):
        target = dict.fromkeys(self.limits, 0.0)
        samples = [{"feedback_time": i * 0.01, "angles_deg": dict.fromkeys(self.limits, 4.0)} for i in range(51)]
        self.assertFalse(self.module.returned_to_reference(samples, target))

    def test_final_check_rejects_motion_inside_position_tolerance(self):
        target = dict.fromkeys(self.limits, 0.0)
        samples = [{"feedback_time": i * 0.01, "angles_deg": dict.fromkeys(self.limits, 0.0)} for i in range(51)]
        samples[25]["angles_deg"]["J1"] = 0.6
        self.assertFalse(self.module.returned_to_reference(samples, target))

    def run_fake_reset(
        self,
        reach_after=5.0,
        rebound=False,
        stop_accepted=True,
        fault=False,
        requested_side="right",
        actual_side="right_hand",
        firmware="V3.0.6",
        initial_ff_mcp=20.0,
    ):
        # Only the hardware boundary and wall clock are replaced; execute() is real.
        from ghand.tasks.cube_spin.core import sdk_joint_contract

        root = SCRIPT.parents[2]
        contract = {
            "joint_contract": sdk_joint_contract(
                json.loads((root / "assets/ghand5_right.manifest.json").read_text()),
                json.loads((root / "config/cube_spin/hand.json").read_text()),
            )
        }
        config = json.loads((SCRIPT.parents[2] / "config/cube_spin/hand.json").read_text())
        ids = IntEnum("JointId", {j["name"].upper(): j["sdk_id"] for j in contract["joint_contract"]})
        limits = {ids[j["id"]]: (j["min"], j["max"]) for j in config["joints"] if "min" in j}
        reference = (
            {"initial_deg": {j.name: max(lo, min(hi, 0.0)) for j, (lo, hi) in limits.items()}}
            if requested_side == "left"
            else json.loads(self.module.REFERENCE.read_text())
        )
        target = self.module.reference_targets(reference["initial_deg"], {j.name: v for j, v in limits.items()})
        actual = target.copy()
        actual["FF_MCP"] = initial_ff_mcp
        now, packets, stopped, closed = [100.0], [], [], []

        def move(packet):
            packets.append((now[0] - 100.0, packet))
            actual.update({c.id.name: c.angle for c in packet})
            return True

        def stop():
            stopped.append(now[0] - 100.0)
            return stop_accepted

        hand = SimpleNamespace(
            _comm=SimpleNamespace(_client=SimpleNamespace(_master=object())),
            _joint_limits=limits,
            open=lambda _: True,
            get_device_name=lambda: "GHand 5",
            get_firmware_version=lambda: firmware,
            get_hand_type=lambda: SimpleNamespace(value=actual_side),
            is_connected=lambda: not closed,
            move_joints=move,
            stop=stop,
            close=lambda: closed.append(True),
        )
        sdk = SimpleNamespace(
            GHand=lambda *args: hand,
            ProductType=SimpleNamespace(GHand5=1),
            CommType=SimpleNamespace(ETHERCAT=1),
            JointCommand=SimpleNamespace,
        )

        def feedback(*_):
            now[0] += 0.001
            angles = actual.copy()
            if reach_after is None or now[0] - 100.0 < reach_after or (stopped and rebound):
                angles["FF_MCP"] = max(4.0, angles["FF_MCP"])
            return {
                "feedback_time": now[0],
                "wkc": 3,
                "hand_state": 0,
                "hand_error": 22 if fault and now[0] > 100.2 else 0,
                "temperature": 45,
                "angles_deg": angles,
                "joint_states": dict.fromkeys(angles, 0),
                "joint_errors": dict.fromkeys(angles, 0),
            }

        def sleep(seconds):
            now[0] += seconds

        result = {"requested_hand_side": requested_side}
        with (
            patch.dict(self.module.sys.modules, {"ghand": sdk}),
            patch.object(self.module, "fresh_feedback", feedback),
            patch.object(self.module.time, "monotonic", lambda: now[0]),
            patch.object(self.module.time, "sleep", sleep),
            patch.object(self.module.signal, "signal"),
            patch.object(self.module.signal, "alarm"),
            redirect_stdout(io.StringIO()),
        ):
            self.module.execute(result, reference)
        self.assertEqual(len(stopped), 1)
        self.assertEqual(closed, [True])
        for _, packet in packets:
            for command in packet:
                self.assertEqual((command.speed, command.torque), (15, 20))
                self.assertTrue(limits[command.id][0] <= command.angle <= limits[command.id][1])
        return result, stopped[0], packets

    def test_reset_has_no_dependency_on_historical_run_artifacts(self):
        read_text = Path.read_text

        def without_runs(path, *args, **kwargs):
            if "runs" in path.parts:
                raise AssertionError("Reset must not need historical run artifacts: " + str(path))
            return read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", without_runs):
            result, _, packets = self.run_fake_reset(reach_after=0.0)
        self.assertTrue(result["complete"], result.get("error"))
        self.assertTrue(packets)

    def test_reset_accepts_finite_feedback_outside_sdk_and_bounds_all_targets(self):
        result, _, packets = self.run_fake_reset(reach_after=0.0, initial_ff_mcp=94.0)
        self.assertTrue(result["complete"], result.get("error"))
        self.assertGreater(len(packets), 0)
        self.assertEqual(result["initial_measured_deg"]["FF_MCP"], 94.0)
        # run_fake_reset checks every emitted command against the SDK limits.

    def test_waits_for_slow_joint_to_arrive_and_stabilize_before_stopping(self):
        result, stop_time, _ = self.run_fake_reset(reach_after=5.0)
        self.assertTrue(result["complete"], result.get("error"))
        self.assertGreaterEqual(stop_time, 6.0)
        self.assertLessEqual(stop_time, 8.1)
        self.assertTrue(result["settled_before_stop"])
        self.assertLessEqual(max(abs(v) for v in result["final_error_deg"].values()), 1.0)

    def test_incomplete_return_times_out_and_names_the_joint(self):
        result, stop_time, _ = self.run_fake_reset(reach_after=None)
        self.assertFalse(result["complete"])
        self.assertIn("FF_MCP", result["error"])
        self.assertIn("timeout", result["error"].lower())
        self.assertLessEqual(stop_time, 8.1)
        self.assertAlmostEqual(result["final_error_deg"]["FF_MCP"], 4.0)

    def test_rebound_after_stop_is_reported_as_failure(self):
        result, _, _ = self.run_fake_reset(reach_after=0.0, rebound=True)
        self.assertFalse(result["complete"])
        self.assertIn("FF_MCP", result["stop_error"])

    def test_rejected_stop_cannot_report_success(self):
        result, _, _ = self.run_fake_reset(reach_after=0.0, stop_accepted=False)
        self.assertFalse(result["complete"])
        self.assertIn("STOP", result["stop_error"])

    def test_firmware_temperature_fault_stops_without_waiting_for_settle(self):
        result, stop_time, _ = self.run_fake_reset(fault=True)
        self.assertFalse(result["complete"])
        self.assertIn("Device fault", result["error"])
        self.assertIn("hand_error=22", result["error"])
        self.assertLess(stop_time, 0.3)


if __name__ == "__main__":
    unittest.main()
