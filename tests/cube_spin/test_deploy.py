import importlib.util
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from enum import IntEnum
import unittest
from unittest.mock import Mock, patch
from tempfile import TemporaryDirectory

import torch

SCRIPT = Path(__file__).parents[2] / "scripts/cube_spin/deploy.py"
PROJECT = Path(__file__).resolve().parents[2]
POLICY = PROJECT / "models/cube_spin/student"


class DeployTests(unittest.TestCase):
    def test_hand_specific_model_rejects_other_hand_before_device_creation(self):
        sdk = SimpleNamespace(
            GHand=Mock(side_effect=AssertionError("Device must not be created")),
            ProductType=SimpleNamespace(GHand5=1),
            CommType=SimpleNamespace(ETHERCAT=1),
            JointCommand=SimpleNamespace,
        )
        with patch.dict(self.module.sys.modules, {"ghand": sdk}), self.assertRaisesRegex(ValueError, "hand side"):
            self.module.execute(dict(self.contract, hand_side="left"), None, {"requested_hand_side": "right"})
        sdk.GHand.assert_not_called()

    def test_trained_target_caps_preserve_physical_normalization_and_command_history(self):
        contract = json.loads(json.dumps(self.contract))
        contract["command_upper_deg"] = {"RF_PIP": 80.0, "LF_PIP": 64.0}
        observations = []

        def actor(obs):
            observations.append(obs.clone())
            return torch.ones(1, 13)

        adapter = self.module.PolicyAdapter(contract, actor, self.high[None])
        target, _ = adapter.propose(self.high[None], 0.0)
        for name, upper in contract["command_upper_deg"].items():
            self.assertAlmostEqual(float(torch.rad2deg(target[0, adapter.names.index(name)])), upper, places=4)
        torch.testing.assert_close(adapter.lower, self.low)
        torch.testing.assert_close(adapter.upper, self.high)
        torch.testing.assert_close(adapter.history.upper, self.high)
        adapter.acknowledge(target)
        adapter.propose(self.high[None], 0.05)
        expected = 2.0 * (target - self.low) / (self.high - self.low) - 1.0
        torch.testing.assert_close(observations[-1][:, -14:-1], expected)
        # A legacy user ceiling must not expand a tighter trained ceiling.
        adapter = self.module.PolicyAdapter(contract, actor, self.high[None], ring_pip_upper_deg=85.0)
        target, _ = adapter.propose(self.high[None], 0.0)
        self.assertAlmostEqual(float(torch.rad2deg(target[0, adapter.names.index("RF_PIP")])), 80.0, places=4)

    def test_margin_targets_cover_all_joints_without_renormalizing_history(self):
        contract = dict(self.contract, command_margin_fraction=0.05)
        lo = self.low + 0.05 * (self.high - self.low)
        hi = self.high - 0.05 * (self.high - self.low)
        for sign, start, expected in ((-1.0, self.low[None], lo[None]), (1.0, self.high[None], hi[None])):
            seen = []

            def actor(obs):
                seen.append(obs.clone())
                return torch.full((1, 13), sign)

            adapter = self.module.PolicyAdapter(contract, actor, start)
            target, _ = adapter.propose(start, 0.0)
            torch.testing.assert_close(target, expected)
            torch.testing.assert_close(adapter.history.lower, self.low)
            torch.testing.assert_close(adapter.history.upper, self.high)
            adapter.acknowledge(target)
            adapter.propose(start, 0.05)
            torch.testing.assert_close(seen[-1][:, -14:-1], 2.0 * (target - self.low) / (self.high - self.low) - 1.0)

    def test_trained_target_caps_reject_invalid_contract(self):
        for caps in ({"unknown": 20.0}, {"RF_PIP": 91.0}, {"LF_PIP": float("nan")}, {"LF_PIP": 0.0}):
            contract = dict(self.contract, command_upper_deg=caps)
            with self.subTest(caps=caps), self.assertRaises(ValueError):
                self.module.PolicyAdapter(contract, lambda obs: torch.ones(1, 13), self.start)

    def test_speed_torque_cli_are_logged_and_leave_policy_targets_unchanged(self):
        previous = None
        for speed, torque in ((15, 20), (25, 20), (15, 30)):
            with TemporaryDirectory() as temporary:
                output = Path(temporary) / "trial.json"
                argv = [
                    str(SCRIPT),
                    "--policy-dir",
                    str(POLICY),
                    "--hand-side",
                    "left",
                    "--duration",
                    ".1",
                    "--speed",
                    str(speed),
                    "--torque",
                    str(torque),
                    "--output",
                    str(output),
                ]
                with (
                    patch.object(self.module.sys, "argv", argv),
                    patch.object(self.module, "execute") as execute,
                    redirect_stdout(io.StringIO()),
                ):
                    self.module.main()
                execute.assert_not_called()
                result = json.loads(output.read_text())
                self.assertEqual((result["speed_percent"], result["torque_percent"]), (speed, torque))
                if previous is not None:
                    self.assertEqual(result["samples"], previous)
                previous = result["samples"]

    def test_speed_torque_percent_reject_invalid_before_device_access(self):
        for value in ("0", "101", "15.5", "nan", "inf", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.sdk_percent(value)
        for value in ("1", "25", "100"):
            self.assertEqual(self.module.sdk_percent(value), int(value))

    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "Deployment entry point is missing")
        spec = importlib.util.spec_from_file_location("deploy_spin", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.contract = json.loads((POLICY / "contract.json").read_text())
        self.low = torch.tensor([j["lower"] for j in self.contract["joint_contract"]])
        self.high = torch.tensor([j["upper"] for j in self.contract["joint_contract"]])
        self.start = (self.low + self.high)[None] * 0.5
        self.observations = []

        def actor(obs):
            self.observations.append(obs.clone())
            return torch.ones((1, 13))

        self.adapter = self.module.PolicyAdapter(self.contract, actor, self.start)

    def test_left_cli_keeps_original_actor_targets_and_no_extra_margin(self):
        results = []
        for side in ("right", "left"):
            with TemporaryDirectory() as temporary:
                output = Path(temporary) / "result.json"
                argv = [
                    str(SCRIPT),
                    "--policy-dir",
                    str(POLICY),
                    "--hand-side",
                    side,
                    "--duration",
                    "10",
                    "--output",
                    str(output),
                ]
                with (
                    patch.object(self.module.sys, "argv", argv),
                    patch.object(self.module, "execute") as execute,
                    redirect_stdout(io.StringIO()),
                ):
                    try:
                        self.module.main()
                    except SystemExit as error:
                        self.fail(f"Explicit hand-side CLI missing: {error}")
                execute.assert_not_called()
                result = json.loads(output.read_text())
                self.assertEqual(result["requested_hand_side"], side)
                self.assertIsNone(result["ring_pip_upper_deg"])
                self.assertIsNone(result["extra_action_cap"])
                self.assertIsNone(result["feedback_tolerance_deg"])
                self.assertIsNone(result["host_temperature_limit_c"])
                self.assertEqual(result["action_scale"], 1.0)
                self.assertEqual(result["offline_steps"], 200)
                results.append(result)
        self.assertEqual(results[0]["samples"], results[1]["samples"])

    def test_left_device_requires_explicit_left_selection_before_position_commands(self):
        joints = self.contract["joint_contract"]
        ids = IntEnum("JointId", {j["name"].upper(): j["sdk_id"] for j in joints})
        limits = {ids[j["name"].upper()]: (math.degrees(j["lower"]), math.degrees(j["upper"])) for j in joints}
        for requested, actual_side, firmware, expected in [
            ("left", "left_hand", "V3.0.6", True),
            ("left", "left_hand", "V3.0.4", True),
            ("left", "right_hand", "V3.0.4", False),
            ("right", "right_hand", "V3.0.4", True),
            ("left", "left_hand", "V3.0.5", False),
            ("left", "right_hand", "V3.0.6", False),
            ("right", "left_hand", "V3.0.6", False),
            ("left", "left_hand", "unknown", False),
        ]:
            with self.subTest(requested=requested, actual=actual_side, firmware=firmware):
                actual = {j["name"].upper(): math.degrees(float(self.start[0, i])) for i, j in enumerate(joints)}
                now = [100.0]
                packets = []
                stopped = []
                closed = []

                def move(packet):
                    packets.append(packet)
                    actual.update({c.id.name: c.angle for c in packet})
                    return True

                hand = SimpleNamespace(
                    _comm=SimpleNamespace(_client=SimpleNamespace(_master=object())),
                    _joint_limits=limits,
                    open=lambda _: True,
                    get_device_name=lambda: "GHand 5",
                    get_firmware_version=lambda: firmware,
                    get_hand_type=lambda: SimpleNamespace(value=actual_side),
                    is_connected=lambda: not closed,
                    move_joints=move,
                    stop=lambda: stopped.append(True) or True,
                    close=lambda: closed.append(True),
                )
                sdk = SimpleNamespace(
                    GHand=lambda *a: hand,
                    ProductType=SimpleNamespace(GHand5=1),
                    CommType=SimpleNamespace(ETHERCAT=1),
                    JointCommand=SimpleNamespace,
                )

                def feedback(*_):
                    now[0] += 0.001
                    return {
                        "feedback_time": now[0],
                        "wkc": 3,
                        "hand_state": 0,
                        "hand_error": 0,
                        "temperature": 42,
                        "angles_deg": actual.copy(),
                        "joint_states": dict.fromkeys(actual, 0),
                        "joint_errors": dict.fromkeys(actual, 0),
                    }

                def sleep(dt):
                    now[0] += dt

                result = {
                    "duration_seconds": 0.1,
                    "requested_hand_side": requested,
                    "hardware_policy_executed": False,
                    "speed_percent": 25,
                    "torque_percent": 20,
                }
                with (
                    patch.dict(self.module.sys.modules, {"ghand": sdk}),
                    patch.object(self.module, "fresh_feedback", feedback),
                    patch.object(self.module.time, "monotonic", lambda: now[0]),
                    patch.object(self.module.time, "sleep", sleep),
                    patch.object(self.module.signal, "signal"),
                    patch.object(self.module.signal, "setitimer"),
                    redirect_stdout(io.StringIO()),
                ):
                    self.module.execute(self.contract, lambda obs: torch.ones(1, 13), result)
                self.assertEqual(bool(result.get("policy_window_completed")), expected, result)
                self.assertEqual(len(packets), 2 if expected else 0)
                if expected:
                    for index, command in enumerate(packets[0]):
                        self.assertAlmostEqual(
                            command.angle, math.degrees(float(self.start[0, index]) + 0.075), places=4
                        )
                        self.assertEqual((command.speed, command.torque), (25, 20))
                self.assertEqual(stopped, [True])
                self.assertEqual(closed, [True])

    def test_known_policy_bundles_run_offline_and_log_selected_checkpoint(self):
        policies = ((POLICY, "c613485833794e46f6024ac343ab934ef7b90908a8ebbeb40a72e77ef03f1a38"),)
        for policy, expected_sha in policies:
            with self.subTest(policy=policy), TemporaryDirectory() as temporary:
                output = Path(temporary) / "offline.json"
                argv = [
                    str(SCRIPT),
                    "--policy-dir",
                    str(policy),
                    "--duration",
                    ".1",
                    "--hand-side",
                    "left",
                    "--output",
                    str(output),
                ]
                with (
                    patch.object(self.module.sys, "argv", argv),
                    patch.object(self.module, "execute") as execute,
                    redirect_stdout(io.StringIO()),
                ):
                    try:
                        self.module.main()
                    except ValueError as error:
                        self.fail(f"Known policy should load: {error}")
                execute.assert_not_called()
                result = json.loads(output.read_text())
                self.assertEqual(result["actor_sha256"], expected_sha)
                self.assertEqual(result["offline_steps"], 2)
                self.assertFalse(result["hardware_policy_executed"])
                self.assertTrue(all(len(row["action"]) == 13 for row in result["samples"]))

    def test_ring_ceiling_preserves_normalization_and_applied_target_history(self):
        seen = []

        def actor(obs):
            seen.append(obs.clone())
            return torch.ones(1, 13)

        start = self.high[None].clone()
        index = self.adapter.names.index("RF_PIP")
        start[0, index] = math.radians(88.0)
        try:
            adapter = self.module.PolicyAdapter(self.contract, actor, start, ring_pip_upper_deg=85.0)
        except TypeError as error:
            self.fail(f"Optional ring ceiling is missing: {error}")
        target, _ = adapter.propose(start, 0.0)
        expected = self.high[None].clone()
        expected[0, index] = math.radians(85.0)
        torch.testing.assert_close(target, expected)
        torch.testing.assert_close(adapter.history.upper, self.high)
        applied = target - 0.001
        adapter.acknowledge(applied)
        adapter.propose(start, 0.05)
        normalized = 2.0 * (applied - self.low) / (self.high - self.low) - 1.0
        torch.testing.assert_close(seen[-1][:, -14:-1], normalized)
        torch.testing.assert_close(seen[-1][:, -27:-14], 2.0 * (start - self.low) / (self.high - self.low) - 1.0)

    def test_ring_ceiling_rejects_nonfinite_or_expanded_limits(self):
        for value in (79.9, 90.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.PolicyAdapter(
                    self.contract, lambda x: torch.ones(1, 13), self.start, ring_pip_upper_deg=value
                )
        contract = json.loads(json.dumps(self.contract))
        next(j for j in contract["joint_contract"] if j["name"].upper() == "RF_PIP")["upper"] = math.radians(84.0)
        with self.assertRaises(ValueError):
            self.module.PolicyAdapter(contract, lambda x: torch.ones(1, 13), self.start, ring_pip_upper_deg=85.0)

    def test_ring_ceiling_cli_runs_offline_and_records_selected_policy(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "margin.json"
            argv = [
                str(SCRIPT),
                "--policy-dir",
                str(POLICY),
                "--duration",
                "10",
                "--ring-pip-upper-deg",
                "85",
                "--output",
                str(output),
            ]
            with (
                patch.object(self.module.sys, "argv", argv),
                patch.object(self.module, "execute") as execute,
                redirect_stdout(io.StringIO()),
            ):
                try:
                    self.module.main()
                except SystemExit as error:
                    self.fail(f"Ring ceiling CLI is missing: {error}")
            execute.assert_not_called()
            result = json.loads(output.read_text())
            self.assertEqual(result["ring_pip_upper_deg"], 85.0)
            self.assertEqual(result["policy_dir"], str(POLICY.resolve()))
            self.assertIsNone(result["feedback_tolerance_deg"])
            self.assertEqual(result["action_scale"], 1.0)
            self.assertEqual(result["offline_steps"], 200)
            index = result["joint_order"].index("RF_PIP")
            peak = max(row["target_deg"][index] for row in result["samples"])
            self.assertLessEqual(peak, 85.00001)
            self.assertGreater(peak, 84.9)

    def test_feedback_includes_raw_sdk_speed_and_torque_without_unit_conversion(self):
        import threading

        # Enum keys match the real SDK TPDO dictionary interface.
        jid = IntEnum("Joint", {"RF_PIP": 1}).RF_PIP
        tpdo = SimpleNamespace(
            hand=SimpleNamespace(state=1, error=0, temperature=40),
            joints={jid: SimpleNamespace(state=1, error=0, angle=61.5, speed=-17, torque=23)},
        )
        protocol = SimpleNamespace(Tpdo=SimpleNamespace(from_bytes=lambda raw, cfg: tpdo))
        client = SimpleNamespace(
            _data_lock=threading.Lock(),
            _last_valid_process_time=100.01,
            _last_valid_tpdo_snapshot=b"frame",
            _actual_wkc=3,
            _expected_wkc=3,
        )
        hand = SimpleNamespace(
            _comm=SimpleNamespace(_client=client, config=object(), _validate_tpdo_data=lambda raw: raw)
        )
        with (
            patch.dict(self.module.sys.modules, {"ghand.comm.ethercat_protocol": protocol}),
            patch.object(self.module.time, "monotonic", lambda: 100.02),
        ):
            sample = self.module.fresh_feedback(hand, 100.0)
        self.assertEqual(sample["angles_deg"], {"RF_PIP": 61.5})
        self.assertEqual(sample.get("joint_speed_sdk_raw"), {"RF_PIP": -17})
        self.assertEqual(sample.get("joint_torque_sdk_raw"), {"RF_PIP": 23})

    def test_unknown_checkpoint_is_rejected_before_actor_or_hardware_loading(self):
        with TemporaryDirectory() as temporary:
            policy = Path(temporary)
            (policy / "contract.json").write_text(json.dumps(self.contract))
            (policy / "actor.ts").write_bytes(b"unknown or damaged checkpoint")
            argv = [str(SCRIPT), "--policy-dir", str(policy), "--output", str(policy / "result.json")]
            with (
                patch.object(self.module.sys, "argv", argv),
                patch.object(self.module.torch.jit, "load") as load,
                patch.object(self.module, "execute") as execute,
                self.assertRaisesRegex(ValueError, "Unexpected actor checkpoint"),
            ):
                self.module.main()
            load.assert_not_called()
            execute.assert_not_called()

    def test_all_thirteen_actions_match_training_integration_without_pilot_cap(self):
        target, action = self.adapter.propose(self.start, 0.0)
        expected = (self.start + 0.075).clamp(self.low, self.high)
        torch.testing.assert_close(target, expected)
        self.assertEqual(int((target > self.start).sum()), 13)
        self.assertGreater(float(torch.rad2deg(target - self.start).max()), 4.0)

    def test_only_acknowledged_commands_enter_history(self):
        target, _ = self.adapter.propose(self.start, 0.0)
        second, _ = self.adapter.propose(self.start, 0.05)
        torch.testing.assert_close(second, target)
        self.adapter.acknowledge(target)
        self.adapter.propose(self.start, 0.10)
        normalized = 2 * (target - self.low) / (self.high - self.low) - 1
        torch.testing.assert_close(self.observations[-1][:, -14:-1], normalized)

    def test_action_scale_halves_increment_and_history_uses_accepted_scaled_target(self):
        seen = []

        def actor(obs):
            seen.append(obs.clone())
            return torch.ones(1, 13)

        try:
            adapter = self.module.PolicyAdapter(self.contract, actor, self.start, action_scale=0.5)
        except TypeError as error:
            self.fail(f"Action scaling is missing: {error}")
        target, action = adapter.propose(self.start, 0.0)
        torch.testing.assert_close(action, torch.ones(1, 13))
        torch.testing.assert_close(target, (self.start + 0.0375).clamp(self.low, self.high))
        again, _ = adapter.propose(self.start, 0.05)
        torch.testing.assert_close(again, target)  # Unacknowledged output is not integrated.
        applied = target - 0.001
        adapter.acknowledge(applied)
        following, _ = adapter.propose(self.start, 0.1)
        torch.testing.assert_close(following, (applied + 0.0375).clamp(self.low, self.high))
        normalized = 2 * (applied - self.low) / (self.high - self.low) - 1
        torch.testing.assert_close(seen[-1][:, -14:-1], normalized)
        for value in (0.0, -1.0, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.module.PolicyAdapter(self.contract, actor, self.start, action_scale=value)

    def test_scaled_cli_is_offline_logs_effective_step_and_rejects_invalid_scale(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "scaled.json"
            argv = [
                str(SCRIPT),
                "--policy-dir",
                str(POLICY),
                "--duration",
                ".1",
                "--action-scale",
                ".5",
                "--output",
                str(output),
            ]
            with (
                patch.object(self.module.sys, "argv", argv),
                patch.object(self.module, "execute") as execute,
                redirect_stdout(io.StringIO()),
            ):
                try:
                    self.module.main()
                except SystemExit as error:
                    self.fail(f"Scaled CLI is missing: {error}")
            execute.assert_not_called()
            d = json.loads(output.read_text())
            self.assertEqual(d["action_scale"], 0.5)
            self.assertEqual(d["effective_action_step_rad"], 0.0375)
            self.assertEqual(d["control_hz"], 20)
            self.assertFalse(d["hardware_policy_executed"])
            for value in ("0", "nan", "inf", "1.1"):
                with (
                    patch.object(self.module.sys, "argv", argv + ["--action-scale", value]),
                    patch.object(self.module, "execute") as execute,
                ):
                    with self.assertRaises(SystemExit):
                        self.module.main()
                execute.assert_not_called()

    def test_hardware_path_scales_targets_and_keeps_sdk_settings(self):
        joints = self.contract["joint_contract"]
        ids = IntEnum("JointId", {j["name"].upper(): j["sdk_id"] for j in joints})
        limits = {ids[j["name"].upper()]: (math.degrees(j["lower"]), math.degrees(j["upper"])) for j in joints}
        actual = {j["name"].upper(): math.degrees(float(self.start[0, i])) for i, j in enumerate(joints)}
        initial = actual.copy()
        now = [100.0]
        packets = []
        stopped = []
        closed = []

        def move(packet):
            packets.append(packet)
            actual.update({c.id.name: c.angle for c in packet})
            return True

        hand = SimpleNamespace(
            _comm=SimpleNamespace(_client=SimpleNamespace(_master=object())),
            _joint_limits=limits,
            open=lambda _: True,
            get_device_name=lambda: "GHand 5",
            get_firmware_version=lambda: "V3.0.6",
            get_hand_type=lambda: SimpleNamespace(value="right_hand"),
            is_connected=lambda: not closed,
            move_joints=move,
            stop=lambda: stopped.append(True) or True,
            close=lambda: closed.append(True),
        )
        sdk = SimpleNamespace(
            GHand=lambda *a: hand,
            ProductType=SimpleNamespace(GHand5=1),
            CommType=SimpleNamespace(ETHERCAT=1),
            JointCommand=SimpleNamespace,
        )

        def feedback(*_):
            now[0] += 0.001
            angles = actual.copy()
            if now[0] > 100.02:
                angles["FF_MCP_AA"] = 11.5
            return {
                "feedback_time": now[0],
                "wkc": 3,
                "hand_state": 0,
                "hand_error": 0,
                "temperature": 45,
                "angles_deg": angles,
                "joint_states": dict.fromkeys(actual, 0),
                "joint_errors": dict.fromkeys(actual, 0),
            }

        def sleep(dt):
            now[0] += dt

        result = {"duration_seconds": 0.1, "action_scale": 0.5, "hardware_policy_executed": False}
        with (
            patch.dict(self.module.sys.modules, {"ghand": sdk}),
            patch.object(self.module, "fresh_feedback", feedback),
            patch.object(self.module.time, "monotonic", lambda: now[0]),
            patch.object(self.module.time, "sleep", sleep),
            patch.object(self.module.signal, "signal"),
            patch.object(self.module.signal, "setitimer"),
            redirect_stdout(io.StringIO()),
        ):
            self.module.execute(self.contract, lambda obs: torch.ones(1, 13), result)
        self.assertTrue(result.get("policy_window_completed"), result)
        self.assertEqual(len(packets), 2)
        self.assertEqual(result["samples"][-1]["feedback"]["angles_deg"]["FF_MCP_AA"], 11.5)
        for packet in packets:
            for c in packet:
                self.assertTrue(limits[c.id][0] <= c.angle <= limits[c.id][1])
        for c in packets[0]:
            self.assertAlmostEqual(c.angle - initial[c.id.name], math.degrees(0.0375), places=4)
            self.assertEqual((c.speed, c.torque), (15, 20))
        self.assertEqual(stopped, [True])
        self.assertEqual(closed, [True])

    def test_joint_limits_and_invalid_policy_outputs(self):
        edge = self.module.PolicyAdapter(self.contract, lambda x: torch.ones(1, 13), self.high[None])
        target, _ = edge.propose(self.high[None], 0.0)
        torch.testing.assert_close(target, self.high[None])
        bad = self.module.PolicyAdapter(self.contract, lambda x: torch.full((1, 13), float("nan")), self.start)
        with self.assertRaises(ValueError):
            bad.propose(self.start, 0.0)

    def test_stop_is_preloaded_before_master_enters_operational_state(self):
        class Master:
            def __init__(self):
                self.slaves = [type("Slave", (), {"output": bytes(80)})()]

            def config_map(self):
                return 746

        master = Master()
        wrapped = self.module.StopPreloadMaster(master)
        self.assertEqual(wrapped.config_map(), 746)
        self.assertEqual(master.slaves[0].output, bytes([0, 1]) + bytes(78))

    def test_feedback_guard_allows_small_tracking_error_but_rejects_faults(self):
        sample = {
            "hand_state": 0,
            "hand_error": 0,
            "temperature": 40,
            "joint_states": {"A": 1},
            "joint_errors": {"A": 0},
            "angles_deg": {"A": 10.8},
        }
        self.module.validate_feedback(sample, {"A": (0.0, 50.0)})
        sample["hand_error"] = 1
        with self.assertRaises(RuntimeError):
            self.module.validate_feedback(sample, {"A": (0.0, 50.0)})
        sample["hand_error"] = 0
        sample["angles_deg"]["A"] = 54.0
        with self.assertRaisesRegex(RuntimeError, "54.*-3.*53"):
            self.module.validate_feedback(sample, {"A": (0.0, 50.0)})

    def test_normal_45_degree_feedback_is_not_a_firmware_temperature_fault(self):
        sample = {
            "hand_state": 0,
            "hand_error": 0,
            "temperature": 45,
            "joint_states": {"A": 1},
            "joint_errors": {"A": 0},
            "angles_deg": {"A": 10.0},
        }
        self.module.validate_feedback(sample, {"A": (0.0, 50.0)})

    def test_firmware_thermal_faults_stop_and_report_codes_even_below_45(self):
        # GHand SDK ErrorCode: driver overtemperature=6, motor overtemperature=8, high temperature=22.
        for code in (6, 8, 22):
            for scope in ("hand", "joint"):
                sample = {
                    "hand_state": 0,
                    "hand_error": code if scope == "hand" else 0,
                    "temperature": 40,
                    "joint_states": {"A": 1},
                    "joint_errors": {"A": code if scope == "joint" else 0},
                    "angles_deg": {"A": 10.0},
                }
                with self.subTest(code=code, scope=scope), self.assertRaises(RuntimeError) as error:
                    self.module.validate_feedback(sample, {"A": (0.0, 50.0)})
                self.assertIn("Device fault", str(error.exception))
                self.assertIn(str(code), str(error.exception))
                self.assertIn("temperature=40", str(error.exception))

    def test_nonfinite_temperature_and_protective_states_still_stop(self):
        sample = {
            "hand_state": 0,
            "hand_error": 0,
            "temperature": 40,
            "joint_states": {"A": 1},
            "joint_errors": {"A": 0},
            "angles_deg": {"A": 10.0},
        }
        for temperature in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(RuntimeError, "Temperature guard"):
                self.module.validate_feedback(dict(sample, temperature=temperature), {"A": (0.0, 50.0)})
        for changes in ({"hand_state": 2}, {"hand_state": 3}, {"joint_states": {"A": 2}}, {"joint_states": {"A": 3}}):
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, "Abnormal device state"):
                self.module.validate_feedback(dict(sample, **changes), {"A": (0.0, 50.0)})

    def test_deployment_does_not_treat_finite_angle_excursion_as_device_fault(self):
        sample = {
            "hand_state": 0,
            "hand_error": 0,
            "temperature": 40,
            "joint_states": {"A": 1},
            "joint_errors": {"A": 0},
            "angles_deg": {"A": 54.0},
        }
        self.module.validate_feedback(sample, {"A": (0.0, 50.0)}, check_angle_bounds=False)
        for angle in (float("nan"), float("inf"), -float("inf")):
            sample["angles_deg"]["A"] = angle
            with self.subTest(angle=angle), self.assertRaisesRegex(RuntimeError, "Invalid joint feedback"):
                self.module.validate_feedback(sample, {"A": (0.0, 50.0)}, check_angle_bounds=False)

    def test_hardware_guard_retains_rejected_feedback_and_closes(self):
        joints = self.contract["joint_contract"]
        joint_ids = IntEnum("JointId", {j["name"].upper(): j["sdk_id"] for j in joints})
        limits = {joint_ids[j["name"].upper()]: (math.degrees(j["lower"]), math.degrees(j["upper"])) for j in joints}
        sample = {
            "feedback_time": 1.0,
            "wkc": 3,
            "hand_state": 0,
            "hand_error": 0,
            "temperature": 42,
            "angles_deg": {j["name"].upper(): math.degrees(j["lower"]) for j in joints},
            "joint_states": {j.name: 0 for j in limits},
            "joint_errors": {j.name: 0 for j in limits},
        }
        sample["joint_errors"]["FF_MCP_AA"] = 5
        closed, stopped = [], []
        hand = SimpleNamespace(
            _comm=SimpleNamespace(_client=SimpleNamespace(_master=object())),
            _joint_limits=limits,
            open=lambda interface: True,
            get_device_name=lambda: "GHand 5",
            get_firmware_version=lambda: "V3.0.6",
            get_hand_type=lambda: SimpleNamespace(value="right_hand"),
            is_connected=lambda: not closed,
            stop=lambda: stopped.append(True) or True,
            close=lambda: closed.append(True),
        )
        sdk = SimpleNamespace(
            GHand=lambda *args: hand,
            ProductType=SimpleNamespace(GHand5=1),
            CommType=SimpleNamespace(ETHERCAT=1),
            JointCommand=object,
        )
        result = {"hardware_policy_executed": False, "duration_seconds": 30.0}
        with (
            patch.dict(self.module.sys.modules, {"ghand": sdk}),
            patch.object(self.module, "fresh_feedback", return_value=sample),
            patch.object(self.module.signal, "signal"),
            patch.object(self.module.signal, "alarm"),
            patch.object(self.module.signal, "setitimer") as timer,
        ):
            self.module.execute(self.contract, None, result)
        self.assertEqual(timer.call_args_list[0].args, (self.module.signal.ITIMER_REAL, 45.0))
        self.assertEqual(timer.call_args_list[-1].args, (self.module.signal.ITIMER_REAL, 0.0))
        self.assertEqual(result.get("last_feedback"), sample)
        self.assertIn("Device fault", result["error"])
        self.assertFalse(result["hardware_policy_executed"])
        self.assertTrue(result["connection_closed"])
        self.assertTrue(stopped)

    def test_policy_window_sends_forty_commands_and_never_retries_rejection(self):
        now = [100.0]
        rows, sent = [], []

        def sleep(dt):
            now[0] += dt

        def sample():
            return {
                "feedback_time": now[0],
                "angles_deg": {
                    j["name"].upper(): float(torch.rad2deg(self.start[0, i]))
                    for i, j in enumerate(self.contract["joint_contract"])
                },
            }

        def send(target):
            sent.append(target.clone())
            return target

        with (
            patch.object(self.module.time, "monotonic", lambda: now[0]),
            patch.object(self.module.time, "sleep", sleep),
        ):
            self.module.run_policy_window(self.adapter, sample, send, rows)
        self.assertEqual(len(sent), 40)
        self.assertTrue(all(r["accepted"] for r in rows))
        self.assertAlmostEqual(now[0], 102.0)
        self.assertLess(rows[-1]["t"], 2.0)

        adapter = self.module.PolicyAdapter(self.contract, lambda x: torch.ones(1, 13), self.start)

        def reject(target):
            raise RuntimeError("SDK rejected")

        with (
            patch.object(self.module.time, "monotonic", lambda: now[0]),
            patch.object(self.module.time, "sleep", sleep),
            self.assertRaises(RuntimeError),
        ):
            self.module.run_policy_window(adapter, sample, reject, [])
        torch.testing.assert_close(adapter.command, self.start)

    def test_missed_control_deadline_sends_nothing(self):
        now, sent = [100.0], []

        def delayed_sample():
            now[0] += 0.04
            return {
                "feedback_time": now[0],
                "angles_deg": {
                    j["name"].upper(): float(torch.rad2deg(self.start[0, i]))
                    for i, j in enumerate(self.contract["joint_contract"])
                },
            }

        with (
            patch.object(self.module.time, "monotonic", lambda: now[0]),
            patch.object(self.module.time, "sleep", lambda dt: None),
            self.assertRaisesRegex(RuntimeError, "deadline"),
        ):
            self.module.run_policy_window(self.adapter, delayed_sample, sent.append, [])
        self.assertEqual(sent, [])

    def test_nonfinite_or_reused_feedback_time_is_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.propose(self.start, float("nan"))
        self.adapter.propose(self.start, 1.0)
        with self.assertRaises(ValueError):
            self.adapter.propose(self.start, 1.0)

    def test_custom_windows_accept_decimals_stop_on_time_and_reject_invalid_durations(self):
        now = [100.0]

        def sleep(dt):
            now[0] += dt

        def sample():
            return {
                "feedback_time": now[0],
                "angles_deg": {
                    j["name"].upper(): float(torch.rad2deg(self.start[0, i]))
                    for i, j in enumerate(self.contract["joint_contract"])
                },
            }

        for duration in (5.0, 10.0, 12.37):
            sent, rows = [], []
            adapter = self.module.PolicyAdapter(self.contract, lambda x: torch.ones(1, 13), self.start)

            def send(target):
                sent.append(target.clone())
                return target

            start = now[0]
            with (
                patch.object(self.module.time, "monotonic", lambda: now[0]),
                patch.object(self.module.time, "sleep", sleep),
            ):
                self.module.run_policy_window(adapter, sample, send, rows, duration=duration)
            self.assertEqual(len(sent), math.ceil(duration / 0.05))
            self.assertAlmostEqual(now[0], start + duration)
            self.assertLess(rows[-1]["t"], duration)
        for duration in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.module.run_policy_window(self.adapter, sample, send, [], duration=duration)

    def test_duration_expiring_during_inference_does_not_send_late_target(self):
        now, sent = [100.0], []

        def sample():
            now[0] += 0.015
            return {
                "feedback_time": now[0],
                "angles_deg": {
                    j["name"].upper(): float(torch.rad2deg(self.start[0, i]))
                    for i, j in enumerate(self.contract["joint_contract"])
                },
            }

        with (
            patch.object(self.module.time, "monotonic", lambda: now[0]),
            patch.object(self.module.time, "sleep", lambda dt: None),
        ):
            self.module.run_policy_window(self.adapter, sample, sent.append, [], duration=0.01)
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
