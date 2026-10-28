import importlib.util
import gc
import io
import json
import math
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from enum import IntEnum
from contextlib import redirect_stdout, redirect_stderr
import unittest
from unittest.mock import patch, Mock
import torch

ROOT = Path(__file__).resolve().parents[2]


class CliTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "scripts/dual_ball_rotation/deploy.py"
        self.assertTrue(path.exists(), "Dual-ball rotation deployment CLI missing")
        spec = importlib.util.spec_from_file_location("deploy_dual_ball_test", path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.c = json.loads((ROOT / "models/dual_ball_rotation/student/contract.json").read_text())
        self.names = [n.upper() for n in self.c["joint_contract"]["names"]]

    def test_offline_default_never_opens_sdk_and_overwrites_one_log(self):
        with TemporaryDirectory() as d:
            out = Path(d) / "latest.json"
            for _ in range(2):
                with (
                    patch.object(self.m, "execute", side_effect=AssertionError("No device access")),
                    redirect_stdout(io.StringIO()),
                ):
                    self.m.main(["--duration", ".2", "--output", str(out)])
            result = json.loads(out.read_text())
            self.assertEqual(result["offline_steps"], 4)
            self.assertFalse(result["hardware_policy_executed"])
            self.assertEqual(result["observation_size"], 780)
            self.assertEqual(len(list(Path(d).iterdir())), 1)

    def test_cli_uses_single_cpu_thread_before_loading_policy(self):
        with TemporaryDirectory() as d:
            observed = []

            def load(_):
                observed.append(torch.get_num_threads())
                raise ValueError("test stops before any SDK access")

            with (
                patch.object(self.m, "load_policy", side_effect=load),
                redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.m.main(["--output", str(Path(d) / "offline.json")])
            self.assertEqual(observed, [1])

    def test_execute_requires_operator_flag(self):
        with patch.object(self.m, "execute") as execute, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.m.main(["--execute"])
        execute.assert_not_called()

    def test_feedback_does_not_reintroduce_angle_or_temperature_cutoffs(self):
        sample = dict(
            hand_error=0,
            hand_state=0,
            temperature=60.0,
            joint_errors={"LF_PIP": 0},
            joint_states={"LF_PIP": 0},
            angles_deg={"LF_PIP": 100.0},
        )
        self.m.validate_feedback(sample, {"LF_PIP": (0.0, 74.0)})
        for bad in [
            dict(sample, hand_error=41),
            dict(sample, joint_errors={"LF_PIP": 5}),
            dict(sample, angles_deg={"LF_PIP": float("nan")}),
            dict(sample, temperature=float("nan")),
        ]:
            with self.assertRaises(RuntimeError):
                self.m.validate_feedback(bad, {"LF_PIP": (0.0, 74.0)})

    def test_parameters_reject_nonfinite_and_out_of_range(self):
        for x in ["0", "-1", "nan", "inf"]:
            with self.assertRaises(ValueError):
                self.m.positive_seconds(x)
        for x in ["0", "101", "15.5", "nan"]:
            with self.assertRaises(ValueError):
                self.m.sdk_percent(x)
        self.assertEqual(self.m.positive_seconds("10.5"), 10.5)

    def fake_run(self, side="right_hand", fault=0, reject=False, stop_ok=True, firmware="V3.0.6", factory=None):
        ids = IntEnum("JointId", dict(zip(self.names, [17, 16, 14, 13, 11, 10, 8, 7, 6, 4, 3, 2, 1])))
        config = json.loads((ROOT / "config/dual_ball_rotation/hand.json").read_text())
        limits = {ids[j["id"]]: (j["min"], j["max"]) for j in config["joints"] if j["id"] in ids.__members__}
        actual = {n: math.degrees(v) for n, v in zip(self.names, self.c["joint_contract"]["lower"])}
        now = [100.0]
        packets = []
        stops = []
        closed = []

        def send(commands, mode=None):
            packets.append(commands)
            if reject:
                return False
            actual.update({c.id.name: c.angle for c in commands})
            return True

        hand = SimpleNamespace(
            _comm=SimpleNamespace(_client=SimpleNamespace(_master=object())),
            _joint_limits=limits,
            open=lambda _: True,
            get_device_name=lambda: "GHand 5",
            get_firmware_version=lambda: firmware,
            get_hand_type=lambda: SimpleNamespace(value=side),
            is_connected=lambda: not closed,
            move_joints=send,
            stop=lambda: stops.append(True) or stop_ok,
            close=lambda: closed.append(True),
        )
        sdk = SimpleNamespace(
            GHand=lambda *a: hand,
            ProductType=SimpleNamespace(GHand5=1),
            CommType=SimpleNamespace(ETHERCAT=1),
            JointCommand=SimpleNamespace,
            CtrlMode=SimpleNamespace(POSITION=1),
        )

        def read(*args):
            now[0] += 0.001
            return dict(
                feedback_time=now[0],
                wkc=3,
                hand_state=0,
                hand_error=fault,
                temperature=50.0,
                angles_deg=actual.copy(),
                joint_states=dict.fromkeys(actual, 0),
                joint_errors=dict.fromkeys(actual, 0),
            )

        def sleep(dt):
            now[0] += dt

        result = dict(
            duration_seconds=0.1,
            speed_percent=15,
            torque_percent=20,
            object_present_reported=True,
            interface="fake",
            sdk_root=str(ROOT / "does_not_open"),
            hardware_policy_executed=False,
        )
        with (
            patch.dict(self.m.sys.modules, {"ghand": sdk}),
            patch.object(self.m, "fresh_feedback", side_effect=read),
            patch.object(self.m, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)),
            patch.object(self.m.signal, "signal"),
            patch.object(self.m.signal, "setitimer"),
            redirect_stdout(io.StringIO()),
        ):
            self.m.execute(self.c, lambda obs: torch.zeros(1, 13), result, adapter_factory=factory)
        return result, packets, stops, closed

    def test_mock_transport_sends_all13_and_cleans_up(self):
        r, p, s, c = self.fake_run()
        self.assertTrue(r["policy_window_completed"])
        self.assertEqual(r["policy_steps_accepted"], 2)
        self.assertEqual(len(p[0]), 13)
        self.assertTrue(s and c)
        self.assertTrue(all(x.speed == 15 and x.torque == 20 for x in p[0]))

    def test_both_firmwares_allow_policy_and_prepare_pose(self):
        from ghand.tasks.dual_ball_rotation.start_pose import PoseAdapter

        def prepare(c, actor, initial):
            return PoseAdapter(c, initial, initial, ramp_seconds=0.05, hold_seconds=0.05)

        for firmware in ("V3.0.4", "V3.0.6"):
            for factory in (None, prepare):
                with self.subTest(firmware=firmware, mode="prepare" if factory else "policy"):
                    r, p, s, c = self.fake_run(firmware=firmware, factory=factory)
                    self.assertNotIn("error", r)
                    self.assertTrue(r["policy_window_completed"])
                    self.assertEqual(r["policy_steps_accepted"], 2)
                    self.assertTrue(all(len(packet) == 13 for packet in p))
                    self.assertEqual(r["hardware_policy_executed"], factory is None)
                    self.assertTrue(s and c)

    def test_wrong_hand_and_firmware_fault_never_send_positions(self):
        for args in [
            {"side": "left_hand"},
            {"side": "left_hand", "firmware": "V3.0.4"},
            {"fault": 41},
            {"firmware": "V3.0.5"},
            {"firmware": "unknown"},
        ]:
            r, p, s, c = self.fake_run(**args)
            self.assertIn("error", r)
            self.assertFalse(p)
            self.assertTrue(c)

    def test_rejected_command_is_not_marked_accepted(self):
        r, p, s, c = self.fake_run(reject=True)
        self.assertEqual(r["policy_steps_accepted"], 0)
        self.assertFalse(r["samples"][0]["accepted"])
        self.assertIn("error", r)
        self.assertTrue(s and c)

    def test_deadline_miss_sends_no_position_commands(self):
        now = [100.0]

        def read():
            now[0] += 0.03
            return dict(angles_deg={n: 0.0 for n in self.names}, feedback_time=now[0])

        adapter = SimpleNamespace(names=self.names, propose=lambda *a: (torch.zeros(1, 13), torch.zeros(1, 13)))
        send = Mock()
        rows = []
        with (
            patch.object(self.m, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _: None)),
            self.assertRaises(RuntimeError),
        ):
            self.m.run_policy_window(adapter, read, send, rows, duration=0.1)
        send.assert_not_called()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["accepted"])
        self.assertTrue(rows[0]["deadline_missed"])
        self.assertAlmostEqual(rows[0]["timing_ms"]["feedback_read"], 30.0)
        self.assertAlmostEqual(rows[0]["timing_ms"]["propose"], 0.0)
        self.assertAlmostEqual(rows[0]["timing_ms"]["wake_lateness"], 0.0)

    def test_control_window_pauses_cyclic_gc_and_restores_entry_state(self):
        entry_state = gc.isenabled()
        try:
            for initially_enabled in (True, False):
                (gc.enable if initially_enabled else gc.disable)()
                now = [100.0]
                sent = []

                def read():
                    self.assertFalse(gc.isenabled(), "Automatic GC must not pause position control")
                    return dict(angles_deg={n: 0.0 for n in self.names}, feedback_time=now[0])

                def send(target):
                    self.assertFalse(gc.isenabled())
                    sent.append(target)
                    return target

                adapter = SimpleNamespace(
                    names=self.names,
                    propose=lambda *a: (torch.zeros(1, 13), torch.zeros(1, 13)),
                    acknowledge=lambda _: None,
                )

                def sleep(dt):
                    now[0] += dt

                with (
                    patch.object(self.m, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)),
                    redirect_stdout(io.StringIO()),
                ):
                    self.m.run_policy_window(adapter, read, send, [], duration=0.1)
                self.assertEqual(len(sent), 2)
                self.assertEqual(gc.isenabled(), initially_enabled)
        finally:
            (gc.enable if entry_state else gc.disable)()

    def test_control_window_restores_gc_when_feedback_raises(self):
        entry_state = gc.isenabled()
        gc.enable()

        def read():
            self.assertFalse(gc.isenabled(), "Automatic GC must already be paused")
            raise RuntimeError("feedback failed")

        try:
            adapter = SimpleNamespace(names=self.names)
            with self.assertRaisesRegex(RuntimeError, "feedback failed"):
                self.m.run_policy_window(adapter, read, Mock(), [], duration=0.1)
            self.assertTrue(gc.isenabled())
        finally:
            (gc.enable if entry_state else gc.disable)()

    def test_failed_stop_is_not_reported_as_clean_completion(self):
        r, p, s, c = self.fake_run(stop_ok=False)
        self.assertTrue(r["policy_window_completed"])
        self.assertIn("error", r)
        self.assertFalse(r["stop_accepted"])

    def test_startup_pdo_preloads_stop_and_rejects_other_layout(self):
        slave = SimpleNamespace(output=bytes(80))
        master = SimpleNamespace(slaves=[slave], config_map=lambda: 7)
        self.assertEqual(self.m.StopPreloadMaster(master).config_map(), 7)
        self.assertEqual(slave.output, bytes([0, 1]) + bytes(78))
        slave.output = bytes(40)
        with self.assertRaises(RuntimeError):
            self.m.StopPreloadMaster(master).config_map()

    def test_pdo_receive_releases_gil_and_preserves_timeout_and_wkc(self):
        for wkc in (0, 3):
            for in_op in (False, True):
                with self.subTest(wkc=wkc, in_op=in_op):
                    receive = Mock(return_value=wkc)
                    master = self.m.StopPreloadMaster(SimpleNamespace(receive_processdata=receive, in_op=in_op))
                    self.assertEqual(master.receive_processdata(15_000), wkc)
                    receive.assert_called_once_with(15_000, release_gil=in_op)

    def test_fresh_feedback_rejects_bad_wkc_old_or_missing_snapshot(self):
        for stamp, wkc in [(100.0, 0), (99.0, 3), (None, 3)]:
            now = [100.0]

            def sleep(dt):
                now[0] += dt

            client = SimpleNamespace(
                _data_lock=threading.Lock(),
                _last_valid_process_time=stamp,
                _last_valid_tpdo_snapshot=b"not-decoded",
                _actual_wkc=wkc,
                _expected_wkc=3,
            )
            hand = SimpleNamespace(_comm=SimpleNamespace(_client=client))
            decoder = Mock(side_effect=AssertionError("Invalid snapshot must not be decoded"))
            protocol = SimpleNamespace(Tpdo=SimpleNamespace(from_bytes=decoder))
            with (
                patch.dict(self.m.sys.modules, {"ghand.comm.ethercat_protocol": protocol}),
                patch.object(self.m, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)),
                self.assertRaises(RuntimeError),
            ):
                self.m.fresh_feedback(hand, 98.0)
            decoder.assert_not_called()

    def test_invalid_sensor_values_can_be_saved_as_strict_json(self):
        value = {"temperature": float("nan"), "nested": [float("inf"), 1.0]}
        encoded = json.dumps(self.m.json_safe(value), allow_nan=False)
        self.assertEqual(json.loads(encoded)["temperature"], "nan")


if __name__ == "__main__":
    unittest.main()
