"""Right GHand5 EtherCAT Dual-ball rotation student runner. Offline unless --execute.

Position increments at 20 Hz; no camera, automatic reset or object-state input.
Firmware faults and fresh-feedback checks remain active. No extra host angle or
Celsius cutoff. Cyclic GC is paused only during control and restored on exit.
SDK STOP is requested on exit; physical stopping is not certified.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import os
import signal
import sys
import tempfile
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from ghand.tasks.dual_ball_rotation.deployment import PolicyAdapter, load_policy, validate_contract
from ghand.tasks.dual_ball_rotation.start_pose import load_start_pose

SDK_ROOT = Path(os.environ.get("GHAND_SDK_ROOT", str(Path.home() / "ghand/GHand-Python-SDK"))).expanduser()
INTERFACE = os.environ.get("GHAND_ETHERCAT_INTERFACE", "enx00e04c665f71")
ALLOWED_FIRMWARES = ("V3.0.4", "V3.0.6")


def sdk_percent(value):
    try:
        percent = int(value)
        if float(value) == percent and 1 <= percent <= 100:
            return percent
    except (ValueError, TypeError, OverflowError):
        pass
    raise ValueError("SDK speed/torque must be an integer percentage in [1, 100]")


class StopPreloadMaster:
    """Preload STOP and keep blocking PDO reception from starving Python inference."""

    def __init__(self, master):
        object.__setattr__(self, "_master", master)

    def __getattr__(self, name):
        return getattr(self._master, name)

    def __setattr__(self, name, value):
        setattr(self._master, name, value)

    def receive_processdata(self, timeout):
        # The SDK holds its data lock here. Release only the Python GIL while
        # pysoem waits (15 ms timeout), once startup has confirmed OP state.
        return self._master.receive_processdata(timeout, release_gil=self._master.in_op)

    def config_map(self):
        result = self._master.config_map()
        if len(self._master.slaves) != 1 or len(self._master.slaves[0].output) != 80:
            raise RuntimeError("Unexpected EtherCAT topology or PDO size")
        self._master.slaves[0].output = bytes([0, 1]) + bytes(78)
        return result


def validate_feedback(sample, limits):
    joint_errors = {name: code for name, code in sample["joint_errors"].items() if code}
    if sample["hand_error"] or joint_errors:
        raise RuntimeError(f"Device fault: hand_error={sample['hand_error']}; joint_errors={joint_errors}")
    if sample["hand_state"] not in (0, 1) or any(state not in (0, 1) for state in sample["joint_states"].values()):
        raise RuntimeError("Abnormal device state")
    if not math.isfinite(sample["temperature"]):
        raise RuntimeError("Invalid temperature feedback")
    for name in limits:
        if not math.isfinite(sample["angles_deg"].get(name, float("nan"))):
            raise RuntimeError("Invalid joint feedback: " + name)


def fresh_feedback(hand, after):
    from ghand.comm.ethercat_protocol import Tpdo

    client = hand._comm._client
    deadline = time.monotonic() + 0.075
    while time.monotonic() < deadline:
        with client._data_lock:
            stamp, raw = client._last_valid_process_time, client._last_valid_tpdo_snapshot
            wkc, expected = client._actual_wkc, client._expected_wkc
        if stamp is not None and stamp > after and raw is not None:
            if wkc < max(1, expected) or not 0.0 <= time.monotonic() - stamp <= 0.05:
                raise RuntimeError("Invalid or stale feedback")
            tpdo = Tpdo.from_bytes(hand._comm._validate_tpdo_data(raw), hand._comm.config)
            return {
                "feedback_time": stamp,
                "wkc": wkc,
                "hand_state": tpdo.hand.state,
                "hand_error": tpdo.hand.error,
                "temperature": tpdo.hand.temperature,
                "angles_deg": {j.name: v.angle for j, v in tpdo.joints.items()},
                # SDK raw fields: no unverified conversion to deg/s or Nm.
                "joint_speed_sdk_raw": {j.name: v.speed for j, v in tpdo.joints.items()},
                "joint_torque_sdk_raw": {j.name: v.torque for j, v in tpdo.joints.items()},
                "joint_states": {j.name: v.state for j, v in tpdo.joints.items()},
                "joint_errors": {j.name: v.error for j, v in tpdo.joints.items()},
            }
        time.sleep(0.001)
    raise RuntimeError("No fresh feedback within 75 ms")


def positive_seconds(value):
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("Duration must be a finite positive number of seconds")
    return duration


def run_policy_window(adapter, read_sample, send_target, rows, duration=2.0):
    duration = positive_seconds(duration)
    steps = math.ceil(duration / 0.05)
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        # Collect before the schedule starts. Full cyclic collections can take
        # tens of milliseconds and otherwise interrupt any control-loop step.
        # Reference counting still releases ordinary temporary objects below.
        gc.collect()
        gc.disable()
    try:
        start = time.monotonic()
        for step in range(steps):
            tick = start + step * 0.05
            time.sleep(max(0.0, tick - time.monotonic()))
            woke = time.monotonic()
            sample = read_sample()
            read_done = time.monotonic()
            q = torch.deg2rad(torch.tensor([[sample["angles_deg"][n] for n in adapter.names]]))
            target, action = adapter.propose(q, sample["feedback_time"])
            now = time.monotonic()
            timing = {
                "wake_lateness": (woke - tick) * 1000.0,
                "feedback_read": (read_done - woke) * 1000.0,
                "propose": (now - read_done) * 1000.0,
                "total_lateness": (now - tick) * 1000.0,
            }
            row = {
                "step": step,
                "t": now - start,
                "feedback": sample,
                "target_deg": torch.rad2deg(target[0]).tolist(),
                "action": action[0].tolist(),
                "accepted": False,
                "timing_ms": timing,
                "deadline_missed": now - tick > 0.025,
            }
            rows.append(row)
            if row["deadline_missed"]:
                raise RuntimeError(
                    "Control deadline missed; no further position commands; "
                    + ", ".join(f"{key}={value:.2f} ms" for key, value in timing.items())
                )
            if now - start >= duration:
                break
            row["send_start_time"] = time.monotonic()
            applied = send_target(target)
            row["send_end_time"] = time.monotonic()  # SDK buffer acceptance, not firmware arrival.
            row["accepted"] = True
            row["target_deg"] = torch.rad2deg(applied[0]).tolist()
            adapter.acknowledge(applied)
            if step % 10 == 0:
                print(f"POLICY {step + 1}/{steps}: all 13 joints", flush=True)
        time.sleep(max(0.0, start + duration - time.monotonic()))
    finally:
        if gc_was_enabled:
            gc.enable()


def execute(contract, actor, result, adapter_factory=None):
    speed = sdk_percent(result.get("speed_percent", 15))
    torque = sdk_percent(result.get("torque_percent", 20))
    hand_side = "right"
    duration = positive_seconds(result.get("duration_seconds", 5.0))
    joints = validate_contract(contract)
    sys.path.insert(0, str(Path(result.get("sdk_root", SDK_ROOT)) / "src"))
    from ghand import GHand, ProductType, CommType, JointCommand, CtrlMode

    hand = GHand(ProductType.GHand5, CommType.ETHERCAT)
    hand._comm._client._master = StopPreloadMaster(hand._comm._client._master)
    rows = result["samples"] = []

    def interrupt(*_):
        raise RuntimeError("Interrupted or overall deadline reached")

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.signal(signal.SIGALRM, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, duration + 15.0)
    try:
        print("Connecting to the right hand; initial output is STOP. No policy actions have been sent.", flush=True)
        if not hand.open(result.get("interface", INTERFACE)):
            raise RuntimeError("SDK open failed")
        identity = {
            "name": hand.get_device_name(),
            "firmware": hand.get_firmware_version(),
            "hand_type": hand.get_hand_type().value,
        }
        result["identity"] = identity
        expected_identity = {"name": "GHand 5", "hand_type": hand_side + "_hand"}
        allowed_firmware = ALLOWED_FIRMWARES
        print("Device identity: " + json.dumps(identity, ensure_ascii=False), flush=True)
        if (
            any(identity[key] != value for key, value in expected_identity.items())
            or identity["firmware"] not in allowed_firmware
        ):
            raise RuntimeError(
                f"Unexpected device identity: expected {expected_identity}, "
                f"firmware in {allowed_firmware}, received {identity}"
            )
        ids = {j.name: j for j in hand._joint_limits}
        limits = {j.name: (float(lo), float(hi)) for j, (lo, hi) in hand._joint_limits.items()}
        if set(ids) != {j["name"].upper() for j in joints}:
            raise RuntimeError("SDK active joint set differs from policy")
        for joint in joints:
            name = joint["name"].upper()
            lo, hi = limits[name]
            if (
                ids[name].value != joint["sdk_id"]
                or math.degrees(joint["lower"]) < lo - 1e-5
                or math.degrees(joint["upper"]) > hi + 1e-5
            ):
                raise RuntimeError("SDK joint mapping/range differs: " + name)
        before = time.monotonic()

        def read_sample():
            nonlocal before
            sample = fresh_feedback(hand, before)
            result["last_feedback"] = sample
            validate_feedback(sample, limits)
            before = sample["feedback_time"]
            return sample

        sample = read_sample()
        initial = torch.deg2rad(torch.tensor([[sample["angles_deg"][j["name"].upper()] for j in joints]]))
        adapter = (adapter_factory or PolicyAdapter)(contract, actor, initial)
        result["initial_deg"] = sample["angles_deg"]
        result["initial_outside_training_limits"] = [
            j["name"]
            for i, j in enumerate(joints)
            if initial[0, i] < adapter.lower[i] - 1e-5 or initial[0, i] > adapter.upper[i] + 1e-5
        ]

        def send_target(target):
            values = torch.rad2deg(target[0]).tolist()
            # Remove floating-point overshoot at exact SDK endpoints.
            values = [max(limits[n][0], min(limits[n][1], v)) for n, v in zip(adapter.names, values)]
            commands = [
                JointCommand(id=ids[n], angle=v, speed=speed, torque=torque) for n, v in zip(adapter.names, values)
            ]
            if not hand.move_joints(commands, mode=CtrlMode.POSITION):
                raise RuntimeError("SDK rejected policy command")
            result["hardware_motion_executed"] = True
            result["hardware_policy_executed"] = adapter_factory is None
            return torch.deg2rad(torch.tensor([values]))

        condition = "two balls present" if result.get("object_present_reported", False) else "empty hand"
        label = "Starting fixed motion" if adapter_factory else "Starting policy"
        print(
            f"{label}: {condition}, all 13 joints, 20 Hz, speed {speed}% / torque {torque}%, duration {duration:g} s. "
            "Manually disconnect power when the run ends or if an abnormality occurs.",
            flush=True,
        )
        run_policy_window(adapter, read_sample, send_target, rows, duration=duration)
        result["policy_window_completed"] = True
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        print("Policy aborted: " + result["error"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        print(
            "Disconnect power now. Stopping position commands, sending SDK STOP, and closing the connection.",
            flush=True,
        )
        try:
            if hand.is_connected():
                result["stop_accepted"] = bool(hand.stop())
        except BaseException as error:
            result["stop_error"] = str(error)
        finally:
            try:
                hand.close()
            except BaseException as error:
                result["close_error"] = str(error)
            result["connection_closed"] = not hand.is_connected()
        if (
            result.get("stop_accepted") is False
            or result.get("stop_error")
            or result.get("close_error")
            or not result["connection_closed"]
        ):
            result.setdefault("error", "SDK STOP/close did not complete; physical power cutoff required")
        result["policy_steps_accepted"] = sum(r["accepted"] for r in rows)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "models/dual_ball_rotation/student")
    parser.add_argument("--execute", action="store_true", help="Connect to the right hand and send policy commands")
    parser.add_argument("--duration", type=positive_seconds, default=5.0)
    parser.add_argument("--speed", type=sdk_percent, default=15, help="SDK percentage, not calibrated deg/s")
    parser.add_argument("--torque", type=sdk_percent, default=20, help="SDK percentage, not calibrated Nm")
    parser.add_argument("--interface", default=INTERFACE)
    parser.add_argument("--sdk-root", type=Path, default=SDK_ROOT)
    parser.add_argument(
        "--object-present", action="store_true", help="Operator reports BOTH balls placed; not object sensing"
    )
    parser.add_argument(
        "--onsite-cutoff-ready",
        action="store_true",
        help="Operator present, hand fixed, human hands clear, physical power cutoff ready",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON result; default overwrites runs/dual_ball_rotation/hardware/latest.json (offline: offline_latest.json)",
    )
    args = parser.parse_args(argv)
    if args.execute and not args.onsite_cutoff_ready:
        parser.error("--execute requires --onsite-cutoff-ready")
    out = args.output or ROOT / "runs/dual_ball_rotation/hardware" / (
        "latest.json" if args.execute else "offline_latest.json"
    )
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Verify output is writable before opening any device; replace the previous log only after finishing.
    reserved = tempfile.NamedTemporaryFile(
        mode="w", dir=out.parent, prefix="." + out.name + ".", suffix=".tmp", delete=False
    )
    result = {
        "mode": "hardware_dual_ball_rotation_trial" if args.execute else "offline_synthetic_replay",
        "requested_hand_side": "right",
        "policy_dir": str(args.policy_dir.resolve()),
        "interface": args.interface,
        "sdk_root": str(args.sdk_root.resolve()),
        "hardware_policy_executed": False,
        "hardware_ready": False,
        "rotation_task_verified": False,
        "object_present_reported": args.object_present,
        "object_pose_observed": False,
        "duration_seconds": args.duration,
        "control_hz": 20,
        "speed_percent": args.speed,
        "torque_percent": args.torque,
        "observation_size": 780,
        "history_steps": 30,
        "action_step_rad": 0.075,
        "software_stop_verified": False,
        "feedback_protection": "sdk_firmware_fault_and_state",
        "host_temperature_limit_c": None,
        "feedback_tolerance_deg": None,
        "initial_command_history": "measured-angle surrogate once, then SDK-buffer-accepted targets",
        "accepted_command_note": "SDK buffer acceptance is not proof of firmware arrival or physical motion",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(
            (ROOT / "source/ghand/tasks/dual_ball_rotation/deployment.py").read_bytes()
        ).hexdigest(),
    }
    try:
        # Batch-one CPU inference: avoid worker-thread scheduling spikes.
        torch.set_num_threads(1)
        result["inference_threads"] = torch.get_num_threads()
        contract, actor, hashes = load_policy(args.policy_dir)
        result.update(hashes)
        trained_initial, trained_setup = load_start_pose(args.policy_dir, contract)
        result["trained_setup"] = trained_setup
        result["rotation_direction"] = contract.get("rotation_direction", 1)
        result["trained_domain_randomization"] = contract.get("domain_randomization", False)
        result["joint_order"] = [n.upper() for n in contract["joint_contract"]["names"]]
        print(
            f"Dual-ball rotation right-hand student: {args.policy_dir.resolve()}; 30-frame joint history, 20 Hz position increments.",
            flush=True,
        )
        if args.execute:
            print(
                "Start from the selected model's saved pose; this script does not reset the hand automatically. Use scripts/dual_ball_rotation/prepare_hand.py to prepare the empty hand first.",
                flush=True,
            )
            print(
                "Training palm forward-tilt range: "
                + str(trained_setup["palm_tilt_range_deg"])
                + " degrees; ball diameter "
                + str(trained_setup["ball_diameter_mm"])
                + " mm.",
                flush=True,
            )
            if trained_setup["palm_side_tilt_deg"]:
                print(
                    "Training palm side tilt: "
                    + str(trained_setup["palm_side_tilt_deg"])
                    + " degrees (negative values tilt the right hand toward the thumb); adjust the mount manually.",
                    flush=True,
                )
            execute(contract, actor, result)
        else:
            q = trained_initial.clone()
            result["offline_initial_deg"] = torch.rad2deg(q[0]).tolist()
            adapter = PolicyAdapter(contract, actor, q)
            samples = []
            for step in range(math.ceil(args.duration / 0.05)):
                target, action = adapter.propose(q, step * 0.05)
                adapter.acknowledge(target)
                q = target.clone()
                samples.append(
                    {"step": step, "target_deg": torch.rad2deg(target[0]).tolist(), "action": action[0].tolist()}
                )
            result.update(
                offline_steps=len(samples),
                samples=samples,
                offline_note="Synthetic ideal tracking verifies software wiring only; no SDK connection or physical validation",
            )
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    finally:
        with reserved:
            json.dump(json_safe(result), reserved, indent=2, allow_nan=False)
            reserved.write("\n")
        os.replace(reserved.name, out)
    summary = {
        k: result[k]
        for k in (
            "mode",
            "hardware_policy_executed",
            "policy_window_completed",
            "policy_steps_accepted",
            "offline_steps",
            "error",
        )
        if k in result
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Results file: " + str(out), flush=True)
    if result.get("error") or (args.execute and not result.get("policy_window_completed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
