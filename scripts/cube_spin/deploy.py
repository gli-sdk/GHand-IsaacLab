"""Supervised timed deployment of the full 13-joint spin actor.

Default is offline; --object-present records an operator-placed cube trial.
Physical power cutoff is required after the hardware trial:
software STOP has not been established as a reliable physical stop on this hand.
This is an experimental deployment, not a validated cube-spinning controller.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch
from ghand.tasks.cube_spin.student import JointHistory
from ghand.tasks.cube_spin.core import command_limits

ACTOR_SHA = "c613485833794e46f6024ac343ab934ef7b90908a8ebbeb40a72e77ef03f1a38"
KNOWN_ACTOR_SHAS = {ACTOR_SHA}  # Validated 35–40 mm / 30–50 g student.
SDK_ROOT = Path(os.environ.get("GHAND_SDK_ROOT", str(Path.home() / "ghand/GHand-Python-SDK"))).expanduser()
INTERFACE = os.environ.get("GHAND_ETHERCAT_INTERFACE", "enx00e04c665f71")
# Device versions admitted for these trials; protocol/mapping/feedback checks still apply.
ALLOWED_FIRMWARES = {
    "right": (
        "V3.0.4",
        "V3.0.6",
    ),
    "left": ("V3.0.4", "V3.0.6"),
}
FEEDBACK_TOLERANCE_DEG = 3.0  # Legacy diagnostic/reference tolerance; not used to stop policy/reset.


def sdk_percent(value):
    try:
        percent = int(value)
        if float(value) == percent and 1 <= percent <= 100:
            return percent
    except (ValueError, TypeError, OverflowError):
        pass
    raise ValueError("SDK speed/torque must be an integer percentage in [1, 100]")


def action_scale_fraction(value):
    scale = float(value)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("Action scale must be finite and in (0, 1]")
    return scale


def ring_pip_upper_degrees(value):
    upper = float(value)
    if not math.isfinite(upper) or not 80.0 <= upper <= 90.0:
        raise ValueError("RF_PIP target ceiling must be finite and between 80 and 90 degrees")
    return upper


class PolicyAdapter:
    def __init__(self, contract, actor, initial, action_scale=1.0, ring_pip_upper_deg=None):
        joints = contract["joint_contract"]
        if (
            len(joints) != 13
            or len({j["sdk_id"] for j in joints}) != 13
            or contract["history_steps"] != 30
            or contract["observation_size"] != 781
            or contract["step_dt"] != 0.05
            or contract["action_step_rad"] != 0.075
            or contract["target_speed"] != 1.0
        ):
            raise ValueError("Unexpected pilot policy contract")
        self.contract, self.actor = contract, actor
        self.action_scale = action_scale_fraction(action_scale)
        self.names = [j["name"].upper() for j in joints]
        self.lower = torch.tensor([j["lower"] for j in joints])
        self.upper = torch.tensor([j["upper"] for j in joints])
        # Target bounds are separate from the trained observation normalization.
        lower, upper = command_limits(
            joints, contract.get("command_upper_deg", {}), contract.get("command_margin_fraction", 0.0)
        )
        self.command_lower, self.command_upper = torch.tensor(lower), torch.tensor(upper)
        if ring_pip_upper_deg is not None:
            index = self.names.index("RF_PIP")
            ceiling = math.radians(ring_pip_upper_degrees(ring_pip_upper_deg))
            if not float(self.command_lower[index]) < ceiling <= float(self.upper[index]) + 1e-6:
                raise ValueError("RF_PIP target ceiling must stay within the policy contract")
            self.command_upper[index] = min(ceiling, float(self.command_upper[index]))
        self.history = JointHistory(1, 30, self.lower, self.upper)
        # No earlier position command exists at startup: use measured q once.
        self.command = initial.clone()
        self.history.reset(initial, self.command)
        self.last_time = None

    def propose(self, measured, timestamp):
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("Feedback time must increase")
        self.history.advance(measured, self.command, torch.zeros(1, dtype=torch.bool))
        with torch.inference_mode():
            action = self.actor(self.history.observation(self.contract["target_speed"]))
        if action.shape != (1, 13) or not bool(torch.isfinite(action).all()) or bool((action.abs() > 1.000001).any()):
            raise ValueError("Invalid actor output")
        self.last_time = timestamp
        target = (self.command + 0.075 * self.action_scale * action.clamp(-1.0, 1.0)).clamp(
            self.command_lower, self.command_upper
        )
        return target, action

    def acknowledge(self, target):
        if target.shape != (1, 13) or not bool(torch.isfinite(target).all()):
            raise ValueError("Invalid applied target")
        self.command = target.clone()


class StopPreloadMaster:
    """Set the initial RPDO to STOP before the SDK transitions into OP."""

    def __init__(self, master):
        object.__setattr__(self, "_master", master)

    def __getattr__(self, name):
        return getattr(self._master, name)

    def __setattr__(self, name, value):
        setattr(self._master, name, value)

    def config_map(self):
        result = self._master.config_map()
        if len(self._master.slaves) != 1 or len(self._master.slaves[0].output) != 80:
            raise RuntimeError("Unexpected EtherCAT topology or PDO size")
        self._master.slaves[0].output = bytes([0, 1]) + bytes(78)
        return result


def validate_feedback(sample, limits, *, check_angle_bounds=True):
    joint_errors = {name: code for name, code in sample["joint_errors"].items() if code}
    if sample["hand_error"] or joint_errors:
        raise RuntimeError(
            f"Device fault: hand_error={sample['hand_error']}; "
            f"joint_errors={joint_errors}; temperature={sample['temperature']} C"
        )
    if sample["hand_state"] not in (0, 1) or any(s not in (0, 1) for s in sample["joint_states"].values()):
        raise RuntimeError("Abnormal device state")
    # The SDK exposes thermal faults (6, 8, 22), not a documented Celsius cutoff.
    # Honor firmware faults/states above instead of an extra 45 C host cutoff.
    if not math.isfinite(sample["temperature"]):
        raise RuntimeError(f"Temperature guard: invalid reading {sample['temperature']}")
    for name, (lower, upper) in limits.items():
        angle = sample["angles_deg"].get(name, float("nan"))
        if not math.isfinite(angle):
            raise RuntimeError(f"Invalid joint feedback: {name}={angle}")
        # Retained for legacy diagnostic trajectories only. Policy/reset use firmware faults.
        low_guard, high_guard = lower - FEEDBACK_TOLERANCE_DEG, upper + FEEDBACK_TOLERANCE_DEG
        if check_angle_bounds and not low_guard <= angle <= high_guard:
            raise RuntimeError(
                f"Feedback outside SDK range: {name}={angle:.6g} deg; "
                f"allowed with feedback tolerance [{low_guard:.6g}, {high_guard:.6g}] deg"
            )


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
    start = time.monotonic()
    for step in range(steps):
        tick = start + step * 0.05
        time.sleep(max(0.0, tick - time.monotonic()))
        sample = read_sample()
        q = torch.deg2rad(torch.tensor([[sample["angles_deg"][n] for n in adapter.names]]))
        target, action = adapter.propose(q, sample["feedback_time"])
        now = time.monotonic()
        if now - tick > 0.025:
            raise RuntimeError("Control deadline missed; no further position commands")
        if now - start >= duration:
            break
        row = {
            "step": step,
            "t": now - start,
            "feedback": sample,
            "target_deg": torch.rad2deg(target[0]).tolist(),
            "action": action[0].tolist(),
            "accepted": False,
        }
        rows.append(row)
        row["send_start_time"] = time.monotonic()
        applied = send_target(target)
        row["send_end_time"] = time.monotonic()  # SDK buffer acceptance, not firmware arrival.
        row["accepted"] = True
        row["target_deg"] = torch.rad2deg(applied[0]).tolist()
        adapter.acknowledge(applied)
        if step % 10 == 0:
            print(f"POLICY {step + 1}/{steps}: all 13 joints", flush=True)
    time.sleep(max(0.0, start + duration - time.monotonic()))


def execute(contract, actor, result):
    speed = sdk_percent(result.get("speed_percent", 15))
    torque = sdk_percent(result.get("torque_percent", 20))
    hand_side = result.get("requested_hand_side", "right")
    if hand_side not in ("left", "right"):
        raise ValueError("hand side must be left or right")
    if contract.get("hand_side", hand_side) != hand_side:
        raise ValueError("Requested hand side differs from trained policy contract")
    duration = positive_seconds(result.get("duration_seconds", 2.0))
    action_scale = action_scale_fraction(result.get("action_scale", 1.0))
    ring_ceiling = result.get("ring_pip_upper_deg")
    if ring_ceiling is not None:
        ring_ceiling = ring_pip_upper_degrees(ring_ceiling)
    sys.path.insert(0, str(SDK_ROOT / "src"))
    from ghand import GHand, ProductType, CommType, JointCommand

    hand = GHand(ProductType.GHand5, CommType.ETHERCAT)
    hand._comm._client._master = StopPreloadMaster(hand._comm._client._master)
    rows = result["samples"] = []

    def interrupt(*_):
        raise RuntimeError("Interrupted or overall deadline reached")

    signal.signal(signal.SIGALRM, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, duration + 15.0)
    try:
        print(
            f"Connecting to the {'left' if hand_side == 'left' else 'right'} hand; initial output is STOP. No policy actions have been sent.",
            flush=True,
        )
        if not hand.open(INTERFACE):
            raise RuntimeError("SDK open failed")
        identity = {
            "name": hand.get_device_name(),
            "firmware": hand.get_firmware_version(),
            "hand_type": hand.get_hand_type().value,
        }
        result["identity"] = identity
        expected_identity = {"name": "GHand 5", "hand_type": hand_side + "_hand"}
        allowed_firmware = ALLOWED_FIRMWARES[hand_side]
        print("Device identity: " + json.dumps(identity, ensure_ascii=False), flush=True)
        if (
            any(identity[key] != value for key, value in expected_identity.items())
            or identity["firmware"] not in allowed_firmware
        ):
            raise RuntimeError(
                f"Unexpected device identity: expected {expected_identity}, "
                f"firmware in {allowed_firmware}, received {identity}"
            )
        joints = contract["joint_contract"]
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
            validate_feedback(sample, limits, check_angle_bounds=False)
            before = sample["feedback_time"]
            return sample

        sample = read_sample()
        initial = torch.deg2rad(torch.tensor([[sample["angles_deg"][j["name"].upper()] for j in joints]]))
        adapter = PolicyAdapter(contract, actor, initial, action_scale=action_scale, ring_pip_upper_deg=ring_ceiling)
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
            if not hand.move_joints(commands):
                raise RuntimeError("SDK rejected policy command")
            result["hardware_policy_executed"] = True
            return torch.deg2rad(torch.tensor([values]))

        condition = "cube present" if result.get("object_present_reported", False) else "empty hand"
        print(
            f"Starting policy: {condition}, all 13 joints, 20 Hz, speed {speed}% / torque {torque}%, action increment scale {action_scale:g}, duration {duration:g} s. "
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
        result["policy_steps_accepted"] = sum(r["accepted"] for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir", type=Path, default=Path(__file__).resolve().parents[2] / "models/cube_spin/student"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--hand-side",
        choices=("right", "left"),
        default="right",
        help="Required device handedness; semantic joint angles are unchanged (default: right)",
    )
    parser.add_argument(
        "--duration",
        type=positive_seconds,
        default=2.0,
        help="Finite positive policy duration in seconds; decimals allowed (default: 2)",
    )
    parser.add_argument(
        "--speed",
        type=sdk_percent,
        default=15,
        help="SDK speed percentage, not a calibrated joint velocity (default: 15)",
    )
    parser.add_argument(
        "--torque",
        type=sdk_percent,
        default=20,
        help="SDK torque percentage, not a calibrated joint torque (default: 20)",
    )
    parser.add_argument(
        "--action-scale",
        type=action_scale_fraction,
        default=1.0,
        help="Multiply each policy target increment by (0, 1]; 20 Hz loop unchanged (default: 1)",
    )
    parser.add_argument(
        "--ring-pip-upper-deg",
        type=ring_pip_upper_degrees,
        default=None,
        help="Optional RF_PIP command ceiling (80..90 degrees); normalization and feedback guard unchanged",
    )
    parser.add_argument(
        "--object-present",
        action="store_true",
        help="Operator reports cube placed and human hands withdrawn; does not add object sensing",
    )
    parser.add_argument(
        "--onsite-cutoff-ready",
        action="store_true",
        help="Operator present, hand mounted, human hands withdrawn, and will cut power at trial end",
    )
    args = parser.parse_args()
    if args.execute and not args.onsite_cutoff_ready:
        parser.error("Hardware trial requires an onsite operator committed to physical power cutoff")
    if args.output.exists():
        parser.error("Refusing to overwrite results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the result before any device access so an unwritable path cannot lose evidence.
    with args.output.open("x") as output:
        torch.set_num_threads(4)
        print(
            f"Policy directory: {args.policy_dir.resolve()}; RF_PIP target upper limit: "
            f"{args.ring_pip_upper_deg if args.ring_pip_upper_deg is not None else 'training/SDK limit'}",
            flush=True,
        )
        contract = json.loads((args.policy_dir / "contract.json").read_text())
        actor_path = args.policy_dir / "actor.ts"
        actor_sha = hashlib.sha256(actor_path.read_bytes()).hexdigest()
        if actor_sha not in KNOWN_ACTOR_SHAS:
            raise ValueError("Unexpected actor checkpoint")
        actor = torch.jit.load(str(actor_path), map_location="cpu").eval()
        with torch.inference_mode():
            for _ in range(4):
                actor(torch.zeros(1, 781))
        trial_mode = "full13_cube_policy_trial" if args.object_present else "full13_empty_hand_policy_trial"
        result = {
            "mode": trial_mode if args.execute else "offline_synthetic_replay",
            "requested_hand_side": args.hand_side,
            "object_present_reported": args.object_present,
            "object_pose_observed": False,
            "hardware_policy_executed": False,
            "hardware_ready": False,
            "cube_task_verified": False,
            "policy_dir": str(args.policy_dir.resolve()),
            "ring_pip_upper_deg": args.ring_pip_upper_deg,
            "trained_command_upper_deg": contract.get("command_upper_deg", {}),
            "trained_command_margin_fraction": contract.get("command_margin_fraction", 0.0),
            "actor_sha256": actor_sha,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "duration_seconds": args.duration,
            "control_hz": 20,
            "speed_percent": args.speed,
            "torque_percent": args.torque,
            "action_scale": args.action_scale,
            "effective_action_step_rad": 0.075 * args.action_scale,
            "extra_action_cap": None,
            "software_stop_verified": False,
            "feedback_tolerance_deg": None,
            "feedback_protection": "sdk_firmware_fault_and_state",
            "temperature_protection": "sdk_firmware_fault_and_state",
            "host_temperature_limit_c": None,
            "initial_command_history": "measured-angle surrogate once, then SDK-accepted position targets",
            "joint_order": [j["name"].upper() for j in contract["joint_contract"]],
        }
        if args.execute:
            execute(contract, actor, result)
        else:
            # Synthetic ideal tracking checks wiring only, not physical performance.
            q = torch.tensor([[j["lower"] for j in contract["joint_contract"]]])
            adapter = PolicyAdapter(
                contract, actor, q, action_scale=args.action_scale, ring_pip_upper_deg=args.ring_pip_upper_deg
            )
            samples = []
            for step in range(math.ceil(args.duration / 0.05)):
                target, action = adapter.propose(q, step * 0.05)
                adapter.acknowledge(target)
                q = target.clone()
                samples.append(
                    {"step": step, "target_deg": torch.rad2deg(target[0]).tolist(), "action": action[0].tolist()}
                )
            result.update(offline_steps=len(samples), samples=samples)
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2), flush=True)
    if args.execute and (not result.get("policy_window_completed") or result.get("error")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
