"""Restore the open pose: recorded right-hand pose or SDK-neutral left-hand pose.

Empty hand only. Three-second ramp, then wait for measured arrival and stability
for up to five more seconds before STOP/close. Timeout is a failed reset.
Default is offline. Disconnect drive power before placing an object by hand.
The output JSON is overwritten each run; default: runs/reset_latest.json.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cube_spin.deploy import (
    ALLOWED_FIRMWARES,
    FEEDBACK_TOLERANCE_DEG,
    INTERFACE,
    SDK_ROOT,
    StopPreloadMaster,
    fresh_feedback,
    validate_feedback,
)

from ghand.tasks.cube_spin.core import sdk_joint_contract

PROJECT = Path(__file__).resolve().parents[2]
REFERENCE = PROJECT / "config/cube_spin/reset_pose.json"
DEFAULT_OUTPUT = PROJECT / "runs/reset_latest.json"
RAMP_SECONDS = 3.0
SETTLE_TIMEOUT_SECONDS = 5.0
SETTLE_SECONDS = 1.0
POSITION_TOLERANCE_DEG = 1.0
STABILITY_TOLERANCE_DEG = 0.5


def reference_targets(angles, limits, *, check_angle_bounds=True):
    if len(limits) != 13 or not set(limits) <= set(angles):
        raise ValueError("All thirteen active angles are required")
    result = {}
    for name, (lower, upper) in limits.items():
        angle = angles[name]
        if not math.isfinite(angle) or (
            check_angle_bounds and not lower - FEEDBACK_TOLERANCE_DEG <= angle <= upper + FEEDBACK_TOLERANCE_DEG
        ):
            raise ValueError("Reference outside SDK limits: " + name)
        result[name] = max(lower, min(upper, angle))
    return result


def ramp_target(initial, target, elapsed):
    if not math.isfinite(elapsed) or elapsed < 0.0 or set(initial) != set(target):
        raise ValueError("Invalid reset interpolation")
    fraction = min(elapsed / RAMP_SECONDS, 1.0)
    return {name: initial[name] + fraction * (target[name] - initial[name]) for name in initial}


def returned_to_reference(samples, target, minimum_seconds=0.5):
    if len(samples) < 2:
        return False
    stamps = [row["feedback_time"] for row in samples]
    if (
        not all(math.isfinite(t) for t in stamps)
        or any(not 0.0 < b - a <= 0.075 for a, b in zip(stamps, stamps[1:]))
        or stamps[-1] - stamps[0] < minimum_seconds - 1e-9
    ):
        return False
    for name, goal in target.items():
        values = [row["angles_deg"][name] for row in samples]
        if (
            not all(math.isfinite(q) and abs(q - goal) <= POSITION_TOLERANCE_DEG for q in values)
            or max(values) - min(values) > STABILITY_TOLERANCE_DEG
        ):
            return False
    return True


def pose_error_summary(sample, target):
    errors = {name: sample["angles_deg"][name] - goal for name, goal in target.items()}
    return ", ".join(
        f"{name}={error:+.2f} deg"
        for name, error in sorted(errors.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    )


def execute(result, reference):
    hand_side = result.get("requested_hand_side", "right")
    if hand_side not in ("left", "right"):
        raise ValueError("hand side must be left or right")
    sys.path.insert(0, str(SDK_ROOT / "src"))
    from ghand import GHand, ProductType, CommType, JointCommand

    hand = GHand(ProductType.GHand5, CommType.ETHERCAT)
    hand._comm._client._master = StopPreloadMaster(hand._comm._client._master)
    rows, stopped_samples = [], []
    result.update(samples=rows, stop_samples=stopped_samples, motion_commands=0, settled_before_stop=False)
    before = time.monotonic()
    target = None

    def interrupt(*_):
        raise RuntimeError("Reset interrupted or deadline reached")

    signal.signal(signal.SIGALRM, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    signal.alarm(15)
    try:
        if not hand.open(INTERFACE):
            raise RuntimeError("SDK open failed")
        identity = {
            "name": hand.get_device_name(),
            "hand_type": hand.get_hand_type().value,
            "firmware": hand.get_firmware_version(),
        }
        result["identity"] = identity
        print("Reset device: " + json.dumps(identity, ensure_ascii=False), flush=True)
        if (
            identity["name"] != "GHand 5"
            or identity["hand_type"] != hand_side + "_hand"
            or identity["firmware"] not in ALLOWED_FIRMWARES[hand_side]
        ):
            raise RuntimeError(f"Unexpected hand identity for {hand_side}: {identity}")
        limits = {j.name: (float(lo), float(hi)) for j, (lo, hi) in hand._joint_limits.items()}
        joints = sdk_joint_contract(
            json.loads((PROJECT / "assets/ghand5_right.manifest.json").read_text()),
            json.loads((PROJECT / "config/cube_spin/hand.json").read_text()),
        )
        if {j.name: j.value for j in hand._joint_limits} != {j["name"].upper(): j["sdk_id"] for j in joints}:
            raise RuntimeError("Unexpected SDK joint mapping")
        target = reference_targets(reference["initial_deg"], limits)
        sample = fresh_feedback(hand, time.monotonic())
        result["last_feedback"] = sample
        validate_feedback(sample, limits, check_angle_bounds=False)
        # Keep the measured start in the log; bound all outgoing ramp targets to SDK limits.
        initial = reference_targets(sample["angles_deg"], limits, check_angle_bounds=False)
        result.update(initial_deg=initial, initial_measured_deg=sample["angles_deg"].copy(), target_deg=target)
        before = sample["feedback_time"]
        start = time.monotonic()
        print(
            "Starting reset: open gradually over 3 s, then wait for all joint errors to stay within 1 degree and remain stable for 1 s (up to 5 additional seconds). Do not place the cube in the hand.",
            flush=True,
        )
        settle_samples = []
        max_seconds = RAMP_SECONDS + SETTLE_TIMEOUT_SECONDS
        for step in range(round(max_seconds / 0.05) + 1):
            tick = start + step * 0.05
            time.sleep(max(0.0, tick - time.monotonic()))
            sample = fresh_feedback(hand, before)
            result["last_feedback"] = sample
            before = sample["feedback_time"]
            validate_feedback(sample, limits, check_angle_bounds=False)
            now = time.monotonic()
            if now - tick > 0.025 or now - start > max_seconds + 0.05:
                raise RuntimeError("Reset control deadline missed")
            goal = ramp_target(initial, target, step * 0.05)
            commands = [JointCommand(id=j, angle=goal[j.name], speed=15, torque=20) for j in hand._joint_limits]
            if not hand.move_joints(commands):
                raise RuntimeError("SDK rejected reset target")
            result["motion_commands"] += 1
            rows.append({"t": now - start, "target_deg": goal, "feedback": sample})
            if step * 0.05 >= RAMP_SECONDS:
                result["ramp_completed"] = True
            # Start checking after the final target has actually been sent.
            if step * 0.05 > RAMP_SECONDS:
                settle_samples.append(sample)
                cutoff = before - SETTLE_SECONDS
                while len(settle_samples) > 1 and settle_samples[1]["feedback_time"] <= cutoff:
                    settle_samples.pop(0)
                if returned_to_reference(settle_samples, target, SETTLE_SECONDS):
                    result["settled_before_stop"] = True
                    result["settle_seconds"] = now - start - RAMP_SECONDS
                    break
            if step % 20 == 0:
                print(f"Reset t={now - start:.2f}s, target error: {pose_error_summary(sample, target)}", flush=True)
        else:
            raise RuntimeError(
                "Reset settle timeout: joints did not reach/stabilize within 1 deg; "
                + pose_error_summary(sample, target)
            )
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        print(
            "Reset aborted: " + result["error"] + "; disconnect power immediately if the hand is still moving.",
            flush=True,
        )
    finally:
        signal.alarm(0)
        try:
            if hand.is_connected():
                result["stop_accepted"] = bool(hand.stop())
                if not result["stop_accepted"]:
                    raise RuntimeError("SDK rejected STOP")
                if target is not None and not result.get("error"):
                    start = time.monotonic()
                    while time.monotonic() - start < 0.55:
                        sample = fresh_feedback(hand, before)
                        result["last_feedback"] = sample
                        before = sample["feedback_time"]
                        stopped_samples.append(sample)
                        validate_feedback(sample, limits, check_angle_bounds=False)
                    if not returned_to_reference(stopped_samples, target):
                        raise RuntimeError(
                            "Pose did not remain at target/stable after STOP: " + pose_error_summary(sample, target)
                        )
        except BaseException as error:
            result["stop_error"] = str(error)
        finally:
            try:
                hand.close()
            except BaseException as error:
                result["close_error"] = str(error)
            result["connection_closed"] = not hand.is_connected()
        result["returned_near_reference"] = bool(target and returned_to_reference(stopped_samples, target))
        if target and "last_feedback" in result:
            result["final_error_deg"] = {
                name: result["last_feedback"]["angles_deg"][name] - goal for name, goal in target.items()
            }
        result["complete"] = bool(
            result["settled_before_stop"]
            and result.get("stop_accepted")
            and result["returned_near_reference"]
            and result["connection_closed"]
            and not any(k.endswith("error") for k in result)
        )
        status = "Reset complete" if result["complete"] else "Reset incomplete"
        connection = "connection closed" if result["connection_closed"] else "failed to close connection"
        print(f"{status}; {connection}. Disconnect actuator power before placing the cube.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--hand-side",
        choices=("right", "left"),
        default="right",
        help="Right: recorded open pose; left: SDK-neutral open pose (default: right)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Reset result JSON, overwritten each run (default: runs/reset_latest.json)",
    )
    args = parser.parse_args()
    config = json.loads((PROJECT / "config/cube_spin/hand.json").read_text())
    limits = {j["id"]: (j["min"], j["max"]) for j in config["joints"] if "min" in j}
    if args.hand_side == "left":
        reference = {"initial_deg": {name: max(lo, min(hi, 0.0)) for name, (lo, hi) in limits.items()}}
        reference_source = "sdk_neutral_left"
    else:
        reference = json.loads(REFERENCE.read_text())
        reference_source = str(REFERENCE)
    target = reference_targets(reference["initial_deg"], limits)
    result = {
        "mode": "restore_pre_policy_open_pose",
        "hardware_executed": args.execute,
        "policy_executed": False,
        "complete": False,
        "target_deg": target,
        "requested_hand_side": args.hand_side,
        "reference": reference_source,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "speed_percent": 15,
        "torque_percent": 20,
        "ramp_seconds": RAMP_SECONDS,
        "settle_timeout_seconds": SETTLE_TIMEOUT_SECONDS,
        "settle_required_seconds": SETTLE_SECONDS,
        "position_tolerance_deg": POSITION_TOLERANCE_DEG,
        "stability_tolerance_deg": STABILITY_TOLERANCE_DEG,
        "software_stop_certified": False,
        "temperature_protection": "sdk_firmware_fault_and_state",
        "host_temperature_limit_c": None,
        "feedback_protection": "sdk_firmware_fault_and_state",
    }
    result["feedback_tolerance_deg"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        if args.execute:
            execute(result, reference)
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("samples", "stop_samples", "last_feedback")}, indent=2
        ),
        flush=True,
    )
    if args.execute and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
