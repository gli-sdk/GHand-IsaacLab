"""Prepare the saved Dual-ball rotation start pose with an EMPTY right hand. Offline by default."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT))
from scripts.dual_ball_rotation.deploy import execute, sdk_percent, positive_seconds, SDK_ROOT, INTERFACE, json_safe
from ghand.tasks.dual_ball_rotation.start_pose import load_start_pose, PoseAdapter


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--empty-hand", action="store_true")
    p.add_argument("--onsite-cutoff-ready", action="store_true")
    p.add_argument("--ramp-seconds", type=positive_seconds, default=4.0)
    p.add_argument("--hold-seconds", type=positive_seconds, default=2.0)
    p.add_argument("--speed", type=sdk_percent, default=25)
    p.add_argument("--torque", type=sdk_percent, default=25)
    p.add_argument("--interface", default=INTERFACE)
    p.add_argument("--sdk-root", type=Path, default=SDK_ROOT)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.execute and not (a.empty_hand and a.onsite_cutoff_ready):
        p.error("--execute requires --empty-hand --onsite-cutoff-ready")
    contract = json.loads((a.policy_dir / "contract.json").read_text())
    target, setup = load_start_pose(a.policy_dir, contract)
    reference = torch.tensor([contract["joint_contract"]["lower"]])
    plan = PoseAdapter(contract, reference, target, a.ramp_seconds, a.hold_seconds).plan
    out = a.output or ROOT / "runs/dual_ball_rotation/hardware" / (
        "prepare_latest.json" if a.execute else "prepare_offline_latest.json"
    )
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    reserved = tempfile.NamedTemporaryFile(
        mode="w", dir=out.parent, prefix="." + out.name + ".", suffix=".tmp", delete=False
    )
    result = {
        "mode": "prepare_trained_start_pose",
        "policy_dir": str(a.policy_dir.resolve()),
        "hardware_motion_executed": False,
        "hardware_policy_executed": False,
        "hardware_ready": False,
        "object_present_reported": False,
        "empty_hand_reported": a.empty_hand,
        "requested_hand_side": "right",
        "interface": a.interface,
        "sdk_root": str(a.sdk_root),
        "trained_setup": setup,
        "duration_seconds": plan["duration_seconds"],
        "control_hz": 20,
        "speed_percent": a.speed,
        "torque_percent": a.torque,
        "trajectory": plan,
        "software_stop_verified": False,
        "offline_start": "SDK lower limits, placeholder; execute uses fresh measured angles",
    }

    def factory(c, actor, initial):
        adapter = PoseAdapter(c, initial, target, a.ramp_seconds, a.hold_seconds)
        result["trajectory"] = adapter.plan
        return adapter

    try:
        torch.set_num_threads(1)
        print(
            "Preparing pose: loading the selected model's initial values for all 13 joints; fixed motion only, no policy execution.",
            flush=True,
        )
        print(
            "Training palm forward-tilt range: "
            + str(setup["palm_tilt_range_deg"])
            + " degrees; adjust the mount angle manually.",
            flush=True,
        )
        if setup["palm_side_tilt_deg"]:
            print(
                "Training palm side tilt: "
                + str(setup["palm_side_tilt_deg"])
                + " degrees (negative values tilt the right hand toward the thumb); adjust the mount manually.",
                flush=True,
            )
        if a.execute:
            execute(contract, None, result, adapter_factory=factory)
            if result.get("last_feedback"):
                result["final_error_deg"] = {
                    n: result["last_feedback"]["angles_deg"][n] - v for n, v in setup["joint_deg"].items()
                }
            print(
                "Fixed motion has ended. Isolate actuator power before placing the balls; confirm the actual pose can support them, then place them and withdraw your hands.",
                flush=True,
            )
        else:
            print(
                "Offline plan generated; no SDK connection. Execution transitions smoothly from the measured pose to the partially closed target pose. Physical pose retention after exit has not been verified.",
                flush=True,
            )
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        with reserved:
            json.dump(json_safe(result), reserved, indent=2, allow_nan=False)
            reserved.write("\n")
        os.replace(reserved.name, out)
    print("Log: " + str(out), flush=True)
    if result.get("error") or (a.execute and not result.get("policy_window_completed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
