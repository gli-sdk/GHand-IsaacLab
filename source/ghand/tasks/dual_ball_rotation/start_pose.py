"""Read a policy's training start pose and prepare a fixed ramp; no device access."""

import math
from pathlib import Path
import torch
from ghand.common.config_io import load_config_yaml
from ghand.tasks.dual_ball_rotation.deployment import validate_contract


def load_start_pose(directory, contract):
    joints = validate_contract(contract)
    cfg = load_config_yaml(Path(directory) / "params/env.yaml")
    values = cfg["robot_cfg"]["init_state"]["joint_pos"]
    try:
        angles = [float(values[j["name"]]) for j in joints]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Saved configuration must name all 13 active start angles") from exc
    for value, joint in zip(angles, joints):
        if not math.isfinite(value) or not joint["lower"] - 1e-6 <= value <= joint["upper"] + 1e-6:
            raise ValueError("Invalid trained initial angle: " + joint["name"])
    setup = {
        "palm_tilt_range_deg": list(cfg.get("palm_tilt_range_deg", (0.0, 0.0))),
        "palm_side_tilt_deg": float(cfg.get("palm_side_tilt_deg", 0.0)),
        "ball_diameter_mm": float(cfg["ball_radius"]) * 2000.0,
        "mass_g_each": float(cfg["ball_mass"]) * 1000.0,
        "joint_deg": {j["name"].upper(): math.degrees(a) for j, a in zip(joints, angles)},
    }
    return torch.tensor([angles]), setup


class PoseAdapter:
    """Move from measured pose to the trained initial pose, then hold it."""

    def __init__(self, contract, initial, target, ramp_seconds=4.0, hold_seconds=2.0):
        joints = validate_contract(contract)
        for duration in (ramp_seconds, hold_seconds):
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Positive finite durations required")
        self.lower = torch.tensor(contract["joint_contract"]["lower"])
        self.upper = torch.tensor(contract["joint_contract"]["upper"])
        for q in (initial, target):
            if q.shape != (1, 13) or not torch.isfinite(q).all():
                raise ValueError("Expected 13 finite angles")
        if ((target < self.lower - 1e-6) | (target > self.upper + 1e-6)).any():
            raise ValueError("Start pose outside trained limits")
        initial = initial.detach().cpu().clamp(self.lower, self.upper)
        target = target.detach().cpu().clamp(self.lower, self.upper)
        ramp_steps = math.ceil(ramp_seconds / 0.05)
        hold_steps = math.ceil(hold_seconds / 0.05)
        t = torch.arange(1, ramp_steps + 1, dtype=torch.float32)[:, None] / ramp_steps
        blend = t * t * (3 - 2 * t)
        self.targets = torch.cat((initial + (target - initial) * blend, target.expand(hold_steps, -1)), 0)
        self.names = [j["name"].upper() for j in joints]
        self.plan = {
            "joint_order": self.names,
            "control_hz": 20,
            "duration_seconds": len(self.targets) * 0.05,
            "ramp_seconds": ramp_steps * 0.05,
            "hold_seconds": hold_steps * 0.05,
            "start_deg": torch.rad2deg(initial[0]).tolist(),
            "target_deg": torch.rad2deg(target[0]).tolist(),
            "targets_deg": torch.rad2deg(self.targets).tolist(),
            "description": "Smooth transition to saved initial pose, then hold; no policy inference",
        }
        self.index = 0
        self.last_time = None
        self.pending = False

    def propose(self, measured, timestamp):
        if self.pending or not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("Commands require acknowledged targets and fresh timestamps")
        if self.index >= len(self.targets):
            raise ValueError("Fixed trajectory finished")
        self.last_time = timestamp
        self.pending = True
        return self.targets[self.index : self.index + 1].clone(), torch.zeros(1, 13)

    def acknowledge(self, target):
        if not self.pending or not torch.allclose(target, self.targets[self.index : self.index + 1], atol=1e-6):
            raise ValueError("Fixed target was not accepted unchanged")
        self.index += 1
        self.pending = False
