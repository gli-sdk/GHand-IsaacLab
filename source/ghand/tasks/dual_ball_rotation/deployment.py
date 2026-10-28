"""CPU-only Dual-ball rotation policy contract and deployment adapter; no device access."""

import hashlib
import json
import math
from pathlib import Path
import torch
from ghand.tasks.dual_ball_rotation.joint_contract import sdk_joint_contract
from ghand.tasks.dual_ball_rotation.student import JointHistory, StudentPolicy

ROOT = Path(__file__).resolve().parents[4]


def validate_contract(contract):
    if contract.get("rotation_direction", 1) not in (-1, 1):
        raise ValueError("Invalid rotation direction")
    for key, expected in [
        ("history_steps", 30),
        ("frame_width", 26),
        ("observation_size", 780),
        ("step_dt", 0.05),
        ("action_step_rad", 0.075),
    ]:
        if contract.get(key) != expected:
            raise ValueError(f"Unexpected Dual-ball rotation contract: {key} must be {expected}")
    joints = sdk_joint_contract(
        json.loads((ROOT / "assets/ghand5_right.manifest.json").read_text()),
        json.loads((ROOT / "config/dual_ball_rotation/hand.json").read_text()),
    )
    saved = contract["joint_contract"]
    if saved["names"] != [j["name"] for j in joints]:
        raise ValueError("Policy joint order does not match right-hand asset")
    for key in ("lower", "upper"):
        values = saved[key]
        if len(values) != 13 or any(not math.isfinite(v) or abs(v - j[key]) > 1e-6 for v, j in zip(values, joints)):
            raise ValueError("Policy joint limits differ from trained asset/SDK: " + key)
    return joints


def load_policy(directory):
    directory = Path(directory).resolve()
    contract = json.loads((directory / "contract.json").read_text())
    validate_contract(contract)
    checkpoint = torch.load(directory / "student.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("rotation_direction", 1) != contract.get("rotation_direction", 1):
        raise ValueError("Checkpoint/contract rotation direction mismatch")
    for key in ("joint_contract", "history_steps", "observation_size", "step_dt", "action_step_rad"):
        if checkpoint[key] != contract[key]:
            raise ValueError("Checkpoint/contract mismatch: " + key)
    reference = StudentPolicy(contract["history_steps"]).eval()
    reference.load_state_dict(checkpoint["model"], strict=True)
    if any(not torch.isfinite(v).all() for v in reference.state_dict().values()):
        raise ValueError("Nonfinite policy weights")
    actor = torch.jit.load(str(directory / "actor.ts"), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(123)
    probes = torch.cat(
        (torch.zeros(1, 780), -torch.ones(1, 780), torch.ones(1, 780), torch.randn(5, 780, generator=generator)), 0
    )
    with torch.inference_mode():
        actual = actor(probes)
        expected = reference(probes)
    if (
        actual.shape != (8, 13)
        or not torch.isfinite(actual).all()
        or not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    ):
        raise ValueError("Exported actor does not match student checkpoint")
    # Warm the exact batch size used by the hardware loop before any SDK connection.
    with torch.inference_mode():
        for _ in range(4):
            actor(torch.zeros(1, 780))
    hashes = {
        name + "_sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for name, filename in [("actor", "actor.ts"), ("student", "student.pt"), ("contract", "contract.json")]
    }
    return contract, actor, hashes


class PolicyAdapter:
    """Accumulate accepted position targets; measurements never replace command history."""

    def __init__(self, contract, actor, initial):
        self.joints = validate_contract(contract)
        self.names = [j["name"].upper() for j in self.joints]
        self.lower = torch.tensor(contract["joint_contract"]["lower"])
        self.upper = torch.tensor(contract["joint_contract"]["upper"])
        self.actor = actor
        self.history = JointHistory(1, 30, self.lower, self.upper)
        self.command = initial.clone()
        self.history.reset(initial, self.command)
        self.last_time = None
        self.pending = False

    def propose(self, measured, timestamp):
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("Feedback timestamp must increase")
        if self.pending:
            raise ValueError("Previous command has not been acknowledged")
        self.history.advance(measured, self.command, torch.zeros(1, dtype=torch.bool))
        with torch.inference_mode():
            action = self.actor(self.history.observation())
        if action.shape != (1, 13) or not torch.isfinite(action).all() or (action.abs() > 1.000001).any():
            raise ValueError("Invalid actor output")
        self.last_time = timestamp
        self.pending = True
        target = (self.command + 0.075 * action.clamp(-1, 1)).clamp(self.lower, self.upper)
        return target, action

    def acknowledge(self, target):
        if not self.pending or target.shape != (1, 13) or not torch.isfinite(target).all():
            raise ValueError("Invalid SDK-accepted command")
        if ((target < self.lower - 1e-6) | (target > self.upper + 1e-6)).any():
            raise ValueError("SDK-accepted target outside trained command limits")
        self.command = target.clone()
        self.pending = False
