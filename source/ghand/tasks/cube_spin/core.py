"""SDK-free joint contracts and batched world-frame spin math."""

import math

import torch


_SDK_NAMES = (
    "THUMB_IP",
    "THUMB_MCP",
    "THUMB_TMC_FE",
    "THUMB_TMC_AA",
    "THUMB_TMC_PS",
    "FF_DIP",
    "FF_PIP",
    "FF_MCP",
    "FF_MCP_AA",
    "MF_DIP",
    "MF_PIP",
    "MF_MCP",
    "RF_DIP",
    "RF_PIP",
    "RF_MCP",
    "LF_DIP",
    "LF_PIP",
    "LF_MCP",
)
_SDK_IDS = {name: index for index, name in enumerate(_SDK_NAMES)}
_ACTIVE = set(_SDK_NAMES) - {"THUMB_IP", "FF_DIP", "MF_DIP", "RF_DIP", "LF_DIP"}


def sdk_joint_contract(manifest: dict, sdk_config: dict) -> list[dict]:
    """Intersect radians in the asset with SDK degrees and mimic feasibility."""
    names = manifest["active_joints"]
    upper_names = [name.upper() for name in names]
    if len(upper_names) != 13 or len(set(upper_names)) != 13 or set(upper_names) != _ACTIVE:
        raise ValueError("Expected exactly the 13 unique known active joints")
    sdk = {}
    for joint in sdk_config["joints"]:
        name = joint["id"]
        if name not in _SDK_IDS or name in sdk:
            raise ValueError(f"Unknown or duplicate SDK joint: {name}")
        sdk[name] = joint
    joints = manifest["joints"]
    contract = []
    for name, sdk_name in zip(names, upper_names):
        if name not in joints or sdk_name not in sdk:
            raise ValueError(f"Missing joint limits: {name}")
        try:
            sim_lower, sim_upper = float(joints[name]["lower"]), float(joints[name]["upper"])
            sdk_lower, sdk_upper = math.radians(sdk[sdk_name]["min"]), math.radians(sdk[sdk_name]["max"])
            if not all(math.isfinite(value) for value in (sim_lower, sim_upper, sdk_lower, sdk_upper)):
                raise ValueError("Limits must be finite")
            lower, upper = max(sim_lower, sdk_lower), min(sim_upper, sdk_upper)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid joint limits: {name}") from error
        for child_name, child in joints.items():
            mimic = child.get("mimic")
            if not mimic or mimic["joint"] != name:
                continue
            multiplier, offset = float(mimic["multiplier"]), float(mimic["offset"])
            child_lower, child_upper = float(child["lower"]), float(child["upper"])
            if not all(math.isfinite(value) for value in (multiplier, offset, child_lower, child_upper)):
                raise ValueError(f"Invalid mimic limits: {child_name}")
            if child_lower > child_upper:
                raise ValueError(f"Reversed mimic limits: {child_name}")
            if multiplier == 0.0:
                if not child_lower <= offset <= child_upper:
                    raise ValueError(f"Infeasible constant mimic: {child_name}")
                continue
            endpoints = ((child_lower - offset) / multiplier, (child_upper - offset) / multiplier)
            lower, upper = max(lower, min(endpoints)), min(upper, max(endpoints))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"Empty or invalid feasible joint interval: {name}")
        contract.append({"name": name, "sdk_id": _SDK_IDS[sdk_name], "lower": lower, "upper": upper})
    return contract


def command_upper_limits(contract: list[dict], command_upper_deg: dict[str, float]) -> list[float]:
    """Separate target ceilings; never modify physical/normalization limits."""
    names = {row["name"].upper() for row in contract}
    if set(command_upper_deg) - names:
        raise ValueError("Command limits contain unknown active joints")
    result = []
    for row in contract:
        name = row["name"].upper()
        upper = math.radians(float(command_upper_deg[name])) if name in command_upper_deg else row["upper"]
        if not math.isfinite(upper) or not row["lower"] < upper <= row["upper"] + 1e-12:
            raise ValueError(f"Command ceiling outside physical joint interval: {name}")
        result.append(min(upper, row["upper"]))
    return result


def command_limits(
    contract: list[dict], command_upper_deg: dict[str, float], margin_fraction: float = 0.0
) -> tuple[list[float], list[float]]:
    """Inset both target endpoints; keep physical limits and normalization intact."""
    if not math.isfinite(margin_fraction) or not 0.0 <= margin_fraction < 0.5:
        raise ValueError("Command margin fraction must be finite and in [0, 0.5)")
    ceilings = command_upper_limits(contract, command_upper_deg)
    lower, upper = [], []
    for row, ceiling in zip(contract, ceilings):
        inset = margin_fraction * (row["upper"] - row["lower"])
        lo, hi = row["lower"] + inset, min(row["upper"] - inset, ceiling)
        if lo >= hi:
            raise ValueError(f"Empty command interval: {row['name']}")
        lower.append(lo)
        upper.append(hi)
    return lower, upper


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)


def rotation_vector(previous, current):
    """Actual shortest world-frame orientation change; quaternions are wxyz.

    Inputs must contain nonzero quaternions. Sample steps
    below pi radians to accumulate unambiguous signed turns.
    """
    previous, current = _unit(previous), _unit(current)
    pw, pv = previous[..., :1], previous[..., 1:]
    cw, cv = current[..., :1], current[..., 1:]
    # current * conjugate(previous), so the delta axis is in the world frame.
    scalar = cw * pw + (cv * pv).sum(dim=-1, keepdim=True)
    vector = -cw * pv + pw * cv - torch.cross(cv, pv, dim=-1)
    sign = torch.where(scalar < 0, -1.0, 1.0)
    scalar, vector = scalar * sign, vector * sign
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, scalar)
    scale = torch.where(norm > 1e-8, angle / norm.clamp_min(1e-8), 2.0)
    return vector * scale


def rotation_increment(previous, current, axis):
    """Signed world-axis progress and perpendicular motion, in radians."""
    delta = rotation_vector(previous, current)
    axis = _unit(axis.to(device=current.device, dtype=current.dtype))
    axial = (delta * axis).sum(dim=-1)
    off_axis = torch.linalg.vector_norm(delta - axial.unsqueeze(-1) * axis, dim=-1)
    return axial, off_axis


def spin_reward(
    angular_velocity,
    linear_velocity,
    displacement,
    action_delta,
    axis,
    target_speed: float,
    *,
    position_penalty_scale: float = 0.5,
) -> dict:
    """Cap forward spin benefit while retaining full reverse-speed penalties.

    Leaving reverse reward uncapped prevents zero-net forward/reset oscillations
    from earning rotation benefit. Stability and action-rate terms penalize motion.
    """
    if not math.isfinite(target_speed) or target_speed <= 0.0:
        raise ValueError("target_speed must be finite and positive")
    if not math.isfinite(position_penalty_scale) or position_penalty_scale < 0.0:
        raise ValueError("position_penalty_scale must be finite and nonnegative")
    axis = _unit(axis.to(device=angular_velocity.device, dtype=angular_velocity.dtype))
    axial_speed = (angular_velocity * axis).sum(dim=-1)
    perpendicular = angular_velocity - axial_speed.unsqueeze(-1) * axis
    terms = {
        "rotation": 2.0 * (axial_speed / target_speed).clamp(max=1.0),
        "off_axis": -0.15 * perpendicular.square().sum(dim=-1),
        "position": -position_penalty_scale * (displacement / 0.05).square().sum(dim=-1),
        "linear_velocity": -0.05 * linear_velocity.square().sum(dim=-1),
        "action_rate": -0.01 * action_delta.square().sum(dim=-1),
    }
    terms["total"] = sum(terms.values())
    return terms
