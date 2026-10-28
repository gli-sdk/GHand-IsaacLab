"""SDK-free joint contracts and batched world-frame spin math."""

import math


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
