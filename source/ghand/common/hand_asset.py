# SPDX-License-Identifier: BSD-3-Clause
"""GHand 5 right hand from the bundled vendor model, with 13 driven joints."""

import os
from pathlib import Path
import hashlib
import json

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

USD_PATH = Path(os.environ.get("GHAND_USD_PATH", str(Path(__file__).resolve().parents[3] / "assets/ghand5_right.usd")))
# Bind the control model to the converted asset, not to a later-changing checkout.
_MANIFEST = json.loads(USD_PATH.with_suffix(".manifest.json").read_text())
if hashlib.sha256(USD_PATH.read_bytes()).hexdigest() != _MANIFEST["usd_sha256"]:
    raise ValueError("GHand USD does not match its asset manifest; regenerate the asset")
_JOINTS = _MANIFEST["joints"]
ACTIVE_JOINTS = list(_MANIFEST["active_joints"])
PASSIVE_JOINTS = list(_MANIFEST["passive_joints"])
MIMIC_JOINTS = {
    name: (j["mimic"]["joint"], j["mimic"]["multiplier"], j["mimic"]["offset"])
    for name, j in _JOINTS.items()
    if j["mimic"] is not None
}
_EFFORT = {name: j["effort"] for name, j in _JOINTS.items()}
_VELOCITY = {name: j["velocity"] for name, j in _JOINTS.items()}
_DEFAULT_POS = {name: max(0.0, j["lower"]) for name, j in _JOINTS.items()}
for child, (parent, multiplier, offset) in MIMIC_JOINTS.items():
    _DEFAULT_POS[parent] = max(_DEFAULT_POS[parent], -offset / multiplier + 0.01)
    _DEFAULT_POS[child] = multiplier * _DEFAULT_POS[parent] + offset

GHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, enable_gyroscopic_forces=False, max_depenetration_velocity=1.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), rot=(0.0, 0.0, -0.70710678, 0.70710678), joint_pos=_DEFAULT_POS
    ),
    actuators={
        "active": ImplicitActuatorCfg(
            joint_names_expr=ACTIVE_JOINTS,
            stiffness=1.0,
            # Numerical regularization for small-link native mimic constraints.
            # This is not calibrated motor rotor inertia.
            armature=1e-4,
            damping={n: 0.25 * _EFFORT[n] / _VELOCITY[n] for n in ACTIVE_JOINTS},
            effort_limit_sim={n: _EFFORT[n] for n in ACTIVE_JOINTS},
            velocity_limit_sim={n: _VELOCITY[n] for n in ACTIVE_JOINTS},
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=PASSIVE_JOINTS,
            stiffness=0.0,
            damping=0.0,
            armature=1e-4,
            effort_limit_sim={n: _EFFORT[n] for n in PASSIVE_JOINTS},
            velocity_limit_sim={n: _VELOCITY[n] for n in PASSIVE_JOINTS},
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
