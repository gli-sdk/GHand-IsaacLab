# SPDX-License-Identifier: BSD-3-Clause
"""Constraint-consistent initialization for the GHand's native mimic joints."""

import math

import torch


def feasible_parent_limits(parent_lower, parent_upper, child_lower, child_upper, multiplier, offset):
    """Intersect a driven joint's range with its follower's range (radians)."""
    if multiplier == 0:
        raise ValueError("A zero mimic multiplier is not supported")
    a = (child_lower - offset) / multiplier
    b = (child_upper - offset) / multiplier
    lower = torch.maximum(parent_lower, torch.minimum(a, b))
    upper = torch.minimum(parent_upper, torch.maximum(a, b))
    if torch.any(lower > upper):
        raise ValueError("Mimic joint limits have no feasible intersection")
    return lower, upper


def apply_mimic_state(position, velocity, relationships):
    """Return states obeying child = multiplier * parent + offset.

    Relationships contain (child index, parent index, multiplier, offset).
    Native physics constraints, not this reset helper, enforce coupling during rollout.
    """
    position = position.clone()
    velocity = velocity.clone()
    for child, parent, multiplier, offset in relationships:
        position[:, child] = multiplier * position[:, parent] + offset
        velocity[:, child] = multiplier * velocity[:, parent]
    return position, velocity


def validate_relative_goal_config(probability, angle_min, angle_max, margin, tolerance):
    """Reject enabled curricula that can generate an already successful target."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("relative_goal_probability must be finite and between zero and one")
    if probability == 0.0:
        return
    if not all(math.isfinite(value) for value in (angle_min, angle_max, margin, tolerance)):
        raise ValueError("Relative goal angles, margin, and success tolerance must be finite")
    if not (0.0 < angle_min <= angle_max <= math.pi and margin > 0.0 and tolerance > 0.0):
        raise ValueError("Require 0 < relative goal minimum <= maximum <= pi and positive margin/tolerance")
    if angle_min < tolerance + margin:
        raise ValueError("relative_goal_angle_min must be at least success_tolerance + relative_goal_margin")


def sample_relative_goal_rotations(object_quat, angle_min, angle_max):
    """Sample fixed wxyz targets with a uniform spherical axis and uniform angle.

    Left multiplication applies a world-frame rotation to the current object pose.
    Keeping this helper in Torch makes the curriculum geometry simulator-independent.
    """
    samples = torch.rand((object_quat.shape[0], 3), device=object_quat.device, dtype=object_quat.dtype)
    z = 2.0 * samples[:, 0] - 1.0
    azimuth = 2.0 * math.pi * samples[:, 1]
    radius = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    axis = torch.stack((radius * torch.cos(azimuth), radius * torch.sin(azimuth), z), dim=-1)
    half_angle = 0.5 * (angle_min + (angle_max - angle_min) * samples[:, 2:3])
    delta_w, delta_v = torch.cos(half_angle), axis * torch.sin(half_angle)
    current = torch.nn.functional.normalize(object_quat, dim=-1)
    current_w, current_v = current[:, :1], current[:, 1:]
    goal = torch.cat(
        (
            delta_w * current_w - (delta_v * current_v).sum(dim=-1, keepdim=True),
            delta_w * current_v + current_w * delta_v + torch.cross(delta_v, current_v, dim=-1),
        ),
        dim=-1,
    )
    return torch.nn.functional.normalize(goal, dim=-1)
