"""Pair position orbit; ball self-spin and translation never count as turns."""

import torch


def orbit_delta(previous, current, diameter, direction=1):
    if direction not in (-1, 1):
        raise ValueError("Rotation direction must be -1 or +1")
    a, b = previous[..., :2], current[..., :2]
    valid = (
        (a.norm(dim=-1) >= 0.65 * diameter)
        & (b.norm(dim=-1) >= 0.65 * diameter)
        & (previous[..., 2].abs() < 0.8 * diameter)
        & (current[..., 2].abs() < 0.8 * diameter)
    )
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    delta = torch.atan2(cross, (a * b).sum(-1))
    # Reject teleports / ambiguous sub-step changes rather than reward them.
    valid &= delta.abs() < 0.8
    return torch.where(valid, direction * delta, 0.0), valid


def novel_progress(net_angle, high_water, max_credit=None):
    next_high = torch.maximum(net_angle, high_water)
    credit = next_high - high_water
    if max_credit is not None:
        credit = credit.clamp_max(max_credit)
    return credit, next_high


def orbit_goal_offsets(phase, radii, direction=1):
    """Positive task phase advances in the requested physical direction."""
    if direction not in (-1, 1):
        raise ValueError("Rotation direction must be -1 or +1")
    angle = direction * phase
    return torch.stack((angle.cos(), angle.sin()), -1) * radii


def stall_fraction(mean_forward_speed, threshold):
    """Soft lack-of-forward-progress cost; reverse motion does not avoid stalling."""
    return (1.0 - mean_forward_speed / threshold).clamp(0.0, 1.0)


def shared_lift_fraction(world_heights, initial_world_heights, tolerance, transition):
    """Bounded cost only when BOTH ball centers rise beyond their own start heights."""
    common_rise = (world_heights - initial_world_heights).amin(dim=-1)
    return ((common_rise - tolerance) / transition).clamp(0.0, 1.0)
