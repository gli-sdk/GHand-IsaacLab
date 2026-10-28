"""Cube property sampling and uniform-density inertia, independent of Isaac Lab."""

import math
import torch


def sample_range(bounds, count, *, seed=None):
    if len(bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in bounds) or bounds[0] > bounds[1]:
        raise ValueError("Cube bounds must be two finite positive numbers in ascending order")
    generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(count, device="cpu", generator=generator)


def cube_inertia(edge_m, mass_kg):
    """Diagonal 3x3 inertia flattened as required by the PhysX tensor API."""
    inertia = edge_m.new_zeros((edge_m.numel(), 9))
    inertia[:, [0, 4, 8]] = (mass_kg * edge_m.square() / 6.0)[:, None]
    return inertia


def friction_buckets(static_range, dynamic_ratio_range, count, *, seed):
    """Finite [static, dynamic, restitution] materials; dynamic = static * ratio.

    A private RNG avoids coupling materials to size/mass draws or reset seeds.
    Bounds are engineering coverage, not measured wood/finger-pad coefficients.
    """
    sample_range(static_range, 0)
    sample_range(dynamic_ratio_range, 0)
    if dynamic_ratio_range[1] > 1.0 or not 1 <= count <= 4096:
        raise ValueError("Dynamic/static ratio must be <= 1 and bucket count in [1, 4096]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.rand((count, 2), generator=generator)
    static = static_range[0] + (static_range[1] - static_range[0]) * draws[:, 0]
    ratio = dynamic_ratio_range[0] + (dynamic_ratio_range[1] - dynamic_ratio_range[0]) * draws[:, 1]
    return torch.stack((static, static * ratio, torch.zeros_like(static)), dim=-1)
