"""Reproducible paired placement perturbations; no simulator or policy dependency."""

import math
import torch


def placement_offsets(num_envs, bound_mm, seed):
    """Common XY shift of both balls, independently uniform on each axis (metres).

    A private CPU generator makes teacher and student conditions identical,
    independent of policy initialization or the simulator's random generator.
    """
    if num_envs < 1 or not math.isfinite(bound_mm) or bound_mm < 0:
        raise ValueError("Need positive environment count and finite nonnegative placement bound")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (2 * torch.rand((num_envs, 2), generator=generator) - 1) * (bound_mm / 1000.0)


def sample_domains(num_envs, bound_mm, friction_scale_range, nominal_fraction, generator):
    """Per-episode common XY shift and shared sphere material scale; fixed buckets."""
    lo, hi = friction_scale_range
    if (
        num_envs < 1
        or not math.isfinite(bound_mm)
        or bound_mm < 0
        or not all(math.isfinite(v) for v in (lo, hi, nominal_fraction))
        or not 0 < lo <= 1 <= hi
        or not 0 <= nominal_fraction <= 1
    ):
        raise ValueError("Invalid placement/friction domain range")
    shift = (2 * torch.rand((num_envs, 2), generator=generator) - 1) * (bound_mm / 1000.0)
    buckets = torch.linspace(lo, hi, 65)
    scales = buckets[torch.randint(65, (num_envs,), generator=generator)]
    nominal = torch.rand(num_envs, generator=generator) < nominal_fraction
    shift[nominal] = 0.0
    scales[nominal] = 1.0
    return shift, scales


def resized_ball_geometry(radius, orbit_radii, initial_heights, diameter_mm):
    """Resize spheres with unchanged bottom heights and ellipse surface clearance.

    An opt-in comparison also needs wider center spacing/goal ellipse; otherwise
    larger spheres start intersecting or receive mutually impossible waypoints.
    This preserves the hand pose, not each sphere's center position.
    """
    if diameter_mm is None:
        return radius, tuple(orbit_radii), tuple(initial_heights)
    if not math.isfinite(diameter_mm) or diameter_mm <= 0.0:
        raise ValueError("Ball diameter must be finite and positive")
    new_radius = diameter_mm / 2000.0
    difference = new_radius - radius
    radii = tuple(value + difference for value in orbit_radii)
    heights = tuple(value + difference for value in initial_heights)
    if min(radii) <= new_radius:
        raise ValueError("Goal ellipse must leave positive sphere surface clearance")
    return new_radius, radii, heights
