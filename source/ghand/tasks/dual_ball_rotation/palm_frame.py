"""Palm-reference coordinates for a fixed per-episode forward inclination.

The existing task's fingers point along -Y. Positive world-X rotation therefore
lowers the fingertips. Gravity is never rotated; positions, vectors and the hand
root are transformed consistently. Zero inclination preserves legacy behavior.
"""

import math
import torch


def sample_tilts(count, bounds_deg, generator):
    lo, hi = bounds_deg
    if count < 1 or not all(math.isfinite(v) for v in (lo, hi)) or not -90 <= lo <= hi <= 90:
        raise ValueError("Require positive count and finite ordered tilt bounds within +/-90 degrees")
    if lo == hi:
        return torch.full((count,), float(lo))
    return lo + (hi - lo) * torch.rand(count, generator=generator)


def rotate_x(vectors, degrees):
    """Rotate (...,3) vectors; first dimension matches the per-environment angles."""
    angle = torch.deg2rad(degrees)
    while angle.ndim < vectors.ndim - 1:
        angle = angle.unsqueeze(-1)
    c, s = angle.cos(), angle.sin()
    x, y, z = vectors.unbind(-1)
    return torch.stack((x, c * y - s * z, s * y + c * z), dim=-1)


def rotate_y(vectors, degrees):
    angle = torch.deg2rad(degrees)
    while angle.ndim < vectors.ndim - 1:
        angle = angle.unsqueeze(-1)
    c, s = angle.cos(), angle.sin()
    x, y, z = vectors.unbind(-1)
    return torch.stack((c * x + s * z, y, -s * x + c * z), dim=-1)


def rotate_palm(vectors, forward_deg, side_deg=None, inverse=False):
    """Side tilt about palm Y, then forward tilt about world X; inverse reverses both."""
    if inverse:
        vectors = rotate_x(vectors, -forward_deg)
        return rotate_y(vectors, -side_deg) if side_deg is not None else vectors
    if side_deg is not None:
        vectors = rotate_y(vectors, side_deg)
    return rotate_x(vectors, forward_deg)


def tilted_quaternion(base_wxyz, degrees, side_degrees=None):
    """Compose the inclination on the world side of the saved hand orientation."""
    if side_degrees is not None:
        half = torch.deg2rad(side_degrees) * 0.5
        c, s = half.cos(), half.sin()
        w, x, y, z = base_wxyz.unbind(-1)
        base_wxyz = torch.stack((c * w - s * y, c * x + s * z, c * y + s * w, c * z - s * x), dim=-1)
    half = torch.deg2rad(degrees) * 0.5
    c, s = half.cos(), half.sin()
    w, x, y, z = base_wxyz.unbind(-1)
    return torch.stack((c * w - s * x, c * x + s * w, c * y - s * z, c * z + s * y), dim=-1)
