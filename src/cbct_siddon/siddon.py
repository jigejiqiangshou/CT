"""Siddon (1985) exact radiological path through a 3D voxel array."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class VoxelGrid:
    """
    Voxel grid aligned with phantom3d.m sampling on [-1, 1]^3.

    Plane boundaries: -1 - dx/2 + k*dx for k = 0..n (n voxels).
    """

    n: int

    @property
    def spacing(self) -> tuple[float, float, float]:
        if self.n < 2:
            return (2.0, 2.0, 2.0)
        d = 2.0 / (self.n - 1)
        return (d, d, d)

    @property
    def origin(self) -> tuple[float, float, float]:
        """Position of the first x/y/z plane (left boundary of voxel 0)."""
        if self.n < 2:
            return (-1.0, -1.0, -1.0)
        d = 2.0 / (self.n - 1)
        return (-1.0 - d / 2.0, -1.0 - d / 2.0, -1.0 - d / 2.0)

    def plane_positions(self, axis: int) -> np.ndarray:
        """n+1 plane coordinates along axis (0=x, 1=y, 2=z)."""
        d = self.spacing[axis]
        o = self.origin[axis]
        return o + np.arange(self.n + 1, dtype=np.float64) * d

    def index_at_midpoint(self, point: np.ndarray) -> tuple[int, int, int] | None:
        """Voxel indices for a point inside the volume, else None."""
        d = self.spacing
        o = self.origin
        idx = []
        for ax in range(3):
            t = (point[ax] - o[ax]) / d[ax]
            if t < 0.0 or t >= self.n:
                return None
            i = int(np.floor(t))
            if i >= self.n:
                i = self.n - 1
            idx.append(i)
        return idx[0], idx[1], idx[2]


def _alphas_for_axis(
    s: np.ndarray,
    p: np.ndarray,
    planes: np.ndarray,
    axis: int,
) -> np.ndarray:
    delta = p[axis] - s[axis]
    if abs(delta) < _EPS:
        return np.array([], dtype=np.float64)
    out = (planes - s[axis]) / delta
    return out[(out >= 0.0) & (out <= 1.0)]


def ray_enter_exit(
    s: np.ndarray,
    p: np.ndarray,
    grid: VoxelGrid,
) -> tuple[float, float] | None:
    """Return (alpha_min, alpha_max) where ray meets the volume, or None."""
    alphas = [0.0, 1.0]
    for ax in range(3):
        planes = grid.plane_positions(ax)
        delta = p[ax] - s[ax]
        if abs(delta) < _EPS:
            if s[ax] < planes[0] or s[ax] > planes[-1]:
                return None
            continue
        a0 = (planes[0] - s[ax]) / delta
        a1 = (planes[-1] - s[ax]) / delta
        alphas.extend([min(a0, a1), max(a0, a1)])

    alpha_min = max(0.0, min(alphas))
    alpha_max = min(1.0, max(alphas))
    if alpha_min >= alpha_max - _EPS:
        return None
    return alpha_min, alpha_max


def _merge_alphas(alphas: np.ndarray) -> np.ndarray:
    """Sort and remove duplicate alpha values (ray through edges/corners)."""
    if alphas.size == 0:
        return alphas
    a = np.sort(alphas)
    keep = [0]
    for i in range(1, a.size):
        if a[i] - a[keep[-1]] > _EPS:
            keep.append(i)
    return a[keep]


def line_integral_siddon(
    s: np.ndarray,
    p: np.ndarray,
    volume: np.ndarray,
    grid: VoxelGrid | None = None,
) -> float:
    """
    Line integral sum(ell_m * V[i,j,k]) along segment s -> p.

    Parameters
    ----------
    s, p : array-like shape (3,)
        Source and endpoint (detector pixel).
    volume : ndarray (n,n,n)
    grid : VoxelGrid, optional
        Defaults to VoxelGrid(volume.shape[0]).
    """
    s = np.asarray(s, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if grid is None:
        grid = VoxelGrid(volume.shape[0])

    bounds = ray_enter_exit(s, p, grid)
    if bounds is None:
        return 0.0
    alpha_min, alpha_max = bounds

    alphas = [alpha_min, alpha_max]
    for ax in range(3):
        alphas.extend(_alphas_for_axis(s, p, grid.plane_positions(ax), ax).tolist())

    alphas = _merge_alphas(np.asarray(alphas, dtype=np.float64))
    if alphas.size < 2:
        return 0.0

    ray_len = float(np.linalg.norm(p - s))
    direction = p - s
    total = 0.0

    for m in range(alphas.size - 1):
        a0, a1 = alphas[m], alphas[m + 1]
        d_alpha = a1 - a0
        if d_alpha <= _EPS:
            continue
        alpha_mid = 0.5 * (a0 + a1)
        point = s + alpha_mid * direction
        idx = grid.index_at_midpoint(point)
        if idx is None:
            continue
        i, j, k = idx
        ell = ray_len * d_alpha
        total += ell * volume[i, j, k]

    return total
