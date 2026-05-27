"""Run a single-ray line integral with source directly above the volume center.

This file is intentionally self-contained so the full flow can be copied from
one place:
- phantom generation
- voxel grid definition
- Siddon line integral
- CLI entry point
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path

import numpy as np

_EPS = 1e-12


# -----------------------------
# Phantom generation
# -----------------------------

MODIFIED_SHEPP_LOGAN: np.ndarray = np.array(
    [
        [1.0, 0.6900, 0.920, 0.810, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.8, 0.6624, 0.874, 0.780, 0.0, -0.0184, 0.0, 0.0, 0.0, 0.0],
        [-0.2, 0.1100, 0.310, 0.220, 0.22, 0.0, 0.0, -18.0, 0.0, 10.0],
        [-0.2, 0.1600, 0.410, 0.280, -0.22, 0.0, 0.0, 18.0, 0.0, 10.0],
        [0.1, 0.2100, 0.250, 0.410, 0.0, 0.35, -0.15, 0.0, 0.0, 0.0],
        [0.1, 0.0460, 0.046, 0.050, 0.0, 0.1, 0.25, 0.0, 0.0, 0.0],
        [0.1, 0.0460, 0.046, 0.050, 0.0, -0.1, 0.25, 0.0, 0.0, 0.0],
        [0.1, 0.0460, 0.046, 0.050, -0.08, -0.605, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0230, 0.023, 0.020, 0.0, -0.606, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0230, 0.046, 0.020, 0.06, -0.605, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _euler_matrix(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:
    """Compute a Z-X-Z Euler rotation matrix in degrees."""
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    psi = np.deg2rad(psi_deg)
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cpsi * cphi - ctheta * sphi * spsi, cpsi * sphi + ctheta * cphi * spsi, spsi * stheta],
            [-spsi * cphi - ctheta * sphi * cpsi, -spsi * sphi + ctheta * cphi * cpsi, cpsi * stheta],
            [stheta * sphi, -stheta * cphi, ctheta],
        ],
        dtype=np.float64,
    )


def build_phantom(n: int = 64, kind: str = "modified") -> np.ndarray:
    """Build an n x n x n Modified Shepp-Logan phantom on [-1, 1]^3."""
    if kind != "modified":
        raise ValueError(f'Unknown phantom kind "{kind}"; use "modified".')
    if n < 2:
        raise ValueError("n must be >= 2")

    rng = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) / ((n - 1) / 2.0)
    x, y, z = np.meshgrid(rng, rng, rng, indexing="ij")
    coord = np.vstack((x.ravel(), y.ravel(), z.ravel()))

    volume = np.zeros(n * n * n, dtype=np.float64)
    for row in MODIFIED_SHEPP_LOGAN:
        a, b, c = row[1], row[2], row[3]
        x0, y0, z0 = row[4], row[5], row[6]
        asq, bsq, csq = a * a, b * b, c * c

        alpha = _euler_matrix(row[7], row[8], row[9])
        coordp = alpha @ coord

        inside = (
            (coordp[0] - x0) ** 2 / asq
            + (coordp[1] - y0) ** 2 / bsq
            + (coordp[2] - z0) ** 2 / csq
        ) <= 1.0
        volume[inside] += row[0]

    return volume.reshape((n, n, n))


# -----------------------------
# Siddon geometry and integral
# -----------------------------


@dataclass(frozen=True)
class VoxelGrid:
    """Voxel grid aligned with phantom sampling on [-1, 1]^3."""

    n: int

    @property
    def spacing(self) -> tuple[float, float, float]:
        if self.n < 2:
            return (2.0, 2.0, 2.0)
        d = 2.0 / (self.n - 1)
        return (d, d, d)

    @property
    def origin(self) -> tuple[float, float, float]:
        if self.n < 2:
            return (-1.0, -1.0, -1.0)
        d = 2.0 / (self.n - 1)
        return (-1.0 - d / 2.0, -1.0 - d / 2.0, -1.0 - d / 2.0)

    def plane_positions(self, axis: int) -> np.ndarray:
        d = self.spacing[axis]
        o = self.origin[axis]
        return o + np.arange(self.n + 1, dtype=np.float64) * d

    def index_at_midpoint(self, point: np.ndarray) -> tuple[int, int, int] | None:
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


def _alphas_for_axis(s: np.ndarray, p: np.ndarray, planes: np.ndarray, axis: int) -> np.ndarray:
    delta = p[axis] - s[axis]
    if abs(delta) < _EPS:
        return np.array([], dtype=np.float64)
    out = (planes - s[axis]) / delta
    return out[(out >= 0.0) & (out <= 1.0)]


def ray_enter_exit(s: np.ndarray, p: np.ndarray, grid: VoxelGrid) -> tuple[float, float] | None:
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


def line_integral_overhead(
    s_z: float,
    p: np.ndarray,
    volume: np.ndarray,
    grid: VoxelGrid | None = None,
) -> float:
    s = np.asarray([0.0, 0.0, float(s_z)], dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return line_integral_siddon(s, p, volume, grid)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-ray line integral (source overhead)")
    parser.add_argument("--s_z", type=float, default=3.0, help="source z coordinate (source at (0,0,s_z))")
    parser.add_argument("--p", type=float, nargs=3, required=True, help="detector point coordinates x y z")
    parser.add_argument("--n", type=int, default=64, help="volume size (n x n x n) if not loading volume")
    parser.add_argument("--value", type=float, default=1.0, help="uniform voxel value when creating a test volume")
    parser.add_argument("--load-volume", type=Path, default=None, help=".npy file path to load a 3D volume")
    parser.add_argument(
        "--use-phantom",
        action="store_true",
        help="generate Modified Shepp-Logan phantom with build_phantom(n) and use it",
    )
    return parser.parse_args(argv)


def load_or_create_volume(path: Path | None, n: int, value: float, use_phantom: bool) -> np.ndarray:
    """Load a 3D volume from file, or create a uniform volume, or generate phantom.

    Priority: if path provided -> load .npy; elif use_phantom -> build_phantom(n); else -> uniform volume.
    """
    if path is not None:
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError("Loaded array is not 3D")
        return arr.astype(np.float64)
    if use_phantom:
        return build_phantom(n)
    return np.full((n, n, n), float(value), dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        t0 = perf_counter()
        vol = load_or_create_volume(args.load_volume, args.n, args.value, args.use_phantom)
        t1 = perf_counter()
    except Exception as e:
        print("Error loading/creating volume:", e, file=sys.stderr)
        return 2

    p = np.asarray(args.p, dtype=np.float64)
    # optional: build a VoxelGrid matching volume
    grid = VoxelGrid(vol.shape[0])

    t_start_compute = perf_counter()
    val = line_integral_overhead(args.s_z, p, vol, grid)
    t_end_compute = perf_counter()

    print(f"line integral = {val}")
    print(f"volume load/generate time: {t1 - t0:.4f} s")
    print(f"line integral compute time: {t_end_compute - t_start_compute:.4f} s")
    print(f"total elapsed: {t_end_compute - t0:.4f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
