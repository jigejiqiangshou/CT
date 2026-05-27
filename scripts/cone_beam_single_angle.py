"""Single-angle cone-beam projection using Siddon line integrals.

Self-contained script: phantom generation, voxel grid, Siddon tracing, and
single-angle cone-beam projection.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

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


def build_phantom(n: int = 256, kind: str = "modified") -> np.ndarray:
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


# -----------------------------
# Single-angle cone-beam
# -----------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-angle cone-beam projection.")
    parser.add_argument("--n", type=int, default=64, help="volume size (default: 64)")
    parser.add_argument("--s-z", type=float, default=3.0, help="source z coordinate (default: 3.0)")
    parser.add_argument("--det-rows", type=int, default=256
    , help="detector rows (default: 256)")
    parser.add_argument("--det-cols", type=int, default=256, help="detector cols (default: 256)")
    parser.add_argument("--det-pixel-size", type=float, default=0.01, help="detector pixel size (default: 0.01)")
    parser.add_argument("--det-z", type=float, default=-1.0, help="detector center z (default: -1.0)")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs") / "cone_beam_projection.png",
        help="output PNG path (default: outputs/cone_beam_projection.png)",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=Path("outputs") / "cone_beam_projection.npy",
        help="output NPY path (default: outputs/cone_beam_projection.npy)",
    )
    parser.add_argument(
        "--auto-window",
        action="store_true",
        help="use percentile-based windowing for display/save",
    )
    parser.add_argument(
        "--log-display",
        action="store_true",
        help="apply log1p transform for display/save",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    volume = build_phantom(args.n)
    grid = VoxelGrid(args.n)

    source = np.asarray([0.0, 0.0, float(args.s_z)], dtype=np.float64)

    cols = int(args.det_cols)
    rows = int(args.det_rows)
    pixel_size = float(args.det_pixel_size)
    det_z = float(args.det_z)
    if det_z >= 1.0:
        print("warning: det_z >= 1.0 may miss the phantom (expect near-zero projection)")

    u = (np.arange(cols, dtype=np.float64) - (cols - 1) / 2.0) * pixel_size
    v = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2.0) * pixel_size
    uu, vv = np.meshgrid(u, v, indexing="xy")

    det_x = uu
    det_y = vv
    det_z_grid = np.full_like(det_x, det_z)

    projection = np.zeros((rows, cols), dtype=np.float64)

    t0 = perf_counter()
    for r in range(rows):
        for c in range(cols):
            pixel_center = np.array([det_x[r, c], det_y[r, c], det_z_grid[r, c]], dtype=np.float64)
            projection[r, c] = line_integral_siddon(source, pixel_center, volume, grid)
    t1 = perf_counter()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)

    np.save(args.output_npy, projection)

    display = projection
    if args.log_display:
        display = np.log1p(display)

    vmin = None
    vmax = None
    if args.auto_window:
        vmin, vmax = np.percentile(display, [5, 95])

    plt.figure(figsize=(6, 6))
    plt.imshow(display, cmap="gray", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Cone-beam projection (single angle)")
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=200)
    plt.show()

    print(
        f"projection stats: min={projection.min():.6f}, max={projection.max():.6f}, mean={projection.mean():.6f}"
    )
    print(f"projection saved: {args.output_npy}")
    print(f"image saved: {args.output_png}")
    print(f"total projection time: {t1 - t0:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
