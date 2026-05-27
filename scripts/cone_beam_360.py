#!/usr/bin/env python3
"""360° cone-beam forward projection using Siddon's algorithm.

This script generates a set of projections (sinogram) for a given phantom
by rotating the source and detector around the Y axis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

_EPS = 1e-12


# ----------------------------------------------------------------------
# Phantom generation (exactly the same as in the original script)
# ----------------------------------------------------------------------

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


def build_phantom(n: int = 64, kind: str = "modified") -> np.ndarray:
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


# ----------------------------------------------------------------------
# Siddon geometry and integral (unchanged)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class VoxelGrid:
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


# ----------------------------------------------------------------------
# Cone‑beam geometry helpers
# ----------------------------------------------------------------------

def rotation_matrix_y(theta_deg: float) -> np.ndarray:
    """3x3 rotation matrix around Y axis (right‑handed)."""
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def cone_beam_pixel_coordinates(
    theta_deg: float,
    sad: float,
    oad: float,
    rows: int,
    cols: int,
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate world coordinates of all detector pixel centers for a given angle.

    Returns:
        pixel_centers: (rows*cols, 3) array
        source: (3,) array
        det_center: (3,) array
    """
    # source and detector center at angle 0
    source0 = np.array([sad, 0.0, 0.0], dtype=np.float64)
    det_center0 = np.array([-oad, 0.0, 0.0], dtype=np.float64)
    # direction from source to detector center (central ray)
    dir0 = det_center0 - source0
    dir0_norm = dir0 / np.linalg.norm(dir0)   # (-1,0,0)

    # For angle 0, we define:
    #   u axis (row direction) = world Y (0,1,0)
    #   v axis (col direction) = direction × u  (cross product)
    # This ensures detector plane is perpendicular to the central ray.
    u_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v_axis = np.cross(dir0_norm, u_axis)   # (0,0,-1) for angle 0
    v_axis = v_axis / np.linalg.norm(v_axis)

    # Pixel offsets in local detector coordinates
    u_offsets = (np.arange(rows) - (rows - 1) / 2.0) * pixel_size
    v_offsets = (np.arange(cols) - (cols - 1) / 2.0) * pixel_size
    uu, vv = np.meshgrid(u_offsets, v_offsets, indexing='ij')
    # local offset vectors
    offsets = uu[..., np.newaxis] * u_axis + vv[..., np.newaxis] * v_axis
    # shape (rows, cols, 3)

    # Pixel centers at angle 0
    pixels0 = det_center0 + offsets.reshape(-1, 3)

    # Rotate everything by theta around Y axis
    R = rotation_matrix_y(theta_deg)
    source = R @ source0
    det_center = R @ det_center0
    pixels = (R @ pixels0.T).T

    return pixels, source, det_center


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="360° cone‑beam forward projection using Siddon")
    parser.add_argument("--n", type=int, default=64, help="Phantom size (N x N x N)")
    parser.add_argument("--sad", type=float, default=3.0, help="Source to axis distance")
    parser.add_argument("--oad", type=float, default=1.0, help="Object to detector distance")
    parser.add_argument("--rows", type=int, default=256, help="Detector rows (u direction, along Y)")
    parser.add_argument("--cols", type=int, default=256, help="Detector columns (v direction)")
    parser.add_argument("--pixel-size", type=float, default=0.01, help="Pixel pitch (world units)")
    parser.add_argument("--start-angle", type=float, default=0.0, help="Start angle (deg)")
    parser.add_argument("--end-angle", type=float, default=360.0, help="End angle (deg, exclusive if step aligns)")
    parser.add_argument("--step", type=float, default=1.0, help="Angular step (deg)")
    parser.add_argument("--output-npy", type=Path, default=Path("outputs/projections_360.npy"),
                        help="Output .npy file path")
    parser.add_argument("--preview", action="store_true", help="Show preview of 0°, 90°, 180°, 270° after computation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Create output directory
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)

    print("Generating phantom ...")
    volume = build_phantom(args.n)
    grid = VoxelGrid(args.n)
    print(f"Phantom size: {args.n}^3 voxels")

    angles = np.arange(args.start_angle, args.end_angle, args.step, dtype=np.float64)
    n_angles = len(angles)
    print(f"Angles: {n_angles} from {args.start_angle}° to {args.end_angle - args.step}° (step {args.step}°)")
    print(f"Detector: {args.rows} x {args.cols}, pixel size = {args.pixel_size}")
    print("Starting forward projection (this may take a while) ...")

    # Pre‑allocate projection array (angles, rows, cols)
    projections = np.zeros((n_angles, args.rows, args.cols), dtype=np.float32)

    total_rays = n_angles * args.rows * args.cols
    print(f"Total rays to compute: {total_rays}")

    # Optional progress bar (tqdm if installed)
    try:
        from tqdm import tqdm
        angle_iter = tqdm(angles, desc="Angles")
    except ImportError:
        angle_iter = angles
        print("Tip: install tqdm for a progress bar.")

    start_time = perf_counter()

    for idx, theta in enumerate(angle_iter):
        if angle_iter is angles:
            print(f"  Angle {theta:.1f}° ({idx+1}/{n_angles})")

        # Get pixel world coordinates for this angle
        pixels, source, _ = cone_beam_pixel_coordinates(
            theta, args.sad, args.oad, args.rows, args.cols, args.pixel_size
        )

        proj_2d = np.zeros((args.rows, args.cols), dtype=np.float32)
        # Loop over all detector pixels
        for r in range(args.rows):
            for c in range(args.cols):
                pidx = r * args.cols + c
                val = line_integral_siddon(source, pixels[pidx], volume, grid)
                proj_2d[r, c] = val
        projections[idx] = proj_2d

        # tqdm handles progress display; no extra work needed

    elapsed = perf_counter() - start_time
    print(f"Total computation time: {elapsed:.2f} s")

    # Save to .npy
    np.save(args.output_npy, projections)
    print(f"Saved projections to {args.output_npy} (shape {projections.shape})")

    # Optional preview
    if args.preview:
        # Show projections closest to 0°, 90°, 180°, 270°
        if n_angles == 0:
            print("No angles available for preview.")
        else:
            targets = [0.0, 90.0, 180.0, 270.0]
            idxs = [int(np.argmin(np.abs(angles - t))) for t in targets]
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for ax, idx, title in zip(axes.flat, idxs, ["0°", "90°", "180°", "270°"]):
                im = ax.imshow(projections[idx], cmap="gray", origin="upper")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            preview_path = args.output_npy.parent / "preview_angles.png"
            plt.savefig(preview_path, dpi=150)
            print(f"Preview image saved to {preview_path}")
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())