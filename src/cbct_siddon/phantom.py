"""3D Modified Shepp-Logan phantom (Toft / phantom3d convention)."""

from __future__ import annotations

import numpy as np

# Columns: A, a, b, c, x0, y0, z0, phi, theta, psi (degrees)
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


def modified_shepp_logan_table() -> np.ndarray:
    """Return the 10x10 ellipsoid parameter table."""
    return MODIFIED_SHEPP_LOGAN.copy()


def _euler_matrix(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:
    """Rotation matrix matching phantom3d.m (Z-X-Z Euler)."""
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
    """
    Build an n×n×n phantom on [-1, 1]^3 (MATLAB phantom3d voxel centers).

    Parameters
    ----------
    n : int
        Grid size per axis.
    kind : str
        Only ``\"modified\"`` is supported (Modified Shepp-Logan).
    """
    if kind != "modified":
        raise ValueError(f'Unknown phantom kind "{kind}"; use "modified".')
    if n < 2:
        raise ValueError("n must be >= 2")

    ellipsoids = MODIFIED_SHEPP_LOGAN
    rng = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) / ((n - 1) / 2.0)
    x, y, z = np.meshgrid(rng, rng, rng, indexing="ij")
    coord = np.vstack((x.ravel(), y.ravel(), z.ravel()))

    volume = np.zeros(n * n * n, dtype=np.float64)
    for row in ellipsoids:
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
