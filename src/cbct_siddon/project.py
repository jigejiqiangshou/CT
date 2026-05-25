"""Cone-beam forward projection using Siddon ray tracing."""

from __future__ import annotations

import numpy as np

from cbct_siddon.geometry import ConeBeamGeometry
from cbct_siddon.siddon import VoxelGrid, line_integral_siddon


def forward_project(
    volume: np.ndarray,
    geom: ConeBeamGeometry,
    *,
    grid: VoxelGrid | None = None,
) -> np.ndarray:
    """
    Siddon cone-beam forward projection.

    Returns
    -------
    proj : ndarray, shape (n_angles, det_nv, det_nu)
    """
    if grid is None:
        grid = VoxelGrid(volume.shape[0])

    u_vals = geom.u_indices()
    v_vals = geom.v_indices()
    betas = geom.angles()
    proj = np.zeros((geom.n_angles, geom.det_nv, geom.det_nu), dtype=np.float64)

    for ia, beta in enumerate(betas):
        s = geom.source_position(beta)
        for iv, v in enumerate(v_vals):
            for iu, u in enumerate(u_vals):
                p = geom.detector_pixel(beta, u, v)
                proj[ia, iv, iu] = line_integral_siddon(s, p, volume, grid)

    return proj
