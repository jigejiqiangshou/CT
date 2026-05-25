"""Cone-beam CT geometry: finite source and flat detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cbct_siddon.siddon import VoxelGrid


@dataclass
class ConeBeamGeometry:
    """
    Circular cone-beam scan in the x-y plane.

    Source rotates at distance ``dso`` from origin; flat detector at distance
    ``dsd`` from source, centered on the line from source through origin.
    Detector u along local x, v along z.
    """

    dso: float = 2.0
    dsd: float = 4.0
    det_nu: int = 64
    det_nv: int = 64
    du: float = 0.05
    dv: float = 0.05
    n_angles: int = 36
    volume_n: int = 64

    def angles(self) -> np.ndarray:
        return np.linspace(0.0, 2.0 * np.pi, self.n_angles, endpoint=False)

    def voxel_grid(self) -> VoxelGrid:
        return VoxelGrid(self.volume_n)

    def source_position(self, beta: float) -> np.ndarray:
        return np.array(
            [self.dso * np.sin(beta), self.dso * np.cos(beta), 0.0],
            dtype=np.float64,
        )

    def detector_pixel(self, beta: float, u: float, v: float) -> np.ndarray:
        """3D point on detector for pixel (u, v) at view angle beta."""
        s = self.source_position(beta)
        # Unit vector from source toward isocenter
        u_iso = -s / np.linalg.norm(s)
        # Detector axes at view beta
        u_hat = np.array([np.cos(beta), -np.sin(beta), 0.0], dtype=np.float64)
        v_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        det_center = s + self.dsd * u_iso
        return det_center + u * u_hat + v * v_hat

    def u_indices(self) -> np.ndarray:
        n = self.det_nu
        return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * self.du

    def v_indices(self) -> np.ndarray:
        n = self.det_nv
        return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * self.dv
