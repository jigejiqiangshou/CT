"""Brute-force line integral for Siddon validation (tests only)."""

from __future__ import annotations

import numpy as np

from cbct_siddon.siddon import VoxelGrid, ray_enter_exit


def line_integral_bruteforce(
    s: np.ndarray,
    p: np.ndarray,
    volume: np.ndarray,
    grid: VoxelGrid,
    n_samples: int = 20000,
) -> float:
    """
    Sample the ray uniformly in alpha and sum voxel values weighted by segment length.
    """
    bounds = ray_enter_exit(s, p, grid)
    if bounds is None:
        return 0.0
    alpha_min, alpha_max = bounds
    s = np.asarray(s, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    direction = p - s
    ray_len = float(np.linalg.norm(direction))

    alphas = np.linspace(alpha_min, alpha_max, n_samples + 1)
    total = 0.0
    for m in range(n_samples):
        a0, a1 = alphas[m], alphas[m + 1]
        d_alpha = a1 - a0
        alpha_mid = 0.5 * (a0 + a1)
        point = s + alpha_mid * direction
        idx = grid.index_at_midpoint(point)
        if idx is None:
            continue
        i, j, k = idx
        total += ray_len * d_alpha * volume[i, j, k]

    return total
