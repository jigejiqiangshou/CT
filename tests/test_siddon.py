import numpy as np

from cbct_siddon.siddon import VoxelGrid, line_integral_siddon

from tests._bruteforce import line_integral_bruteforce


def test_uniform_cube():
    n = 32
    c = 2.5
    volume = np.full((n, n, n), c, dtype=np.float64)
    grid = VoxelGrid(n)
    s = np.array([-3.0, 0.0, 0.0])
    p = np.array([3.0, 0.0, 0.0])
    val = line_integral_siddon(s, p, volume, grid)
    # Path along x through voxel grid extent (slightly wider than [-1, 1])
    dx = grid.spacing[0]
    path_len = 2.0 + dx
    assert np.isclose(val, c * path_len, rtol=1e-5)


def test_siddon_vs_bruteforce():
    rng = np.random.default_rng(42)
    n = 32
    volume = np.random.rand(n, n, n)
    grid = VoxelGrid(n)
    for _ in range(8):
        s = rng.uniform(-2.5, 2.5, 3)
        p = rng.uniform(-2.5, 2.5, 3)
        if np.linalg.norm(p - s) < 0.5:
            continue
        a = line_integral_siddon(s, p, volume, grid)
        b = line_integral_bruteforce(s, p, volume, grid, n_samples=80000)
        assert np.isclose(a, b, rtol=2e-3, atol=1e-4), f"siddon={a}, brute={b}"


def test_empty_ray():
    n = 32
    volume = np.ones((n, n, n))
    grid = VoxelGrid(n)
    s = np.array([0.0, 0.0, 5.0])
    p = np.array([0.0, 0.0, 6.0])
    assert line_integral_siddon(s, p, volume, grid) == 0.0


def test_axis_parallel():
    n = 32
    volume = np.random.rand(n, n, n)
    grid = VoxelGrid(n)
    s = np.array([-2.0, 0.0, 0.0])
    p = np.array([2.0, 0.0, 0.0])
    val = line_integral_siddon(s, p, volume, grid)
    assert np.isfinite(val)
