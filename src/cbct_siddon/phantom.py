"""3D Modified Shepp-Logan phantom (Toft / phantom3d convention).

本模块生成与 MATLAB `phantom3d.m` 兼容的 Modified Shepp–Logan 三维模体。
主要功能：提供椭球参数表、计算 Z-X-Z Euler 旋转矩阵、以及在 [-1,1]^3 上构建体积数据。
"""

from __future__ import annotations

import numpy as np

# 椭球参数表（10×10），每行表示一个椭球：
# 列依次为：A, a, b, c, x0, y0, z0, phi, theta, psi（角度单位：度）
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
    """返回参数表的副本（10×10），以防外部修改原表。

    返回值为 `numpy.ndarray`，dtype 为 `float64`。
    """
    return MODIFIED_SHEPP_LOGAN.copy()


def _euler_matrix(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:
    """根据 Z-X-Z Euler 角计算 3×3 旋转矩阵（角度单位：度）。

    说明：此旋转矩阵与 MATLAB 的 `phantom3d.m` 保持一致，用于将全局坐标旋转到
    椭球的局部坐标系，从而支持椭球的方向和倾斜。
    """
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    psi = np.deg2rad(psi_deg)
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    # 返回显式展开的矩阵元素，便于审阅和与 MATLAB 对齐
    return np.array(
        [
            [cpsi * cphi - ctheta * sphi * spsi, cpsi * sphi + ctheta * cphi * spsi, spsi * stheta],
            [-spsi * cphi - ctheta * sphi * cpsi, -spsi * sphi + ctheta * cphi * cpsi, cpsi * stheta],
            [stheta * sphi, -stheta * cphi, ctheta],
        ],
        dtype=np.float64,
    )


def build_phantom(n: int = 64, kind: str = "modified") -> np.ndarray:
    """在 [-1, 1]^3 上构建 n×n×n 的 Modified Shepp–Logan 体积模体。

    算法概述：
    - 将索引 0..n-1 映射到 [-1, 1] 上的 voxel center 坐标（与 MATLAB phantom3d 一致）；
    - 对参数表中每个椭球：构造旋转矩阵，将所有 voxel center 旋转到椭球参考系；
    - 判断哪些点落在椭球内部，对应位置累加该椭球的振幅 A；
    - 最后 reshape 回 (n, n, n) 并返回。

    参数
    ------
    n : int
        每轴体素数（必须 >= 2）。
    kind : str
        当前仅支持 "modified"（Modified Shepp-Logan）。

    返回
    ------
    volume : ndarray
        大小为 (n, n, n) 的体积数组，dtype=float64。
    """
    if kind != "modified":
        raise ValueError(f'Unknown phantom kind "{kind}"; use "modified".')
    if n < 2:
        raise ValueError("n must be >= 2")

    ellipsoids = MODIFIED_SHEPP_LOGAN

    # 将网格索引映射到 [-1, 1]，使样本点位于 voxel center
    rng = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) / ((n - 1) / 2.0)
    x, y, z = np.meshgrid(rng, rng, rng, indexing="ij")
    coord = np.vstack((x.ravel(), y.ravel(), z.ravel()))  # 3 × (n^3)

    volume = np.zeros(n * n * n, dtype=np.float64)

    # 遍历每个椭球条目并将其贡献累加到体积上
    for row in ellipsoids:
        # 半轴长度与中心偏移
        a, b, c = row[1], row[2], row[3]
        x0, y0, z0 = row[4], row[5], row[6]
        asq, bsq, csq = a * a, b * b, c * c

        # 生成椭球的旋转矩阵并将坐标转换到椭球参考系
        alpha = _euler_matrix(row[7], row[8], row[9])
        coordp = alpha @ coord

        # 椭球内部判定：((x-x0)/a)^2 + ((y-y0)/b)^2 + ((z-z0)/c)^2 <= 1
        inside = (
            (coordp[0] - x0) ** 2 / asq
            + (coordp[1] - y0) ** 2 / bsq
            + (coordp[2] - z0) ** 2 / csq
        ) <= 1.0

        # 对落在椭球内的体素累加振幅 A
        volume[inside] += row[0]

    return volume.reshape((n, n, n))
