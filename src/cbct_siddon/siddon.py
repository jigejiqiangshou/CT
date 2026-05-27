"""Siddon (1985) exact radiological path through a 3D voxel array.

中文注释说明（对应论文实现步骤，未引用论文原文）：
- 步骤1：定义体素网格（VoxelGrid），包括体素间距、起点(origin)和各轴平面位置。
- 步骤2：计算射线与体积的进入/退出参数(alpha)：ray_enter_exit。
- 步骤3：对每个轴，计算射线与该轴平面相交的参数集合(_alphas_for_axis)。
- 步骤4：合并并去重所有alpha值，得到射线在体积内的分段边界(_merge_alphas)。
- 步骤5：对每段区间取中点，确定所属体素，计算线积分贡献并累加(line_integral_siddon)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class VoxelGrid:
    """
    Voxel grid aligned with phantom3d.m sampling on [-1, 1]^3.

    Plane boundaries: -1 - dx/2 + k*dx for k = 0..n (n voxels).

    中文说明：
    - 本实现假定体素网格与 phantom3d.m 的采样在 [-1,1]^3 区域对齐。
    - `n` 表示每轴上的体素数（各轴等长，每轴体素数相同）。
    """

    n: int

    @property
    def spacing(self) -> tuple[float, float, float]:
        # 体素间距：对于 n<2 的退化情况，返回全域跨度 2.0
        if self.n < 2:
            return (2.0, 2.0, 2.0)
        d = 2.0 / (self.n - 1)
        return (d, d, d)

    @property
    def origin(self) -> tuple[float, float, float]:
        """Position of the first x/y/z plane (left boundary of voxel 0)."""
        # 原点（第一个平面的位置），注意这里使用左边界（voxel 0 的左侧）
        if self.n < 2:
            return (-1.0, -1.0, -1.0)
        d = 2.0 / (self.n - 1)
        return (-1.0 - d / 2.0, -1.0 - d / 2.0, -1.0 - d / 2.0)

    def plane_positions(self, axis: int) -> np.ndarray:
        """n+1 plane coordinates along axis (0=x, 1=y, 2=z)."""
        # 返回沿指定轴的所有平面坐标（有 n+1 个平面，分割 n 个体素）
        d = self.spacing[axis]
        o = self.origin[axis]
        return o + np.arange(self.n + 1, dtype=np.float64) * d

    def index_at_midpoint(self, point: np.ndarray) -> tuple[int, int, int] | None:
        """Voxel indices for a point inside the volume, else None."""
        # 对于分段中点，计算其所属体素的索引（若不在体积内则返回 None）
        d = self.spacing
        o = self.origin
        idx = []
        for ax in range(3):
            # 将坐标转换到以平面索引为单位的参数 t
            t = (point[ax] - o[ax]) / d[ax]
            # 若超出范围，返回 None（该点不在体积内）
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
    # 计算射线 s->p 与指定轴上每个平面的相交参数 alpha
    # alpha 定义为 s + alpha*(p-s)，取值范围 [0,1]
    delta = p[axis] - s[axis]
    # 如果在该轴上没有分量（平行于轴平面），则不会与这些平面相交
    if abs(delta) < _EPS:
        return np.array([], dtype=np.float64)
    out = (planes - s[axis]) / delta
    # 只保留在射线段 [0,1] 范围内的交点
    return out[(out >= 0.0) & (out <= 1.0)]


def ray_enter_exit(
    s: np.ndarray,
    p: np.ndarray,
    grid: VoxelGrid,
) -> tuple[float, float] | None:
    """Return (alpha_min, alpha_max) where ray meets the volume, or None."""
    # 计算射线与体积包围盒（由每轴的最小/最大平面定义）的进入/退出 alpha
    # 初始考虑整条射线区间 [0,1]
    alphas = [0.0, 1.0]
    for ax in range(3):
        planes = grid.plane_positions(ax)
        delta = p[ax] - s[ax]
        # 若在该轴方向没有分量，检查射线在该轴是否位于包围盒内
        if abs(delta) < _EPS:
            if s[ax] < planes[0] or s[ax] > planes[-1]:
                # 射线平行于平面且在外部 -> 不相交
                return None
            continue
        # 计算射线与该轴最小/最大平面对应的 alpha
        a0 = (planes[0] - s[ax]) / delta
        a1 = (planes[-1] - s[ax]) / delta
        alphas.extend([min(a0, a1), max(a0, a1)])

    # 射线进入和退出体积的 alpha 值（截取到 [0,1]）
    alpha_min = max(0.0, min(alphas))
    alpha_max = min(1.0, max(alphas))
    # 若没有有效区间则不相交
    if alpha_min >= alpha_max - _EPS:
        return None
    return alpha_min, alpha_max


def _merge_alphas(alphas: np.ndarray) -> np.ndarray:
    """Sort and remove duplicate alpha values (ray through edges/corners)."""
    # 将所有 alpha 排序并去除非常接近的重复值（处理穿过体素边界/角落的情况）
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
    # 准备向量并构建默认网格（如果未提供）
    s = np.asarray(s, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if grid is None:
        grid = VoxelGrid(volume.shape[0])

    # 步骤2：计算射线与体积的进入/退出 alpha
    bounds = ray_enter_exit(s, p, grid)
    if bounds is None:
        return 0.0
    alpha_min, alpha_max = bounds

    # 步骤3：收集所有轴上与平面的交点 alpha（包括进入/退出）
    alphas = [alpha_min, alpha_max]
    for ax in range(3):
        alphas.extend(_alphas_for_axis(s, p, grid.plane_positions(ax), ax).tolist())

    # 步骤4：排序并去重 alpha，得到射线在体积内的分段边界
    alphas = _merge_alphas(np.asarray(alphas, dtype=np.float64))
    if alphas.size < 2:
        return 0.0

    # 射线真实长度与方向
    ray_len = float(np.linalg.norm(p - s))
    direction = p - s
    total = 0.0

    # 步骤5：遍历每个相交区间，取中点判断所属体素，累加长度*体素值
    for m in range(alphas.size - 1):
        a0, a1 = alphas[m], alphas[m + 1]
        d_alpha = a1 - a0
        if d_alpha <= _EPS:
            continue
        # 区间中点对应射线上的位置，用于确定体素索引（射线与体素边界相交时取中点避免模糊）
        alpha_mid = 0.5 * (a0 + a1)
        point = s + alpha_mid * direction
        idx = grid.index_at_midpoint(point)
        if idx is None:
            continue
        i, j, k = idx
        # 该区间在物理空间中的长度
        ell = ray_len * d_alpha
        total += ell * volume[i, j, k]

    return total


def line_integral_overhead(
    s_z: float,
    p: np.ndarray,
    volume: np.ndarray,
    grid: VoxelGrid | None = None,
) -> float:
    """Compute line integral for a single ray with source directly above the volume center.

    参数（中文）：
    - s_z: 源的 z 坐标（源点位于 (0, 0, s_z)）
    - p: 探测器像素在三维空间的坐标（长度为 3 的数组）
    - volume: 三维体素数组，形状 (n, n, n)
    - grid: 可选的 `VoxelGrid`，若未提供将基于 `volume` 大小构建

    返回值：该射线穿过体积的线积分（float）。
    """
    s = np.asarray([0.0, 0.0, float(s_z)], dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return line_integral_siddon(s, p, volume, grid)
