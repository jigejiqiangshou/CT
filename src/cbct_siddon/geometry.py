"""Cone-beam CT geometry: finite source and flat detector.

中文说明：
- 实现圆形轨迹的圆锥束（cone-beam）几何：源点绕 z=0 平面上原点做圆周运动。
- 检测器为平面（flat detector），以源-等中心线为中心，局部 u 轴沿横向，v 轴沿 z 轴。
- 角度采样为 [0, 2π)（默认不包含端点），因此为 360° 扫描。
"""

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
        """返回视角数组：从 0 到 2π 等间隔采样（endpoint=False 表示不包含 2π）。

        说明：此处生成的角度覆盖完整一圈（360°），若需 180° 扫描可改为 np.linspace(0, np.pi, ...)
        """
        return np.linspace(0.0, 2.0 * np.pi, self.n_angles, endpoint=False)

    def voxel_grid(self) -> VoxelGrid:
        """构建与重建体素体积对应的 VoxelGrid（供 Siddon 投影使用）。"""
        return VoxelGrid(self.volume_n)

    def source_position(self, beta: float) -> np.ndarray:
        """给定视角 beta，返回源点在三维空间中的坐标。

        实现细节：源点沿以原点为中心半径为 dso 的圆周运动，坐标采用 (dso*sin(beta), dso*cos(beta), 0)。
        注意：beta=0 时源位于 y 轴正方向；随 beta 增大源逆时针旋转。
        """
        return np.array(
            [self.dso * np.sin(beta), self.dso * np.cos(beta), 0.0],
            dtype=np.float64,
        )

    def detector_pixel(self, beta: float, u: float, v: float) -> np.ndarray:
        """给定视角 beta 与检测器局部坐标 (u, v)，返回检测器上对应的 3D 点。

        计算步骤：
        1. 计算当前视角下的源点 s。
        2. 计算从源指向等中心（原点）的单位向量 u_iso（检测器中心的方向）。
        3. 构造检测器局部坐标轴：u_hat（横向，旋转相关）和 v_hat（竖直，沿 z 轴）。
        4. 计算检测器中心 det_center = s + dsd * u_iso（从源出发沿等中心方向到达检测器平面）。
        5. 返回 det_center + u * u_hat + v * v_hat（在检测器平面上偏移到像素位置）。
        """
        s = self.source_position(beta)
        # 指向等中心的单位向量（从源指向原点）
        u_iso = -s / np.linalg.norm(s)
        # 检测器在当前视角下的局部 u, v 轴（u 随视角旋转，v 固定为 z 方向）
        u_hat = np.array([np.cos(beta), -np.sin(beta), 0.0], dtype=np.float64)
        v_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        det_center = s + self.dsd * u_iso
        return det_center + u * u_hat + v * v_hat

    def u_indices(self) -> np.ndarray:
        """返回检测器 u 方向上每个像素中心在物理坐标系下的 u 值（以中心为原点对称分布）。"""
        n = self.det_nu
        return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * self.du

    def v_indices(self) -> np.ndarray:
        """返回检测器 v 方向上每个像素中心在物理坐标系下的 v 值（以中心为原点对称分布）。"""
        n = self.det_nv
        return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * self.dv
