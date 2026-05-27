"""Siddon 线性缩放实验：验证单射线积分的 O(N) 行为。

该脚本是自包含版本，包含模体生成、体素网格和 Siddon 线积分实现，
便于直接运行并复现实验流程。
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
# 模体生成
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
    # Z-X-Z 欧拉角旋转矩阵（角度单位：度）
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
    # 在 [-1, 1]^3 上生成 Modified Shepp-Logan 体模
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
# Siddon 几何与线积分
# -----------------------------


@dataclass(frozen=True)
class VoxelGrid:
    """与体模采样对齐的体素网格（坐标范围 [-1, 1]^3）。"""

    n: int

    @property
    def spacing(self) -> tuple[float, float, float]:
        # 体素间距
        if self.n < 2:
            return (2.0, 2.0, 2.0)
        d = 2.0 / (self.n - 1)
        return (d, d, d)

    @property
    def origin(self) -> tuple[float, float, float]:
        # 第一个平面的坐标（体素 0 的左边界）
        if self.n < 2:
            return (-1.0, -1.0, -1.0)
        d = 2.0 / (self.n - 1)
        return (-1.0 - d / 2.0, -1.0 - d / 2.0, -1.0 - d / 2.0)

    def plane_positions(self, axis: int) -> np.ndarray:
        # 沿指定轴的 n+1 个平面坐标
        d = self.spacing[axis]
        o = self.origin[axis]
        return o + np.arange(self.n + 1, dtype=np.float64) * d

    def index_at_midpoint(self, point: np.ndarray) -> tuple[int, int, int] | None:
        # 根据区间中点确定其所在体素索引
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
    # 射线与指定轴平面的交点参数 alpha（仅保留 [0,1]）
    delta = p[axis] - s[axis]
    if abs(delta) < _EPS:
        return np.array([], dtype=np.float64)
    out = (planes - s[axis]) / delta
    return out[(out >= 0.0) & (out <= 1.0)]


def ray_enter_exit(s: np.ndarray, p: np.ndarray, grid: VoxelGrid) -> tuple[float, float] | None:
    # 计算射线进入/离开体素包围盒的 alpha 范围
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
    # 排序并去除重复 alpha（穿过边/角的情况）
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
    # Siddon 算法：沿射线分段累加体素值 * 路径长度
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
# Experiment
# -----------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # 命令行参数：N 列表、终点网格密度、源位置等
    parser = argparse.ArgumentParser(description="Siddon scaling experiment (single-ray O(N)).")
    parser.add_argument(
        "--n-vals",
        type=str,
        default="20,40,60,80,100",
        help="comma-separated N values (default: 20,40,60,80,100)",
    )
    parser.add_argument("--grid-size", type=int, default=21, help="number of points per axis (default: 21)")
    parser.add_argument("--s-z", type=float, default=3.0, help="source z coordinate (default: 3.0)")
    parser.add_argument("--fit", action="store_true", help="print linear fit slope/intercept")
    return parser.parse_args(argv)


def _parse_n_vals(text: str) -> list[int]:
    # 解析逗号分隔的 N 列表
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("--n-vals is empty")
    return out


def main(argv: list[str] | None = None) -> int:
    # 主流程：生成终点网格 -> 遍历 N -> 逐射线计时 -> 输出与绘图
    args = parse_args(argv)
    n_vals = _parse_n_vals(args.n_vals)
    grid_size = int(args.grid_size)
    s_z = float(args.s_z)

    # 终点网格（不计入计时）
    lin = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    endpoints = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    total_rays = endpoints.shape[0]

    rows: list[tuple[int, int, int, float, float]] = []
    s = np.asarray([0.0, 0.0, s_z], dtype=np.float64)

    for n in n_vals:
        # 构建体模与网格（不计入计时）
        volume = build_phantom(n)
        grid = VoxelGrid(n)

        valid_count = 0
        total_time = 0.0
        for p in endpoints:
            # 仅计时 Siddon 单次调用
            t0 = perf_counter()
            val = line_integral_siddon(s, p, volume, grid)
            t1 = perf_counter()
            total_time += t1 - t0
            if val != 0.0:
                valid_count += 1

        avg_ms = (total_time / total_rays) * 1000.0
        rows.append((n, total_rays, valid_count, total_time, avg_ms))

    header = "N  total_rays  valid_rays  total_time(s)  avg_time(ms)"
    print(header)
    print("-" * len(header))
    for n, total_rays, valid_rays, total_time, avg_ms in rows:
        print(f"{n:>3}  {total_rays:>10}  {valid_rays:>10}  {total_time:>13.6f}  {avg_ms:>12.6f}")

    ns = np.array([r[0] for r in rows], dtype=np.float64)
    avg_ms = np.array([r[4] for r in rows], dtype=np.float64)

    # 过原点参考线：y = c * N
    c = float(np.dot(ns, avg_ms) / np.dot(ns, ns))
    ns_ref = np.concatenate(([0.0], ns))
    ref = c * ns_ref

    plt.figure(figsize=(6, 4))
    plt.plot(ns, avg_ms, "o-", label="avg time (ms)")
    plt.plot(ns_ref, ref, "--", label=f"y = {c:.4f} * N")
    plt.plot([0.0], [0.0], "o", label="origin")
    plt.xlabel("N (volume size)")
    plt.ylabel("avg time per ray (ms)")
    plt.title("Siddon scaling experiment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(left=0.0)

    # 保存图像到 outputs
    out_path = Path("outputs") / "siddon_scaling.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")

    if args.fit:
        m, b = np.polyfit(ns, avg_ms, 1)
        print(f"linear fit: slope={m:.6f} ms/N, intercept={b:.6f} ms")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
