"""Siddon 线性缩放实验：验证单射线积分的 O(N) 行为。

该脚本是自包含版本，包含模体生成、体素网格和 Siddon 线积分实现，
便于直接运行并复现实验流程。
"""  # 脚本说明：验证单射线积分的线性缩放性质
from __future__ import annotations  # 启用延迟类型注解解析

import argparse  # 导入命令行参数解析模块
from dataclasses import dataclass  # 导入数据类装饰器
from pathlib import Path  # 导入路径处理类
from time import perf_counter  # 导入高精度计时函数

import numpy as np  # 导入 NumPy 数值计算库
import matplotlib.pyplot as plt  # 导入绘图接口

_EPS = 1e-12  # 定义数值比较阈值


# -----------------------------  # 模体生成部分分隔线
# 模体生成  # 本节负责构造三维体模
# -----------------------------  # 模体生成部分分隔线结束

MODIFIED_SHEPP_LOGAN: np.ndarray = np.array(  # 定义修改版 Shepp-Logan 参数表
    [  # 这里的每一行对应一个椭球体
        [1.0, 0.6900, 0.920, 0.810, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 主椭球
        [-0.8, 0.6624, 0.874, 0.780, 0.0, -0.0184, 0.0, 0.0, 0.0, 0.0],  # 负密度椭球
        [-0.2, 0.1100, 0.310, 0.220, 0.22, 0.0, 0.0, -18.0, 0.0, 10.0],  # 倾斜椭球 1
        [-0.2, 0.1600, 0.410, 0.280, -0.22, 0.0, 0.0, 18.0, 0.0, 10.0],  # 倾斜椭球 2
        [0.1, 0.2100, 0.250, 0.410, 0.0, 0.35, -0.15, 0.0, 0.0, 0.0],  # 上部结构
        [0.1, 0.0460, 0.046, 0.050, 0.0, 0.1, 0.25, 0.0, 0.0, 0.0],  # 小椭球 1
        [0.1, 0.0460, 0.046, 0.050, 0.0, -0.1, 0.25, 0.0, 0.0, 0.0],  # 小椭球 2
        [0.1, 0.0460, 0.046, 0.050, -0.08, -0.605, 0.0, 0.0, 0.0, 0.0],  # 侧边小椭球
        [0.1, 0.0230, 0.023, 0.020, 0.0, -0.606, 0.0, 0.0, 0.0, 0.0],  # 更小椭球 1
        [0.1, 0.0230, 0.046, 0.020, 0.06, -0.605, 0.0, 0.0, 0.0, 0.0],  # 更小椭球 2
    ],  # 参数表结束
    dtype=np.float64,  # 使用双精度浮点保存
)


def _euler_matrix(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:  # 计算 Z-X-Z 欧拉角旋转矩阵
    phi = np.deg2rad(phi_deg)  # 将 phi 从角度转为弧度
    theta = np.deg2rad(theta_deg)  # 将 theta 从角度转为弧度
    psi = np.deg2rad(psi_deg)  # 将 psi 从角度转为弧度
    cphi, sphi = np.cos(phi), np.sin(phi)  # 计算 phi 的余弦和正弦
    ctheta, stheta = np.cos(theta), np.sin(theta)  # 计算 theta 的余弦和正弦
    cpsi, spsi = np.cos(psi), np.sin(psi)  # 计算 psi 的余弦和正弦
    return np.array(  # 返回 3x3 旋转矩阵
        [  # 矩阵第一行到第三行
            [cpsi * cphi - ctheta * sphi * spsi, cpsi * sphi + ctheta * cphi * spsi, spsi * stheta],  # 第一行
            [-spsi * cphi - ctheta * sphi * cpsi, -spsi * sphi + ctheta * cphi * cpsi, cpsi * stheta],  # 第二行
            [stheta * sphi, -stheta * cphi, ctheta],  # 第三行
        ],  # 行列表结束
        dtype=np.float64,  # 输出为双精度矩阵
    )  # 旋转矩阵构造结束


def build_phantom(n: int = 64, kind: str = "modified") -> np.ndarray:  # 在 [-1,1]^3 上生成体模
    if kind != "modified":  # 检查体模类型是否支持
        raise ValueError(f'Unknown phantom kind "{kind}"; use "modified".')  # 不支持则报错
    if n < 2:  # 检查尺寸是否有效
        raise ValueError("n must be >= 2")  # 尺寸过小则报错

    rng = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) / ((n - 1) / 2.0)  # 生成归一化坐标轴
    x, y, z = np.meshgrid(rng, rng, rng, indexing="ij")  # 生成三维坐标网格
    coord = np.vstack((x.ravel(), y.ravel(), z.ravel()))  # 展平并堆叠为 3xN 坐标矩阵

    volume = np.zeros(n * n * n, dtype=np.float64)  # 初始化体模数组
    for row in MODIFIED_SHEPP_LOGAN:  # 遍历每个椭球参数行
        a, b, c = row[1], row[2], row[3]  # 读取椭球三轴长度
        x0, y0, z0 = row[4], row[5], row[6]  # 读取椭球中心偏移
        asq, bsq, csq = a * a, b * b, c * c  # 预先计算平方项

        alpha = _euler_matrix(row[7], row[8], row[9])  # 生成椭球旋转矩阵
        coordp = alpha @ coord  # 把体素坐标旋转到椭球局部坐标系

        inside = (  # 判断哪些点落在椭球内部
            (coordp[0] - x0) ** 2 / asq  # x 方向归一化距离
            + (coordp[1] - y0) ** 2 / bsq  # y 方向归一化距离
            + (coordp[2] - z0) ** 2 / csq  # z 方向归一化距离
        ) <= 1.0  # 满足椭球方程即视为内部
        volume[inside] += row[0]  # 将椭球密度加到内部体素上

    return volume.reshape((n, n, n))  # 恢复为三维数组


# -----------------------------  # Siddon 几何与线积分部分分隔线
# Siddon 几何与线积分  # 本节负责网格与射线积分计算
# -----------------------------  # Siddon 几何与线积分部分分隔线结束


@dataclass(frozen=True)  # 定义不可变数据类
class VoxelGrid:  # 表示与体模采样对齐的体素网格
    """与体模采样对齐的体素网格（坐标范围 [-1, 1]^3）。"""  # 类说明

    n: int  # 网格边长

    @property
    def spacing(self) -> tuple[float, float, float]:  # 返回三个方向的体素间距
        if self.n < 2:  # 尺寸过小时返回退化值
            return (2.0, 2.0, 2.0)  # 避免除零
        d = 2.0 / (self.n - 1)  # 计算均匀网格间距
        return (d, d, d)  # 三个方向相同

    @property
    def origin(self) -> tuple[float, float, float]:  # 返回第一个平面的坐标
        if self.n < 2:  # 尺寸过小时返回默认原点
            return (-1.0, -1.0, -1.0)  # 退化情形下的默认值
        d = 2.0 / (self.n - 1)  # 重新计算间距
        return (-1.0 - d / 2.0, -1.0 - d / 2.0, -1.0 - d / 2.0)  # 返回左边界平面坐标

    def plane_positions(self, axis: int) -> np.ndarray:  # 生成指定轴上的平面坐标
        d = self.spacing[axis]  # 读取该轴间距
        o = self.origin[axis]  # 读取该轴原点
        return o + np.arange(self.n + 1, dtype=np.float64) * d  # 生成 n+1 个边界平面

    def index_at_midpoint(self, point: np.ndarray) -> tuple[int, int, int] | None:  # 根据中点查找体素索引
        d = self.spacing  # 获取三轴间距
        o = self.origin  # 获取三轴原点
        idx = []  # 保存每个轴的索引
        for ax in range(3):  # 逐轴计算索引
            t = (point[ax] - o[ax]) / d[ax]  # 映射到网格坐标
            if t < 0.0 or t >= self.n:  # 超出范围则返回空
                return None  # 表示不在网格内
            i = int(np.floor(t))  # 向下取整得到体素编号
            if i >= self.n:  # 防止越界
                i = self.n - 1  # 截断到最后一个体素
            idx.append(i)  # 存入索引结果
        return idx[0], idx[1], idx[2]  # 返回三维索引


def _alphas_for_axis(s: np.ndarray, p: np.ndarray, planes: np.ndarray, axis: int) -> np.ndarray:  # 计算与平面的交点参数
    delta = p[axis] - s[axis]  # 计算该轴方向增量
    if abs(delta) < _EPS:  # 若射线与该轴平行
        return np.array([], dtype=np.float64)  # 返回空数组
    out = (planes - s[axis]) / delta  # 计算所有平面对应的 alpha
    return out[(out >= 0.0) & (out <= 1.0)]  # 仅保留射线段上的交点


def ray_enter_exit(s: np.ndarray, p: np.ndarray, grid: VoxelGrid) -> tuple[float, float] | None:  # 计算射线与包围盒交段
    alphas = [0.0, 1.0]  # 初始包含线段端点
    for ax in range(3):  # 检查三个轴向边界
        planes = grid.plane_positions(ax)  # 获取边界平面坐标
        delta = p[ax] - s[ax]  # 获取该轴方向变化量
        if abs(delta) < _EPS:  # 若该轴方向几乎不变
            if s[ax] < planes[0] or s[ax] > planes[-1]:  # 且起点不在范围内
                return None  # 则无相交区间
            continue  # 否则继续检查下一轴
        a0 = (planes[0] - s[ax]) / delta  # 计算进入候选参数
        a1 = (planes[-1] - s[ax]) / delta  # 计算离开候选参数
        alphas.extend([min(a0, a1), max(a0, a1)])  # 记录该轴的参数区间

    alpha_min = max(0.0, min(alphas))  # 取全局进入参数下界
    alpha_max = min(1.0, max(alphas))  # 取全局离开参数上界
    if alpha_min >= alpha_max - _EPS:  # 若区间过短或无效
        return None  # 返回无交段
    return alpha_min, alpha_max  # 返回有效交段


def _merge_alphas(alphas: np.ndarray) -> np.ndarray:  # 对 alpha 排序并去重
    if alphas.size == 0:  # 若输入为空
        return alphas  # 直接返回空数组
    a = np.sort(alphas)  # 对参数排序
    keep = [0]  # 保留第一个值
    for i in range(1, a.size):  # 遍历其余值
        if a[i] - a[keep[-1]] > _EPS:  # 与上一个保留值差异足够大时保留
            keep.append(i)  # 记录当前索引
    return a[keep]  # 返回去重后的数组


def line_integral_siddon(  # 计算一条射线穿过体模的线积分
    s: np.ndarray,  # 射线起点
    p: np.ndarray,  # 射线终点
    volume: np.ndarray,  # 三维体模
    grid: VoxelGrid | None = None,  # 可选体素网格
) -> float:  # 返回线积分结果
    s = np.asarray(s, dtype=np.float64)  # 将起点转为浮点数组
    p = np.asarray(p, dtype=np.float64)  # 将终点转为浮点数组
    if grid is None:  # 若未显式提供网格
        grid = VoxelGrid(volume.shape[0])  # 根据体模尺寸构造网格

    bounds = ray_enter_exit(s, p, grid)  # 求射线与网格包围盒交段
    if bounds is None:  # 若无交段
        return 0.0  # 直接返回零
    alpha_min, alpha_max = bounds  # 解包进入与离开参数

    alphas = [alpha_min, alpha_max]  # 先放入交段边界
    for ax in range(3):  # 再加入各轴平面的交点参数
        alphas.extend(_alphas_for_axis(s, p, grid.plane_positions(ax), ax).tolist())  # 扩展候选参数

    alphas = _merge_alphas(np.asarray(alphas, dtype=np.float64))  # 排序并去重
    if alphas.size < 2:  # 若参数数量太少
        return 0.0  # 返回零

    ray_len = float(np.linalg.norm(p - s))  # 计算整条射线长度
    direction = p - s  # 计算射线方向向量
    total = 0.0  # 初始化累加值

    for m in range(alphas.size - 1):  # 逐段处理相邻 alpha 区间
        a0, a1 = alphas[m], alphas[m + 1]  # 取区间两端
        d_alpha = a1 - a0  # 计算参数区间长度
        if d_alpha <= _EPS:  # 若区间过短
            continue  # 跳过该段
        alpha_mid = 0.5 * (a0 + a1)  # 取区间中点参数
        point = s + alpha_mid * direction  # 计算中点坐标
        idx = grid.index_at_midpoint(point)  # 找出中点所在体素
        if idx is None:  # 若不在网格内
            continue  # 跳过
        i, j, k = idx  # 解包体素索引
        ell = ray_len * d_alpha  # 计算当前段实际长度
        total += ell * volume[i, j, k]  # 累加长度乘以体素值

    return total  # 返回总线积分


# -----------------------------  # 实验部分分隔线
# Experiment  # 本节负责缩放实验和结果可视化
# -----------------------------  # 实验部分分隔线结束


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # 解析命令行参数
    # 命令行参数：N 列表、终点网格密度、源位置等  # 参数说明
    parser = argparse.ArgumentParser(description="Siddon scaling experiment (single-ray O(N)).")  # 创建参数解析器
    parser.add_argument(
        "--n-vals",  # N 值列表参数名
        type=str,  # 以字符串形式输入
        default="20,40,60,80,100",  # 默认的 N 值序列
        help="comma-separated N values (default: 20,40,60,80,100)",  # 参数提示
    )  # 参数添加结束
    parser.add_argument("--grid-size", type=int, default=21, help="number of points per axis (default: 21)")  # 终点网格尺寸
    parser.add_argument("--s-z", type=float, default=3.0, help="source z coordinate (default: 3.0)")  # 源点 z 坐标
    parser.add_argument("--fit", action="store_true", help="print linear fit slope/intercept")  # 是否输出线性拟合
    return parser.parse_args(argv)  # 返回解析结果


def _parse_n_vals(text: str) -> list[int]:  # 解析逗号分隔的 N 值
    out: list[int] = []  # 初始化结果列表
    for token in text.split(","):  # 按逗号拆分输入字符串
        token = token.strip()  # 去除空白字符
        if not token:  # 跳过空 token
            continue  # 继续下一个
        out.append(int(token))  # 将 token 转为整数后保存
    if not out:  # 若列表为空
        raise ValueError("--n-vals is empty")  # 抛出异常提示输入错误
    return out  # 返回解析结果


def main(argv: list[str] | None = None) -> int:  # 主函数入口
    # 主流程：生成终点网格 -> 遍历 N -> 逐射线计时 -> 输出与绘图  # 整体流程说明
    args = parse_args(argv)  # 解析参数
    n_vals = _parse_n_vals(args.n_vals)  # 解析 N 列表
    grid_size = int(args.grid_size)  # 读取网格大小
    s_z = float(args.s_z)  # 读取源点 z 坐标

    lin = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)  # 生成终点网格的一维坐标
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")  # 生成三维终点网格
    endpoints = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # 展平并组合终点坐标
    total_rays = endpoints.shape[0]  # 统计终点数量，也就是射线数

    rows: list[tuple[int, int, int, float, float]] = []  # 存储每个 N 的实验结果
    s = np.asarray([0.0, 0.0, s_z], dtype=np.float64)  # 定义固定源点坐标

    for n in n_vals:  # 遍历不同体模尺寸
        volume = build_phantom(n)  # 构建体模
        grid = VoxelGrid(n)  # 构建对应网格

        valid_count = 0  # 统计非零积分的射线数
        total_time = 0.0  # 累加总耗时
        for p in endpoints:  # 遍历每个终点
            t0 = perf_counter()  # 记录开始时间
            val = line_integral_siddon(s, p, volume, grid)  # 计算一次 Siddon 线积分
            t1 = perf_counter()  # 记录结束时间
            total_time += t1 - t0  # 累加单次耗时
            if val != 0.0:  # 若线积分非零
                valid_count += 1  # 统计有效射线

        avg_ms = (total_time / total_rays) * 1000.0  # 计算单射线平均耗时
        rows.append((n, total_rays, valid_count, total_time, avg_ms))  # 保存结果

    header = "N  total_rays  valid_rays  total_time(s)  avg_time(ms)"  # 输出表头
    print(header)  # 打印表头
    print("-" * len(header))  # 打印分隔线
    for n, total_rays, valid_rays, total_time, avg_ms in rows:  # 打印每行结果
        print(f"{n:>3}  {total_rays:>10}  {valid_rays:>10}  {total_time:>13.6f}  {avg_ms:>12.6f}")  # 逐行格式化输出

    ns = np.array([r[0] for r in rows], dtype=np.float64)  # 提取 N 数组
    avg_ms = np.array([r[4] for r in rows], dtype=np.float64)  # 提取平均耗时数组

    c = float(np.dot(ns, avg_ms) / np.dot(ns, ns))  # 计算过原点拟合系数
    ns_ref = np.concatenate(([0.0], ns))  # 构造参考直线横坐标
    ref = c * ns_ref  # 计算参考直线纵坐标

    plt.figure(figsize=(6, 4))  # 创建绘图窗口
    plt.plot(ns, avg_ms, "o-", label="avg time (ms)")  # 绘制实验点
    plt.plot(ns_ref, ref, "--", label=f"y = {c:.4f} * N")  # 绘制参考线
    plt.plot([0.0], [0.0], "o", label="origin")  # 绘制原点
    plt.xlabel("N (volume size)")  # 设置横轴标签
    plt.ylabel("avg time per ray (ms)")  # 设置纵轴标签
    plt.title("Siddon scaling experiment")  # 设置标题
    plt.grid(True, alpha=0.3)  # 添加网格
    plt.legend()  # 显示图例
    plt.xlim(left=0.0)  # 设置横轴左边界

    out_path = Path("outputs") / "siddon_scaling.png"  # 定义输出图片路径
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    plt.savefig(out_path, dpi=200, bbox_inches="tight")  # 保存图像
    print(f"saved figure: {out_path}")  # 打印保存位置

    if args.fit:  # 如果需要打印线性拟合
        m, b = np.polyfit(ns, avg_ms, 1)  # 进行一次线性拟合
        print(f"linear fit: slope={m:.6f} ms/N, intercept={b:.6f} ms")  # 打印拟合结果

    plt.show()  # 显示图像
    return 0  # 返回成功状态码


if __name__ == "__main__":  # 仅当脚本直接运行时执行主函数
    raise SystemExit(main())  # 用主函数返回码退出
