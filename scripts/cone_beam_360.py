#!/usr/bin/env python3  # 指定使用当前环境的 Python 解释器执行脚本
"""使用 Siddon 算法进行 360° 锥束正向投影。

本脚本通过绕 Y 轴旋转射线源与探测器，生成给定体模的投影集合（正弦图）。
"""

from __future__ import annotations  # 启用延迟类型注解解析

import argparse  # 解析命令行参数
from dataclasses import dataclass  # 定义只读数据类
from pathlib import Path  # 处理文件路径
from time import perf_counter  # 统计计算耗时

import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图与预览输出

_EPS = 1e-12  # 判断数值是否接近相等的阈值


# ----------------------------------------------------------------------  # 体模生成部分的分隔线
# 体模生成（与原始脚本保持一致）  # 说明这一段负责构造体模
# ----------------------------------------------------------------------  # 体模生成部分的分隔线结束
# 椭球参数表（10×10），每行表示一个椭球：
# 列依次为：A, a, b, c, x0, y0, z0, phi, theta, psi（角度单位：度）
MODIFIED_SHEPP_LOGAN: np.ndarray = np.array(  # 定义修改版 Shepp-Logan 体模参数表
    [  # 每一行表示一个椭球体的参数
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
    ],  # 参数矩阵结束
    dtype=np.float64,  # 使用双精度浮点保存参数
)


def _euler_matrix(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:  # 生成欧拉旋转矩阵
    phi = np.deg2rad(phi_deg)  # 将 phi 从角度转换为弧度
    theta = np.deg2rad(theta_deg)  # 将 theta 从角度转换为弧度
    psi = np.deg2rad(psi_deg)  # 将 psi 从角度转换为弧度
    cphi, sphi = np.cos(phi), np.sin(phi)  # 计算 phi 的余弦与正弦
    ctheta, stheta = np.cos(theta), np.sin(theta)  # 计算 theta 的余弦与正弦
    cpsi, spsi = np.cos(psi), np.sin(psi)  # 计算 psi 的余弦与正弦
    return np.array(  # 返回 3x3 旋转矩阵
        [  # 矩阵第一行到第三行
            [cpsi * cphi - ctheta * sphi * spsi, cpsi * sphi + ctheta * cphi * spsi, spsi * stheta],  # 第一行
            [-spsi * cphi - ctheta * sphi * cpsi, -spsi * sphi + ctheta * cphi * cpsi, cpsi * stheta],  # 第二行
            [stheta * sphi, -stheta * cphi, ctheta],  # 第三行
        ],  # 矩阵行列表结束
        dtype=np.float64,  # 输出使用双精度
    )  # 旋转矩阵构造结束


def build_phantom(n: int = 64, kind: str = "modified") -> np.ndarray:  # 构建三维体模数组
    if kind != "modified":  # 检查体模类型是否受支持
        raise ValueError(f'Unknown phantom kind "{kind}"; use "modified".')  # 不支持时直接报错
    if n < 2:  # 检查体模尺寸是否合法
        raise ValueError("n must be >= 2")  # 尺寸过小时抛出异常

    rng = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) / ((n - 1) / 2.0)  # 构造 [-1,1] 均匀坐标
    x, y, z = np.meshgrid(rng, rng, rng, indexing="ij")  # 生成三维网格坐标
    coord = np.vstack((x.ravel(), y.ravel(), z.ravel()))  # 展平成 3xN 的坐标矩阵

    volume = np.zeros(n * n * n, dtype=np.float64)  # 初始化体素值数组
    for row in MODIFIED_SHEPP_LOGAN:  # 逐个叠加椭球体
        a, b, c = row[1], row[2], row[3]  # 读取椭球三轴长度
        x0, y0, z0 = row[4], row[5], row[6]  # 读取椭球中心偏移
        asq, bsq, csq = a * a, b * b, c * c  # 预先计算平方项

        alpha = _euler_matrix(row[7], row[8], row[9])  # 计算椭球旋转矩阵
        coordp = alpha @ coord  # 将坐标旋转到椭球局部坐标系

        inside = (  # 判断哪些点落在椭球内部
            (coordp[0] - x0) ** 2 / asq  # x 方向归一化距离
            + (coordp[1] - y0) ** 2 / bsq  # y 方向归一化距离
            + (coordp[2] - z0) ** 2 / csq  # z 方向归一化距离
        ) <= 1.0  # 只有满足椭球方程的点才算内部
        volume[inside] += row[0]  # 内部体素累加该椭球的密度值

    return volume.reshape((n, n, n))  # 将一维数组恢复为三维体数据


# ----------------------------------------------------------------------  # Siddon 几何与积分部分的分隔线
# Siddon 几何与积分（未改动）  # 说明这一段负责射线与体素相交计算
# ----------------------------------------------------------------------  # Siddon 几何与积分部分的分隔线结束

@dataclass(frozen=True)
class VoxelGrid:  # 表示规则三维体素网格
    n: int  # 网格边长

    @property
    def spacing(self) -> tuple[float, float, float]:  # 返回体素间距
        if self.n < 2:  # 尺寸过小时使用退化间距
            return (2.0, 2.0, 2.0)  # 仅用于避免除零
        d = 2.0 / (self.n - 1)  # 计算均匀网格的实际间距
        return (d, d, d)  # 三个方向间距相同

    @property
    def origin(self) -> tuple[float, float, float]:  # 返回第一个体素中心之前的网格原点
        if self.n < 2:  # 尺寸过小时返回默认原点
            return (-1.0, -1.0, -1.0)  # 退化情形下的原点
        d = 2.0 / (self.n - 1)  # 计算体素间距
        return (-1.0 - d / 2.0, -1.0 - d / 2.0, -1.0 - d / 2.0)  # 返回外延半个体素的原点

    def plane_positions(self, axis: int) -> np.ndarray:  # 返回指定轴上的所有体素边界平面位置
        d = self.spacing[axis]  # 获取该轴上的体素间距
        o = self.origin[axis]  # 获取该轴上的起始原点
        return o + np.arange(self.n + 1, dtype=np.float64) * d  # 生成 n+1 个平面坐标

    def index_at_midpoint(self, point: np.ndarray) -> tuple[int, int, int] | None:  # 根据中点坐标查找体素索引
        d = self.spacing  # 获取三个方向的体素间距
        o = self.origin  # 获取三个方向的网格原点
        idx = []  # 用于存放每个轴上的索引
        for ax in range(3):  # 逐轴计算索引
            t = (point[ax] - o[ax]) / d[ax]  # 将坐标映射到网格索引空间
            if t < 0.0 or t >= self.n:  # 若点超出体素范围则返回空
                return None  # 表示该点不在网格内
            i = int(np.floor(t))  # 取下取整得到体素编号
            if i >= self.n:  # 保险性截断，避免越界
                i = self.n - 1  # 限制到最后一个体素
            idx.append(i)  # 保存该轴索引
        return idx[0], idx[1], idx[2]  # 返回三维索引


def _alphas_for_axis(s: np.ndarray, p: np.ndarray, planes: np.ndarray, axis: int) -> np.ndarray:  # 计算与某一轴平面的交点参数
    delta = p[axis] - s[axis]  # 计算该轴上的方向增量
    if abs(delta) < _EPS:  # 若与该轴平行
        return np.array([], dtype=np.float64)  # 则没有交点参数可返回
    out = (planes - s[axis]) / delta  # 计算所有平面对应的参数值
    return out[(out >= 0.0) & (out <= 1.0)]  # 只保留射线段上的参数


def ray_enter_exit(s: np.ndarray, p: np.ndarray, grid: VoxelGrid) -> tuple[float, float] | None:  # 计算射线进入和离开网格的参数区间
    alphas = [0.0, 1.0]  # 初始包含整条线段端点
    for ax in range(3):  # 检查三个轴向上的边界
        planes = grid.plane_positions(ax)  # 获取该轴上的所有平面
        delta = p[ax] - s[ax]  # 计算该轴方向分量
        if abs(delta) < _EPS:  # 若射线与该轴平行
            if s[ax] < planes[0] or s[ax] > planes[-1]:  # 且起点不在平面范围内
                return None  # 则整条射线不与网格相交
            continue  # 起点在范围内则继续检查其他轴
        a0 = (planes[0] - s[ax]) / delta  # 计算与最小平面的交点参数
        a1 = (planes[-1] - s[ax]) / delta  # 计算与最大平面的交点参数
        alphas.extend([min(a0, a1), max(a0, a1)])  # 将区间端点加入候选集合

    alpha_min = max(0.0, min(alphas))  # 取所有下界中的最大值
    alpha_max = min(1.0, max(alphas))  # 取所有上界中的最小值
    if alpha_min >= alpha_max - _EPS:  # 若区间长度过小
        return None  # 则视为没有有效交段
    return alpha_min, alpha_max  # 返回有效进入与离开参数


def _merge_alphas(alphas: np.ndarray) -> np.ndarray:  # 合并重复或几乎重复的参数值
    if alphas.size == 0:  # 若输入为空
        return alphas  # 直接返回空数组
    a = np.sort(alphas)  # 对参数值排序
    keep = [0]  # 保留第一个元素
    for i in range(1, a.size):  # 逐个比较后续元素
        if a[i] - a[keep[-1]] > _EPS:  # 若与上一个保留值相差足够大
            keep.append(i)  # 则保留下来
    return a[keep]  # 返回去重后的参数数组


def line_integral_siddon(  # 使用 Siddon 算法计算线积分
    s: np.ndarray,  # 射线起点
    p: np.ndarray,  # 射线终点
    volume: np.ndarray,  # 三维体模数据
    grid: VoxelGrid | None = None,  # 可选的体素网格描述
) -> float:  # 返回线积分值
    s = np.asarray(s, dtype=np.float64)  # 将起点转换为浮点数组
    p = np.asarray(p, dtype=np.float64)  # 将终点转换为浮点数组
    if grid is None:  # 若未显式提供网格
        grid = VoxelGrid(volume.shape[0])  # 根据体模尺寸自动构造网格

    bounds = ray_enter_exit(s, p, grid)  # 先计算射线与网格的交段
    if bounds is None:  # 若没有交段
        return 0.0  # 线积分直接为零
    alpha_min, alpha_max = bounds  # 解包进入与离开参数

    alphas = [alpha_min, alpha_max]  # 先加入区间端点
    for ax in range(3):  # 再加入与各轴平面的交点参数
        alphas.extend(_alphas_for_axis(s, p, grid.plane_positions(ax), ax).tolist())  # 汇总候选参数

    alphas = _merge_alphas(np.asarray(alphas, dtype=np.float64))  # 排序并去重
    if alphas.size < 2:  # 若有效区间太短
        return 0.0  # 直接返回零

    ray_len = float(np.linalg.norm(p - s))  # 计算整段射线长度
    direction = p - s  # 计算射线方向向量
    total = 0.0  # 累加线积分结果

    for m in range(alphas.size - 1):  # 遍历相邻参数区间
        a0, a1 = alphas[m], alphas[m + 1]  # 取相邻端点
        d_alpha = a1 - a0  # 计算参数区间长度
        if d_alpha <= _EPS:  # 若区间过短
            continue  # 跳过该区间
        alpha_mid = 0.5 * (a0 + a1)  # 取区间中点参数
        point = s + alpha_mid * direction  # 计算中点坐标
        idx = grid.index_at_midpoint(point)  # 查找中点所属体素
        if idx is None:  # 若中点不在网格内
            continue  # 跳过该区间
        i, j, k = idx  # 解包体素索引
        ell = ray_len * d_alpha  # 计算当前段的物理长度
        total += ell * volume[i, j, k]  # 累加长度乘以体素值

    return total  # 返回总线积分


# ----------------------------------------------------------------------  # 锥束几何部分的分隔线
# 锥束几何辅助函数  # 说明这一段负责源点和探测器坐标生成
# ----------------------------------------------------------------------  # 锥束几何部分的分隔线结束

def rotation_matrix_y(theta_deg: float) -> np.ndarray:
    """绕 Y 轴的 3x3 旋转矩阵（右手系）。"""  # 说明该函数返回绕 Y 轴旋转矩阵
    theta = np.deg2rad(theta_deg)  # 将角度转换为弧度
    c, s = np.cos(theta), np.sin(theta)  # 计算余弦与正弦
    return np.array([[c, 0, s],  # 第一行：X-Z 平面旋转
                     [0, 1, 0],  # 第二行：Y 轴保持不变
                     [-s, 0, c]], dtype=np.float64)  # 第三行：X-Z 平面旋转


def cone_beam_pixel_coordinates(
    theta_deg: float,
    sad: float,
    oad: float,
    rows: int,
    cols: int,
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # 返回像素中心、源点和探测器中心
    """生成指定角度下所有探测器像素中心的世界坐标。"""  # 函数用途说明
    # 返回值说明如下  # 说明返回数组的含义
    # pixel_centers: (rows*cols, 3) 数组  # 所有像素中心坐标
    # source: (3,) 数组  # 当前角度下的射线源坐标
    # det_center: (3,) 数组  # 当前角度下的探测器中心坐标
    source0 = np.array([sad, 0.0, 0.0], dtype=np.float64)  # 角度为 0 时的源点
    det_center0 = np.array([-oad, 0.0, 0.0], dtype=np.float64)  # 角度为 0 时的探测器中心
    dir0 = det_center0 - source0  # 中心射线方向向量
    dir0_norm = dir0 / np.linalg.norm(dir0)  # 对中心射线方向做归一化

    # 角度为 0 时定义局部坐标轴  # 说明探测器局部坐标系的构造方式
    u_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # 行方向对齐世界 Y 轴
    v_axis = np.cross(dir0_norm, u_axis)  # 列方向由叉乘确定以保证正交
    v_axis = v_axis / np.linalg.norm(v_axis)  # 将列方向单位化

    u_offsets = (np.arange(rows) - (rows - 1) / 2.0) * pixel_size  # 行方向像素偏移
    v_offsets = (np.arange(cols) - (cols - 1) / 2.0) * pixel_size  # 列方向像素偏移
    uu, vv = np.meshgrid(u_offsets, v_offsets, indexing='ij')  # 生成二维偏移网格
    offsets = uu[..., np.newaxis] * u_axis + vv[..., np.newaxis] * v_axis  # 组合成三维偏移向量

    pixels0 = det_center0 + offsets.reshape(-1, 3)  # 计算角度为 0 时的像素中心

    R = rotation_matrix_y(theta_deg)  # 计算绕 Y 轴的旋转矩阵
    source = R @ source0  # 旋转射线源
    det_center = R @ det_center0  # 旋转探测器中心
    pixels = (R @ pixels0.T).T  # 旋转所有像素中心到世界坐标系

    return pixels, source, det_center  # 返回像素、源点与探测器中心


# ----------------------------------------------------------------------  # 主流程部分的分隔线
# 主流程  # 说明这一段负责命令行执行与批量投影计算
# ----------------------------------------------------------------------  # 主流程部分的分隔线结束

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Siddon 的 360° 锥束正向投影")  # 创建参数解析器
    parser.add_argument("--n", type=int, default=64, help="体模尺寸（N x N x N）")  # 体模边长
    parser.add_argument("--sad", type=float, default=3.0, help="射线源到旋转轴距离")  # 源到轴距离
    parser.add_argument("--oad", type=float, default=1.0, help="物体到探测器距离")  # 物体到探测器距离
    parser.add_argument("--rows", type=int, default=256, help="探测器行数（u 方向，沿 Y 轴）")  # 探测器行数
    parser.add_argument("--cols", type=int, default=256, help="探测器列数（v 方向）")  # 探测器列数
    parser.add_argument("--pixel-size", type=float, default=0.01, help="像素间距（世界单位）")  # 像素尺寸
    parser.add_argument("--start-angle", type=float, default=0.0, help="起始角度（度）")  # 起始角度
    parser.add_argument("--end-angle", type=float, default=360.0, help="结束角度（度，若步长整除则不包含）")  # 结束角度
    parser.add_argument("--step", type=float, default=1.0, help="角度步长（度）")  # 角度步长
    parser.add_argument("--output-npy", type=Path, default=Path("outputs/projections_360.npy"),  # 输出文件路径参数
                        help="输出 .npy 文件路径")  # 输出路径说明
    parser.add_argument("--preview", action="store_true", help="计算后显示 0°、90°、180°、270° 的预览")  # 是否显示预览
    return parser.parse_args()  # 解析并返回参数


def main() -> int:
    args = parse_args()  # 解析命令行参数

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录，若已存在则忽略

    print("Generating phantom ...")  # 打印体模生成提示
    volume = build_phantom(args.n)  # 构造三维体模
    grid = VoxelGrid(args.n)  # 构造对应的体素网格
    print(f"Phantom size: {args.n}^3 voxels")  # 打印体模尺寸

    angles = np.arange(args.start_angle, args.end_angle, args.step, dtype=np.float64)  # 生成角度序列
    n_angles = len(angles)  # 统计角度数量
    print(f"Angles: {n_angles} from {args.start_angle}° to {args.end_angle - args.step}° (step {args.step}°)")  # 打印角度范围
    print(f"Detector: {args.rows} x {args.cols}, pixel size = {args.pixel_size}")  # 打印探测器参数
    print("Starting forward projection (this may take a while) ...")  # 提示开始计算

    projections = np.zeros((n_angles, args.rows, args.cols), dtype=np.float32)  # 预分配投影数组

    total_rays = n_angles * args.rows * args.cols  # 统计总射线数
    print(f"Total rays to compute: {total_rays}")  # 打印总射线数量

    try:  # 尝试启用进度条
        from tqdm import tqdm  # 导入 tqdm 进度条
        angle_iter = tqdm(angles, desc="Angles")  # 用 tqdm 包装角度迭代器
    except ImportError:  # 若未安装 tqdm
        angle_iter = angles  # 退回到普通迭代
        print("Tip: install tqdm for a progress bar.")  # 提示安装 tqdm

    start_time = perf_counter()  # 记录开始时间

    for idx, theta in enumerate(angle_iter):  # 逐角度进行投影计算
        if angle_iter is angles:  # 若没有进度条
            print(f"  Angle {theta:.1f}° ({idx+1}/{n_angles})")  # 手动打印当前进度

        pixels, source, _ = cone_beam_pixel_coordinates(  # 计算该角度下像素与源点坐标
            theta, args.sad, args.oad, args.rows, args.cols, args.pixel_size  # 传入几何参数
        )  # 函数调用结束

        proj_2d = np.zeros((args.rows, args.cols), dtype=np.float32)  # 初始化当前角度的二维投影
        for r in range(args.rows):  # 遍历探测器行
            for c in range(args.cols):  # 遍历探测器列
                pidx = r * args.cols + c  # 将二维索引映射为一维索引
                val = line_integral_siddon(source, pixels[pidx], volume, grid)  # 计算该射线的线积分
                proj_2d[r, c] = val  # 将结果写入投影图像
        projections[idx] = proj_2d  # 保存当前角度的二维投影

    elapsed = perf_counter() - start_time  # 计算总耗时
    print(f"Total computation time: {elapsed:.2f} s")  # 打印总耗时

    np.save(args.output_npy, projections)  # 将投影结果保存为 npy 文件
    print(f"Saved projections to {args.output_npy} (shape {projections.shape})")  # 打印保存信息

    if args.preview:  # 若需要预览
        if n_angles == 0:  # 若没有角度可供预览
            print("No angles available for preview.")  # 提示无可用角度
        else:  # 否则生成预览图
            targets = [0.0, 90.0, 180.0, 270.0]  # 选取四个典型角度
            idxs = [int(np.argmin(np.abs(angles - t))) for t in targets]  # 找到最接近的角度索引
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 创建 2x2 子图
            for ax, idx, title in zip(axes.flat, idxs, ["0°", "90°", "180°", "270°"]):  # 逐个绘制子图
                im = ax.imshow(projections[idx], cmap="gray", origin="upper")  # 显示投影图
                ax.set_title(title)  # 设置标题
                plt.colorbar(im, ax=ax, fraction=0.046)  # 添加颜色条
            plt.tight_layout()  # 调整布局
            preview_path = args.output_npy.parent / "preview_angles.png"  # 构造预览图片路径
            plt.savefig(preview_path, dpi=150)  # 保存预览图片
            print(f"Preview image saved to {preview_path}")  # 打印预览保存信息
            plt.show()  # 弹出显示窗口

    return 0  # 正常结束


if __name__ == "__main__":  # 仅在脚本直接运行时执行主函数
    raise SystemExit(main())  # 以主函数返回码退出进程