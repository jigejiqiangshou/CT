# CBCT Siddon — 锥束正投影

Python 实现：**Modified Shepp–Logan 3D 模体** + **Siddon (1985) 射线追踪** + **锥束正投影**（非平行束）。

## 环境（Conda `medical_reg`）

本项目**不在仓库内创建 venv**，统一使用已有 Conda 环境：

```text
conda activate medical_reg
```

| 包 | 用途 | `medical_reg` 状态 |
|----|------|-------------------|
| `numpy` | 数组 / Siddon | 已包含 |
| `scipy` | 椭球旋转等 | 已包含 |
| `matplotlib` | 验收图 | 已包含 |
| `pytest` / `pytest-cov` | 测试 | 已包含 |

若缺少某包，**仅在该环境中**安装（勿改动其他 env）：

```bash
conda activate medical_reg
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` 仅列出运行时依赖；开发可选见 `requirements-dev.txt`（`ruff` 可选，非必须）。

## 运行

```bash
conda activate medical_reg
cd c:\Users\Zhaoji\Desktop\CT

# 单元 / 集成测试
python -m pytest tests/ -v

# 覆盖率（可选）
python -m pytest tests/ --cov=cbct_siddon --cov-report=term-missing

# 演示：模体切片 + 锥束投影 + 验收日志

# 体模生成

python scripts/generate_phantom.py --size 128
# 模体切片 + 锥束投影 + 验收日志
python scripts/run_forward_demo.py
```

## 可运行脚本（新增）

### 1) 单射线线积分（源在上方）

```bash
python scripts/line_integral_overhead.py --use-phantom --n 64 --s_z 3.0 --p 0.1 0.0 -1.0
```

### 2) Siddon 线性缩放实验（O(N)）

```bash
python scripts/siddon_scaling_experiment.py
```

可选：

```bash
python scripts/siddon_scaling_experiment.py --n-vals 20,40,60,80,100 --grid-size 21 --s-z 3.0 --fit
```

### 3) 单角度锥束正投影

```bash
python scripts/cone_beam_single_angle.py
```

可选（更小尺寸快速验证）：

```bash
python scripts/cone_beam_single_angle.py --n 32 --det-rows 32 --det-cols 32 --det-pixel-size 0.05
```

### 4) 360° 锥束正投影（多角度）

```bash
python scripts/cone_beam_360.py --n 32 --rows 32 --cols 32 --step 30 --preview
```

> 提示：角度数量与探测器分辨率会显著影响耗时，建议先用小尺寸验证。

产物目录 [`outputs/`](outputs/)：

| 文件 | 说明 |
|------|------|
| `phantom_slices.png` | 三正交切片 |
| `projection_beta000.png` | 首视角投影 |
| `projections.npz` | `proj`, `angles`, `volume` |
| `acceptance_log.txt` | 参数、耗时、pytest 摘要 |

## 默认几何（调试）

| 参数 | 值 |
|------|-----|
| 体素 | 64³ |
| `dso` / `dsd` | 2.0 / 4.0 |
| 探测器 | 64×64，`du=dv=0.05` |
| 视角 | 36（10° 间隔） |

## 模块

- `phantom.py` — Toft / phantom3d 10 椭球表
- `geometry.py` — 锥束源与平板探测器
- `siddon.py` — 单射线 Siddon 线积分
- `project.py` — 多视角正投影

## 验收清单

**自动门禁**

1. `python -m pytest tests/ -v` 全部通过  
2. `python scripts/run_forward_demo.py` 退出码 0  
3. `outputs/projections.npz` 中 `proj` 为 `float64` 且有限  

**人工检查**

1. 模体切片：颅骨外椭球、内部结构可辨  
2. 单视角投影：头模轮廓正常，无全黑/竖条异常  
3. 多视角：`projections.npz` 中各角度投影随旋转变化  
4. `test_siddon_vs_bruteforce` 通过（Siddon 与暴力法一致）  
5. 规模：`n=64`, 36 视角, 64² 探测器；记录 `acceptance_log.txt` 中耗时  

## 验收记录

| 日期 | pytest | demo 耗时 (s) | 备注 |
|------|--------|---------------|------|
| 2026-05-25 | 17 passed | 108.5 | `medical_reg`, n=64, 36×64² |

## 参考

- Siddon R. L., *Medical Physics* 12, 252 (1985), [doi:10.1118/1.595715](https://doi.org/10.1118/1.595715)
- `phantom3d.m` — Modified Shepp–Logan 椭球参数
