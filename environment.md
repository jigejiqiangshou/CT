# Conda 环境说明

- **环境名称**：`medical_reg`
- **Python**：`C:\Users\Zhaoji\miniconda3\envs\medical_reg\python.exe`
- **约定**：本仓库所有测试与脚本均在该环境中执行；**禁止**在 `CT/` 下创建 `.venv` 或向其他 Conda 环境安装本包。

## 一次性设置

```bash
conda activate medical_reg
cd c:\Users\Zhaoji\Desktop\CT
pip install -e .
```

若 `pytest` / `matplotlib` 等已存在则无需重复安装 `requirements.txt`。

## 常用命令

```bash
conda activate medical_reg
python -m pytest tests/ -v
python scripts/run_forward_demo.py
```
