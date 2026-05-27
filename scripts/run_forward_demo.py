#!/usr/bin/env python
"""Build phantom, run cone-beam Siddon forward projection, save acceptance artifacts."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbct_siddon.geometry import ConeBeamGeometry  # noqa: E402
from cbct_siddon.phantom import build_phantom  # noqa: E402
from cbct_siddon.project import forward_project  # noqa: E402

OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    n = 256
    geom = ConeBeamGeometry(
        dso=2.0,
        dsd=4.0,
        det_nu=64,
        det_nv=64,
        du=0.05,
        dv=0.05,
        n_angles=36,
        volume_n=n,
    )

    print("Building Modified Shepp-Logan phantom...")
    volume = build_phantom(n)
    c = n // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[c, :, :], cmap="gray", origin="lower")
    axes[0].set_title("Axial (z mid)")
    axes[1].imshow(volume[:, c, :], cmap="gray", origin="lower")
    axes[1].set_title("Coronal (y mid)")
    axes[2].imshow(volume[:, :, c], cmap="gray", origin="lower")
    axes[2].set_title("Sagittal (x mid)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "phantom_slices.png", dpi=120)
    plt.close(fig)

    print("Forward projecting (cone-beam Siddon)...")
    t0 = time.perf_counter()
    proj = forward_project(volume, geom)
    elapsed = time.perf_counter() - t0
    print(f"  shape={proj.shape}, elapsed={elapsed:.2f}s")

    np.savez(
        OUTPUTS / "projections.npz",
        proj=proj,
        angles=geom.angles(),
        volume=volume,
    )

    # 尝试将模体保存为 NIfTI（如果可用）以便 3D 可视化 / 互操作
    try:
        import nibabel as nib  # type: ignore

        affine = np.eye(4)
        # 保存为 float32，文件名为 phantom.nii.gz
        nifti_path = OUTPUTS / "phantom.nii.gz"
        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), nifti_path)
        print(f"Wrote {nifti_path}")
    except Exception:
        print("nibabel not available: to save NIfTI install with 'pip install nibabel'")

    plt.figure(figsize=(6, 5))
    plt.imshow(proj[0], cmap="gray", origin="lower")
    plt.title("Cone-beam projection (beta=0)")
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "projection_beta000.png", dpi=120)
    plt.close()

    # pytest summary for acceptance log
    pytest_rc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    log_lines = [
        f"acceptance_run_utc: {datetime.now(timezone.utc).isoformat()}",
        f"phantom_n: {n}",
        f"geometry: dso={geom.dso}, dsd={geom.dsd}, det={geom.det_nu}x{geom.det_nv}, "
        f"angles={geom.n_angles}",
        f"forward_project_seconds: {elapsed:.3f}",
        f"proj_shape: {proj.shape}",
        f"proj_dtype: {proj.dtype}",
        f"proj_finite: {np.all(np.isfinite(proj))}",
        "pytest_output:",
        pytest_rc.stdout.strip() or "(empty)",
        pytest_rc.stderr.strip() or "(empty)",
        f"pytest_exit_code: {pytest_rc.returncode}",
    ]
    log_path = OUTPUTS / "acceptance_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Wrote {log_path}")

    if pytest_rc.returncode != 0:
        sys.exit(pytest_rc.returncode)


if __name__ == "__main__":
    main()
