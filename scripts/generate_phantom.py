#!/usr/bin/env python
"""Generate Modified Shepp-Logan phantom and save visualization/NIfTI for inspection.

This script exposes a function `generate_and_save_phantom(n, out_dir, save_nifti=True)`
which can be imported in tests or run as a CLI for manual checks.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cbct_siddon.phantom import build_phantom  # noqa: E402


def generate_and_save_phantom(n: int = 64, out_dir: Path | str | None = None, save_nifti: bool = True) -> np.ndarray:
    """Generate phantom, save orthogonal slice image and optionally a NIfTI file.

    Parameters
    ----------
    n : int
        Grid size per axis.
    out_dir : Path | str | None
        Directory to save outputs. Defaults to repository `outputs/` directory.
    save_nifti : bool
        Whether to attempt to save a NIfTI file (requires `nibabel`).

    Returns
    -------
    volume : ndarray
        Generated (n,n,n) phantom volume.
    """
    if out_dir is None:
        out_dir = ROOT / "outputs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    volume = build_phantom(n)

    # Save three orthogonal slices (axial, coronal, sagittal)
    c = n // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[c, :, :], cmap="gray", origin="lower")
    axes[0].set_title("Axial (z mid)")
    axes[1].imshow(volume[:, c, :], cmap="gray", origin="lower")
    axes[1].set_title("Coronal (y mid)")
    axes[2].imshow(volume[:, :, c], cmap="gray", origin="lower")
    axes[2].set_title("Sagittal (x mid)")
    fig.tight_layout()
    slice_path = out_dir / f"phantom_slices_n{n}.png"
    fig.savefig(slice_path, dpi=120)
    plt.close(fig)

    # Optionally save NIfTI
    nifti_path = out_dir / f"phantom_n{n}.nii.gz"
    if save_nifti:
        try:
            import nibabel as nib  # type: ignore

            affine = np.eye(4)
            nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), nifti_path)
            nifti_saved = True
        except Exception:
            nifti_saved = False
    else:
        nifti_saved = False

    print(f"Wrote {slice_path}")
    if nifti_saved:
        print(f"Wrote {nifti_path}")
    else:
        if save_nifti:
            print("nibabel not available — NIfTI not saved. Install with: pip install nibabel")

    return volume


def _main() -> None:
    p = argparse.ArgumentParser(description="Generate Modified Shepp-Logan phantom and save outputs")
    p.add_argument("-n", "--size", type=int, default=64, help="phantom size per axis")
    p.add_argument("-o", "--out", type=Path, default=ROOT / "outputs", help="output directory")
    p.add_argument("--no-nifti", dest="nifti", action="store_false", help="do not attempt to save NIfTI")
    args = p.parse_args()

    generate_and_save_phantom(n=args.size, out_dir=args.out, save_nifti=args.nifti)


if __name__ == "__main__":
    _main()
