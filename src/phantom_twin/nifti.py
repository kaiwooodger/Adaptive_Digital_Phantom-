from __future__ import annotations

from pathlib import Path

import numpy as np


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "NIfTI support requires nibabel. Install with: "
            "python3 -m pip install nibabel"
        ) from exc
    return nib


def inspect_nifti(path: str | Path) -> dict[str, object]:
    nib = _import_nibabel()
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)

    return {
        "path": str(path),
        "shape": tuple(int(value) for value in data.shape),
        "dtype": str(data.dtype),
        "spacing": tuple(float(value) for value in image.header.get_zooms()[: data.ndim]),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "affine": image.affine.tolist(),
    }


def format_nifti_summary(summary: dict[str, object]) -> str:
    lines = [
        f"Path: {summary['path']}",
        f"Shape: {summary['shape']}",
        f"Dtype: {summary['dtype']}",
        f"Spacing: {summary['spacing']}",
        f"Value range: {summary['min']:.3f} to {summary['max']:.3f}",
        f"Mean: {summary['mean']:.3f}",
        "Affine:",
    ]
    affine = summary["affine"]
    if isinstance(affine, list):
        for row in affine:
            lines.append("  " + " ".join(f"{float(value): .6g}" for value in row))
    return "\n".join(lines)
