from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _import_mesh_dependencies():
    try:
        import nibabel as nib  # type: ignore
        from skimage import measure  # type: ignore
        import trimesh  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Mesh export requires nibabel, scikit-image, and trimesh. Install with: "
            "python3 -m pip install nibabel scikit-image trimesh"
        ) from exc
    return nib, measure, trimesh


def export_label_mesh(
    labels_path: str | Path,
    label_id: int | Sequence[int],
    output_path: str | Path,
    level: float = 0.5,
) -> dict[str, object]:
    nib, measure, trimesh = _import_mesh_dependencies()
    image = nib.load(str(labels_path))
    labels = np.asanyarray(image.dataobj)
    label_ids = (label_id,) if isinstance(label_id, int) else tuple(label_id)
    if not label_ids:
        raise ValueError("At least one label ID is required")

    # Some public medical label maps store integer labels as scaled floats
    # such as 2.9999999993. Round before matching to avoid empty masks.
    rounded_labels = np.rint(labels).astype(np.int32)
    mask = np.isin(rounded_labels, label_ids).astype(np.uint8)

    if int(mask.sum()) == 0:
        raise ValueError(f"Labels {label_ids} were not found in {labels_path}")

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    vertices, faces, normals, _ = measure.marching_cubes(mask, level=level, spacing=spacing)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)

    return {
        "labels_path": str(labels_path),
        "label_ids": label_ids,
        "output_path": str(output),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "volume_voxels": int(mask.sum()),
        "spacing": spacing,
    }
