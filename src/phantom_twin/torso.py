from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .abdominal_organs import ABDOMINAL_REGION_SPECS
from .materials import MaterialLibrary, load_material_library


def _import_dependencies():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from matplotlib.colors import ListedColormap  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
        from scipy import ndimage  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Digital torso phantom generation requires nibabel, scipy, matplotlib, and PyYAML."
        ) from exc
    return plt, ListedColormap, Patch, nib, ndimage, yaml


@dataclass(frozen=True)
class MaterialRegion:
    index: int
    name: str
    material_id: str
    target_hu_midpoint: float
    mass_density_g_cm3: float
    relative_electron_density: float
    color: str


@dataclass(frozen=True)
class MaterialRegionStats:
    index: int
    name: str
    material_id: str
    voxel_count: int
    volume_cm3: float
    mean_source_hu: float | None
    target_hu_midpoint: float
    mass_density_g_cm3: float
    relative_electron_density: float


@dataclass(frozen=True)
class DigitalTorsoResult:
    case_id: str
    output_dir: str
    material_label_path: str
    density_path: str
    relative_electron_density_path: str
    synthetic_hu_path: str
    body_mask_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    body_voxels: int
    body_volume_cm3: float
    voxel_volume_mm3: float
    regions: tuple[MaterialRegion, ...]
    stats: tuple[MaterialRegionStats, ...]
    notes: tuple[str, ...]


def _load_labelmap(path: str | Path) -> dict[int, dict[str, Any]]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    labels = data.get("labels", {})
    return {int(key): value for key, value in labels.items()}


def _target_midpoint(material_id: str, library: MaterialLibrary) -> tuple[float, float, float]:
    material = library.by_id[material_id]
    hu_min, hu_max = material.target_hu
    return (
        (hu_min + hu_max) / 2.0,
        material.target_mass_density_g_cm3,
        material.target_relative_electron_density,
    )


def _region_table(library: MaterialLibrary) -> tuple[MaterialRegion, ...]:
    raw = [
        (0, "external_air", "air", "#0d1b2a"),
        (1, "internal_air_cavity", "air", "#6c757d"),
        (2, "low_density_lung_like_tissue", "lung_inflated", "#8ecae6"),
        (3, "adipose_envelope", "adipose", "#ffd166"),
        (4, "generic_muscle_soft_tissue", "muscle", "#d95d39"),
        (5, "water_equivalent_fluid_or_soft_tissue", "water_equivalent_soft_tissue", "#4cc9f0"),
        (6, "liver", "liver", "#9d4edd"),
        (7, "kidneys", "kidney", "#f72585"),
        (8, "lungs", "lung_inflated", "#48cae4"),
        (9, "bladder", "water_equivalent_soft_tissue", "#f4d35e"),
        (10, "trabecular_bone", "trabecular_bone", "#e9ecef"),
        (11, "cortical_bone", "cortical_bone", "#ffffff"),
        (12, "brain_or_out_of_scope_soft_tissue", "water_equivalent_soft_tissue", "#80ed99"),
        *ABDOMINAL_REGION_SPECS,
    ]
    regions: list[MaterialRegion] = []
    for index, name, material_id, color in raw:
        hu, density, red = _target_midpoint(material_id, library)
        regions.append(
            MaterialRegion(
                index=index,
                name=name,
                material_id=material_id,
                target_hu_midpoint=hu,
                mass_density_g_cm3=density,
                relative_electron_density=red,
                color=color,
            )
        )
    return tuple(regions)


def _infer_body_mask(
    ct_hu: np.ndarray,
    labels: np.ndarray,
    body_threshold_hu: float,
    ndimage,
) -> np.ndarray:
    organ_mask = labels > 0
    raw = (ct_hu > body_threshold_hu) | organ_mask
    body = np.zeros(raw.shape, dtype=bool)
    structure_2d = np.ones((5, 5), dtype=bool)

    for z_index in range(raw.shape[2]):
        slice_mask = raw[:, :, z_index]
        if not slice_mask.any():
            continue

        closed = ndimage.binary_closing(slice_mask, structure=structure_2d, iterations=2)
        connected, count = ndimage.label(closed)
        if count == 0:
            continue

        organ_slice = organ_mask[:, :, z_index]
        if organ_slice.any():
            overlaps = np.bincount(connected[organ_slice].ravel(), minlength=count + 1)
            overlaps[0] = 0
            selected = int(overlaps.argmax())
            if overlaps[selected] == 0:
                sizes = np.bincount(connected.ravel(), minlength=count + 1)
                sizes[0] = 0
                selected = int(sizes.argmax())
        else:
            sizes = np.bincount(connected.ravel(), minlength=count + 1)
            sizes[0] = 0
            selected = int(sizes.argmax())

        filled = ndimage.binary_fill_holes(connected == selected)
        body[:, :, z_index] = filled

    body = ndimage.binary_closing(body, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    connected, count = ndimage.label(body, structure=np.ones((3, 3, 3), dtype=bool))
    if count <= 1:
        return body | organ_mask

    sizes = np.bincount(connected.ravel(), minlength=count + 1)
    sizes[0] = 0
    largest = int(sizes.argmax())
    return (connected == largest) | organ_mask


def _assign_material_labels(ct_hu: np.ndarray, labels: np.ndarray, body: np.ndarray) -> np.ndarray:
    material = np.zeros(ct_hu.shape, dtype=np.int16)
    unlabeled_body = body & (labels == 0)

    material[unlabeled_body & (ct_hu < -500)] = 1
    material[unlabeled_body & (ct_hu >= -500) & (ct_hu < -190)] = 2
    material[unlabeled_body & (ct_hu >= -190) & (ct_hu < -30)] = 3
    material[unlabeled_body & (ct_hu >= -30) & (ct_hu < 120)] = 4
    material[unlabeled_body & (ct_hu >= 120) & (ct_hu < 500)] = 10
    material[unlabeled_body & (ct_hu >= 500)] = 11

    material[labels == 1] = 6
    material[labels == 2] = 9
    material[labels == 3] = 8
    material[labels == 4] = 7
    material[(labels == 5) & (ct_hu < 500)] = 10
    material[(labels == 5) & (ct_hu >= 500)] = 11
    material[labels == 6] = 12
    return material


def _make_property_volume(material_labels: np.ndarray, regions: tuple[MaterialRegion, ...], attribute: str) -> np.ndarray:
    output = np.zeros(material_labels.shape, dtype=np.float32)
    for region in regions:
        output[material_labels == region.index] = float(getattr(region, attribute))
    return output


def _write_nifti(path: Path, data: np.ndarray, reference_image, nib) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _slice_indices(body: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(body)
    if len(coords) == 0:
        return tuple(value // 2 for value in body.shape)
    return tuple(int(round(float(np.median(coords[:, axis])))) for axis in range(3))


def _render_preview(
    path: Path,
    ct_hu: np.ndarray,
    material_labels: np.ndarray,
    body: np.ndarray,
    regions: tuple[MaterialRegion, ...],
    spacing: tuple[float, float, float],
) -> None:
    plt, ListedColormap, Patch, *_ = _import_dependencies()
    x_index, y_index, z_index = _slice_indices(body)
    colors = [region.color for region in sorted(regions, key=lambda item: item.index)]
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    for ax in axes.ravel():
        ax.set_facecolor("#f6f1e8")
        ax.axis("off")

    extent_xy = (0.0, ct_hu.shape[0] * spacing[0], 0.0, ct_hu.shape[1] * spacing[1])
    extent_xz = (0.0, ct_hu.shape[0] * spacing[0], 0.0, ct_hu.shape[2] * spacing[2])
    extent_yz = (0.0, ct_hu.shape[1] * spacing[1], 0.0, ct_hu.shape[2] * spacing[2])

    ct_views = [
        (np.rot90(ct_hu[:, :, z_index]), f"Source CT axial z={z_index}", extent_xy),
        (np.rot90(ct_hu[:, y_index, :]), f"Source CT coronal y={y_index}", extent_xz),
        (np.rot90(ct_hu[x_index, :, :]), f"Source CT sagittal x={x_index}", extent_yz),
    ]
    label_views = [
        (np.rot90(material_labels[:, :, z_index]), "Material map axial", extent_xy),
        (np.rot90(material_labels[:, y_index, :]), "Material map coronal", extent_xz),
        (np.rot90(material_labels[x_index, :, :]), "Material map sagittal", extent_yz),
    ]

    for ax, (view, title, extent) in zip(axes[0], ct_views):
        ax.imshow(
            np.clip(view, -1000, 1000),
            cmap="gray",
            vmin=-1000,
            vmax=1000,
            extent=extent,
            aspect="equal",
        )
        ax.set_title(title, color="#1e2a32", fontsize=10)

    for ax, (view, title, extent) in zip(axes[1], label_views):
        ax.imshow(
            view,
            cmap=cmap,
            vmin=0,
            vmax=max(region.index for region in regions),
            interpolation="nearest",
            extent=extent,
            aspect="equal",
        )
        ax.set_title(title, color="#1e2a32", fontsize=10)

    nonzero_indices = sorted(int(value) for value in np.unique(material_labels) if value > 0)
    by_index = {region.index: region for region in regions}
    handles = [
        Patch(facecolor=by_index[index].color, label=f"{index}: {by_index[index].name}")
        for index in nonzero_indices
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.5), fontsize=7)
    fig.suptitle(
        "Digital Torso Phantom Stage 001: CT-Derived Material Volume "
        f"(physical aspect, spacing {spacing[0]:.3g} x {spacing[1]:.3g} x {spacing[2]:.3g} mm)",
        fontsize=15,
        color="#13202a",
    )
    fig.tight_layout(rect=(0, 0, 0.90, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _compute_stats(
    material_labels: np.ndarray,
    ct_hu: np.ndarray,
    regions: tuple[MaterialRegion, ...],
    voxel_volume_mm3: float,
) -> tuple[MaterialRegionStats, ...]:
    stats: list[MaterialRegionStats] = []
    for region in regions:
        mask = material_labels == region.index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_hu = float(np.mean(ct_hu[mask])) if count else None
        stats.append(
            MaterialRegionStats(
                index=region.index,
                name=region.name,
                material_id=region.material_id,
                voxel_count=count,
                volume_cm3=count * voxel_volume_mm3 / 1000.0,
                mean_source_hu=mean_hu,
                target_hu_midpoint=region.target_hu_midpoint,
                mass_density_g_cm3=region.mass_density_g_cm3,
                relative_electron_density=region.relative_electron_density,
            )
        )
    return tuple(stats)


def _write_spec(path: Path, result: DigitalTorsoResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "coordinate_units": "mm",
        "volume_units": "cm3",
        "outputs": {
            "material_label_map": result.material_label_path,
            "mass_density_g_cm3": result.density_path,
            "relative_electron_density": result.relative_electron_density_path,
            "synthetic_hu": result.synthetic_hu_path,
            "body_mask": result.body_mask_path,
            "preview_png": result.preview_png_path,
        },
        "regions": [
            {
                "index": region.index,
                "name": region.name,
                "material_id": region.material_id,
                "target_hu_midpoint": region.target_hu_midpoint,
                "mass_density_g_cm3": region.mass_density_g_cm3,
                "relative_electron_density": region.relative_electron_density,
                "color": region.color,
            }
            for region in result.regions
        ],
        "stats": [
            {
                "index": stat.index,
                "name": stat.name,
                "material_id": stat.material_id,
                "voxel_count": stat.voxel_count,
                "volume_cm3": stat.volume_cm3,
                "mean_source_hu": stat.mean_source_hu,
                "target_hu_midpoint": stat.target_hu_midpoint,
                "mass_density_g_cm3": stat.mass_density_g_cm3,
                "relative_electron_density": stat.relative_electron_density,
            }
            for stat in result.stats
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: DigitalTorsoResult) -> str:
    lines = [
        "# Digital Torso Phantom Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Outputs",
        "",
        f"- Material label map: `{Path(result.material_label_path).name}`",
        f"- Mass-density map: `{Path(result.density_path).name}`",
        f"- Relative electron-density map: `{Path(result.relative_electron_density_path).name}`",
        f"- Synthetic HU target map: `{Path(result.synthetic_hu_path).name}`",
        f"- Body mask: `{Path(result.body_mask_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Construction Method",
        "",
        "- Inferred a torso/body envelope from CT HU thresholding plus organ-label inclusion.",
        "- Assigned unlabeled torso voxels from source HU ranges: internal air, lung-like low density, adipose, muscle/soft tissue, trabecular bone, and cortical bone.",
        "- Overrode the inferred map with CT-ORG organ labels for liver, bladder, lungs, kidneys, and bone.",
        "- Split CT-ORG bone into trabecular/cortical classes using a 500 HU threshold.",
        "",
        "## Volume Summary",
        "",
        f"- Body voxels: {result.body_voxels}",
        f"- Body volume: {result.body_volume_cm3:.2f} cm3",
        f"- Voxel volume: {result.voxel_volume_mm3:.3f} mm3",
        "",
        "## Region Material Stats",
        "",
        "| index | region | material | voxels | volume cm3 | mean source HU | target HU | density | RED |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for stat in result.stats:
        mean_hu = "n/a" if stat.mean_source_hu is None else f"{stat.mean_source_hu:.1f}"
        lines.append(
            f"| {stat.index} | {stat.name} | {stat.material_id} | {stat.voxel_count} | "
            f"{stat.volume_cm3:.2f} | {mean_hu} | {stat.target_hu_midpoint:.1f} | "
            f"{stat.mass_density_g_cm3:.3f} | {stat.relative_electron_density:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Digital Phantom Notes",
            "",
            "- This is a voxel/material digital phantom, not a print-prep model.",
            "- The CT-ORG case does not include every torso organ or a curated external-body contour, so the body envelope is inferred and should be visually reviewed.",
            "- The material maps are suitable for early CT/radiotherapy simulation experiments, but material targets still need protocol-specific calibration before any experimental validation.",
            "- The ImageTBAD vascular module remains separate until we register or embed the vascular geometry into this CT-ORG torso coordinate system.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_digital_torso_phantom(
    ct_path: str | Path,
    labels_path: str | Path,
    labelmap_path: str | Path,
    materials_path: str | Path,
    output_dir: str | Path,
    case_id: str = "ct_org_case0",
    body_threshold_hu: float = -500.0,
    report_path: str | Path | None = None,
) -> DigitalTorsoResult:
    _, _, _, nib, ndimage, *_ = _import_dependencies()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    labelmap = _load_labelmap(labelmap_path)
    library = load_material_library(materials_path)
    regions = _region_table(library)

    ct_image = nib.load(str(ct_path))
    labels_image = nib.load(str(labels_path))
    ct_hu = np.asanyarray(ct_image.dataobj).astype(np.float32)
    raw_labels = np.asanyarray(labels_image.dataobj)
    labels = np.rint(raw_labels).astype(np.int16)
    if ct_hu.shape != labels.shape:
        raise ValueError(f"CT and label shapes differ: {ct_hu.shape} vs {labels.shape}")

    spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))
    body = _infer_body_mask(ct_hu, labels, body_threshold_hu=body_threshold_hu, ndimage=ndimage)
    material_labels = _assign_material_labels(ct_hu, labels, body)
    density = _make_property_volume(material_labels, regions, "mass_density_g_cm3")
    red = _make_property_volume(material_labels, regions, "relative_electron_density")
    synthetic_hu = _make_property_volume(material_labels, regions, "target_hu_midpoint")

    present_labels = sorted(int(value) for value in np.unique(labels) if value != 0)
    notes.append(
        "ct_org_labels_present="
        + ",".join(f"{value}:{labelmap.get(value, {}).get('name', 'unknown')}" for value in present_labels)
    )
    notes.append(f"body_threshold_hu={body_threshold_hu:g}")

    material_path = output / f"{case_id}_torso_material_labels_v001.nii.gz"
    density_path = output / f"{case_id}_torso_mass_density_g_cm3_v001.nii.gz"
    red_path = output / f"{case_id}_torso_relative_electron_density_v001.nii.gz"
    hu_path = output / f"{case_id}_torso_synthetic_hu_v001.nii.gz"
    body_path = output / f"{case_id}_torso_body_mask_v001.nii.gz"
    preview_png = output / f"{case_id}_torso_material_preview_v001.png"
    spec_yaml = output / f"{case_id}_torso_material_spec_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_digital_torso_report_v001.md"

    _write_nifti(material_path, material_labels, ct_image, nib)
    _write_nifti(density_path, density, ct_image, nib)
    _write_nifti(red_path, red, ct_image, nib)
    _write_nifti(hu_path, synthetic_hu, ct_image, nib)
    _write_nifti(body_path, body.astype(np.uint8), ct_image, nib)
    _render_preview(preview_png, ct_hu, material_labels, body, regions, spacing)

    stats = _compute_stats(material_labels, ct_hu, regions, voxel_volume_mm3)
    result = DigitalTorsoResult(
        case_id=case_id,
        output_dir=str(output),
        material_label_path=str(material_path),
        density_path=str(density_path),
        relative_electron_density_path=str(red_path),
        synthetic_hu_path=str(hu_path),
        body_mask_path=str(body_path),
        preview_png_path=str(preview_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        body_voxels=int(body.sum()),
        body_volume_cm3=float(body.sum() * voxel_volume_mm3 / 1000.0),
        voxel_volume_mm3=voxel_volume_mm3,
        regions=regions,
        stats=stats,
        notes=tuple(notes),
    )

    _write_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_digital_torso_result(result: DigitalTorsoResult) -> str:
    return _format_report(result)
