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
            "Combined digital phantom generation requires nibabel, scipy, matplotlib, and PyYAML."
        ) from exc
    return plt, ListedColormap, Patch, nib, ndimage, yaml


@dataclass(frozen=True)
class CombinedRegion:
    index: int
    name: str
    material_id: str
    target_hu_midpoint: float
    mass_density_g_cm3: float
    relative_electron_density: float
    color: str


@dataclass(frozen=True)
class BoundaryRegion:
    label: int
    name: str
    role: str
    center_mm: tuple[float, float, float]
    center_ijk: tuple[float, float, float]
    radius_mm: float
    voxel_count: int


@dataclass(frozen=True)
class CombinedDigitalPhantomResult:
    case_id: str
    output_dir: str
    blood_material_labels_path: str
    blood_density_path: str
    blood_relative_electron_density_path: str
    blood_synthetic_hu_path: str
    contrast_material_labels_path: str
    contrast_density_path: str
    contrast_relative_electron_density_path: str
    contrast_synthetic_hu_path: str
    vascular_fluid_mask_path: str
    vessel_wall_mask_path: str
    flow_boundary_labels_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    source_vascular_center_mm: tuple[float, float, float]
    target_torso_center_mm: tuple[float, float, float]
    vascular_scale: float
    vessel_wall_thickness_mm: float
    vascular_fluid_voxels: int
    vessel_wall_voxels: int
    vascular_fluid_volume_cm3: float
    vessel_wall_volume_cm3: float
    vascular_outside_body_fraction_before_clip: float
    vascular_connected_components: int
    regions: tuple[CombinedRegion, ...]
    boundaries: tuple[BoundaryRegion, ...]
    notes: tuple[str, ...]


def _material_midpoint(material_id: str, library: MaterialLibrary) -> tuple[float, float, float]:
    material = library.by_id[material_id]
    hu_min, hu_max = material.target_hu
    return (
        (hu_min + hu_max) / 2.0,
        material.target_mass_density_g_cm3,
        material.target_relative_electron_density,
    )


def _combined_regions(library: MaterialLibrary) -> tuple[CombinedRegion, ...]:
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
        (13, "vessel_wall", "vessel_wall", "#ff9f1c"),
        (14, "blood_equivalent_fluid", "blood_equivalent_fluid", "#0077b6"),
        (15, "contrast_filled_blood", "contrast_filled_blood", "#ef476f"),
        *ABDOMINAL_REGION_SPECS,
    ]
    regions: list[CombinedRegion] = []
    for index, name, material_id, color in raw:
        hu, density, red = _material_midpoint(material_id, library)
        regions.append(
            CombinedRegion(
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


def _write_nifti(path: Path, data: np.ndarray, reference_image, nib) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _mask_bbox_center_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Cannot compute center for an empty mask")
    mins = coords.min(axis=0) * np.array(spacing)
    maxs = coords.max(axis=0) * np.array(spacing)
    center = (mins + maxs) / 2.0
    return tuple(float(value) for value in center)


def _auto_target_center_mm(body_mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    coords = np.argwhere(body_mask)
    if len(coords) == 0:
        shape = np.array(body_mask.shape, dtype=float)
        return tuple(float(value) for value in (shape * np.array(spacing) / 2.0))

    physical = coords * np.array(spacing)
    mins = physical.min(axis=0)
    maxs = physical.max(axis=0)
    median = np.median(physical, axis=0)
    target = np.array(
        [
            median[0],
            median[1],
            mins[2] + 0.55 * (maxs[2] - mins[2]),
        ]
    )
    return tuple(float(value) for value in target)


def _resample_vascular_to_torso(
    vascular_mask: np.ndarray,
    vascular_spacing: tuple[float, float, float],
    torso_shape: tuple[int, int, int],
    torso_spacing: tuple[float, float, float],
    source_center_mm: tuple[float, float, float],
    target_center_mm: tuple[float, float, float],
    scale: float,
    ndimage,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError("vascular scale must be positive")

    source_spacing = np.array(vascular_spacing, dtype=float)
    target_spacing = np.array(torso_spacing, dtype=float)
    source_center_index = np.array(source_center_mm, dtype=float) / source_spacing
    target_center = np.array(target_center_mm, dtype=float)

    matrix = np.diag(target_spacing / (source_spacing * scale))
    offset = source_center_index - target_center / (source_spacing * scale)
    resampled = ndimage.affine_transform(
        vascular_mask.astype(np.uint8),
        matrix=matrix,
        offset=offset,
        output_shape=torso_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return resampled.astype(bool)


def _property_volume(material_labels: np.ndarray, regions: tuple[CombinedRegion, ...], attribute: str) -> np.ndarray:
    output = np.zeros(material_labels.shape, dtype=np.float32)
    for region in regions:
        output[material_labels == region.index] = float(getattr(region, attribute))
    return output


def _paint_sphere(
    labels: np.ndarray,
    center_mm: tuple[float, float, float],
    radius_mm: float,
    value: int,
    spacing: tuple[float, float, float],
    restrict_mask: np.ndarray | None = None,
) -> int:
    center = np.array(center_mm, dtype=float)
    spacing_array = np.array(spacing, dtype=float)
    center_index = center / spacing_array
    radius_index = np.ceil(radius_mm / spacing_array).astype(int) + 1
    mins = np.maximum(np.floor(center_index).astype(int) - radius_index, 0)
    maxs = np.minimum(np.ceil(center_index).astype(int) + radius_index + 1, np.array(labels.shape))
    if np.any(maxs <= mins):
        return 0

    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    grids = np.meshgrid(
        *[
            np.arange(mins[axis], maxs[axis], dtype=float) * spacing_array[axis]
            for axis in range(3)
        ],
        indexing="ij",
    )
    distance_sq = sum((grid - center[axis]) ** 2 for axis, grid in enumerate(grids))
    sphere = distance_sq <= radius_mm**2
    if restrict_mask is not None:
        sphere &= restrict_mask[slices]
    labels[slices][sphere] = value
    return int(sphere.sum())


def _boundary_center_ijk(center_mm: tuple[float, float, float], spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in (np.array(center_mm) / np.array(spacing)))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    return yaml.safe_load(Path(path).read_text())


def _transform_point(
    point_mm: tuple[float, float, float] | list[float],
    source_center_mm: tuple[float, float, float],
    target_center_mm: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    point = np.array(point_mm, dtype=float)
    source_center = np.array(source_center_mm, dtype=float)
    target_center = np.array(target_center_mm, dtype=float)
    transformed = target_center + (point - source_center) * scale
    return tuple(float(value) for value in transformed)


def _build_boundary_labels(
    flow_loop_spec: dict[str, Any],
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    source_center_mm: tuple[float, float, float],
    target_center_mm: tuple[float, float, float],
    scale: float,
    vascular_fluid: np.ndarray,
    vessel_wall: np.ndarray,
) -> tuple[np.ndarray, tuple[BoundaryRegion, ...]]:
    boundary = np.zeros(shape, dtype=np.int16)
    boundary[vessel_wall] = 5

    roles = {
        "z_min_port": ("inlet", 1),
        "z_max_port": ("outlet", 2),
    }
    boundaries: list[BoundaryRegion] = []

    for adapter in flow_loop_spec.get("adapters", []):
        adapter_id = str(adapter["id"])
        role, label = roles.get(adapter_id, (adapter_id, 0))
        if label:
            center = _transform_point(adapter["center_mm"], source_center_mm, target_center_mm, scale)
            radius = max(3.0, float(adapter["tube_inner_diameter_mm"]) * scale / 2.0)
            count = _paint_sphere(boundary, center, radius, label, spacing, restrict_mask=vascular_fluid)
            if count == 0:
                count = _paint_sphere(boundary, center, radius * 1.8, label, spacing, restrict_mask=vascular_fluid)
            boundaries.append(
                BoundaryRegion(
                    label=label,
                    name=adapter_id,
                    role=role,
                    center_mm=center,
                    center_ijk=_boundary_center_ijk(center, spacing),
                    radius_mm=radius,
                    voxel_count=count,
                )
            )

        tap_center = _transform_point(adapter["pressure_tap_center_mm"], source_center_mm, target_center_mm, scale)
        tap_label = 3 if adapter_id == "z_min_port" else 4 if adapter_id == "z_max_port" else 0
        if tap_label:
            tap_radius = max(2.0, float(adapter["pressure_tap_diameter_mm"]) * scale / 2.0)
            count = _paint_sphere(boundary, tap_center, tap_radius, tap_label, spacing)
            boundaries.append(
                BoundaryRegion(
                    label=tap_label,
                    name=f"{adapter_id}_pressure_tap",
                    role="pressure_tap",
                    center_mm=tap_center,
                    center_ijk=_boundary_center_ijk(tap_center, spacing),
                    radius_mm=tap_radius,
                    voxel_count=count,
                )
            )

    wall_count = int((boundary == 5).sum())
    wall_center = tuple(float(value) for value in (np.argwhere(vessel_wall).mean(axis=0) * np.array(spacing))) if wall_count else (0.0, 0.0, 0.0)
    boundaries.append(
        BoundaryRegion(
            label=5,
            name="vessel_wall",
            role="wall",
            center_mm=wall_center,
            center_ijk=_boundary_center_ijk(wall_center, spacing),
            radius_mm=0.0,
            voxel_count=wall_count,
        )
    )
    return boundary, tuple(boundaries)


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return tuple(value // 2 for value in mask.shape)
    return tuple(int(round(float(np.median(coords[:, axis])))) for axis in range(3))


def _render_combined_preview(
    path: Path,
    source_ct: np.ndarray,
    blood_material_labels: np.ndarray,
    vascular_fluid: np.ndarray,
    vessel_wall: np.ndarray,
    regions: tuple[CombinedRegion, ...],
    spacing: tuple[float, float, float],
) -> None:
    plt, ListedColormap, Patch, *_ = _import_dependencies()
    x_index, y_index, z_index = _slice_indices(vascular_fluid)
    colors = [region.color for region in sorted(regions, key=lambda item: item.index)]
    cmap = ListedColormap(colors)
    vmax = max(region.index for region in regions)

    extent_xy = (0.0, source_ct.shape[0] * spacing[0], 0.0, source_ct.shape[1] * spacing[1])
    extent_xz = (0.0, source_ct.shape[0] * spacing[0], 0.0, source_ct.shape[2] * spacing[2])
    extent_yz = (0.0, source_ct.shape[1] * spacing[1], 0.0, source_ct.shape[2] * spacing[2])
    views = [
        (
            np.rot90(source_ct[:, :, z_index]),
            np.rot90(blood_material_labels[:, :, z_index]),
            np.rot90(vascular_fluid[:, :, z_index]),
            np.rot90(vessel_wall[:, :, z_index]),
            f"Axial z={z_index}",
            extent_xy,
        ),
        (
            np.rot90(source_ct[:, y_index, :]),
            np.rot90(blood_material_labels[:, y_index, :]),
            np.rot90(vascular_fluid[:, y_index, :]),
            np.rot90(vessel_wall[:, y_index, :]),
            f"Coronal y={y_index}",
            extent_xz,
        ),
        (
            np.rot90(source_ct[x_index, :, :]),
            np.rot90(blood_material_labels[x_index, :, :]),
            np.rot90(vascular_fluid[x_index, :, :]),
            np.rot90(vessel_wall[x_index, :, :]),
            f"Sagittal x={x_index}",
            extent_yz,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    for ax in axes.ravel():
        ax.set_facecolor("#f6f1e8")
        ax.axis("off")

    for ax, (ct_view, _, fluid_view, wall_view, title, extent) in zip(axes[0], views):
        ax.imshow(np.clip(ct_view, -1000, 1000), cmap="gray", vmin=-1000, vmax=1000, extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~wall_view, wall_view), cmap=ListedColormap(["#ff9f1c"]), alpha=0.55, extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~fluid_view, fluid_view), cmap=ListedColormap(["#0077b6"]), alpha=0.70, extent=extent, aspect="equal")
        ax.set_title(f"CT + vascular overlay {title}", color="#1e2a32", fontsize=10)

    for ax, (_, material_view, fluid_view, wall_view, title, extent) in zip(axes[1], views):
        ax.imshow(material_view, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~wall_view, wall_view), cmap=ListedColormap(["#ff9f1c"]), alpha=0.35, extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~fluid_view, fluid_view), cmap=ListedColormap(["#003566"]), alpha=0.35, extent=extent, aspect="equal")
        ax.set_title(f"Combined material map {title}", color="#1e2a32", fontsize=10)

    handles = [
        Patch(facecolor="#0077b6", label="blood-equivalent vascular fluid"),
        Patch(facecolor="#ff9f1c", label="vessel wall"),
        Patch(facecolor="#ffd166", label="adipose envelope"),
        Patch(facecolor="#d95d39", label="muscle / soft tissue"),
        Patch(facecolor="#48cae4", label="lungs"),
        Patch(facecolor="#9d4edd", label="liver"),
        Patch(facecolor="#e9ecef", label="trabecular bone"),
        Patch(facecolor="#ffffff", label="cortical bone"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.5), fontsize=7)
    fig.suptitle(
        "Combined Digital Torso + Vascular Phantom "
        f"(engineering placement, spacing {spacing[0]:.3g} x {spacing[1]:.3g} x {spacing[2]:.3g} mm)",
        fontsize=15,
        color="#13202a",
    )
    fig.tight_layout(rect=(0, 0, 0.90, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_spec(path: Path, result: CombinedDigitalPhantomResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "coordinate_units": "mm",
        "placement": {
            "type": "engineering_embedding_not_patient_registration",
            "source_vascular_center_mm": list(result.source_vascular_center_mm),
            "target_torso_center_mm": list(result.target_torso_center_mm),
            "vascular_scale": result.vascular_scale,
            "vessel_wall_thickness_mm": result.vessel_wall_thickness_mm,
            "vascular_outside_body_fraction_before_clip": result.vascular_outside_body_fraction_before_clip,
        },
        "outputs": {
            "blood_material_labels": result.blood_material_labels_path,
            "blood_mass_density_g_cm3": result.blood_density_path,
            "blood_relative_electron_density": result.blood_relative_electron_density_path,
            "blood_synthetic_hu": result.blood_synthetic_hu_path,
            "contrast_material_labels": result.contrast_material_labels_path,
            "contrast_mass_density_g_cm3": result.contrast_density_path,
            "contrast_relative_electron_density": result.contrast_relative_electron_density_path,
            "contrast_synthetic_hu": result.contrast_synthetic_hu_path,
            "vascular_fluid_mask": result.vascular_fluid_mask_path,
            "vessel_wall_mask": result.vessel_wall_mask_path,
            "flow_boundary_labels": result.flow_boundary_labels_path,
            "preview_png": result.preview_png_path,
        },
        "flow_boundary_labels": [
            {
                "label": boundary.label,
                "name": boundary.name,
                "role": boundary.role,
                "center_mm": list(boundary.center_mm),
                "center_ijk": list(boundary.center_ijk),
                "radius_mm": boundary.radius_mm,
                "voxel_count": boundary.voxel_count,
            }
            for boundary in result.boundaries
        ],
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
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: CombinedDigitalPhantomResult) -> str:
    lines = [
        "# Combined Digital Torso + Vascular Phantom Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Outputs",
        "",
        f"- Blood-filled material labels: `{Path(result.blood_material_labels_path).name}`",
        f"- Blood-filled density/RED/HU maps: `{Path(result.blood_density_path).name}`, `{Path(result.blood_relative_electron_density_path).name}`, `{Path(result.blood_synthetic_hu_path).name}`",
        f"- Contrast-filled material labels: `{Path(result.contrast_material_labels_path).name}`",
        f"- Contrast-filled density/RED/HU maps: `{Path(result.contrast_density_path).name}`, `{Path(result.contrast_relative_electron_density_path).name}`, `{Path(result.contrast_synthetic_hu_path).name}`",
        f"- Vascular fluid mask: `{Path(result.vascular_fluid_mask_path).name}`",
        f"- Vessel wall mask: `{Path(result.vessel_wall_mask_path).name}`",
        f"- Flow boundary labels: `{Path(result.flow_boundary_labels_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Placement",
        "",
        "- Placement type: engineering embedding, not anatomical patient-to-patient registration.",
        f"- Source vascular center mm: {', '.join(f'{value:.2f}' for value in result.source_vascular_center_mm)}",
        f"- Target torso center mm: {', '.join(f'{value:.2f}' for value in result.target_torso_center_mm)}",
        f"- Vascular scale: {result.vascular_scale:.3f}",
        f"- Vessel wall thickness: {result.vessel_wall_thickness_mm:.2f} mm",
        f"- Vascular outside-body fraction before clipping: {result.vascular_outside_body_fraction_before_clip:.4f}",
        "",
        "## Vascular Volume Summary",
        "",
        f"- Fluid voxels: {result.vascular_fluid_voxels}",
        f"- Vessel wall voxels: {result.vessel_wall_voxels}",
        f"- Fluid volume: {result.vascular_fluid_volume_cm3:.2f} cm3",
        f"- Vessel wall volume: {result.vessel_wall_volume_cm3:.2f} cm3",
        f"- Fluid connected components: {result.vascular_connected_components}",
        "",
        "## Flow Boundary Labels",
        "",
        "| label | name | role | center mm | center ijk | radius mm | voxels |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for boundary in result.boundaries:
        center_mm = ", ".join(f"{value:.2f}" for value in boundary.center_mm)
        center_ijk = ", ".join(f"{value:.1f}" for value in boundary.center_ijk)
        lines.append(
            f"| {boundary.label} | {boundary.name} | {boundary.role} | {center_mm} | "
            f"{center_ijk} | {boundary.radius_mm:.2f} | {boundary.voxel_count} |"
        )

    lines.extend(
        [
            "",
            "## Material Labels Added",
            "",
            "- `13`: vessel wall, using the `vessel_wall` material target.",
            "- `14`: blood-equivalent circulating fluid.",
            "- `15`: contrast-filled blood for CTA-style contrast scenarios.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_combined_digital_phantom(
    torso_material_labels_path: str | Path,
    torso_body_mask_path: str | Path,
    source_ct_path: str | Path,
    vascular_labels_path: str | Path,
    flow_loop_spec_path: str | Path,
    materials_path: str | Path,
    output_dir: str | Path,
    case_id: str = "ct_org_case0_imagetbad_case125",
    vascular_label_id: int = 1,
    target_center_mm: tuple[float, float, float] | None = None,
    vascular_scale: float = 1.0,
    vessel_wall_thickness_mm: float = 2.0,
    report_path: str | Path | None = None,
) -> CombinedDigitalPhantomResult:
    _, _, _, nib, ndimage, *_ = _import_dependencies()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    library = load_material_library(materials_path)
    regions = _combined_regions(library)
    by_index = {region.index: region for region in regions}

    material_image = nib.load(str(torso_material_labels_path))
    body_image = nib.load(str(torso_body_mask_path))
    ct_image = nib.load(str(source_ct_path))
    vascular_image = nib.load(str(vascular_labels_path))

    torso_material = np.asanyarray(material_image.dataobj).astype(np.int16)
    body_mask = np.asanyarray(body_image.dataobj) > 0
    source_ct = np.asanyarray(ct_image.dataobj).astype(np.float32)
    vascular_labels = np.rint(np.asanyarray(vascular_image.dataobj)).astype(np.int16)
    vascular_source = vascular_labels == vascular_label_id
    if not vascular_source.any():
        raise ValueError(f"Vascular label {vascular_label_id} was not found in {vascular_labels_path}")
    if source_ct.shape != torso_material.shape:
        raise ValueError(f"Source CT and torso material map differ: {source_ct.shape} vs {torso_material.shape}")

    torso_spacing = tuple(float(value) for value in material_image.header.get_zooms()[:3])
    vascular_spacing = tuple(float(value) for value in vascular_image.header.get_zooms()[:3])
    voxel_volume_cm3 = float(np.prod(torso_spacing) / 1000.0)
    source_center_mm = _mask_bbox_center_mm(vascular_source, vascular_spacing)
    target = target_center_mm if target_center_mm is not None else _auto_target_center_mm(body_mask, torso_spacing)

    vascular_resampled = _resample_vascular_to_torso(
        vascular_source,
        vascular_spacing=vascular_spacing,
        torso_shape=torso_material.shape,
        torso_spacing=torso_spacing,
        source_center_mm=source_center_mm,
        target_center_mm=target,
        scale=vascular_scale,
        ndimage=ndimage,
    )
    vascular_resampled = ndimage.binary_closing(
        vascular_resampled,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=1,
    )
    outside_before_clip = int((vascular_resampled & ~body_mask).sum())
    total_before_clip = int(vascular_resampled.sum())
    outside_fraction = outside_before_clip / total_before_clip if total_before_clip else 0.0
    vascular_fluid = vascular_resampled & body_mask

    connected, component_count = ndimage.label(vascular_fluid, structure=np.ones((3, 3, 3), dtype=bool))
    if component_count > 1:
        sizes = np.bincount(connected.ravel(), minlength=component_count + 1)
        sizes[0] = 0
        largest = int(sizes.argmax())
        notes.append(f"vascular_components_before_largest_filter={component_count}")
        vascular_fluid = connected == largest
        component_count = 1

    distance_to_fluid = ndimage.distance_transform_edt(~vascular_fluid, sampling=torso_spacing)
    vessel_wall = (distance_to_fluid <= vessel_wall_thickness_mm) & ~vascular_fluid & body_mask

    blood_labels = torso_material.copy()
    contrast_labels = torso_material.copy()
    blood_labels[vessel_wall] = 13
    contrast_labels[vessel_wall] = 13
    blood_labels[vascular_fluid] = 14
    contrast_labels[vascular_fluid] = 15

    blood_density = _property_volume(blood_labels, regions, "mass_density_g_cm3")
    blood_red = _property_volume(blood_labels, regions, "relative_electron_density")
    blood_hu = _property_volume(blood_labels, regions, "target_hu_midpoint")
    contrast_density = _property_volume(contrast_labels, regions, "mass_density_g_cm3")
    contrast_red = _property_volume(contrast_labels, regions, "relative_electron_density")
    contrast_hu = _property_volume(contrast_labels, regions, "target_hu_midpoint")

    flow_loop_spec = _load_yaml(flow_loop_spec_path)
    boundary_labels, boundaries = _build_boundary_labels(
        flow_loop_spec,
        shape=torso_material.shape,
        spacing=torso_spacing,
        source_center_mm=source_center_mm,
        target_center_mm=target,
        scale=vascular_scale,
        vascular_fluid=vascular_fluid,
        vessel_wall=vessel_wall,
    )

    base = output / case_id
    blood_labels_path = base.with_name(f"{case_id}_combined_material_labels_blood_v001.nii.gz")
    blood_density_path = base.with_name(f"{case_id}_combined_mass_density_blood_v001.nii.gz")
    blood_red_path = base.with_name(f"{case_id}_combined_relative_electron_density_blood_v001.nii.gz")
    blood_hu_path = base.with_name(f"{case_id}_combined_synthetic_hu_blood_v001.nii.gz")
    contrast_labels_path = base.with_name(f"{case_id}_combined_material_labels_contrast_v001.nii.gz")
    contrast_density_path = base.with_name(f"{case_id}_combined_mass_density_contrast_v001.nii.gz")
    contrast_red_path = base.with_name(f"{case_id}_combined_relative_electron_density_contrast_v001.nii.gz")
    contrast_hu_path = base.with_name(f"{case_id}_combined_synthetic_hu_contrast_v001.nii.gz")
    vascular_mask_path = base.with_name(f"{case_id}_vascular_fluid_mask_v001.nii.gz")
    wall_mask_path = base.with_name(f"{case_id}_vessel_wall_mask_v001.nii.gz")
    boundary_path = base.with_name(f"{case_id}_flow_boundary_labels_v001.nii.gz")
    preview_png = base.with_name(f"{case_id}_combined_preview_v001.png")
    spec_yaml = base.with_name(f"{case_id}_combined_spec_v001.yaml")
    report = Path(report_path) if report_path else base.with_name(f"{case_id}_combined_report_v001.md")

    _write_nifti(blood_labels_path, blood_labels, material_image, nib)
    _write_nifti(blood_density_path, blood_density, material_image, nib)
    _write_nifti(blood_red_path, blood_red, material_image, nib)
    _write_nifti(blood_hu_path, blood_hu, material_image, nib)
    _write_nifti(contrast_labels_path, contrast_labels, material_image, nib)
    _write_nifti(contrast_density_path, contrast_density, material_image, nib)
    _write_nifti(contrast_red_path, contrast_red, material_image, nib)
    _write_nifti(contrast_hu_path, contrast_hu, material_image, nib)
    _write_nifti(vascular_mask_path, vascular_fluid.astype(np.uint8), material_image, nib)
    _write_nifti(wall_mask_path, vessel_wall.astype(np.uint8), material_image, nib)
    _write_nifti(boundary_path, boundary_labels, material_image, nib)

    notes.append("vascular_embedding_is_engineering_placement_not_patient_registration")
    if outside_fraction > 0:
        notes.append(f"vascular_voxels_clipped_outside_body_fraction={outside_fraction:.4f}")
    notes.append(f"blood_fluid_label={by_index[14].material_id}")
    notes.append(f"contrast_fluid_label={by_index[15].material_id}")

    result = CombinedDigitalPhantomResult(
        case_id=case_id,
        output_dir=str(output),
        blood_material_labels_path=str(blood_labels_path),
        blood_density_path=str(blood_density_path),
        blood_relative_electron_density_path=str(blood_red_path),
        blood_synthetic_hu_path=str(blood_hu_path),
        contrast_material_labels_path=str(contrast_labels_path),
        contrast_density_path=str(contrast_density_path),
        contrast_relative_electron_density_path=str(contrast_red_path),
        contrast_synthetic_hu_path=str(contrast_hu_path),
        vascular_fluid_mask_path=str(vascular_mask_path),
        vessel_wall_mask_path=str(wall_mask_path),
        flow_boundary_labels_path=str(boundary_path),
        preview_png_path=str(preview_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        source_vascular_center_mm=source_center_mm,
        target_torso_center_mm=target,
        vascular_scale=vascular_scale,
        vessel_wall_thickness_mm=vessel_wall_thickness_mm,
        vascular_fluid_voxels=int(vascular_fluid.sum()),
        vessel_wall_voxels=int(vessel_wall.sum()),
        vascular_fluid_volume_cm3=float(vascular_fluid.sum() * voxel_volume_cm3),
        vessel_wall_volume_cm3=float(vessel_wall.sum() * voxel_volume_cm3),
        vascular_outside_body_fraction_before_clip=float(outside_fraction),
        vascular_connected_components=int(component_count),
        regions=regions,
        boundaries=boundaries,
        notes=tuple(notes),
    )

    _render_combined_preview(
        preview_png,
        source_ct=source_ct,
        blood_material_labels=blood_labels,
        vascular_fluid=vascular_fluid,
        vessel_wall=vessel_wall,
        regions=regions,
        spacing=torso_spacing,
    )
    _write_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_combined_digital_phantom_result(result: CombinedDigitalPhantomResult) -> str:
    return _format_report(result)
