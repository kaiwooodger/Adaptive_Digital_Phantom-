from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .combined import _combined_regions, _property_volume, _write_nifti
from .materials import load_material_library
from .vessel_radius_profile import edge_radius_at_fraction


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
            "Vascular network voxelization requires nibabel, scipy, matplotlib, and PyYAML."
        ) from exc
    return plt, ListedColormap, Patch, nib, ndimage, yaml


@dataclass(frozen=True)
class VoxelizedBoundary:
    label: int
    node_id: str
    role: str
    center_mm: tuple[float, float, float]
    center_ijk: tuple[float, float, float]
    radius_mm: float
    voxel_count: int


@dataclass(frozen=True)
class VascularNetworkVoxelizationResult:
    case_id: str
    output_dir: str
    graph_yaml_path: str
    source_combined_labels_path: str
    arterial_lumen_mask_path: str
    venous_lumen_mask_path: str
    combined_lumen_mask_path: str
    collision_mask_path: str
    collision_owner_labels_path: str
    flow_domain_labels_path: str
    vessel_wall_mask_path: str
    boundary_labels_path: str
    blood_material_labels_path: str | None
    blood_density_path: str | None
    blood_relative_electron_density_path: str | None
    blood_synthetic_hu_path: str | None
    contrast_material_labels_path: str | None
    contrast_density_path: str | None
    contrast_relative_electron_density_path: str | None
    contrast_synthetic_hu_path: str | None
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    arterial_voxels: int
    venous_voxels: int
    overlap_voxels: int
    overlap_voxels_before_cleanup: int
    overlap_voxels_after_cleanup: int
    combined_lumen_voxels: int
    vessel_wall_voxels: int
    arterial_volume_cm3: float
    venous_volume_cm3: float
    combined_lumen_volume_cm3: float
    vessel_wall_volume_cm3: float
    outside_body_fraction_before_clip: float
    connected_components: int
    arterial_components: int
    venous_components: int
    boundaries: tuple[VoxelizedBoundary, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _spacing_from_image(image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _center_ijk(center_mm: tuple[float, float, float], spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in (np.array(center_mm, dtype=float) / np.array(spacing, dtype=float)))


def _paint_sphere(
    target: np.ndarray,
    center_mm: tuple[float, float, float],
    radius_mm: float,
    spacing: tuple[float, float, float],
    value: int | bool = True,
    restrict_mask: np.ndarray | None = None,
) -> int:
    center = np.array(center_mm, dtype=float)
    spacing_array = np.array(spacing, dtype=float)
    center_index = center / spacing_array
    radius_index = np.ceil(radius_mm / spacing_array).astype(int) + 1
    mins = np.maximum(np.floor(center_index).astype(int) - radius_index, 0)
    maxs = np.minimum(np.ceil(center_index).astype(int) + radius_index + 1, np.array(target.shape))
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
    view = target[slices]
    before = int(np.count_nonzero(view[sphere])) if target.dtype == bool else int(np.count_nonzero(view[sphere] == value))
    view[sphere] = value
    after = int(np.count_nonzero(sphere))
    return max(after - before, 0) if target.dtype == bool else after


def _mark_seed(
    target: np.ndarray,
    center_mm: tuple[float, float, float],
    spacing: tuple[float, float, float],
    restrict_mask: np.ndarray | None = None,
) -> None:
    index = np.rint(np.array(center_mm, dtype=float) / np.array(spacing, dtype=float)).astype(int)
    if np.any(index < 0) or np.any(index >= np.array(target.shape)):
        return
    location = tuple(int(value) for value in index)
    if restrict_mask is not None and not bool(restrict_mask[location]):
        return
    target[location] = True


def _edge_samples(edge: dict[str, Any], sample_step_mm: float) -> tuple[tuple[tuple[float, float, float], float], ...]:
    points = np.array(edge["polyline_mm"], dtype=float)
    if len(points) < 2:
        return ()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 0:
        return ()

    samples: list[tuple[tuple[float, float, float], float]] = []
    running = 0.0
    step = max(sample_step_mm, 0.1)

    for segment_index, length in enumerate(segment_lengths):
        if length <= 1e-6:
            continue
        count = max(1, int(np.ceil(float(length) / step)))
        for local_index in range(count + 1):
            if segment_index > 0 and local_index == 0:
                continue
            local_t = local_index / count
            edge_t = (running + local_t * float(length)) / total_length
            point = points[segment_index] + (points[segment_index + 1] - points[segment_index]) * local_t
            radius = edge_radius_at_fraction(edge, edge_t)
            samples.append(((float(point[0]), float(point[1]), float(point[2])), float(radius)))
        running += float(length)
    return tuple(samples)


def _paint_edges(
    graph: dict[str, Any],
    vessel_type: str,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    sample_step_mm: float,
    restrict_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    clipped = 0
    unclipped = np.zeros(shape, dtype=bool)
    clipped_mask = np.zeros(shape, dtype=bool)
    seed_mask = np.zeros(shape, dtype=bool)

    for edge in graph.get("edges", []):
        if str(edge.get("vessel_type")) != vessel_type:
            continue
        for center, radius in _edge_samples(edge, sample_step_mm=sample_step_mm):
            _paint_sphere(unclipped, center, radius, spacing, True)
            _paint_sphere(clipped_mask, center, radius, spacing, True, restrict_mask=restrict_mask)
            _mark_seed(seed_mask, center, spacing, restrict_mask=restrict_mask)

    if restrict_mask is not None:
        clipped = int(unclipped.sum() - clipped_mask.sum())
    return clipped_mask, seed_mask & clipped_mask, clipped


def _cleanup_collision_voxels(
    arterial_mask: np.ndarray,
    venous_mask: np.ndarray,
    arterial_seed: np.ndarray,
    venous_seed: np.ndarray,
    spacing: tuple[float, float, float],
    ndimage,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if method not in {"nearest-centerline", "arterial-priority", "venous-priority", "none"}:
        raise ValueError("collision cleanup must be one of: nearest-centerline, arterial-priority, venous-priority, none")

    collision = arterial_mask & venous_mask
    owner_labels = np.zeros(arterial_mask.shape, dtype=np.int16)
    if not collision.any() or method == "none":
        return arterial_mask.copy(), venous_mask.copy(), collision, owner_labels

    arterial_clean = arterial_mask.copy()
    venous_clean = venous_mask.copy()

    if method == "arterial-priority":
        venous_clean[collision] = False
        owner_labels[collision] = 1
        return arterial_clean, venous_clean, collision, owner_labels

    if method == "venous-priority":
        arterial_clean[collision] = False
        owner_labels[collision] = 2
        return arterial_clean, venous_clean, collision, owner_labels

    arterial_reference = arterial_seed if arterial_seed.any() else arterial_mask
    venous_reference = venous_seed if venous_seed.any() else venous_mask
    arterial_distance = ndimage.distance_transform_edt(~arterial_reference, sampling=spacing)
    venous_distance = ndimage.distance_transform_edt(~venous_reference, sampling=spacing)
    arterial_owner = collision & (arterial_distance <= venous_distance)
    venous_owner = collision & ~arterial_owner

    arterial_clean[venous_owner] = False
    venous_clean[arterial_owner] = False
    owner_labels[arterial_owner] = 1
    owner_labels[venous_owner] = 2
    return arterial_clean, venous_clean, collision, owner_labels


def _boundary_label_for_role(role: str) -> int:
    return {
        "arterial_inlet": 1,
        "arterial_outlet": 2,
        "venous_inlet": 3,
        "venous_outlet": 4,
    }.get(role, 0)


def _paint_boundary_labels(
    graph: dict[str, Any],
    spacing: tuple[float, float, float],
    arterial_mask: np.ndarray,
    venous_mask: np.ndarray,
    vessel_wall: np.ndarray,
) -> tuple[np.ndarray, tuple[VoxelizedBoundary, ...]]:
    labels = np.zeros(arterial_mask.shape, dtype=np.int16)
    labels[vessel_wall] = 5
    boundaries: list[VoxelizedBoundary] = []

    for node in graph.get("nodes", []):
        role = str(node.get("boundary_role", ""))
        label = _boundary_label_for_role(role)
        if label == 0:
            continue
        center = tuple(float(value) for value in node["position_mm"])
        radius = max(2.0, float(node["radius_mm"]))
        restrict_mask = arterial_mask if role.startswith("arterial") else venous_mask
        count = _paint_sphere(labels, center, radius, spacing, label, restrict_mask=restrict_mask)
        boundaries.append(
            VoxelizedBoundary(
                label=label,
                node_id=str(node["id"]),
                role=role,
                center_mm=center,
                center_ijk=_center_ijk(center, spacing),
                radius_mm=radius,
                voxel_count=count,
            )
        )

    wall_coords = np.argwhere(vessel_wall)
    wall_center = tuple(float(value) for value in (wall_coords.mean(axis=0) * np.array(spacing))) if len(wall_coords) else (0.0, 0.0, 0.0)
    boundaries.append(
        VoxelizedBoundary(
            label=5,
            node_id="vascular_network_vessel_wall",
            role="wall",
            center_mm=wall_center,
            center_ijk=_center_ijk(wall_center, spacing),
            radius_mm=0.0,
            voxel_count=int(vessel_wall.sum()),
        )
    )
    return labels, tuple(boundaries)


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return tuple(value // 2 for value in mask.shape)
    return tuple(int(round(float(np.median(coords[:, axis])))) for axis in range(3))


def _render_preview(
    path: Path,
    source_ct: np.ndarray | None,
    material_labels: np.ndarray,
    arterial_mask: np.ndarray,
    venous_mask: np.ndarray,
    vessel_wall: np.ndarray,
    spacing: tuple[float, float, float],
) -> None:
    plt, ListedColormap, Patch, *_ = _import_dependencies()
    combined = arterial_mask | venous_mask
    x_index, y_index, z_index = _slice_indices(combined)
    background = source_ct if source_ct is not None else material_labels.astype(np.float32)

    extent_xy = (0.0, material_labels.shape[0] * spacing[0], 0.0, material_labels.shape[1] * spacing[1])
    extent_xz = (0.0, material_labels.shape[0] * spacing[0], 0.0, material_labels.shape[2] * spacing[2])
    extent_yz = (0.0, material_labels.shape[1] * spacing[1], 0.0, material_labels.shape[2] * spacing[2])
    views = [
        (
            np.rot90(background[:, :, z_index]),
            np.rot90(material_labels[:, :, z_index]),
            np.rot90(arterial_mask[:, :, z_index]),
            np.rot90(venous_mask[:, :, z_index]),
            np.rot90(vessel_wall[:, :, z_index]),
            f"Axial z={z_index}",
            extent_xy,
        ),
        (
            np.rot90(background[:, y_index, :]),
            np.rot90(material_labels[:, y_index, :]),
            np.rot90(arterial_mask[:, y_index, :]),
            np.rot90(venous_mask[:, y_index, :]),
            np.rot90(vessel_wall[:, y_index, :]),
            f"Coronal y={y_index}",
            extent_xz,
        ),
        (
            np.rot90(background[x_index, :, :]),
            np.rot90(material_labels[x_index, :, :]),
            np.rot90(arterial_mask[x_index, :, :]),
            np.rot90(venous_mask[x_index, :, :]),
            np.rot90(vessel_wall[x_index, :, :]),
            f"Sagittal x={x_index}",
            extent_yz,
        ),
    ]

    cmap = ListedColormap(
        [
            "#0d1b2a",
            "#6c757d",
            "#8ecae6",
            "#ffd166",
            "#d95d39",
            "#4cc9f0",
            "#9d4edd",
            "#f72585",
            "#48cae4",
            "#f4d35e",
            "#e9ecef",
            "#ffffff",
            "#80ed99",
            "#ff9f1c",
            "#0077b6",
            "#ef476f",
        ]
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    for ax in axes.ravel():
        ax.set_facecolor("#f6f1e8")
        ax.axis("off")

    for ax, (ct_view, _, artery_view, vein_view, wall_view, title, extent) in zip(axes[0], views):
        ax.imshow(np.clip(ct_view, -1000, 1000), cmap="gray", vmin=-1000, vmax=1000, extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~wall_view, wall_view), cmap=ListedColormap(["#ff9f1c"]), alpha=0.45, extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~vein_view, vein_view), cmap=ListedColormap(["#2878b8"]), alpha=0.65, extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~artery_view, artery_view), cmap=ListedColormap(["#dc3b2a"]), alpha=0.72, extent=extent, aspect="equal")
        ax.set_title(f"CT + voxelized network {title}", color="#1e2a32", fontsize=10)

    for ax, (_, material_view, artery_view, vein_view, wall_view, title, extent) in zip(axes[1], views):
        ax.imshow(material_view, cmap=cmap, vmin=0, vmax=15, interpolation="nearest", extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~wall_view, wall_view), cmap=ListedColormap(["#ff9f1c"]), alpha=0.35, extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~vein_view, vein_view), cmap=ListedColormap(["#2878b8"]), alpha=0.38, extent=extent, aspect="equal")
        ax.imshow(np.ma.masked_where(~artery_view, artery_view), cmap=ListedColormap(["#dc3b2a"]), alpha=0.42, extent=extent, aspect="equal")
        ax.set_title(f"Material labels + network {title}", color="#1e2a32", fontsize=10)

    handles = [
        Patch(facecolor="#dc3b2a", label="arterial scaffold lumen"),
        Patch(facecolor="#2878b8", label="venous return lumen"),
        Patch(facecolor="#ff9f1c", label="network vessel wall"),
        Patch(facecolor="#0077b6", label="blood-equivalent material"),
        Patch(facecolor="#ef476f", label="contrast-filled arterial material"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.5), fontsize=8)
    fig.suptitle(
        "Voxelized Vascular Network Integrated Into Combined Digital Phantom",
        fontsize=15,
        color="#13202a",
    )
    fig.tight_layout(rect=(0, 0, 0.90, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_spec(
    path: Path,
    result: VascularNetworkVoxelizationResult,
    contrast_mode: str,
    sample_step_mm: float,
    vessel_wall_thickness_mm: float,
    collision_cleanup: str,
    write_material_volumes: bool,
) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "coordinate_units": "mm",
        "voxelization": {
            "source_graph": result.graph_yaml_path,
            "source_combined_labels": result.source_combined_labels_path,
            "sample_step_mm": sample_step_mm,
            "vessel_wall_thickness_mm": vessel_wall_thickness_mm,
            "contrast_mode": contrast_mode,
            "collision_cleanup": collision_cleanup,
            "material_property_volumes_written": write_material_volumes,
            "outside_body_fraction_before_clip": result.outside_body_fraction_before_clip,
            "connected_components": result.connected_components,
            "arterial_components": result.arterial_components,
            "venous_components": result.venous_components,
            "arterial_venous_overlap_voxels_before_cleanup": result.overlap_voxels_before_cleanup,
            "arterial_venous_overlap_voxels_after_cleanup": result.overlap_voxels_after_cleanup,
        },
        "outputs": {
            "arterial_lumen_mask": result.arterial_lumen_mask_path,
            "venous_lumen_mask": result.venous_lumen_mask_path,
            "combined_lumen_mask": result.combined_lumen_mask_path,
            "collision_mask": result.collision_mask_path,
            "collision_owner_labels": result.collision_owner_labels_path,
            "flow_domain_labels": result.flow_domain_labels_path,
            "vessel_wall_mask": result.vessel_wall_mask_path,
            "boundary_labels": result.boundary_labels_path,
            "blood_material_labels": result.blood_material_labels_path,
            "blood_mass_density_g_cm3": result.blood_density_path,
            "blood_relative_electron_density": result.blood_relative_electron_density_path,
            "blood_synthetic_hu": result.blood_synthetic_hu_path,
            "contrast_material_labels": result.contrast_material_labels_path,
            "contrast_mass_density_g_cm3": result.contrast_density_path,
            "contrast_relative_electron_density": result.contrast_relative_electron_density_path,
            "contrast_synthetic_hu": result.contrast_synthetic_hu_path,
            "preview_png": result.preview_png_path,
        },
        "boundary_labels": [
            {
                "label": boundary.label,
                "node_id": boundary.node_id,
                "role": boundary.role,
                "center_mm": list(boundary.center_mm),
                "center_ijk": list(boundary.center_ijk),
                "radius_mm": boundary.radius_mm,
                "voxel_count": boundary.voxel_count,
            }
            for boundary in result.boundaries
        ],
        "material_label_mapping": {
            13: "vessel_wall",
            14: "blood_equivalent_fluid",
            15: "contrast_filled_blood",
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VascularNetworkVoxelizationResult) -> str:
    lines = [
        "# Vascular Network Voxelization + Collision Cleanup Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Arterial lumen voxels: {result.arterial_voxels} ({result.arterial_volume_cm3:.2f} cm3)",
        f"- Venous lumen voxels: {result.venous_voxels} ({result.venous_volume_cm3:.2f} cm3)",
        f"- Arterial/venous overlap before cleanup: {result.overlap_voxels_before_cleanup}",
        f"- Arterial/venous overlap after cleanup: {result.overlap_voxels_after_cleanup}",
        f"- Combined network lumen voxels: {result.combined_lumen_voxels} ({result.combined_lumen_volume_cm3:.2f} cm3)",
        f"- Network vessel wall voxels: {result.vessel_wall_voxels} ({result.vessel_wall_volume_cm3:.2f} cm3)",
        f"- Connected lumen components: {result.connected_components}",
        f"- Arterial components: {result.arterial_components}",
        f"- Venous components: {result.venous_components}",
        f"- Outside-body fraction before clipping: {result.outside_body_fraction_before_clip:.4f}",
        "",
        "## Outputs",
        "",
        f"- Arterial lumen mask: `{Path(result.arterial_lumen_mask_path).name}`",
        f"- Venous lumen mask: `{Path(result.venous_lumen_mask_path).name}`",
        f"- Combined network lumen mask: `{Path(result.combined_lumen_mask_path).name}`",
        f"- Original collision mask: `{Path(result.collision_mask_path).name}`",
        f"- Collision owner labels: `{Path(result.collision_owner_labels_path).name}`",
        f"- Flow-domain labels: `{Path(result.flow_domain_labels_path).name}`",
        f"- Network vessel wall mask: `{Path(result.vessel_wall_mask_path).name}`",
        f"- Boundary labels: `{Path(result.boundary_labels_path).name}`",
        f"- Blood material labels: `{Path(result.blood_material_labels_path).name if result.blood_material_labels_path else 'not_written'}`",
        f"- Contrast material labels: `{Path(result.contrast_material_labels_path).name if result.contrast_material_labels_path else 'not_written'}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Boundary Labels",
        "",
        "| label | node | role | center mm | center ijk | radius mm | voxels |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for boundary in result.boundaries:
        center_mm = ", ".join(f"{value:.2f}" for value in boundary.center_mm)
        center_ijk = ", ".join(f"{value:.1f}" for value in boundary.center_ijk)
        lines.append(
            f"| {boundary.label} | `{boundary.node_id}` | {boundary.role} | "
            f"{center_mm} | {center_ijk} | {boundary.radius_mm:.2f} | {boundary.voxel_count} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The scaffold graph is now rasterized into NIfTI masks aligned to the combined digital phantom grid.",
            "- Collision voxels are assigned to exactly one flow domain, so arterial and venous lumen masks can be used independently.",
            "- Existing material labels are preserved unless overwritten by scaffold vessel wall or lumen voxels.",
            "- Blood outputs paint arterial and venous scaffold lumen as label `14`; contrast outputs use label `15` according to the selected contrast mode.",
            "- The collision owner-label mask preserves auditability of every voxel reassigned during cleanup.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def voxelize_vascular_network(
    graph_yaml_path: str | Path,
    combined_labels_path: str | Path,
    materials_path: str | Path,
    output_dir: str | Path = "outputs/digital/vascular_network_voxelized",
    case_id: str = "ct_org_case0_imagetbad_case125",
    source_ct_path: str | Path | None = None,
    body_mask_path: str | Path | None = None,
    sample_step_mm: float = 0.75,
    vessel_wall_thickness_mm: float = 2.0,
    contrast_mode: str = "arterial",
    collision_cleanup: str = "nearest-centerline",
    clip_to_body: bool = True,
    write_material_volumes: bool = True,
    report_path: str | Path | None = "outputs/reports/vascular_network_voxelized_stage001.md",
) -> VascularNetworkVoxelizationResult:
    if contrast_mode not in {"arterial", "all", "none"}:
        raise ValueError("contrast_mode must be one of: arterial, all, none")
    if collision_cleanup not in {"nearest-centerline", "arterial-priority", "venous-priority", "none"}:
        raise ValueError("collision_cleanup must be one of: nearest-centerline, arterial-priority, venous-priority, none")
    if vessel_wall_thickness_mm < 0:
        raise ValueError("vessel_wall_thickness_mm must be non-negative")

    _, _, _, nib, ndimage, _ = _import_dependencies()
    graph_path = Path(graph_yaml_path)
    labels_path = Path(combined_labels_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    graph = _load_yaml(graph_path)
    label_image = nib.load(str(labels_path))
    base_labels = np.rint(np.asanyarray(label_image.dataobj)).astype(np.int16)
    spacing = _spacing_from_image(label_image)
    voxel_volume_cm3 = float(np.prod(spacing) / 1000.0)

    if body_mask_path is not None:
        body_image = nib.load(str(body_mask_path))
        body_mask = np.asanyarray(body_image.dataobj) > 0
        if body_mask.shape != base_labels.shape:
            raise ValueError(f"Body mask and combined labels differ: {body_mask.shape} vs {base_labels.shape}")
    else:
        body_mask = base_labels != 0

    restrict_mask = body_mask if clip_to_body else None
    arterial_raw, arterial_seed, arterial_clipped = _paint_edges(
        graph,
        "arterial",
        base_labels.shape,
        spacing,
        sample_step_mm=sample_step_mm,
        restrict_mask=restrict_mask,
    )
    venous_raw, venous_seed, venous_clipped = _paint_edges(
        graph,
        "venous",
        base_labels.shape,
        spacing,
        sample_step_mm=sample_step_mm,
        restrict_mask=restrict_mask,
    )
    arterial_mask, venous_mask, collision_mask, collision_owner_labels = _cleanup_collision_voxels(
        arterial_raw,
        venous_raw,
        arterial_seed,
        venous_seed,
        spacing=spacing,
        ndimage=ndimage,
        method=collision_cleanup,
    )

    lumen_mask = arterial_mask | venous_mask
    if not lumen_mask.any():
        raise ValueError("Voxelized vascular network is empty after clipping")

    existing_lumen = np.isin(base_labels, (14, 15))
    distance_to_lumen = ndimage.distance_transform_edt(~lumen_mask, sampling=spacing)
    vessel_wall = (distance_to_lumen <= vessel_wall_thickness_mm) & ~lumen_mask & ~existing_lumen
    if clip_to_body:
        vessel_wall &= body_mask

    boundary_labels, boundaries = _paint_boundary_labels(graph, spacing, arterial_mask, venous_mask, vessel_wall)
    connected, component_count = ndimage.label(lumen_mask, structure=np.ones((3, 3, 3), dtype=bool))
    _ = connected
    _, arterial_component_count = ndimage.label(arterial_mask, structure=np.ones((3, 3, 3), dtype=bool))
    _, venous_component_count = ndimage.label(venous_mask, structure=np.ones((3, 3, 3), dtype=bool))

    flow_domain_labels = np.zeros(base_labels.shape, dtype=np.int16)
    flow_domain_labels[vessel_wall] = 5
    flow_domain_labels[arterial_mask] = 1
    flow_domain_labels[venous_mask] = 2

    blood_labels = base_labels.copy()
    contrast_labels = base_labels.copy()
    blood_labels[vessel_wall] = 13
    contrast_labels[vessel_wall] = 13
    blood_labels[lumen_mask] = 14
    contrast_labels[lumen_mask] = 14
    if contrast_mode == "arterial":
        contrast_labels[arterial_mask] = 15
    elif contrast_mode == "all":
        contrast_labels[lumen_mask] = 15

    blood_density = blood_red = blood_hu = None
    contrast_density = contrast_red = contrast_hu = None
    if write_material_volumes:
        library = load_material_library(materials_path)
        regions = _combined_regions(library)
        blood_density = _property_volume(blood_labels, regions, "mass_density_g_cm3")
        blood_red = _property_volume(blood_labels, regions, "relative_electron_density")
        blood_hu = _property_volume(blood_labels, regions, "target_hu_midpoint")
        contrast_density = _property_volume(contrast_labels, regions, "mass_density_g_cm3")
        contrast_red = _property_volume(contrast_labels, regions, "relative_electron_density")
        contrast_hu = _property_volume(contrast_labels, regions, "target_hu_midpoint")

    source_ct = None
    if source_ct_path is not None:
        ct_image = nib.load(str(source_ct_path))
        source_ct = np.asanyarray(ct_image.dataobj).astype(np.float32)
        if source_ct.shape != base_labels.shape:
            raise ValueError(f"Source CT and combined labels differ: {source_ct.shape} vs {base_labels.shape}")

    base = output / case_id
    arterial_path = base.with_name(f"{case_id}_vascular_network_arterial_lumen_mask_v001.nii.gz")
    venous_path = base.with_name(f"{case_id}_vascular_network_venous_lumen_mask_v001.nii.gz")
    lumen_path = base.with_name(f"{case_id}_vascular_network_lumen_mask_v001.nii.gz")
    collision_path = base.with_name(f"{case_id}_vascular_network_collision_mask_v001.nii.gz")
    collision_owner_path = base.with_name(f"{case_id}_vascular_network_collision_owner_labels_v001.nii.gz")
    flow_domain_path = base.with_name(f"{case_id}_vascular_network_flow_domain_labels_v001.nii.gz")
    wall_path = base.with_name(f"{case_id}_vascular_network_vessel_wall_mask_v001.nii.gz")
    boundary_path = base.with_name(f"{case_id}_vascular_network_boundary_labels_v001.nii.gz")
    blood_labels_path = base.with_name(f"{case_id}_vascular_network_material_labels_blood_v001.nii.gz") if write_material_volumes else None
    blood_density_path = base.with_name(f"{case_id}_vascular_network_mass_density_blood_v001.nii.gz") if write_material_volumes else None
    blood_red_path = base.with_name(f"{case_id}_vascular_network_relative_electron_density_blood_v001.nii.gz") if write_material_volumes else None
    blood_hu_path = base.with_name(f"{case_id}_vascular_network_synthetic_hu_blood_v001.nii.gz") if write_material_volumes else None
    contrast_labels_path = base.with_name(f"{case_id}_vascular_network_material_labels_contrast_v001.nii.gz") if write_material_volumes else None
    contrast_density_path = base.with_name(f"{case_id}_vascular_network_mass_density_contrast_v001.nii.gz") if write_material_volumes else None
    contrast_red_path = base.with_name(f"{case_id}_vascular_network_relative_electron_density_contrast_v001.nii.gz") if write_material_volumes else None
    contrast_hu_path = base.with_name(f"{case_id}_vascular_network_synthetic_hu_contrast_v001.nii.gz") if write_material_volumes else None
    preview_png = base.with_name(f"{case_id}_vascular_network_voxelized_preview_v001.png")
    spec_yaml = base.with_name(f"{case_id}_vascular_network_voxelized_spec_v001.yaml")
    report = Path(report_path) if report_path else base.with_name(f"{case_id}_vascular_network_voxelized_report_v001.md")

    _write_nifti(arterial_path, arterial_mask.astype(np.uint8), label_image, nib)
    _write_nifti(venous_path, venous_mask.astype(np.uint8), label_image, nib)
    _write_nifti(lumen_path, lumen_mask.astype(np.uint8), label_image, nib)
    _write_nifti(collision_path, collision_mask.astype(np.uint8), label_image, nib)
    _write_nifti(collision_owner_path, collision_owner_labels, label_image, nib)
    _write_nifti(flow_domain_path, flow_domain_labels, label_image, nib)
    _write_nifti(wall_path, vessel_wall.astype(np.uint8), label_image, nib)
    _write_nifti(boundary_path, boundary_labels, label_image, nib)
    if write_material_volumes:
        _write_nifti(blood_labels_path, blood_labels, label_image, nib)
        _write_nifti(contrast_labels_path, contrast_labels, label_image, nib)
        _write_nifti(blood_density_path, blood_density, label_image, nib)
        _write_nifti(blood_red_path, blood_red, label_image, nib)
        _write_nifti(blood_hu_path, blood_hu, label_image, nib)
        _write_nifti(contrast_density_path, contrast_density, label_image, nib)
        _write_nifti(contrast_red_path, contrast_red, label_image, nib)
        _write_nifti(contrast_hu_path, contrast_hu, label_image, nib)

    outside_before_clip = arterial_clipped + venous_clipped
    total_before_clip = int(arterial_raw.sum() + arterial_clipped + venous_raw.sum() + venous_clipped)
    outside_fraction = outside_before_clip / total_before_clip if total_before_clip else 0.0
    overlap_before = int(collision_mask.sum())
    overlap_after = int((arterial_mask & venous_mask).sum())
    notes = [
        "vascular_network_graph_voxelized_into_combined_phantom_grid",
        "coordinates_follow_existing_project_convention=index_times_spacing_mm",
        f"contrast_mode={contrast_mode}",
        f"collision_cleanup={collision_cleanup}",
    ]
    if clip_to_body:
        notes.append("voxelized_network_clipped_to_body_mask")
    if not write_material_volumes:
        notes.append("material_property_volumes_skipped_for_disk_light_flow_rerun")
    if overlap_before:
        notes.append(f"arterial_venous_overlap_voxels_before_cleanup={overlap_before}")
        notes.append(f"arterial_venous_overlap_voxels_after_cleanup={overlap_after}")
    if overlap_after:
        notes.append(f"arterial_venous_overlap_voxels_remaining={overlap_after}_requires_review")

    result = VascularNetworkVoxelizationResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_path),
        source_combined_labels_path=str(labels_path),
        arterial_lumen_mask_path=str(arterial_path),
        venous_lumen_mask_path=str(venous_path),
        combined_lumen_mask_path=str(lumen_path),
        collision_mask_path=str(collision_path),
        collision_owner_labels_path=str(collision_owner_path),
        flow_domain_labels_path=str(flow_domain_path),
        vessel_wall_mask_path=str(wall_path),
        boundary_labels_path=str(boundary_path),
        blood_material_labels_path=None if blood_labels_path is None else str(blood_labels_path),
        blood_density_path=None if blood_density_path is None else str(blood_density_path),
        blood_relative_electron_density_path=None if blood_red_path is None else str(blood_red_path),
        blood_synthetic_hu_path=None if blood_hu_path is None else str(blood_hu_path),
        contrast_material_labels_path=None if contrast_labels_path is None else str(contrast_labels_path),
        contrast_density_path=None if contrast_density_path is None else str(contrast_density_path),
        contrast_relative_electron_density_path=None if contrast_red_path is None else str(contrast_red_path),
        contrast_synthetic_hu_path=None if contrast_hu_path is None else str(contrast_hu_path),
        preview_png_path=str(preview_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        arterial_voxels=int(arterial_mask.sum()),
        venous_voxels=int(venous_mask.sum()),
        overlap_voxels=overlap_after,
        overlap_voxels_before_cleanup=overlap_before,
        overlap_voxels_after_cleanup=overlap_after,
        combined_lumen_voxels=int(lumen_mask.sum()),
        vessel_wall_voxels=int(vessel_wall.sum()),
        arterial_volume_cm3=float(arterial_mask.sum() * voxel_volume_cm3),
        venous_volume_cm3=float(venous_mask.sum() * voxel_volume_cm3),
        combined_lumen_volume_cm3=float(lumen_mask.sum() * voxel_volume_cm3),
        vessel_wall_volume_cm3=float(vessel_wall.sum() * voxel_volume_cm3),
        outside_body_fraction_before_clip=float(outside_fraction),
        connected_components=int(component_count),
        arterial_components=int(arterial_component_count),
        venous_components=int(venous_component_count),
        boundaries=boundaries,
        notes=tuple(notes),
    )

    _render_preview(
        preview_png,
        source_ct=source_ct,
        material_labels=contrast_labels,
        arterial_mask=arterial_mask,
        venous_mask=venous_mask,
        vessel_wall=vessel_wall,
        spacing=spacing,
    )
    _write_spec(
        spec_yaml,
        result,
        contrast_mode=contrast_mode,
        sample_step_mm=sample_step_mm,
        vessel_wall_thickness_mm=vessel_wall_thickness_mm,
        collision_cleanup=collision_cleanup,
        write_material_volumes=write_material_volumes,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_vascular_network_voxelization_result(result: VascularNetworkVoxelizationResult) -> str:
    return _format_report(result)
