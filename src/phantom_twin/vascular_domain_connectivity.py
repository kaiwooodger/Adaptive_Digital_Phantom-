from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np

from .combined import _combined_regions, _property_volume, _write_nifti
from .materials import load_material_library
from .vascular_voxelize import (
    VascularNetworkVoxelizationResult,
    _edge_samples,
    _format_report as _format_voxelization_report,
    _import_dependencies,
    _load_yaml,
    _paint_boundary_labels,
    _paint_sphere,
    _render_preview,
    _spacing_from_image,
    _write_spec as _write_voxelization_spec,
)
from .vessel_anatomy_validation import _resolve_path


@dataclass(frozen=True)
class DomainRepairSummary:
    domain: str
    components_before: int
    components_after: int
    voxels_before: int
    voxels_after: int
    seeded_component_count_before: int
    pruned_component_count: int
    pruned_voxel_count: int
    connector_voxel_count: int


@dataclass(frozen=True)
class VascularDomainConnectivityRepairResult:
    case_id: str
    output_dir: str
    source_voxelized_spec_path: str
    source_graph_path: str
    source_combined_labels_path: str
    repaired_spec_yaml_path: str
    repaired_arterial_lumen_mask_path: str
    repaired_venous_lumen_mask_path: str
    repaired_combined_lumen_mask_path: str
    repaired_flow_domain_labels_path: str
    component_summary_csv_path: str
    preview_png_path: str
    report_path: str
    arterial_summary: DomainRepairSummary
    venous_summary: DomainRepairSummary
    combined_components_before: int
    combined_components_after: int
    overlap_after_repair: int
    outside_body_fraction_after_repair: float
    notes: tuple[str, ...]


def _component_count(mask: np.ndarray, ndimage) -> int:
    _, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    return int(count)


def _load_mask(path: str | Path, nib) -> tuple[np.ndarray, Any]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj) > 0, image


def _component_ids_near_index(
    component_labels: np.ndarray,
    index: np.ndarray,
    search_radius_voxels: int,
) -> set[int]:
    shape = np.asarray(component_labels.shape, dtype=int)
    if np.any(index < 0) or np.any(index >= shape):
        return set()
    direct = int(component_labels[tuple(int(value) for value in index)])
    if direct:
        return {direct}
    radius = max(int(search_radius_voxels), 0)
    if radius <= 0:
        return set()
    slices = tuple(
        slice(max(0, int(index[axis]) - radius), min(int(shape[axis]), int(index[axis]) + radius + 1))
        for axis in range(3)
    )
    return {int(value) for value in np.unique(component_labels[slices]) if int(value) > 0}


def _seeded_components_for_domain(
    graph: dict[str, Any],
    component_labels: np.ndarray,
    domain: str,
    spacing: tuple[float, float, float],
    sample_step_mm: float,
    seed_search_radius_voxels: int,
) -> set[int]:
    seeded: set[int] = set()
    spacing_array = np.asarray(spacing, dtype=float)
    for edge in graph.get("edges", []):
        if str(edge.get("vessel_type")) != domain:
            continue
        for center, _ in _edge_samples(edge, sample_step_mm=sample_step_mm):
            index = np.rint(np.asarray(center, dtype=float) / spacing_array).astype(int)
            seeded.update(_component_ids_near_index(component_labels, index, seed_search_radius_voxels))
    for node in graph.get("nodes", []):
        role = str(node.get("boundary_role", ""))
        if domain == "arterial" and not role.startswith("arterial"):
            continue
        if domain == "venous" and not role.startswith("venous"):
            continue
        if "position_mm" not in node:
            continue
        index = np.rint(np.asarray(node["position_mm"], dtype=float) / spacing_array).astype(int)
        seeded.update(_component_ids_near_index(component_labels, index, seed_search_radius_voxels))
    return seeded


def _paint_domain_connectors(
    graph: dict[str, Any],
    domain: str,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    body_mask: np.ndarray,
    sample_step_mm: float,
    connector_radius_mm: float,
) -> np.ndarray:
    connector = np.zeros(shape, dtype=bool)
    for edge in graph.get("edges", []):
        if str(edge.get("vessel_type")) != domain:
            continue
        for center, _ in _edge_samples(edge, sample_step_mm=sample_step_mm):
            _paint_sphere(connector, center, connector_radius_mm, spacing, True, restrict_mask=body_mask)
    return connector


def _repair_domain(
    mask: np.ndarray,
    *,
    graph: dict[str, Any],
    domain: str,
    spacing: tuple[float, float, float],
    ndimage,
    body_mask: np.ndarray,
    sample_step_mm: float,
    seed_search_radius_voxels: int,
    max_unseeded_component_voxels: int,
    connector_radius_mm: float,
    connect_seeded_components: bool,
) -> tuple[np.ndarray, DomainRepairSummary, np.ndarray]:
    component_labels, components_before = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    seeded = _seeded_components_for_domain(
        graph,
        component_labels,
        domain,
        spacing,
        sample_step_mm=sample_step_mm,
        seed_search_radius_voxels=seed_search_radius_voxels,
    )
    sizes = np.bincount(component_labels.ravel())
    repaired = mask.copy()
    pruned_components = 0
    pruned_voxels = 0
    for component_id in range(1, int(components_before) + 1):
        size = int(sizes[component_id]) if component_id < len(sizes) else 0
        if component_id in seeded:
            continue
        if size <= max_unseeded_component_voxels:
            component_mask = component_labels == component_id
            repaired[component_mask] = False
            pruned_components += 1
            pruned_voxels += size

    connector = np.zeros(mask.shape, dtype=bool)
    labels_after_prune, components_after_prune = ndimage.label(repaired, structure=np.ones((3, 3, 3), dtype=bool))
    seeded_after_prune = _seeded_components_for_domain(
        graph,
        labels_after_prune,
        domain,
        spacing,
        sample_step_mm=sample_step_mm,
        seed_search_radius_voxels=seed_search_radius_voxels,
    )
    if connect_seeded_components and len(seeded_after_prune) > 1:
        connector = _paint_domain_connectors(
            graph,
            domain,
            mask.shape,
            spacing,
            body_mask,
            sample_step_mm=min(sample_step_mm, 0.75),
            connector_radius_mm=connector_radius_mm,
        )
        repaired |= connector

    _, components_after = ndimage.label(repaired, structure=np.ones((3, 3, 3), dtype=bool))
    summary = DomainRepairSummary(
        domain=domain,
        components_before=int(components_before),
        components_after=int(components_after),
        voxels_before=int(np.count_nonzero(mask)),
        voxels_after=int(np.count_nonzero(repaired)),
        seeded_component_count_before=len(seeded),
        pruned_component_count=pruned_components,
        pruned_voxel_count=pruned_voxels,
        connector_voxel_count=int(np.count_nonzero(connector & ~mask)),
    )
    return repaired, summary, connector


def _write_component_summary(path: Path, summaries: tuple[DomainRepairSummary, ...]) -> None:
    fields = [
        "domain",
        "components_before",
        "components_after",
        "voxels_before",
        "voxels_after",
        "seeded_component_count_before",
        "pruned_component_count",
        "pruned_voxel_count",
        "connector_voxel_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: getattr(summary, field) for field in fields})


def _format_repair_report(result: VascularDomainConnectivityRepairResult) -> str:
    lines = [
        "# Vascular Domain Connectivity Repair",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Combined lumen components: {result.combined_components_before} -> {result.combined_components_after}",
        f"- Arterial components: {result.arterial_summary.components_before} -> {result.arterial_summary.components_after}",
        f"- Venous components: {result.venous_summary.components_before} -> {result.venous_summary.components_after}",
        f"- Arterial pruned islands/voxels: {result.arterial_summary.pruned_component_count} / {result.arterial_summary.pruned_voxel_count}",
        f"- Venous pruned islands/voxels: {result.venous_summary.pruned_component_count} / {result.venous_summary.pruned_voxel_count}",
        f"- Connector voxels arterial/venous: {result.arterial_summary.connector_voxel_count} / {result.venous_summary.connector_voxel_count}",
        f"- Arterial/venous overlap after repair: {result.overlap_after_repair}",
        f"- Outside-body fraction after repair: {result.outside_body_fraction_after_repair:.6f}",
        "",
        "## Outputs",
        "",
        f"- Repaired spec YAML: `{Path(result.repaired_spec_yaml_path).name}`",
        f"- Repaired arterial mask: `{Path(result.repaired_arterial_lumen_mask_path).name}`",
        f"- Repaired venous mask: `{Path(result.repaired_venous_lumen_mask_path).name}`",
        f"- Repaired combined lumen mask: `{Path(result.repaired_combined_lumen_mask_path).name}`",
        f"- Repaired flow-domain labels: `{Path(result.repaired_flow_domain_labels_path).name}`",
        f"- Component summary CSV: `{Path(result.component_summary_csv_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        "",
        "## Interpretation",
        "",
        "- Unseeded tiny islands are removed when they are not touched by graph edge or boundary-node samples.",
        "- If graph-seeded components are genuinely split, the optional connector pass can add minimal centerline voxels.",
        "- Arterial/venous overlap is still forced to zero after repair by assigning connector conflicts to the repaired domain.",
        "- This is voxel-domain cleanup; it does not change graph centerlines, graph radii, flow roles, or boundary IDs.",
        "",
        "## Notes",
    ]
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def repair_vascular_domain_connectivity(
    voxelized_spec_path: str | Path,
    graph_yaml_path: str | Path | None = None,
    combined_labels_path: str | Path | None = None,
    materials_path: str | Path = "configs/materials.yaml",
    output_dir: str | Path = "outputs/digital/vascular_domain_connectivity_repaired",
    case_id: str = "vascular_domain_connectivity_repaired",
    sample_step_mm: float | None = None,
    seed_search_radius_voxels: int = 2,
    max_unseeded_component_voxels: int = 500,
    connector_radius_mm: float = 0.8,
    connect_seeded_components: bool = True,
    contrast_mode: str | None = None,
    vessel_wall_thickness_mm: float | None = None,
    write_material_volumes: bool = True,
    report_path: str | Path | None = "outputs/reports/vascular_domain_connectivity_repair_stage001.md",
) -> VascularDomainConnectivityRepairResult:
    _, _, _, nib, ndimage, _ = _import_dependencies()
    spec_path = Path(voxelized_spec_path)
    spec = _load_yaml(spec_path)
    voxelization = spec.get("voxelization", {})
    outputs = spec.get("outputs", {})
    graph_path = _resolve_path(graph_yaml_path or voxelization.get("source_graph"), spec_path)
    labels_path = _resolve_path(combined_labels_path or voxelization.get("source_combined_labels"), spec_path)
    if graph_path is None or labels_path is None:
        raise ValueError("Voxelized spec must provide source graph and source combined labels")
    graph = _load_yaml(graph_path)
    label_image = nib.load(str(labels_path))
    base_labels = np.rint(np.asanyarray(label_image.dataobj)).astype(np.int16)
    spacing = _spacing_from_image(label_image)
    voxel_volume_cm3 = float(np.prod(spacing) / 1000.0)
    body_mask = base_labels != 0

    arterial_path = _resolve_path(outputs.get("arterial_lumen_mask"), spec_path)
    venous_path = _resolve_path(outputs.get("venous_lumen_mask"), spec_path)
    if arterial_path is None or venous_path is None:
        raise ValueError("Voxelized spec must provide arterial and venous lumen masks")
    arterial_mask, source_mask_image = _load_mask(arterial_path, nib)
    venous_mask, _ = _load_mask(venous_path, nib)
    if arterial_mask.shape != base_labels.shape or venous_mask.shape != base_labels.shape:
        raise ValueError("Source masks and combined labels have different shapes")

    resolved_sample_step = float(sample_step_mm if sample_step_mm is not None else voxelization.get("sample_step_mm", 0.75))
    resolved_contrast_mode = str(contrast_mode or voxelization.get("contrast_mode", "arterial"))
    resolved_wall_thickness = float(vessel_wall_thickness_mm if vessel_wall_thickness_mm is not None else voxelization.get("vessel_wall_thickness_mm", 2.0))

    combined_before = _component_count(arterial_mask | venous_mask, ndimage)
    arterial_repaired, arterial_summary, arterial_connector = _repair_domain(
        arterial_mask,
        graph=graph,
        domain="arterial",
        spacing=spacing,
        ndimage=ndimage,
        body_mask=body_mask,
        sample_step_mm=resolved_sample_step,
        seed_search_radius_voxels=seed_search_radius_voxels,
        max_unseeded_component_voxels=max_unseeded_component_voxels,
        connector_radius_mm=connector_radius_mm,
        connect_seeded_components=connect_seeded_components,
    )
    venous_repaired, venous_summary, venous_connector = _repair_domain(
        venous_mask,
        graph=graph,
        domain="venous",
        spacing=spacing,
        ndimage=ndimage,
        body_mask=body_mask,
        sample_step_mm=resolved_sample_step,
        seed_search_radius_voxels=seed_search_radius_voxels,
        max_unseeded_component_voxels=max_unseeded_component_voxels,
        connector_radius_mm=connector_radius_mm,
        connect_seeded_components=connect_seeded_components,
    )

    if np.any(arterial_connector):
        venous_repaired[arterial_connector] = False
    if np.any(venous_connector):
        arterial_repaired[venous_connector] = False
    overlap = arterial_repaired & venous_repaired
    if np.any(overlap):
        # Preserve arterial ownership for residual conflicts; this mirrors arterial-priority cleanup only for repaired voxels.
        venous_repaired[overlap] = False
    arterial_repaired &= body_mask
    venous_repaired &= body_mask
    lumen_mask = arterial_repaired | venous_repaired
    combined_after = _component_count(lumen_mask, ndimage)
    overlap_after = int(np.count_nonzero(arterial_repaired & venous_repaired))
    outside_after = int(np.count_nonzero(lumen_mask & ~body_mask))
    outside_fraction = outside_after / int(np.count_nonzero(lumen_mask)) if np.any(lumen_mask) else 0.0

    existing_lumen = np.isin(base_labels, (14, 15))
    distance_to_lumen = ndimage.distance_transform_edt(~lumen_mask, sampling=spacing)
    vessel_wall = (distance_to_lumen <= resolved_wall_thickness) & ~lumen_mask & ~existing_lumen & body_mask
    boundary_labels, boundaries = _paint_boundary_labels(graph, spacing, arterial_repaired, venous_repaired, vessel_wall)
    flow_domain_labels = np.zeros(base_labels.shape, dtype=np.int16)
    flow_domain_labels[vessel_wall] = 5
    flow_domain_labels[arterial_repaired] = 1
    flow_domain_labels[venous_repaired] = 2

    blood_labels = base_labels.copy()
    contrast_labels = base_labels.copy()
    blood_labels[vessel_wall] = 13
    contrast_labels[vessel_wall] = 13
    blood_labels[lumen_mask] = 14
    contrast_labels[lumen_mask] = 14
    if resolved_contrast_mode == "arterial":
        contrast_labels[arterial_repaired] = 15
    elif resolved_contrast_mode == "all":
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

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = output / case_id
    repaired_arterial_path = base.with_name(f"{case_id}_vascular_network_arterial_lumen_mask_v001.nii.gz")
    repaired_venous_path = base.with_name(f"{case_id}_vascular_network_venous_lumen_mask_v001.nii.gz")
    repaired_lumen_path = base.with_name(f"{case_id}_vascular_network_lumen_mask_v001.nii.gz")
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
    repaired_spec = base.with_name(f"{case_id}_vascular_network_voxelized_spec_v001.yaml")
    component_csv = base.with_name(f"{case_id}_vascular_domain_connectivity_summary_v001.csv")
    repair_report = Path(report_path) if report_path is not None else base.with_name(f"{case_id}_vascular_domain_connectivity_repair_report_v001.md")

    collision_mask = arterial_mask & venous_mask
    collision_owner = np.zeros(base_labels.shape, dtype=np.int16)
    collision_owner[collision_mask & arterial_repaired] = 1
    collision_owner[collision_mask & venous_repaired] = 2

    _write_nifti(repaired_arterial_path, arterial_repaired.astype(np.uint8), label_image, nib)
    _write_nifti(repaired_venous_path, venous_repaired.astype(np.uint8), label_image, nib)
    _write_nifti(repaired_lumen_path, lumen_mask.astype(np.uint8), label_image, nib)
    _write_nifti(collision_path, collision_mask.astype(np.uint8), label_image, nib)
    _write_nifti(collision_owner_path, collision_owner, label_image, nib)
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

    arterial_components_after = _component_count(arterial_repaired, ndimage)
    venous_components_after = _component_count(venous_repaired, ndimage)
    notes = [
        "vascular_domain_connectivity_repair_applied",
        f"source_voxelized_spec={spec_path}",
        f"max_unseeded_component_voxels={max_unseeded_component_voxels}",
        f"connect_seeded_components={connect_seeded_components}",
    ]
    if arterial_summary.pruned_voxel_count or venous_summary.pruned_voxel_count:
        notes.append(
            f"pruned_unseeded_domain_island_voxels_arterial_venous={arterial_summary.pruned_voxel_count}_{venous_summary.pruned_voxel_count}"
        )
    if arterial_summary.connector_voxel_count or venous_summary.connector_voxel_count:
        notes.append(
            f"connector_voxels_arterial_venous={arterial_summary.connector_voxel_count}_{venous_summary.connector_voxel_count}"
        )

    voxel_result = VascularNetworkVoxelizationResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_path),
        source_combined_labels_path=str(labels_path),
        arterial_lumen_mask_path=str(repaired_arterial_path),
        venous_lumen_mask_path=str(repaired_venous_path),
        combined_lumen_mask_path=str(repaired_lumen_path),
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
        spec_yaml_path=str(repaired_spec),
        report_path=str(repair_report),
        arterial_voxels=int(np.count_nonzero(arterial_repaired)),
        venous_voxels=int(np.count_nonzero(venous_repaired)),
        overlap_voxels=int(np.count_nonzero(collision_mask)),
        overlap_voxels_before_cleanup=int(spec.get("voxelization", {}).get("arterial_venous_overlap_voxels_before_cleanup", np.count_nonzero(collision_mask))),
        overlap_voxels_after_cleanup=overlap_after,
        combined_lumen_voxels=int(np.count_nonzero(lumen_mask)),
        vessel_wall_voxels=int(np.count_nonzero(vessel_wall)),
        arterial_volume_cm3=float(np.count_nonzero(arterial_repaired) * voxel_volume_cm3),
        venous_volume_cm3=float(np.count_nonzero(venous_repaired) * voxel_volume_cm3),
        combined_lumen_volume_cm3=float(np.count_nonzero(lumen_mask) * voxel_volume_cm3),
        vessel_wall_volume_cm3=float(np.count_nonzero(vessel_wall) * voxel_volume_cm3),
        outside_body_fraction_before_clip=outside_fraction,
        connected_components=combined_after,
        arterial_components=arterial_components_after,
        venous_components=venous_components_after,
        boundaries=boundaries,
        notes=tuple(notes),
    )
    _render_preview(preview_png, None, base_labels, arterial_repaired, venous_repaired, vessel_wall, spacing)
    _write_voxelization_spec(
        repaired_spec,
        voxel_result,
        contrast_mode=resolved_contrast_mode,
        sample_step_mm=resolved_sample_step,
        vessel_wall_thickness_mm=resolved_wall_thickness,
        collision_cleanup=f"{voxelization.get('collision_cleanup', 'unknown')}+domain_connectivity_repair",
        write_material_volumes=write_material_volumes,
    )
    _write_component_summary(component_csv, (arterial_summary, venous_summary))

    result = VascularDomainConnectivityRepairResult(
        case_id=case_id,
        output_dir=str(output),
        source_voxelized_spec_path=str(spec_path),
        source_graph_path=str(graph_path),
        source_combined_labels_path=str(labels_path),
        repaired_spec_yaml_path=str(repaired_spec),
        repaired_arterial_lumen_mask_path=str(repaired_arterial_path),
        repaired_venous_lumen_mask_path=str(repaired_venous_path),
        repaired_combined_lumen_mask_path=str(repaired_lumen_path),
        repaired_flow_domain_labels_path=str(flow_domain_path),
        component_summary_csv_path=str(component_csv),
        preview_png_path=str(preview_png),
        report_path=str(repair_report),
        arterial_summary=DomainRepairSummary(
            domain="arterial",
            components_before=arterial_summary.components_before,
            components_after=arterial_components_after,
            voxels_before=arterial_summary.voxels_before,
            voxels_after=int(np.count_nonzero(arterial_repaired)),
            seeded_component_count_before=arterial_summary.seeded_component_count_before,
            pruned_component_count=arterial_summary.pruned_component_count,
            pruned_voxel_count=arterial_summary.pruned_voxel_count,
            connector_voxel_count=arterial_summary.connector_voxel_count,
        ),
        venous_summary=DomainRepairSummary(
            domain="venous",
            components_before=venous_summary.components_before,
            components_after=venous_components_after,
            voxels_before=venous_summary.voxels_before,
            voxels_after=int(np.count_nonzero(venous_repaired)),
            seeded_component_count_before=venous_summary.seeded_component_count_before,
            pruned_component_count=venous_summary.pruned_component_count,
            pruned_voxel_count=venous_summary.pruned_voxel_count,
            connector_voxel_count=venous_summary.connector_voxel_count,
        ),
        combined_components_before=combined_before,
        combined_components_after=combined_after,
        overlap_after_repair=overlap_after,
        outside_body_fraction_after_repair=outside_fraction,
        notes=tuple(notes),
    )
    repair_report.parent.mkdir(parents=True, exist_ok=True)
    repair_report.write_text(_format_repair_report(result) + "\n\n" + _format_voxelization_report(voxel_result) + "\n")
    return result


def format_vascular_domain_connectivity_repair_result(result: VascularDomainConnectivityRepairResult) -> str:
    return "\n".join(
        [
            "Vascular domain connectivity repair completed",
            f"Case ID: {result.case_id}",
            f"Combined components before/after: {result.combined_components_before}/{result.combined_components_after}",
            f"Arterial components before/after: {result.arterial_summary.components_before}/{result.arterial_summary.components_after}",
            f"Venous components before/after: {result.venous_summary.components_before}/{result.venous_summary.components_after}",
            f"Pruned arterial/venous voxels: {result.arterial_summary.pruned_voxel_count}/{result.venous_summary.pruned_voxel_count}",
            f"Overlap after repair: {result.overlap_after_repair}",
            f"Repaired spec YAML: {result.repaired_spec_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
