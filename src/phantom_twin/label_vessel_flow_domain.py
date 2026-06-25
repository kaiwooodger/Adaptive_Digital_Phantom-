from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .combined import _combined_regions, _property_volume, _write_nifti
from .materials import load_material_library
from .validation_intake import DEFAULT_REQUIRED_VESSEL_LABELS


DEFAULT_ARTERIAL_LABELS = frozenset((2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 27, 28, 41))
DEFAULT_VENOUS_LABELS = frozenset(
    label for label in range(1, 44) if label not in DEFAULT_ARTERIAL_LABELS
)


@dataclass(frozen=True)
class LabelVesselFlowDomainResult:
    case_id: str
    anatomy_labels_path: str
    vessel_labels_path: str
    source_graph_path: str
    output_dir: str
    arterial_lumen_mask_path: str
    venous_lumen_mask_path: str
    combined_lumen_mask_path: str
    flow_domain_labels_path: str
    vessel_wall_mask_path: str
    boundary_labels_path: str
    collision_mask_path: str
    collision_owner_labels_path: str
    blood_material_labels_path: str | None
    contrast_material_labels_path: str | None
    blood_density_path: str | None
    blood_relative_electron_density_path: str | None
    blood_synthetic_hu_path: str | None
    contrast_density_path: str | None
    contrast_relative_electron_density_path: str | None
    contrast_synthetic_hu_path: str | None
    flow_graph_yaml_path: str
    spec_yaml_path: str
    manifest_yaml_path: str
    label_summary_csv_path: str
    preview_png_path: str
    report_path: str
    arterial_voxels: int
    venous_voxels: int
    vessel_wall_voxels: int
    combined_lumen_voxels: int
    arterial_volume_cm3: float
    venous_volume_cm3: float
    vessel_wall_volume_cm3: float
    combined_lumen_volume_cm3: float
    arterial_components: int
    venous_components: int
    connected_components: int
    label_count: int
    classified_label_count: int
    unclassified_labels: tuple[int, ...]
    boundary_node_count: int
    snapped_boundary_node_count: int
    boundary_label_voxels: int
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from scipy import ndimage  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Labelled-vessel flow-domain generation requires matplotlib, nibabel, and scipy.") from exc
    return plt, nib, ndimage


def _geometry_match(reference_image: Any, candidate_image: Any) -> bool:
    if tuple(reference_image.shape) != tuple(candidate_image.shape):
        return False
    ref_spacing = reference_image.header.get_zooms()[: len(reference_image.shape)]
    cand_spacing = candidate_image.header.get_zooms()[: len(candidate_image.shape)]
    return bool(
        np.allclose(np.asarray(ref_spacing, dtype=float), np.asarray(cand_spacing, dtype=float), atol=1e-3)
        and np.allclose(np.asarray(reference_image.affine, dtype=float), np.asarray(candidate_image.affine, dtype=float), atol=1e-3)
    )


def _spacing(image: Any) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _voxel_volume_cm3(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(np.asarray(spacing_mm, dtype=float)) / 1000.0)


def _center_mm(coords: np.ndarray, spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in (coords.mean(axis=0) * np.asarray(spacing_mm, dtype=float)))


def _slice_index(mask: np.ndarray, axis: int) -> int:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return int(mask.shape[axis] // 2)
    return int(np.clip(np.median(coords[:, axis]), 0, mask.shape[axis] - 1))


def _load_label_names(path: str | Path | None) -> dict[int, str]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        return {}
    names: dict[int, str] = {}
    for key, value in data.get("labels", {}).items():
        try:
            names[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return names


def _load_edge_label_mapping(path: str | Path | None) -> dict[str, tuple[int, ...]]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        return {}
    mapping: dict[str, tuple[int, ...]] = {}
    for edge_id, payload in data.get("graph_edge_mapping", {}).items():
        if not isinstance(payload, dict):
            continue
        labels: list[int] = []
        for value in payload.get("labels", []):
            try:
                labels.append(int(value))
            except (TypeError, ValueError):
                continue
        if labels:
            mapping[str(edge_id)] = tuple(labels)
    return mapping


def _classify_labels(label_names: dict[int, str]) -> tuple[frozenset[int], frozenset[int]]:
    arterial = set(DEFAULT_ARTERIAL_LABELS)
    venous = set(DEFAULT_VENOUS_LABELS)
    for label, raw_name in label_names.items():
        name = raw_name.lower()
        is_arterial = any(term in name for term in ("artery", "aorta", "coeliac", "celiac"))
        is_venous = any(term in name for term in ("vein", "vena", "cava", "portal", "azygos", "plexus"))
        if is_arterial and not is_venous:
            arterial.add(label)
            venous.discard(label)
        elif is_venous and not is_arterial:
            venous.add(label)
            arterial.discard(label)
    return frozenset(arterial), frozenset(venous)


def _label_counts(vessel_labels: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(vessel_labels.astype(np.int64), return_counts=True)
    return {int(label): int(count) for label, count in zip(labels.tolist(), counts.tolist()) if int(label) != 0}


def _domain_for_role(role: str) -> str | None:
    role_lower = role.lower()
    if "arterial" in role_lower:
        return "arterial"
    if "venous" in role_lower:
        return "venous"
    return None


def _node_position_mm(node: dict[str, Any]) -> tuple[float, float, float]:
    return tuple(float(value) for value in node.get("position_mm", (0.0, 0.0, 0.0))[:3])


def _connected_edge_ids(graph: dict[str, Any], node_id: str) -> tuple[str, ...]:
    edge_ids: list[str] = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if str(edge.get("source", "")) == node_id or str(edge.get("target", "")) == node_id:
            edge_ids.append(str(edge.get("id", "")))
    return tuple(edge_id for edge_id in edge_ids if edge_id)


def _preferred_label_groups(
    *,
    node_id: str,
    role: str,
    connected_edge_ids: tuple[str, ...],
    edge_label_mapping: dict[str, tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    node_lower = node_id.lower()
    role_lower = role.lower()
    groups: list[tuple[int, ...]] = []
    if "arterial" in role_lower:
        if "aorta" in node_lower:
            groups.append((4,))
        if "left_common_iliac" in node_lower:
            groups.append((2,))
        if "right_common_iliac" in node_lower:
            groups.append((3,))
        if "left_renal" in node_lower:
            groups.append((28,))
        if "right_renal" in node_lower:
            groups.append((27,))
        if "hepatic" in node_lower:
            groups.append((6, 8, 9, 10))
            groups.append((5,))
        if "splenic" in node_lower:
            groups.append((13,))
            groups.append((5,))
    if "venous" in role_lower:
        if "ivc" in node_lower or "lower_return" in node_lower:
            groups.append((1,))
            groups.append((43,))
        if "left_renal" in node_lower:
            groups.append((24,))
        if "right_renal" in node_lower:
            groups.append((25,))
        if "hepatic" in node_lower:
            groups.append((33, 34, 35))
        if "splenic" in node_lower:
            groups.append((21,))
            groups.append((14, 15, 16))
    for edge_id in connected_edge_ids:
        labels = edge_label_mapping.get(edge_id)
        if labels:
            groups.append(tuple(labels))
    deduped: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for group in groups:
        normalized = tuple(int(label) for label in group)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return tuple(deduped)


def _nearest_mask_point_mm(
    mask: np.ndarray,
    center_mm: tuple[float, float, float],
    spacing_mm: tuple[float, float, float],
    search_radius_mm: float,
) -> tuple[float, float, float] | None:
    if not mask.any():
        return None
    spacing_array = np.asarray(spacing_mm, dtype=float)
    center_index = np.asarray(center_mm, dtype=float) / spacing_array
    radius_index = np.ceil(search_radius_mm / spacing_array).astype(int) + 1
    mins = np.maximum(np.floor(center_index).astype(int) - radius_index, 0)
    maxs = np.minimum(np.ceil(center_index).astype(int) + radius_index + 1, np.asarray(mask.shape))
    if np.any(maxs <= mins):
        return None
    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    local_coords = np.argwhere(mask[slices])
    if len(local_coords) == 0:
        return None
    global_coords = local_coords + mins
    distances = np.linalg.norm((global_coords.astype(float) - center_index) * spacing_array, axis=1)
    best = global_coords[int(np.argmin(distances))]
    if float(np.min(distances)) > search_radius_mm:
        return None
    return tuple(float(value) for value in (best.astype(float) * spacing_array))


def _snap_graph_boundaries(
    graph: dict[str, Any],
    *,
    vessel_labels: np.ndarray,
    arterial_mask: np.ndarray,
    venous_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    search_radius_mm: float,
    edge_label_mapping: dict[str, tuple[int, ...]],
) -> tuple[dict[str, Any], int, int]:
    import copy

    snapped_graph = copy.deepcopy(graph)
    node_positions: dict[str, list[float]] = {}
    boundary_count = 0
    snapped_count = 0
    for node in snapped_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        role = str(node.get("boundary_role", ""))
        if not role:
            continue
        boundary_count += 1
        domain = _domain_for_role(role)
        domain_mask = arterial_mask if domain == "arterial" else venous_mask if domain == "venous" else None
        if domain_mask is None:
            continue
        original = _node_position_mm(node)
        connected_edge_ids = _connected_edge_ids(snapped_graph, str(node.get("id", "")))
        preferred_groups = _preferred_label_groups(
            node_id=str(node.get("id", "")),
            role=role,
            connected_edge_ids=connected_edge_ids,
            edge_label_mapping=edge_label_mapping,
        )
        snapped = None
        snapped_label_group: tuple[int, ...] | None = None
        for label_group in preferred_groups:
            preferred_mask = np.isin(vessel_labels, label_group) & domain_mask
            snapped = _nearest_mask_point_mm(preferred_mask, original, spacing_mm, search_radius_mm)
            if snapped is not None:
                snapped_label_group = label_group
                break
        if snapped is None:
            snapped = _nearest_mask_point_mm(domain_mask, original, spacing_mm, search_radius_mm)
        if snapped is None:
            node_positions[str(node.get("id", ""))] = list(original)
            continue
        node["position_mm"] = [float(value) for value in snapped]
        notes = list(node.get("notes", []))
        notes.append(f"boundary_node_snapped_to_corrected_label_lumen_within_{search_radius_mm:.1f}_mm")
        if snapped_label_group is not None:
            notes.append(f"boundary_node_snap_preferred_labels={','.join(str(label) for label in snapped_label_group)}")
        node["notes"] = notes
        node_positions[str(node.get("id", ""))] = [float(value) for value in snapped]
        snapped_count += 1

    for node in snapped_graph.get("nodes", []):
        if isinstance(node, dict) and str(node.get("id", "")) not in node_positions:
            node_positions[str(node.get("id", ""))] = [float(value) for value in _node_position_mm(node)]

    for edge in snapped_graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        polyline = edge.get("polyline_mm")
        if not isinstance(polyline, list) or len(polyline) == 0:
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in node_positions:
            polyline[0] = node_positions[source]
        if target in node_positions:
            polyline[-1] = node_positions[target]

    metadata = dict(snapped_graph.get("graph_metadata", {}))
    metadata["boundary_snap_search_radius_mm"] = float(search_radius_mm)
    metadata["boundary_node_count"] = int(boundary_count)
    metadata["snapped_boundary_node_count"] = int(snapped_count)
    metadata["flow_domain_source"] = "corrected_ct_grid_labelled_vessel_mask"
    snapped_graph["graph_metadata"] = metadata
    provenance = list(snapped_graph.get("provenance_notes", []))
    provenance.append("boundary_nodes_snapped_to_corrected_ct_grid_labelled_lumen_for_flow_domain_mapping")
    snapped_graph["provenance_notes"] = provenance
    return snapped_graph, boundary_count, snapped_count


def _paint_boundary_labels(
    graph: dict[str, Any],
    *,
    arterial_mask: np.ndarray,
    venous_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> tuple[np.ndarray, int]:
    labels = np.zeros(arterial_mask.shape, dtype=np.int16)
    spacing_array = np.asarray(spacing_mm, dtype=float)
    painted = 0
    boundary_nodes = [
        node for node in graph.get("nodes", []) if isinstance(node, dict) and str(node.get("boundary_role", ""))
    ]
    boundary_nodes.sort(key=lambda node: str(node.get("id", "")))
    for index, node in enumerate(boundary_nodes, start=1):
        domain = _domain_for_role(str(node.get("boundary_role", "")))
        domain_mask = arterial_mask if domain == "arterial" else venous_mask if domain == "venous" else None
        if domain_mask is None:
            continue
        center_mm = np.asarray(_node_position_mm(node), dtype=float)
        radius_mm = max(float(node.get("radius_mm", 1.0)) * 1.5, 2.0)
        center_index = center_mm / spacing_array
        radius_index = np.ceil(radius_mm / spacing_array).astype(int) + 1
        mins = np.maximum(np.floor(center_index).astype(int) - radius_index, 0)
        maxs = np.minimum(np.ceil(center_index).astype(int) + radius_index + 1, np.asarray(labels.shape))
        if np.any(maxs <= mins):
            continue
        slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
        grids = np.meshgrid(
            *[np.arange(mins[axis], maxs[axis], dtype=float) * spacing_array[axis] for axis in range(3)],
            indexing="ij",
        )
        distance_sq = sum((grid - center_mm[axis]) ** 2 for axis, grid in enumerate(grids))
        sphere = (distance_sq <= radius_mm**2) & domain_mask[slices]
        if not sphere.any():
            continue
        view = labels[slices]
        view[sphere] = index
        painted += int(np.count_nonzero(sphere))
    return labels, painted


def _write_label_summary(
    path: Path,
    *,
    counts: dict[int, int],
    label_names: dict[int, str],
    arterial_labels: frozenset[int],
    venous_labels: frozenset[int],
    spacing_mm: tuple[float, float, float],
) -> None:
    import csv

    voxel_volume = _voxel_volume_cm3(spacing_mm)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("label", "name", "domain", "voxels", "volume_cm3"))
        writer.writeheader()
        for label, voxels in sorted(counts.items()):
            if label in arterial_labels:
                domain = "arterial"
            elif label in venous_labels:
                domain = "venous"
            else:
                domain = "unclassified"
            writer.writerow(
                {
                    "label": label,
                    "name": label_names.get(label, ""),
                    "domain": domain,
                    "voxels": voxels,
                    "volume_cm3": f"{voxels * voxel_volume:.6f}",
                }
            )


def _region_payloads(regions: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "index": int(region.index),
            "name": str(region.name),
            "material_id": str(region.material_id),
            "target_hu_midpoint": float(region.target_hu_midpoint),
            "mass_density_g_cm3": float(region.mass_density_g_cm3),
            "relative_electron_density": float(region.relative_electron_density),
            "color": str(region.color),
        }
        for region in regions
    ]


def _write_preview(
    path: Path,
    *,
    anatomy: np.ndarray,
    vessel_labels: np.ndarray,
    arterial_mask: np.ndarray,
    venous_mask: np.ndarray,
    vessel_wall: np.ndarray,
) -> None:
    plt, _, _ = _import_dependencies()
    lumen = arterial_mask | venous_mask
    z_index = _slice_index(lumen | vessel_wall, 2)
    y_index = _slice_index(lumen | vessel_wall, 1)
    x_index = _slice_index(lumen | vessel_wall, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#f7f4ef")
    views = (
        ("Axial", anatomy[:, :, z_index], vessel_labels[:, :, z_index], arterial_mask[:, :, z_index], venous_mask[:, :, z_index], vessel_wall[:, :, z_index]),
        ("Coronal", anatomy[:, y_index, :], vessel_labels[:, y_index, :], arterial_mask[:, y_index, :], venous_mask[:, y_index, :], vessel_wall[:, y_index, :]),
        ("Sagittal", anatomy[x_index, :, :], vessel_labels[x_index, :, :], arterial_mask[x_index, :, :], venous_mask[x_index, :, :], vessel_wall[x_index, :, :]),
    )
    for ax, (title, anatomy_view, labels_view, arterial_view, venous_view, wall_view) in zip(axes, views):
        ax.imshow(np.rot90(anatomy_view), cmap="gray", interpolation="nearest")
        ax.contour(np.rot90(wall_view.astype(float)), levels=[0.5], colors=["#2d1c11"], linewidths=0.6)
        ax.contour(np.rot90(arterial_view.astype(float)), levels=[0.5], colors=["#d7263d"], linewidths=1.4)
        ax.contour(np.rot90(venous_view.astype(float)), levels=[0.5], colors=["#1b62b7"], linewidths=1.4)
        ax.imshow(np.ma.masked_where(np.rot90(labels_view) == 0, np.rot90(labels_view)), cmap="turbo", alpha=0.22, interpolation="nearest")
        ax.set_title(title, fontsize=12, color="#13202a")
        ax.axis("off")
    fig.suptitle("Corrected CT-Grid Labelled Vessel Flow Domains", fontsize=15, color="#13202a")
    fig.text(0.5, 0.02, "Red = arterial lumen, blue = venous lumen, brown = vessel wall, color fill = branch labels", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_spec(
    path: Path,
    result: LabelVesselFlowDomainResult,
    *,
    contrast_mode: str,
    wall_thickness_mm: float,
    boundary_snap_radius_mm: float,
    write_material_volumes: bool,
    regions: tuple[Any, ...],
) -> None:
    payload = {
        "case_id": result.case_id,
        "coordinate_units": "mm",
        "voxelization": {
            "source_graph": result.flow_graph_yaml_path,
            "source_graph_before_boundary_snap": result.source_graph_path,
            "source_combined_labels": result.anatomy_labels_path,
            "source_labelled_vessel_mask": result.vessel_labels_path,
            "sample_step_mm": None,
            "vessel_wall_thickness_mm": wall_thickness_mm,
            "contrast_mode": contrast_mode,
            "collision_cleanup": "not_required_label_volume_domains_are_disjoint",
            "material_property_volumes_written": write_material_volumes,
            "boundary_snap_radius_mm": boundary_snap_radius_mm,
            "connected_components": result.connected_components,
            "arterial_components": result.arterial_components,
            "venous_components": result.venous_components,
            "arterial_venous_overlap_voxels_before_cleanup": 0,
            "arterial_venous_overlap_voxels_after_cleanup": 0,
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
        "regions": _region_payloads(regions),
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_manifest(path: Path, result: LabelVesselFlowDomainResult) -> None:
    payload = {
        "case_id": result.case_id,
        "status": "flow_domain_ready" if not result.unclassified_labels else "flow_domain_ready_with_unclassified_labels_excluded",
        "source": {
            "anatomy_labels": result.anatomy_labels_path,
            "vessel_labels": result.vessel_labels_path,
            "source_graph": result.source_graph_path,
            "flow_graph": result.flow_graph_yaml_path,
        },
        "outputs": {
            "voxelized_spec": result.spec_yaml_path,
            "flow_domain_labels": result.flow_domain_labels_path,
            "arterial_lumen_mask": result.arterial_lumen_mask_path,
            "venous_lumen_mask": result.venous_lumen_mask_path,
            "label_summary_csv": result.label_summary_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "metrics": {
            "arterial_voxels": result.arterial_voxels,
            "venous_voxels": result.venous_voxels,
            "vessel_wall_voxels": result.vessel_wall_voxels,
            "boundary_node_count": result.boundary_node_count,
            "snapped_boundary_node_count": result.snapped_boundary_node_count,
            "unclassified_labels": list(result.unclassified_labels),
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: LabelVesselFlowDomainResult) -> str:
    lines = [
        "# Labelled Vessel Flow-Domain Generation",
        "",
        f"- Case ID: `{result.case_id}`",
        f"- Anatomy labels: `{Path(result.anatomy_labels_path).name}`",
        f"- Corrected vessel labels: `{Path(result.vessel_labels_path).name}`",
        f"- Flow graph: `{Path(result.flow_graph_yaml_path).name}`",
        "",
        "## Domain Summary",
        "",
        f"- Arterial lumen: {result.arterial_voxels} voxels ({result.arterial_volume_cm3:.2f} cm3), {result.arterial_components} components",
        f"- Venous lumen: {result.venous_voxels} voxels ({result.venous_volume_cm3:.2f} cm3), {result.venous_components} components",
        f"- Combined lumen: {result.combined_lumen_voxels} voxels ({result.combined_lumen_volume_cm3:.2f} cm3), {result.connected_components} components",
        f"- Vessel wall: {result.vessel_wall_voxels} voxels ({result.vessel_wall_volume_cm3:.2f} cm3)",
        f"- Branch labels present/classified: {result.label_count}/{result.classified_label_count}",
        f"- Unclassified labels excluded from flow domains: {', '.join(str(label) for label in result.unclassified_labels) if result.unclassified_labels else 'none'}",
        "",
        "## Boundary Mapping",
        "",
        f"- Boundary nodes in graph: {result.boundary_node_count}",
        f"- Boundary nodes snapped to corrected lumen: {result.snapped_boundary_node_count}",
        f"- Boundary-label voxels painted: {result.boundary_label_voxels}",
        "",
        "## Outputs",
        "",
        f"- Voxelized flow spec: `{Path(result.spec_yaml_path).name}`",
        f"- Flow-domain labels: `{Path(result.flow_domain_labels_path).name}`",
        f"- Arterial lumen mask: `{Path(result.arterial_lumen_mask_path).name}`",
        f"- Venous lumen mask: `{Path(result.venous_lumen_mask_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        "",
        "## Notes",
    ]
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_label_vessel_flow_domain(
    anatomy_labels_path: str | Path,
    vessel_labels_path: str | Path,
    graph_yaml_path: str | Path,
    output_dir: str | Path = "outputs/digital/label_vessel_flow_domain",
    case_id: str = "label_vessel_flow_domain",
    vessel_label_config: str | Path | None = "configs/labelmaps/medseg_abdominal_vasculature.yaml",
    materials_path: str | Path = "configs/materials.yaml",
    vessel_wall_thickness_mm: float = 1.5,
    contrast_mode: str = "arterial",
    boundary_snap_radius_mm: float = 40.0,
    snap_boundary_nodes: bool = True,
    write_material_volumes: bool = False,
    report_path: str | Path | None = None,
) -> LabelVesselFlowDomainResult:
    _, nib, ndimage = _import_dependencies()
    anatomy_image = nib.load(str(anatomy_labels_path))
    vessel_image = nib.load(str(vessel_labels_path))
    if not _geometry_match(anatomy_image, vessel_image):
        raise ValueError("Anatomy labels and labelled vessel mask must be co-registered on the same CT grid.")
    anatomy = np.rint(np.asanyarray(anatomy_image.dataobj)).astype(np.int16)
    vessel_labels = np.rint(np.asanyarray(vessel_image.dataobj)).astype(np.int16)
    spacing_mm = _spacing(anatomy_image)
    voxel_volume = _voxel_volume_cm3(spacing_mm)
    material_regions = _combined_regions(load_material_library(materials_path))
    body = anatomy != 0
    bone = np.isin(anatomy, (10, 11))
    allowed = body & ~bone

    label_names = _load_label_names(vessel_label_config)
    edge_label_mapping = _load_edge_label_mapping(vessel_label_config)
    arterial_labels, venous_labels = _classify_labels(label_names)
    counts = _label_counts(vessel_labels)
    present_labels = set(counts)
    arterial_mask = np.isin(vessel_labels, tuple(arterial_labels)) & allowed
    venous_mask = np.isin(vessel_labels, tuple(venous_labels)) & allowed
    overlap = arterial_mask & venous_mask
    if overlap.any():
        venous_mask &= ~overlap
    lumen_mask = arterial_mask | venous_mask
    if not lumen_mask.any():
        raise ValueError("No classified arterial or venous lumen voxels remain after body/bone clipping.")

    distance_to_lumen = ndimage.distance_transform_edt(~lumen_mask, sampling=spacing_mm)
    vessel_wall = (distance_to_lumen <= vessel_wall_thickness_mm) & ~lumen_mask & allowed
    flow_domain_labels = np.zeros(anatomy.shape, dtype=np.int16)
    flow_domain_labels[vessel_wall] = 5
    flow_domain_labels[arterial_mask] = 1
    flow_domain_labels[venous_mask] = 2
    collision_mask = np.zeros(anatomy.shape, dtype=np.uint8)
    collision_owner_labels = np.zeros(anatomy.shape, dtype=np.int16)

    graph_path = Path(graph_yaml_path)
    graph = yaml.safe_load(graph_path.read_text())
    if not isinstance(graph, dict):
        raise ValueError(f"Graph YAML is not a mapping: {graph_yaml_path}")
    if snap_boundary_nodes:
        flow_graph, boundary_count, snapped_count = _snap_graph_boundaries(
            graph,
            vessel_labels=vessel_labels,
            arterial_mask=arterial_mask,
            venous_mask=venous_mask,
            spacing_mm=spacing_mm,
            search_radius_mm=boundary_snap_radius_mm,
            edge_label_mapping=edge_label_mapping,
        )
    else:
        flow_graph = graph
        boundary_count = sum(1 for node in graph.get("nodes", []) if isinstance(node, dict) and str(node.get("boundary_role", "")))
        snapped_count = 0
    boundary_labels, boundary_voxels = _paint_boundary_labels(
        flow_graph,
        arterial_mask=arterial_mask,
        venous_mask=venous_mask,
        spacing_mm=spacing_mm,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = output / case_id
    arterial_path = base.with_name(f"{case_id}_label_vessel_arterial_lumen_mask_v001.nii.gz")
    venous_path = base.with_name(f"{case_id}_label_vessel_venous_lumen_mask_v001.nii.gz")
    lumen_path = base.with_name(f"{case_id}_label_vessel_lumen_mask_v001.nii.gz")
    flow_domain_path = base.with_name(f"{case_id}_label_vessel_flow_domain_labels_v001.nii.gz")
    wall_path = base.with_name(f"{case_id}_label_vessel_vessel_wall_mask_v001.nii.gz")
    boundary_path = base.with_name(f"{case_id}_label_vessel_boundary_labels_v001.nii.gz")
    collision_path = base.with_name(f"{case_id}_label_vessel_collision_mask_v001.nii.gz")
    collision_owner_path = base.with_name(f"{case_id}_label_vessel_collision_owner_labels_v001.nii.gz")
    blood_labels_path = base.with_name(f"{case_id}_label_vessel_material_labels_blood_v001.nii.gz")
    contrast_labels_path = base.with_name(f"{case_id}_label_vessel_material_labels_contrast_v001.nii.gz")
    blood_density_path = base.with_name(f"{case_id}_label_vessel_mass_density_blood_v001.nii.gz") if write_material_volumes else None
    blood_red_path = base.with_name(f"{case_id}_label_vessel_relative_electron_density_blood_v001.nii.gz") if write_material_volumes else None
    blood_hu_path = base.with_name(f"{case_id}_label_vessel_synthetic_hu_blood_v001.nii.gz") if write_material_volumes else None
    contrast_density_path = base.with_name(f"{case_id}_label_vessel_mass_density_contrast_v001.nii.gz") if write_material_volumes else None
    contrast_red_path = base.with_name(f"{case_id}_label_vessel_relative_electron_density_contrast_v001.nii.gz") if write_material_volumes else None
    contrast_hu_path = base.with_name(f"{case_id}_label_vessel_synthetic_hu_contrast_v001.nii.gz") if write_material_volumes else None
    flow_graph_path = base.with_name(f"{case_id}_label_vessel_flow_graph_v001.yaml")
    spec_path = base.with_name(f"{case_id}_label_vessel_flow_domain_spec_v001.yaml")
    manifest_path = base.with_name(f"{case_id}_label_vessel_flow_domain_manifest_v001.yaml")
    label_summary_csv = base.with_name(f"{case_id}_label_vessel_flow_domain_label_summary_v001.csv")
    preview_png = base.with_name(f"{case_id}_label_vessel_flow_domain_preview_v001.png")
    report = Path(report_path) if report_path else base.with_name(f"{case_id}_label_vessel_flow_domain_report_v001.md")

    blood_labels = anatomy.copy()
    contrast_labels = anatomy.copy()
    blood_labels[vessel_wall] = 13
    contrast_labels[vessel_wall] = 13
    blood_labels[lumen_mask] = 14
    contrast_labels[lumen_mask] = 14
    if contrast_mode == "arterial":
        contrast_labels[arterial_mask] = 15
    elif contrast_mode == "all":
        contrast_labels[lumen_mask] = 15

    _write_nifti(arterial_path, arterial_mask.astype(np.uint8), anatomy_image, nib)
    _write_nifti(venous_path, venous_mask.astype(np.uint8), anatomy_image, nib)
    _write_nifti(lumen_path, lumen_mask.astype(np.uint8), anatomy_image, nib)
    _write_nifti(flow_domain_path, flow_domain_labels, anatomy_image, nib)
    _write_nifti(wall_path, vessel_wall.astype(np.uint8), anatomy_image, nib)
    _write_nifti(boundary_path, boundary_labels, anatomy_image, nib)
    _write_nifti(collision_path, collision_mask, anatomy_image, nib)
    _write_nifti(collision_owner_path, collision_owner_labels, anatomy_image, nib)
    _write_nifti(blood_labels_path, blood_labels.astype(np.int16), anatomy_image, nib)
    _write_nifti(contrast_labels_path, contrast_labels.astype(np.int16), anatomy_image, nib)

    if write_material_volumes:
        _write_nifti(blood_density_path, _property_volume(blood_labels, material_regions, "mass_density_g_cm3"), anatomy_image, nib)
        _write_nifti(blood_red_path, _property_volume(blood_labels, material_regions, "relative_electron_density"), anatomy_image, nib)
        _write_nifti(blood_hu_path, _property_volume(blood_labels, material_regions, "target_hu_midpoint"), anatomy_image, nib)
        _write_nifti(contrast_density_path, _property_volume(contrast_labels, material_regions, "mass_density_g_cm3"), anatomy_image, nib)
        _write_nifti(contrast_red_path, _property_volume(contrast_labels, material_regions, "relative_electron_density"), anatomy_image, nib)
        _write_nifti(contrast_hu_path, _property_volume(contrast_labels, material_regions, "target_hu_midpoint"), anatomy_image, nib)

    _, connected_components = ndimage.label(lumen_mask, structure=np.ones((3, 3, 3), dtype=bool))
    _, arterial_components = ndimage.label(arterial_mask, structure=np.ones((3, 3, 3), dtype=bool))
    _, venous_components = ndimage.label(venous_mask, structure=np.ones((3, 3, 3), dtype=bool))
    classified = present_labels & (set(arterial_labels) | set(venous_labels))
    unclassified = tuple(sorted(present_labels - classified))
    missing_required = tuple(sorted(set(DEFAULT_REQUIRED_VESSEL_LABELS) - present_labels))
    notes = [
        "corrected_branch_label_volume_used_as_flow_lumen_source",
        "anatomy_body_label_nonzero_used_for_body_clip",
        "bone_labels_10_11_excluded_from_flow_lumen_and_wall",
        f"contrast_mode={contrast_mode}",
        f"required_p1_labels_missing={','.join(str(label) for label in missing_required) if missing_required else 'none'}",
    ]
    if snap_boundary_nodes:
        notes.append("graph_boundary_nodes_snapped_to_nearest_corrected_lumen_before_boundary_condition_generation")
    if not write_material_volumes:
        notes.append("material_property_float_volumes_skipped_to_limit_disk_use")
    if unclassified:
        notes.append("unclassified_branch_labels_are_excluded_from_arterial_venous_flow_domains")

    result = LabelVesselFlowDomainResult(
        case_id=case_id,
        anatomy_labels_path=str(anatomy_labels_path),
        vessel_labels_path=str(vessel_labels_path),
        source_graph_path=str(graph_yaml_path),
        output_dir=str(output),
        arterial_lumen_mask_path=str(arterial_path),
        venous_lumen_mask_path=str(venous_path),
        combined_lumen_mask_path=str(lumen_path),
        flow_domain_labels_path=str(flow_domain_path),
        vessel_wall_mask_path=str(wall_path),
        boundary_labels_path=str(boundary_path),
        collision_mask_path=str(collision_path),
        collision_owner_labels_path=str(collision_owner_path),
        blood_material_labels_path=str(blood_labels_path),
        contrast_material_labels_path=str(contrast_labels_path),
        blood_density_path=None if blood_density_path is None else str(blood_density_path),
        blood_relative_electron_density_path=None if blood_red_path is None else str(blood_red_path),
        blood_synthetic_hu_path=None if blood_hu_path is None else str(blood_hu_path),
        contrast_density_path=None if contrast_density_path is None else str(contrast_density_path),
        contrast_relative_electron_density_path=None if contrast_red_path is None else str(contrast_red_path),
        contrast_synthetic_hu_path=None if contrast_hu_path is None else str(contrast_hu_path),
        flow_graph_yaml_path=str(flow_graph_path),
        spec_yaml_path=str(spec_path),
        manifest_yaml_path=str(manifest_path),
        label_summary_csv_path=str(label_summary_csv),
        preview_png_path=str(preview_png),
        report_path=str(report),
        arterial_voxels=int(arterial_mask.sum()),
        venous_voxels=int(venous_mask.sum()),
        vessel_wall_voxels=int(vessel_wall.sum()),
        combined_lumen_voxels=int(lumen_mask.sum()),
        arterial_volume_cm3=float(arterial_mask.sum() * voxel_volume),
        venous_volume_cm3=float(venous_mask.sum() * voxel_volume),
        vessel_wall_volume_cm3=float(vessel_wall.sum() * voxel_volume),
        combined_lumen_volume_cm3=float(lumen_mask.sum() * voxel_volume),
        arterial_components=int(arterial_components),
        venous_components=int(venous_components),
        connected_components=int(connected_components),
        label_count=len(present_labels),
        classified_label_count=len(classified),
        unclassified_labels=unclassified,
        boundary_node_count=int(boundary_count),
        snapped_boundary_node_count=int(snapped_count),
        boundary_label_voxels=int(boundary_voxels),
        notes=tuple(notes),
    )

    flow_graph_path.write_text(yaml.safe_dump(flow_graph, sort_keys=False))
    _write_label_summary(
        label_summary_csv,
        counts=counts,
        label_names=label_names,
        arterial_labels=arterial_labels,
        venous_labels=venous_labels,
        spacing_mm=spacing_mm,
    )
    _write_preview(
        preview_png,
        anatomy=anatomy,
        vessel_labels=vessel_labels,
        arterial_mask=arterial_mask,
        venous_mask=venous_mask,
        vessel_wall=vessel_wall,
    )
    _write_spec(
        spec_path,
        result,
        contrast_mode=contrast_mode,
        wall_thickness_mm=vessel_wall_thickness_mm,
        boundary_snap_radius_mm=boundary_snap_radius_mm,
        write_material_volumes=write_material_volumes,
        regions=material_regions,
    )
    _write_manifest(manifest_path, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_label_vessel_flow_domain_result(result: LabelVesselFlowDomainResult) -> str:
    return _format_report(result)
