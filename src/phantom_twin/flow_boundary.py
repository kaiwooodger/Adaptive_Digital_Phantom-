from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np


def _import_dependencies():
    try:
        import nibabel as nib  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Flow boundary packaging requires nibabel and PyYAML.") from exc
    return nib, yaml


@dataclass(frozen=True)
class FlowBoundary:
    boundary_id: int
    node_id: str
    label: str
    vessel_type: str
    role: str
    boundary_condition_type: str
    flow_domain_label: int
    center_mm: tuple[float, float, float]
    center_ijk: tuple[float, float, float]
    flow_direction: tuple[float, float, float]
    outward_normal: tuple[float, float, float]
    graph_radius_mm: float
    measured_voxels: int
    area_mm2: float
    equivalent_diameter_mm: float
    assigned_flow_ml_s: float | None
    pressure_pa: float | None
    resistance_pa_s_per_m3: float | None
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FlowBoundaryPackageResult:
    case_id: str
    output_dir: str
    config_yaml_path: str
    boundaries_csv_path: str
    waveform_csv_path: str
    unique_boundary_labels_path: str
    report_path: str
    boundary_count: int
    mapped_boundary_count: int
    arterial_inlet_count: int
    arterial_outlet_count: int
    venous_inlet_count: int
    venous_outlet_count: int
    total_arterial_outlet_flow_ml_s: float
    total_venous_inlet_flow_ml_s: float
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    _, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _write_nifti(path: Path, data: np.ndarray, reference_image, nib) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def _spacing(image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _center_ijk(center_mm: tuple[float, float, float], spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in (np.array(center_mm, dtype=float) / np.array(spacing, dtype=float)))


def _load_mask(path: str | Path, reference_shape: tuple[int, int, int], reference_spacing: tuple[float, float, float]):
    nib, _ = _import_dependencies()
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if tuple(data.shape) != reference_shape:
        raise ValueError(f"Mask shape differs from reference: {data.shape} vs {reference_shape} for {path}")
    image_spacing = _spacing(image)
    if any(abs(image_spacing[index] - reference_spacing[index]) > 1e-6 for index in range(3)):
        raise ValueError(f"Mask spacing differs from reference: {image_spacing} vs {reference_spacing} for {path}")
    return data


def _edge_for_node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    matches = [
        edge
        for edge in graph.get("edges", [])
        if str(edge.get("source")) == node_id or str(edge.get("target")) == node_id
    ]
    if not matches:
        raise ValueError(f"No graph edge touches boundary node {node_id}")
    return matches[0]


def _flow_direction_for_node(edge: dict[str, Any], node_id: str) -> tuple[float, float, float]:
    points = np.array(edge["polyline_mm"], dtype=float)
    if len(points) < 2:
        raise ValueError(f"Edge {edge.get('id')} has fewer than two polyline points")

    if str(edge["source"]) == node_id:
        direction = points[1] - points[0]
    elif str(edge["target"]) == node_id:
        direction = points[-1] - points[-2]
    else:
        raise ValueError(f"Node {node_id} is not on edge {edge.get('id')}")
    return tuple(float(value) for value in _unit(direction))


def _outward_normal_for_role(flow_direction: tuple[float, float, float], role: str) -> tuple[float, float, float]:
    direction = np.array(flow_direction, dtype=float)
    if role.endswith("inlet"):
        direction = -direction
    return tuple(float(value) for value in direction)


def _boundary_condition_type(role: str) -> str:
    return {
        "arterial_inlet": "flow_rate_waveform",
        "arterial_outlet": "resistance_outlet_placeholder",
        "venous_inlet": "return_flow_split_placeholder",
        "venous_outlet": "pressure_outlet_placeholder",
    }[role]


def _domain_for_role(role: str) -> tuple[str, int]:
    if role.startswith("arterial"):
        return "arterial", 1
    if role.startswith("venous"):
        return "venous", 2
    raise ValueError(f"Unsupported boundary role: {role}")


def _boundary_region_mask(
    domain_mask: np.ndarray,
    center_mm: tuple[float, float, float],
    normal: tuple[float, float, float],
    radius_mm: float,
    spacing: tuple[float, float, float],
    slab_thickness_mm: float,
) -> tuple[np.ndarray, str]:
    center = np.array(center_mm, dtype=float)
    normal_array = _unit(np.array(normal, dtype=float))
    spacing_array = np.array(spacing, dtype=float)
    search_radius = radius_mm + slab_thickness_mm + max(spacing)
    center_index = center / spacing_array
    radius_index = np.ceil(search_radius / spacing_array).astype(int) + 1
    mins = np.maximum(np.floor(center_index).astype(int) - radius_index, 0)
    maxs = np.minimum(np.ceil(center_index).astype(int) + radius_index + 1, np.array(domain_mask.shape))
    output = np.zeros(domain_mask.shape, dtype=bool)
    if np.any(maxs <= mins):
        return output, "empty_outside_volume"

    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    grids = np.meshgrid(
        *[
            np.arange(mins[axis], maxs[axis], dtype=float) * spacing_array[axis]
            for axis in range(3)
        ],
        indexing="ij",
    )
    rel = np.stack([grid - center[axis] for axis, grid in enumerate(grids)], axis=-1)
    axial = rel @ normal_array
    radial_vectors = rel - axial[..., None] * normal_array
    radial_distance = np.linalg.norm(radial_vectors, axis=-1)
    slab = (np.abs(axial) <= slab_thickness_mm / 2.0) & (radial_distance <= radius_mm * 1.15)
    slab &= domain_mask[slices]
    if np.any(slab):
        output[slices] = slab
        return output, "plane_slab"

    sphere = np.linalg.norm(rel, axis=-1) <= radius_mm * 1.35
    sphere &= domain_mask[slices]
    if np.any(sphere):
        output[slices] = sphere
        return output, "fallback_sphere"

    return output, "empty_no_domain_voxels"


def _area_from_boundary_voxels(
    voxel_count: int,
    spacing: tuple[float, float, float],
    slab_thickness_mm: float,
    graph_radius_mm: float,
    method: str,
) -> tuple[float, float]:
    if voxel_count > 0 and method == "plane_slab":
        area = voxel_count * float(np.prod(spacing)) / slab_thickness_mm
    else:
        area = math.pi * graph_radius_mm**2
    diameter = 2.0 * math.sqrt(max(area, 0.0) / math.pi)
    return float(area), float(diameter)


def _waveform_samples() -> tuple[tuple[float, float], ...]:
    raw = np.array(
        [
            0.62,
            0.70,
            0.95,
            1.34,
            1.68,
            1.55,
            1.22,
            0.98,
            0.82,
            0.72,
            0.66,
            0.63,
            0.61,
            0.60,
            0.61,
            0.63,
            0.66,
            0.69,
            0.70,
            0.66,
        ],
        dtype=float,
    )
    raw = raw / float(raw.mean())
    times = np.linspace(0.0, 1.0, len(raw), endpoint=False)
    return tuple((float(t), float(v)) for t, v in zip(times, raw))


def _flow_weights(boundaries: list[FlowBoundary], role: str) -> dict[str, float]:
    selected = [boundary for boundary in boundaries if boundary.role == role and boundary.measured_voxels > 0]
    if not selected:
        return {}
    weights = np.array([max(boundary.equivalent_diameter_mm / 2.0, boundary.graph_radius_mm, 0.1) ** 3 for boundary in selected])
    weights = weights / float(weights.sum())
    return {boundary.node_id: float(weight) for boundary, weight in zip(selected, weights)}


def _with_flow_assignment(
    boundaries: list[FlowBoundary],
    arterial_inlet_flow_ml_s: float,
    nominal_outlet_pressure_drop_pa: float,
    venous_outlet_pressure_pa: float,
) -> list[FlowBoundary]:
    arterial_outlet_weights = _flow_weights(boundaries, "arterial_outlet")
    venous_inlet_weights = _flow_weights(boundaries, "venous_inlet")
    updated: list[FlowBoundary] = []
    for boundary in boundaries:
        assigned_flow: float | None = None
        pressure: float | None = None
        resistance: float | None = None
        notes = list(boundary.notes)
        if boundary.role == "arterial_inlet":
            assigned_flow = arterial_inlet_flow_ml_s
            notes.append("mean_flow_waveform_placeholder")
        elif boundary.role == "arterial_outlet":
            assigned_flow = -arterial_inlet_flow_ml_s * arterial_outlet_weights.get(boundary.node_id, 0.0)
            flow_m3_s = abs(assigned_flow) * 1e-6
            resistance = nominal_outlet_pressure_drop_pa / flow_m3_s if flow_m3_s > 0 else None
            notes.append("murray_radius_cubed_split_placeholder")
        elif boundary.role == "venous_inlet":
            assigned_flow = arterial_inlet_flow_ml_s * venous_inlet_weights.get(boundary.node_id, 0.0)
            notes.append("venous_return_split_placeholder")
        elif boundary.role == "venous_outlet":
            assigned_flow = -arterial_inlet_flow_ml_s
            pressure = venous_outlet_pressure_pa
            notes.append("central_venous_pressure_placeholder")

        updated.append(
            FlowBoundary(
                boundary_id=boundary.boundary_id,
                node_id=boundary.node_id,
                label=boundary.label,
                vessel_type=boundary.vessel_type,
                role=boundary.role,
                boundary_condition_type=boundary.boundary_condition_type,
                flow_domain_label=boundary.flow_domain_label,
                center_mm=boundary.center_mm,
                center_ijk=boundary.center_ijk,
                flow_direction=boundary.flow_direction,
                outward_normal=boundary.outward_normal,
                graph_radius_mm=boundary.graph_radius_mm,
                measured_voxels=boundary.measured_voxels,
                area_mm2=boundary.area_mm2,
                equivalent_diameter_mm=boundary.equivalent_diameter_mm,
                assigned_flow_ml_s=assigned_flow,
                pressure_pa=pressure,
                resistance_pa_s_per_m3=resistance,
                status=boundary.status,
                notes=tuple(notes),
            )
        )
    return updated


def _boundary_to_payload(boundary: FlowBoundary) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "boundary_id": boundary.boundary_id,
        "node_id": boundary.node_id,
        "label": boundary.label,
        "vessel_type": boundary.vessel_type,
        "role": boundary.role,
        "boundary_condition_type": boundary.boundary_condition_type,
        "flow_domain_label": boundary.flow_domain_label,
        "center_mm": list(boundary.center_mm),
        "center_ijk": list(boundary.center_ijk),
        "flow_direction": list(boundary.flow_direction),
        "outward_normal": list(boundary.outward_normal),
        "graph_radius_mm": boundary.graph_radius_mm,
        "measured_voxels": boundary.measured_voxels,
        "area_mm2": boundary.area_mm2,
        "equivalent_diameter_mm": boundary.equivalent_diameter_mm,
        "assigned_flow_ml_s": boundary.assigned_flow_ml_s,
        "pressure_pa": boundary.pressure_pa,
        "resistance_pa_s_per_m3": boundary.resistance_pa_s_per_m3,
        "status": boundary.status,
        "notes": list(boundary.notes),
    }
    return payload


def _write_config_yaml(
    path: Path,
    case_id: str,
    graph_path: Path,
    voxelized_spec_path: Path,
    result: FlowBoundaryPackageResult,
    boundaries: list[FlowBoundary],
    arterial_inlet_flow_ml_s: float,
    nominal_outlet_pressure_drop_pa: float,
    venous_outlet_pressure_pa: float,
    slab_thickness_mm: float,
) -> None:
    _, yaml = _import_dependencies()
    waveform = _waveform_samples()
    payload = {
        "case_id": case_id,
        "coordinate_units": "mm",
        "flow_units": "mL/s",
        "pressure_units": "Pa",
        "source_graph": str(graph_path),
        "source_voxelized_spec": str(voxelized_spec_path),
        "boundary_measurement": {
            "method": "plane_slab_cut_through_cleaned_domain_mask",
            "slab_thickness_mm": slab_thickness_mm,
            "area_units": "mm^2",
            "equivalent_diameter_units": "mm",
        },
        "solver_notes": [
            "Boundary conditions are first-pass placeholders for digital phantom testing.",
            "Positive assigned_flow_ml_s follows graph flow direction into the modeled network segment; negative values leave the modeled segment.",
            "Outward normals follow a CFD-style domain boundary convention: opposite flow at inlets, with flow at outlets.",
            "Resistance values are placeholders derived from nominal pressure drop and radius-cubed flow splits, not calibrated physiology.",
        ],
        "global_placeholders": {
            "arterial_inlet_mean_flow_ml_s": arterial_inlet_flow_ml_s,
            "nominal_arterial_outlet_pressure_drop_pa": nominal_outlet_pressure_drop_pa,
            "venous_outlet_pressure_pa": venous_outlet_pressure_pa,
        },
        "waveforms": {
            "arterial_inlet_unit_cycle": [
                {"phase": phase, "normalized_flow_multiplier": multiplier}
                for phase, multiplier in waveform
            ]
        },
        "outputs": {
            "boundaries_csv": result.boundaries_csv_path,
            "waveform_csv": result.waveform_csv_path,
            "unique_boundary_labels": result.unique_boundary_labels_path,
            "report": result.report_path,
        },
        "boundaries": [_boundary_to_payload(boundary) for boundary in boundaries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_boundaries_csv(path: Path, boundaries: list[FlowBoundary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "boundary_id",
                "node_id",
                "label",
                "vessel_type",
                "role",
                "bc_type",
                "flow_domain_label",
                "center_x_mm",
                "center_y_mm",
                "center_z_mm",
                "normal_x",
                "normal_y",
                "normal_z",
                "area_mm2",
                "equivalent_diameter_mm",
                "measured_voxels",
                "assigned_flow_ml_s",
                "pressure_pa",
                "resistance_pa_s_per_m3",
                "status",
                "notes",
            ]
        )
        for boundary in boundaries:
            writer.writerow(
                [
                    boundary.boundary_id,
                    boundary.node_id,
                    boundary.label,
                    boundary.vessel_type,
                    boundary.role,
                    boundary.boundary_condition_type,
                    boundary.flow_domain_label,
                    *[f"{value:.6f}" for value in boundary.center_mm],
                    *[f"{value:.8f}" for value in boundary.outward_normal],
                    f"{boundary.area_mm2:.6f}",
                    f"{boundary.equivalent_diameter_mm:.6f}",
                    boundary.measured_voxels,
                    "" if boundary.assigned_flow_ml_s is None else f"{boundary.assigned_flow_ml_s:.6f}",
                    "" if boundary.pressure_pa is None else f"{boundary.pressure_pa:.6f}",
                    "" if boundary.resistance_pa_s_per_m3 is None else f"{boundary.resistance_pa_s_per_m3:.6f}",
                    boundary.status,
                    ";".join(boundary.notes),
                ]
            )


def _write_waveform_csv(path: Path, arterial_inlet_flow_ml_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["phase", "normalized_flow_multiplier", "flow_ml_s"])
        for phase, multiplier in _waveform_samples():
            writer.writerow([f"{phase:.6f}", f"{multiplier:.8f}", f"{arterial_inlet_flow_ml_s * multiplier:.6f}"])


def _format_report(result: FlowBoundaryPackageResult, boundaries: list[FlowBoundary]) -> str:
    lines = [
        "# Flow Boundary-Condition Package Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Boundary count: {result.boundary_count}",
        f"- Boundaries mapped to cleaned flow-domain voxels: {result.mapped_boundary_count}",
        f"- Arterial inlets/outlets: {result.arterial_inlet_count} / {result.arterial_outlet_count}",
        f"- Venous inlets/outlets: {result.venous_inlet_count} / {result.venous_outlet_count}",
        f"- Total assigned arterial outlet flow: {result.total_arterial_outlet_flow_ml_s:.3f} mL/s",
        f"- Total assigned venous inlet flow: {result.total_venous_inlet_flow_ml_s:.3f} mL/s",
        "",
        "## Outputs",
        "",
        f"- Boundary config YAML: `{Path(result.config_yaml_path).name}`",
        f"- Boundary table CSV: `{Path(result.boundaries_csv_path).name}`",
        f"- Arterial inlet waveform CSV: `{Path(result.waveform_csv_path).name}`",
        f"- Unique boundary-label NIfTI: `{Path(result.unique_boundary_labels_path).name}`",
        "",
        "## Boundary Table",
        "",
        "| id | node | role | bc type | voxels | area mm2 | equiv diameter mm | assigned flow mL/s | pressure Pa | status |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for boundary in boundaries:
        flow = "n/a" if boundary.assigned_flow_ml_s is None else f"{boundary.assigned_flow_ml_s:.3f}"
        pressure = "n/a" if boundary.pressure_pa is None else f"{boundary.pressure_pa:.1f}"
        lines.append(
            f"| {boundary.boundary_id} | `{boundary.node_id}` | {boundary.role} | "
            f"{boundary.boundary_condition_type} | {boundary.measured_voxels} | "
            f"{boundary.area_mm2:.2f} | {boundary.equivalent_diameter_mm:.2f} | "
            f"{flow} | {pressure} | {boundary.status} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This package is simulation metadata for the digital phantom, not a calibrated physiological model.",
            "- The cleaned arterial and venous masks are used as independent flow domains.",
            "- Boundary areas are estimated from a cut slab through the voxelized lumen; small branches are limited by the source CT z-spacing.",
            "- The arterial waveform and outlet resistances are placeholders intended to be replaced by bench or CFD calibration values.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_flow_boundary_package(
    voxelized_spec_path: str | Path,
    graph_yaml_path: str | Path | None = None,
    output_dir: str | Path = "outputs/sim/flow_boundary_conditions",
    case_id: str = "ct_org_case0_imagetbad_case125",
    arterial_inlet_flow_ml_s: float = 80.0,
    nominal_outlet_pressure_drop_pa: float = 8000.0,
    venous_outlet_pressure_pa: float = 667.0,
    boundary_slab_thickness_mm: float = 5.0,
    report_path: str | Path | None = "outputs/reports/flow_boundary_conditions_stage001.md",
) -> FlowBoundaryPackageResult:
    nib, _ = _import_dependencies()
    voxelized_spec = _load_yaml(voxelized_spec_path)
    outputs = voxelized_spec["outputs"]
    graph_path = Path(graph_yaml_path or voxelized_spec["voxelization"]["source_graph"])
    graph = _load_yaml(graph_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    flow_domain_image = nib.load(str(outputs["flow_domain_labels"]))
    flow_domains = np.rint(np.asanyarray(flow_domain_image.dataobj)).astype(np.int16)
    reference_shape = tuple(int(value) for value in flow_domains.shape)
    spacing = _spacing(flow_domain_image)
    arterial_mask = _load_mask(outputs["arterial_lumen_mask"], reference_shape, spacing) > 0
    venous_mask = _load_mask(outputs["venous_lumen_mask"], reference_shape, spacing) > 0

    node_items = [node for node in graph.get("nodes", []) if str(node.get("boundary_role", ""))]
    node_items.sort(key=lambda item: str(item["id"]))
    unique_labels = np.zeros(flow_domains.shape, dtype=np.int16)
    boundaries: list[FlowBoundary] = []

    for index, node in enumerate(node_items, start=1):
        node_id = str(node["id"])
        role = str(node["boundary_role"])
        vessel_type, flow_domain_label = _domain_for_role(role)
        edge = _edge_for_node(graph, node_id)
        flow_direction = _flow_direction_for_node(edge, node_id)
        outward_normal = _outward_normal_for_role(flow_direction, role)
        center_mm = tuple(float(value) for value in node["position_mm"])
        graph_radius = float(node["radius_mm"])
        domain_mask = arterial_mask if vessel_type == "arterial" else venous_mask
        boundary_mask, method = _boundary_region_mask(
            domain_mask,
            center_mm=center_mm,
            normal=outward_normal,
            radius_mm=graph_radius,
            spacing=spacing,
            slab_thickness_mm=boundary_slab_thickness_mm,
        )
        unique_labels[boundary_mask] = index
        voxel_count = int(boundary_mask.sum())
        area, diameter = _area_from_boundary_voxels(
            voxel_count,
            spacing=spacing,
            slab_thickness_mm=boundary_slab_thickness_mm,
            graph_radius_mm=graph_radius,
            method=method,
        )
        notes = [f"area_method={method}", f"edge_id={edge.get('id')}"]
        status = "mapped" if voxel_count > 0 else "missing_voxels"
        boundaries.append(
            FlowBoundary(
                boundary_id=index,
                node_id=node_id,
                label=str(node.get("label", node_id)),
                vessel_type=vessel_type,
                role=role,
                boundary_condition_type=_boundary_condition_type(role),
                flow_domain_label=flow_domain_label,
                center_mm=center_mm,
                center_ijk=_center_ijk(center_mm, spacing),
                flow_direction=flow_direction,
                outward_normal=outward_normal,
                graph_radius_mm=graph_radius,
                measured_voxels=voxel_count,
                area_mm2=area,
                equivalent_diameter_mm=diameter,
                assigned_flow_ml_s=None,
                pressure_pa=None,
                resistance_pa_s_per_m3=None,
                status=status,
                notes=tuple(notes),
            )
        )

    boundaries = _with_flow_assignment(
        boundaries,
        arterial_inlet_flow_ml_s=arterial_inlet_flow_ml_s,
        nominal_outlet_pressure_drop_pa=nominal_outlet_pressure_drop_pa,
        venous_outlet_pressure_pa=venous_outlet_pressure_pa,
    )

    base = output / case_id
    config_yaml = base.with_name(f"{case_id}_flow_boundary_conditions_v001.yaml")
    boundaries_csv = base.with_name(f"{case_id}_flow_boundaries_v001.csv")
    waveform_csv = base.with_name(f"{case_id}_arterial_inlet_waveform_v001.csv")
    unique_boundary_labels = base.with_name(f"{case_id}_flow_boundary_unique_labels_v001.nii.gz")
    report = Path(report_path) if report_path else base.with_name(f"{case_id}_flow_boundary_conditions_report_v001.md")

    _write_nifti(unique_boundary_labels, unique_labels, flow_domain_image, nib)
    mapped_count = sum(1 for boundary in boundaries if boundary.measured_voxels > 0)
    total_arterial_outlet_flow = sum(
        abs(boundary.assigned_flow_ml_s or 0.0)
        for boundary in boundaries
        if boundary.role == "arterial_outlet"
    )
    total_venous_inlet_flow = sum(
        abs(boundary.assigned_flow_ml_s or 0.0)
        for boundary in boundaries
        if boundary.role == "venous_inlet"
    )
    notes = [
        "flow_boundary_package_uses_cleaned_non_overlapping_arterial_and_venous_masks",
        "boundary_conditions_are_placeholders_for_solver_setup_not_calibrated_physiology",
        f"boundary_slab_thickness_mm={boundary_slab_thickness_mm}",
    ]
    if mapped_count == len(boundaries):
        notes.append("all_inlet_outlet_boundaries_mapped_to_cleaned_domain_voxels")
    else:
        notes.append(f"unmapped_boundaries={len(boundaries) - mapped_count}")

    result = FlowBoundaryPackageResult(
        case_id=case_id,
        output_dir=str(output),
        config_yaml_path=str(config_yaml),
        boundaries_csv_path=str(boundaries_csv),
        waveform_csv_path=str(waveform_csv),
        unique_boundary_labels_path=str(unique_boundary_labels),
        report_path=str(report),
        boundary_count=len(boundaries),
        mapped_boundary_count=mapped_count,
        arterial_inlet_count=sum(1 for boundary in boundaries if boundary.role == "arterial_inlet"),
        arterial_outlet_count=sum(1 for boundary in boundaries if boundary.role == "arterial_outlet"),
        venous_inlet_count=sum(1 for boundary in boundaries if boundary.role == "venous_inlet"),
        venous_outlet_count=sum(1 for boundary in boundaries if boundary.role == "venous_outlet"),
        total_arterial_outlet_flow_ml_s=total_arterial_outlet_flow,
        total_venous_inlet_flow_ml_s=total_venous_inlet_flow,
        notes=tuple(notes),
    )

    _write_boundaries_csv(boundaries_csv, boundaries)
    _write_waveform_csv(waveform_csv, arterial_inlet_flow_ml_s)
    _write_config_yaml(
        config_yaml,
        case_id=case_id,
        graph_path=graph_path,
        voxelized_spec_path=Path(voxelized_spec_path),
        result=result,
        boundaries=boundaries,
        arterial_inlet_flow_ml_s=arterial_inlet_flow_ml_s,
        nominal_outlet_pressure_drop_pa=nominal_outlet_pressure_drop_pa,
        venous_outlet_pressure_pa=venous_outlet_pressure_pa,
        slab_thickness_mm=boundary_slab_thickness_mm,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, boundaries))
    return result


def format_flow_boundary_package_result(result: FlowBoundaryPackageResult) -> str:
    lines = [
        "Flow boundary-condition package created",
        f"Case ID: {result.case_id}",
        f"Boundaries: {result.boundary_count}",
        f"Mapped boundaries: {result.mapped_boundary_count}/{result.boundary_count}",
        f"Arterial inlets/outlets: {result.arterial_inlet_count}/{result.arterial_outlet_count}",
        f"Venous inlets/outlets: {result.venous_inlet_count}/{result.venous_outlet_count}",
        f"Config YAML: {result.config_yaml_path}",
        f"Boundaries CSV: {result.boundaries_csv_path}",
        f"Waveform CSV: {result.waveform_csv_path}",
        f"Unique boundary labels: {result.unique_boundary_labels_path}",
        f"Report: {result.report_path}",
    ]
    return "\n".join(lines)
