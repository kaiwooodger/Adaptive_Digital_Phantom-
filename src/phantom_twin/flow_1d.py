from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np

from .vessel_radius_profile import edge_radius_at_fraction


def _import_dependencies():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("1D flow modeling requires PyYAML.") from exc
    return yaml


@dataclass(frozen=True)
class Flow1DSegmentResult:
    edge_id: str
    label: str
    source: str
    target: str
    vessel_type: str
    flow_role: str
    length_mm: float
    radius_start_mm: float
    radius_end_mm: float
    radius_mean_mm: float
    area_mm2: float
    hydraulic_resistance_pa_s_per_m3: float
    flow_ml_s: float
    mean_velocity_cm_s: float
    pressure_source_pa: float
    pressure_target_pa: float
    pressure_drop_pa: float
    pressure_equation_residual_pa: float
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Flow1DNodeResult:
    node_id: str
    label: str
    vessel_type: str
    role: str
    boundary_role: str
    boundary_condition_type: str
    boundary_flow_ml_s: float
    boundary_pressure_pa: float | None
    boundary_resistance_pa_s_per_m3: float | None
    pressure_pa: float
    incoming_edge_flow_ml_s: float
    outgoing_edge_flow_ml_s: float
    net_residual_ml_s: float
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Flow1DMassBalanceResult:
    vessel_type: str
    node_count: int
    edge_count: int
    boundary_inflow_ml_s: float
    boundary_outflow_ml_s: float
    signed_boundary_net_ml_s: float
    max_abs_node_residual_ml_s: float
    rms_node_residual_ml_s: float
    status: str


@dataclass(frozen=True)
class Flow1DModelResult:
    case_id: str
    output_dir: str
    model_yaml_path: str
    segments_csv_path: str
    nodes_csv_path: str
    mass_balance_csv_path: str
    report_path: str
    source_graph_path: str
    source_boundary_config_path: str
    blood_viscosity_cp: float
    arterial_inlet_pressure_pa: float
    venous_outlet_pressure_pa: float
    segment_count: int
    node_count: int
    arterial_total_flow_ml_s: float
    venous_total_flow_ml_s: float
    max_abs_mass_balance_residual_ml_s: float
    max_abs_pressure_equation_residual_pa: float
    segments: tuple[Flow1DSegmentResult, ...]
    nodes: tuple[Flow1DNodeResult, ...]
    mass_balance: tuple[Flow1DMassBalanceResult, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _node_payload_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in graph.get("nodes", [])}


def _boundary_payload_by_node(boundary_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(boundary["node_id"]): boundary
        for boundary in boundary_config.get("boundaries", [])
        if boundary.get("node_id") is not None
    }


def _edge_points(edge: dict[str, Any], node_by_id: dict[str, dict[str, Any]]) -> np.ndarray:
    raw_points = edge.get("polyline_mm")
    if raw_points:
        points = np.array(raw_points, dtype=float)
        if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 3:
            return points

    source = node_by_id[str(edge["source"])]
    target = node_by_id[str(edge["target"])]
    return np.array([source["position_mm"], target["position_mm"]], dtype=float)


def _edge_resistance(
    edge: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    viscosity_pa_s: float,
) -> tuple[float, float, float]:
    points = _edge_points(edge, node_by_id)
    deltas = np.diff(points, axis=0)
    segment_lengths_mm = np.linalg.norm(deltas, axis=1)
    valid = segment_lengths_mm > 1e-9
    segment_lengths_mm = segment_lengths_mm[valid]
    if segment_lengths_mm.size == 0:
        length_mm = _safe_float(edge.get("length_mm"))
        radius = max(_safe_float(edge.get("radius_start_mm"), 1.0), 0.1)
        resistance = 8.0 * viscosity_pa_s * (length_mm * 1e-3) / (math.pi * (radius * 1e-3) ** 4)
        return float(length_mm), float(radius), float(resistance)

    length_mm = float(segment_lengths_mm.sum())
    cumulative_end = np.cumsum(segment_lengths_mm)
    cumulative_mid = cumulative_end - segment_lengths_mm / 2.0
    fraction = cumulative_mid / max(length_mm, 1e-9)
    radii_mm = np.maximum(
        np.asarray([edge_radius_at_fraction(edge, float(item), minimum_radius_mm=0.1) for item in fraction], dtype=float),
        0.1,
    )

    segment_lengths_m = segment_lengths_mm * 1e-3
    radii_m = radii_mm * 1e-3
    resistance = float(np.sum(8.0 * viscosity_pa_s * segment_lengths_m / (math.pi * radii_m**4)))
    radius_mean = float(np.average(radii_mm, weights=segment_lengths_mm))
    return length_mm, radius_mean, resistance


def _vessel_types(edges: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(edge.get("vessel_type", "unknown")) for edge in edges}))


def _nodes_for_edges(edges: list[dict[str, Any]]) -> tuple[str, ...]:
    node_ids = {str(edge["source"]) for edge in edges}
    node_ids.update(str(edge["target"]) for edge in edges)
    return tuple(sorted(node_ids))


def _solve_edge_flows(
    node_ids: tuple[str, ...],
    edges: list[dict[str, Any]],
    boundary_flow_by_node: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    if not edges:
        return {}, {}

    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    incidence = np.zeros((len(node_ids), len(edges)), dtype=float)
    for edge_index, edge in enumerate(edges):
        incidence[node_index[str(edge["source"])], edge_index] = -1.0
        incidence[node_index[str(edge["target"])], edge_index] = 1.0

    boundary = np.array([boundary_flow_by_node.get(node_id, 0.0) for node_id in node_ids], dtype=float)
    flows, *_ = np.linalg.lstsq(incidence, -boundary, rcond=None)
    residuals = incidence @ flows + boundary
    flow_by_edge = {str(edge["id"]): float(flow) for edge, flow in zip(edges, flows)}
    residual_by_node = {node_id: float(residual) for node_id, residual in zip(node_ids, residuals)}
    return flow_by_edge, residual_by_node


def _reference_pressures(
    vessel_type: str,
    node_ids: tuple[str, ...],
    boundary_by_node: dict[str, dict[str, Any]],
    arterial_inlet_pressure_pa: float,
    venous_outlet_pressure_pa: float,
) -> dict[str, float]:
    references: dict[str, float] = {}
    for node_id in node_ids:
        boundary = boundary_by_node.get(node_id)
        if not boundary:
            continue
        role = str(boundary.get("role", ""))
        configured_pressure = _optional_float(boundary.get("pressure_pa"))
        if vessel_type == "arterial" and role == "arterial_inlet":
            references[node_id] = arterial_inlet_pressure_pa
        elif vessel_type == "venous" and role == "venous_outlet":
            references[node_id] = venous_outlet_pressure_pa
        elif configured_pressure is not None:
            references[node_id] = configured_pressure

    if references:
        return references
    if node_ids:
        references[node_ids[0]] = 0.0
    return references


def _solve_pressures(
    node_ids: tuple[str, ...],
    edges: list[dict[str, Any]],
    flow_by_edge: dict[str, float],
    resistance_by_edge: dict[str, float],
    references: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    if not node_ids:
        return {}, {}

    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for edge in edges:
        row = np.zeros(len(node_ids), dtype=float)
        row[node_index[str(edge["source"])]] = 1.0
        row[node_index[str(edge["target"])]] = -1.0
        edge_id = str(edge["id"])
        rows.append(row)
        rhs.append(resistance_by_edge[edge_id] * flow_by_edge.get(edge_id, 0.0) * 1e-6)

    for node_id, pressure_pa in references.items():
        if node_id not in node_index:
            continue
        row = np.zeros(len(node_ids), dtype=float)
        row[node_index[node_id]] = 1.0
        rows.append(row)
        rhs.append(float(pressure_pa))

    matrix = np.vstack(rows)
    vector = np.array(rhs, dtype=float)
    solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    pressure_by_node = {node_id: float(solution[index]) for node_id, index in node_index.items()}

    residual_by_edge: dict[str, float] = {}
    for edge in edges:
        edge_id = str(edge["id"])
        source = str(edge["source"])
        target = str(edge["target"])
        expected_drop = resistance_by_edge[edge_id] * flow_by_edge.get(edge_id, 0.0) * 1e-6
        actual_drop = pressure_by_node[source] - pressure_by_node[target]
        residual_by_edge[edge_id] = float(actual_drop - expected_drop)
    return pressure_by_node, residual_by_edge


def _segment_status(flow_ml_s: float, residual_pa: float) -> str:
    if abs(residual_pa) > 1e-3:
        return "pressure_residual_warning"
    if flow_ml_s < -1e-6:
        return "reverse_flow"
    if abs(flow_ml_s) <= 1e-6:
        return "zero_flow"
    return "ok"


def _node_status(residual_ml_s: float) -> str:
    return "balanced" if abs(residual_ml_s) <= 1e-6 else "mass_balance_warning"


def _segment_to_payload(segment: Flow1DSegmentResult) -> dict[str, Any]:
    return {
        "edge_id": segment.edge_id,
        "label": segment.label,
        "source": segment.source,
        "target": segment.target,
        "vessel_type": segment.vessel_type,
        "flow_role": segment.flow_role,
        "length_mm": segment.length_mm,
        "radius_start_mm": segment.radius_start_mm,
        "radius_end_mm": segment.radius_end_mm,
        "radius_mean_mm": segment.radius_mean_mm,
        "area_mm2": segment.area_mm2,
        "hydraulic_resistance_pa_s_per_m3": segment.hydraulic_resistance_pa_s_per_m3,
        "flow_ml_s": segment.flow_ml_s,
        "mean_velocity_cm_s": segment.mean_velocity_cm_s,
        "pressure_source_pa": segment.pressure_source_pa,
        "pressure_target_pa": segment.pressure_target_pa,
        "pressure_drop_pa": segment.pressure_drop_pa,
        "pressure_equation_residual_pa": segment.pressure_equation_residual_pa,
        "status": segment.status,
        "notes": list(segment.notes),
    }


def _node_to_payload(node: Flow1DNodeResult) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "label": node.label,
        "vessel_type": node.vessel_type,
        "role": node.role,
        "boundary_role": node.boundary_role,
        "boundary_condition_type": node.boundary_condition_type,
        "boundary_flow_ml_s": node.boundary_flow_ml_s,
        "boundary_pressure_pa": node.boundary_pressure_pa,
        "boundary_resistance_pa_s_per_m3": node.boundary_resistance_pa_s_per_m3,
        "pressure_pa": node.pressure_pa,
        "incoming_edge_flow_ml_s": node.incoming_edge_flow_ml_s,
        "outgoing_edge_flow_ml_s": node.outgoing_edge_flow_ml_s,
        "net_residual_ml_s": node.net_residual_ml_s,
        "status": node.status,
        "notes": list(node.notes),
    }


def _mass_balance_to_payload(row: Flow1DMassBalanceResult) -> dict[str, Any]:
    return {
        "vessel_type": row.vessel_type,
        "node_count": row.node_count,
        "edge_count": row.edge_count,
        "boundary_inflow_ml_s": row.boundary_inflow_ml_s,
        "boundary_outflow_ml_s": row.boundary_outflow_ml_s,
        "signed_boundary_net_ml_s": row.signed_boundary_net_ml_s,
        "max_abs_node_residual_ml_s": row.max_abs_node_residual_ml_s,
        "rms_node_residual_ml_s": row.rms_node_residual_ml_s,
        "status": row.status,
    }


def _write_segments_csv(path: Path, segments: tuple[Flow1DSegmentResult, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "edge_id",
                "label",
                "source",
                "target",
                "vessel_type",
                "flow_role",
                "length_mm",
                "radius_start_mm",
                "radius_end_mm",
                "radius_mean_mm",
                "area_mm2",
                "hydraulic_resistance_pa_s_per_m3",
                "flow_ml_s",
                "mean_velocity_cm_s",
                "pressure_source_pa",
                "pressure_target_pa",
                "pressure_drop_pa",
                "pressure_equation_residual_pa",
                "status",
                "notes",
            ]
        )
        for segment in segments:
            writer.writerow(
                [
                    segment.edge_id,
                    segment.label,
                    segment.source,
                    segment.target,
                    segment.vessel_type,
                    segment.flow_role,
                    f"{segment.length_mm:.6f}",
                    f"{segment.radius_start_mm:.6f}",
                    f"{segment.radius_end_mm:.6f}",
                    f"{segment.radius_mean_mm:.6f}",
                    f"{segment.area_mm2:.6f}",
                    f"{segment.hydraulic_resistance_pa_s_per_m3:.6f}",
                    f"{segment.flow_ml_s:.6f}",
                    f"{segment.mean_velocity_cm_s:.6f}",
                    f"{segment.pressure_source_pa:.6f}",
                    f"{segment.pressure_target_pa:.6f}",
                    f"{segment.pressure_drop_pa:.6f}",
                    f"{segment.pressure_equation_residual_pa:.9f}",
                    segment.status,
                    ";".join(segment.notes),
                ]
            )


def _write_nodes_csv(path: Path, nodes: tuple[Flow1DNodeResult, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "node_id",
                "label",
                "vessel_type",
                "role",
                "boundary_role",
                "bc_type",
                "boundary_flow_ml_s",
                "boundary_pressure_pa",
                "boundary_resistance_pa_s_per_m3",
                "pressure_pa",
                "incoming_edge_flow_ml_s",
                "outgoing_edge_flow_ml_s",
                "net_residual_ml_s",
                "status",
                "notes",
            ]
        )
        for node in nodes:
            writer.writerow(
                [
                    node.node_id,
                    node.label,
                    node.vessel_type,
                    node.role,
                    node.boundary_role,
                    node.boundary_condition_type,
                    f"{node.boundary_flow_ml_s:.6f}",
                    "" if node.boundary_pressure_pa is None else f"{node.boundary_pressure_pa:.6f}",
                    "" if node.boundary_resistance_pa_s_per_m3 is None else f"{node.boundary_resistance_pa_s_per_m3:.6f}",
                    f"{node.pressure_pa:.6f}",
                    f"{node.incoming_edge_flow_ml_s:.6f}",
                    f"{node.outgoing_edge_flow_ml_s:.6f}",
                    f"{node.net_residual_ml_s:.9f}",
                    node.status,
                    ";".join(node.notes),
                ]
            )


def _write_mass_balance_csv(path: Path, rows: tuple[Flow1DMassBalanceResult, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "vessel_type",
                "node_count",
                "edge_count",
                "boundary_inflow_ml_s",
                "boundary_outflow_ml_s",
                "signed_boundary_net_ml_s",
                "max_abs_node_residual_ml_s",
                "rms_node_residual_ml_s",
                "status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.vessel_type,
                    row.node_count,
                    row.edge_count,
                    f"{row.boundary_inflow_ml_s:.6f}",
                    f"{row.boundary_outflow_ml_s:.6f}",
                    f"{row.signed_boundary_net_ml_s:.9f}",
                    f"{row.max_abs_node_residual_ml_s:.9f}",
                    f"{row.rms_node_residual_ml_s:.9f}",
                    row.status,
                ]
            )


def _write_model_yaml(path: Path, result: Flow1DModelResult) -> None:
    yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "model_type": "first_pass_steady_1d_flow_network",
        "source_graph": result.source_graph_path,
        "source_boundary_config": result.source_boundary_config_path,
        "units": {
            "length": "mm",
            "radius": "mm",
            "area": "mm^2",
            "flow": "mL/s",
            "pressure": "Pa",
            "viscosity": "cP",
            "hydraulic_resistance": "Pa*s/m^3",
        },
        "inputs": {
            "blood_viscosity_cp": result.blood_viscosity_cp,
            "arterial_inlet_pressure_pa": result.arterial_inlet_pressure_pa,
            "venous_outlet_pressure_pa": result.venous_outlet_pressure_pa,
        },
        "assumptions": [
            "steady_mean_flow_only",
            "rigid_circular_tubes",
            "laminar_poiseuille_resistance_from_scaffold_radii",
            "no_wall_compliance_or_pulsatile_wave_propagation",
            "no_minor_loss_terms_at_bifurcations_or_diameter_changes",
            "terminal_boundary_resistances_not_yet_coupled_into_network_solve",
        ],
        "outputs": {
            "segments_csv": result.segments_csv_path,
            "nodes_csv": result.nodes_csv_path,
            "mass_balance_csv": result.mass_balance_csv_path,
            "report": result.report_path,
        },
        "summary": {
            "segment_count": result.segment_count,
            "node_count": result.node_count,
            "arterial_total_flow_ml_s": result.arterial_total_flow_ml_s,
            "venous_total_flow_ml_s": result.venous_total_flow_ml_s,
            "max_abs_mass_balance_residual_ml_s": result.max_abs_mass_balance_residual_ml_s,
            "max_abs_pressure_equation_residual_pa": result.max_abs_pressure_equation_residual_pa,
        },
        "mass_balance": [_mass_balance_to_payload(row) for row in result.mass_balance],
        "segments": [_segment_to_payload(segment) for segment in result.segments],
        "nodes": [_node_to_payload(node) for node in result.nodes],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: Flow1DModelResult) -> str:
    lines = [
        "# 1D Flow Model Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Segments solved: {result.segment_count}",
        f"- Nodes solved: {result.node_count}",
        f"- Blood viscosity: {result.blood_viscosity_cp:.3f} cP",
        f"- Arterial reference pressure: {result.arterial_inlet_pressure_pa:.1f} Pa",
        f"- Venous outlet reference pressure: {result.venous_outlet_pressure_pa:.1f} Pa",
        f"- Arterial mean flow through inlet tree: {result.arterial_total_flow_ml_s:.3f} mL/s",
        f"- Venous mean return flow: {result.venous_total_flow_ml_s:.3f} mL/s",
        f"- Max node mass-balance residual: {result.max_abs_mass_balance_residual_ml_s:.9f} mL/s",
        f"- Max pressure-equation residual: {result.max_abs_pressure_equation_residual_pa:.9f} Pa",
        "",
        "## Outputs",
        "",
        f"- Model YAML: `{Path(result.model_yaml_path).name}`",
        f"- Segment table CSV: `{Path(result.segments_csv_path).name}`",
        f"- Node table CSV: `{Path(result.nodes_csv_path).name}`",
        f"- Mass-balance CSV: `{Path(result.mass_balance_csv_path).name}`",
        "",
        "## Mass Balance",
        "",
        "| domain | nodes | edges | inflow mL/s | outflow mL/s | signed net mL/s | max residual mL/s | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result.mass_balance:
        lines.append(
            f"| {row.vessel_type} | {row.node_count} | {row.edge_count} | "
            f"{row.boundary_inflow_ml_s:.3f} | {row.boundary_outflow_ml_s:.3f} | "
            f"{row.signed_boundary_net_ml_s:.9f} | {row.max_abs_node_residual_ml_s:.9f} | {row.status} |"
        )

    lines.extend(
        [
            "",
            "## Segment Flow And Pressure",
            "",
            "| edge | domain | flow mL/s | radius mm | resistance Pa*s/m3 | pressure drop Pa | status |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for segment in result.segments:
        lines.append(
            f"| `{segment.edge_id}` | {segment.vessel_type} | {segment.flow_ml_s:.3f} | "
            f"{segment.radius_mean_mm:.2f} | {segment.hydraulic_resistance_pa_s_per_m3:.3e} | "
            f"{segment.pressure_drop_pa:.2f} | {segment.status} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the first computational flow layer for the digital phantom, not a calibrated patient-specific hemodynamic model.",
            "- Flows come from the existing boundary-condition package and are balanced over the graph using directed node conservation.",
            "- Pressure drops use Poiseuille resistance for the synthetic circular vessel scaffold, so small placeholder branches can produce exaggerated drops.",
            "- The next upgrade is coupling outlet resistance/compliance terms and adding a pulsatile waveform simulation.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_flow_1d_model(
    graph_yaml_path: str | Path,
    boundary_config_path: str | Path,
    output_dir: str | Path = "outputs/sim/flow_1d",
    case_id: str = "ct_org_case0_imagetbad_case125",
    blood_viscosity_cp: float = 3.5,
    arterial_inlet_pressure_pa: float = 13332.0,
    venous_outlet_pressure_pa: float | None = None,
    report_path: str | Path | None = "outputs/reports/flow_1d_model_stage001.md",
) -> Flow1DModelResult:
    graph_path = Path(graph_yaml_path)
    boundary_path = Path(boundary_config_path)
    graph = _load_yaml(graph_path)
    boundary_config = _load_yaml(boundary_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    global_placeholders = boundary_config.get("global_placeholders", {})
    resolved_venous_outlet_pressure = (
        float(venous_outlet_pressure_pa)
        if venous_outlet_pressure_pa is not None
        else float(global_placeholders.get("venous_outlet_pressure_pa", 667.0))
    )
    viscosity_pa_s = blood_viscosity_cp * 0.001
    node_by_id = _node_payload_by_id(graph)
    boundary_by_node = _boundary_payload_by_node(boundary_config)
    graph_edges = [edge for edge in graph.get("edges", []) if edge.get("id") is not None]

    all_segments: list[Flow1DSegmentResult] = []
    all_nodes: list[Flow1DNodeResult] = []
    mass_rows: list[Flow1DMassBalanceResult] = []
    max_pressure_residual = 0.0

    for vessel_type in _vessel_types(graph_edges):
        domain_edges = [edge for edge in graph_edges if str(edge.get("vessel_type", "unknown")) == vessel_type]
        node_ids = _nodes_for_edges(domain_edges)
        boundary_flow_by_node = {
            node_id: _safe_float(boundary_by_node.get(node_id, {}).get("assigned_flow_ml_s"))
            for node_id in node_ids
        }
        flow_by_edge, residual_by_node = _solve_edge_flows(node_ids, domain_edges, boundary_flow_by_node)

        resistance_by_edge: dict[str, float] = {}
        geometry_by_edge: dict[str, tuple[float, float]] = {}
        for edge in domain_edges:
            edge_id = str(edge["id"])
            length_mm, radius_mean_mm, resistance = _edge_resistance(edge, node_by_id, viscosity_pa_s)
            resistance_by_edge[edge_id] = resistance
            geometry_by_edge[edge_id] = (length_mm, radius_mean_mm)

        references = _reference_pressures(
            vessel_type,
            node_ids,
            boundary_by_node=boundary_by_node,
            arterial_inlet_pressure_pa=arterial_inlet_pressure_pa,
            venous_outlet_pressure_pa=resolved_venous_outlet_pressure,
        )
        pressure_by_node, pressure_residual_by_edge = _solve_pressures(
            node_ids,
            domain_edges,
            flow_by_edge,
            resistance_by_edge,
            references,
        )

        incoming_by_node = {node_id: 0.0 for node_id in node_ids}
        outgoing_by_node = {node_id: 0.0 for node_id in node_ids}
        for edge in domain_edges:
            edge_id = str(edge["id"])
            flow_ml_s = flow_by_edge.get(edge_id, 0.0)
            outgoing_by_node[str(edge["source"])] += flow_ml_s
            incoming_by_node[str(edge["target"])] += flow_ml_s

            length_mm, radius_mean_mm = geometry_by_edge[edge_id]
            radius_start_mm = edge_radius_at_fraction(edge, 0.0, minimum_radius_mm=0.1)
            radius_end_mm = edge_radius_at_fraction(edge, 1.0, minimum_radius_mm=0.1)
            area_mm2 = math.pi * radius_mean_mm**2
            velocity_cm_s = (flow_ml_s * 1000.0 / area_mm2) / 10.0 if area_mm2 > 0 else 0.0
            pressure_source = pressure_by_node.get(str(edge["source"]), 0.0)
            pressure_target = pressure_by_node.get(str(edge["target"]), 0.0)
            pressure_drop = pressure_source - pressure_target
            pressure_residual = pressure_residual_by_edge.get(edge_id, 0.0)
            max_pressure_residual = max(max_pressure_residual, abs(pressure_residual))
            notes = tuple(str(note) for note in edge.get("notes", []))
            all_segments.append(
                Flow1DSegmentResult(
                    edge_id=edge_id,
                    label=str(edge.get("label", edge_id)),
                    source=str(edge["source"]),
                    target=str(edge["target"]),
                    vessel_type=vessel_type,
                    flow_role=str(edge.get("flow_role", "")),
                    length_mm=length_mm,
                    radius_start_mm=radius_start_mm,
                    radius_end_mm=radius_end_mm,
                    radius_mean_mm=radius_mean_mm,
                    area_mm2=area_mm2,
                    hydraulic_resistance_pa_s_per_m3=resistance_by_edge[edge_id],
                    flow_ml_s=flow_ml_s,
                    mean_velocity_cm_s=velocity_cm_s,
                    pressure_source_pa=pressure_source,
                    pressure_target_pa=pressure_target,
                    pressure_drop_pa=pressure_drop,
                    pressure_equation_residual_pa=pressure_residual,
                    status=_segment_status(flow_ml_s, pressure_residual),
                    notes=notes,
                )
            )

        for node_id in node_ids:
            node = node_by_id.get(node_id, {})
            boundary = boundary_by_node.get(node_id, {})
            boundary_flow = boundary_flow_by_node.get(node_id, 0.0)
            residual = residual_by_node.get(node_id, 0.0)
            notes = tuple(str(note) for note in node.get("notes", []))
            all_nodes.append(
                Flow1DNodeResult(
                    node_id=node_id,
                    label=str(node.get("label", node_id)),
                    vessel_type=vessel_type,
                    role=str(node.get("role", "")),
                    boundary_role=str(node.get("boundary_role", boundary.get("role", ""))),
                    boundary_condition_type=str(boundary.get("boundary_condition_type", "")),
                    boundary_flow_ml_s=boundary_flow,
                    boundary_pressure_pa=_optional_float(boundary.get("pressure_pa")),
                    boundary_resistance_pa_s_per_m3=_optional_float(boundary.get("resistance_pa_s_per_m3")),
                    pressure_pa=pressure_by_node.get(node_id, 0.0),
                    incoming_edge_flow_ml_s=incoming_by_node.get(node_id, 0.0),
                    outgoing_edge_flow_ml_s=outgoing_by_node.get(node_id, 0.0),
                    net_residual_ml_s=residual,
                    status=_node_status(residual),
                    notes=notes,
                )
            )

        boundary_values = np.array(list(boundary_flow_by_node.values()), dtype=float)
        residual_values = np.array(list(residual_by_node.values()), dtype=float)
        boundary_inflow = float(boundary_values[boundary_values > 0.0].sum()) if boundary_values.size else 0.0
        boundary_outflow = float(abs(boundary_values[boundary_values < 0.0].sum())) if boundary_values.size else 0.0
        max_residual = float(np.max(np.abs(residual_values))) if residual_values.size else 0.0
        rms_residual = float(np.sqrt(np.mean(residual_values**2))) if residual_values.size else 0.0
        mass_rows.append(
            Flow1DMassBalanceResult(
                vessel_type=vessel_type,
                node_count=len(node_ids),
                edge_count=len(domain_edges),
                boundary_inflow_ml_s=boundary_inflow,
                boundary_outflow_ml_s=boundary_outflow,
                signed_boundary_net_ml_s=float(boundary_values.sum()) if boundary_values.size else 0.0,
                max_abs_node_residual_ml_s=max_residual,
                rms_node_residual_ml_s=rms_residual,
                status="balanced" if max_residual <= 1e-6 else "mass_balance_warning",
            )
        )

    all_segments.sort(key=lambda segment: (segment.vessel_type, segment.edge_id))
    all_nodes.sort(key=lambda node: (node.vessel_type, node.node_id))
    mass_rows.sort(key=lambda row: row.vessel_type)

    base = output / case_id
    model_yaml = base.with_name(f"{case_id}_flow_1d_model_v001.yaml")
    segments_csv = base.with_name(f"{case_id}_flow_1d_segments_v001.csv")
    nodes_csv = base.with_name(f"{case_id}_flow_1d_nodes_v001.csv")
    mass_balance_csv = base.with_name(f"{case_id}_flow_1d_mass_balance_v001.csv")
    report = Path(report_path) if report_path else base.with_name(f"{case_id}_flow_1d_model_report_v001.md")

    max_mass_residual = max((row.max_abs_node_residual_ml_s for row in mass_rows), default=0.0)
    arterial_flow = next((row.boundary_inflow_ml_s for row in mass_rows if row.vessel_type == "arterial"), 0.0)
    venous_flow = next((row.boundary_inflow_ml_s for row in mass_rows if row.vessel_type == "venous"), 0.0)
    notes = [
        "first_pass_steady_1d_network_from_existing_graph_and_boundary_package",
        "positive_segment_flow_follows_graph_edge_direction",
        "mass_balance_solved_with_linear_node_conservation",
        "pressure_estimates_anchor_arterial_inlet_and_venous_outlet_reference_pressures",
        "not_calibrated_for_patient_specific_or_clinical_use",
    ]

    result = Flow1DModelResult(
        case_id=case_id,
        output_dir=str(output),
        model_yaml_path=str(model_yaml),
        segments_csv_path=str(segments_csv),
        nodes_csv_path=str(nodes_csv),
        mass_balance_csv_path=str(mass_balance_csv),
        report_path=str(report),
        source_graph_path=str(graph_path),
        source_boundary_config_path=str(boundary_path),
        blood_viscosity_cp=blood_viscosity_cp,
        arterial_inlet_pressure_pa=arterial_inlet_pressure_pa,
        venous_outlet_pressure_pa=resolved_venous_outlet_pressure,
        segment_count=len(all_segments),
        node_count=len(all_nodes),
        arterial_total_flow_ml_s=arterial_flow,
        venous_total_flow_ml_s=venous_flow,
        max_abs_mass_balance_residual_ml_s=max_mass_residual,
        max_abs_pressure_equation_residual_pa=max_pressure_residual,
        segments=tuple(all_segments),
        nodes=tuple(all_nodes),
        mass_balance=tuple(mass_rows),
        notes=tuple(notes),
    )

    _write_segments_csv(segments_csv, result.segments)
    _write_nodes_csv(nodes_csv, result.nodes)
    _write_mass_balance_csv(mass_balance_csv, result.mass_balance)
    _write_model_yaml(model_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_flow_1d_model_result(result: Flow1DModelResult) -> str:
    lines = [
        "1D flow model created",
        f"Case ID: {result.case_id}",
        f"Segments: {result.segment_count}",
        f"Nodes: {result.node_count}",
        f"Arterial flow: {result.arterial_total_flow_ml_s:.3f} mL/s",
        f"Venous flow: {result.venous_total_flow_ml_s:.3f} mL/s",
        f"Max mass-balance residual: {result.max_abs_mass_balance_residual_ml_s:.9f} mL/s",
        f"Max pressure residual: {result.max_abs_pressure_equation_residual_pa:.9f} Pa",
        f"Model YAML: {result.model_yaml_path}",
        f"Segments CSV: {result.segments_csv_path}",
        f"Nodes CSV: {result.nodes_csv_path}",
        f"Mass-balance CSV: {result.mass_balance_csv_path}",
        f"Report: {result.report_path}",
    ]
    return "\n".join(lines)
