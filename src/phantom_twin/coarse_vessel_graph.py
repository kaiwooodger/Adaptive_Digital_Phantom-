from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import csv

import numpy as np
import yaml


@dataclass(frozen=True)
class CoarseVesselGraphResult:
    case_id: str
    output_dir: str
    source_graph_path: str
    graph_yaml_path: str
    nodes_csv_path: str
    edges_csv_path: str
    preview_png_path: str
    report_path: str
    retained_node_count: int
    retained_edge_count: int
    dropped_node_count: int
    dropped_edge_count: int
    expected_lumen_components: int
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _line_length(points: list[list[float]]) -> float:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or len(array) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def _cap_node_radius(node: dict[str, Any], radius_mm: float) -> None:
    node["radius_mm"] = float(radius_mm)


def _cap_edge_radius(edge: dict[str, Any], radius_mm: float) -> None:
    edge["radius_start_mm"] = float(radius_mm)
    edge["radius_end_mm"] = float(radius_mm)
    edge.pop("radius_profile", None)


def _append_note(item: dict[str, Any], note: str) -> None:
    notes = list(item.get("notes", []))
    if note not in notes:
        notes.append(note)
    item["notes"] = notes


def _edge_points(edge: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> list[list[float]]:
    points = edge.get("polyline_mm", [])
    if isinstance(points, list) and len(points) >= 2:
        return [[float(value) for value in point[:3]] for point in points]
    source = nodes_by_id.get(str(edge.get("source")))
    target = nodes_by_id.get(str(edge.get("target")))
    if source is None or target is None:
        return []
    return [
        [float(value) for value in source.get("position_mm", [0.0, 0.0, 0.0])[:3]],
        [float(value) for value in target.get("position_mm", [0.0, 0.0, 0.0])[:3]],
    ]


def _concat_polylines(existing: list[list[float]], addition: list[list[float]]) -> list[list[float]]:
    if not addition:
        return existing
    if not existing:
        return [list(point) for point in addition]
    if np.linalg.norm(np.asarray(existing[-1], dtype=float) - np.asarray(addition[0], dtype=float)) < 1e-6:
        return [*existing, *[list(point) for point in addition[1:]]]
    return [*existing, *[list(point) for point in addition]]


def _follow_role_chain(
    edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    *,
    start_node_id: str,
    flow_role: str,
) -> tuple[list[list[float]], str, list[str]]:
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        if str(edge.get("flow_role", "")) != flow_role:
            continue
        outgoing.setdefault(str(edge.get("source", "")), []).append(edge)

    points: list[list[float]] = []
    edge_ids: list[str] = []
    current = start_node_id
    visited: set[str] = set()
    while current in outgoing:
        candidates = [edge for edge in outgoing[current] if str(edge.get("id", "")) not in visited]
        if not candidates:
            break
        candidates.sort(key=lambda item: float(item.get("length_mm", _line_length(_edge_points(item, nodes_by_id)))), reverse=True)
        edge = candidates[0]
        edge_id = str(edge.get("id", ""))
        visited.add(edge_id)
        edge_ids.append(edge_id)
        points = _concat_polylines(points, _edge_points(edge, nodes_by_id))
        next_node = str(edge.get("target", ""))
        if next_node == current:
            break
        current = next_node
    return points, current, edge_ids


def _write_nodes_csv(path: Path, nodes: list[dict[str, Any]]) -> None:
    fields = ["id", "label", "kind", "role", "boundary_role", "radius_mm", "x_mm", "y_mm", "z_mm", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for node in nodes:
            position = node.get("position_mm", [None, None, None])
            writer.writerow(
                {
                    "id": node.get("id", ""),
                    "label": node.get("label", ""),
                    "kind": node.get("kind", ""),
                    "role": node.get("role", ""),
                    "boundary_role": node.get("boundary_role", ""),
                    "radius_mm": node.get("radius_mm", ""),
                    "x_mm": position[0],
                    "y_mm": position[1],
                    "z_mm": position[2],
                    "notes": ";".join(str(note) for note in node.get("notes", [])),
                }
            )


def _write_edges_csv(path: Path, edges: list[dict[str, Any]], source_edge_ids: dict[str, str]) -> None:
    fields = [
        "id",
        "source_edge_id",
        "source",
        "target",
        "vessel_type",
        "flow_role",
        "radius_start_mm",
        "radius_end_mm",
        "length_mm",
        "point_count",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "id": edge.get("id", ""),
                    "source_edge_id": source_edge_ids.get(str(edge.get("id", "")), ""),
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "vessel_type": edge.get("vessel_type", ""),
                    "flow_role": edge.get("flow_role", ""),
                    "radius_start_mm": edge.get("radius_start_mm", ""),
                    "radius_end_mm": edge.get("radius_end_mm", ""),
                    "length_mm": edge.get("length_mm", ""),
                    "point_count": len(edge.get("polyline_mm", [])),
                    "notes": ";".join(str(note) for note in edge.get("notes", [])),
                }
            )


def _render_preview(path: Path, source_graph: dict[str, Any], coarse_graph: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Coarse vessel graph preview generation requires matplotlib.") from exc

    fig = plt.figure(figsize=(10, 8), dpi=170)
    fig.patch.set_facecolor("#f7f1e3")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f7f1e3")
    for graph, color, alpha, width, label in (
        (source_graph, "#94a3b8", 0.25, 0.7, "source graph"),
        (coarse_graph, "#dc2626", 0.95, 2.0, "BTCV coarse graph"),
    ):
        first = True
        for edge in graph.get("edges", []):
            points = np.asarray(edge.get("polyline_mm", []), dtype=float)
            if points.ndim != 2 or len(points) < 2:
                continue
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=alpha, linewidth=width, label=label if first else None)
            first = False
    ax.set_title("BTCV Coarse Major-Vessel Graph")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.view_init(elev=18, azim=-58)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, result: CoarseVesselGraphResult) -> None:
    lines = [
        "# BTCV Coarse Major-Vessel Graph",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Retained nodes/edges: {result.retained_node_count} / {result.retained_edge_count}",
        f"- Dropped nodes/edges: {result.dropped_node_count} / {result.dropped_edge_count}",
        f"- Expected lumen components for QA: {result.expected_lumen_components}",
        "",
        "## Outputs",
        "",
        f"- Graph YAML: `{result.graph_yaml_path}`",
        f"- Nodes CSV: `{result.nodes_csv_path}`",
        f"- Edges CSV: `{result.edges_csv_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        "",
        "## Interpretation",
        "",
        "- This graph is a coarse BTCV-compatible major-vessel domain, not a branch-rich vascular network.",
        "- Renal, hepatic, splenic, iliac, and placeholder venous branches are removed because the BTCV coarse labels do not support those branches.",
        "- The retained aorta and IVC segments are radius-capped so the digital flow domain stays within the available anatomy labels.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_btcv_coarse_vessel_graph(
    *,
    graph_yaml_path: str | Path,
    output_dir: str | Path = "outputs/digital/btcv_coarse_vessel_graph",
    case_id: str = "btcv_coarse_major_vessels",
    arterial_radius_mm: float = 2.5,
    venous_radius_mm: float = 3.0,
    report_path: str | Path | None = "outputs/reports/btcv_coarse_vessel_graph.md",
) -> CoarseVesselGraphResult:
    source_path = Path(graph_yaml_path)
    source_graph = _load_yaml(source_path)
    nodes_by_id = {str(node["id"]): copy.deepcopy(node) for node in source_graph.get("nodes", [])}
    edges_by_id = {str(edge["id"]): copy.deepcopy(edge) for edge in source_graph.get("edges", [])}
    source_edges = [copy.deepcopy(edge) for edge in source_graph.get("edges", []) if isinstance(edge, dict)]
    aorta_chain_points, aorta_chain_target_id, aorta_chain_edge_ids = _follow_role_chain(
        source_edges,
        nodes_by_id,
        start_node_id="aorta_inlet",
        flow_role="aorta_trunk",
    )
    first_aorta_points = _edge_points(edges_by_id.get("aorta_inlet_to_descending", {}), nodes_by_id)
    use_full_aorta_chain = (
        bool(aorta_chain_edge_ids)
        and _line_length(aorta_chain_points) > max(_line_length(first_aorta_points) + 1e-6, _line_length(first_aorta_points) * 1.25)
    )

    required_edges = {
        "aorta_inlet_to_descending": ("coarse_aorta_trunk", arterial_radius_mm),
        "ivc_lower_to_bifurcation_return": ("coarse_ivc_return", venous_radius_mm),
    }
    missing = [edge_id for edge_id in required_edges if edge_id not in edges_by_id]
    if missing:
        raise ValueError(f"Source graph is missing required BTCV coarse edge(s): {', '.join(missing)}")

    retained_node_ids = ("aorta_inlet", "descending_aorta_mid", "ivc_lower_return_inlet", "ivc_bifurcation_return")
    missing_nodes = [node_id for node_id in retained_node_ids if node_id not in nodes_by_id]
    if missing_nodes:
        raise ValueError(f"Source graph is missing required BTCV coarse node(s): {', '.join(missing_nodes)}")

    nodes: list[dict[str, Any]] = []
    for node_id in retained_node_ids:
        node = nodes_by_id[node_id]
        node.pop("boundary_role", None)
        if node_id == "aorta_inlet":
            node["role"] = "inlet"
            node["boundary_role"] = "arterial_inlet"
            _cap_node_radius(node, arterial_radius_mm)
        elif node_id == "descending_aorta_mid":
            node["role"] = "outlet"
            node["label"] = "Coarse aorta outlet"
            node["boundary_role"] = "arterial_outlet"
            if use_full_aorta_chain and aorta_chain_points:
                node["position_mm"] = [float(value) for value in aorta_chain_points[-1]]
                source_target = nodes_by_id.get(aorta_chain_target_id)
                if source_target is not None and source_target.get("radius_mm") is not None:
                    node["source_full_trunk_node_id"] = aorta_chain_target_id
                _append_note(node, "position_retargeted_to_full_aorta_trunk_outlet_for_coarse_mode")
            _cap_node_radius(node, arterial_radius_mm)
        elif node_id == "ivc_lower_return_inlet":
            node["role"] = "inlet"
            node["boundary_role"] = "venous_inlet"
            _cap_node_radius(node, venous_radius_mm)
        elif node_id == "ivc_bifurcation_return":
            node["role"] = "outlet"
            node["label"] = "Coarse IVC outlet"
            node["boundary_role"] = "venous_outlet"
            _cap_node_radius(node, venous_radius_mm)
        _append_note(node, "btcv_coarse_major_vessel_pruned_graph")
        nodes.append(node)

    source_edge_ids: dict[str, str] = {}
    edges: list[dict[str, Any]] = []
    for source_edge_id, (coarse_edge_id, radius_mm) in required_edges.items():
        edge = edges_by_id[source_edge_id]
        if coarse_edge_id == "coarse_aorta_trunk" and use_full_aorta_chain:
            edge["source"] = "aorta_inlet"
            edge["target"] = "descending_aorta_mid"
            edge["polyline_mm"] = [[float(value) for value in point] for point in aorta_chain_points]
            _append_note(edge, "full_source_aorta_trunk_chain_preserved_for_coarse_mode")
        edge["id"] = coarse_edge_id
        edge["label"] = "BTCV coarse major vessel"
        edge["flow_role"] = "aorta_trunk" if str(edge.get("vessel_type")) == "arterial" else "venous_return_trunk"
        _cap_edge_radius(edge, radius_mm)
        edge["length_mm"] = _line_length(edge.get("polyline_mm", []))
        _append_note(edge, "btcv_coarse_major_vessel_pruned_graph")
        _append_note(edge, "unsupported_btcv_branch_placeholders_removed")
        _append_note(edge, "radius_capped_for_coarse_digital_validation")
        edges.append(edge)
        source_edge_ids[coarse_edge_id] = "+".join(aorta_chain_edge_ids) if coarse_edge_id == "coarse_aorta_trunk" and use_full_aorta_chain else source_edge_id

    coarse_graph = copy.deepcopy(source_graph)
    coarse_graph["case_id"] = case_id
    coarse_graph["scaffold_type"] = "btcv_coarse_major_vessels_only"
    coarse_graph["nodes"] = nodes
    coarse_graph["edges"] = edges
    metadata = dict(coarse_graph.get("graph_metadata", {}))
    metadata.update(
        {
            "source_graph": str(source_path),
            "coarse_vessel_mode": "btcv_major_vessels_only",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "retained_edge_ids": [edge["id"] for edge in edges],
            "source_edge_ids": source_edge_ids,
            "coarse_aorta_source_chain_edge_ids": aorta_chain_edge_ids if use_full_aorta_chain else [],
            "coarse_aorta_source_chain_length_mm": _line_length(aorta_chain_points) if use_full_aorta_chain else 0.0,
            "coarse_aorta_source_chain_target_node_id": aorta_chain_target_id if use_full_aorta_chain else "",
            "dropped_edge_count": max(0, len(source_graph.get("edges", [])) - len(edges)),
            "arterial_radius_mm": float(arterial_radius_mm),
            "venous_radius_mm": float(venous_radius_mm),
            "expected_lumen_components": 2,
            "product_scope": "coarse_major_vessels_only_not_branch_rich_vascular_network",
        }
    )
    coarse_graph["graph_metadata"] = metadata
    coarse_graph["provenance_notes"] = list(coarse_graph.get("provenance_notes", [])) + [
        "btcv_coarse_major_vessel_pruning_applied",
        "unsupported_branch_placeholders_removed_for_product_QA_mode",
    ]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_yaml = output / f"{case_id}_btcv_coarse_vascular_graph_v001.yaml"
    nodes_csv = output / f"{case_id}_btcv_coarse_vascular_graph_nodes_v001.csv"
    edges_csv = output / f"{case_id}_btcv_coarse_vascular_graph_edges_v001.csv"
    preview_png = output / f"{case_id}_btcv_coarse_vascular_graph_preview_v001.png"
    report = Path(report_path) if report_path else output / f"{case_id}_btcv_coarse_vascular_graph_report_v001.md"
    graph_yaml.write_text(yaml.safe_dump(coarse_graph, sort_keys=False))
    _write_nodes_csv(nodes_csv, nodes)
    _write_edges_csv(edges_csv, edges, source_edge_ids)
    _render_preview(preview_png, source_graph, coarse_graph)

    result = CoarseVesselGraphResult(
        case_id=case_id,
        output_dir=str(output),
        source_graph_path=str(source_path),
        graph_yaml_path=str(graph_yaml),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        preview_png_path=str(preview_png),
        report_path=str(report),
        retained_node_count=len(nodes),
        retained_edge_count=len(edges),
        dropped_node_count=max(0, len(source_graph.get("nodes", [])) - len(nodes)),
        dropped_edge_count=max(0, len(source_graph.get("edges", [])) - len(edges)),
        expected_lumen_components=2,
        notes=(
        "btcv_case_supports_coarse_major_vessels_aorta_ivc_portal_not_branch_rich_network",
        (
            "coarse_aorta_trunk_preserves_full_available_source_aorta_chain"
            if use_full_aorta_chain
            else "coarse_aorta_trunk_uses_single_source_aorta_segment"
        ),
        "branch_placeholders_removed_instead_of_relabelled_as_patient_specific_vessels",
            "coarse_graph_is_suitable_for_research_product_demonstrator_QA_not_clinical_vessel_anatomy_claims",
        ),
    )
    _write_report(report, result)
    return result


def format_coarse_vessel_graph_result(result: CoarseVesselGraphResult) -> str:
    return "\n".join(
        [
            "BTCV coarse major-vessel graph built",
            f"Case ID: {result.case_id}",
            f"Retained nodes/edges: {result.retained_node_count}/{result.retained_edge_count}",
            f"Dropped nodes/edges: {result.dropped_node_count}/{result.dropped_edge_count}",
            f"Expected lumen components: {result.expected_lumen_components}",
            f"Graph YAML: {result.graph_yaml_path}",
            f"Report: {result.report_path}",
        ]
    )
