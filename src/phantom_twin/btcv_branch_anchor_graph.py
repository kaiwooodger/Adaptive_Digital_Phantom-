from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import numpy as np

from .cta_vascular_graph import _line_length, _write_edges_csv, _write_nodes_csv


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        import yaml  # type: ignore
        from scipy import ndimage as ndi  # type: ignore
    except ImportError as exc:
        raise RuntimeError("BTCV branch-anchor graph building requires matplotlib, nibabel, scipy, and PyYAML.") from exc
    return plt, nib, yaml, ndi


@dataclass(frozen=True)
class BtcvBranchAnchorGraphResult:
    case_id: str
    output_dir: str
    graph_yaml_path: str
    nodes_csv_path: str
    edges_csv_path: str
    anchors_csv_path: str
    preview_png_path: str
    report_path: str
    node_count: int
    edge_count: int
    branch_anchor_count: int
    placeholder_edge_count: int
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    _, _, yaml, _ = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _body_stats(labels: np.ndarray, spacing_mm: np.ndarray) -> dict[str, np.ndarray]:
    ijk = np.argwhere(labels > 0)
    if len(ijk) == 0:
        raise ValueError("Anatomy labels contain no non-zero body voxels.")
    points = ijk.astype(float) * spacing_mm
    return {"centroid": points.mean(axis=0), "min": points.min(axis=0), "max": points.max(axis=0)}


def _label_centroid(labels: np.ndarray, spacing_mm: np.ndarray, label_id: int) -> np.ndarray | None:
    ijk = np.argwhere(labels == label_id)
    if len(ijk) == 0:
        return None
    return (ijk.astype(float) * spacing_mm).mean(axis=0)


def _kidney_centroids(labels: np.ndarray, spacing_mm: np.ndarray, ndi) -> tuple[np.ndarray, np.ndarray]:
    mask = labels == 7
    body = _body_stats(labels, spacing_mm)
    fallback_left = body["centroid"] + np.array([-38.0, -22.0, 92.0])
    fallback_right = body["centroid"] + np.array([38.0, -22.0, 92.0])
    if not np.any(mask):
        return fallback_left, fallback_right

    components, component_count = ndi.label(mask)
    centroids: list[tuple[int, np.ndarray]] = []
    for component_id in range(1, component_count + 1):
        ijk = np.argwhere(components == component_id)
        if len(ijk) == 0:
            continue
        centroids.append((len(ijk), (ijk.astype(float) * spacing_mm).mean(axis=0)))
    if len(centroids) >= 2:
        selected = [item[1] for item in sorted(centroids, key=lambda item: item[0], reverse=True)[:2]]
        selected.sort(key=lambda point: float(point[0]))
        return selected[0], selected[1]

    ijk = np.argwhere(mask)
    points = ijk.astype(float) * spacing_mm
    median_x = float(np.median(points[:, 0]))
    left_points = points[points[:, 0] <= median_x]
    right_points = points[points[:, 0] > median_x]
    left = left_points.mean(axis=0) if len(left_points) else fallback_left
    right = right_points.mean(axis=0) if len(right_points) else fallback_right
    if left[0] > right[0]:
        left, right = right, left
    return left, right


def _clip_to_body(point: np.ndarray, body: dict[str, np.ndarray], margin_mm: float = 6.0) -> np.ndarray:
    lo = body["min"] + margin_mm
    hi = body["max"] - margin_mm
    return np.minimum(np.maximum(point.astype(float), lo), hi)


def _edge_by_role(graph: dict[str, Any], flow_role: str) -> dict[str, Any] | None:
    for edge in graph.get("edges", []):
        if isinstance(edge, dict) and str(edge.get("flow_role", "")) == flow_role:
            return edge
    return None


def _edge_by_id_or_role(graph: dict[str, Any], edge_id: str, flow_role: str) -> dict[str, Any]:
    for edge in graph.get("edges", []):
        if isinstance(edge, dict) and str(edge.get("id", "")) == edge_id:
            return edge
    edge = _edge_by_role(graph, flow_role)
    if edge is None:
        raise ValueError(f"Could not find edge id={edge_id!r} or flow_role={flow_role!r} in source graph.")
    return edge


def _as_points(edge: dict[str, Any]) -> np.ndarray:
    points = np.asarray(edge.get("polyline_mm", []), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"Edge {edge.get('id', '<unknown>')} does not contain a usable polyline_mm.")
    return points


def _sort_by_z(points: np.ndarray) -> np.ndarray:
    order = np.argsort(points[:, 2], kind="mergesort")
    sorted_points = points[order]
    unique: list[np.ndarray] = []
    for point in sorted_points:
        if unique and abs(float(point[2] - unique[-1][2])) < 1e-6:
            unique[-1] = (unique[-1] + point) / 2.0
        else:
            unique.append(point.astype(float))
    return np.asarray(unique, dtype=float)


def _sample_at_z(points_by_z: np.ndarray, z_mm: float) -> np.ndarray:
    z_values = points_by_z[:, 2]
    z_clamped = float(np.clip(z_mm, float(z_values.min()), float(z_values.max())))
    x = float(np.interp(z_clamped, z_values, points_by_z[:, 0]))
    y = float(np.interp(z_clamped, z_values, points_by_z[:, 1]))
    return np.asarray([x, y, z_clamped], dtype=float)


def _segment_between_z(points_by_z: np.ndarray, start_z: float, end_z: float) -> list[list[float]]:
    start = _sample_at_z(points_by_z, start_z)
    end = _sample_at_z(points_by_z, end_z)
    z_min = min(float(start[2]), float(end[2]))
    z_max = max(float(start[2]), float(end[2]))
    interior = points_by_z[(points_by_z[:, 2] > z_min) & (points_by_z[:, 2] < z_max)]
    segment = [start]
    if len(interior):
        if start[2] > end[2]:
            interior = interior[::-1]
        segment.extend(interior)
    segment.append(end)
    return [[float(value) for value in point] for point in segment]


def _node(
    node_id: str,
    label: str,
    kind: str,
    role: str,
    position: np.ndarray,
    radius_mm: float,
    *,
    boundary_role: str | None = None,
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": node_id,
        "label": label,
        "kind": kind,
        "role": role,
        "position_mm": [float(value) for value in position],
        "radius_mm": float(radius_mm),
        "notes": list(notes),
    }
    if boundary_role is not None:
        node["boundary_role"] = boundary_role
    return node


def _curved_edge_points(source: np.ndarray, target: np.ndarray, body: dict[str, np.ndarray], bend_mm: float = 10.0) -> list[list[float]]:
    midpoint = (source + target) / 2.0
    # Bend toward the body centroid to avoid perfectly straight synthetic chords.
    direction = body["centroid"] - midpoint
    norm = float(np.linalg.norm(direction[:2]))
    if norm > 1e-6:
        midpoint[:2] += direction[:2] / norm * bend_mm
    return [[float(value) for value in point] for point in (source, midpoint, target)]


def _edge(
    edge_id: str,
    label: str,
    source: str,
    target: str,
    vessel_type: str,
    flow_role: str,
    nodes: dict[str, dict[str, Any]],
    points: list[list[float]],
    *,
    radius_start_mm: float | None = None,
    radius_end_mm: float | None = None,
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    start_radius = float(radius_start_mm if radius_start_mm is not None else nodes[source]["radius_mm"])
    end_radius = float(radius_end_mm if radius_end_mm is not None else nodes[target]["radius_mm"])
    return {
        "id": edge_id,
        "label": label,
        "source": source,
        "target": target,
        "vessel_type": vessel_type,
        "flow_role": flow_role,
        "radius_start_mm": start_radius,
        "radius_end_mm": end_radius,
        "length_mm": _line_length(points),
        "polyline_mm": points,
        "notes": list(notes),
    }


def _write_anchor_csv(path: Path, nodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["node_id", "kind", "role", "boundary_role", "x_mm", "y_mm", "z_mm", "radius_mm", "notes"])
        for node in nodes:
            writer.writerow(
                [
                    node.get("id", ""),
                    node.get("kind", ""),
                    node.get("role", ""),
                    node.get("boundary_role", ""),
                    *[f"{float(value):.6f}" for value in node.get("position_mm", [0.0, 0.0, 0.0])],
                    f"{float(node.get('radius_mm', 0.0)):.6f}",
                    " | ".join(str(item) for item in node.get("notes", [])),
                ]
            )


def _write_preview(path: Path, source_graph: dict[str, Any], graph: dict[str, Any], body: dict[str, np.ndarray]) -> None:
    plt, *_ = _import_dependencies()
    fig = plt.figure(figsize=(9.5, 7.5), dpi=170)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")
    all_points: list[np.ndarray] = []

    for edge in source_graph.get("edges", []):
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if len(points) >= 2:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#6b7280", linewidth=1.1, alpha=0.35)

    for edge in graph.get("edges", []):
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if len(points) < 2:
            continue
        all_points.append(points)
        vessel_type = str(edge.get("vessel_type", ""))
        notes = tuple(str(item) for item in edge.get("notes", []))
        if "btc_anchor_corrected_aorta_segment" in notes:
            color = "#dc2626"
            linewidth = 3.1
            alpha = 0.96
        elif vessel_type == "arterial":
            color = "#f97316"
            linewidth = 2.0
            alpha = 0.88
        else:
            color = "#2563eb"
            linewidth = 2.0
            alpha = 0.82
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth, alpha=alpha)

    node_points = []
    for node in graph.get("nodes", []):
        point = np.asarray(node.get("position_mm", []), dtype=float)
        if point.shape == (3,):
            node_points.append(point)
            ax.scatter(point[0], point[1], point[2], color="#111827", s=10, alpha=0.7)

    bounds = [body["min"], body["max"]]
    if all_points:
        bounds.extend([np.vstack(all_points).min(axis=0), np.vstack(all_points).max(axis=0)])
    if node_points:
        node_array = np.vstack(node_points)
        bounds.extend([node_array.min(axis=0), node_array.max(axis=0)])
    bound_array = np.vstack(bounds)
    center = (bound_array.min(axis=0) + bound_array.max(axis=0)) / 2.0
    radius = float((bound_array.max(axis=0) - bound_array.min(axis=0)).max() / 2.0) * 1.06
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_title("BTCV Branch-Anchor Vessel Graph\nred = corrected aorta, orange = arterial anchors, blue = venous anchors", pad=16)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _format_report(result: BtcvBranchAnchorGraphResult) -> str:
    return "\n".join(
        [
            "# BTCV Branch-Anchor Vascular Graph",
            "",
            f"Case ID: `{result.case_id}`",
            "",
            "## Summary",
            "",
            f"- Nodes: {result.node_count}",
            f"- Edges: {result.edge_count}",
            f"- Branch anchor nodes: {result.branch_anchor_count}",
            f"- Placeholder/template-replaceable edges: {result.placeholder_edge_count}",
            "",
            "## Outputs",
            "",
            f"- Graph YAML: `{Path(result.graph_yaml_path).name}`",
            f"- Nodes CSV: `{Path(result.nodes_csv_path).name}`",
            f"- Edges CSV: `{Path(result.edges_csv_path).name}`",
            f"- Anchor CSV: `{Path(result.anchors_csv_path).name}`",
            f"- Preview PNG: `{Path(result.preview_png_path).name}`",
            "",
            "## Interpretation",
            "",
            "- This rebuilds the BTCV graph from a coarse aorta/IVC product graph into a branch-capable target graph.",
            "- Aorta trunk anchors are sampled from the previously bone-rerouted corrected aorta.",
            "- Renal and visceral branch targets are estimated from BTCV kidney/liver/body anatomy labels.",
            "- The output is designed for branch-rich MedSeg/CTA template registration; it is not a patient-specific CTA extraction by itself.",
            "",
            "## Notes",
            *[f"- {note}" for note in result.notes],
        ]
    )


def build_btcv_branch_anchor_vascular_graph(
    coarse_graph_path: str | Path,
    anatomy_labels_path: str | Path,
    output_dir: str | Path = "outputs/digital/btcv_branch_anchor_graph",
    case_id: str = "btcv_branch_anchor_vascular_graph",
    report_path: str | Path | None = "outputs/reports/btcv_branch_anchor_vascular_graph.md",
) -> BtcvBranchAnchorGraphResult:
    _, nib, yaml, ndi = _import_dependencies()
    source_graph = _load_yaml(coarse_graph_path)
    anatomy_image = nib.load(str(anatomy_labels_path))
    labels = np.asanyarray(anatomy_image.dataobj).astype(np.int16)
    spacing_mm = np.asarray(anatomy_image.header.get_zooms()[:3], dtype=float)
    body = _body_stats(labels, spacing_mm)
    liver = _label_centroid(labels, spacing_mm, 6)
    left_kidney, right_kidney = _kidney_centroids(labels, spacing_mm, ndi)

    aorta_edge = _edge_by_id_or_role(source_graph, "coarse_aorta_trunk", "aorta_trunk")
    aorta_points = _sort_by_z(_as_points(aorta_edge))
    aorta_z_min = float(aorta_points[:, 2].min())
    aorta_z_max = float(aorta_points[:, 2].max())
    aorta_span = max(aorta_z_max - aorta_z_min, 1.0)

    kidney_z = float(np.mean([left_kidney[2], right_kidney[2]]))
    liver_z = float(liver[2]) if liver is not None else kidney_z + 24.0
    renal_z = float(np.clip(kidney_z, aorta_z_min + 0.18 * aorta_span, aorta_z_max - 0.30 * aorta_span))
    visceral_z = float(np.clip(max(liver_z, renal_z + 12.0), renal_z + 8.0, aorta_z_max - 0.18 * aorta_span))
    descending_z = float(np.clip(visceral_z + 0.28 * (aorta_z_max - visceral_z), visceral_z + 6.0, aorta_z_max - 6.0))
    distal_z = float(np.clip(aorta_z_min + 0.14 * aorta_span, aorta_z_min + 6.0, renal_z - 8.0))
    bifurcation_z = aorta_z_min

    aorta_inlet = _sample_at_z(aorta_points, aorta_z_max)
    descending = _sample_at_z(aorta_points, descending_z)
    visceral = _sample_at_z(aorta_points, visceral_z)
    renal = _sample_at_z(aorta_points, renal_z)
    distal = _sample_at_z(aorta_points, distal_z)
    bifurcation = _sample_at_z(aorta_points, bifurcation_z)

    body_extent = body["max"] - body["min"]
    lateral_sep = float(np.clip(body_extent[0] * 0.17, 32.0, 62.0))
    anterior_shift = float(np.clip(body_extent[1] * 0.06, 10.0, 20.0))
    iliac_drop = float(np.clip(aorta_span * 0.06, 8.0, 16.0))
    left_iliac = _clip_to_body(bifurcation + np.array([-lateral_sep, anterior_shift, -iliac_drop]), body)
    right_iliac = _clip_to_body(bifurcation + np.array([lateral_sep, anterior_shift, -iliac_drop]), body)

    liver_point = _clip_to_body(liver if liver is not None else body["centroid"] + np.array([-55.0, 35.0, 110.0]), body)
    splenic_point = np.array(
        [
            body["centroid"][0] + abs(float(liver_point[0] - body["centroid"][0])),
            liver_point[1] - 8.0,
            liver_point[2] - 10.0,
        ],
        dtype=float,
    )
    splenic_point = _clip_to_body(splenic_point, body)

    left_renal = _clip_to_body(left_kidney, body)
    right_renal = _clip_to_body(right_kidney, body)

    ivc_edge = _edge_by_role(source_graph, "ivc_return")
    if ivc_edge is not None:
        ivc_points = _as_points(ivc_edge)
        reference_aorta = _sample_at_z(aorta_points, aorta_z_min)
        ivc_offset_xy = ivc_points[:, :2].mean(axis=0) - reference_aorta[:2]
    else:
        ivc_offset_xy = np.array([-22.0, 56.0], dtype=float)
    ivc_offset_xy = np.asarray(
        [
            float(np.clip(ivc_offset_xy[0], -42.0, 42.0)),
            float(np.clip(ivc_offset_xy[1], 24.0, 72.0)),
        ],
        dtype=float,
    )

    def ivc_at(z_mm: float) -> np.ndarray:
        point = _sample_at_z(aorta_points, z_mm).copy()
        point[:2] += ivc_offset_xy
        return _clip_to_body(point, body)

    ivc_lower = ivc_at(aorta_z_min + 0.06 * aorta_span)
    ivc_bifurcation = ivc_at(aorta_z_min + 0.12 * aorta_span)
    ivc_renal = ivc_at(renal_z)
    ivc_hepatic = ivc_at(visceral_z + 0.10 * aorta_span)
    ivc_outlet = ivc_at(aorta_z_max - 0.03 * aorta_span)
    left_renal_vein = _clip_to_body(left_renal + np.array([0.0, 10.0, 0.0]), body)
    right_renal_vein = _clip_to_body(right_renal + np.array([0.0, 10.0, 0.0]), body)
    hepatic_vein = _clip_to_body(liver_point * 0.60 + ivc_hepatic * 0.40, body)
    splenic_vein = _clip_to_body(splenic_point + np.array([0.0, 8.0, -4.0]), body)

    nodes_list = [
        _node("aorta_inlet", "Aorta inlet", "arterial", "inlet", aorta_inlet, 5.5, boundary_role="arterial_inlet", notes=("sampled_from_bone_rerouted_btcv_aorta",)),
        _node("descending_aorta_mid", "Descending aorta control", "arterial", "junction", descending, 5.3, notes=("sampled_from_bone_rerouted_btcv_aorta",)),
        _node("visceral_branch_origin", "Celiac/visceral arterial origin", "arterial", "junction", visceral, 4.8, notes=("estimated_from_liver_and_aorta_geometry",)),
        _node("renal_branch_origin", "Renal arterial origin", "arterial", "junction", renal, 4.5, notes=("estimated_from_kidney_and_aorta_geometry",)),
        _node("aorta_distal_anchor", "Distal abdominal aorta anchor", "arterial", "junction", distal, 4.2, notes=("sampled_from_bone_rerouted_btcv_aorta",)),
        _node("aortic_bifurcation", "Aortic bifurcation anchor", "arterial", "junction", bifurcation, 4.0, notes=("inferior_anchor_from_corrected_aorta_extent",)),
        _node("left_common_iliac_outlet", "Left common iliac outlet", "arterial", "outlet", left_iliac, 2.8, boundary_role="arterial_outlet", notes=("estimated_branch_endpoint_from_body_bbox",)),
        _node("right_common_iliac_outlet", "Right common iliac outlet", "arterial", "outlet", right_iliac, 2.8, boundary_role="arterial_outlet", notes=("estimated_branch_endpoint_from_body_bbox",)),
        _node("left_renal_outlet", "Left renal artery outlet", "arterial", "outlet", left_renal, 2.2, boundary_role="arterial_outlet", notes=("estimated_from_left_kidney_centroid",)),
        _node("right_renal_outlet", "Right renal artery outlet", "arterial", "outlet", right_renal, 2.2, boundary_role="arterial_outlet", notes=("estimated_from_right_kidney_centroid",)),
        _node("hepatic_placeholder_outlet", "Hepatic arterial outlet", "arterial", "outlet", liver_point, 2.1, boundary_role="arterial_outlet", notes=("estimated_from_liver_centroid",)),
        _node("splenic_placeholder_outlet", "Splenic arterial outlet surrogate", "arterial", "outlet", splenic_point, 2.0, boundary_role="arterial_outlet", notes=("spleen_label_unavailable_mirrored_from_liver_across_body_center",)),
        _node("ivc_lower_return_inlet", "Lower venous return inlet", "venous", "inlet", ivc_lower, 4.0, boundary_role="venous_inlet", notes=("estimated_ivc_offset_from_btcv_coarse_graph",)),
        _node("ivc_bifurcation_return", "Iliocaval bifurcation return", "venous", "junction", ivc_bifurcation, 4.0, notes=("estimated_ivc_offset_from_btcv_coarse_graph",)),
        _node("ivc_renal_junction", "IVC renal junction", "venous", "junction", ivc_renal, 4.2, notes=("estimated_ivc_offset_from_btcv_coarse_graph",)),
        _node("ivc_hepatic_junction", "IVC hepatic junction", "venous", "junction", ivc_hepatic, 4.4, notes=("estimated_ivc_offset_from_btcv_coarse_graph",)),
        _node("ivc_outlet", "IVC outlet", "venous", "outlet", ivc_outlet, 4.6, boundary_role="venous_outlet", notes=("estimated_ivc_offset_from_btcv_coarse_graph",)),
        _node("left_renal_vein_inlet", "Left renal vein inlet", "venous", "inlet", left_renal_vein, 3.0, boundary_role="venous_inlet", notes=("estimated_from_left_kidney_centroid",)),
        _node("right_renal_vein_inlet", "Right renal vein inlet", "venous", "inlet", right_renal_vein, 3.0, boundary_role="venous_inlet", notes=("estimated_from_right_kidney_centroid",)),
        _node("hepatic_venous_placeholder_inlet", "Hepatic venous inlet", "venous", "inlet", hepatic_vein, 3.2, boundary_role="venous_inlet", notes=("estimated_from_liver_to_ivc_geometry",)),
        _node("splenic_venous_placeholder_inlet", "Splenic/portal venous inlet surrogate", "venous", "inlet", splenic_vein, 3.0, boundary_role="venous_inlet", notes=("spleen_label_unavailable_mirrored_from_liver_across_body_center",)),
    ]
    nodes = {str(node["id"]): node for node in nodes_list}

    def aorta_segment(start: np.ndarray, end: np.ndarray) -> list[list[float]]:
        return _segment_between_z(aorta_points, float(start[2]), float(end[2]))

    ivc_points_by_z = _sort_by_z(np.asarray([ivc_at(z) for z in np.linspace(aorta_z_min, aorta_z_max, 30)], dtype=float))
    edges = [
        _edge("aorta_inlet_to_descending", "Aorta trunk: inlet to descending control", "aorta_inlet", "descending_aorta_mid", "arterial", "aorta_trunk", nodes, aorta_segment(aorta_inlet, descending), notes=("btc_anchor_corrected_aorta_segment",)),
        _edge("descending_to_visceral_origin", "Aorta trunk: descending to visceral origin", "descending_aorta_mid", "visceral_branch_origin", "arterial", "aorta_trunk", nodes, aorta_segment(descending, visceral), notes=("btc_anchor_corrected_aorta_segment",)),
        _edge("visceral_to_renal_origin", "Aorta trunk: visceral origin to renal origin", "visceral_branch_origin", "renal_branch_origin", "arterial", "aorta_trunk", nodes, aorta_segment(visceral, renal), notes=("btc_anchor_corrected_aorta_segment",)),
        _edge("renal_origin_to_distal_aorta", "Aorta trunk: renal origin to distal anchor", "renal_branch_origin", "aorta_distal_anchor", "arterial", "aorta_trunk", nodes, aorta_segment(renal, distal), notes=("btc_anchor_corrected_aorta_segment",)),
        _edge("distal_aorta_to_bifurcation", "Aorta trunk: distal anchor to bifurcation", "aorta_distal_anchor", "aortic_bifurcation", "arterial", "aorta_trunk", nodes, aorta_segment(distal, bifurcation), notes=("btc_anchor_corrected_aorta_segment",)),
        _edge("bifurcation_to_left_common_iliac", "Left common iliac artery placeholder", "aortic_bifurcation", "left_common_iliac_outlet", "arterial", "iliac_branch", nodes, _curved_edge_points(bifurcation, left_iliac, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("bifurcation_to_right_common_iliac", "Right common iliac artery placeholder", "aortic_bifurcation", "right_common_iliac_outlet", "arterial", "iliac_branch", nodes, _curved_edge_points(bifurcation, right_iliac, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("renal_origin_to_left_renal", "Left renal artery placeholder", "renal_branch_origin", "left_renal_outlet", "arterial", "renal_branch", nodes, _curved_edge_points(renal, left_renal, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("renal_origin_to_right_renal", "Right renal artery placeholder", "renal_branch_origin", "right_renal_outlet", "arterial", "renal_branch", nodes, _curved_edge_points(renal, right_renal, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("visceral_origin_to_hepatic_placeholder", "Celiac/common hepatic axis placeholder", "visceral_branch_origin", "hepatic_placeholder_outlet", "arterial", "hepatic_placeholder_branch", nodes, _curved_edge_points(visceral, liver_point, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("visceral_origin_to_splenic_placeholder", "Celiac/splenic axis placeholder", "visceral_branch_origin", "splenic_placeholder_outlet", "arterial", "splenic_placeholder_branch", nodes, _curved_edge_points(visceral, splenic_point, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("ivc_lower_to_bifurcation_return", "Iliocaval return placeholder", "ivc_lower_return_inlet", "ivc_bifurcation_return", "venous", "venous_return", nodes, _segment_between_z(ivc_points_by_z, float(ivc_lower[2]), float(ivc_bifurcation[2])), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("ivc_bifurcation_to_renal_junction", "IVC lower segment placeholder", "ivc_bifurcation_return", "ivc_renal_junction", "venous", "venous_return", nodes, _segment_between_z(ivc_points_by_z, float(ivc_bifurcation[2]), float(ivc_renal[2])), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("ivc_renal_to_hepatic_junction", "IVC mid segment placeholder", "ivc_renal_junction", "ivc_hepatic_junction", "venous", "venous_return", nodes, _segment_between_z(ivc_points_by_z, float(ivc_renal[2]), float(ivc_hepatic[2])), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("ivc_hepatic_to_outlet", "IVC upper segment placeholder", "ivc_hepatic_junction", "ivc_outlet", "venous", "venous_return", nodes, _segment_between_z(ivc_points_by_z, float(ivc_hepatic[2]), float(ivc_outlet[2])), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("left_renal_vein_to_ivc", "Left renal vein placeholder", "left_renal_vein_inlet", "ivc_renal_junction", "venous", "renal_venous_return", nodes, _curved_edge_points(left_renal_vein, ivc_renal, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("right_renal_vein_to_ivc", "Right renal vein placeholder", "right_renal_vein_inlet", "ivc_renal_junction", "venous", "renal_venous_return", nodes, _curved_edge_points(right_renal_vein, ivc_renal, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("hepatic_venous_placeholder_to_ivc", "Hepatic venous return placeholder", "hepatic_venous_placeholder_inlet", "ivc_hepatic_junction", "venous", "hepatic_venous_return", nodes, _curved_edge_points(hepatic_vein, ivc_hepatic, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
        _edge("splenic_venous_placeholder_to_ivc", "Splenic/portal venous return placeholder", "splenic_venous_placeholder_inlet", "ivc_hepatic_junction", "venous", "splenic_portal_venous_return", nodes, _curved_edge_points(splenic_vein, ivc_hepatic, body), notes=("branch_anchor_placeholder_replaceable_by_medseg",)),
    ]

    placeholder_edges = [
        edge
        for edge in edges
        if "branch_anchor_placeholder_replaceable_by_medseg" in tuple(str(item) for item in edge.get("notes", []))
    ]
    graph = {
        "case_id": case_id,
        "coordinate_units": "mm",
        "scaffold_type": "btcv_branch_anchor_graph_from_corrected_aorta_and_organ_centroids",
        "source_graph": str(coarse_graph_path),
        "source_anatomy_labels": str(anatomy_labels_path),
        "include_venous_return": True,
        "graph_metadata": {
            "node_count": len(nodes_list),
            "edge_count": len(edges),
            "branch_anchor_node_count": len(nodes_list) - 7,
            "placeholder_edge_count": len(placeholder_edges),
            "aorta_source": "bone_rerouted_btcv_coarse_aorta_trunk",
            "aorta_z_range_mm": [aorta_z_min, aorta_z_max],
            "renal_origin_z_mm": renal_z,
            "visceral_origin_z_mm": visceral_z,
            "ivc_offset_xy_mm": [float(value) for value in ivc_offset_xy],
            "boundary_ids": {
                "arterial_inlet_ids": ["aorta_inlet"],
                "arterial_outlet_ids": [
                    "left_common_iliac_outlet",
                    "right_common_iliac_outlet",
                    "left_renal_outlet",
                    "right_renal_outlet",
                    "hepatic_placeholder_outlet",
                    "splenic_placeholder_outlet",
                ],
                "venous_inlet_ids": [
                    "ivc_lower_return_inlet",
                    "left_renal_vein_inlet",
                    "right_renal_vein_inlet",
                    "hepatic_venous_placeholder_inlet",
                    "splenic_venous_placeholder_inlet",
                ],
                "venous_outlet_ids": ["ivc_outlet"],
            },
            "registration_intent": "target_graph_for_medseg_branch_rich_labeled_vessel_registration",
            "clinical_status": "research_demo_anchor_graph_not_patient_specific_cta_extraction",
        },
        "provenance_notes": [
            "corrected_btcv_aorta_split_into_branch_capable_trunk_segments",
            "renal_branch_anchors_estimated_from_kidney_centroids",
            "visceral_branch_anchors_estimated_from_liver_centroid",
            "splenic_anchor_is_surrogate_because_current_material_labels_do_not_include_spleen",
            "venous_return_anchors_estimated_from_ivc_offset_relative_to_corrected_aorta",
            "designed_for_registered_medseg_or_cta_ctv_branch_template_replacement",
        ],
        "nodes": nodes_list,
        "edges": edges,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_yaml = output / f"{case_id}_btcv_branch_anchor_vascular_graph_v001.yaml"
    nodes_csv = output / f"{case_id}_btcv_branch_anchor_vascular_graph_nodes_v001.csv"
    edges_csv = output / f"{case_id}_btcv_branch_anchor_vascular_graph_edges_v001.csv"
    anchors_csv = output / f"{case_id}_btcv_branch_anchor_nodes_v001.csv"
    preview = output / f"{case_id}_btcv_branch_anchor_vascular_graph_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_btcv_branch_anchor_vascular_graph_report_v001.md"

    graph_yaml.write_text(yaml.safe_dump(graph, sort_keys=False))
    _write_nodes_csv(nodes_csv, nodes_list)
    _write_edges_csv(edges_csv, edges)
    _write_anchor_csv(anchors_csv, nodes_list)
    _write_preview(preview, source_graph, graph, body)

    result = BtcvBranchAnchorGraphResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_yaml),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        anchors_csv_path=str(anchors_csv),
        preview_png_path=str(preview),
        report_path=str(report),
        node_count=len(nodes_list),
        edge_count=len(edges),
        branch_anchor_count=len(nodes_list) - 7,
        placeholder_edge_count=len(placeholder_edges),
        notes=(
            "btcv_coarse_graph_rebuilt_with_branch_anchor_nodes",
            "aorta_segments_sampled_from_bone_rerouted_aorta",
            "branch_targets_derived_from_btcv_liver_kidney_body_geometry",
            "splenic_target_is_a_surrogate_without_spleen_label",
            "ready_for_registered_labeled_vessel_branch_replacement",
        ),
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result) + "\n")
    return result


def format_btcv_branch_anchor_graph_result(result: BtcvBranchAnchorGraphResult) -> str:
    return "\n".join(
        [
            "BTCV branch-anchor vascular graph built",
            f"Case ID: {result.case_id}",
            f"Nodes/edges: {result.node_count}/{result.edge_count}",
            f"Branch anchors: {result.branch_anchor_count}",
            f"Template-replaceable placeholder edges: {result.placeholder_edge_count}",
            f"Graph YAML: {result.graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
