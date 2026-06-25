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
    except ImportError as exc:
        raise RuntimeError("BTCV IVC tightening requires matplotlib, nibabel, and PyYAML.") from exc
    return plt, nib, yaml


@dataclass(frozen=True)
class BtcvIvcTighteningResult:
    case_id: str
    output_dir: str
    source_graph_path: str
    corrected_graph_yaml_path: str
    edge_metrics_csv_path: str
    nodes_csv_path: str
    edges_csv_path: str
    preview_png_path: str
    report_path: str
    corrected_edge_count: int
    mean_length_before_mm: float
    mean_length_after_mm: float
    mean_outside_before_fraction: float
    mean_outside_after_fraction: float
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    _, _, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _node_lookup(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in graph.get("nodes", []) if isinstance(node, dict) and "id" in node}


def _body_centroid(labels: np.ndarray, spacing_mm: np.ndarray) -> np.ndarray:
    ijk = np.argwhere(labels > 0)
    if len(ijk) == 0:
        raise ValueError("Anatomy labels contain no non-zero body voxels.")
    return (ijk.astype(float) * spacing_mm).mean(axis=0)


def _inside_body(point_mm: np.ndarray, body_mask: np.ndarray, spacing_mm: np.ndarray) -> bool:
    index = np.rint(point_mm / spacing_mm).astype(int)
    if np.any(index < 0) or np.any(index >= np.asarray(body_mask.shape)):
        return False
    return bool(body_mask[tuple(index)])


def _pull_inside_body(point_mm: np.ndarray, body_mask: np.ndarray, spacing_mm: np.ndarray, body_centroid_mm: np.ndarray) -> np.ndarray:
    point = point_mm.astype(float).copy()
    if _inside_body(point, body_mask, spacing_mm):
        return point
    target = body_centroid_mm.astype(float)
    # Keep the same axial station as much as possible; most failures are lateral/anterior bows.
    target[2] = point[2]
    for fraction in np.linspace(0.08, 1.0, 24):
        candidate = point + (target - point) * float(fraction)
        if _inside_body(candidate, body_mask, spacing_mm):
            return candidate
    # Last-resort fallback to the body centroid at that axial level.
    fallback = body_centroid_mm.astype(float).copy()
    fallback[2] = point[2]
    return fallback


def _resample_curve(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < 2:
        return points.astype(float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-6:
        return np.repeat(points[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, total, max(count, 2))
    output = np.empty((len(targets), 3), dtype=float)
    for index, distance in enumerate(targets):
        segment = min(max(int(np.searchsorted(cumulative, distance, side="right")) - 1, 0), len(lengths) - 1)
        denom = max(float(lengths[segment]), 1e-6)
        t = float((distance - cumulative[segment]) / denom)
        output[index] = points[segment] + (points[segment + 1] - points[segment]) * t
    output[0] = points[0]
    output[-1] = points[-1]
    return output


def _tight_segment(
    start: np.ndarray,
    end: np.ndarray,
    *,
    body_mask: np.ndarray,
    spacing_mm: np.ndarray,
    body_centroid_mm: np.ndarray,
    point_count: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(point_count, 12))[:, None]
    points = start + (end - start) * t
    # Mild posterior/anterior bend toward the body centroid prevents perfectly
    # straight synthetic-looking paths while keeping the trunk compact.
    midpoint = 0.5 * (start + end)
    bend = body_centroid_mm - midpoint
    bend[2] = 0.0
    norm = float(np.linalg.norm(bend))
    if norm > 1e-6:
        bend = bend / norm * min(8.0, 0.10 * float(np.linalg.norm(end - start)))
        points += np.sin(np.pi * t) * bend
    for index in range(1, len(points) - 1):
        points[index] = _pull_inside_body(points[index], body_mask, spacing_mm, body_centroid_mm)
    points[0] = start
    points[-1] = end
    return points


def _outside_fraction(points: np.ndarray, body_mask: np.ndarray, spacing_mm: np.ndarray) -> float:
    if len(points) == 0:
        return 1.0
    outside = sum(0 if _inside_body(point, body_mask, spacing_mm) else 1 for point in points)
    return float(outside / len(points))


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "edge_id",
        "length_before_mm",
        "length_after_mm",
        "outside_before_fraction",
        "outside_after_fraction",
        "point_count_before",
        "point_count_after",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_preview(path: Path, original_edges: list[dict[str, Any]], corrected_edges: list[dict[str, Any]]) -> None:
    plt, *_ = _import_dependencies()
    fig = plt.figure(figsize=(9, 7), dpi=170)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")
    all_points: list[np.ndarray] = []
    for edge in original_edges:
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if len(points) >= 2:
            all_points.append(points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#94a3b8", linewidth=1.0, alpha=0.40)
    for edge in corrected_edges:
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if len(points) >= 2:
            all_points.append(points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#2563eb", linewidth=3.0, alpha=0.96)
    if all_points:
        stacked = np.vstack(all_points)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = float((maxs - mins).max() / 2.0) * 1.15
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_title("BTCV IVC Trunk Tightening\nblue = tightened compact IVC, grey = previous registered IVC", pad=16)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _format_report(result: BtcvIvcTighteningResult) -> str:
    return "\n".join(
        [
            "# BTCV IVC Trunk Tightening",
            "",
            f"Case ID: `{result.case_id}`",
            "",
            "## Summary",
            "",
            f"- Corrected IVC trunk edges: {result.corrected_edge_count}",
            f"- Mean edge length before/after: {result.mean_length_before_mm:.2f} / {result.mean_length_after_mm:.2f} mm",
            f"- Mean outside-body fraction before/after: {result.mean_outside_before_fraction:.4f} / {result.mean_outside_after_fraction:.4f}",
            "",
            "## Outputs",
            "",
            f"- Corrected graph YAML: `{Path(result.corrected_graph_yaml_path).name}`",
            f"- Edge metrics CSV: `{Path(result.edge_metrics_csv_path).name}`",
            f"- Preview PNG: `{Path(result.preview_png_path).name}`",
            "",
            "## Interpretation",
            "",
            "- This pass replaces duplicated full-IVC template curves with compact local IVC trunk segments between the same graph anchors.",
            "- It is intended to keep venous trunk geometry inside the BTCV body before branch rerouting and voxelized flow QA.",
            "",
            "## Notes",
            *[f"- {note}" for note in result.notes],
        ]
    )


def tighten_btcv_ivc_trunk(
    graph_path: str | Path,
    anatomy_labels_path: str | Path,
    output_dir: str | Path = "outputs/digital/btcv_ivc_tightened",
    case_id: str = "btcv_ivc_tightened",
    edge_ids: tuple[str, ...] = (
        "ivc_lower_to_bifurcation_return",
        "ivc_bifurcation_to_renal_junction",
        "ivc_renal_to_hepatic_junction",
        "ivc_hepatic_to_outlet",
    ),
    point_count: int = 36,
    report_path: str | Path | None = "outputs/reports/btcv_ivc_tightening.md",
) -> BtcvIvcTighteningResult:
    _, nib, yaml = _import_dependencies()
    graph = _load_yaml(graph_path)
    image = nib.load(str(anatomy_labels_path))
    labels = np.asanyarray(image.dataobj).astype(np.int16)
    spacing_mm = np.asarray(image.header.get_zooms()[:3], dtype=float)
    body_mask = labels > 0
    body_centroid_mm = _body_centroid(labels, spacing_mm)
    nodes = [dict(node) for node in graph.get("nodes", [])]
    edges = [dict(edge) for edge in graph.get("edges", [])]
    lookup = _node_lookup({"nodes": nodes})

    original_ivc_edges: list[dict[str, Any]] = []
    corrected_ivc_edges: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    target_edge_ids = set(edge_ids)
    corrected = 0
    for edge in edges:
        edge_id = str(edge.get("id", ""))
        if edge_id not in target_edge_ids:
            continue
        source = lookup.get(str(edge.get("source")))
        target = lookup.get(str(edge.get("target")))
        if source is None or target is None:
            continue
        before_points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if before_points.ndim != 2 or before_points.shape[1] != 3 or len(before_points) < 2:
            before_points = np.vstack([source["position_mm"], target["position_mm"]]).astype(float)
        start = np.asarray(source["position_mm"], dtype=float)
        end = np.asarray(target["position_mm"], dtype=float)
        start = _pull_inside_body(start, body_mask, spacing_mm, body_centroid_mm)
        end = _pull_inside_body(end, body_mask, spacing_mm, body_centroid_mm)
        source["position_mm"] = [float(value) for value in start]
        target["position_mm"] = [float(value) for value in end]
        after_points = _tight_segment(
            start,
            end,
            body_mask=body_mask,
            spacing_mm=spacing_mm,
            body_centroid_mm=body_centroid_mm,
            point_count=point_count,
        )
        notes = list(edge.get("notes", []))
        notes.extend(
            [
                "ivc_trunk_tightened_inside_btcv_body",
                "duplicated_full_ivc_template_replaced_by_compact_local_segment",
            ]
        )
        edge["polyline_mm"] = [[float(value) for value in point] for point in after_points]
        edge["length_mm"] = _line_length(after_points)
        edge["notes"] = sorted(set(str(item) for item in notes))
        original_ivc_edges.append({"id": edge_id, "polyline_mm": [[float(value) for value in point] for point in before_points]})
        corrected_ivc_edges.append(edge)
        rows.append(
            {
                "edge_id": edge_id,
                "length_before_mm": f"{_line_length(before_points):.6f}",
                "length_after_mm": f"{_line_length(after_points):.6f}",
                "outside_before_fraction": f"{_outside_fraction(before_points, body_mask, spacing_mm):.8f}",
                "outside_after_fraction": f"{_outside_fraction(after_points, body_mask, spacing_mm):.8f}",
                "point_count_before": len(before_points),
                "point_count_after": len(after_points),
            }
        )
        corrected += 1

    metadata = dict(graph.get("graph_metadata", {}))
    metadata.update(
        {
            "ivc_tightening_source_graph": str(graph_path),
            "ivc_tightening_anatomy_labels": str(anatomy_labels_path),
            "ivc_tightened_edge_count": corrected,
            "ivc_tightening_method": "compact_local_segments_between_existing_ivc_anchors_with_body_clip",
            "ivc_tightening_point_count": int(point_count),
        }
    )
    provenance = list(graph.get("provenance_notes", []))
    provenance.append("ivc_trunk_tightened_inside_btcv_body_after_medseg_registration")
    graph.update(
        {
            "case_id": case_id,
            "graph_metadata": metadata,
            "provenance_notes": sorted(set(str(item) for item in provenance)),
            "nodes": nodes,
            "edges": edges,
        }
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_yaml = output / f"{case_id}_ivc_tightened_vascular_graph_v001.yaml"
    metrics_csv = output / f"{case_id}_ivc_tightening_edge_metrics_v001.csv"
    nodes_csv = output / f"{case_id}_ivc_tightened_vascular_graph_nodes_v001.csv"
    edges_csv = output / f"{case_id}_ivc_tightened_vascular_graph_edges_v001.csv"
    preview = output / f"{case_id}_ivc_tightening_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_ivc_tightening_report_v001.md"

    graph_yaml.write_text(yaml.safe_dump(graph, sort_keys=False))
    _write_metrics(metrics_csv, rows)
    _write_nodes_csv(nodes_csv, nodes)
    _write_edges_csv(edges_csv, edges)
    _write_preview(preview, original_ivc_edges, corrected_ivc_edges)

    before_lengths = [float(row["length_before_mm"]) for row in rows]
    after_lengths = [float(row["length_after_mm"]) for row in rows]
    before_outside = [float(row["outside_before_fraction"]) for row in rows]
    after_outside = [float(row["outside_after_fraction"]) for row in rows]
    result = BtcvIvcTighteningResult(
        case_id=case_id,
        output_dir=str(output),
        source_graph_path=str(graph_path),
        corrected_graph_yaml_path=str(graph_yaml),
        edge_metrics_csv_path=str(metrics_csv),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        preview_png_path=str(preview),
        report_path=str(report),
        corrected_edge_count=corrected,
        mean_length_before_mm=float(np.mean(before_lengths)) if before_lengths else 0.0,
        mean_length_after_mm=float(np.mean(after_lengths)) if after_lengths else 0.0,
        mean_outside_before_fraction=float(np.mean(before_outside)) if before_outside else 0.0,
        mean_outside_after_fraction=float(np.mean(after_outside)) if after_outside else 0.0,
        notes=(
            "ivc_trunk_edges_replaced_with_compact_inside_body_segments",
            "same_source_target_nodes_and_boundary_roles_preserved",
            "use_before_branch_reroute_and_final_vascular_qa",
        ),
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result) + "\n")
    return result


def format_btcv_ivc_tightening_result(result: BtcvIvcTighteningResult) -> str:
    return "\n".join(
        [
            "BTCV IVC trunk tightening completed",
            f"Case ID: {result.case_id}",
            f"Corrected IVC edges: {result.corrected_edge_count}",
            f"Mean length before/after: {result.mean_length_before_mm:.2f}/{result.mean_length_after_mm:.2f} mm",
            f"Mean outside-body fraction before/after: {result.mean_outside_before_fraction:.4f}/{result.mean_outside_after_fraction:.4f}",
            f"Corrected graph YAML: {result.corrected_graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
