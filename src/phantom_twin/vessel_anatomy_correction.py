from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np

from .cta_vascular_graph import _line_length
from .vessel_anatomy_validation import (
    _edge_samples,
    _import_dependencies,
    _load_yaml,
    _mask_for_group,
    _points_to_indices,
    _resolve_path,
    _sample_mask_fraction,
    _spacing_from_image,
)


@dataclass(frozen=True)
class VesselAnatomyCorrectionResult:
    case_id: str
    output_dir: str
    source_graph_path: str
    anatomy_labels_path: str
    corrected_graph_yaml_path: str
    node_corrections_csv_path: str
    edge_corrections_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    corrected_node_count: int
    corrected_edge_count: int
    mean_node_shift_mm: float
    max_node_shift_mm: float
    mean_edge_shift_mm: float
    max_edge_shift_mm: float
    notes: tuple[str, ...]


def _read_review_edges(path: str | Path | None, threshold: float) -> set[str] | None:
    if path is None:
        return None
    review_edges: set[str] = set()
    with Path(path).open(newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            try:
                bone_fraction = float(row.get("inside_bone_fraction", "0") or 0.0)
            except ValueError:
                bone_fraction = 0.0
            if row.get("status") != "pass" or bone_fraction > threshold:
                review_edges.add(str(row["edge_id"]))
    return review_edges


def _signed_bone_field(bone_mask: np.ndarray, spacing_mm: tuple[float, float, float], ndimage) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    outside_distance = ndimage.distance_transform_edt(~bone_mask, sampling=spacing_mm).astype(np.float32)
    inside_distance = ndimage.distance_transform_edt(bone_mask, sampling=spacing_mm).astype(np.float32)
    signed = outside_distance - inside_distance
    gradients = tuple(np.asarray(item, dtype=np.float32) for item in np.gradient(signed, *spacing_mm))
    return signed, gradients  # positive outside bone; negative inside bone.


def _nearest_index(point_mm: np.ndarray, spacing_mm: tuple[float, float, float], shape: tuple[int, int, int]) -> tuple[int, int, int] | None:
    index = np.rint(point_mm / np.asarray(spacing_mm, dtype=float)).astype(int)
    if np.any(index < 0) or np.any(index >= np.asarray(shape, dtype=int)):
        return None
    return tuple(int(value) for value in index)


def _signed_value(point_mm: np.ndarray, spacing_mm: tuple[float, float, float], signed: np.ndarray) -> float:
    index = _nearest_index(point_mm, spacing_mm, signed.shape)
    if index is None:
        return float("-inf")
    return float(signed[index])


def _inside_body(point_mm: np.ndarray, spacing_mm: tuple[float, float, float], body_mask: np.ndarray) -> bool:
    index = _nearest_index(point_mm, spacing_mm, body_mask.shape)
    return False if index is None else bool(body_mask[index])


def _fallback_direction(point_mm: np.ndarray, bone_centroid_mm: np.ndarray, body_centroid_mm: np.ndarray) -> np.ndarray:
    direction = point_mm - bone_centroid_mm
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = point_mm - body_centroid_mm
        norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        norm = 1.0
    return direction / norm


def _nudge_point_out_of_bone(
    point_mm: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    signed: np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray, np.ndarray],
    body_mask: np.ndarray,
    bone_centroid_mm: np.ndarray,
    body_centroid_mm: np.ndarray,
    clearance_mm: float,
    max_shift_mm: float,
) -> tuple[np.ndarray, float, float, bool]:
    start_signed = _signed_value(point_mm, spacing_mm, signed)
    if start_signed >= clearance_mm:
        return point_mm.astype(float), 0.0, start_signed, False

    index = _nearest_index(point_mm, spacing_mm, signed.shape)
    if index is None:
        return point_mm.astype(float), 0.0, start_signed, False

    gradient = np.asarray([gradients[axis][index] for axis in range(3)], dtype=float)
    norm = float(np.linalg.norm(gradient))
    direction = gradient / norm if norm > 1e-6 else _fallback_direction(point_mm, bone_centroid_mm, body_centroid_mm)
    needed = max(0.0, clearance_mm - start_signed)
    shift = min(float(max_shift_mm), needed)

    best = point_mm.astype(float)
    best_signed = start_signed
    best_shift = 0.0
    for scale in (1.0, 0.75, 0.5, 0.35, 0.2, 0.1):
        candidate_shift = shift * scale
        candidate = point_mm + direction * candidate_shift
        candidate_signed = _signed_value(candidate, spacing_mm, signed)
        if _inside_body(candidate, spacing_mm, body_mask) and candidate_signed > best_signed:
            best = candidate
            best_signed = candidate_signed
            best_shift = candidate_shift
            if candidate_signed >= clearance_mm:
                break
    return best, float(best_shift), float(best_signed), best_shift > 1e-6


def _smooth_polyline(points: np.ndarray, iterations: int) -> np.ndarray:
    if len(points) <= 2 or iterations <= 0:
        return points
    smoothed = points.astype(float).copy()
    for _ in range(iterations):
        updated = smoothed.copy()
        updated[1:-1] = 0.25 * smoothed[:-2] + 0.50 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed = updated
    return smoothed


def _write_node_corrections(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "node_id",
        "boundary_role",
        "corrected",
        "original_x_mm",
        "original_y_mm",
        "original_z_mm",
        "corrected_x_mm",
        "corrected_y_mm",
        "corrected_z_mm",
        "shift_mm",
        "signed_distance_before_mm",
        "signed_distance_after_mm",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_edge_corrections(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "edge_id",
        "corrected",
        "targeted_for_bone_review",
        "vessel_type",
        "flow_role",
        "point_count",
        "corrected_point_count",
        "mean_point_shift_mm",
        "max_point_shift_mm",
        "bone_fraction_before",
        "bone_fraction_after",
        "length_before_mm",
        "length_after_mm",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _render_preview(path: Path, labels: np.ndarray, original_graph: dict[str, Any], corrected_graph: dict[str, Any], spacing_mm: tuple[float, float, float]) -> None:
    plt, *_ = _import_dependencies()
    bone = _mask_for_group(labels, "bone")
    body = _mask_for_group(labels, "body")
    coords = np.argwhere(bone if np.any(bone) else body)
    z_index = int(round(float(np.median(coords[:, 2])))) if len(coords) else labels.shape[2] // 2

    fig = plt.figure(figsize=(12, 7), dpi=170)
    fig.patch.set_facecolor("#f8f3e8")
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in (ax1,):
        ax.set_facecolor("#f8f3e8")
        ax.axis("off")

    ax1.imshow(np.rot90(labels[:, :, z_index]), cmap="bone", interpolation="nearest")
    if np.any(bone[:, :, z_index]):
        ax1.contour(np.rot90(bone[:, :, z_index].astype(float)), levels=[0.5], colors=["#ffffff"], linewidths=1.0)
    for graph, color, width, alpha, label in (
        (original_graph, "#94a3b8", 0.8, 0.45, "original"),
        (corrected_graph, "#ef4444", 1.3, 0.95, "corrected"),
    ):
        for edge in graph.get("edges", []):
            points = np.asarray(edge.get("polyline_mm", []), dtype=float)
            if points.ndim != 2 or len(points) < 2:
                continue
            z_dist = np.abs(points[:, 2] / spacing_mm[2] - z_index)
            near = z_dist <= 2.5
            if np.count_nonzero(near) >= 2:
                xy = points[near, :2] / np.asarray(spacing_mm[:2], dtype=float)
                ax1.plot(xy[:, 1], labels.shape[0] - xy[:, 0], color=color, linewidth=width, alpha=alpha, label=label)

    for graph, color, alpha, label in (
        (original_graph, "#94a3b8", 0.32, "original"),
        (corrected_graph, "#dc2626", 0.95, "corrected"),
    ):
        for edge in graph.get("edges", []):
            points = np.asarray(edge.get("polyline_mm", []), dtype=float)
            if points.ndim == 2 and len(points) >= 2:
                ax2.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=1.3, alpha=alpha)
    ax2.set_facecolor("#f8f3e8")
    ax2.set_title("3D Centerline Correction")
    ax2.set_xlabel("x mm")
    ax2.set_ylabel("y mm")
    ax2.set_zlabel("z mm")
    ax2.view_init(elev=20, azim=-58)
    ax1.set_title(f"Bone Slice + Centerlines z={z_index}")
    fig.suptitle("Organ-Aware Vascular Bone-Conflict Correction", fontsize=15, color="#111827")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_spec(path: Path, result: VesselAnatomyCorrectionResult, clearance_mm: float, edge_bone_threshold: float) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "correction_type": "organ_aware_vascular_bone_conflict_correction",
        "source_graph": result.source_graph_path,
        "source_anatomy_labels": result.anatomy_labels_path,
        "clearance_mm": clearance_mm,
        "edge_bone_review_threshold": edge_bone_threshold,
        "summary": {
            "corrected_node_count": result.corrected_node_count,
            "corrected_edge_count": result.corrected_edge_count,
            "mean_node_shift_mm": result.mean_node_shift_mm,
            "max_node_shift_mm": result.max_node_shift_mm,
            "mean_edge_shift_mm": result.mean_edge_shift_mm,
            "max_edge_shift_mm": result.max_edge_shift_mm,
        },
        "outputs": {
            "corrected_graph": result.corrected_graph_yaml_path,
            "node_corrections_csv": result.node_corrections_csv_path,
            "edge_corrections_csv": result.edge_corrections_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VesselAnatomyCorrectionResult, edge_rows: list[dict[str, Any]], node_rows: list[dict[str, Any]]) -> str:
    moved_edges = [row for row in edge_rows if int(row["corrected"]) == 1]
    moved_nodes = [row for row in node_rows if int(row["corrected"]) == 1]
    lines = [
        "# Organ-Aware Vascular Bone-Conflict Correction",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Corrected nodes: {result.corrected_node_count}",
        f"- Corrected edges: {result.corrected_edge_count}",
        f"- Mean / max node shift: {result.mean_node_shift_mm:.2f} / {result.max_node_shift_mm:.2f} mm",
        f"- Mean / max edge point shift: {result.mean_edge_shift_mm:.2f} / {result.max_edge_shift_mm:.2f} mm",
        "",
        "## Outputs",
        "",
        f"- Corrected graph YAML: `{Path(result.corrected_graph_yaml_path).name}`",
        f"- Node corrections CSV: `{Path(result.node_corrections_csv_path).name}`",
        f"- Edge corrections CSV: `{Path(result.edge_corrections_csv_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Spec YAML: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Largest Edge Corrections",
        "",
        "| edge | targeted | bone before % | bone after % | mean shift mm | max shift mm |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(moved_edges, key=lambda item: float(item["max_point_shift_mm"]), reverse=True)[:12]:
        lines.append(
            f"| `{row['edge_id']}` | {row['targeted_for_bone_review']} | "
            f"{float(row['bone_fraction_before']) * 100.0:.2f} | {float(row['bone_fraction_after']) * 100.0:.2f} | "
            f"{float(row['mean_point_shift_mm']):.2f} | {float(row['max_point_shift_mm']):.2f} |"
        )
    if not moved_edges:
        lines.append("| none | no | 0.00 | 0.00 | 0.00 | 0.00 |")

    lines.extend(["", "## Corrected Nodes", "", "| node | boundary role | shift mm | signed distance before/after mm |", "| --- | --- | ---: | --- |"])
    for row in sorted(moved_nodes, key=lambda item: float(item["shift_mm"]), reverse=True)[:12]:
        lines.append(
            f"| `{row['node_id']}` | {row['boundary_role'] or 'internal'} | {float(row['shift_mm']):.2f} | "
            f"{float(row['signed_distance_before_mm']):.2f} / {float(row['signed_distance_after_mm']):.2f} |"
        )
    if not moved_nodes:
        lines.append("| none | n/a | 0.00 | n/a |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Node IDs, boundary roles, vessel types, radii, and flow roles are preserved.",
            "- Node positions and edge interior points are moved only enough to improve signed distance from bone within the configured shift limits.",
            "- This is an engineering geometry-correction pass; it should be followed by voxelization and organ-aware validation before being used downstream.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def correct_vessel_bone_conflicts(
    graph_yaml_path: str | Path,
    anatomy_labels_path: str | Path,
    edge_metrics_csv_path: str | Path | None = None,
    output_dir: str | Path = "outputs/digital/vessel_anatomy_corrected",
    case_id: str = "ct_org_vessel_bone_corrected",
    clearance_mm: float = 8.0,
    edge_bone_review_threshold: float = 0.05,
    max_node_shift_mm: float = 24.0,
    max_point_shift_mm: float = 24.0,
    smooth_iterations: int = 1,
    report_path: str | Path | None = "outputs/reports/vessel_anatomy_correction_stage001.md",
) -> VesselAnatomyCorrectionResult:
    plt, nib, ndimage, yaml = _import_dependencies()
    _ = plt
    graph_path = Path(graph_yaml_path)
    labels_path = Path(anatomy_labels_path)
    graph = _load_yaml(graph_path)
    labels_image = nib.load(str(labels_path))
    labels = np.rint(np.asanyarray(labels_image.dataobj)).astype(np.int16)
    spacing = _spacing_from_image(labels_image)
    bone = _mask_for_group(labels, "bone")
    body = _mask_for_group(labels, "body")
    if not np.any(bone):
        raise ValueError("Anatomy labels do not contain a bone label to correct against")
    signed, gradients = _signed_bone_field(bone, spacing, ndimage)
    bone_coords = np.argwhere(bone).astype(float) * np.asarray(spacing, dtype=float)
    body_coords = np.argwhere(body).astype(float) * np.asarray(spacing, dtype=float)
    bone_centroid = bone_coords.mean(axis=0)
    body_centroid = body_coords.mean(axis=0) if len(body_coords) else bone_centroid
    review_edges = _read_review_edges(edge_metrics_csv_path, edge_bone_review_threshold)

    corrected_graph = dict(graph)
    nodes = [dict(node) for node in graph.get("nodes", [])]
    edges = [dict(edge) for edge in graph.get("edges", [])]
    node_positions: dict[str, np.ndarray] = {}
    node_rows: list[dict[str, Any]] = []

    for node in nodes:
        node_id = str(node.get("id", ""))
        original = np.asarray(node.get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
        before = _signed_value(original, spacing, signed)
        corrected, shift, after, changed = _nudge_point_out_of_bone(
            original,
            spacing_mm=spacing,
            signed=signed,
            gradients=gradients,
            body_mask=body,
            bone_centroid_mm=bone_centroid,
            body_centroid_mm=body_centroid,
            clearance_mm=clearance_mm,
            max_shift_mm=max_node_shift_mm,
        )
        if changed:
            node["position_mm"] = [float(value) for value in corrected]
            notes = list(node.get("notes", []))
            notes.append("position_adjusted_by_organ_aware_bone_conflict_correction")
            node["notes"] = sorted(set(str(item) for item in notes))
        node_positions[node_id] = corrected
        node_rows.append(
            {
                "node_id": node_id,
                "boundary_role": str(node.get("boundary_role", "")),
                "corrected": int(changed),
                "original_x_mm": f"{original[0]:.6f}",
                "original_y_mm": f"{original[1]:.6f}",
                "original_z_mm": f"{original[2]:.6f}",
                "corrected_x_mm": f"{corrected[0]:.6f}",
                "corrected_y_mm": f"{corrected[1]:.6f}",
                "corrected_z_mm": f"{corrected[2]:.6f}",
                "shift_mm": f"{shift:.6f}",
                "signed_distance_before_mm": f"{before:.6f}",
                "signed_distance_after_mm": f"{after:.6f}",
            }
        )

    edge_rows: list[dict[str, Any]] = []
    edge_shift_values: list[float] = []
    for edge in edges:
        edge_id = str(edge.get("id", ""))
        targeted = review_edges is None or edge_id in review_edges
        original_points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if original_points.ndim != 2 or len(original_points) < 2:
            edge_rows.append(
                {
                    "edge_id": edge_id,
                    "corrected": 0,
                    "targeted_for_bone_review": int(targeted),
                    "vessel_type": edge.get("vessel_type", ""),
                    "flow_role": edge.get("flow_role", ""),
                    "point_count": 0,
                    "corrected_point_count": 0,
                    "mean_point_shift_mm": "0.000000",
                    "max_point_shift_mm": "0.000000",
                    "bone_fraction_before": "0.000000",
                    "bone_fraction_after": "0.000000",
                    "length_before_mm": "0.000000",
                    "length_after_mm": "0.000000",
                }
            )
            continue

        before_indices, before_valid = _points_to_indices(_edge_samples(edge, 2.0), spacing, labels.shape)
        before_bone_fraction = _sample_mask_fraction(bone, before_indices, before_valid)

        corrected_points = original_points.copy()
        source_position = node_positions.get(str(edge.get("source")))
        target_position = node_positions.get(str(edge.get("target")))
        if source_position is not None:
            corrected_points[0] = source_position
        if target_position is not None:
            corrected_points[-1] = target_position

        shifts = np.zeros(len(corrected_points), dtype=float)
        if targeted:
            for index in range(1, len(corrected_points) - 1):
                corrected, shift, _, changed = _nudge_point_out_of_bone(
                    corrected_points[index],
                    spacing_mm=spacing,
                    signed=signed,
                    gradients=gradients,
                    body_mask=body,
                    bone_centroid_mm=bone_centroid,
                    body_centroid_mm=body_centroid,
                    clearance_mm=clearance_mm,
                    max_shift_mm=max_point_shift_mm,
                )
                if changed:
                    corrected_points[index] = corrected
                    shifts[index] = shift
            corrected_points = _smooth_polyline(corrected_points, smooth_iterations)
            corrected_points[0] = source_position if source_position is not None else corrected_points[0]
            corrected_points[-1] = target_position if target_position is not None else corrected_points[-1]
            for index in range(1, len(corrected_points) - 1):
                corrected, shift, _, changed = _nudge_point_out_of_bone(
                    corrected_points[index],
                    spacing_mm=spacing,
                    signed=signed,
                    gradients=gradients,
                    body_mask=body,
                    bone_centroid_mm=bone_centroid,
                    body_centroid_mm=body_centroid,
                    clearance_mm=clearance_mm,
                    max_shift_mm=max_point_shift_mm * 0.5,
                )
                if changed:
                    corrected_points[index] = corrected
                    shifts[index] = max(shifts[index], shift)

        total_point_shift = np.linalg.norm(corrected_points - original_points, axis=1)
        changed_edge = bool(np.max(total_point_shift) > 1e-6)
        if changed_edge:
            edge["polyline_mm"] = [[float(value) for value in point] for point in corrected_points]
            edge["length_mm"] = _line_length(corrected_points)
            notes = list(edge.get("notes", []))
            notes.append("polyline_adjusted_by_organ_aware_bone_conflict_correction")
            edge["notes"] = sorted(set(str(item) for item in notes))
            edge_shift_values.extend([float(value) for value in total_point_shift if value > 1e-6])

        after_edge = dict(edge)
        after_edge["polyline_mm"] = [[float(value) for value in point] for point in corrected_points]
        after_indices, after_valid = _points_to_indices(_edge_samples(after_edge, 2.0), spacing, labels.shape)
        after_bone_fraction = _sample_mask_fraction(bone, after_indices, after_valid)
        edge_rows.append(
            {
                "edge_id": edge_id,
                "corrected": int(changed_edge),
                "targeted_for_bone_review": int(targeted),
                "vessel_type": edge.get("vessel_type", ""),
                "flow_role": edge.get("flow_role", ""),
                "point_count": int(len(original_points)),
                "corrected_point_count": int(np.count_nonzero(total_point_shift > 1e-6)),
                "mean_point_shift_mm": f"{float(np.mean(total_point_shift)):.6f}",
                "max_point_shift_mm": f"{float(np.max(total_point_shift)):.6f}",
                "bone_fraction_before": f"{before_bone_fraction:.6f}",
                "bone_fraction_after": f"{after_bone_fraction:.6f}",
                "length_before_mm": f"{_line_length(original_points):.6f}",
                "length_after_mm": f"{_line_length(corrected_points):.6f}",
            }
        )

    corrected_graph["nodes"] = nodes
    corrected_graph["edges"] = edges
    metadata = dict(corrected_graph.get("graph_metadata", {}))
    metadata.update(
        {
            "organ_aware_bone_correction": {
                "source_graph": str(graph_path),
                "source_anatomy_labels": str(labels_path),
                "source_edge_metrics": str(edge_metrics_csv_path) if edge_metrics_csv_path is not None else None,
                "clearance_mm": float(clearance_mm),
                "corrected_node_count": int(sum(1 for row in node_rows if int(row["corrected"]) == 1)),
                "corrected_edge_count": int(sum(1 for row in edge_rows if int(row["corrected"]) == 1)),
            }
        }
    )
    corrected_graph["graph_metadata"] = metadata
    notes = list(corrected_graph.get("provenance_notes", []))
    notes.append("organ_aware_bone_conflict_correction_applied")
    corrected_graph["provenance_notes"] = sorted(set(str(item) for item in notes))
    corrected_graph["case_id"] = case_id

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corrected_graph_yaml = output / f"{case_id}_organ_aware_bone_corrected_vascular_graph_v001.yaml"
    node_csv = output / f"{case_id}_organ_aware_bone_node_corrections_v001.csv"
    edge_csv = output / f"{case_id}_organ_aware_bone_edge_corrections_v001.csv"
    preview = output / f"{case_id}_organ_aware_bone_correction_preview_v001.png"
    spec_yaml = output / f"{case_id}_organ_aware_bone_correction_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_organ_aware_bone_correction_report_v001.md"

    corrected_graph_yaml.write_text(yaml.safe_dump(corrected_graph, sort_keys=False))
    _write_node_corrections(node_csv, node_rows)
    _write_edge_corrections(edge_csv, edge_rows)
    _render_preview(preview, labels, graph, corrected_graph, spacing)

    node_shifts = [float(row["shift_mm"]) for row in node_rows if int(row["corrected"]) == 1]
    corrected_node_count = len(node_shifts)
    corrected_edge_count = sum(1 for row in edge_rows if int(row["corrected"]) == 1)
    mean_node_shift = float(np.mean(node_shifts)) if node_shifts else 0.0
    max_node_shift = float(np.max(node_shifts)) if node_shifts else 0.0
    mean_edge_shift = float(np.mean(edge_shift_values)) if edge_shift_values else 0.0
    max_edge_shift = float(np.max(edge_shift_values)) if edge_shift_values else 0.0
    result = VesselAnatomyCorrectionResult(
        case_id=case_id,
        output_dir=str(output),
        source_graph_path=str(graph_path),
        anatomy_labels_path=str(labels_path),
        corrected_graph_yaml_path=str(corrected_graph_yaml),
        node_corrections_csv_path=str(node_csv),
        edge_corrections_csv_path=str(edge_csv),
        preview_png_path=str(preview),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        corrected_node_count=corrected_node_count,
        corrected_edge_count=corrected_edge_count,
        mean_node_shift_mm=mean_node_shift,
        max_node_shift_mm=max_node_shift,
        mean_edge_shift_mm=mean_edge_shift,
        max_edge_shift_mm=max_edge_shift,
        notes=(
            "node_ids_boundary_roles_vessel_types_and_flow_roles_preserved",
            "positions_adjusted_by_signed_bone_distance_gradient",
            "correction_requires_downstream_voxelization_and_vessel_organ_validation",
        ),
    )
    _write_spec(spec_yaml, result, clearance_mm=clearance_mm, edge_bone_threshold=edge_bone_review_threshold)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, edge_rows, node_rows) + "\n")
    return result


def format_vessel_anatomy_correction_result(result: VesselAnatomyCorrectionResult) -> str:
    return "\n".join(
        [
            "Organ-aware vascular bone-conflict correction completed",
            f"Case ID: {result.case_id}",
            f"Corrected nodes/edges: {result.corrected_node_count}/{result.corrected_edge_count}",
            f"Mean/max node shift: {result.mean_node_shift_mm:.2f}/{result.max_node_shift_mm:.2f} mm",
            f"Mean/max edge point shift: {result.mean_edge_shift_mm:.2f}/{result.max_edge_shift_mm:.2f} mm",
            f"Corrected graph YAML: {result.corrected_graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
