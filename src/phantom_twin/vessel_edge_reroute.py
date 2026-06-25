from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import math

import numpy as np

from .cta_vascular_graph import _line_length
from .vessel_anatomy_correction import (
    _fallback_direction,
    _nudge_point_out_of_bone,
    _signed_bone_field,
    _signed_value,
    _smooth_polyline,
)
from .vessel_anatomy_validation import (
    _edge_samples,
    _import_dependencies,
    _load_yaml,
    _mask_for_group,
    _points_to_indices,
    _sample_mask_fraction,
    _spacing_from_image,
)


@dataclass(frozen=True)
class VesselEdgeRerouteResult:
    case_id: str
    output_dir: str
    source_graph_path: str
    anatomy_labels_path: str
    edge_id: str
    corrected_graph_yaml_path: str
    candidate_metrics_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    status: str
    bone_fraction_before: float
    bone_fraction_after: float
    outside_body_fraction_before: float
    outside_body_fraction_after: float
    min_signed_bone_distance_before_mm: float
    min_signed_bone_distance_after_mm: float
    length_before_mm: float
    length_after_mm: float
    selected_candidate_id: str
    selected_detour_mm: float
    notes: tuple[str, ...]


def _resample_polyline(points: np.ndarray, step_mm: float, minimum_points: int = 24) -> np.ndarray:
    if len(points) < 2:
        return points.astype(float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-6:
        return points.astype(float)
    count = max(minimum_points, int(math.ceil(total / max(step_mm, 0.5))) + 1)
    targets = np.linspace(0.0, total, count)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    resampled = np.empty((count, 3), dtype=float)
    for target_index, distance in enumerate(targets):
        segment = min(max(int(np.searchsorted(cumulative, distance, side="right")) - 1, 0), len(lengths) - 1)
        segment_length = max(float(lengths[segment]), 1e-6)
        local_t = float((distance - cumulative[segment]) / segment_length)
        resampled[target_index] = points[segment] + (points[segment + 1] - points[segment]) * local_t
    resampled[0] = points[0]
    resampled[-1] = points[-1]
    return resampled


def _project_perpendicular(vector: np.ndarray, chord: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    chord = np.asarray(chord, dtype=float)
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm < 1e-6:
        projected = vector
    else:
        unit = chord / chord_norm
        projected = vector - unit * float(np.dot(vector, unit))
    norm = float(np.linalg.norm(projected))
    if norm < 1e-6:
        return None
    return projected / norm


def _candidate_directions(
    start: np.ndarray,
    end: np.ndarray,
    bone_centroid_mm: np.ndarray,
    body_centroid_mm: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    midpoint = 0.5 * (start + end)
    chord = end - start
    raw = [
        ("away_from_bone_centroid", midpoint - bone_centroid_mm),
        ("toward_body_centroid", body_centroid_mm - bone_centroid_mm),
        ("positive_x", np.array([1.0, 0.0, 0.0])),
        ("negative_x", np.array([-1.0, 0.0, 0.0])),
        ("positive_y", np.array([0.0, 1.0, 0.0])),
        ("negative_y", np.array([0.0, -1.0, 0.0])),
        ("positive_z", np.array([0.0, 0.0, 1.0])),
        ("negative_z", np.array([0.0, 0.0, -1.0])),
    ]

    directions: list[tuple[str, np.ndarray]] = []
    seen: list[np.ndarray] = []
    for name, vector in raw:
        projected = _project_perpendicular(vector, chord)
        if projected is None:
            continue
        if any(abs(float(np.dot(projected, existing))) > 0.985 for existing in seen):
            continue
        seen.append(projected)
        directions.append((name, projected))

    if not directions:
        fallback = _fallback_direction(midpoint, bone_centroid_mm, body_centroid_mm)
        directions.append(("fallback_radial", fallback))
    return directions


def _metrics_for_points(
    points: np.ndarray,
    *,
    template_edge: dict[str, Any],
    spacing_mm: tuple[float, float, float],
    shape: tuple[int, int, int],
    body_mask: np.ndarray,
    bone_mask: np.ndarray,
    signed_bone_mm: np.ndarray,
    sample_step_mm: float,
) -> dict[str, float]:
    edge = dict(template_edge)
    edge["polyline_mm"] = [[float(value) for value in point] for point in points]
    samples = _edge_samples(edge, sample_step_mm)
    indices, valid = _points_to_indices(samples, spacing_mm, shape)
    if len(samples) == 0:
        return {
            "sample_count": 0.0,
            "outside_body_fraction": 1.0,
            "bone_fraction": 1.0,
            "min_signed_bone_distance_mm": float("-inf"),
            "mean_signed_bone_distance_mm": float("-inf"),
        }

    valid_indices = indices[valid]
    invalid_count = int(len(samples) - int(np.count_nonzero(valid)))
    inside_body = int(np.count_nonzero(body_mask[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]])) if len(valid_indices) else 0
    outside_body = invalid_count + int(len(valid_indices) - inside_body)
    signed_values = signed_bone_mm[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] if len(valid_indices) else np.asarray([float("-inf")])

    return {
        "sample_count": float(len(samples)),
        "outside_body_fraction": float(outside_body / len(samples)),
        "bone_fraction": _sample_mask_fraction(bone_mask, indices, valid),
        "min_signed_bone_distance_mm": float(np.min(signed_values)),
        "mean_signed_bone_distance_mm": float(np.mean(signed_values)),
    }


def _clean_candidate_with_signed_field(
    points: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    signed: np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray, np.ndarray],
    body_mask: np.ndarray,
    bone_centroid_mm: np.ndarray,
    body_centroid_mm: np.ndarray,
    clearance_mm: float,
    max_point_shift_mm: float,
    smooth_iterations: int,
) -> np.ndarray:
    cleaned = np.asarray(points, dtype=float).copy()
    if len(cleaned) <= 2:
        return cleaned
    for index in range(1, len(cleaned) - 1):
        corrected, _, _, changed = _nudge_point_out_of_bone(
            cleaned[index],
            spacing_mm=spacing_mm,
            signed=signed,
            gradients=gradients,
            body_mask=body_mask,
            bone_centroid_mm=bone_centroid_mm,
            body_centroid_mm=body_centroid_mm,
            clearance_mm=clearance_mm,
            max_shift_mm=max_point_shift_mm,
        )
        if changed:
            cleaned[index] = corrected
    cleaned = _smooth_polyline(cleaned, smooth_iterations)
    cleaned[0] = points[0]
    cleaned[-1] = points[-1]
    for index in range(1, len(cleaned) - 1):
        corrected, _, _, changed = _nudge_point_out_of_bone(
            cleaned[index],
            spacing_mm=spacing_mm,
            signed=signed,
            gradients=gradients,
            body_mask=body_mask,
            bone_centroid_mm=bone_centroid_mm,
            body_centroid_mm=body_centroid_mm,
            clearance_mm=clearance_mm,
            max_shift_mm=max_point_shift_mm * 0.5,
        )
        if changed:
            cleaned[index] = corrected
    cleaned[0] = points[0]
    cleaned[-1] = points[-1]
    return cleaned


def _write_candidate_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "direction",
        "detour_mm",
        "score",
        "selected",
        "sample_count",
        "outside_body_fraction",
        "bone_fraction",
        "min_signed_bone_distance_mm",
        "mean_signed_bone_distance_mm",
        "length_mm",
        "length_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _render_preview(
    path: Path,
    *,
    labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    edge_id: str,
    original_points: np.ndarray,
    rerouted_points: np.ndarray,
) -> None:
    plt, *_ = _import_dependencies()
    bone = _mask_for_group(labels, "bone")
    coords = np.argwhere(bone)
    if len(coords):
        z_index = int(round(float(np.median((rerouted_points[:, 2] / spacing_mm[2]).clip(0, labels.shape[2] - 1)))))
    else:
        z_index = labels.shape[2] // 2

    fig = plt.figure(figsize=(12, 7), dpi=170)
    fig.patch.set_facecolor("#f7f1e3")
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_facecolor("#f7f1e3")
    ax1.axis("off")
    ax1.imshow(np.rot90(labels[:, :, z_index]), cmap="bone", interpolation="nearest")
    if np.any(bone[:, :, z_index]):
        ax1.contour(np.rot90(bone[:, :, z_index].astype(float)), levels=[0.5], colors=["#ffffff"], linewidths=1.0)
    for points, color, label, linewidth in (
        (original_points, "#64748b", "before", 1.0),
        (rerouted_points, "#ef4444", "after", 1.8),
    ):
        near = np.abs(points[:, 2] / spacing_mm[2] - z_index) <= 4.0
        if np.count_nonzero(near) >= 2:
            xy = points[near, :2] / np.asarray(spacing_mm[:2], dtype=float)
            ax1.plot(xy[:, 1], labels.shape[0] - xy[:, 0], color=color, linewidth=linewidth, label=label)

    for points, color, label, alpha in (
        (original_points, "#64748b", "before", 0.45),
        (rerouted_points, "#dc2626", "after", 0.95),
    ):
        ax2.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=1.6, alpha=alpha, label=label)
        ax2.scatter(points[[0, -1], 0], points[[0, -1], 1], points[[0, -1], 2], color=color, s=16, alpha=alpha)
    ax2.set_facecolor("#f7f1e3")
    ax2.set_title("Rerouted Centerline")
    ax2.set_xlabel("x mm")
    ax2.set_ylabel("y mm")
    ax2.set_zlabel("z mm")
    ax2.view_init(elev=20, azim=-58)
    ax2.legend(loc="upper left")
    ax1.set_title(f"Bone Slice + Edge z={z_index}")
    fig.suptitle(f"Targeted Vessel Reroute: {edge_id}", fontsize=15, color="#111827")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_spec(path: Path, result: VesselEdgeRerouteResult, params: dict[str, Any]) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "correction_type": "targeted_vessel_edge_bone_avoidance_reroute",
        "edge_id": result.edge_id,
        "source_graph": result.source_graph_path,
        "source_anatomy_labels": result.anatomy_labels_path,
        "parameters": params,
        "summary": {
            "status": result.status,
            "bone_fraction_before": result.bone_fraction_before,
            "bone_fraction_after": result.bone_fraction_after,
            "outside_body_fraction_before": result.outside_body_fraction_before,
            "outside_body_fraction_after": result.outside_body_fraction_after,
            "min_signed_bone_distance_before_mm": result.min_signed_bone_distance_before_mm,
            "min_signed_bone_distance_after_mm": result.min_signed_bone_distance_after_mm,
            "length_before_mm": result.length_before_mm,
            "length_after_mm": result.length_after_mm,
            "selected_candidate_id": result.selected_candidate_id,
            "selected_detour_mm": result.selected_detour_mm,
        },
        "outputs": {
            "corrected_graph": result.corrected_graph_yaml_path,
            "candidate_metrics_csv": result.candidate_metrics_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VesselEdgeRerouteResult, candidate_rows: list[dict[str, Any]]) -> str:
    top = sorted(candidate_rows, key=lambda row: float(row["score"]))[:8]
    lines = [
        "# Targeted Vessel Edge Reroute",
        "",
        f"Case ID: `{result.case_id}`",
        f"Edge ID: `{result.edge_id}`",
        f"Status: `{result.status}`",
        "",
        "## Summary",
        "",
        f"- Centerline bone fraction: {result.bone_fraction_before * 100.0:.2f}% -> {result.bone_fraction_after * 100.0:.2f}%",
        f"- Outside-body fraction: {result.outside_body_fraction_before * 100.0:.2f}% -> {result.outside_body_fraction_after * 100.0:.2f}%",
        f"- Minimum signed bone distance: {result.min_signed_bone_distance_before_mm:.2f} -> {result.min_signed_bone_distance_after_mm:.2f} mm",
        f"- Length: {result.length_before_mm:.2f} -> {result.length_after_mm:.2f} mm",
        f"- Selected candidate: `{result.selected_candidate_id}` with {result.selected_detour_mm:.1f} mm detour",
        "",
        "## Outputs",
        "",
        f"- Corrected graph YAML: `{Path(result.corrected_graph_yaml_path).name}`",
        f"- Candidate metrics CSV: `{Path(result.candidate_metrics_csv_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Spec YAML: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Best Candidate Scores",
        "",
        "| candidate | direction | detour mm | score | outside body % | bone % | min signed mm | length ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        selected = " selected" if str(row.get("selected")) == "1" else ""
        lines.append(
            f"| `{row['candidate_id']}`{selected} | {row['direction']} | {float(row['detour_mm']):.1f} | "
            f"{float(row['score']):.3f} | {float(row['outside_body_fraction']) * 100.0:.2f} | "
            f"{float(row['bone_fraction']) * 100.0:.2f} | {float(row['min_signed_bone_distance_mm']):.2f} | "
            f"{float(row['length_ratio']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Source node, target node, vessel type, flow role, and radii are preserved.",
            "- Only the selected edge polyline and derived length are changed.",
            "- This edge-level reroute should be followed by voxelization plus organ-aware and radius-aware QA before downstream flow or RT use.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def reroute_vessel_edge_around_bone(
    graph_yaml_path: str | Path,
    anatomy_labels_path: str | Path,
    edge_id: str,
    output_dir: str | Path = "outputs/digital/vessel_edge_reroutes",
    case_id: str = "targeted_vessel_edge_reroute",
    clearance_mm: float = 8.0,
    max_detour_mm: float = 90.0,
    detour_step_mm: float = 6.0,
    sample_step_mm: float = 2.0,
    resample_step_mm: float = 3.0,
    max_point_shift_mm: float = 18.0,
    smooth_iterations: int = 2,
    report_path: str | Path | None = "outputs/reports/vessel_edge_reroute_stage001.md",
) -> VesselEdgeRerouteResult:
    plt, nib, ndimage, yaml = _import_dependencies()
    _ = plt
    graph_path = Path(graph_yaml_path)
    labels_path = Path(anatomy_labels_path)
    graph = _load_yaml(graph_path)
    labels_image = nib.load(str(labels_path))
    labels = np.rint(np.asanyarray(labels_image.dataobj)).astype(np.int16)
    spacing = _spacing_from_image(labels_image)
    body = _mask_for_group(labels, "body")
    bone = _mask_for_group(labels, "bone")
    if not np.any(body):
        raise ValueError("Anatomy labels do not contain a body mask")
    if not np.any(bone):
        raise ValueError("Anatomy labels do not contain bone labels")

    signed, gradients = _signed_bone_field(bone, spacing, ndimage)
    bone_coords = np.argwhere(bone).astype(float) * np.asarray(spacing, dtype=float)
    body_coords = np.argwhere(body).astype(float) * np.asarray(spacing, dtype=float)
    bone_centroid = bone_coords.mean(axis=0)
    body_centroid = body_coords.mean(axis=0)

    edges = [dict(edge) for edge in graph.get("edges", [])]
    try:
        edge_index = next(index for index, edge in enumerate(edges) if str(edge.get("id")) == edge_id)
    except StopIteration as exc:
        raise ValueError(f"Edge ID not found in graph: {edge_id}") from exc
    original_edge = dict(edges[edge_index])
    original_points = np.asarray(original_edge.get("polyline_mm", []), dtype=float)
    if original_points.ndim != 2 or len(original_points) < 2:
        raise ValueError(f"Edge {edge_id} does not contain a usable polyline_mm")

    base_points = _resample_polyline(original_points, resample_step_mm)
    before_metrics = _metrics_for_points(
        base_points,
        template_edge=original_edge,
        spacing_mm=spacing,
        shape=labels.shape,
        body_mask=body,
        bone_mask=bone,
        signed_bone_mm=signed,
        sample_step_mm=sample_step_mm,
    )
    length_before = _line_length(base_points)
    directions = _candidate_directions(base_points[0], base_points[-1], bone_centroid, body_centroid)
    detours = np.arange(0.0, max(float(max_detour_mm), 0.0) + max(detour_step_mm, 1.0), max(detour_step_mm, 1.0))
    candidate_rows: list[dict[str, Any]] = []
    candidate_points: dict[str, np.ndarray] = {}

    t = np.linspace(0.0, 1.0, len(base_points))
    envelope = np.sin(np.pi * t) ** 1.15
    for direction_name, direction in directions:
        for detour in detours:
            candidate_id = f"{direction_name}_{detour:.1f}mm".replace(".", "p")
            points = base_points + direction.reshape(1, 3) * float(detour) * envelope.reshape(-1, 1)
            points[0] = base_points[0]
            points[-1] = base_points[-1]
            points = _clean_candidate_with_signed_field(
                points,
                spacing_mm=spacing,
                signed=signed,
                gradients=gradients,
                body_mask=body,
                bone_centroid_mm=bone_centroid,
                body_centroid_mm=body_centroid,
                clearance_mm=clearance_mm,
                max_point_shift_mm=max_point_shift_mm,
                smooth_iterations=smooth_iterations,
            )
            metrics = _metrics_for_points(
                points,
                template_edge=original_edge,
                spacing_mm=spacing,
                shape=labels.shape,
                body_mask=body,
                bone_mask=bone,
                signed_bone_mm=signed,
                sample_step_mm=sample_step_mm,
            )
            length = _line_length(points)
            length_ratio = length / max(length_before, 1e-6)
            clearance_penalty = max(0.0, float(clearance_mm) - metrics["min_signed_bone_distance_mm"])
            score = (
                metrics["outside_body_fraction"] * 10000.0
                + metrics["bone_fraction"] * 10000.0
                + clearance_penalty * 45.0
                + max(0.0, length_ratio - 1.0) * 35.0
                + abs(float(detour)) * 0.015
            )
            candidate_points[candidate_id] = points
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "direction": direction_name,
                    "detour_mm": f"{float(detour):.6f}",
                    "score": f"{score:.9f}",
                    "selected": 0,
                    "sample_count": f"{metrics['sample_count']:.0f}",
                    "outside_body_fraction": f"{metrics['outside_body_fraction']:.8f}",
                    "bone_fraction": f"{metrics['bone_fraction']:.8f}",
                    "min_signed_bone_distance_mm": f"{metrics['min_signed_bone_distance_mm']:.6f}",
                    "mean_signed_bone_distance_mm": f"{metrics['mean_signed_bone_distance_mm']:.6f}",
                    "length_mm": f"{length:.6f}",
                    "length_ratio": f"{length_ratio:.8f}",
                }
            )

    selected = min(candidate_rows, key=lambda row: float(row["score"]))
    selected["selected"] = 1
    selected_id = str(selected["candidate_id"])
    selected_points = candidate_points[selected_id]
    after_metrics = _metrics_for_points(
        selected_points,
        template_edge=original_edge,
        spacing_mm=spacing,
        shape=labels.shape,
        body_mask=body,
        bone_mask=bone,
        signed_bone_mm=signed,
        sample_step_mm=sample_step_mm,
    )
    length_after = _line_length(selected_points)

    corrected_graph = dict(graph)
    corrected_edges = [dict(edge) for edge in graph.get("edges", [])]
    corrected_edge = dict(corrected_edges[edge_index])
    corrected_edge["polyline_mm"] = [[float(value) for value in point] for point in selected_points]
    corrected_edge["length_mm"] = float(length_after)
    notes = list(corrected_edge.get("notes", []))
    notes.append("polyline_rerouted_by_targeted_bone_avoidance")
    corrected_edge["notes"] = sorted(set(str(note) for note in notes))
    corrected_edges[edge_index] = corrected_edge
    corrected_graph["edges"] = corrected_edges
    metadata = dict(corrected_graph.get("graph_metadata", {}))
    metadata.update(
        {
            "targeted_edge_reroute": {
                "edge_id": edge_id,
                "source_graph": str(graph_path),
                "source_anatomy_labels": str(labels_path),
                "selected_candidate_id": selected_id,
                "selected_detour_mm": float(selected["detour_mm"]),
                "bone_fraction_before": before_metrics["bone_fraction"],
                "bone_fraction_after": after_metrics["bone_fraction"],
            }
        }
    )
    corrected_graph["graph_metadata"] = metadata
    provenance = list(corrected_graph.get("provenance_notes", []))
    provenance.append("targeted_vessel_edge_bone_avoidance_reroute_applied")
    corrected_graph["provenance_notes"] = sorted(set(str(note) for note in provenance))
    corrected_graph["case_id"] = case_id

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corrected_graph_yaml = output / f"{case_id}_{edge_id}_rerouted_vascular_graph_v001.yaml"
    candidate_csv = output / f"{case_id}_{edge_id}_reroute_candidates_v001.csv"
    preview_png = output / f"{case_id}_{edge_id}_reroute_preview_v001.png"
    spec_yaml = output / f"{case_id}_{edge_id}_reroute_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_{edge_id}_reroute_report_v001.md"

    corrected_graph_yaml.write_text(yaml.safe_dump(corrected_graph, sort_keys=False))
    _write_candidate_metrics(candidate_csv, candidate_rows)
    _render_preview(
        preview_png,
        labels=labels,
        spacing_mm=spacing,
        edge_id=edge_id,
        original_points=base_points,
        rerouted_points=selected_points,
    )

    improved_bone = after_metrics["bone_fraction"] < before_metrics["bone_fraction"]
    no_outside_body = after_metrics["outside_body_fraction"] <= 0.02
    status = "improved" if improved_bone and no_outside_body else "review_required"
    if after_metrics["bone_fraction"] <= 0.05 and no_outside_body:
        status = "qa_candidate"
    notes_tuple = (
        "source_target_nodes_vessel_type_flow_role_and_radii_preserved",
        "only_selected_edge_polyline_and_length_modified",
        "reroute_requires_downstream_voxelization_and_vessel_qa",
    )
    result = VesselEdgeRerouteResult(
        case_id=case_id,
        output_dir=str(output),
        source_graph_path=str(graph_path),
        anatomy_labels_path=str(labels_path),
        edge_id=edge_id,
        corrected_graph_yaml_path=str(corrected_graph_yaml),
        candidate_metrics_csv_path=str(candidate_csv),
        preview_png_path=str(preview_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        status=status,
        bone_fraction_before=before_metrics["bone_fraction"],
        bone_fraction_after=after_metrics["bone_fraction"],
        outside_body_fraction_before=before_metrics["outside_body_fraction"],
        outside_body_fraction_after=after_metrics["outside_body_fraction"],
        min_signed_bone_distance_before_mm=before_metrics["min_signed_bone_distance_mm"],
        min_signed_bone_distance_after_mm=after_metrics["min_signed_bone_distance_mm"],
        length_before_mm=length_before,
        length_after_mm=length_after,
        selected_candidate_id=selected_id,
        selected_detour_mm=float(selected["detour_mm"]),
        notes=notes_tuple,
    )
    params = {
        "clearance_mm": float(clearance_mm),
        "max_detour_mm": float(max_detour_mm),
        "detour_step_mm": float(detour_step_mm),
        "sample_step_mm": float(sample_step_mm),
        "resample_step_mm": float(resample_step_mm),
        "max_point_shift_mm": float(max_point_shift_mm),
        "smooth_iterations": int(smooth_iterations),
    }
    _write_spec(spec_yaml, result, params)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, candidate_rows) + "\n")
    return result


def format_vessel_edge_reroute_result(result: VesselEdgeRerouteResult) -> str:
    return "\n".join(
        [
            "Targeted vessel edge reroute completed",
            f"Case ID: {result.case_id}",
            f"Edge ID: {result.edge_id}",
            f"Status: {result.status}",
            f"Bone fraction before/after: {result.bone_fraction_before * 100.0:.2f}%/{result.bone_fraction_after * 100.0:.2f}%",
            f"Outside-body before/after: {result.outside_body_fraction_before * 100.0:.2f}%/{result.outside_body_fraction_after * 100.0:.2f}%",
            f"Length before/after: {result.length_before_mm:.2f}/{result.length_after_mm:.2f} mm",
            f"Corrected graph YAML: {result.corrected_graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
