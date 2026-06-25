from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np

from .cta_vascular_graph import _line_length
from .vessel_anatomy_correction import _signed_bone_field
from .vessel_anatomy_validation import _import_dependencies, _load_yaml, _mask_for_group, _points_to_indices, _spacing_from_image
from .vessel_radius_profile import edge_radius_at_fraction, edge_radius_profile_max


@dataclass(frozen=True)
class VesselRadiusTuningResult:
    case_id: str
    output_dir: str
    source_graph_path: str
    anatomy_labels_path: str
    source_radius_metrics_csv_path: str | None
    tuned_graph_yaml_path: str
    tuning_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    edge_count: int
    tuned_edge_count: int
    mean_radius_reduction_percent: float
    max_radius_reduction_percent: float
    notes: tuple[str, ...]


def _read_review_edges(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    edge_ids: set[str] = set()
    with Path(path).open(newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            if row.get("status") != "pass":
                edge_ids.add(str(row.get("edge_id", "")))
    return edge_ids


def _edge_radius_bounds(
    edge: dict[str, Any],
    *,
    min_radius_mm: float,
    branch_max_radius_mm: float,
    arterial_trunk_min_radius_mm: float,
    arterial_trunk_max_radius_mm: float,
    venous_trunk_min_radius_mm: float,
    venous_trunk_max_radius_mm: float,
) -> tuple[float, float, str]:
    edge_id = str(edge.get("id", "")).lower()
    flow_role = str(edge.get("flow_role", "")).lower()
    vessel_type = str(edge.get("vessel_type", "")).lower()
    descriptor = f"{edge_id} {flow_role}"
    if vessel_type == "arterial" and ("aorta" in descriptor or "aorta_trunk" in descriptor or "visceral_to_renal" in descriptor):
        return max(min_radius_mm, arterial_trunk_min_radius_mm), arterial_trunk_max_radius_mm, "arterial_trunk"
    if vessel_type == "venous" and ("ivc" in descriptor or "venous_return_trunk" in descriptor):
        return max(min_radius_mm, venous_trunk_min_radius_mm), venous_trunk_max_radius_mm, "venous_trunk"
    return min_radius_mm, branch_max_radius_mm, "branch"


def _edge_station_samples(edge: dict[str, Any], sample_step_mm: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(edge.get("polyline_mm", []), dtype=float)
    if points.ndim != 2 or len(points) < 2:
        return np.empty((0,), dtype=float), np.empty((0, 3), dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(segment_lengths.sum())
    if total <= 1e-6:
        return np.empty((0,), dtype=float), np.empty((0, 3), dtype=float)
    stations: list[float] = []
    samples: list[np.ndarray] = []
    running = 0.0
    step = max(sample_step_mm, 0.25)
    for segment_index, length in enumerate(segment_lengths):
        if length <= 1e-6:
            continue
        count = max(1, int(np.ceil(float(length) / step)))
        for local_index in range(count + 1):
            if segment_index > 0 and local_index == 0:
                continue
            local_t = local_index / count
            station = (running + local_t * float(length)) / total
            point = points[segment_index] + (points[segment_index + 1] - points[segment_index]) * local_t
            stations.append(float(station))
            samples.append(point.astype(float))
        running += float(length)
    return np.asarray(stations, dtype=float), np.asarray(samples, dtype=float)


def _sample_signed_bone_distance(
    points_mm: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    signed_bone_mm: np.ndarray,
) -> np.ndarray:
    indices, valid = _points_to_indices(points_mm, spacing_mm, signed_bone_mm.shape)
    values = np.full((len(points_mm),), -np.inf, dtype=float)
    if np.any(valid):
        valid_indices = indices[valid]
        values[valid] = signed_bone_mm[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
    return values


def _smooth_radius_profile(values: np.ndarray, upper_bounds: np.ndarray, lower_bound: float, iterations: int) -> np.ndarray:
    if len(values) <= 2 or iterations <= 0:
        return np.minimum(np.maximum(values, lower_bound), upper_bounds)
    smoothed = values.astype(float).copy()
    for _ in range(iterations):
        updated = smoothed.copy()
        updated[1:-1] = 0.25 * smoothed[:-2] + 0.50 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed = np.minimum(np.maximum(updated, lower_bound), upper_bounds)
    return smoothed


def _profile_payload(stations: np.ndarray, radii: np.ndarray, max_profile_points: int) -> list[dict[str, float]]:
    if len(stations) == 0:
        return []
    count = min(max(int(max_profile_points), 2), len(stations))
    target_stations = np.linspace(0.0, 1.0, count)
    target_radii = np.interp(target_stations, stations, radii)
    return [
        {"station": float(round(station, 6)), "radius_mm": float(round(radius, 6))}
        for station, radius in zip(target_stations, target_radii)
    ]


def _write_tuning_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "edge_id",
        "tuned",
        "radius_class",
        "sample_count",
        "profile_point_count",
        "radius_min_before_mm",
        "radius_mean_before_mm",
        "radius_max_before_mm",
        "radius_min_after_mm",
        "radius_mean_after_mm",
        "radius_max_after_mm",
        "mean_reduction_percent",
        "max_reduction_percent",
        "min_signed_bone_distance_mm",
        "bone_clearance_mm",
        "lower_bound_mm",
        "upper_bound_mm",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _render_preview(path: Path, rows: list[dict[str, Any]]) -> None:
    plt, *_ = _import_dependencies()
    tuned = [row for row in rows if int(row["tuned"]) == 1]
    top = sorted(tuned, key=lambda row: float(row["mean_reduction_percent"]), reverse=True)[:12]
    labels = [row["edge_id"] for row in top]
    before = [float(row["radius_mean_before_mm"]) for row in top]
    after = [float(row["radius_mean_after_mm"]) for row in top]
    reduction = [float(row["mean_reduction_percent"]) for row in top]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=170)
    fig.patch.set_facecolor("#f8f3e8")
    for ax in axes:
        ax.set_facecolor("#f8f3e8")
    if top:
        y = np.arange(len(top))
        axes[0].barh(y, before, color="#94a3b8", alpha=0.65, label="before")
        axes[0].barh(y, after, color="#dc2626", alpha=0.8, label="after")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Mean radius (mm)")
        axes[0].legend(fontsize=8)
        axes[1].barh(y, reduction, color="#f97316", alpha=0.82)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Mean radius reduction (%)")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No tuned edges", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
    axes[0].set_title("Radius Profiles Before/After")
    axes[1].set_title("Tuning Magnitude")
    fig.suptitle("Anatomy-Aware Vessel Radius Tuning", fontsize=15, color="#111827")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _format_report(result: VesselRadiusTuningResult, rows: list[dict[str, Any]]) -> str:
    tuned_rows = [row for row in rows if int(row["tuned"]) == 1]
    lines = [
        "# Anatomy-Aware Vessel Radius Tuning",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Edges evaluated: {result.edge_count}",
        f"- Tuned edges: {result.tuned_edge_count}",
        f"- Mean / max radius reduction: {result.mean_radius_reduction_percent:.2f}% / {result.max_radius_reduction_percent:.2f}%",
        "",
        "## Outputs",
        "",
        f"- Tuned graph YAML: `{Path(result.tuned_graph_yaml_path).name}`",
        f"- Tuning CSV: `{Path(result.tuning_csv_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Spec YAML: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Tuned Edges",
        "",
        "| edge | class | mean radius before/after mm | max radius before/after mm | mean reduction % | min bone distance mm |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(tuned_rows, key=lambda item: float(item["mean_reduction_percent"]), reverse=True):
        lines.append(
            f"| `{row['edge_id']}` | {row['radius_class']} | "
            f"{float(row['radius_mean_before_mm']):.2f} / {float(row['radius_mean_after_mm']):.2f} | "
            f"{float(row['radius_max_before_mm']):.2f} / {float(row['radius_max_after_mm']):.2f} | "
            f"{float(row['mean_reduction_percent']):.2f} | {float(row['min_signed_bone_distance_mm']):.2f} |"
        )
    if not tuned_rows:
        lines.append("| none | n/a | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 | 0.00 |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Source/target nodes, vessel types, flow roles, and centerline coordinates are preserved.",
            "- Tuned edges receive a per-edge `radius_profile` so local lumen size can change along the vessel.",
            "- Original radius fields are preserved in `radius_tuning.original_*` metadata for auditability.",
            "- This is an engineering geometry correction; downstream voxelization, radius-aware QA, flow, and RT QA must be rerun.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _write_spec(path: Path, result: VesselRadiusTuningResult, params: dict[str, Any]) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "correction_type": "anatomy_aware_vessel_radius_tuning",
        "source_graph": result.source_graph_path,
        "source_anatomy_labels": result.anatomy_labels_path,
        "source_radius_metrics_csv": result.source_radius_metrics_csv_path,
        "parameters": params,
        "summary": {
            "edge_count": result.edge_count,
            "tuned_edge_count": result.tuned_edge_count,
            "mean_radius_reduction_percent": result.mean_radius_reduction_percent,
            "max_radius_reduction_percent": result.max_radius_reduction_percent,
        },
        "outputs": {
            "tuned_graph": result.tuned_graph_yaml_path,
            "tuning_csv": result.tuning_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def tune_vessel_radii_against_bone(
    graph_yaml_path: str | Path,
    anatomy_labels_path: str | Path,
    radius_metrics_csv_path: str | Path | None = None,
    output_dir: str | Path = "outputs/digital/vessel_radius_tuned",
    case_id: str = "vessel_radius_tuned",
    edge_ids: tuple[str, ...] = (),
    tune_review_edges_only: bool = True,
    bone_clearance_mm: float = 0.5,
    sample_step_mm: float = 2.0,
    max_profile_points: int = 56,
    smooth_iterations: int = 2,
    min_radius_mm: float = 1.5,
    branch_max_radius_mm: float = 8.0,
    arterial_trunk_min_radius_mm: float = 4.0,
    arterial_trunk_max_radius_mm: float = 18.0,
    venous_trunk_min_radius_mm: float = 3.0,
    venous_trunk_max_radius_mm: float = 12.0,
    report_path: str | Path | None = "outputs/reports/vessel_radius_tuning_stage001.md",
) -> VesselRadiusTuningResult:
    _, nib, ndimage, yaml = _import_dependencies()
    graph_path = Path(graph_yaml_path)
    labels_path = Path(anatomy_labels_path)
    graph = _load_yaml(graph_path)
    labels_image = nib.load(str(labels_path))
    labels = np.rint(np.asanyarray(labels_image.dataobj)).astype(np.int16)
    spacing = _spacing_from_image(labels_image)
    bone = _mask_for_group(labels, "bone")
    if not np.any(bone):
        raise ValueError("Anatomy labels do not contain bone labels")
    signed, _ = _signed_bone_field(bone, spacing, ndimage)
    review_edges = _read_review_edges(radius_metrics_csv_path) if tune_review_edges_only else None
    explicit_edges = set(edge_ids)

    tuned_graph = dict(graph)
    tuned_edges: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    reduction_values: list[float] = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            tuned_edges.append(edge)
            continue
        edge_id = str(edge.get("id", ""))
        should_tune = bool(explicit_edges and edge_id in explicit_edges)
        if not explicit_edges:
            should_tune = review_edges is None or edge_id in review_edges
        if review_edges is not None and explicit_edges and edge_id not in explicit_edges:
            should_tune = False

        stations, samples = _edge_station_samples(edge, sample_step_mm=sample_step_mm)
        lower_bound, upper_bound, radius_class = _edge_radius_bounds(
            edge,
            min_radius_mm=min_radius_mm,
            branch_max_radius_mm=branch_max_radius_mm,
            arterial_trunk_min_radius_mm=arterial_trunk_min_radius_mm,
            arterial_trunk_max_radius_mm=arterial_trunk_max_radius_mm,
            venous_trunk_min_radius_mm=venous_trunk_min_radius_mm,
            venous_trunk_max_radius_mm=venous_trunk_max_radius_mm,
        )
        if len(stations) == 0:
            tuned_edges.append(dict(edge))
            continue

        original_radii = np.asarray([edge_radius_at_fraction(edge, float(station)) for station in stations], dtype=float)
        signed_values = _sample_signed_bone_distance(samples, spacing_mm=spacing, signed_bone_mm=signed)
        safe_by_bone = np.maximum(lower_bound, signed_values - float(bone_clearance_mm))
        upper_bounds = np.minimum(upper_bound, safe_by_bone)
        upper_bounds = np.maximum(upper_bounds, lower_bound)
        tuned_radii = np.minimum(original_radii, upper_bounds)
        tuned_radii = _smooth_radius_profile(tuned_radii, upper_bounds, lower_bound, smooth_iterations)

        tuned = should_tune and bool(np.max(np.abs(tuned_radii - original_radii)) > 1e-6)
        tuned_edge = dict(edge)
        if tuned:
            profile = _profile_payload(stations, tuned_radii, max_profile_points=max_profile_points)
            tuned_edge["radius_profile"] = profile
            tuned_edge["radius_start_mm"] = float(profile[0]["radius_mm"])
            tuned_edge["radius_end_mm"] = float(profile[-1]["radius_mm"])
            notes = list(tuned_edge.get("notes", []))
            notes.append("radius_profile_tuned_by_anatomy_aware_bone_clearance")
            tuned_edge["notes"] = sorted(set(str(note) for note in notes))
            existing_tuning = dict(tuned_edge.get("radius_tuning", {}))
            existing_tuning.update(
                {
                    "method": "anatomy_aware_bone_clearance_radius_profile",
                    "source_graph": str(graph_path),
                    "source_anatomy_labels": str(labels_path),
                    "source_radius_metrics_csv": str(radius_metrics_csv_path) if radius_metrics_csv_path is not None else None,
                    "original_radius_start_mm": float(edge.get("radius_start_mm", edge_radius_at_fraction(edge, 0.0))),
                    "original_radius_end_mm": float(edge.get("radius_end_mm", edge_radius_at_fraction(edge, 1.0))),
                    "original_radius_max_mm": float(edge_radius_profile_max(edge)),
                    "bone_clearance_mm": float(bone_clearance_mm),
                    "lower_bound_mm": float(lower_bound),
                    "upper_bound_mm": float(upper_bound),
                }
            )
            tuned_edge["radius_tuning"] = existing_tuning
            mean_reduction = float((np.mean(original_radii) - np.mean(tuned_radii)) / max(np.mean(original_radii), 1e-9) * 100.0)
            max_reduction = float((np.max(original_radii) - np.min(tuned_radii)) / max(np.max(original_radii), 1e-9) * 100.0)
            reduction_values.append(mean_reduction)
        else:
            tuned_radii = original_radii
            mean_reduction = 0.0
            max_reduction = 0.0

        tuned_edges.append(tuned_edge)
        rows.append(
            {
                "edge_id": edge_id,
                "tuned": int(tuned),
                "radius_class": radius_class,
                "sample_count": int(len(stations)),
                "profile_point_count": len(tuned_edge.get("radius_profile", [])) if tuned else 0,
                "radius_min_before_mm": f"{float(np.min(original_radii)):.6f}",
                "radius_mean_before_mm": f"{float(np.mean(original_radii)):.6f}",
                "radius_max_before_mm": f"{float(np.max(original_radii)):.6f}",
                "radius_min_after_mm": f"{float(np.min(tuned_radii)):.6f}",
                "radius_mean_after_mm": f"{float(np.mean(tuned_radii)):.6f}",
                "radius_max_after_mm": f"{float(np.max(tuned_radii)):.6f}",
                "mean_reduction_percent": f"{mean_reduction:.6f}",
                "max_reduction_percent": f"{max_reduction:.6f}",
                "min_signed_bone_distance_mm": f"{float(np.min(signed_values)):.6f}",
                "bone_clearance_mm": f"{float(bone_clearance_mm):.6f}",
                "lower_bound_mm": f"{float(lower_bound):.6f}",
                "upper_bound_mm": f"{float(upper_bound):.6f}",
                "notes": "radius_profile_written" if tuned else "not_tuned",
            }
        )

    tuned_graph["edges"] = tuned_edges
    metadata = dict(tuned_graph.get("graph_metadata", {}))
    metadata["anatomy_aware_radius_tuning"] = {
        "source_graph": str(graph_path),
        "source_anatomy_labels": str(labels_path),
        "source_radius_metrics_csv": str(radius_metrics_csv_path) if radius_metrics_csv_path is not None else None,
        "tuned_edge_count": int(sum(1 for row in rows if int(row["tuned"]) == 1)),
        "bone_clearance_mm": float(bone_clearance_mm),
        "profile_support": "edge.radius_profile list of station/radius_mm",
    }
    tuned_graph["graph_metadata"] = metadata
    provenance = list(tuned_graph.get("provenance_notes", []))
    provenance.append("anatomy_aware_radius_tuning_applied")
    tuned_graph["provenance_notes"] = sorted(set(str(note) for note in provenance))
    tuned_graph["case_id"] = case_id

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tuned_graph_yaml = output / f"{case_id}_radius_tuned_vascular_graph_v001.yaml"
    tuning_csv = output / f"{case_id}_radius_tuning_metrics_v001.csv"
    preview_png = output / f"{case_id}_radius_tuning_preview_v001.png"
    spec_yaml = output / f"{case_id}_radius_tuning_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_radius_tuning_report_v001.md"

    tuned_graph_yaml.write_text(yaml.safe_dump(tuned_graph, sort_keys=False))
    _write_tuning_csv(tuning_csv, rows)
    _render_preview(preview_png, rows)

    tuned_edge_count = int(sum(1 for row in rows if int(row["tuned"]) == 1))
    result = VesselRadiusTuningResult(
        case_id=case_id,
        output_dir=str(output),
        source_graph_path=str(graph_path),
        anatomy_labels_path=str(labels_path),
        source_radius_metrics_csv_path=str(radius_metrics_csv_path) if radius_metrics_csv_path is not None else None,
        tuned_graph_yaml_path=str(tuned_graph_yaml),
        tuning_csv_path=str(tuning_csv),
        preview_png_path=str(preview_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        edge_count=len(rows),
        tuned_edge_count=tuned_edge_count,
        mean_radius_reduction_percent=float(np.mean(reduction_values)) if reduction_values else 0.0,
        max_radius_reduction_percent=float(np.max(reduction_values)) if reduction_values else 0.0,
        notes=(
            "centerlines_topology_flow_roles_and_boundary_ids_preserved",
            "tuned_edges_use_radius_profile_station_radius_mm",
            "downstream_voxelization_radius_qa_flow_and_rt_require_rerun",
        ),
    )
    params = {
        "tune_review_edges_only": bool(tune_review_edges_only),
        "edge_ids": list(edge_ids),
        "bone_clearance_mm": float(bone_clearance_mm),
        "sample_step_mm": float(sample_step_mm),
        "max_profile_points": int(max_profile_points),
        "smooth_iterations": int(smooth_iterations),
        "min_radius_mm": float(min_radius_mm),
        "branch_max_radius_mm": float(branch_max_radius_mm),
        "arterial_trunk_min_radius_mm": float(arterial_trunk_min_radius_mm),
        "arterial_trunk_max_radius_mm": float(arterial_trunk_max_radius_mm),
        "venous_trunk_min_radius_mm": float(venous_trunk_min_radius_mm),
        "venous_trunk_max_radius_mm": float(venous_trunk_max_radius_mm),
    }
    _write_spec(spec_yaml, result, params)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, rows) + "\n")
    return result


def format_vessel_radius_tuning_result(result: VesselRadiusTuningResult) -> str:
    return "\n".join(
        [
            "Anatomy-aware vessel radius tuning completed",
            f"Case ID: {result.case_id}",
            f"Tuned edges: {result.tuned_edge_count}/{result.edge_count}",
            f"Mean/max radius reduction: {result.mean_radius_reduction_percent:.2f}%/{result.max_radius_reduction_percent:.2f}%",
            f"Tuned graph YAML: {result.tuned_graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
