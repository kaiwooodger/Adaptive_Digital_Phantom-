from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np

from .vessel_radius_profile import edge_radius_at_fraction, edge_radius_profile_max
from .vessel_anatomy_validation import (
    _import_dependencies,
    _load_yaml,
    _mask_for_group,
    _points_to_indices,
    _resolve_path,
    _sample_mask_fraction,
    _spacing_from_image,
)


@dataclass(frozen=True)
class VesselRadiusValidationResult:
    case_id: str
    output_dir: str
    graph_yaml_path: str
    anatomy_labels_path: str
    voxelized_spec_path: str
    edge_metrics_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    edge_count: int
    pass_count: int
    review_count: int
    fail_count: int
    radius_tuning_candidate_count: int
    reroute_candidate_count: int
    preserve_candidate_count: int
    notes: tuple[str, ...]


def _edge_samples_with_radius(edge: dict[str, Any], sample_step_mm: float, radius_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(edge.get("polyline_mm", []), dtype=float)
    if points.ndim != 2 or len(points) < 2:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-6:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    step = max(float(sample_step_mm), 0.25)
    samples: list[np.ndarray] = []
    radii: list[float] = []
    running = 0.0
    for segment_index, length in enumerate(segment_lengths):
        if length <= 1e-6:
            continue
        count = max(1, int(np.ceil(float(length) / step)))
        for local_index in range(count + 1):
            if segment_index > 0 and local_index == 0:
                continue
            local_t = local_index / count
            edge_t = (running + local_t * float(length)) / total_length
            point = points[segment_index] + (points[segment_index + 1] - points[segment_index]) * local_t
            radius = edge_radius_at_fraction(edge, edge_t, radius_scale=radius_scale)
            samples.append(point.astype(float))
            radii.append(float(max(radius, 0.05)))
        running += float(length)
    return np.asarray(samples, dtype=float), np.asarray(radii, dtype=float)


def _paint_sphere(mask: np.ndarray, center_mm: np.ndarray, radius_mm: float, spacing_mm: tuple[float, float, float]) -> None:
    spacing = np.asarray(spacing_mm, dtype=float)
    center_index = center_mm / spacing
    radius_index = np.ceil(radius_mm / spacing).astype(int) + 1
    mins = np.maximum(np.floor(center_index).astype(int) - radius_index, 0)
    maxs = np.minimum(np.ceil(center_index).astype(int) + radius_index + 1, np.asarray(mask.shape, dtype=int))
    if np.any(maxs <= mins):
        return
    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    grids = np.meshgrid(
        *[np.arange(mins[axis], maxs[axis], dtype=float) * spacing[axis] for axis in range(3)],
        indexing="ij",
    )
    distance_sq = sum((grid - float(center_mm[axis])) ** 2 for axis, grid in enumerate(grids))
    mask[slices] |= distance_sq <= float(radius_mm) ** 2


def _edge_tube_mask(
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    edge: dict[str, Any],
    sample_step_mm: float,
    radius_scale: float = 1.0,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(shape, dtype=bool)
    samples, radii = _edge_samples_with_radius(edge, sample_step_mm=sample_step_mm, radius_scale=radius_scale)
    for center, radius in zip(samples, radii):
        _paint_sphere(mask, center, float(radius), spacing_mm)
    return mask, int(len(samples))


def _centerline_bone_fraction(edge: dict[str, Any], spacing_mm: tuple[float, float, float], bone: np.ndarray, sample_step_mm: float) -> float:
    samples, _ = _edge_samples_with_radius(edge, sample_step_mm=sample_step_mm, radius_scale=0.0)
    indices, valid = _points_to_indices(samples, spacing_mm, bone.shape)
    return _sample_mask_fraction(bone, indices, valid)


def _recommendation(
    *,
    centerline_bone_fraction: float,
    lumen_bone_fraction: float,
    scaled_lumen_bone_fraction: float,
    review_threshold: float,
    fail_threshold: float,
    edge_id: str,
    vessel_type: str,
) -> tuple[str, str]:
    if lumen_bone_fraction <= review_threshold:
        return "pass", "within_radius_aware_bone_overlap_gate"
    if centerline_bone_fraction > 0.05:
        status = "fail" if lumen_bone_fraction > fail_threshold else "review"
        return status, "reroute_centerline_before_radius_tuning"
    if scaled_lumen_bone_fraction <= review_threshold:
        return "review", "radius_tuning_candidate"
    lowered = edge_id.lower()
    if vessel_type == "arterial" and ("aorta" in lowered or "visceral_to_renal" in lowered):
        return "review", "large_trunk_overlap_preserve_or_reroute_decision_required"
    return "review", "radius_tuning_insufficient_consider_reroute_or_accept_stress_test"


def _write_edge_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "edge_id",
        "vessel_type",
        "flow_role",
        "radius_start_mm",
        "radius_end_mm",
        "radius_max_mm",
        "length_mm",
        "sample_count",
        "lumen_voxels",
        "lumen_cm3",
        "outside_body_fraction",
        "centerline_bone_fraction",
        "lumen_bone_fraction",
        "lumen_bone_cm3",
        "scaled_radius_factor",
        "scaled_lumen_bone_fraction",
        "scaled_lumen_bone_cm3",
        "liver_overlap_fraction",
        "kidney_overlap_fraction",
        "lung_overlap_fraction",
        "status",
        "recommendation",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_spec(
    path: Path,
    result: VesselRadiusValidationResult,
    sample_step_mm: float,
    scaled_radius_factor: float,
    review_threshold: float,
    fail_threshold: float,
) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "validation_type": "vessel_radius_aware_anatomy_validation",
        "source_voxelized_spec": result.voxelized_spec_path,
        "source_graph": result.graph_yaml_path,
        "source_anatomy_labels": result.anatomy_labels_path,
        "sample_step_mm": float(sample_step_mm),
        "scaled_radius_factor": float(scaled_radius_factor),
        "review_lumen_bone_fraction": float(review_threshold),
        "fail_lumen_bone_fraction": float(fail_threshold),
        "summary": {
            "edge_count": result.edge_count,
            "pass_count": result.pass_count,
            "review_count": result.review_count,
            "fail_count": result.fail_count,
            "radius_tuning_candidate_count": result.radius_tuning_candidate_count,
            "reroute_candidate_count": result.reroute_candidate_count,
            "preserve_candidate_count": result.preserve_candidate_count,
        },
        "outputs": {
            "edge_metrics_csv": result.edge_metrics_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _render_preview(path: Path, rows: list[dict[str, Any]]) -> None:
    plt, *_ = _import_dependencies()
    top = sorted(rows, key=lambda row: float(row["lumen_bone_fraction"]), reverse=True)[:12]
    labels = [row["edge_id"] for row in top]
    full = [float(row["lumen_bone_fraction"]) * 100.0 for row in top]
    scaled = [float(row["scaled_lumen_bone_fraction"]) * 100.0 for row in top]
    center = [float(row["centerline_bone_fraction"]) * 100.0 for row in top]
    colors = ["#dc2626" if row["status"] == "fail" else "#f59e0b" if row["status"] == "review" else "#16a34a" for row in top]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=170)
    fig.patch.set_facecolor("#f8f3e8")
    for ax in axes:
        ax.set_facecolor("#f8f3e8")
    y = np.arange(len(top))
    axes[0].barh(y, full, color=colors, alpha=0.84, label="current radius")
    axes[0].barh(y, scaled, color="#2563eb", alpha=0.45, label="scaled-radius test")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Lumen volume overlapping bone (%)")
    axes[0].set_title("Radius-Aware Bone Overlap")
    axes[0].legend(fontsize=8)

    axes[1].scatter(center, full, c=colors, s=60, edgecolor="#111827", linewidth=0.4)
    for row, x, y_value in zip(top, center, full):
        axes[1].annotate(row["edge_id"], (x, y_value), fontsize=6, alpha=0.8)
    axes[1].set_xlabel("Centerline samples in bone (%)")
    axes[1].set_ylabel("Lumen volume in bone (%)")
    axes[1].set_title("Centerline vs Tube-Volume Conflict")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Vessel Radius-Aware Anatomy Validation", fontsize=15, color="#111827")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _format_report(result: VesselRadiusValidationResult, rows: list[dict[str, Any]]) -> str:
    review_rows = [row for row in rows if row["status"] != "pass"]
    lines = [
        "# Vessel Radius-Aware Anatomy Validation",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Edges evaluated: {result.edge_count}",
        f"- Pass / review / fail: {result.pass_count} / {result.review_count} / {result.fail_count}",
        f"- Radius-tuning candidates: {result.radius_tuning_candidate_count}",
        f"- Reroute candidates: {result.reroute_candidate_count}",
        f"- Preserve/stress-test decision candidates: {result.preserve_candidate_count}",
        "",
        "## Outputs",
        "",
        f"- Edge metrics CSV: `{Path(result.edge_metrics_csv_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Spec YAML: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Radius-Aware Review Edges",
        "",
        "| edge | status | lumen in bone % | scaled-radius bone % | centerline bone % | recommendation |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(review_rows, key=lambda item: float(item["lumen_bone_fraction"]), reverse=True)[:14]:
        lines.append(
            f"| `{row['edge_id']}` | `{row['status']}` | "
            f"{float(row['lumen_bone_fraction']) * 100.0:.2f} | "
            f"{float(row['scaled_lumen_bone_fraction']) * 100.0:.2f} | "
            f"{float(row['centerline_bone_fraction']) * 100.0:.2f} | {row['recommendation']} |"
        )
    if not review_rows:
        lines.append("| none | pass | 0.00 | 0.00 | 0.00 | no review items |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This QA paints each graph edge as a tube using its radius_start/radius_end values or optional radius_profile, so it evaluates lumen volume clearance rather than centerline-only clearance.",
            "- The scaled-radius column is a diagnostic what-if test; it does not change the graph.",
            "- If centerline overlap remains high, rerouting is favored before radius tuning.",
            "- If centerline overlap is low but tube-volume overlap remains high, local radius tuning or an explicit stress-test acceptance decision is the safer next choice.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def validate_vessel_radius_anatomy(
    voxelized_spec_path: str | Path,
    graph_yaml_path: str | Path | None = None,
    anatomy_labels_path: str | Path | None = None,
    output_dir: str | Path = "outputs/validation/vessel_radius_anatomy",
    case_id: str = "ct_org_vessel_radius_validation",
    sample_step_mm: float = 2.0,
    scaled_radius_factor: float = 0.75,
    review_lumen_bone_fraction: float = 0.10,
    fail_lumen_bone_fraction: float = 0.35,
    report_path: str | Path | None = "outputs/reports/vessel_radius_anatomy_validation_stage001.md",
) -> VesselRadiusValidationResult:
    _, nib, _, _ = _import_dependencies()
    spec_path = Path(voxelized_spec_path)
    spec = _load_yaml(spec_path)
    voxelization = spec.get("voxelization", {})
    graph_path = _resolve_path(graph_yaml_path or voxelization.get("source_graph"), spec_path)
    labels_path = _resolve_path(anatomy_labels_path or voxelization.get("source_combined_labels"), spec_path)
    if graph_path is None or labels_path is None:
        raise ValueError("Voxelized spec must provide source graph and source anatomy labels")
    graph = _load_yaml(graph_path)
    labels_image = nib.load(str(labels_path))
    labels = np.rint(np.asanyarray(labels_image.dataobj)).astype(np.int16)
    spacing = _spacing_from_image(labels_image)
    voxel_volume_cm3 = float(np.prod(np.asarray(spacing, dtype=float)) / 1000.0)
    body = _mask_for_group(labels, "body")
    bone = _mask_for_group(labels, "bone")
    liver = _mask_for_group(labels, "liver")
    kidneys = _mask_for_group(labels, "kidneys")
    lungs = _mask_for_group(labels, "lungs")

    rows: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        tube, sample_count = _edge_tube_mask(labels.shape, spacing, edge, sample_step_mm, radius_scale=1.0)
        scaled_tube, _ = _edge_tube_mask(labels.shape, spacing, edge, sample_step_mm, radius_scale=scaled_radius_factor)
        lumen_voxels = int(np.count_nonzero(tube))
        scaled_voxels = int(np.count_nonzero(scaled_tube))
        bone_voxels = int(np.count_nonzero(tube & bone))
        scaled_bone_voxels = int(np.count_nonzero(scaled_tube & bone))
        outside_body_voxels = int(np.count_nonzero(tube & ~body))
        centerline_bone = _centerline_bone_fraction(edge, spacing, bone, sample_step_mm)
        lumen_bone_fraction = bone_voxels / lumen_voxels if lumen_voxels else 0.0
        scaled_lumen_bone_fraction = scaled_bone_voxels / scaled_voxels if scaled_voxels else 0.0
        status, recommendation = _recommendation(
            centerline_bone_fraction=centerline_bone,
            lumen_bone_fraction=lumen_bone_fraction,
            scaled_lumen_bone_fraction=scaled_lumen_bone_fraction,
            review_threshold=review_lumen_bone_fraction,
            fail_threshold=fail_lumen_bone_fraction,
            edge_id=str(edge.get("id", "")),
            vessel_type=str(edge.get("vessel_type", "")),
        )
        row = {
            "edge_id": str(edge.get("id", "")),
            "vessel_type": str(edge.get("vessel_type", "")),
            "flow_role": str(edge.get("flow_role", "")),
            "radius_start_mm": f"{float(edge.get('radius_start_mm', 0.0)):.6f}",
            "radius_end_mm": f"{float(edge.get('radius_end_mm', 0.0)):.6f}",
            "radius_max_mm": f"{edge_radius_profile_max(edge):.6f}",
            "length_mm": f"{float(edge.get('length_mm', 0.0)):.6f}",
            "sample_count": sample_count,
            "lumen_voxels": lumen_voxels,
            "lumen_cm3": f"{lumen_voxels * voxel_volume_cm3:.6f}",
            "outside_body_fraction": f"{(outside_body_voxels / lumen_voxels) if lumen_voxels else 0.0:.8f}",
            "centerline_bone_fraction": f"{centerline_bone:.8f}",
            "lumen_bone_fraction": f"{lumen_bone_fraction:.8f}",
            "lumen_bone_cm3": f"{bone_voxels * voxel_volume_cm3:.6f}",
            "scaled_radius_factor": f"{float(scaled_radius_factor):.6f}",
            "scaled_lumen_bone_fraction": f"{scaled_lumen_bone_fraction:.8f}",
            "scaled_lumen_bone_cm3": f"{scaled_bone_voxels * voxel_volume_cm3:.6f}",
            "liver_overlap_fraction": f"{(np.count_nonzero(tube & liver) / lumen_voxels) if lumen_voxels else 0.0:.8f}",
            "kidney_overlap_fraction": f"{(np.count_nonzero(tube & kidneys) / lumen_voxels) if lumen_voxels else 0.0:.8f}",
            "lung_overlap_fraction": f"{(np.count_nonzero(tube & lungs) / lumen_voxels) if lumen_voxels else 0.0:.8f}",
            "status": status,
            "recommendation": recommendation,
        }
        rows.append(row)

    pass_count = sum(1 for row in rows if row["status"] == "pass")
    review_count = sum(1 for row in rows if row["status"] == "review")
    fail_count = sum(1 for row in rows if row["status"] == "fail")
    radius_tuning_count = sum(1 for row in rows if "radius_tuning" in row["recommendation"])
    reroute_count = sum(1 for row in rows if "reroute" in row["recommendation"])
    preserve_count = sum(1 for row in rows if "preserve" in row["recommendation"] or "stress_test" in row["recommendation"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    edge_csv = output / f"{case_id}_vessel_radius_edge_metrics_v001.csv"
    preview = output / f"{case_id}_vessel_radius_validation_preview_v001.png"
    spec_out = output / f"{case_id}_vessel_radius_validation_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_vessel_radius_validation_report_v001.md"
    result = VesselRadiusValidationResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_path),
        anatomy_labels_path=str(labels_path),
        voxelized_spec_path=str(spec_path),
        edge_metrics_csv_path=str(edge_csv),
        preview_png_path=str(preview),
        spec_yaml_path=str(spec_out),
        report_path=str(report),
        edge_count=len(rows),
        pass_count=pass_count,
        review_count=review_count,
        fail_count=fail_count,
        radius_tuning_candidate_count=radius_tuning_count,
        reroute_candidate_count=reroute_count,
        preserve_candidate_count=preserve_count,
        notes=(
            "edge_lumen_tubes_painted_from_graph_radius_start_end_or_radius_profile_values",
            "scaled_radius_test_is_diagnostic_and_does_not_modify_graph",
            "recommendations_are_engineering_triage_not_clinical_anatomy_acceptance",
        ),
    )
    _write_edge_metrics(edge_csv, rows)
    _render_preview(preview, rows)
    _write_spec(
        spec_out,
        result,
        sample_step_mm=sample_step_mm,
        scaled_radius_factor=scaled_radius_factor,
        review_threshold=review_lumen_bone_fraction,
        fail_threshold=fail_lumen_bone_fraction,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, rows) + "\n")
    return result


def format_vessel_radius_validation_result(result: VesselRadiusValidationResult) -> str:
    return "\n".join(
        [
            "Vessel radius-aware anatomy validation completed",
            f"Case ID: {result.case_id}",
            f"Edges pass/review/fail: {result.pass_count}/{result.review_count}/{result.fail_count}",
            f"Radius-tuning/reroute/preserve candidates: {result.radius_tuning_candidate_count}/{result.reroute_candidate_count}/{result.preserve_candidate_count}",
            f"Edge metrics CSV: {result.edge_metrics_csv_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
