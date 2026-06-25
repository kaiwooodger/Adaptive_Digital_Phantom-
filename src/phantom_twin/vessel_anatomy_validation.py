from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from scipy import ndimage  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Vessel anatomy validation requires matplotlib, nibabel, scipy, and PyYAML.") from exc
    return plt, nib, ndimage, yaml


@dataclass(frozen=True)
class VesselAnatomyValidationResult:
    case_id: str
    output_dir: str
    graph_yaml_path: str
    anatomy_labels_path: str
    voxelized_spec_path: str
    edge_metrics_csv_path: str
    node_metrics_csv_path: str
    organ_stats_csv_path: str
    overlap_metrics_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    edge_count: int
    node_count: int
    pass_count: int
    review_count: int
    fail_count: int
    outside_body_edge_count: int
    bone_intersection_edge_count: int
    missing_expected_target_count: int
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(value: str | Path | None, anchor: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    anchored = anchor.parent / path
    if anchored.exists():
        return anchored
    return path


def _spacing_from_image(image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _mask_for_group(labels: np.ndarray, group_id: str) -> np.ndarray:
    if group_id == "body":
        return labels != 0
    if group_id == "liver":
        return labels == 6
    if group_id == "kidneys":
        return labels == 7
    if group_id in {"left_kidney", "right_kidney"}:
        kidneys = labels == 7
        coords = np.argwhere(kidneys)
        if len(coords) == 0:
            return kidneys
        split_x = float(np.median(coords[:, 0]))
        x_grid = np.indices(labels.shape)[0]
        return kidneys & (x_grid <= split_x if group_id == "left_kidney" else x_grid > split_x)
    if group_id == "lungs":
        return labels == 8
    if group_id == "bone":
        return np.isin(labels, (10, 11))
    if group_id == "vessel_wall":
        return labels == 13
    if group_id == "vascular_fluid":
        return np.isin(labels, (14, 15))
    raise ValueError(f"Unknown anatomy group: {group_id}")


def _mask_stats(group_id: str, mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> dict[str, Any]:
    coords = np.argwhere(mask)
    spacing = np.asarray(spacing_mm, dtype=float)
    voxel_volume_cm3 = float(np.prod(spacing) / 1000.0)
    if len(coords) == 0:
        zeros = (0.0, 0.0, 0.0)
        return {
            "group_id": group_id,
            "voxel_count": 0,
            "volume_cm3": 0.0,
            "centroid_mm": zeros,
            "bbox_min_mm": zeros,
            "bbox_max_mm": zeros,
            "extent_mm": zeros,
        }
    coords_mm = coords.astype(float) * spacing
    bbox_min = coords_mm.min(axis=0)
    bbox_max = coords_mm.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, spacing)
    centroid = coords_mm.mean(axis=0)
    return {
        "group_id": group_id,
        "voxel_count": int(len(coords)),
        "volume_cm3": float(len(coords) * voxel_volume_cm3),
        "centroid_mm": tuple(float(value) for value in centroid),
        "bbox_min_mm": tuple(float(value) for value in bbox_min),
        "bbox_max_mm": tuple(float(value) for value in bbox_max),
        "extent_mm": tuple(float(value) for value in extent),
    }


def _edge_samples(edge: dict[str, Any], sample_step_mm: float) -> np.ndarray:
    points = np.asarray(edge.get("polyline_mm", []), dtype=float)
    if points.ndim != 2 or len(points) < 2:
        return np.empty((0, 3), dtype=float)
    samples: list[np.ndarray] = []
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    step = max(float(sample_step_mm), 0.25)
    for segment_index, length in enumerate(segment_lengths):
        if length <= 1e-6:
            continue
        count = max(1, int(np.ceil(float(length) / step)))
        for local_index in range(count + 1):
            if segment_index > 0 and local_index == 0:
                continue
            t = local_index / count
            samples.append(points[segment_index] + (points[segment_index + 1] - points[segment_index]) * t)
    return np.asarray(samples, dtype=float)


def _points_to_indices(points_mm: np.ndarray, spacing_mm: tuple[float, float, float], shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    if len(points_mm) == 0:
        return np.empty((0, 3), dtype=int), np.empty((0,), dtype=bool)
    spacing = np.asarray(spacing_mm, dtype=float)
    indices = np.rint(points_mm / spacing).astype(int)
    shape_array = np.asarray(shape, dtype=int)
    valid = np.all((indices >= 0) & (indices < shape_array), axis=1)
    return indices, valid


def _sample_mask_fraction(mask: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> float:
    if len(indices) == 0 or not np.any(valid):
        return 0.0
    valid_indices = indices[valid]
    values = mask[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
    return float(np.count_nonzero(values) / len(values))


def _sample_distance(distance_mm: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    if len(indices) == 0 or not np.any(valid):
        return float("nan"), float("nan"), float("nan")
    valid_indices = indices[valid]
    values = distance_mm[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
    return float(np.min(values)), float(np.mean(values)), float(np.median(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _expected_target(edge_id: str, flow_role: str) -> tuple[str | None, float | None, str]:
    lowered = f"{edge_id} {flow_role}".lower()
    if "left_renal" in lowered:
        return "left_kidney", 35.0, "left renal vessel should approach the left kidney label"
    if "right_renal" in lowered:
        return "right_kidney", 35.0, "right renal vessel should approach the right kidney label"
    if "renal" in lowered:
        return "kidneys", 45.0, "renal-level trunk or junction should remain near kidney labels"
    if "hepatic" in lowered:
        return "liver", 40.0, "hepatic vessel should approach the liver label"
    if "visceral_origin_to_hepatic" in lowered:
        return "liver", 45.0, "hepatic visceral branch should approach the liver label"
    if "splenic" in lowered:
        return None, None, "spleen is not represented in the current CT-ORG material labels"
    if "iliac" in lowered:
        return "body", 0.0, "iliac branch should remain inside the lower body envelope"
    if "aorta" in lowered or "ivc" in lowered:
        return "body", 0.0, "major trunk should remain inside the body envelope"
    return None, None, "no organ-specific rule configured for this edge"


def _edge_status(row: dict[str, Any]) -> tuple[str, str]:
    notes: list[str] = []
    status = "pass"
    if _safe_float(row.get("outside_body_fraction", 0.0)) > 0.02:
        return "fail", "centerline_samples_outside_body"
    if _safe_float(row.get("inside_bone_fraction", 0.0)) > 0.05:
        status = "review"
        notes.append("centerline_intersects_bone_label")
    expected = row["expected_target"]
    threshold = row["expected_threshold_mm"]
    if expected and expected != "body" and threshold != "":
        distance = row.get(f"min_distance_to_{expected}_mm", "")
        if distance == "" or not np.isfinite(float(distance)):
            return "fail", f"missing_expected_target_mask={expected}"
        distance_value = float(distance)
        threshold_value = float(threshold)
        if distance_value > threshold_value * 1.8:
            return "fail", f"{expected}_distance_exceeds_fail_threshold"
        if distance_value > threshold_value:
            status = "review"
            notes.append(f"{expected}_distance_exceeds_review_threshold")
    if expected == "":
        notes.append("no_direct_organ_target_available")
    return status, ";".join(notes) if notes else "within_engineering_rules"


def _write_organ_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "group_id",
                "voxel_count",
                "volume_cm3",
                "centroid_x_mm",
                "centroid_y_mm",
                "centroid_z_mm",
                "bbox_min_x_mm",
                "bbox_min_y_mm",
                "bbox_min_z_mm",
                "bbox_max_x_mm",
                "bbox_max_y_mm",
                "bbox_max_z_mm",
                "extent_x_mm",
                "extent_y_mm",
                "extent_z_mm",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["group_id"],
                    row["voxel_count"],
                    f"{row['volume_cm3']:.6f}",
                    *[f"{value:.6f}" for value in row["centroid_mm"]],
                    *[f"{value:.6f}" for value in row["bbox_min_mm"]],
                    *[f"{value:.6f}" for value in row["bbox_max_mm"]],
                    *[f"{value:.6f}" for value in row["extent_mm"]],
                ]
            )


def _write_edge_metrics(path: Path, rows: list[dict[str, Any]], organ_ids: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "edge_id",
        "source",
        "target",
        "vessel_type",
        "flow_role",
        "expected_target",
        "expected_threshold_mm",
        "status",
        "status_note",
        "sample_count",
        "valid_sample_count",
        "outside_body_fraction",
        "inside_bone_fraction",
        "length_mm",
    ]
    for organ_id in organ_ids:
        fields.extend(
            [
                f"min_distance_to_{organ_id}_mm",
                f"mean_distance_to_{organ_id}_mm",
                f"median_distance_to_{organ_id}_mm",
                f"inside_{organ_id}_fraction",
            ]
        )
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_node_metrics(path: Path, rows: list[dict[str, Any]], organ_ids: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["node_id", "kind", "role", "boundary_role", "x_mm", "y_mm", "z_mm", "inside_body", "inside_bone"]
    for organ_id in organ_ids:
        fields.extend([f"distance_to_{organ_id}_mm", f"inside_{organ_id}"])
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_overlap_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["domain", "organ", "overlap_voxels", "overlap_cm3", "domain_fraction", "organ_fraction"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_spec(path: Path, result: VesselAnatomyValidationResult, sample_step_mm: float, organ_ids: tuple[str, ...]) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "validation_type": "organ_aware_vascular_anatomy_validation",
        "source_voxelized_spec": result.voxelized_spec_path,
        "source_graph": result.graph_yaml_path,
        "source_anatomy_labels": result.anatomy_labels_path,
        "sample_step_mm": sample_step_mm,
        "organ_groups": list(organ_ids),
        "summary": {
            "edge_count": result.edge_count,
            "node_count": result.node_count,
            "pass_count": result.pass_count,
            "review_count": result.review_count,
            "fail_count": result.fail_count,
            "outside_body_edge_count": result.outside_body_edge_count,
            "bone_intersection_edge_count": result.bone_intersection_edge_count,
            "missing_expected_target_count": result.missing_expected_target_count,
        },
        "outputs": {
            "edge_metrics_csv": result.edge_metrics_csv_path,
            "node_metrics_csv": result.node_metrics_csv_path,
            "organ_stats_csv": result.organ_stats_csv_path,
            "overlap_metrics_csv": result.overlap_metrics_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _slice_index(mask: np.ndarray, axis: int) -> int:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return mask.shape[axis] // 2
    return int(round(float(np.median(coords[:, axis]))))


def _render_preview(
    path: Path,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    arterial: np.ndarray,
    venous: np.ndarray,
    wall: np.ndarray,
    edge_rows: list[dict[str, Any]],
) -> None:
    plt, *_ = _import_dependencies()
    liver = masks["liver"]
    kidneys = masks["kidneys"]
    bone = masks["bone"]
    vessel = arterial | venous
    z_liver = _slice_index(liver if liver.any() else vessel, 2)
    z_kidney = _slice_index(kidneys if kidneys.any() else vessel, 2)
    y_vessel = _slice_index(vessel, 1)
    x_vessel = _slice_index(vessel, 0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=170)
    fig.patch.set_facecolor("#f8f3e8")
    for ax in axes.ravel():
        ax.set_facecolor("#f8f3e8")
        ax.axis("off")

    def draw_axial(ax, z_index: int, title: str) -> None:
        ax.imshow(np.rot90(labels[:, :, z_index]), cmap="bone", interpolation="nearest")
        for mask, color, width in (
            (bone[:, :, z_index], "#ffffff", 0.8),
            (liver[:, :, z_index], "#8f3ffc", 1.1),
            (kidneys[:, :, z_index], "#f72585", 1.1),
            (wall[:, :, z_index], "#f59e0b", 0.8),
            (venous[:, :, z_index], "#2563eb", 1.0),
            (arterial[:, :, z_index], "#dc2626", 1.1),
        ):
            if np.any(mask):
                ax.contour(np.rot90(mask.astype(float)), levels=[0.5], colors=[color], linewidths=width)
        ax.set_title(title, fontsize=10, color="#17202a")

    def draw_plane(ax, plane: str, index: int, title: str) -> None:
        if plane == "coronal":
            label_view = labels[:, index, :]
            masks_view = [(m[:, index, :], color, width) for m, color, width in ((bone, "#ffffff", 0.8), (liver, "#8f3ffc", 1.1), (kidneys, "#f72585", 1.1), (wall, "#f59e0b", 0.8), (venous, "#2563eb", 1.0), (arterial, "#dc2626", 1.1))]
        else:
            label_view = labels[index, :, :]
            masks_view = [(m[index, :, :], color, width) for m, color, width in ((bone, "#ffffff", 0.8), (liver, "#8f3ffc", 1.1), (kidneys, "#f72585", 1.1), (wall, "#f59e0b", 0.8), (venous, "#2563eb", 1.0), (arterial, "#dc2626", 1.1))]
        ax.imshow(np.rot90(label_view), cmap="bone", interpolation="nearest")
        for mask, color, width in masks_view:
            if np.any(mask):
                ax.contour(np.rot90(mask.astype(float)), levels=[0.5], colors=[color], linewidths=width)
        ax.set_title(title, fontsize=10, color="#17202a")

    draw_axial(axes[0, 0], z_kidney, f"Renal axial z={z_kidney}")
    draw_axial(axes[0, 1], z_liver, f"Hepatic axial z={z_liver}")
    draw_plane(axes[0, 2], "coronal", y_vessel, f"Coronal vessel y={y_vessel}")
    draw_plane(axes[1, 0], "sagittal", x_vessel, f"Sagittal vessel x={x_vessel}")

    status_counts = {"pass": 0, "review": 0, "fail": 0}
    for row in edge_rows:
        status_counts[str(row.get("status", "review"))] = status_counts.get(str(row.get("status", "review")), 0) + 1
    axes[1, 1].bar(
        list(status_counts),
        [status_counts["pass"], status_counts["review"], status_counts["fail"]],
        color=["#16a34a", "#f59e0b", "#dc2626"],
    )
    axes[1, 1].set_title("Edge QA Status", fontsize=10, color="#17202a")
    axes[1, 1].axis("on")
    axes[1, 1].set_facecolor("#f8f3e8")
    axes[1, 1].tick_params(axis="x", labelrotation=0)
    axes[1, 1].tick_params(axis="y", labelsize=8)

    review_edges = [row for row in edge_rows if row.get("status") != "pass"][:8]
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Top Review Items", fontsize=10, color="#17202a")
    text_lines = ["red=artery, blue=vein, orange=wall", "purple=liver, pink=kidneys, white=bone", ""]
    for row in review_edges:
        text_lines.append(f"{row['status']}: {row['edge_id']}")
    if not review_edges:
        text_lines.append("No edge review items.")
    axes[1, 2].text(0.0, 0.95, "\n".join(text_lines), va="top", ha="left", fontsize=8, color="#17202a", transform=axes[1, 2].transAxes)

    fig.suptitle("Organ-Aware Vascular Anatomy Validation", fontsize=15, color="#111827")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _format_report(
    result: VesselAnatomyValidationResult,
    edge_rows: list[dict[str, Any]],
    organ_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> str:
    review_rows = [row for row in edge_rows if row["status"] != "pass"]
    renal_rows = [row for row in edge_rows if "renal" in row["edge_id"].lower()]
    hepatic_rows = [row for row in edge_rows if "hepatic" in row["edge_id"].lower()]
    lines = [
        "# Organ-Aware Vascular Anatomy Validation",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Edges evaluated: {result.edge_count}",
        f"- Nodes evaluated: {result.node_count}",
        f"- Pass / review / fail: {result.pass_count} / {result.review_count} / {result.fail_count}",
        f"- Edges with any centerline sample outside body: {result.outside_body_edge_count}",
        f"- Edges with >5% centerline samples intersecting bone label: {result.bone_intersection_edge_count}",
        f"- Edges missing a direct expected organ target: {result.missing_expected_target_count}",
        "",
        "## Outputs",
        "",
        f"- Edge metrics CSV: `{Path(result.edge_metrics_csv_path).name}`",
        f"- Node metrics CSV: `{Path(result.node_metrics_csv_path).name}`",
        f"- Organ stats CSV: `{Path(result.organ_stats_csv_path).name}`",
        f"- Voxel overlap CSV: `{Path(result.overlap_metrics_csv_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Spec YAML: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Organ Volumes",
        "",
        "| organ | volume cm3 | centroid mm |",
        "| --- | ---: | --- |",
    ]
    for row in organ_rows:
        if row["group_id"] in {"body", "liver", "kidneys", "left_kidney", "right_kidney", "lungs", "bone"}:
            centroid = ", ".join(f"{value:.1f}" for value in row["centroid_mm"])
            lines.append(f"| `{row['group_id']}` | {row['volume_cm3']:.2f} | {centroid} |")

    lines.extend(["", "## Review / Fail Edges", "", "| edge | status | expected target | outside body % | bone intersection % | note |", "| --- | --- | --- | ---: | ---: | --- |"])
    if review_rows:
        for row in review_rows[:12]:
            lines.append(
                f"| `{row['edge_id']}` | `{row['status']}` | `{row['expected_target'] or 'none'}` | "
                f"{_safe_float(row.get('outside_body_fraction')) * 100.0:.2f} | "
                f"{_safe_float(row.get('inside_bone_fraction')) * 100.0:.2f} | {row['status_note']} |"
            )
    else:
        lines.append("| none | pass | n/a | 0.00 | 0.00 | no review items |")

    lines.extend(["", "## Renal Branch Distances", "", "| edge | expected | min expected distance mm | status |", "| --- | --- | ---: | --- |"])
    for row in renal_rows:
        expected = row["expected_target"]
        distance = row.get(f"min_distance_to_{expected}_mm", "") if expected else ""
        distance_text = f"{float(distance):.2f}" if distance != "" and np.isfinite(float(distance)) else "n/a"
        lines.append(f"| `{row['edge_id']}` | `{expected or 'none'}` | {distance_text} | `{row['status']}` |")

    lines.extend(["", "## Hepatic Branch Distances", "", "| edge | expected | min expected distance mm | status |", "| --- | --- | ---: | --- |"])
    for row in hepatic_rows:
        expected = row["expected_target"]
        distance = row.get(f"min_distance_to_{expected}_mm", "") if expected else ""
        distance_text = f"{float(distance):.2f}" if distance != "" and np.isfinite(float(distance)) else "n/a"
        lines.append(f"| `{row['edge_id']}` | `{expected or 'none'}` | {distance_text} | `{row['status']}` |")

    key_overlaps = [
        row
        for row in overlap_rows
        if row["organ"] in {"liver", "kidneys", "lungs", "bone"} and row["domain"] in {"arterial_lumen", "venous_lumen", "vessel_wall"}
    ]
    lines.extend(["", "## Voxel Overlap Highlights", "", "| domain | organ | overlap cm3 | domain fraction % |", "| --- | --- | ---: | ---: |"])
    for row in sorted(key_overlaps, key=lambda item: float(item["domain_fraction"]), reverse=True)[:12]:
        lines.append(
            f"| `{row['domain']}` | `{row['organ']}` | {float(row['overlap_cm3']):.3f} | {float(row['domain_fraction']) * 100.0:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This validates vessel-organ relationships in the current digital phantom coordinate system using the pre-vascular anatomy labels plus cleaned arterial/venous/wall masks.",
            "- Pass/review/fail thresholds are engineering plausibility gates, not clinical anatomical acceptance criteria.",
            "- Bone intersections are highlighted as review items because the current target and vessel scaffold are near the synthetic vertebral/bone region; these need organ-aware correction before stronger anatomy claims.",
            "- Splenic validation is limited because the current CT-ORG-derived material labels do not include a spleen label.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def validate_vessel_organ_anatomy(
    voxelized_spec_path: str | Path,
    graph_yaml_path: str | Path | None = None,
    anatomy_labels_path: str | Path | None = None,
    output_dir: str | Path = "outputs/validation/vessel_organ_anatomy",
    case_id: str = "ct_org_vessel_organ_validation",
    sample_step_mm: float = 2.0,
    report_path: str | Path | None = "outputs/reports/vessel_organ_anatomy_validation_stage001.md",
) -> VesselAnatomyValidationResult:
    _, nib, ndimage, yaml = _import_dependencies()
    spec_path = Path(voxelized_spec_path)
    spec = _load_yaml(spec_path)
    outputs = spec.get("outputs", {})
    voxelization = spec.get("voxelization", {})
    graph_path = _resolve_path(graph_yaml_path or voxelization.get("source_graph"), spec_path)
    labels_path = _resolve_path(anatomy_labels_path or voxelization.get("source_combined_labels"), spec_path)
    arterial_path = _resolve_path(outputs.get("arterial_lumen_mask"), spec_path)
    venous_path = _resolve_path(outputs.get("venous_lumen_mask"), spec_path)
    wall_path = _resolve_path(outputs.get("vessel_wall_mask"), spec_path)
    if graph_path is None or labels_path is None or arterial_path is None or venous_path is None or wall_path is None:
        raise ValueError("Voxelized spec must provide source graph, source anatomy labels, arterial mask, venous mask, and vessel wall mask")

    graph = yaml.safe_load(Path(graph_path).read_text())
    if not isinstance(graph, dict):
        raise ValueError(f"Graph YAML is not a mapping: {graph_path}")
    label_image = nib.load(str(labels_path))
    labels = np.rint(np.asanyarray(label_image.dataobj)).astype(np.int16)
    spacing = _spacing_from_image(label_image)
    shape = labels.shape

    arterial = np.asanyarray(nib.load(str(arterial_path)).dataobj) > 0
    venous = np.asanyarray(nib.load(str(venous_path)).dataobj) > 0
    wall = np.asanyarray(nib.load(str(wall_path)).dataobj) > 0
    for name, mask in (("arterial", arterial), ("venous", venous), ("wall", wall)):
        if mask.shape != shape:
            raise ValueError(f"{name} mask and anatomy labels differ: {mask.shape} vs {shape}")

    organ_ids = ("body", "liver", "kidneys", "left_kidney", "right_kidney", "lungs", "bone")
    organ_masks = {organ_id: _mask_for_group(labels, organ_id) for organ_id in organ_ids}
    organ_rows = [_mask_stats(organ_id, organ_masks[organ_id], spacing) for organ_id in organ_ids]

    edge_rows: list[dict[str, Any]] = []
    edge_samples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        edge_id = str(edge.get("id", ""))
        points = _edge_samples(edge, sample_step_mm=sample_step_mm)
        indices, valid = _points_to_indices(points, spacing, shape)
        flow_role = str(edge.get("flow_role", ""))
        expected, threshold, _ = _expected_target(edge_id, flow_role)
        body_fraction = _sample_mask_fraction(organ_masks["body"], indices, valid)
        bone_fraction = _sample_mask_fraction(organ_masks["bone"], indices, valid)
        row: dict[str, Any] = {
            "edge_id": edge_id,
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "vessel_type": edge.get("vessel_type", ""),
            "flow_role": flow_role,
            "expected_target": expected or "",
            "expected_threshold_mm": "" if threshold is None else f"{threshold:.3f}",
            "sample_count": int(len(points)),
            "valid_sample_count": int(np.count_nonzero(valid)),
            "outside_body_fraction": f"{max(0.0, 1.0 - body_fraction):.6f}",
            "inside_bone_fraction": f"{bone_fraction:.6f}",
            "length_mm": f"{float(edge.get('length_mm', 0.0)):.6f}",
        }
        edge_rows.append(row)
        edge_samples[edge_id] = (points, indices, valid)

    node_rows: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        position = np.asarray(node.get("position_mm", [0.0, 0.0, 0.0]), dtype=float).reshape(1, 3)
        indices, valid = _points_to_indices(position, spacing, shape)
        inside_body = _sample_mask_fraction(organ_masks["body"], indices, valid) > 0
        inside_bone = _sample_mask_fraction(organ_masks["bone"], indices, valid) > 0
        node_rows.append(
            {
                "node_id": str(node.get("id", "")),
                "kind": str(node.get("kind", "")),
                "role": str(node.get("role", "")),
                "boundary_role": str(node.get("boundary_role", "")),
                "x_mm": f"{float(position[0, 0]):.6f}",
                "y_mm": f"{float(position[0, 1]):.6f}",
                "z_mm": f"{float(position[0, 2]):.6f}",
                "inside_body": int(inside_body),
                "inside_bone": int(inside_bone),
            }
        )

    for organ_id in organ_ids:
        mask = organ_masks[organ_id]
        if not np.any(mask):
            for row in edge_rows:
                row[f"min_distance_to_{organ_id}_mm"] = ""
                row[f"mean_distance_to_{organ_id}_mm"] = ""
                row[f"median_distance_to_{organ_id}_mm"] = ""
                row[f"inside_{organ_id}_fraction"] = ""
            for row in node_rows:
                row[f"distance_to_{organ_id}_mm"] = ""
                row[f"inside_{organ_id}"] = ""
            continue
        distance = ndimage.distance_transform_edt(~mask, sampling=spacing)
        for row in edge_rows:
            _, indices, valid = edge_samples[row["edge_id"]]
            min_distance, mean_distance, median_distance = _sample_distance(distance, indices, valid)
            row[f"min_distance_to_{organ_id}_mm"] = f"{min_distance:.6f}" if np.isfinite(min_distance) else ""
            row[f"mean_distance_to_{organ_id}_mm"] = f"{mean_distance:.6f}" if np.isfinite(mean_distance) else ""
            row[f"median_distance_to_{organ_id}_mm"] = f"{median_distance:.6f}" if np.isfinite(median_distance) else ""
            row[f"inside_{organ_id}_fraction"] = f"{_sample_mask_fraction(mask, indices, valid):.6f}"
        for row in node_rows:
            position = np.asarray([[float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])]], dtype=float)
            indices, valid = _points_to_indices(position, spacing, shape)
            min_distance, _, _ = _sample_distance(distance, indices, valid)
            row[f"distance_to_{organ_id}_mm"] = f"{min_distance:.6f}" if np.isfinite(min_distance) else ""
            row[f"inside_{organ_id}"] = int(_sample_mask_fraction(mask, indices, valid) > 0)
        del distance

    for row in edge_rows:
        status, note = _edge_status(row)
        row["status"] = status
        row["status_note"] = note

    voxel_volume_cm3 = float(np.prod(np.asarray(spacing, dtype=float)) / 1000.0)
    domains = {
        "arterial_lumen": arterial,
        "venous_lumen": venous,
        "combined_lumen": arterial | venous,
        "vessel_wall": wall,
    }
    overlap_rows: list[dict[str, Any]] = []
    for domain_id, domain_mask in domains.items():
        domain_voxels = int(np.count_nonzero(domain_mask))
        for organ_id, organ_mask in organ_masks.items():
            organ_voxels = int(np.count_nonzero(organ_mask))
            overlap = int(np.count_nonzero(domain_mask & organ_mask))
            overlap_rows.append(
                {
                    "domain": domain_id,
                    "organ": organ_id,
                    "overlap_voxels": overlap,
                    "overlap_cm3": f"{overlap * voxel_volume_cm3:.6f}",
                    "domain_fraction": f"{(overlap / domain_voxels) if domain_voxels else 0.0:.8f}",
                    "organ_fraction": f"{(overlap / organ_voxels) if organ_voxels else 0.0:.8f}",
                }
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    edge_csv = output / f"{case_id}_vessel_organ_edge_metrics_v001.csv"
    node_csv = output / f"{case_id}_vessel_organ_node_metrics_v001.csv"
    organ_csv = output / f"{case_id}_vessel_organ_stats_v001.csv"
    overlap_csv = output / f"{case_id}_vessel_organ_overlap_metrics_v001.csv"
    preview = output / f"{case_id}_vessel_organ_validation_preview_v001.png"
    spec_out = output / f"{case_id}_vessel_organ_validation_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_vessel_organ_validation_report_v001.md"

    pass_count = sum(1 for row in edge_rows if row["status"] == "pass")
    review_count = sum(1 for row in edge_rows if row["status"] == "review")
    fail_count = sum(1 for row in edge_rows if row["status"] == "fail")
    outside_count = sum(1 for row in edge_rows if _safe_float(row.get("outside_body_fraction")) > 0.0)
    bone_count = sum(1 for row in edge_rows if _safe_float(row.get("inside_bone_fraction")) > 0.05)
    missing_count = sum(1 for row in edge_rows if row["expected_target"] == "")
    notes = (
        "organ_relationships_measured_against_pre_vascular_material_labels",
        "coordinates_follow_existing_project_convention=index_times_spacing_mm",
        "thresholds_are_engineering_plausibility_gates_not_clinical_acceptance_criteria",
        "spleen_specific_validation_unavailable_in_current_ct_org_material_labels",
    )
    result = VesselAnatomyValidationResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_path),
        anatomy_labels_path=str(labels_path),
        voxelized_spec_path=str(spec_path),
        edge_metrics_csv_path=str(edge_csv),
        node_metrics_csv_path=str(node_csv),
        organ_stats_csv_path=str(organ_csv),
        overlap_metrics_csv_path=str(overlap_csv),
        preview_png_path=str(preview),
        spec_yaml_path=str(spec_out),
        report_path=str(report),
        edge_count=len(edge_rows),
        node_count=len(node_rows),
        pass_count=pass_count,
        review_count=review_count,
        fail_count=fail_count,
        outside_body_edge_count=outside_count,
        bone_intersection_edge_count=bone_count,
        missing_expected_target_count=missing_count,
        notes=notes,
    )

    _write_edge_metrics(edge_csv, edge_rows, organ_ids)
    _write_node_metrics(node_csv, node_rows, organ_ids)
    _write_organ_stats(organ_csv, organ_rows)
    _write_overlap_metrics(overlap_csv, overlap_rows)
    _render_preview(preview, labels, organ_masks, arterial, venous, wall, edge_rows)
    _write_spec(spec_out, result, sample_step_mm=sample_step_mm, organ_ids=organ_ids)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, edge_rows, organ_rows, overlap_rows) + "\n")
    return result


def format_vessel_anatomy_validation_result(result: VesselAnatomyValidationResult) -> str:
    return "\n".join(
        [
            "Organ-aware vascular anatomy validation completed",
            f"Case ID: {result.case_id}",
            f"Edges pass/review/fail: {result.pass_count}/{result.review_count}/{result.fail_count}",
            f"Bone-intersection review edges: {result.bone_intersection_edge_count}",
            f"Edge metrics CSV: {result.edge_metrics_csv_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
