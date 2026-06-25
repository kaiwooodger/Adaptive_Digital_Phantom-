from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np

from .cta_vascular_graph import _line_length
from .vascular_voxelize import _import_dependencies, _load_yaml, _spacing_from_image
from .vessel_radius_profile import edge_radius_at_fraction


@dataclass(frozen=True)
class VesselOutsideBodyRepairResult:
    case_id: str
    output_dir: str
    source_graph_path: str
    anatomy_labels_path: str
    repaired_graph_yaml_path: str
    edge_metrics_csv_path: str
    sample_metrics_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    status: str
    targeted_edge_count: int
    repaired_edge_count: int
    moved_node_count: int
    outside_voxels_before: int
    outside_voxels_after: int
    max_edge_outside_fraction_before: float
    max_edge_outside_fraction_after: float
    mean_radius_reduction_percent: float
    max_radius_reduction_percent: float
    notes: tuple[str, ...]


def _sample_edge(edge: dict[str, Any], sample_step_mm: float) -> list[dict[str, Any]]:
    points = np.asarray(edge.get("polyline_mm", []), dtype=float)
    if points.ndim != 2 or len(points) < 2:
        return []
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-6:
        return []

    samples: list[dict[str, Any]] = []
    running = 0.0
    step = max(float(sample_step_mm), 0.1)
    for segment_index, length in enumerate(segment_lengths):
        if length <= 1e-6:
            continue
        count = max(1, int(np.ceil(float(length) / step)))
        for local_index in range(count + 1):
            if segment_index > 0 and local_index == 0:
                continue
            local_t = local_index / count
            station = (running + local_t * float(length)) / total_length
            point = points[segment_index] + (points[segment_index + 1] - points[segment_index]) * local_t
            samples.append(
                {
                    "station": float(station),
                    "point": point.astype(float),
                    "radius_mm": float(edge_radius_at_fraction(edge, station)),
                }
            )
        running += float(length)
    return samples


def _signed_body_field(body_mask: np.ndarray, spacing_mm: tuple[float, float, float], ndimage) -> np.ndarray:
    inside_distance = ndimage.distance_transform_edt(body_mask, sampling=spacing_mm).astype(np.float32)
    outside_distance = ndimage.distance_transform_edt(~body_mask, sampling=spacing_mm).astype(np.float32)
    return inside_distance - outside_distance


def _nearest_index(
    point_mm: np.ndarray,
    spacing_mm: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    index = np.rint(point_mm / np.asarray(spacing_mm, dtype=float)).astype(int)
    if np.any(index < 0) or np.any(index >= np.asarray(shape, dtype=int)):
        return None
    return tuple(int(value) for value in index)


def _signed_value(point_mm: np.ndarray, spacing_mm: tuple[float, float, float], signed_body_mm: np.ndarray) -> float:
    index = _nearest_index(point_mm, spacing_mm, signed_body_mm.shape)
    if index is None:
        return float("-inf")
    return float(signed_body_mm[index])


def _sphere_counts(
    center_mm: np.ndarray,
    radius_mm: float,
    *,
    spacing_mm: tuple[float, float, float],
    shape: tuple[int, int, int],
    body_mask: np.ndarray,
) -> tuple[int, int]:
    center = np.asarray(center_mm, dtype=float)
    spacing = np.asarray(spacing_mm, dtype=float)
    center_index = center / spacing
    radius_index = np.ceil(float(radius_mm) / spacing).astype(int) + 1
    mins = np.maximum(np.floor(center_index).astype(int) - radius_index, 0)
    maxs = np.minimum(np.ceil(center_index).astype(int) + radius_index + 1, np.asarray(shape, dtype=int))
    if np.any(maxs <= mins):
        return 0, 0

    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    grids = np.meshgrid(
        *[
            np.arange(mins[axis], maxs[axis], dtype=float) * spacing[axis]
            for axis in range(3)
        ],
        indexing="ij",
    )
    distance_sq = sum((grid - center[axis]) ** 2 for axis, grid in enumerate(grids))
    sphere = distance_sq <= float(radius_mm) ** 2
    total = int(np.count_nonzero(sphere))
    outside = int(np.count_nonzero(sphere & ~body_mask[slices]))
    return outside, total


def _edge_metrics(
    edge: dict[str, Any],
    *,
    spacing_mm: tuple[float, float, float],
    shape: tuple[int, int, int],
    body_mask: np.ndarray,
    signed_body_mm: np.ndarray,
    sample_step_mm: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = _sample_edge(edge, sample_step_mm)
    edge_mask = np.zeros(shape, dtype=bool)
    clipped_mask = np.zeros(shape, dtype=bool)
    sample_rows: list[dict[str, Any]] = []

    for sample in samples:
        point = np.asarray(sample["point"], dtype=float)
        radius = float(sample["radius_mm"])
        center_index = point / np.asarray(spacing_mm, dtype=float)
        radius_index = np.ceil(radius / np.asarray(spacing_mm, dtype=float)).astype(int) + 1
        mins = np.maximum(np.floor(center_index).astype(int) - radius_index, 0)
        maxs = np.minimum(np.ceil(center_index).astype(int) + radius_index + 1, np.asarray(shape, dtype=int))
        if np.any(maxs <= mins):
            outside, total = 0, 0
        else:
            slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
            grids = np.meshgrid(
                *[
                    np.arange(mins[axis], maxs[axis], dtype=float) * float(spacing_mm[axis])
                    for axis in range(3)
                ],
                indexing="ij",
            )
            distance_sq = sum((grid - point[axis]) ** 2 for axis, grid in enumerate(grids))
            sphere = distance_sq <= radius**2
            edge_mask[slices] |= sphere
            clipped_mask[slices] |= sphere & body_mask[slices]
            outside, total = _sphere_counts(
                point,
                radius,
                spacing_mm=spacing_mm,
                shape=shape,
                body_mask=body_mask,
            )
        sample_rows.append(
            {
                "edge_id": str(edge.get("id", "")),
                "station": float(sample["station"]),
                "x_mm": float(point[0]),
                "y_mm": float(point[1]),
                "z_mm": float(point[2]),
                "radius_mm": radius,
                "signed_body_distance_mm": _signed_value(point, spacing_mm, signed_body_mm),
                "sample_outside_voxels": int(outside),
                "sample_total_voxels": int(total),
                "sample_outside_fraction": float(outside / total) if total else 0.0,
            }
        )

    outside_voxels = int(edge_mask.sum() - clipped_mask.sum())
    total_voxels = int(edge_mask.sum())
    centerline_outside = sum(1 for row in sample_rows if float(row["signed_body_distance_mm"]) <= 0.0)
    metrics = {
        "edge_id": str(edge.get("id", "")),
        "vessel_type": str(edge.get("vessel_type", "")),
        "flow_role": str(edge.get("flow_role", "")),
        "sample_count": len(sample_rows),
        "outside_voxels": outside_voxels,
        "total_voxels": total_voxels,
        "outside_fraction": float(outside_voxels / total_voxels) if total_voxels else 0.0,
        "centerline_outside_sample_fraction": float(centerline_outside / len(sample_rows)) if sample_rows else 0.0,
        "max_sample_outside_voxels": max((int(row["sample_outside_voxels"]) for row in sample_rows), default=0),
        "min_signed_body_distance_mm": min((float(row["signed_body_distance_mm"]) for row in sample_rows), default=0.0),
    }
    return metrics, sample_rows


def _node_degrees(graph: dict[str, Any]) -> dict[str, int]:
    degrees: dict[str, int] = {}
    for edge in graph.get("edges", []):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        degrees[source] = degrees.get(source, 0) + 1
        degrees[target] = degrees.get(target, 0) + 1
    return degrees


def _trim_bounds(
    edge: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    graph: dict[str, Any],
) -> tuple[float, float, bool, bool]:
    if not sample_rows:
        return 0.0, 1.0, False, False
    degrees = _node_degrees(graph)
    node_by_id = {str(node.get("id")): node for node in graph.get("nodes", [])}
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    source_boundary = bool(node_by_id.get(source, {}).get("boundary_role"))
    target_boundary = bool(node_by_id.get(target, {}).get("boundary_role"))

    bad = [int(row["sample_outside_voxels"]) > 0 or float(row["signed_body_distance_mm"]) <= 0.0 for row in sample_rows]
    trim_start = 0.0
    trim_end = 1.0
    moved_source = False
    moved_target = False

    if bad[0] and source_boundary and degrees.get(source, 0) <= 1:
        for row, is_bad in zip(sample_rows, bad, strict=False):
            if not is_bad:
                trim_start = float(row["station"])
                moved_source = trim_start > 0.0
                break

    if bad[-1] and target_boundary and degrees.get(target, 0) <= 1:
        for row, is_bad in zip(reversed(sample_rows), reversed(bad), strict=False):
            if not is_bad:
                trim_end = float(row["station"])
                moved_target = trim_end < 1.0
                break

    if trim_start >= trim_end:
        return 0.0, 1.0, False, False
    return trim_start, trim_end, moved_source, moved_target


def _polyline_between(edge: dict[str, Any], start_station: float, end_station: float, sample_step_mm: float) -> np.ndarray:
    samples = _sample_edge(edge, sample_step_mm)
    kept = [
        np.asarray(row["point"], dtype=float)
        for row in samples
        if start_station - 1e-9 <= float(row["station"]) <= end_station + 1e-9
    ]
    if len(kept) < 2:
        return np.asarray(edge.get("polyline_mm", []), dtype=float)
    return np.asarray(kept, dtype=float)


def _radius_profile_for_body(
    original_edge: dict[str, Any],
    new_edge: dict[str, Any],
    *,
    trim_start: float,
    trim_end: float,
    signed_body_mm: np.ndarray,
    spacing_mm: tuple[float, float, float],
    body_margin_mm: float,
    min_radius_mm: float,
    max_profile_points: int,
    sample_step_mm: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    samples = _sample_edge(new_edge, sample_step_mm)
    if not samples:
        return [], {
            "mean_reduction_percent": 0.0,
            "max_reduction_percent": 0.0,
            "radius_min_after_mm": 0.0,
            "radius_mean_after_mm": 0.0,
            "radius_max_after_mm": 0.0,
        }
    stations = np.asarray([float(row["station"]) for row in samples], dtype=float)
    old_stations = trim_start + stations * max(trim_end - trim_start, 1e-9)
    before = np.asarray([edge_radius_at_fraction(original_edge, station) for station in old_stations], dtype=float)
    signed = np.asarray(
        [
            _signed_value(np.asarray(row["point"], dtype=float), spacing_mm, signed_body_mm)
            for row in samples
        ],
        dtype=float,
    )
    safe = signed - float(body_margin_mm)
    after = np.minimum(before, np.maximum(float(min_radius_mm), safe))
    after = np.minimum(after, before)
    after = np.maximum(after, 0.05)

    count = min(max(int(max_profile_points), 2), len(stations))
    target_stations = np.linspace(0.0, 1.0, count)
    target_radii = np.interp(target_stations, stations, after)
    profile = [
        {"station": float(round(station, 6)), "radius_mm": float(round(radius, 6))}
        for station, radius in zip(target_stations, target_radii)
    ]
    reductions = np.maximum(before - after, 0.0) / np.maximum(before, 1e-9) * 100.0
    stats = {
        "mean_reduction_percent": float(np.mean(reductions)),
        "max_reduction_percent": float(np.max(reductions)),
        "radius_min_after_mm": float(np.min(after)),
        "radius_mean_after_mm": float(np.mean(after)),
        "radius_max_after_mm": float(np.max(after)),
    }
    return profile, stats


def _write_edge_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "edge_id",
        "targeted",
        "repaired",
        "vessel_type",
        "flow_role",
        "outside_voxels_before",
        "outside_voxels_after",
        "outside_fraction_before",
        "outside_fraction_after",
        "centerline_outside_fraction_before",
        "centerline_outside_fraction_after",
        "trim_start_station",
        "trim_end_station",
        "source_node_moved",
        "target_node_moved",
        "radius_profile_applied",
        "mean_radius_reduction_percent",
        "max_radius_reduction_percent",
        "min_signed_body_distance_before_mm",
        "min_signed_body_distance_after_mm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_sample_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "edge_id",
        "station",
        "x_mm",
        "y_mm",
        "z_mm",
        "radius_mm",
        "signed_body_distance_mm",
        "sample_outside_voxels",
        "sample_total_voxels",
        "sample_outside_fraction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _render_preview(path: Path, rows: list[dict[str, Any]]) -> None:
    plt, *_ = _import_dependencies()
    targeted = [row for row in rows if int(row.get("targeted", 0)) == 1]
    top = sorted(targeted, key=lambda row: int(row.get("outside_voxels_before", 0)), reverse=True)[:12]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=170)
    fig.patch.set_facecolor("#f8f3e8")
    for ax in axes:
        ax.set_facecolor("#f8f3e8")

    if top:
        labels = [str(row["edge_id"]) for row in top]
        before = [int(row["outside_voxels_before"]) for row in top]
        after = [int(row["outside_voxels_after"]) for row in top]
        reduction = [float(row.get("mean_radius_reduction_percent", 0.0)) for row in top]
        y = np.arange(len(top))
        axes[0].barh(y, before, color="#94a3b8", alpha=0.7, label="before")
        axes[0].barh(y, after, color="#dc2626", alpha=0.85, label="after")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Outside-body voxels before clipping")
        axes[0].legend(fontsize=8)
        axes[1].barh(y, reduction, color="#f97316", alpha=0.85)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Mean radius reduction (%)")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No outside-body vessel margin found", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    axes[0].set_title("Outside-Body Margin Repair")
    axes[1].set_title("Local Radius Profile Adjustment")
    fig.suptitle("Vascular Outside-Body Release-Gate Repair", fontsize=15, color="#111827")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_spec(path: Path, result: VesselOutsideBodyRepairResult, params: dict[str, Any]) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "correction_type": "vascular_outside_body_margin_repair",
        "source_graph": result.source_graph_path,
        "source_anatomy_labels": result.anatomy_labels_path,
        "parameters": params,
        "summary": {
            "status": result.status,
            "targeted_edge_count": result.targeted_edge_count,
            "repaired_edge_count": result.repaired_edge_count,
            "moved_node_count": result.moved_node_count,
            "outside_voxels_before": result.outside_voxels_before,
            "outside_voxels_after": result.outside_voxels_after,
            "max_edge_outside_fraction_before": result.max_edge_outside_fraction_before,
            "max_edge_outside_fraction_after": result.max_edge_outside_fraction_after,
            "mean_radius_reduction_percent": result.mean_radius_reduction_percent,
            "max_radius_reduction_percent": result.max_radius_reduction_percent,
        },
        "outputs": {
            "repaired_graph": result.repaired_graph_yaml_path,
            "edge_metrics_csv": result.edge_metrics_csv_path,
            "sample_metrics_csv": result.sample_metrics_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VesselOutsideBodyRepairResult, rows: list[dict[str, Any]]) -> str:
    targeted = [row for row in rows if int(row.get("targeted", 0)) == 1]
    lines = [
        "# Vascular Outside-Body Margin Repair",
        "",
        f"Case ID: `{result.case_id}`",
        f"Status: `{result.status}`",
        "",
        "## Summary",
        "",
        f"- Targeted edges: {result.targeted_edge_count}",
        f"- Repaired edges: {result.repaired_edge_count}",
        f"- Moved boundary nodes: {result.moved_node_count}",
        f"- Outside-body voxels before/after: {result.outside_voxels_before} / {result.outside_voxels_after}",
        f"- Max edge outside fraction before/after: {result.max_edge_outside_fraction_before * 100.0:.3f}% / {result.max_edge_outside_fraction_after * 100.0:.3f}%",
        f"- Mean / max local radius reduction: {result.mean_radius_reduction_percent:.2f}% / {result.max_radius_reduction_percent:.2f}%",
        "",
        "## Targeted Edges",
        "",
        "| edge | outside voxels before/after | outside % before/after | trim station | node moved | radius reduction mean/max % |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in targeted:
        moved = []
        if int(row.get("source_node_moved", 0)):
            moved.append("source")
        if int(row.get("target_node_moved", 0)):
            moved.append("target")
        lines.append(
            f"| `{row['edge_id']}` | {int(row['outside_voxels_before'])} / {int(row['outside_voxels_after'])} | "
            f"{float(row['outside_fraction_before']) * 100.0:.3f}% / {float(row['outside_fraction_after']) * 100.0:.3f}% | "
            f"{float(row['trim_start_station']):.4f}-{float(row['trim_end_station']):.4f} | "
            f"{', '.join(moved) if moved else 'none'} | "
            f"{float(row['mean_radius_reduction_percent']):.2f} / {float(row['max_radius_reduction_percent']):.2f} |"
        )
    if not targeted:
        lines.append("| none | 0 / 0 | 0.000% / 0.000% | 0.0000-1.0000 | none | 0.00 / 0.00 |")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Repaired graph YAML: `{Path(result.repaired_graph_yaml_path).name}`",
            f"- Edge metrics CSV: `{Path(result.edge_metrics_csv_path).name}`",
            f"- Sample metrics CSV: `{Path(result.sample_metrics_csv_path).name}`",
            f"- Preview PNG: `{Path(result.preview_png_path).name}`",
            f"- Spec YAML: `{Path(result.spec_yaml_path).name}`",
            "",
            "## Interpretation",
            "",
            "- Graph node IDs, edge IDs, vessel types, flow roles, and boundary roles are preserved.",
            "- Terminal boundary nodes are moved only when the offending outside-body samples touch a degree-one inlet/outlet endpoint.",
            "- Local `radius_profile` entries are added only to targeted edges so vessel tubes fit inside the body before clipping.",
            "- This repair should be followed by vascular voxelization, organ/radius QA, flow boundary rebuild, and pulsatile flow rerun.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def repair_vessel_outside_body_margin(
    graph_yaml_path: str | Path,
    anatomy_labels_path: str | Path,
    output_dir: str | Path = "outputs/digital/vessel_outside_body_repair",
    case_id: str = "vessel_outside_body_repaired",
    edge_ids: tuple[str, ...] | list[str] = (),
    sample_step_mm: float = 0.9,
    body_margin_mm: float = 0.75,
    min_radius_mm: float = 0.5,
    max_profile_points: int = 128,
    report_path: str | Path | None = "outputs/reports/vessel_outside_body_repair.md",
) -> VesselOutsideBodyRepairResult:
    plt, _, _, nib, ndimage, yaml = _import_dependencies()
    _ = plt
    graph_path = Path(graph_yaml_path)
    labels_path = Path(anatomy_labels_path)
    graph = _load_yaml(graph_path)
    labels_image = nib.load(str(labels_path))
    labels = np.rint(np.asanyarray(labels_image.dataobj)).astype(np.int16)
    spacing = _spacing_from_image(labels_image)
    body = labels != 0
    if not np.any(body):
        raise ValueError("Anatomy labels do not contain a non-air body mask")
    signed_body = _signed_body_field(body, spacing, ndimage)

    explicit_edge_ids = {str(edge_id) for edge_id in edge_ids if str(edge_id)}
    before_by_edge: dict[str, dict[str, Any]] = {}
    before_samples_by_edge: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges", []):
        metrics, samples = _edge_metrics(
            edge,
            spacing_mm=spacing,
            shape=labels.shape,
            body_mask=body,
            signed_body_mm=signed_body,
            sample_step_mm=sample_step_mm,
        )
        before_by_edge[str(edge.get("id", ""))] = metrics
        before_samples_by_edge[str(edge.get("id", ""))] = samples

    target_ids = explicit_edge_ids or {
        edge_id for edge_id, metrics in before_by_edge.items() if int(metrics["outside_voxels"]) > 0
    }

    corrected = dict(graph)
    corrected_nodes = [dict(node) for node in graph.get("nodes", [])]
    node_index = {str(node.get("id")): index for index, node in enumerate(corrected_nodes)}
    corrected_edges: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    moved_nodes: set[str] = set()
    repaired_ids: set[str] = set()
    reduction_means: list[float] = []
    reduction_maxes: list[float] = []

    for edge in graph.get("edges", []):
        edge_id = str(edge.get("id", ""))
        before_metrics = before_by_edge[edge_id]
        before_samples = before_samples_by_edge[edge_id]
        targeted = edge_id in target_ids
        new_edge = dict(edge)
        trim_start = 0.0
        trim_end = 1.0
        moved_source = False
        moved_target = False
        radius_stats = {
            "mean_reduction_percent": 0.0,
            "max_reduction_percent": 0.0,
        }

        if targeted:
            trim_start, trim_end, moved_source, moved_target = _trim_bounds(edge, before_samples, graph)
            new_points = _polyline_between(edge, trim_start, trim_end, sample_step_mm)
            if len(new_points) >= 2:
                source_radius = float(edge_radius_at_fraction(edge, trim_start))
                target_radius = float(edge_radius_at_fraction(edge, trim_end))
                new_edge["polyline_mm"] = [[float(value) for value in point] for point in new_points]
                new_edge["length_mm"] = float(_line_length(new_points))
                new_edge["radius_start_mm"] = source_radius
                new_edge["radius_end_mm"] = target_radius
                if moved_source:
                    source_id = str(edge.get("source", ""))
                    if source_id in node_index:
                        node = dict(corrected_nodes[node_index[source_id]])
                        node["position_mm"] = [float(value) for value in new_points[0]]
                        node["radius_mm"] = source_radius
                        notes = list(node.get("notes", []))
                        notes.append("moved_inward_by_outside_body_margin_repair")
                        node["notes"] = sorted(set(str(note) for note in notes))
                        corrected_nodes[node_index[source_id]] = node
                        moved_nodes.add(source_id)
                if moved_target:
                    target_id = str(edge.get("target", ""))
                    if target_id in node_index:
                        node = dict(corrected_nodes[node_index[target_id]])
                        node["position_mm"] = [float(value) for value in new_points[-1]]
                        node["radius_mm"] = target_radius
                        notes = list(node.get("notes", []))
                        notes.append("moved_inward_by_outside_body_margin_repair")
                        node["notes"] = sorted(set(str(note) for note in notes))
                        corrected_nodes[node_index[target_id]] = node
                        moved_nodes.add(target_id)

                profile, radius_stats = _radius_profile_for_body(
                    edge,
                    new_edge,
                    trim_start=trim_start,
                    trim_end=trim_end,
                    signed_body_mm=signed_body,
                    spacing_mm=spacing,
                    body_margin_mm=body_margin_mm,
                    min_radius_mm=min_radius_mm,
                    max_profile_points=max_profile_points,
                    sample_step_mm=sample_step_mm,
                )
                if profile:
                    new_edge["radius_profile"] = profile
                    new_edge["radius_start_mm"] = float(profile[0]["radius_mm"])
                    new_edge["radius_end_mm"] = float(profile[-1]["radius_mm"])
                    repair_meta = dict(new_edge.get("outside_body_repair", {}))
                    repair_meta.update(
                        {
                            "method": "terminal_trim_plus_body_clearance_radius_profile",
                            "source_graph": str(graph_path),
                            "source_anatomy_labels": str(labels_path),
                            "trim_start_station": float(trim_start),
                            "trim_end_station": float(trim_end),
                            "original_radius_start_mm": float(edge.get("radius_start_mm", edge.get("radius_mm", 0.0))),
                            "original_radius_end_mm": float(edge.get("radius_end_mm", edge.get("radius_mm", 0.0))),
                            "body_margin_mm": float(body_margin_mm),
                            "min_radius_mm": float(min_radius_mm),
                        }
                    )
                    new_edge["outside_body_repair"] = repair_meta
                notes = list(new_edge.get("notes", []))
                notes.append("outside_body_margin_repair_applied")
                new_edge["notes"] = sorted(set(str(note) for note in notes))

        corrected_edges.append(new_edge)

    corrected["nodes"] = corrected_nodes
    corrected["edges"] = corrected_edges
    corrected["case_id"] = case_id
    metadata = dict(corrected.get("graph_metadata", {}))
    metadata["outside_body_margin_repair"] = {
        "source_graph": str(graph_path),
        "source_anatomy_labels": str(labels_path),
        "targeted_edge_ids": sorted(target_ids),
        "sample_step_mm": float(sample_step_mm),
        "body_margin_mm": float(body_margin_mm),
        "min_radius_mm": float(min_radius_mm),
    }
    corrected["graph_metadata"] = metadata
    provenance = list(corrected.get("provenance_notes", []))
    provenance.append("vascular_outside_body_margin_repair_applied")
    corrected["provenance_notes"] = sorted(set(str(note) for note in provenance))

    after_by_edge: dict[str, dict[str, Any]] = {}
    after_samples_by_edge: dict[str, list[dict[str, Any]]] = {}
    for edge in corrected_edges:
        metrics, samples = _edge_metrics(
            edge,
            spacing_mm=spacing,
            shape=labels.shape,
            body_mask=body,
            signed_body_mm=signed_body,
            sample_step_mm=sample_step_mm,
        )
        after_by_edge[str(edge.get("id", ""))] = metrics
        after_samples_by_edge[str(edge.get("id", ""))] = samples

    for edge in corrected_edges:
        edge_id = str(edge.get("id", ""))
        targeted = edge_id in target_ids
        before_metrics = before_by_edge[edge_id]
        after_metrics = after_by_edge[edge_id]
        repair_meta = edge.get("outside_body_repair", {}) if isinstance(edge.get("outside_body_repair"), dict) else {}
        mean_reduction = 0.0
        max_reduction = 0.0
        if targeted:
            stats_profile, stats = _radius_profile_for_body(
                next(item for item in graph.get("edges", []) if str(item.get("id", "")) == edge_id),
                edge,
                trim_start=float(repair_meta.get("trim_start_station", 0.0)),
                trim_end=float(repair_meta.get("trim_end_station", 1.0)),
                signed_body_mm=signed_body,
                spacing_mm=spacing,
                body_margin_mm=body_margin_mm,
                min_radius_mm=min_radius_mm,
                max_profile_points=max_profile_points,
                sample_step_mm=sample_step_mm,
            )
            _ = stats_profile
            mean_reduction = float(stats["mean_reduction_percent"])
            max_reduction = float(stats["max_reduction_percent"])
            reduction_means.append(mean_reduction)
            reduction_maxes.append(max_reduction)
            if int(after_metrics["outside_voxels"]) < int(before_metrics["outside_voxels"]):
                repaired_ids.add(edge_id)

        row = {
            "edge_id": edge_id,
            "targeted": int(targeted),
            "repaired": int(edge_id in repaired_ids),
            "vessel_type": before_metrics["vessel_type"],
            "flow_role": before_metrics["flow_role"],
            "outside_voxels_before": int(before_metrics["outside_voxels"]),
            "outside_voxels_after": int(after_metrics["outside_voxels"]),
            "outside_fraction_before": f"{float(before_metrics['outside_fraction']):.8f}",
            "outside_fraction_after": f"{float(after_metrics['outside_fraction']):.8f}",
            "centerline_outside_fraction_before": f"{float(before_metrics['centerline_outside_sample_fraction']):.8f}",
            "centerline_outside_fraction_after": f"{float(after_metrics['centerline_outside_sample_fraction']):.8f}",
            "trim_start_station": f"{float(repair_meta.get('trim_start_station', 0.0)):.8f}",
            "trim_end_station": f"{float(repair_meta.get('trim_end_station', 1.0)):.8f}",
            "source_node_moved": int(targeted and str(edge.get("source", "")) in moved_nodes),
            "target_node_moved": int(targeted and str(edge.get("target", "")) in moved_nodes),
            "radius_profile_applied": int(targeted and isinstance(edge.get("radius_profile"), list) and bool(edge.get("radius_profile"))),
            "mean_radius_reduction_percent": f"{mean_reduction:.6f}",
            "max_radius_reduction_percent": f"{max_reduction:.6f}",
            "min_signed_body_distance_before_mm": f"{float(before_metrics['min_signed_body_distance_mm']):.6f}",
            "min_signed_body_distance_after_mm": f"{float(after_metrics['min_signed_body_distance_mm']):.6f}",
        }
        edge_rows.append(row)

        if targeted:
            for phase, rows_for_phase in (("before", before_samples_by_edge[edge_id]), ("after", after_samples_by_edge[edge_id])):
                for sample in rows_for_phase:
                    sample_rows.append({"phase": phase, **sample})

    outside_before = int(sum(int(row["outside_voxels_before"]) for row in edge_rows))
    outside_after = int(sum(int(row["outside_voxels_after"]) for row in edge_rows))
    max_before = max((float(row["outside_fraction_before"]) for row in edge_rows), default=0.0)
    max_after = max((float(row["outside_fraction_after"]) for row in edge_rows), default=0.0)
    status = "outside_body_margin_repaired" if outside_after == 0 else "review_required"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repaired_graph = output / f"{case_id}_outside_body_repaired_vascular_graph_v001.yaml"
    edge_csv = output / f"{case_id}_outside_body_edge_metrics_v001.csv"
    sample_csv = output / f"{case_id}_outside_body_sample_metrics_v001.csv"
    preview_png = output / f"{case_id}_outside_body_repair_preview_v001.png"
    spec_yaml = output / f"{case_id}_outside_body_repair_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_outside_body_repair_report_v001.md"

    repaired_graph.write_text(yaml.safe_dump(corrected, sort_keys=False))
    _write_edge_metrics(edge_csv, edge_rows)
    _write_sample_metrics(sample_csv, sample_rows)
    _render_preview(preview_png, edge_rows)

    notes = (
        "topology_edge_ids_flow_roles_and_boundary_roles_preserved",
        "degree_one_boundary_nodes_may_move_inward_when_terminal_samples_start_outside_body",
        "targeted_edges_receive_body_clearance_radius_profile",
        "rerun_voxelization_and_flow_qa_after_repair",
    )
    result = VesselOutsideBodyRepairResult(
        case_id=case_id,
        output_dir=str(output),
        source_graph_path=str(graph_path),
        anatomy_labels_path=str(labels_path),
        repaired_graph_yaml_path=str(repaired_graph),
        edge_metrics_csv_path=str(edge_csv),
        sample_metrics_csv_path=str(sample_csv),
        preview_png_path=str(preview_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        status=status,
        targeted_edge_count=len(target_ids),
        repaired_edge_count=len(repaired_ids),
        moved_node_count=len(moved_nodes),
        outside_voxels_before=outside_before,
        outside_voxels_after=outside_after,
        max_edge_outside_fraction_before=max_before,
        max_edge_outside_fraction_after=max_after,
        mean_radius_reduction_percent=float(np.mean(reduction_means)) if reduction_means else 0.0,
        max_radius_reduction_percent=float(np.max(reduction_maxes)) if reduction_maxes else 0.0,
        notes=notes,
    )
    params = {
        "edge_ids": sorted(target_ids),
        "sample_step_mm": float(sample_step_mm),
        "body_margin_mm": float(body_margin_mm),
        "min_radius_mm": float(min_radius_mm),
        "max_profile_points": int(max_profile_points),
    }
    _write_spec(spec_yaml, result, params)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, edge_rows) + "\n")
    return result


def format_vessel_outside_body_repair_result(result: VesselOutsideBodyRepairResult) -> str:
    return "\n".join(
        [
            "Vascular outside-body margin repair completed",
            f"Case ID: {result.case_id}",
            f"Status: {result.status}",
            f"Targeted/repaired edges: {result.targeted_edge_count}/{result.repaired_edge_count}",
            f"Moved boundary nodes: {result.moved_node_count}",
            f"Outside voxels before/after: {result.outside_voxels_before}/{result.outside_voxels_after}",
            f"Max edge outside fraction before/after: {result.max_edge_outside_fraction_before * 100.0:.3f}%/{result.max_edge_outside_fraction_after * 100.0:.3f}%",
            f"Repaired graph YAML: {result.repaired_graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
