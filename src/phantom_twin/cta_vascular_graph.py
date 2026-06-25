from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        import yaml  # type: ignore
        from scipy import ndimage as ndi  # type: ignore
        from scipy.signal import savgol_filter  # type: ignore
    except ImportError as exc:
        raise RuntimeError("CTA-derived vascular graph replacement requires matplotlib, nibabel, scipy, and PyYAML.") from exc
    return plt, nib, yaml, savgol_filter, ndi


@dataclass(frozen=True)
class CtaDerivedGraphResult:
    case_id: str
    output_dir: str
    graph_yaml_path: str
    nodes_csv_path: str
    edges_csv_path: str
    branch_candidates_csv_path: str
    preview_png_path: str
    report_path: str
    centerline_points: int
    centerline_length_mm: float
    replaced_node_count: int
    replaced_edge_count: int
    branch_candidate_count: int
    promoted_branch_count: int
    rejected_branch_count: int
    retained_synthetic_edge_count: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class BranchCandidate:
    candidate_id: str
    point_count: int
    z_span_mm: float
    xy_span_mm: float
    length_mm: float
    mean_radius_mm: float
    mean_trunk_distance_mm: float
    classification: str
    promoted_edge_id: str | None
    rejection_reason: str
    polyline_mm: tuple[tuple[float, float, float], ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    _, _, yaml, _, _ = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _line_length(points: list[list[float]] | np.ndarray) -> float:
    array = np.asarray(points, dtype=float)
    if len(array) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def _smooth(values: np.ndarray, savgol_filter) -> np.ndarray:
    if len(values) < 7:
        return values
    window = min(17, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 7:
        return values
    return savgol_filter(values, window_length=window, polyorder=3, mode="interp")


def _centerline_from_mask(mask: np.ndarray, spacing_mm: tuple[float, float, float], savgol_filter) -> np.ndarray:
    rows: list[tuple[float, float, float, float, float, int]] = []
    voxel_area = spacing_mm[0] * spacing_mm[1]
    for z_index in range(mask.shape[2]):
        coords = np.argwhere(mask[:, :, z_index])
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)
        area = float(len(coords) * voxel_area)
        radius = math.sqrt(max(area, 1e-6) / math.pi)
        rows.append(
            (
                float(centroid[0] * spacing_mm[0]),
                float(centroid[1] * spacing_mm[1]),
                float(z_index * spacing_mm[2]),
                float(radius),
                area,
                int(z_index),
            )
        )
    if len(rows) < 2:
        raise ValueError("CTA vascular mask needs at least two occupied slices for centerline replacement")
    centerline = np.asarray(rows, dtype=float)
    centerline[:, 0] = _smooth(centerline[:, 0], savgol_filter)
    centerline[:, 1] = _smooth(centerline[:, 1], savgol_filter)
    centerline[:, 3] = np.maximum(_smooth(centerline[:, 3], savgol_filter), 0.1)
    centerline[:, 4] = math.pi * centerline[:, 3] ** 2
    return centerline


def _interp_centerline(centerline: np.ndarray, z_mm: float) -> tuple[list[float], float]:
    z_values = centerline[:, 2]
    z_clamped = float(np.clip(z_mm, z_values.min(), z_values.max()))
    x = float(np.interp(z_clamped, z_values, centerline[:, 0]))
    y = float(np.interp(z_clamped, z_values, centerline[:, 1]))
    radius = float(np.interp(z_clamped, z_values, centerline[:, 3]))
    return [x, y, z_clamped], radius


def _centerline_segment(centerline: np.ndarray, start: list[float], end: list[float]) -> list[list[float]]:
    start_z = float(start[2])
    end_z = float(end[2])
    z_min = min(start_z, end_z)
    z_max = max(start_z, end_z)
    interior = centerline[(centerline[:, 2] > z_min) & (centerline[:, 2] < z_max), :3]
    points = [list(map(float, start))]
    if len(interior):
        if start_z > end_z:
            interior = interior[::-1]
        points.extend([[float(value) for value in point] for point in interior])
    points.append(list(map(float, end)))
    return points


def _slice_component_tracks(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    centerline: np.ndarray,
    ndi,
    *,
    min_component_voxels: int = 20,
    max_link_distance_mm: float = 28.0,
) -> list[BranchCandidate]:
    """Find off-axis CTA mask components and classify whether they look branch-like.

    ImageTBAD-style masks often contain true/false lumen channels rather than named
    side branches. This routine deliberately rejects long axial channels so they are
    not mislabeled as renal/hepatic/iliac anatomy.
    """

    tracks: list[list[dict[str, Any]]] = []
    active: dict[int, int] = {}
    spacing = np.asarray(spacing_mm, dtype=float)

    for z_index in range(mask.shape[2]):
        slice_mask = mask[:, :, z_index]
        labels, component_count = ndi.label(slice_mask)
        if component_count <= 1:
            active = {}
            continue

        z_mm = float(z_index * spacing[2])
        trunk_center, trunk_radius = _interp_centerline(centerline, z_mm)
        components: list[dict[str, Any]] = []
        for label_id in range(1, component_count + 1):
            coords = np.argwhere(labels == label_id)
            if len(coords) < min_component_voxels:
                continue
            xy_mm = coords.astype(float) * spacing[:2]
            centroid_xy = xy_mm.mean(axis=0)
            area_mm2 = float(len(coords) * spacing[0] * spacing[1])
            radius_mm = math.sqrt(max(area_mm2, 1e-6) / math.pi)
            trunk_distance = float(np.linalg.norm(centroid_xy - np.asarray(trunk_center[:2], dtype=float)))
            components.append(
                {
                    "point_mm": (float(centroid_xy[0]), float(centroid_xy[1]), z_mm),
                    "radius_mm": float(radius_mm),
                    "voxel_count": int(len(coords)),
                    "trunk_distance_mm": trunk_distance,
                    "trunk_radius_mm": float(trunk_radius),
                }
            )

        if len(components) <= 1:
            active = {}
            continue

        # Treat the largest component as the dominant aortic lumen for that slice.
        components.sort(key=lambda item: int(item["voxel_count"]), reverse=True)
        off_axis_components = components[1:]
        next_active: dict[int, int] = {}
        used_tracks: set[int] = set()
        for component in off_axis_components:
            point = np.asarray(component["point_mm"], dtype=float)
            best_track: int | None = None
            best_distance = float("inf")
            for track_index in active.values():
                if track_index in used_tracks:
                    continue
                last_point = np.asarray(tracks[track_index][-1]["point_mm"], dtype=float)
                distance = float(np.linalg.norm(point - last_point))
                if distance < best_distance and distance <= max_link_distance_mm:
                    best_distance = distance
                    best_track = track_index
            if best_track is None:
                tracks.append([component])
                best_track = len(tracks) - 1
            else:
                tracks[best_track].append(component)
            used_tracks.add(best_track)
            next_active[best_track] = best_track
        active = next_active

    candidates: list[BranchCandidate] = []
    for index, track in enumerate(tracks, start=1):
        points = np.asarray([item["point_mm"] for item in track], dtype=float)
        if len(points) >= 3:
            points[:, 0] = _smooth(points[:, 0], lambda values, **_: values)
        z_span = float(points[:, 2].max() - points[:, 2].min()) if len(points) else 0.0
        xy_extent = points[:, :2].max(axis=0) - points[:, :2].min(axis=0) if len(points) else np.zeros(2)
        xy_span = float(np.linalg.norm(xy_extent))
        length = _line_length(points)
        mean_radius = float(np.mean([item["radius_mm"] for item in track])) if track else 0.0
        mean_trunk_distance = float(np.mean([item["trunk_distance_mm"] for item in track])) if track else 0.0

        if len(points) < 3:
            classification = "rejected_short_component_track"
            reason = "fewer_than_three_axial_samples"
        elif z_span > 40.0 and xy_span < 80.0:
            classification = "rejected_longitudinal_lumen_channel"
            reason = "track_runs_axially_like_aortic_true_false_lumen_not_named_branch"
        elif length < 12.0:
            classification = "rejected_short_component_track"
            reason = "centerline_length_below_branch_threshold"
        elif mean_trunk_distance < max(10.0, mean_radius * 1.25):
            classification = "rejected_near_trunk_component"
            reason = "component_too_close_to_aortic_lumen"
        else:
            classification = "plausible_cta_branch_candidate"
            reason = ""

        candidates.append(
            BranchCandidate(
                candidate_id=f"cta_branch_candidate_{index:02d}",
                point_count=int(len(points)),
                z_span_mm=z_span,
                xy_span_mm=xy_span,
                length_mm=length,
                mean_radius_mm=mean_radius,
                mean_trunk_distance_mm=mean_trunk_distance,
                classification=classification,
                promoted_edge_id=None,
                rejection_reason=reason,
                polyline_mm=tuple(tuple(float(value) for value in point) for point in points),
            )
        )
    return candidates


def _promote_branch_candidates(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    candidates: list[BranchCandidate],
) -> tuple[list[BranchCandidate], int]:
    lookup = _node_lookup({"nodes": nodes})
    available = [candidate for candidate in candidates if candidate.classification == "plausible_cta_branch_candidate"]
    used: set[str] = set()
    promoted_candidates: list[BranchCandidate] = []
    promoted_edges = 0
    branch_roles = {
        "iliac_branch",
        "renal_branch",
        "hepatic_placeholder_branch",
        "splenic_placeholder_branch",
    }

    for edge in edges:
        flow_role = str(edge.get("flow_role", ""))
        if flow_role not in branch_roles:
            continue
        source = lookup.get(str(edge.get("source")))
        target = lookup.get(str(edge.get("target")))
        if source is None or target is None:
            continue
        source_position = np.asarray(source.get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
        target_position = np.asarray(target.get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
        edge_direction = target_position - source_position
        edge_norm = float(np.linalg.norm(edge_direction))
        if edge_norm == 0.0:
            continue
        edge_direction = edge_direction / edge_norm

        best: tuple[float, BranchCandidate, np.ndarray] | None = None
        for candidate in available:
            if candidate.candidate_id in used or len(candidate.polyline_mm) < 2:
                continue
            points = np.asarray(candidate.polyline_mm, dtype=float)
            start_first = float(np.linalg.norm(points[0] - source_position))
            start_last = float(np.linalg.norm(points[-1] - source_position))
            if start_last < start_first:
                points = points[::-1]
                start_distance = start_last
            else:
                start_distance = start_first
            candidate_direction = points[-1] - points[0]
            candidate_norm = float(np.linalg.norm(candidate_direction))
            if candidate_norm == 0.0:
                continue
            alignment = float(np.dot(candidate_direction / candidate_norm, edge_direction))
            endpoint_distance = float(np.linalg.norm(points[-1] - target_position))
            z_overlap = abs(float(points[0, 2]) - float(source_position[2]))
            if start_distance > 35.0 or endpoint_distance > 90.0 or alignment < 0.25 or z_overlap > 45.0:
                continue
            score = start_distance + 0.45 * endpoint_distance + 20.0 * (1.0 - alignment)
            if best is None or score < best[0]:
                best = (score, candidate, points)

        if best is None:
            continue
        _, candidate, points = best
        used.add(candidate.candidate_id)
        target["position_mm"] = [float(value) for value in points[-1]]
        target["radius_mm"] = max(1.0, float(candidate.mean_radius_mm))
        target_notes = list(target.get("notes", []))
        target_notes.append("position_replaced_from_cta_branch_candidate")
        target["notes"] = sorted(set(str(item) for item in target_notes))
        edge["polyline_mm"] = [[float(value) for value in source_position], *[[float(value) for value in point] for point in points]]
        edge["radius_start_mm"] = float(source.get("radius_mm", edge.get("radius_start_mm", 0.0)))
        edge["radius_end_mm"] = float(target.get("radius_mm", edge.get("radius_end_mm", 0.0)))
        notes = list(edge.get("notes", []))
        notes.append("polyline_replaced_from_cta_branch_candidate")
        edge["notes"] = sorted(set(str(item) for item in notes))
        edge["length_mm"] = _line_length(edge.get("polyline_mm", []))
        promoted_edges += 1
        promoted_candidates.append(
            BranchCandidate(
                candidate_id=candidate.candidate_id,
                point_count=candidate.point_count,
                z_span_mm=candidate.z_span_mm,
                xy_span_mm=candidate.xy_span_mm,
                length_mm=candidate.length_mm,
                mean_radius_mm=candidate.mean_radius_mm,
                mean_trunk_distance_mm=candidate.mean_trunk_distance_mm,
                classification="promoted_cta_branch_candidate",
                promoted_edge_id=str(edge.get("id", "")),
                rejection_reason="",
                polyline_mm=candidate.polyline_mm,
            )
        )

    promoted_ids = {candidate.candidate_id for candidate in promoted_candidates}
    merged: list[BranchCandidate] = []
    for candidate in candidates:
        promoted = next((item for item in promoted_candidates if item.candidate_id == candidate.candidate_id), None)
        if promoted is not None:
            merged.append(promoted)
        elif candidate.classification == "plausible_cta_branch_candidate" and candidate.candidate_id not in promoted_ids:
            merged.append(
                BranchCandidate(
                    candidate_id=candidate.candidate_id,
                    point_count=candidate.point_count,
                    z_span_mm=candidate.z_span_mm,
                    xy_span_mm=candidate.xy_span_mm,
                    length_mm=candidate.length_mm,
                    mean_radius_mm=candidate.mean_radius_mm,
                    mean_trunk_distance_mm=candidate.mean_trunk_distance_mm,
                    classification="rejected_unmatched_branch_candidate",
                    promoted_edge_id=None,
                    rejection_reason="no_existing_branch_edge_matched_candidate_geometry",
                    polyline_mm=candidate.polyline_mm,
                )
            )
        else:
            merged.append(candidate)
    return merged, promoted_edges


def _node_lookup(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in graph.get("nodes", []) if isinstance(node, dict) and "id" in node}


def _is_coarse_aorta_edge(edge: dict[str, Any], edges: list[dict[str, Any]], metadata: dict[str, Any]) -> bool:
    if str(edge.get("flow_role", "")) != "aorta_trunk":
        return False
    edge_id = str(edge.get("id", "")).lower()
    if "coarse_aorta" in edge_id:
        return True
    product_scope = str(metadata.get("product_scope", "")).lower()
    coarse_mode = str(metadata.get("coarse_vessel_mode", "")).lower()
    if "coarse_major" in product_scope or "btcv_major" in coarse_mode:
        return True
    trunk_count = sum(1 for item in edges if str(item.get("flow_role", "")) == "aorta_trunk")
    return trunk_count == 1


def _full_trunk_endpoint(centerline: np.ndarray, source_position: list[float]) -> tuple[list[float], float]:
    z_values = centerline[:, 2]
    source_z = float(source_position[2])
    z_min = float(z_values.min())
    z_max = float(z_values.max())
    midpoint = (z_min + z_max) / 2.0
    target_z = z_min if source_z >= midpoint else z_max
    return _interp_centerline(centerline, target_z)


def _write_nodes_csv(path: Path, nodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["node_id", "kind", "role", "x_mm", "y_mm", "z_mm", "radius_mm", "boundary_role", "source"])
        for node in nodes:
            position = node.get("position_mm", [0.0, 0.0, 0.0])
            notes = tuple(str(item) for item in node.get("notes", []))
            if "position_replaced_from_cta_branch_candidate" in notes:
                source = "cta_branch_candidate"
            elif "position_replaced_from_cta_lumen_centerline" in notes:
                source = "cta_centerline"
            elif "endpoint_retained_after_landmark_registered_labeled_vessel_fit" in notes:
                source = "registered_labeled_vessel_endpoint"
            elif "endpoint_retained_graph_anchor_for_labeled_template" in notes:
                source = "labeled_vessel_template_endpoint"
            else:
                source = "synthetic_or_placeholder"
            writer.writerow(
                [
                    node.get("id", ""),
                    node.get("kind", ""),
                    node.get("role", ""),
                    *[f"{float(value):.6f}" for value in position],
                    f"{float(node.get('radius_mm', 0.0)):.6f}",
                    node.get("boundary_role", ""),
                    source,
                ]
            )


def _write_edges_csv(path: Path, edges: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["edge_id", "source", "target", "vessel_type", "flow_role", "length_mm", "point_count", "source"])
        for edge in edges:
            notes = tuple(str(item) for item in edge.get("notes", []))
            if "polyline_replaced_from_cta_branch_candidate" in notes:
                source = "cta_branch_candidate"
            elif "polyline_replaced_from_cta_lumen_centerline" in notes:
                source = "cta_centerline"
            elif "polyline_replaced_from_registered_labeled_vessel_centerline" in notes:
                source = "registered_labeled_vessel"
            elif "polyline_replaced_from_labeled_vessel_template" in notes:
                source = "labeled_vessel_template"
            else:
                source = "synthetic_or_placeholder"
            writer.writerow(
                [
                    edge.get("id", ""),
                    edge.get("source", ""),
                    edge.get("target", ""),
                    edge.get("vessel_type", ""),
                    edge.get("flow_role", ""),
                    f"{float(edge.get('length_mm', 0.0)):.6f}",
                    len(edge.get("polyline_mm", [])),
                    source,
                ]
            )


def _write_branch_candidates_csv(path: Path, candidates: list[BranchCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "candidate_id",
                "classification",
                "promoted_edge_id",
                "rejection_reason",
                "point_count",
                "z_span_mm",
                "xy_span_mm",
                "length_mm",
                "mean_radius_mm",
                "mean_trunk_distance_mm",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.candidate_id,
                    candidate.classification,
                    candidate.promoted_edge_id or "",
                    candidate.rejection_reason,
                    candidate.point_count,
                    f"{candidate.z_span_mm:.6f}",
                    f"{candidate.xy_span_mm:.6f}",
                    f"{candidate.length_mm:.6f}",
                    f"{candidate.mean_radius_mm:.6f}",
                    f"{candidate.mean_trunk_distance_mm:.6f}",
                ]
            )


def _write_preview(
    path: Path,
    baseline_graph: dict[str, Any],
    derived_graph: dict[str, Any],
    centerline: np.ndarray,
    branch_candidates: list[BranchCandidate],
) -> None:
    plt, *_ = _import_dependencies()
    fig = plt.figure(figsize=(9.5, 7.5), dpi=170)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")

    for edge in baseline_graph.get("edges", []):
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if len(points) >= 2:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#9aa0a6", linewidth=1.2, alpha=0.35)

    for edge in derived_graph.get("edges", []):
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if len(points) < 2:
            continue
        notes = tuple(str(item) for item in edge.get("notes", []))
        if "polyline_replaced_from_cta_lumen_centerline" in notes:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#dc3b2a", linewidth=3.0, alpha=0.95)
        elif "polyline_replaced_from_cta_branch_candidate" in notes:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#0f9f6e", linewidth=2.7, alpha=0.95)
        else:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#2878b8", linewidth=1.7, alpha=0.70, linestyle="--")

    for candidate in branch_candidates:
        points = np.asarray(candidate.polyline_mm, dtype=float)
        if len(points) < 2:
            continue
        if candidate.classification == "promoted_cta_branch_candidate":
            color = "#0f9f6e"
            alpha = 0.9
            linewidth = 2.2
        else:
            color = "#f59e0b"
            alpha = 0.35
            linewidth = 1.2
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth, alpha=alpha)

    ax.scatter(centerline[:, 0], centerline[:, 1], centerline[:, 2], s=7, color="#111827", alpha=0.55, label="CTA mask centerline samples")
    all_points = np.vstack(
        [
            centerline[:, :3],
            *[
                np.asarray(candidate.polyline_mm, dtype=float)
                for candidate in branch_candidates
                if len(candidate.polyline_mm) >= 2
            ],
            *[
                np.asarray(edge.get("polyline_mm", []), dtype=float)
                for edge in derived_graph.get("edges", [])
                if len(edge.get("polyline_mm", [])) >= 2
            ],
        ]
    )
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0) * 1.12
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_title(
        "CTA-Derived Vascular Graph\nred = mask-derived trunk, green = promoted branch, orange = rejected candidate",
        pad=16,
    )
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _format_report(result: CtaDerivedGraphResult) -> str:
    return "\n".join(
        [
            "# CTA-Derived Vascular Graph Stage 002",
            "",
            f"Case ID: `{result.case_id}`",
            "",
            "## Summary",
            "",
            f"- CTA centerline points: {result.centerline_points}",
            f"- CTA centerline length: {result.centerline_length_mm:.2f} mm",
            f"- Replaced nodes: {result.replaced_node_count}",
            f"- Replaced trunk edges: {result.replaced_edge_count}",
            f"- CTA branch candidates detected: {result.branch_candidate_count}",
            f"- CTA branch candidates promoted: {result.promoted_branch_count}",
            f"- CTA branch candidates rejected: {result.rejected_branch_count}",
            f"- Retained synthetic/placeholder edges: {result.retained_synthetic_edge_count}",
            "",
            "## Outputs",
            "",
            f"- Graph YAML: `{Path(result.graph_yaml_path).name}`",
            f"- Nodes CSV: `{Path(result.nodes_csv_path).name}`",
            f"- Edges CSV: `{Path(result.edges_csv_path).name}`",
            f"- Branch candidate CSV: `{Path(result.branch_candidates_csv_path).name}`",
            f"- Preview PNG: `{Path(result.preview_png_path).name}`",
            "",
            "## Interpretation",
            "",
            "- The aorta trunk geometry is replaced from the embedded CTA-derived vascular-fluid mask centerline.",
            "- Off-axis CTA mask components are tracked as branch candidates, but only anatomically plausible branch-like traces are promoted.",
            "- Long axial side channels are rejected because ImageTBAD labels can represent true/false aortic lumen channels rather than named side branches.",
            "- Iliac, renal, hepatic/splenic, and venous return paths remain simulation placeholders if no promoted CTA branch candidate is available.",
            "- This is a stronger vascular geometry input than the original all-synthetic trunk, but it is still not a complete patient-specific vascular network.",
            "",
            "## Notes",
            *[f"- {note}" for note in result.notes],
        ]
    )


def build_cta_derived_vascular_graph(
    baseline_graph_path: str | Path,
    vascular_mask_path: str | Path,
    output_dir: str | Path = "outputs/digital/vascular_network_cta_derived",
    case_id: str = "ct_org_case0_imagetbad_case125_cta_derived",
    use_full_trunk_for_coarse_aorta: bool = True,
    report_path: str | Path | None = "outputs/reports/cta_derived_vascular_graph_stage002.md",
) -> CtaDerivedGraphResult:
    _, nib, yaml, savgol_filter, ndi = _import_dependencies()
    baseline_path = Path(baseline_graph_path)
    baseline_graph = _load_yaml(baseline_path)
    mask_image = nib.load(str(vascular_mask_path))
    mask = np.asanyarray(mask_image.dataobj) > 0
    spacing_mm = tuple(float(value) for value in mask_image.header.get_zooms()[:3])
    centerline = _centerline_from_mask(mask, spacing_mm, savgol_filter)
    branch_candidates = _slice_component_tracks(mask, spacing_mm, centerline, ndi)
    z_min = float(centerline[:, 2].min())
    z_max = float(centerline[:, 2].max())

    graph = dict(baseline_graph)
    nodes = [dict(node) for node in baseline_graph.get("nodes", [])]
    edges = [dict(edge) for edge in baseline_graph.get("edges", [])]
    lookup = _node_lookup({"nodes": nodes})
    trunk_node_ids = {
        "aorta_inlet",
        "descending_aorta_mid",
        "visceral_branch_origin",
        "renal_branch_origin",
        "aorta_distal_anchor",
    }
    replaced_nodes = 0
    for node_id in trunk_node_ids:
        node = lookup.get(node_id)
        if node is None:
            continue
        original_position = [float(value) for value in node.get("position_mm", [0.0, 0.0, 0.0])]
        if not (z_min <= original_position[2] <= z_max):
            continue
        new_position, radius = _interp_centerline(centerline, original_position[2])
        node["position_mm"] = new_position
        node["radius_mm"] = radius
        notes = list(node.get("notes", []))
        notes.append("position_replaced_from_cta_lumen_centerline")
        node["notes"] = sorted(set(str(item) for item in notes))
        replaced_nodes += 1

    lookup = _node_lookup({"nodes": nodes})
    replaced_edges = 0
    retained_edges = 0
    full_trunk_replaced_edges = 0
    metadata = dict(graph.get("graph_metadata", {}))
    for edge in edges:
        source = lookup[str(edge.get("source"))]
        target = lookup[str(edge.get("target"))]
        source_position = [float(value) for value in source.get("position_mm", [0.0, 0.0, 0.0])]
        target_position = [float(value) for value in target.get("position_mm", [0.0, 0.0, 0.0])]
        source_real = "position_replaced_from_cta_lumen_centerline" in tuple(str(item) for item in source.get("notes", []))
        target_real = "position_replaced_from_cta_lumen_centerline" in tuple(str(item) for item in target.get("notes", []))
        flow_role = str(edge.get("flow_role", ""))
        notes = list(edge.get("notes", []))
        full_coarse_trunk = bool(use_full_trunk_for_coarse_aorta and _is_coarse_aorta_edge(edge, edges, metadata) and source_real)
        if full_coarse_trunk:
            target_position, target_radius = _full_trunk_endpoint(centerline, source_position)
            target["position_mm"] = target_position
            target["radius_mm"] = target_radius
            target_notes = list(target.get("notes", []))
            target_notes.append("position_replaced_from_full_cta_lumen_centerline_for_coarse_trunk")
            target["notes"] = sorted(set(str(item) for item in target_notes))
            edge["polyline_mm"] = _centerline_segment(centerline, source_position, target_position)
            edge["radius_start_mm"] = float(source.get("radius_mm", edge.get("radius_start_mm", 0.0)))
            edge["radius_end_mm"] = float(target_radius)
            notes.append("polyline_replaced_from_full_cta_lumen_centerline_for_coarse_trunk")
            notes.append("polyline_replaced_from_cta_lumen_centerline")
            replaced_edges += 1
            full_trunk_replaced_edges += 1
        elif flow_role == "aorta_trunk" and source_real and target_real:
            edge["polyline_mm"] = _centerline_segment(centerline, source_position, target_position)
            edge["radius_start_mm"] = float(source.get("radius_mm", edge.get("radius_start_mm", 0.0)))
            edge["radius_end_mm"] = float(target.get("radius_mm", edge.get("radius_end_mm", 0.0)))
            notes.append("polyline_replaced_from_cta_lumen_centerline")
            replaced_edges += 1
        else:
            polyline = edge.get("polyline_mm", [])
            if isinstance(polyline, list) and len(polyline) >= 2:
                updated = [list(map(float, point)) for point in polyline]
                updated[0] = source_position
                updated[-1] = target_position
                edge["polyline_mm"] = updated
            retained_edges += 1
        edge["length_mm"] = _line_length(edge.get("polyline_mm", []))
        edge["notes"] = sorted(set(str(item) for item in notes))

    branch_candidates, promoted_branch_edges = _promote_branch_candidates(nodes, edges, branch_candidates)
    retained_edges = 0
    for edge in edges:
        notes = tuple(str(item) for item in edge.get("notes", []))
        if (
            "polyline_replaced_from_cta_lumen_centerline" not in notes
            and "polyline_replaced_from_cta_branch_candidate" not in notes
        ):
            retained_edges += 1

    metadata.update(
        {
            "source_graph": str(baseline_graph_path),
            "source_vascular_mask": str(vascular_mask_path),
            "cta_centerline_points": int(len(centerline)),
            "cta_centerline_length_mm": _line_length(centerline[:, :3]),
            "replaced_node_count": replaced_nodes,
            "replaced_trunk_edge_count": replaced_edges,
            "full_trunk_coarse_aorta_replaced_edge_count": full_trunk_replaced_edges,
            "coarse_aorta_full_trunk_mode": bool(use_full_trunk_for_coarse_aorta),
            "cta_branch_candidate_count": len(branch_candidates),
            "promoted_branch_edge_count": promoted_branch_edges,
            "rejected_branch_candidate_count": len(branch_candidates) - promoted_branch_edges,
            "retained_synthetic_edge_count": retained_edges,
            "geometry_status": (
                "cta_derived_trunk_with_promoted_branch_candidates"
                if promoted_branch_edges
                else "cta_derived_aorta_trunk_with_retained_placeholder_branches"
            ),
        }
    )
    graph.update(
        {
            "case_id": case_id,
            "scaffold_type": "cta_derived_aorta_trunk_with_synthetic_branch_placeholders",
            "source_baseline_graph": str(baseline_graph_path),
            "source_vascular_mask": str(vascular_mask_path),
            "graph_metadata": metadata,
            "provenance_notes": [
                "aorta_trunk_centerline_replaced_from_embedded_cta_derived_vascular_mask",
                (
                    "coarse_aorta_trunk_replaced_with_full_cta_centerline_extent"
                    if full_trunk_replaced_edges
                    else "coarse_aorta_trunk_full_extent_not_applied"
                ),
                "off_axis_mask_components_were_screened_as_cta_branch_candidates",
                (
                    "some_branch_edges_replaced_from_cta_branch_candidates"
                    if promoted_branch_edges
                    else "no_anatomically_plausible_branch_candidates_promoted_from_current_mask"
                ),
                "stage_branch_rich_cta_or_ctv_segmentations_to_replace_remaining_placeholder_edges",
            ],
            "nodes": nodes,
            "edges": edges,
        }
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_yaml = output / f"{case_id}_cta_derived_vascular_graph_v001.yaml"
    nodes_csv = output / f"{case_id}_cta_derived_vascular_graph_nodes_v001.csv"
    edges_csv = output / f"{case_id}_cta_derived_vascular_graph_edges_v001.csv"
    branch_candidates_csv = output / f"{case_id}_cta_branch_candidates_v001.csv"
    preview = output / f"{case_id}_cta_derived_vascular_graph_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_cta_derived_vascular_graph_report_v001.md"

    graph_yaml.write_text(yaml.safe_dump(graph, sort_keys=False))
    _write_nodes_csv(nodes_csv, nodes)
    _write_edges_csv(edges_csv, edges)
    _write_branch_candidates_csv(branch_candidates_csv, branch_candidates)
    _write_preview(preview, baseline_graph, graph, centerline, branch_candidates)
    result = CtaDerivedGraphResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_yaml),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        branch_candidates_csv_path=str(branch_candidates_csv),
        preview_png_path=str(preview),
        report_path=str(report),
        centerline_points=int(len(centerline)),
        centerline_length_mm=_line_length(centerline[:, :3]),
        replaced_node_count=replaced_nodes,
        replaced_edge_count=replaced_edges,
        branch_candidate_count=len(branch_candidates),
        promoted_branch_count=promoted_branch_edges,
        rejected_branch_count=len(branch_candidates) - promoted_branch_edges,
        retained_synthetic_edge_count=retained_edges,
        notes=(
            "kaggle_single_file_download_unavailable_without_authentication_in_this_session",
            "used_existing_embedded_imagetbad_cta_derived_vascular_mask_as_real_geometry_source",
            (
                "coarse_aorta_trunk_uses_full_cta_centerline_extent"
                if full_trunk_replaced_edges
                else "coarse_aorta_full_trunk_extent_not_used"
            ),
            (
                "branch_candidates_promoted_from_current_cta_mask"
                if promoted_branch_edges
                else "current_cta_mask_did_not_contain_promotable_named_branch_centerlines"
            ),
            "branch_placeholder_replacement_requires_additional_branch_rich_cta_or_ctv_segmentations",
        ),
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result) + "\n")
    return result


def format_cta_derived_vascular_graph_result(result: CtaDerivedGraphResult) -> str:
    return "\n".join(
        [
            "CTA-derived vascular graph built",
            f"Case ID: {result.case_id}",
            f"Centerline points: {result.centerline_points}",
            f"Replaced nodes/edges: {result.replaced_node_count}/{result.replaced_edge_count}",
            f"Branch candidates promoted/rejected: {result.promoted_branch_count}/{result.rejected_branch_count}",
            f"Graph YAML: {result.graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
        ]
    )
