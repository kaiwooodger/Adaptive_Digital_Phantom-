from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np

from .cta_vascular_graph import _load_yaml, _line_length, _node_lookup, _write_edges_csv, _write_nodes_csv


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Labeled vessel graph replacement requires matplotlib, nibabel, and PyYAML.") from exc
    return plt, nib, yaml


@dataclass(frozen=True)
class MedsegVascularStagingResult:
    case_id: str
    image_path: str
    mask_path: str
    label_config_path: str
    manifest_path: str
    label_summary_csv_path: str
    preview_png_path: str
    report_path: str
    label_count: int
    populated_label_count: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class LabeledVesselGraphResult:
    case_id: str
    output_dir: str
    graph_yaml_path: str
    nodes_csv_path: str
    edges_csv_path: str
    replacements_csv_path: str
    preview_png_path: str
    report_path: str
    attempted_replacements: int
    successful_replacements: int
    retained_edges: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RegisteredLabeledVesselGraphResult:
    case_id: str
    output_dir: str
    graph_yaml_path: str
    nodes_csv_path: str
    edges_csv_path: str
    replacements_csv_path: str
    landmarks_csv_path: str
    transform_csv_path: str
    preview_png_path: str
    report_path: str
    attempted_replacements: int
    successful_replacements: int
    retained_edges: int
    landmark_count: int
    registration_rms_error_mm: float
    registration_max_error_mm: float
    deformable_registration_rms_error_mm: float
    deformable_registration_max_error_mm: float
    notes: tuple[str, ...]


def _parse_medseg_conf(path: str | Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in Path(path).read_text().splitlines():
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 2:
            continue
        try:
            label = int(parts[0])
        except ValueError:
            continue
        labels[label] = parts[1]
    return labels


def _points_for_labels(mask: np.ndarray, affine: np.ndarray, label_ids: tuple[int, ...]) -> np.ndarray:
    selector = np.isin(mask, np.asarray(label_ids, dtype=int))
    ijk = np.argwhere(selector)
    if len(ijk) == 0:
        return np.empty((0, 3), dtype=float)
    points = np.c_[ijk.astype(float), np.ones(len(ijk), dtype=float)] @ affine.T
    return points[:, :3].astype(float)


def _centerline_from_points(points: np.ndarray, *, min_bins: int = 6, max_bins: int = 44) -> np.ndarray:
    if len(points) < 2:
        return points.astype(float)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = vh[0]
    projection = (points - centroid) @ axis
    p_min = float(projection.min())
    p_max = float(projection.max())
    if math.isclose(p_min, p_max):
        return np.vstack([points[0], points[-1]]).astype(float)

    length = p_max - p_min
    bin_count = int(np.clip(math.ceil(length / 4.0), min_bins, max_bins))
    bins = np.linspace(p_min, p_max, bin_count + 1)
    centerline: list[np.ndarray] = []
    for index in range(bin_count):
        if index == bin_count - 1:
            selected = (projection >= bins[index]) & (projection <= bins[index + 1])
        else:
            selected = (projection >= bins[index]) & (projection < bins[index + 1])
        if np.any(selected):
            centerline.append(points[selected].mean(axis=0))
    if len(centerline) < 2:
        return np.vstack([points[projection.argmin()], points[projection.argmax()]]).astype(float)
    return np.asarray(centerline, dtype=float)


def _estimate_radius_mm(points: np.ndarray, centerline: np.ndarray) -> float:
    if len(points) == 0 or len(centerline) < 2:
        return 1.0
    length = max(_line_length(centerline), 1.0)
    # Voxel volume is unknown here because labels may be combined; point count gives a
    # stable relative estimate that is clamped against existing graph radii later.
    volume_proxy = float(len(points))
    return max(0.6, math.sqrt(volume_proxy / (math.pi * length)))


def _ordered_centerline(points: np.ndarray, reference_points: np.ndarray | None) -> np.ndarray:
    centerline = _centerline_from_points(points)
    if len(centerline) < 2 or reference_points is None or len(reference_points) == 0:
        return centerline
    start_distance = float(np.linalg.norm(reference_points - centerline[0], axis=1).min())
    end_distance = float(np.linalg.norm(reference_points - centerline[-1], axis=1).min())
    if end_distance < start_distance:
        centerline = centerline[::-1]
    return centerline


def _label_points(mask: np.ndarray, affine: np.ndarray, label_ids: tuple[int, ...]) -> np.ndarray:
    return _points_for_labels(mask, affine, label_ids)


def _centroid(points: np.ndarray) -> np.ndarray | None:
    if len(points) == 0:
        return None
    return points.mean(axis=0)


def _axis_fraction_centroid(points: np.ndarray, axis: int, high: bool, fraction: float = 0.10) -> np.ndarray | None:
    if len(points) == 0:
        return None
    values = points[:, axis]
    threshold = np.quantile(values, 1.0 - fraction if high else fraction)
    selected = points[values >= threshold] if high else points[values <= threshold]
    if len(selected) == 0:
        selected = points
    return selected.mean(axis=0)


def _nearest_axis_fraction_centroid(points: np.ndarray, target: np.ndarray, axis: int, fraction: float = 0.16) -> np.ndarray | None:
    if len(points) == 0:
        return None
    values = np.abs(points[:, axis] - target[axis])
    threshold = np.quantile(values, fraction)
    selected = points[values <= threshold]
    if len(selected) == 0:
        selected = points
    return selected.mean(axis=0)


def _axis_quantile_centroid(points: np.ndarray, axis: int, quantile: float, window_fraction: float = 0.10) -> np.ndarray | None:
    if len(points) == 0:
        return None
    values = points[:, axis]
    center = float(np.quantile(values, quantile))
    half_width = max((float(values.max()) - float(values.min())) * window_fraction * 0.5, 1e-6)
    selected = points[np.abs(values - center) <= half_width]
    if len(selected) == 0:
        selected = points[np.argsort(np.abs(values - center))[: max(1, len(points) // 10)]]
    return selected.mean(axis=0)


def _source_landmarks_from_mask(mask: np.ndarray, affine: np.ndarray) -> dict[str, np.ndarray]:
    aorta = _label_points(mask, affine, (4,))
    ivc = _label_points(mask, affine, (1,))
    left_iliac = _label_points(mask, affine, (2,))
    right_iliac = _label_points(mask, affine, (3,))
    left_renal_artery = _label_points(mask, affine, (28,))
    right_renal_artery = _label_points(mask, affine, (27,))
    celiac = _label_points(mask, affine, (5,))
    hepatic_artery = _label_points(mask, affine, (6, 8))
    splenic_artery = _label_points(mask, affine, (13,))
    left_renal_vein = _label_points(mask, affine, (24,))
    right_renal_vein = _label_points(mask, affine, (25,))
    hepatic_veins = _label_points(mask, affine, (33, 34, 35))
    splenic_portal = _label_points(mask, affine, (21, 14))
    right_iliac_vein = _label_points(mask, affine, (43,))

    source: dict[str, np.ndarray] = {}
    candidates = {
        "aorta_inlet": _axis_fraction_centroid(aorta, 2, True),
        "descending_aorta_mid": _axis_quantile_centroid(aorta, 2, 0.72),
        "visceral_branch_origin": _centroid(celiac),
        "renal_branch_origin": _centroid(np.vstack([arr for arr in (left_renal_artery, right_renal_artery) if len(arr)]))
        if len(left_renal_artery) or len(right_renal_artery)
        else None,
        "aorta_distal_anchor": _axis_fraction_centroid(aorta, 2, False),
        "aortic_bifurcation": _centroid(np.vstack([arr for arr in (left_iliac, right_iliac) if len(arr)])) if len(left_iliac) or len(right_iliac) else None,
        "left_common_iliac_outlet": _centroid(left_iliac),
        "right_common_iliac_outlet": _centroid(right_iliac),
        "left_renal_outlet": _centroid(left_renal_artery),
        "right_renal_outlet": _centroid(right_renal_artery),
        "hepatic_placeholder_outlet": _centroid(hepatic_artery),
        "splenic_placeholder_outlet": _centroid(splenic_artery),
        "ivc_lower_return_inlet": _centroid(right_iliac_vein) if len(right_iliac_vein) else _axis_fraction_centroid(ivc, 2, False),
        "ivc_bifurcation_return": _axis_fraction_centroid(ivc, 2, False),
        "ivc_renal_junction": _nearest_axis_fraction_centroid(ivc, _centroid(np.vstack([arr for arr in (left_renal_vein, right_renal_vein) if len(arr)])) if len(left_renal_vein) or len(right_renal_vein) else np.zeros(3), 2),
        "ivc_hepatic_junction": _nearest_axis_fraction_centroid(ivc, _centroid(hepatic_veins) if len(hepatic_veins) else np.zeros(3), 2),
        "ivc_outlet": _axis_fraction_centroid(ivc, 2, True),
        "left_renal_vein_inlet": _centroid(left_renal_vein),
        "right_renal_vein_inlet": _centroid(right_renal_vein),
        "hepatic_venous_placeholder_inlet": _centroid(hepatic_veins),
        "splenic_venous_placeholder_inlet": _centroid(splenic_portal),
    }
    for key, value in candidates.items():
        if value is not None and np.all(np.isfinite(value)):
            source[key] = np.asarray(value, dtype=float)
    return source


def _source_landmarks_from_edge_templates(mask: np.ndarray, affine: np.ndarray, label_config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Use labelled-vessel centerline endpoints as source landmarks.

    Centroids are poor landmarks for long vessels. For example, a renal artery
    centroid sits midway along the branch, but the graph landmark is the outlet.
    This endpoint-based pass overrides centroid estimates for mapped branches.
    """

    edge_mapping = label_config.get("graph_edge_mapping", {})
    if not isinstance(edge_mapping, dict):
        return {}
    source: dict[str, list[np.ndarray]] = {}
    aorta_reference = _points_for_labels(mask, affine, (4,))
    ivc_reference = _points_for_labels(mask, affine, (1,))

    arterial_parent_edges = {
        "bifurcation_to_left_common_iliac",
        "bifurcation_to_right_common_iliac",
        "renal_origin_to_left_renal",
        "renal_origin_to_right_renal",
        "visceral_origin_to_hepatic_placeholder",
        "visceral_origin_to_splenic_placeholder",
    }
    edge_nodes = {
        "bifurcation_to_left_common_iliac": ("aortic_bifurcation", "left_common_iliac_outlet"),
        "bifurcation_to_right_common_iliac": ("aortic_bifurcation", "right_common_iliac_outlet"),
        "renal_origin_to_left_renal": ("renal_branch_origin", "left_renal_outlet"),
        "renal_origin_to_right_renal": ("renal_branch_origin", "right_renal_outlet"),
        "visceral_origin_to_hepatic_placeholder": ("visceral_branch_origin", "hepatic_placeholder_outlet"),
        "visceral_origin_to_splenic_placeholder": ("visceral_branch_origin", "splenic_placeholder_outlet"),
        "left_renal_vein_to_ivc": ("left_renal_vein_inlet", "ivc_renal_junction"),
        "right_renal_vein_to_ivc": ("right_renal_vein_inlet", "ivc_renal_junction"),
        "hepatic_venous_placeholder_to_ivc": ("hepatic_venous_placeholder_inlet", "ivc_hepatic_junction"),
        "splenic_venous_placeholder_to_ivc": ("splenic_venous_placeholder_inlet", "ivc_hepatic_junction"),
    }

    for edge_id, node_pair in edge_nodes.items():
        mapping = edge_mapping.get(edge_id)
        if not isinstance(mapping, dict):
            continue
        label_ids = tuple(int(value) for value in mapping.get("labels", []))
        points = _points_for_labels(mask, affine, label_ids)
        if len(points) < 2:
            continue
        reference = aorta_reference if edge_id in arterial_parent_edges else ivc_reference
        centerline = _ordered_centerline(points, reference)
        if len(centerline) < 2:
            continue
        source_node, target_node = node_pair
        if edge_id in arterial_parent_edges:
            start, end = centerline[0], centerline[-1]
        else:
            # Venous graph edges generally flow from peripheral inlet to IVC.
            start, end = centerline[-1], centerline[0]
        source.setdefault(source_node, []).append(np.asarray(start, dtype=float))
        source.setdefault(target_node, []).append(np.asarray(end, dtype=float))

    merged: dict[str, np.ndarray] = {}
    for key, values in source.items():
        if values:
            merged[key] = np.vstack(values).mean(axis=0)
    return merged


def _target_landmarks_from_graph(graph: dict[str, Any]) -> dict[str, np.ndarray]:
    landmarks: dict[str, np.ndarray] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or "id" not in node:
            continue
        position = np.asarray(node.get("position_mm", []), dtype=float)
        if position.shape == (3,) and np.all(np.isfinite(position)):
            landmarks[str(node["id"])] = position
    return landmarks


def _fit_landmark_affine(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]], float, float]:
    common = sorted(set(source) & set(target))
    if len(common) < 4:
        raise ValueError(f"At least four common landmarks are required; found {len(common)}")
    src = np.vstack([source[key] for key in common])
    dst = np.vstack([target[key] for key in common])
    src_aug = np.c_[src, np.ones(len(src), dtype=float)]
    coeff, *_ = np.linalg.lstsq(src_aug, dst, rcond=None)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = coeff[:3, :]
    matrix[3, :3] = coeff[3, :]
    registered = src_aug @ coeff
    errors = np.linalg.norm(registered - dst, axis=1)
    rows = []
    for key, src_point, dst_point, reg_point, error in zip(common, src, dst, registered, errors):
        rows.append(
            {
                "landmark_id": key,
                "source": src_point,
                "target": dst_point,
                "registered": reg_point,
                "error_mm": float(error),
            }
        )
    rms = float(np.sqrt(np.mean(errors**2)))
    max_error = float(errors.max())
    return matrix, rows, rms, max_error


def _apply_row_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(float)
    transformed = np.c_[points.astype(float), np.ones(len(points), dtype=float)] @ matrix
    return transformed[:, :3]


def _landmark_residual_controls(landmark_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not landmark_rows:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    anchors = np.vstack([np.asarray(row["registered"], dtype=float) for row in landmark_rows])
    targets = np.vstack([np.asarray(row["target"], dtype=float) for row in landmark_rows])
    return anchors, targets - anchors


def _apply_landmark_residual_warp(
    points: np.ndarray,
    anchors: np.ndarray,
    displacements: np.ndarray,
    *,
    influence_radius_mm: float = 90.0,
    power: float = 2.0,
    top_k: int = 8,
) -> np.ndarray:
    """Apply a smooth inverse-distance residual warp after global affine registration."""

    if len(points) == 0 or len(anchors) == 0:
        return points.astype(float)
    warped: list[np.ndarray] = []
    radius = max(float(influence_radius_mm), 1e-6)
    neighbor_count = min(max(int(top_k), 1), len(anchors))
    for point in points.astype(float):
        distances = np.linalg.norm(anchors - point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < 1e-6:
            warped.append(point + displacements[nearest])
            continue
        neighbor_indices = np.argsort(distances)[:neighbor_count]
        local_distances = distances[neighbor_indices]
        weights = 1.0 / np.maximum(local_distances, 1e-6) ** power
        weights *= np.exp(-((local_distances / radius) ** 2))
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            warped.append(point)
            continue
        local_displacement = (weights[:, None] * displacements[neighbor_indices]).sum(axis=0) / weight_sum
        warped.append(point + local_displacement)
    return np.asarray(warped, dtype=float)


def _add_deformable_landmark_errors(
    landmark_rows: list[dict[str, Any]],
    anchors: np.ndarray,
    displacements: np.ndarray,
) -> tuple[float, float]:
    if not landmark_rows:
        return 0.0, 0.0
    errors: list[float] = []
    for row in landmark_rows:
        registered = np.asarray(row["registered"], dtype=float).reshape(1, 3)
        deformed = _apply_landmark_residual_warp(registered, anchors, displacements)[0]
        target = np.asarray(row["target"], dtype=float)
        error = float(np.linalg.norm(deformed - target))
        row["deformed"] = deformed
        row["deformable_error_mm"] = error
        errors.append(error)
    error_array = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(error_array**2))), float(error_array.max())


def _snap_registered_centerline(centerline: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    if len(centerline) < 2:
        return np.vstack([start, end])
    direct = float(np.linalg.norm(centerline[0] - start) + np.linalg.norm(centerline[-1] - end))
    reversed_distance = float(np.linalg.norm(centerline[-1] - start) + np.linalg.norm(centerline[0] - end))
    if reversed_distance < direct:
        centerline = centerline[::-1]
    t = np.linspace(0.0, 1.0, len(centerline))[:, None]
    start_delta = start - centerline[0]
    end_delta = end - centerline[-1]
    snapped = centerline + (1.0 - t) * start_delta + t * end_delta
    snapped[0] = start
    snapped[-1] = end
    return snapped


def _write_landmarks_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "landmark_id",
                "source_x_mm",
                "source_y_mm",
                "source_z_mm",
                "registered_x_mm",
                "registered_y_mm",
                "registered_z_mm",
                "deformed_x_mm",
                "deformed_y_mm",
                "deformed_z_mm",
                "target_x_mm",
                "target_y_mm",
                "target_z_mm",
                "affine_error_mm",
                "deformable_error_mm",
                "error_mm",
            ]
        )
        for row in rows:
            deformed = row.get("deformed", row["registered"])
            writer.writerow(
                [
                    row["landmark_id"],
                    *[f"{value:.6f}" for value in row["source"]],
                    *[f"{value:.6f}" for value in row["registered"]],
                    *[f"{value:.6f}" for value in deformed],
                    *[f"{value:.6f}" for value in row["target"]],
                    f"{row['error_mm']:.6f}",
                    f"{float(row.get('deformable_error_mm', row['error_mm'])):.6f}",
                    f"{row['error_mm']:.6f}",
                ]
            )


def _write_transform_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["row", "c0", "c1", "c2", "c3"])
        for index, row in enumerate(matrix):
            writer.writerow([index, *[f"{value:.10f}" for value in row]])


def _orthonormal_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1 = direction / max(float(np.linalg.norm(direction)), 1e-9)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(e1, reference))) > 0.88:
        reference = np.array([0.0, 1.0, 0.0])
    e2 = reference - e1 * float(np.dot(reference, e1))
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-9)
    e3 = np.cross(e1, e2)
    e3 = e3 / max(float(np.linalg.norm(e3)), 1e-9)
    return e1, e2, e3


def _template_centerline_to_edge(centerline: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    if len(centerline) < 2:
        return np.vstack([start, end])
    source_start = centerline[0]
    source_end = centerline[-1]
    source_chord = source_end - source_start
    source_length = max(float(np.linalg.norm(source_chord)), 1e-9)
    target_chord = end - start
    target_length = max(float(np.linalg.norm(target_chord)), 1e-9)
    e1s, e2s, e3s = _orthonormal_frame(source_chord)
    e1t, e2t, e3t = _orthonormal_frame(target_chord)
    offset_scale = min(1.0, target_length / source_length)
    mapped: list[np.ndarray] = []
    for point in centerline:
        delta = point - source_start
        t = float(np.dot(delta, e1s) / source_length)
        t = min(1.0, max(0.0, t))
        y = float(np.dot(delta, e2s)) * offset_scale
        z = float(np.dot(delta, e3s)) * offset_scale
        mapped.append(start + target_chord * t + e2t * y + e3t * z)
    mapped[0] = start
    mapped[-1] = end
    return np.asarray(mapped, dtype=float)


def _write_label_summary_csv(path: Path, labels: dict[int, str], mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    populated = 0
    voxel_volume_cm3 = float(np.prod(spacing_mm) / 1000.0)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label_id", "name", "voxel_count", "volume_cm3"])
        for label_id, name in sorted(labels.items()):
            count = int(np.count_nonzero(mask == label_id))
            if count:
                populated += 1
            writer.writerow([label_id, name, count, f"{count * voxel_volume_cm3:.6f}"])
    return populated


def _write_replacements_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "edge_id",
                "replacement_role",
                "label_ids",
                "label_names",
                "status",
                "point_count",
                "template_length_mm",
                "mapped_length_mm",
                "estimated_radius_mm",
                "note",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["edge_id"],
                    row["replacement_role"],
                    " ".join(str(label) for label in row["label_ids"]),
                    " | ".join(row["label_names"]),
                    row["status"],
                    row["point_count"],
                    f"{row['template_length_mm']:.6f}",
                    f"{row['mapped_length_mm']:.6f}",
                    f"{row['estimated_radius_mm']:.6f}",
                    row["note"],
                ]
            )


def _write_stage_preview(path: Path, image_path: str | Path, mask_path: str | Path, labels: dict[int, str]) -> None:
    plt, nib, _ = _import_dependencies()
    image = np.asanyarray(nib.load(str(image_path)).dataobj)
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj).astype(int)
    z_counts = np.count_nonzero(mask, axis=(0, 1))
    z_index = int(np.argmax(z_counts)) if np.any(z_counts) else mask.shape[2] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=160)
    fig.patch.set_facecolor("#f6f1e8")
    axes[0].imshow(np.rot90(image[:, :, z_index]), cmap="gray", vmin=-200, vmax=700)
    axes[0].set_title(f"MedSeg CT slice z={z_index}")
    axes[0].axis("off")
    axes[1].imshow(np.rot90(image[:, :, z_index]), cmap="gray", vmin=-200, vmax=700, alpha=0.45)
    axes[1].imshow(np.rot90(np.ma.masked_where(mask[:, :, z_index] == 0, mask[:, :, z_index])), cmap="turbo", alpha=0.72)
    axes[1].set_title(f"Vascular labels ({len(labels)} classes)")
    axes[1].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_graph_preview(path: Path, graph: dict[str, Any]) -> None:
    plt, _, _ = _import_dependencies()
    fig = plt.figure(figsize=(9.5, 7.5), dpi=170)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")
    points_for_bounds: list[np.ndarray] = []
    for edge in graph.get("edges", []):
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if len(points) < 2:
            continue
        points_for_bounds.append(points)
        notes = tuple(str(item) for item in edge.get("notes", []))
        vessel_type = str(edge.get("vessel_type", ""))
        if "polyline_replaced_from_registered_labeled_vessel_centerline" in notes:
            color = "#16a34a" if vessel_type == "arterial" else "#1d4ed8"
            linewidth = 2.7
            alpha = 0.98
        elif "polyline_replaced_from_labeled_vessel_template" in notes:
            color = "#0f9f6e" if vessel_type == "arterial" else "#2563eb"
            linewidth = 2.5
            alpha = 0.96
        elif "polyline_replaced_from_cta_lumen_centerline" in notes:
            color = "#dc3b2a"
            linewidth = 2.8
            alpha = 0.85
        else:
            color = "#9aa0a6"
            linewidth = 1.2
            alpha = 0.35
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth, alpha=alpha)
    if points_for_bounds:
        all_points = np.vstack(points_for_bounds)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = float((maxs - mins).max() / 2.0) * 1.12
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_title("MedSeg Branch-Template Vascular Graph\nred = CTA aorta, green/blue = labeled vessel replacements", pad=16)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def stage_medseg_abdominal_vasculature(
    raw_dir: str | Path = "data/raw/medseg_vasculature_abdomen",
    label_config_path: str | Path = "configs/labelmaps/medseg_abdominal_vasculature.yaml",
    output_dir: str | Path = "data/processed/medseg_vasculature_abdomen",
    case_id: str = "medseg_abdominal_vasculature_case001",
    report_path: str | Path | None = "outputs/reports/medseg_abdominal_vasculature_staging_stage001.md",
) -> MedsegVascularStagingResult:
    _, nib, yaml = _import_dependencies()
    raw = Path(raw_dir)
    image_path = raw / "img.nii.gz"
    mask_path = raw / "msk.nii.gz"
    conf_path = raw / "conf.txt"
    for path in (image_path, mask_path, conf_path):
        if not path.exists():
            raise FileNotFoundError(f"Required MedSeg file is missing: {path}")
    labels = _parse_medseg_conf(conf_path)
    image = nib.load(str(image_path))
    mask_image = nib.load(str(mask_path))
    if image.shape != mask_image.shape:
        raise ValueError(f"MedSeg image/mask shape mismatch: {image.shape} vs {mask_image.shape}")
    mask = np.asanyarray(mask_image.dataobj).astype(int)
    spacing_mm = tuple(float(value) for value in mask_image.header.get_zooms()[:3])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_summary = output / f"{case_id}_label_summary_v001.csv"
    manifest = output / f"{case_id}_manifest_v001.yaml"
    preview = output / f"{case_id}_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_staging_report_v001.md"
    populated = _write_label_summary_csv(label_summary, labels, mask, spacing_mm)
    _write_stage_preview(preview, image_path, mask_path, labels)
    manifest_payload = {
        "case_id": case_id,
        "dataset": "medseg_abdominal_vasculature",
        "source": "https://www.medseg.ai/database/vasculature-of-the-abdomen",
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "source_config_path": str(conf_path),
        "label_config_path": str(label_config_path),
        "shape": list(mask.shape),
        "spacing_mm": list(spacing_mm),
        "label_count": len(labels),
        "populated_label_count": populated,
        "outputs": {
            "manifest": str(manifest),
            "label_summary_csv": str(label_summary),
            "preview_png": str(preview),
            "report": str(report),
        },
    }
    manifest.write_text(yaml.safe_dump(manifest_payload, sort_keys=False))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# MedSeg Abdominal Vasculature Staging",
                "",
                f"Case ID: `{case_id}`",
                "",
                "## Summary",
                "",
                f"- Image: `{image_path}`",
                f"- Mask: `{mask_path}`",
                f"- Shape: `{tuple(mask.shape)}`",
                f"- Spacing: {spacing_mm[0]:.4f}, {spacing_mm[1]:.4f}, {spacing_mm[2]:.4f} mm",
                f"- Labels in config: {len(labels)}",
                f"- Populated labels: {populated}",
                "",
                "## Outputs",
                "",
                f"- Manifest: `{manifest}`",
                f"- Label summary CSV: `{label_summary}`",
                f"- Preview PNG: `{preview}`",
                "",
                "## Interpretation",
                "",
                "- This case is staged as a branch-rich abdominal vascular template source.",
                "- It is not inherently registered to the CT-ORG/PCA phantom grid; graph replacement must anchor or register labels before use.",
            ]
        )
        + "\n"
    )
    return MedsegVascularStagingResult(
        case_id=case_id,
        image_path=str(image_path),
        mask_path=str(mask_path),
        label_config_path=str(label_config_path),
        manifest_path=str(manifest),
        label_summary_csv_path=str(label_summary),
        preview_png_path=str(preview),
        report_path=str(report),
        label_count=len(labels),
        populated_label_count=populated,
        notes=("branch_rich_medseg_case_staged", "independent_case_requires_template_anchoring_or_registration"),
    )


def build_labeled_vessel_vascular_graph(
    baseline_graph_path: str | Path,
    labeled_mask_path: str | Path,
    label_config_path: str | Path = "configs/labelmaps/medseg_abdominal_vasculature.yaml",
    output_dir: str | Path = "outputs/digital/vascular_network_medseg_branch_template",
    case_id: str = "ct_org_case0_medseg_branch_template",
    report_path: str | Path | None = "outputs/reports/medseg_branch_template_vascular_graph_stage001.md",
) -> LabeledVesselGraphResult:
    _, nib, yaml = _import_dependencies()
    baseline_graph = _load_yaml(baseline_graph_path)
    label_config = _load_yaml(label_config_path)
    mask_image = nib.load(str(labeled_mask_path))
    mask = np.asanyarray(mask_image.dataobj).astype(int)
    affine = np.asarray(mask_image.affine, dtype=float)
    labels = {int(key): str(value) for key, value in label_config.get("labels", {}).items()}
    edge_mapping = label_config.get("graph_edge_mapping", {})
    if not isinstance(edge_mapping, dict):
        raise ValueError("Label config graph_edge_mapping must be a mapping")

    graph = dict(baseline_graph)
    nodes = [dict(node) for node in baseline_graph.get("nodes", [])]
    edges = [dict(edge) for edge in baseline_graph.get("edges", [])]
    lookup = _node_lookup({"nodes": nodes})
    aorta_reference = _points_for_labels(mask, affine, (4,))
    ivc_reference = _points_for_labels(mask, affine, (1,))
    replacement_rows: list[dict[str, Any]] = []
    successful = 0

    for edge in edges:
        edge_id = str(edge.get("id", ""))
        mapping = edge_mapping.get(edge_id)
        if not isinstance(mapping, dict):
            continue
        label_ids = tuple(int(value) for value in mapping.get("labels", []))
        replacement_role = str(mapping.get("replacement_role", edge_id))
        label_points = _points_for_labels(mask, affine, label_ids)
        source = lookup.get(str(edge.get("source")))
        target = lookup.get(str(edge.get("target")))
        if source is None or target is None:
            status = "failed_missing_graph_node"
            note = "source_or_target_node_missing"
            centerline = np.empty((0, 3), dtype=float)
            radius = 0.0
        elif len(label_points) < 2:
            status = "failed_missing_label_voxels"
            note = "no_voxels_for_configured_labels"
            centerline = np.empty((0, 3), dtype=float)
            radius = 0.0
        else:
            reference = ivc_reference if str(edge.get("vessel_type", "")) == "venous" else aorta_reference
            centerline = _ordered_centerline(label_points, reference)
            start = np.asarray(source.get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
            end = np.asarray(target.get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
            mapped = _template_centerline_to_edge(centerline, start, end)
            existing_radius_start = float(edge.get("radius_start_mm", source.get("radius_mm", 1.0)))
            existing_radius_end = float(edge.get("radius_end_mm", target.get("radius_mm", 1.0)))
            radius = _estimate_radius_mm(label_points, centerline)
            clamped_radius = float(np.clip(radius, min(existing_radius_start, existing_radius_end) * 0.55, max(existing_radius_start, existing_radius_end) * 1.55))
            edge["polyline_mm"] = [[float(value) for value in point] for point in mapped]
            edge["radius_start_mm"] = existing_radius_start
            edge["radius_end_mm"] = clamped_radius
            edge["length_mm"] = _line_length(mapped)
            notes = list(edge.get("notes", []))
            notes.extend(
                [
                    "polyline_replaced_from_labeled_vessel_template",
                    f"labeled_template_source={label_config.get('dataset', 'unknown')}",
                    f"labeled_template_role={replacement_role}",
                ]
            )
            edge["notes"] = sorted(set(str(item) for item in notes))
            target_notes = list(target.get("notes", []))
            target_notes.append("endpoint_retained_graph_anchor_for_labeled_template")
            target["notes"] = sorted(set(str(item) for item in target_notes))
            status = "replaced_from_labeled_vessel_template"
            note = "template_centerline_mapped_between_existing_graph_anchors"
            successful += 1
        replacement_rows.append(
            {
                "edge_id": edge_id,
                "replacement_role": replacement_role,
                "label_ids": label_ids,
                "label_names": [labels.get(label_id, f"label_{label_id}") for label_id in label_ids],
                "status": status,
                "point_count": int(len(centerline)),
                "template_length_mm": _line_length(centerline) if len(centerline) >= 2 else 0.0,
                "mapped_length_mm": float(edge.get("length_mm", 0.0)),
                "estimated_radius_mm": radius,
                "note": note,
            }
        )

    retained = 0
    for edge in edges:
        notes = tuple(str(item) for item in edge.get("notes", []))
        if "polyline_replaced_from_labeled_vessel_template" not in notes:
            retained += 1

    metadata = dict(graph.get("graph_metadata", {}))
    metadata.update(
        {
            "source_graph": str(baseline_graph_path),
            "source_labeled_vessel_mask": str(labeled_mask_path),
            "source_label_config": str(label_config_path),
            "attempted_labeled_vessel_replacements": len(replacement_rows),
            "successful_labeled_vessel_replacements": successful,
            "retained_non_labeled_template_edges": retained,
            "geometry_status": "cta_aorta_with_medseg_branch_template_replacements",
            "registration_status": "template_centerlines_anchored_to_existing_phantom_graph_nodes",
        }
    )
    graph.update(
        {
            "case_id": case_id,
            "scaffold_type": "cta_aorta_with_labeled_medseg_branch_templates",
            "source_baseline_graph": str(baseline_graph_path),
            "source_labeled_vessel_mask": str(labeled_mask_path),
            "source_label_config": str(label_config_path),
            "graph_metadata": metadata,
            "provenance_notes": [
                "aorta_trunk_retained_from_baseline_cta_derived_graph",
                "branch_and_venous_edges_replaced_from_medseg_labeled_vessel_centerline_templates",
                "medseg_case_is_independent_and_anchored_to_existing_graph_not_deformably_registered",
            ],
            "nodes": nodes,
            "edges": edges,
        }
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_yaml = output / f"{case_id}_labeled_vessel_vascular_graph_v001.yaml"
    nodes_csv = output / f"{case_id}_labeled_vessel_vascular_graph_nodes_v001.csv"
    edges_csv = output / f"{case_id}_labeled_vessel_vascular_graph_edges_v001.csv"
    replacements_csv = output / f"{case_id}_labeled_vessel_replacements_v001.csv"
    preview = output / f"{case_id}_labeled_vessel_vascular_graph_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_labeled_vessel_vascular_graph_report_v001.md"
    graph_yaml.write_text(yaml.safe_dump(graph, sort_keys=False))
    _write_nodes_csv(nodes_csv, nodes)
    _write_edges_csv(edges_csv, edges)
    _write_replacements_csv(replacements_csv, replacement_rows)
    _write_graph_preview(preview, graph)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_labeled_graph_report(case_id, result_paths=(graph_yaml, nodes_csv, edges_csv, replacements_csv, preview), rows=replacement_rows, successful=successful, retained=retained) + "\n")

    return LabeledVesselGraphResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_yaml),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        replacements_csv_path=str(replacements_csv),
        preview_png_path=str(preview),
        report_path=str(report),
        attempted_replacements=len(replacement_rows),
        successful_replacements=successful,
        retained_edges=retained,
        notes=("branch_rich_labeled_vessel_templates_applied", "template_anchor_registration_not_patient_specific"),
    )


def build_registered_labeled_vessel_vascular_graph(
    target_graph_path: str | Path,
    labeled_mask_path: str | Path,
    label_config_path: str | Path = "configs/labelmaps/medseg_abdominal_vasculature.yaml",
    target_labels_path: str | Path | None = None,
    output_dir: str | Path = "outputs/digital/vascular_network_medseg_registered",
    case_id: str = "ct_org_mode03_neg_medseg_registered",
    report_path: str | Path | None = "outputs/reports/medseg_registered_vascular_graph_stage001.md",
) -> RegisteredLabeledVesselGraphResult:
    _, nib, yaml = _import_dependencies()
    target_graph = _load_yaml(target_graph_path)
    label_config = _load_yaml(label_config_path)
    mask_image = nib.load(str(labeled_mask_path))
    mask = np.asanyarray(mask_image.dataobj).astype(int)
    affine = np.asarray(mask_image.affine, dtype=float)
    labels = {int(key): str(value) for key, value in label_config.get("labels", {}).items()}
    edge_mapping = label_config.get("graph_edge_mapping", {})
    if not isinstance(edge_mapping, dict):
        raise ValueError("Label config graph_edge_mapping must be a mapping")

    source_landmarks = _source_landmarks_from_mask(mask, affine)
    source_landmarks.update(_source_landmarks_from_edge_templates(mask, affine, label_config))
    target_landmarks = _target_landmarks_from_graph(target_graph)
    registration_matrix, landmark_rows, rms_error, max_error = _fit_landmark_affine(source_landmarks, target_landmarks)
    residual_anchors, residual_displacements = _landmark_residual_controls(landmark_rows)
    deformable_rms_error, deformable_max_error = _add_deformable_landmark_errors(landmark_rows, residual_anchors, residual_displacements)

    graph = dict(target_graph)
    nodes = [dict(node) for node in target_graph.get("nodes", [])]
    edges = [dict(edge) for edge in target_graph.get("edges", [])]
    lookup = _node_lookup({"nodes": nodes})
    aorta_reference = _points_for_labels(mask, affine, (4,))
    ivc_reference = _points_for_labels(mask, affine, (1,))
    replacement_rows: list[dict[str, Any]] = []
    successful = 0

    for edge in edges:
        edge_id = str(edge.get("id", ""))
        mapping = edge_mapping.get(edge_id)
        if not isinstance(mapping, dict):
            continue
        label_ids = tuple(int(value) for value in mapping.get("labels", []))
        replacement_role = str(mapping.get("replacement_role", edge_id))
        label_points = _points_for_labels(mask, affine, label_ids)
        source = lookup.get(str(edge.get("source")))
        target = lookup.get(str(edge.get("target")))
        if source is None or target is None:
            status = "failed_missing_graph_node"
            note = "source_or_target_node_missing"
            source_centerline = np.empty((0, 3), dtype=float)
            mapped = np.empty((0, 3), dtype=float)
            radius = 0.0
        elif len(label_points) < 2:
            status = "failed_missing_label_voxels"
            note = "no_voxels_for_configured_labels"
            source_centerline = np.empty((0, 3), dtype=float)
            mapped = np.empty((0, 3), dtype=float)
            radius = 0.0
        else:
            reference = ivc_reference if str(edge.get("vessel_type", "")) == "venous" else aorta_reference
            source_centerline = _ordered_centerline(label_points, reference)
            registered_centerline = _apply_row_affine(source_centerline, registration_matrix)
            deformed_centerline = _apply_landmark_residual_warp(registered_centerline, residual_anchors, residual_displacements)
            start = np.asarray(source.get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
            end = np.asarray(target.get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
            mapped = _snap_registered_centerline(deformed_centerline, start, end)
            existing_radius_start = float(edge.get("radius_start_mm", source.get("radius_mm", 1.0)))
            existing_radius_end = float(edge.get("radius_end_mm", target.get("radius_mm", 1.0)))
            radius = _estimate_radius_mm(label_points, source_centerline)
            clamped_radius = float(np.clip(radius, min(existing_radius_start, existing_radius_end) * 0.55, max(existing_radius_start, existing_radius_end) * 1.55))
            edge["polyline_mm"] = [[float(value) for value in point] for point in mapped]
            edge["radius_start_mm"] = existing_radius_start
            edge["radius_end_mm"] = clamped_radius
            edge["length_mm"] = _line_length(mapped)
            notes = list(edge.get("notes", []))
            notes.extend(
                [
                    "polyline_replaced_from_registered_labeled_vessel_centerline",
                    f"registered_labeled_source={label_config.get('dataset', 'unknown')}",
                    f"registered_labeled_role={replacement_role}",
                    "registration_model=landmark_affine_plus_local_residual_warp_with_endpoint_snap",
                ]
            )
            edge["notes"] = sorted(set(str(item) for item in notes))
            target_notes = list(target.get("notes", []))
            target_notes.append("endpoint_retained_after_landmark_registered_labeled_vessel_fit")
            target["notes"] = sorted(set(str(item) for item in target_notes))
            status = "replaced_from_registered_labeled_vessel_centerline"
            note = "source_centerline_affine_registered_then_local_landmark_residual_warped_and_endpoint_snapped"
            successful += 1
        replacement_rows.append(
            {
                "edge_id": edge_id,
                "replacement_role": replacement_role,
                "label_ids": label_ids,
                "label_names": [labels.get(label_id, f"label_{label_id}") for label_id in label_ids],
                "status": status,
                "point_count": int(len(source_centerline)),
                "template_length_mm": _line_length(source_centerline) if len(source_centerline) >= 2 else 0.0,
                "mapped_length_mm": _line_length(mapped) if len(mapped) >= 2 else float(edge.get("length_mm", 0.0)),
                "estimated_radius_mm": radius,
                "note": note,
            }
        )

    retained = 0
    for edge in edges:
        notes = tuple(str(item) for item in edge.get("notes", []))
        if "polyline_replaced_from_registered_labeled_vessel_centerline" not in notes:
            retained += 1

    metadata = dict(graph.get("graph_metadata", {}))
    metadata.update(
        {
            "source_target_graph": str(target_graph_path),
            "source_labeled_vessel_mask": str(labeled_mask_path),
            "source_label_config": str(label_config_path),
            "target_labels": str(target_labels_path) if target_labels_path is not None else None,
            "attempted_registered_labeled_vessel_replacements": len(replacement_rows),
            "successful_registered_labeled_vessel_replacements": successful,
            "retained_non_registered_labeled_edges": retained,
            "landmark_count": len(landmark_rows),
            "landmark_registration_rms_error_mm": rms_error,
            "landmark_registration_max_error_mm": max_error,
            "landmark_affine_registration_rms_error_mm": rms_error,
            "landmark_affine_registration_max_error_mm": max_error,
            "landmark_deformable_registration_rms_error_mm": deformable_rms_error,
            "landmark_deformable_registration_max_error_mm": deformable_max_error,
            "geometry_status": "landmark_registered_medseg_vessels_in_target_phantom_graph",
            "registration_status": "landmark_affine_plus_local_residual_warp_then_endpoint_snapped",
        }
    )
    graph.update(
        {
            "case_id": case_id,
            "scaffold_type": "landmark_registered_cta_aorta_with_medseg_labeled_branch_centerlines",
            "source_target_graph": str(target_graph_path),
            "source_labeled_vessel_mask": str(labeled_mask_path),
            "source_label_config": str(label_config_path),
            "target_labels": str(target_labels_path) if target_labels_path is not None else None,
            "graph_metadata": metadata,
            "provenance_notes": [
                "target_graph_is_already_in_phantom_or_pca_anatomy_coordinates",
                "medseg_labeled_vessels_registered_to_target_graph_landmarks_by_affine_fit_plus_local_residual_warp",
                "edge_endpoints_are_snapped_to_existing_solver_boundary_nodes_after_registration",
                "this_is_landmark_deformable_template_registration_not_full_patient_specific_cta_to_ct_registration",
            ],
            "nodes": nodes,
            "edges": edges,
        }
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_yaml = output / f"{case_id}_registered_labeled_vessel_vascular_graph_v001.yaml"
    nodes_csv = output / f"{case_id}_registered_labeled_vessel_vascular_graph_nodes_v001.csv"
    edges_csv = output / f"{case_id}_registered_labeled_vessel_vascular_graph_edges_v001.csv"
    replacements_csv = output / f"{case_id}_registered_labeled_vessel_replacements_v001.csv"
    landmarks_csv = output / f"{case_id}_registration_landmarks_v001.csv"
    transform_csv = output / f"{case_id}_registration_transform_v001.csv"
    preview = output / f"{case_id}_registered_labeled_vessel_vascular_graph_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_registered_labeled_vessel_vascular_graph_report_v001.md"
    graph_yaml.write_text(yaml.safe_dump(graph, sort_keys=False))
    _write_nodes_csv(nodes_csv, nodes)
    _write_edges_csv(edges_csv, edges)
    _write_replacements_csv(replacements_csv, replacement_rows)
    _write_landmarks_csv(landmarks_csv, landmark_rows)
    _write_transform_csv(transform_csv, registration_matrix)
    _write_graph_preview(preview, graph)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        _format_registered_graph_report(
            case_id,
            result_paths=(graph_yaml, nodes_csv, edges_csv, replacements_csv, landmarks_csv, transform_csv, preview),
            rows=replacement_rows,
            landmark_rows=landmark_rows,
            successful=successful,
            retained=retained,
            rms_error=rms_error,
            max_error=max_error,
            deformable_rms_error=deformable_rms_error,
            deformable_max_error=deformable_max_error,
        )
        + "\n"
    )

    return RegisteredLabeledVesselGraphResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_yaml),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        replacements_csv_path=str(replacements_csv),
        landmarks_csv_path=str(landmarks_csv),
        transform_csv_path=str(transform_csv),
        preview_png_path=str(preview),
        report_path=str(report),
        attempted_replacements=len(replacement_rows),
        successful_replacements=successful,
        retained_edges=retained,
        landmark_count=len(landmark_rows),
        registration_rms_error_mm=rms_error,
        registration_max_error_mm=max_error,
        deformable_registration_rms_error_mm=deformable_rms_error,
        deformable_registration_max_error_mm=deformable_max_error,
        notes=("landmark_deformable_registered_labeled_vessel_centerlines_applied", "not_patient_specific_cta_to_ct_registration"),
    )


def _format_labeled_graph_report(
    case_id: str,
    *,
    result_paths: tuple[Path, Path, Path, Path, Path],
    rows: list[dict[str, Any]],
    successful: int,
    retained: int,
) -> str:
    graph_yaml, nodes_csv, edges_csv, replacements_csv, preview = result_paths
    lines = [
        "# Labeled Vessel Branch-Template Graph",
        "",
        f"Case ID: `{case_id}`",
        "",
        "## Summary",
        "",
        f"- Attempted edge replacements: {len(rows)}",
        f"- Successful labeled-template replacements: {successful}",
        f"- Retained non-template edges: {retained}",
        "",
        "## Outputs",
        "",
        f"- Graph YAML: `{graph_yaml.name}`",
        f"- Nodes CSV: `{nodes_csv.name}`",
        f"- Edges CSV: `{edges_csv.name}`",
        f"- Replacements CSV: `{replacements_csv.name}`",
        f"- Preview PNG: `{preview.name}`",
        "",
        "## Replacements",
        "",
        "| edge | role | labels | status | template length mm | mapped length mm |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        labels = ", ".join(row["label_names"])
        lines.append(
            f"| `{row['edge_id']}` | `{row['replacement_role']}` | {labels} | `{row['status']}` | {row['template_length_mm']:.2f} | {row['mapped_length_mm']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- MedSeg labeled vessels are used as branch-rich centerline templates anchored to existing graph endpoints.",
            "- This replaces placeholder edge polylines with real labeled-vessel shapes while preserving solver-compatible node IDs and boundary IDs.",
            "- The MedSeg case is not deformably registered into CT-ORG/PCA anatomy, so this is an anatomical template upgrade rather than patient-specific CTA registration.",
        ]
    )
    return "\n".join(lines)


def _format_registered_graph_report(
    case_id: str,
    *,
    result_paths: tuple[Path, Path, Path, Path, Path, Path, Path],
    rows: list[dict[str, Any]],
    landmark_rows: list[dict[str, Any]],
    successful: int,
    retained: int,
    rms_error: float,
    max_error: float,
    deformable_rms_error: float,
    deformable_max_error: float,
) -> str:
    graph_yaml, nodes_csv, edges_csv, replacements_csv, landmarks_csv, transform_csv, preview = result_paths
    worst = sorted(landmark_rows, key=lambda row: float(row["error_mm"]), reverse=True)[:8]
    worst_deformable = sorted(landmark_rows, key=lambda row: float(row.get("deformable_error_mm", row["error_mm"])), reverse=True)[:8]
    lines = [
        "# Registered Labeled Vessel Graph",
        "",
        f"Case ID: `{case_id}`",
        "",
        "## Summary",
        "",
        f"- Landmark pairs: {len(landmark_rows)}",
        f"- Affine landmark RMS error: {rms_error:.2f} mm",
        f"- Affine landmark max error: {max_error:.2f} mm",
        f"- Local deformable-control RMS error: {deformable_rms_error:.2f} mm",
        f"- Local deformable-control max error: {deformable_max_error:.2f} mm",
        f"- Attempted edge replacements: {len(rows)}",
        f"- Successful registered-vessel replacements: {successful}",
        f"- Retained non-registered edges: {retained}",
        "",
        "## Outputs",
        "",
        f"- Graph YAML: `{graph_yaml.name}`",
        f"- Nodes CSV: `{nodes_csv.name}`",
        f"- Edges CSV: `{edges_csv.name}`",
        f"- Replacements CSV: `{replacements_csv.name}`",
        f"- Landmark CSV: `{landmarks_csv.name}`",
        f"- Transform CSV: `{transform_csv.name}`",
        f"- Preview PNG: `{preview.name}`",
        "",
        "## Largest Affine Landmark Errors",
        "",
        "| landmark | error mm |",
        "| --- | ---: |",
    ]
    for row in worst:
        lines.append(f"| `{row['landmark_id']}` | {float(row['error_mm']):.2f} |")
    lines.extend(
        [
            "",
            "## Largest Deformable-Control Errors",
            "",
            "| landmark | error mm |",
            "| --- | ---: |",
        ]
    )
    for row in worst_deformable:
        lines.append(f"| `{row['landmark_id']}` | {float(row.get('deformable_error_mm', row['error_mm'])):.2f} |")
    lines.extend(
        [
            "",
            "## Replacements",
            "",
            "| edge | role | labels | status | source length mm | registered length mm |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        labels = ", ".join(row["label_names"])
        lines.append(
            f"| `{row['edge_id']}` | `{row['replacement_role']}` | {labels} | `{row['status']}` | {row['template_length_mm']:.2f} | {row['mapped_length_mm']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- MedSeg labelled vessel centerlines are first transformed into the target phantom graph coordinate frame using an affine fit from shared vascular landmarks.",
            "- A local inverse-distance residual warp then interpolates landmark corrections onto the vessel centerlines, making this a landmark-deformable template registration layer rather than a pure anchor-only mapping.",
            "- Endpoints are then snapped to existing graph nodes so boundary IDs remain solver-compatible.",
            "- The deformable-control residual is an internal control-point fit, not an independent validation metric; full patient-specific CTA-to-CT deformable registration still requires paired CTA/CT anatomy or organ/vessel landmark annotations.",
        ]
    )
    return "\n".join(lines)


def format_medseg_vascular_staging_result(result: MedsegVascularStagingResult) -> str:
    return "\n".join(
        [
            "MedSeg abdominal vasculature staged",
            f"Case ID: {result.case_id}",
            f"Labels populated: {result.populated_label_count}/{result.label_count}",
            f"Manifest: {result.manifest_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )


def format_labeled_vessel_graph_result(result: LabeledVesselGraphResult) -> str:
    return "\n".join(
        [
            "Labeled vessel vascular graph built",
            f"Case ID: {result.case_id}",
            f"Replacements: {result.successful_replacements}/{result.attempted_replacements}",
            f"Graph YAML: {result.graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )


def format_registered_labeled_vessel_graph_result(result: RegisteredLabeledVesselGraphResult) -> str:
    return "\n".join(
        [
            "Registered labeled vessel vascular graph built",
            f"Case ID: {result.case_id}",
            f"Affine landmark RMS/max error: {result.registration_rms_error_mm:.2f}/{result.registration_max_error_mm:.2f} mm",
            f"Deformable-control RMS/max error: {result.deformable_registration_rms_error_mm:.2f}/{result.deformable_registration_max_error_mm:.2f} mm",
            f"Replacements: {result.successful_replacements}/{result.attempted_replacements}",
            f"Graph YAML: {result.graph_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
