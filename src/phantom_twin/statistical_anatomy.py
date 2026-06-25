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
        from matplotlib.colors import ListedColormap  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
        from scipy import ndimage  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Statistical anatomy morphing requires matplotlib, nibabel, scipy, and PyYAML."
        ) from exc
    return plt, ListedColormap, Patch, nib, ndimage, yaml


GROUP_DEFINITIONS: tuple[tuple[str, str, tuple[int, ...], float, float], ...] = (
    ("body", "Body envelope", tuple(range(1, 16)), 1.00, 1.00),
    ("lungs", "Lungs", (8,), 0.35, 0.70),
    ("liver", "Liver", (6,), 0.50, 0.70),
    ("kidneys", "Kidneys", (7,), 0.45, 0.70),
    ("bladder", "Bladder", (9,), 0.55, 0.75),
    ("bone", "Trabecular + cortical bone", (10, 11), 0.20, 0.45),
    ("vessel_wall", "Vessel wall", (13,), 0.55, 0.75),
    ("vascular_fluid", "Vascular fluid", (14, 15), 0.58, 0.80),
)

GROUP_BY_ID = {group_id: (name, labels, xy_coupling, z_coupling) for group_id, name, labels, xy_coupling, z_coupling in GROUP_DEFINITIONS}
DEEP_STRUCTURE_LABELS = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


@dataclass(frozen=True)
class ShapeGroupStats:
    group_id: str
    name: str
    labels: tuple[int, ...]
    present: bool
    voxel_count: int
    volume_cm3: float
    centroid_ijk: tuple[float, float, float]
    centroid_mm: tuple[float, float, float]
    bbox_voxels: tuple[float, float, float]
    bbox_mm: tuple[float, float, float]
    waist_cm: float | None = None


@dataclass(frozen=True)
class PopulationRegistrationStats:
    case_id: str
    source_path: str
    registered_body_volume_cm3: float
    registered_waist_cm: float
    registered_bbox_mm: tuple[float, float, float]
    registration_scale: tuple[float, float, float]
    registration_translation_voxels: tuple[float, float, float]
    groups: tuple[ShapeGroupStats, ...]


@dataclass(frozen=True)
class StatisticalAnatomyResult:
    case_id: str
    output_dir: str
    morphed_blood_material_labels_path: str
    morphed_blood_density_path: str
    morphed_blood_relative_electron_density_path: str
    morphed_blood_synthetic_hu_path: str
    morphed_contrast_material_labels_path: str
    morphed_contrast_density_path: str
    morphed_contrast_relative_electron_density_path: str
    morphed_contrast_synthetic_hu_path: str
    morphed_body_mask_path: str
    morphed_vascular_fluid_mask_path: str
    morphed_vessel_wall_mask_path: str
    shape_model_npz_path: str
    registration_csv_path: str
    deformation_transforms_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    population_case_count: int
    shape_mode_count: int
    target_height_cm: float | None
    target_bmi: float | None
    target_waist_cm: float | None
    achieved_waist_cm: float
    baseline_body_volume_cm3: float
    morphed_body_volume_cm3: float
    body_volume_change_percent: float
    baseline_bbox_mm: tuple[float, float, float]
    morphed_bbox_mm: tuple[float, float, float]
    vascular_components: int
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(raw_path: str | Path, reference_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    candidate = reference_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _write_nifti(path: Path, data: np.ndarray, reference_image, nib) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _voxel_volume_cm3(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(spacing_mm) / 1000.0)


def _bbox_voxels(mask: np.ndarray) -> tuple[float, float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0.0, 0.0, 0.0)
    extent = coords.max(axis=0) - coords.min(axis=0) + 1
    return tuple(float(value) for value in extent)


def _bbox_mm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.array(_bbox_voxels(mask), dtype=float) * np.array(spacing_mm, dtype=float))


def _centroid_ijk(mask: np.ndarray, fallback_shape: tuple[int, int, int]) -> tuple[float, float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(float((value - 1) / 2.0) for value in fallback_shape)
    return tuple(float(value) for value in coords.mean(axis=0))


def _ellipse_circumference_cm(width_mm: float, depth_mm: float) -> float:
    a = max(width_mm / 2.0, 1e-6)
    b = max(depth_mm / 2.0, 1e-6)
    circumference_mm = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
    return float(circumference_mm / 10.0)


def _body_z_bounds(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, mask.shape[2] - 1
    return int(coords[:, 2].min()), int(coords[:, 2].max())


def _estimate_waist_cm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    z_min, z_max = _body_z_bounds(mask)
    z_index = int(round(z_min + 0.46 * (z_max - z_min)))
    slice_mask = mask[:, :, z_index]
    coords = np.argwhere(slice_mask)
    if coords.size == 0:
        bbox = _bbox_mm(mask, spacing_mm)
        return _ellipse_circumference_cm(bbox[0], bbox[1])
    width_mm = float((coords[:, 0].max() - coords[:, 0].min() + 1) * spacing_mm[0])
    depth_mm = float((coords[:, 1].max() - coords[:, 1].min() + 1) * spacing_mm[1])
    return _ellipse_circumference_cm(width_mm, depth_mm)


def _derive_target_bmi(target_height_cm: float | None, target_weight_kg: float | None, target_bmi: float | None) -> float | None:
    if target_bmi is not None:
        return float(target_bmi)
    if target_weight_kg is None or target_height_cm is None:
        return None
    height_m = target_height_cm / 100.0
    if height_m <= 0.0:
        raise ValueError("target_height_cm must be positive")
    return float(target_weight_kg / (height_m**2))


def _derive_target_waist_cm(
    target_waist_cm: float | None,
    baseline_waist_cm: float,
    target_bmi: float | None,
    baseline_bmi: float,
) -> float | None:
    if target_waist_cm is not None:
        return float(target_waist_cm)
    if target_bmi is None:
        return None
    if baseline_bmi <= 0.0:
        raise ValueError("baseline_bmi must be positive")
    return float(baseline_waist_cm * math.sqrt(max(target_bmi, 1e-6) / baseline_bmi))


def _property_volume(material_labels: np.ndarray, regions: list[dict[str, Any]], key: str) -> np.ndarray:
    output = np.zeros(material_labels.shape, dtype=np.float32)
    for region in regions:
        output[material_labels == int(region["index"])] = float(region[key])
    return output


def _group_mask(labels: np.ndarray, group_labels: tuple[int, ...]) -> np.ndarray:
    return np.isin(labels, group_labels)


def _group_stats(
    group_id: str,
    labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    voxel_volume: float,
) -> ShapeGroupStats:
    name, group_labels, *_ = GROUP_BY_ID[group_id]
    mask = _group_mask(labels, group_labels)
    voxels = int(mask.sum())
    present = voxels > 0
    centroid = _centroid_ijk(mask, labels.shape)
    bbox_vox = _bbox_voxels(mask)
    bbox = tuple(float(value) for value in np.array(bbox_vox, dtype=float) * np.array(spacing_mm, dtype=float))
    centroid_mm = tuple(float(value) for value in np.array(centroid, dtype=float) * np.array(spacing_mm, dtype=float))
    waist = _estimate_waist_cm(mask, spacing_mm) if group_id == "body" and present else None
    return ShapeGroupStats(
        group_id=group_id,
        name=name,
        labels=group_labels,
        present=present,
        voxel_count=voxels,
        volume_cm3=float(voxels * voxel_volume),
        centroid_ijk=centroid,
        centroid_mm=centroid_mm,
        bbox_voxels=bbox_vox,
        bbox_mm=bbox,
        waist_cm=waist,
    )


def _all_group_stats(
    labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    voxel_volume: float,
) -> dict[str, ShapeGroupStats]:
    return {group_id: _group_stats(group_id, labels, spacing_mm, voxel_volume) for group_id, *_ in GROUP_DEFINITIONS}


def _register_labels_to_reference(
    labels: np.ndarray,
    reference_body: np.ndarray,
    ndimage,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    source_body = labels > 0
    if not np.any(source_body):
        raise ValueError("Population segmentation has no non-zero anatomy labels")
    reference_center = np.array(_centroid_ijk(reference_body, reference_body.shape), dtype=float)
    source_center = np.array(_centroid_ijk(source_body, labels.shape), dtype=float)
    reference_extent = np.maximum(np.array(_bbox_voxels(reference_body), dtype=float), 1.0)
    source_extent = np.maximum(np.array(_bbox_voxels(source_body), dtype=float), 1.0)
    scale = source_extent / reference_extent
    matrix = np.diag(scale)
    offset = source_center - matrix @ reference_center
    registered = ndimage.affine_transform(
        labels,
        matrix=matrix,
        offset=offset,
        output_shape=reference_body.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(labels.dtype)
    translation = tuple(float(value) for value in source_center - reference_center)
    return registered, tuple(float(value) for value in scale), translation


def _load_label_like_reference(path: str | Path, reference_image, nib, ndimage) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape == reference_image.shape:
        return np.rint(np.asanyarray(image.dataobj)).astype(np.int16)

    data = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    try:
        from nibabel.processing import resample_from_to  # type: ignore

        resampled = resample_from_to(image, reference_image, order=0)
        labels = np.rint(np.asanyarray(resampled.dataobj)).astype(np.int16)
        if np.any(labels) or not np.any(data):
            return labels
    except Exception:
        pass

    zoom = tuple(float(dst) / float(src) for src, dst in zip(data.shape, reference_image.shape, strict=True))
    return ndimage.zoom(data, zoom=zoom, order=0).astype(np.int16)


def _population_case_ids(paths: tuple[str | Path, ...], supplied_ids: tuple[str, ...] | None) -> tuple[str, ...]:
    if supplied_ids is not None:
        if len(supplied_ids) != len(paths):
            raise ValueError("--population-case-ids must match the number of --population-labels")
        return supplied_ids
    return tuple(Path(path).name.replace(".nii.gz", "").replace(".nii", "") for path in paths)


def _case_feature_vector(group_stats: dict[str, ShapeGroupStats]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[float] = []
    for group_id, *_ in GROUP_DEFINITIONS:
        stats = group_stats[group_id]
        prefix = group_id
        fields = {
            "present": 1.0 if stats.present else 0.0,
            "volume_cm3": stats.volume_cm3,
            "centroid_x_mm": stats.centroid_mm[0],
            "centroid_y_mm": stats.centroid_mm[1],
            "centroid_z_mm": stats.centroid_mm[2],
            "bbox_x_mm": stats.bbox_mm[0],
            "bbox_y_mm": stats.bbox_mm[1],
            "bbox_z_mm": stats.bbox_mm[2],
        }
        if group_id == "body":
            fields["waist_cm"] = float(stats.waist_cm or 0.0)
        for field, value in fields.items():
            names.append(f"{prefix}.{field}")
            values.append(float(value))
    return names, np.array(values, dtype=np.float64)


def _fit_feature_pca(feature_matrix: np.ndarray, max_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = feature_matrix.mean(axis=0)
    if feature_matrix.shape[0] <= 1:
        std = np.ones(feature_matrix.shape[1], dtype=np.float64)
        return mean, std, np.zeros((0, feature_matrix.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.float64)
    std = feature_matrix.std(axis=0, ddof=1)
    std = np.where(std > 1e-8, std, 1.0)
    centered = (feature_matrix - mean) / std
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    mode_count = max(0, min(max_modes, feature_matrix.shape[0] - 1, components.shape[0]))
    return mean, std, components[:mode_count], singular_values[:mode_count]


def _target_feature_map(
    feature_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    components: np.ndarray,
    mode_weights: tuple[float, ...],
) -> dict[str, float]:
    normalized = np.zeros_like(mean)
    for index, weight in enumerate(mode_weights[: components.shape[0]]):
        normalized += float(weight) * components[index]
    values = mean + normalized * std
    return {name: float(value) for name, value in zip(feature_names, values, strict=True)}


def _stats_from_feature_map(
    feature_map: dict[str, float],
    reference_stats: dict[str, ShapeGroupStats],
    spacing_mm: tuple[float, float, float],
) -> dict[str, ShapeGroupStats]:
    output: dict[str, ShapeGroupStats] = {}
    for group_id, name, group_labels, *_ in GROUP_DEFINITIONS:
        reference = reference_stats[group_id]
        present_value = feature_map.get(f"{group_id}.present", 1.0 if reference.present else 0.0)
        volume_value = max(feature_map.get(f"{group_id}.volume_cm3", reference.volume_cm3), 0.0)
        if reference.present and present_value < 0.5 and volume_value <= 1e-6:
            output[group_id] = reference
            continue
        present = present_value >= 0.5
        volume = volume_value
        centroid_mm = (
            feature_map.get(f"{group_id}.centroid_x_mm", reference.centroid_mm[0]),
            feature_map.get(f"{group_id}.centroid_y_mm", reference.centroid_mm[1]),
            feature_map.get(f"{group_id}.centroid_z_mm", reference.centroid_mm[2]),
        )
        bbox_mm = (
            max(feature_map.get(f"{group_id}.bbox_x_mm", reference.bbox_mm[0]), spacing_mm[0]),
            max(feature_map.get(f"{group_id}.bbox_y_mm", reference.bbox_mm[1]), spacing_mm[1]),
            max(feature_map.get(f"{group_id}.bbox_z_mm", reference.bbox_mm[2]), spacing_mm[2]),
        )
        centroid_ijk = tuple(float(value) for value in np.array(centroid_mm, dtype=float) / np.array(spacing_mm, dtype=float))
        bbox_voxels = tuple(float(value) for value in np.array(bbox_mm, dtype=float) / np.array(spacing_mm, dtype=float))
        output[group_id] = ShapeGroupStats(
            group_id=group_id,
            name=name,
            labels=group_labels,
            present=present,
            voxel_count=reference.voxel_count,
            volume_cm3=float(volume),
            centroid_ijk=centroid_ijk,
            centroid_mm=tuple(float(value) for value in centroid_mm),
            bbox_voxels=bbox_voxels,
            bbox_mm=tuple(float(value) for value in bbox_mm),
            waist_cm=feature_map.get(f"{group_id}.waist_cm", reference.waist_cm),
        )
    return output


def _apply_anthropometry_to_target_stats(
    target_stats: dict[str, ShapeGroupStats],
    reference_stats: dict[str, ShapeGroupStats],
    target_height_cm: float | None,
    target_waist_cm: float | None,
    baseline_height_cm: float,
) -> dict[str, ShapeGroupStats]:
    body = target_stats["body"]
    baseline_waist = float(body.waist_cm or reference_stats["body"].waist_cm or 1.0)
    xy_scale = 1.0 if target_waist_cm is None else float(target_waist_cm / max(baseline_waist, 1e-6))
    z_scale = 1.0 if target_height_cm is None else float(target_height_cm / max(baseline_height_cm, 1e-6))
    xy_scale = float(np.clip(xy_scale, 0.70, 1.60))
    z_scale = float(np.clip(z_scale, 0.82, 1.18))
    body_center = np.array(body.centroid_ijk, dtype=float)

    adjusted: dict[str, ShapeGroupStats] = {}
    for group_id, name, group_labels, xy_coupling, z_coupling in GROUP_DEFINITIONS:
        stats = target_stats[group_id]
        reference = reference_stats[group_id]
        group_xy = 1.0 + xy_coupling * (xy_scale - 1.0)
        group_z = 1.0 + z_coupling * (z_scale - 1.0)
        centroid = np.array(stats.centroid_ijk, dtype=float)
        relative_centroid = centroid - body_center
        target_centroid = body_center + relative_centroid * np.array([group_xy, group_xy, group_z], dtype=float)
        bbox_voxels = np.maximum(
            np.array(stats.bbox_voxels, dtype=float) * np.array([group_xy, group_xy, group_z], dtype=float),
            1.0,
        )
        if group_id == "body":
            target_centroid = body_center
            bbox_voxels = np.maximum(
                np.array(stats.bbox_voxels, dtype=float) * np.array([xy_scale, xy_scale, z_scale], dtype=float),
                1.0,
            )
        spacing = np.array(
            [
                reference.bbox_mm[0] / max(reference.bbox_voxels[0], 1e-6),
                reference.bbox_mm[1] / max(reference.bbox_voxels[1], 1e-6),
                reference.bbox_mm[2] / max(reference.bbox_voxels[2], 1e-6),
            ],
            dtype=float,
        )
        bbox_mm = tuple(float(value) for value in bbox_voxels * spacing)
        centroid_mm = tuple(float(value) for value in target_centroid * spacing)
        volume = stats.volume_cm3 * group_xy * group_xy * group_z
        waist = target_waist_cm if group_id == "body" and target_waist_cm is not None else stats.waist_cm
        adjusted[group_id] = ShapeGroupStats(
            group_id=group_id,
            name=name,
            labels=group_labels,
            present=stats.present,
            voxel_count=stats.voxel_count,
            volume_cm3=float(volume),
            centroid_ijk=tuple(float(value) for value in target_centroid),
            centroid_mm=centroid_mm,
            bbox_voxels=tuple(float(value) for value in bbox_voxels),
            bbox_mm=bbox_mm,
            waist_cm=waist,
        )
    return adjusted


def _scale_target_stats_xy(
    target_stats: dict[str, ShapeGroupStats],
    correction: float,
) -> dict[str, ShapeGroupStats]:
    correction = float(np.clip(correction, 0.85, 1.18))
    body_center_ijk = np.array(target_stats["body"].centroid_ijk, dtype=float)
    body_center_mm = np.array(target_stats["body"].centroid_mm, dtype=float)
    adjusted: dict[str, ShapeGroupStats] = {}
    for group_id, name, group_labels, xy_coupling, _ in GROUP_DEFINITIONS:
        stats = target_stats[group_id]
        group_correction = correction if group_id == "body" else 1.0 + xy_coupling * (correction - 1.0)
        centroid_ijk = np.array(stats.centroid_ijk, dtype=float)
        centroid_mm = np.array(stats.centroid_mm, dtype=float)
        relative_ijk = centroid_ijk - body_center_ijk
        relative_mm = centroid_mm - body_center_mm
        target_centroid_ijk = body_center_ijk + relative_ijk * np.array([group_correction, group_correction, 1.0])
        target_centroid_mm = body_center_mm + relative_mm * np.array([group_correction, group_correction, 1.0])
        bbox_voxels = np.array(stats.bbox_voxels, dtype=float) * np.array([group_correction, group_correction, 1.0])
        bbox_mm = np.array(stats.bbox_mm, dtype=float) * np.array([group_correction, group_correction, 1.0])
        adjusted[group_id] = ShapeGroupStats(
            group_id=group_id,
            name=name,
            labels=group_labels,
            present=stats.present,
            voxel_count=stats.voxel_count,
            volume_cm3=float(stats.volume_cm3 * group_correction * group_correction),
            centroid_ijk=tuple(float(value) for value in target_centroid_ijk),
            centroid_mm=tuple(float(value) for value in target_centroid_mm),
            bbox_voxels=tuple(float(max(value, 1.0)) for value in bbox_voxels),
            bbox_mm=tuple(float(max(value, 0.0)) for value in bbox_mm),
            waist_cm=float(stats.waist_cm * correction) if group_id == "body" and stats.waist_cm else stats.waist_cm,
        )
    return adjusted


def _warp_binary_mask_to_stats(
    mask: np.ndarray,
    source_stats: ShapeGroupStats,
    target_stats: ShapeGroupStats,
    output_shape: tuple[int, int, int],
    ndimage,
) -> np.ndarray:
    source_center = np.array(source_stats.centroid_ijk, dtype=float)
    target_center = np.array(target_stats.centroid_ijk, dtype=float)
    source_extent = np.maximum(np.array(source_stats.bbox_voxels, dtype=float), 1.0)
    target_extent = np.maximum(np.array(target_stats.bbox_voxels, dtype=float), 1.0)
    matrix = np.diag(source_extent / target_extent)
    offset = source_center - matrix @ target_center
    warped = ndimage.affine_transform(
        mask.astype(np.uint8),
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return warped > 0


def _warp_labels_to_stats(
    labels: np.ndarray,
    source_stats: ShapeGroupStats,
    target_stats: ShapeGroupStats,
    output_shape: tuple[int, int, int],
    ndimage,
) -> np.ndarray:
    source_center = np.array(source_stats.centroid_ijk, dtype=float)
    target_center = np.array(target_stats.centroid_ijk, dtype=float)
    source_extent = np.maximum(np.array(source_stats.bbox_voxels, dtype=float), 1.0)
    target_extent = np.maximum(np.array(target_stats.bbox_voxels, dtype=float), 1.0)
    matrix = np.diag(source_extent / target_extent)
    offset = source_center - matrix @ target_center
    return ndimage.affine_transform(
        labels,
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(labels.dtype)


def _clean_body(mask: np.ndarray, ndimage) -> np.ndarray:
    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    filled = np.zeros_like(closed, dtype=bool)
    for z_index in range(closed.shape[2]):
        if np.any(closed[:, :, z_index]):
            filled[:, :, z_index] = ndimage.binary_fill_holes(closed[:, :, z_index])
    connected, count = ndimage.label(filled, structure=np.ones((3, 3, 3), dtype=bool))
    if count <= 1:
        return filled
    sizes = np.bincount(connected.ravel(), minlength=count + 1)
    sizes[0] = 0
    return connected == int(sizes.argmax())


def _fill_body_background(labels: np.ndarray, body: np.ndarray, spacing_mm: tuple[float, float, float], adipose_layer_mm: float, ndimage) -> np.ndarray:
    output = labels.copy()
    output[np.isin(output, DEEP_STRUCTURE_LABELS)] = 4
    distance_to_skin = ndimage.distance_transform_edt(body, sampling=spacing_mm)
    empty_body = body & (output == 0)
    output[empty_body & (distance_to_skin <= adipose_layer_mm)] = 3
    output[empty_body & (distance_to_skin > adipose_layer_mm)] = 4
    output[~body] = 0
    return output


def _overlay_group_labels(
    morphed: np.ndarray,
    reference_labels: np.ndarray,
    reference_stats: dict[str, ShapeGroupStats],
    target_stats: dict[str, ShapeGroupStats],
    body: np.ndarray,
    ndimage,
) -> np.ndarray:
    output = morphed.copy()
    for group_id, _, group_labels, *_ in GROUP_DEFINITIONS:
        if group_id == "body" or not target_stats[group_id].present:
            continue
        for label_value in group_labels:
            source_mask = reference_labels == label_value
            if not np.any(source_mask):
                continue
            warped = _warp_binary_mask_to_stats(
                source_mask,
                source_stats=reference_stats[group_id],
                target_stats=target_stats[group_id],
                output_shape=reference_labels.shape,
                ndimage=ndimage,
            )
            output[warped & body] = int(label_value)
    output[~body] = 0
    return output


def _write_registration_csv(path: Path, registrations: tuple[PopulationRegistrationStats, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "case_id",
                "source_path",
                "body_volume_cm3",
                "waist_cm",
                "bbox_x_mm",
                "bbox_y_mm",
                "bbox_z_mm",
                "registration_scale_x",
                "registration_scale_y",
                "registration_scale_z",
                "registration_translation_x_vox",
                "registration_translation_y_vox",
                "registration_translation_z_vox",
            ]
        )
        for item in registrations:
            writer.writerow(
                [
                    item.case_id,
                    item.source_path,
                    f"{item.registered_body_volume_cm3:.6f}",
                    f"{item.registered_waist_cm:.6f}",
                    *[f"{value:.6f}" for value in item.registered_bbox_mm],
                    *[f"{value:.6f}" for value in item.registration_scale],
                    *[f"{value:.6f}" for value in item.registration_translation_voxels],
                ]
            )


def _write_deformation_transforms_csv(
    path: Path,
    reference_stats: dict[str, ShapeGroupStats],
    target_stats: dict[str, ShapeGroupStats],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "group_id",
                "group_name",
                "labels",
                "source_centroid_i",
                "source_centroid_j",
                "source_centroid_k",
                "target_centroid_i",
                "target_centroid_j",
                "target_centroid_k",
                "scale_x",
                "scale_y",
                "scale_z",
            ]
        )
        for group_id, name, group_labels, *_ in GROUP_DEFINITIONS:
            source = reference_stats[group_id]
            target = target_stats[group_id]
            source_extent = np.maximum(np.array(source.bbox_voxels, dtype=float), 1.0)
            target_extent = np.maximum(np.array(target.bbox_voxels, dtype=float), 1.0)
            scale = target_extent / source_extent
            writer.writerow(
                [
                    group_id,
                    name,
                    " ".join(str(label) for label in group_labels),
                    *[f"{value:.6f}" for value in source.centroid_ijk],
                    *[f"{value:.6f}" for value in target.centroid_ijk],
                    *[f"{value:.6f}" for value in scale],
                ]
            )


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(value // 2 for value in mask.shape)
    return tuple(int(round(float(np.median(coords[:, axis])))) for axis in range(3))


def _render_preview(
    path: Path,
    reference_labels: np.ndarray,
    morphed_labels: np.ndarray,
    reference_body: np.ndarray,
    morphed_body: np.ndarray,
    regions: list[dict[str, Any]],
    summary: dict[str, float],
) -> None:
    plt, ListedColormap, Patch, *_ = _import_dependencies()
    colors = [str(region.get("color", "#000000")) for region in sorted(regions, key=lambda item: int(item["index"]))]
    cmap = ListedColormap(colors)
    vmax = max(int(region["index"]) for region in regions)
    _, _, z_index = _slice_indices(reference_body | morphed_body)
    x_index, y_index, _ = _slice_indices(morphed_body)
    views = [
        ("Axial", reference_labels[:, :, z_index], morphed_labels[:, :, z_index]),
        ("Coronal", reference_labels[:, y_index, :], morphed_labels[:, y_index, :]),
        ("Sagittal", reference_labels[x_index, :, :], morphed_labels[x_index, :, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    for ax in axes.ravel():
        ax.axis("off")
        ax.set_facecolor("#f6f1e8")
    for col, (title, reference_view, morphed_view) in enumerate(views):
        axes[0, col].imshow(np.rot90(reference_view), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[0, col].set_title(f"Reference {title}", fontsize=10, color="#13202a")
        axes[1, col].imshow(np.rot90(morphed_view), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[1, col].set_title(f"Statistical Morph {title}", fontsize=10, color="#13202a")

    handles = [
        Patch(facecolor="#ffd166", label="adipose envelope"),
        Patch(facecolor="#d95d39", label="muscle / soft tissue"),
        Patch(facecolor="#48cae4", label="lungs"),
        Patch(facecolor="#9d4edd", label="liver"),
        Patch(facecolor="#f72585", label="kidneys"),
        Patch(facecolor="#e9ecef", label="bone"),
        Patch(facecolor="#ff9f1c", label="vessel wall"),
        Patch(facecolor="#0077b6", label="vascular fluid"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.50), fontsize=7)
    fig.suptitle("Registration-Based Statistical Anatomy Morph", fontsize=15, color="#13202a")
    fig.text(
        0.08,
        0.02,
        f"Population cases: {summary['population_case_count']:.0f}; PCA modes: {summary['shape_mode_count']:.0f}; "
        f"waist proxy: {summary['baseline_waist_cm']:.1f} -> {summary['achieved_waist_cm']:.1f} cm; "
        f"body volume: {summary['baseline_body_volume_l']:.2f} -> {summary['morphed_body_volume_l']:.2f} L",
        fontsize=9,
        color="#1e2a32",
    )
    fig.tight_layout(rect=(0, 0.04, 0.90, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_spec(
    path: Path,
    source_spec_path: Path,
    population_label_paths: tuple[str | Path, ...],
    mode_weights: tuple[float, ...],
    result: StatisticalAnatomyResult,
) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "package_type": "statistical_registration_anatomy_morph",
        "source_combined_spec": str(source_spec_path),
        "population_label_paths": [str(path) for path in population_label_paths],
        "model": {
            "population_case_count": result.population_case_count,
            "shape_mode_count": result.shape_mode_count,
            "mode_weights": list(mode_weights),
        },
        "target": {
            "target_height_cm": result.target_height_cm,
            "target_bmi": result.target_bmi,
            "target_waist_cm": result.target_waist_cm,
            "achieved_waist_cm": result.achieved_waist_cm,
        },
        "outputs": {
            "blood_material_labels": result.morphed_blood_material_labels_path,
            "blood_mass_density_g_cm3": result.morphed_blood_density_path,
            "blood_relative_electron_density": result.morphed_blood_relative_electron_density_path,
            "blood_synthetic_hu": result.morphed_blood_synthetic_hu_path,
            "contrast_material_labels": result.morphed_contrast_material_labels_path,
            "contrast_mass_density_g_cm3": result.morphed_contrast_density_path,
            "contrast_relative_electron_density": result.morphed_contrast_relative_electron_density_path,
            "contrast_synthetic_hu": result.morphed_contrast_synthetic_hu_path,
            "body_mask": result.morphed_body_mask_path,
            "vascular_fluid_mask": result.morphed_vascular_fluid_mask_path,
            "vessel_wall_mask": result.morphed_vessel_wall_mask_path,
            "shape_model_npz": result.shape_model_npz_path,
            "registration_csv": result.registration_csv_path,
            "deformation_transforms_csv": result.deformation_transforms_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "quality_summary": {
            "baseline_body_volume_cm3": result.baseline_body_volume_cm3,
            "morphed_body_volume_cm3": result.morphed_body_volume_cm3,
            "body_volume_change_percent": result.body_volume_change_percent,
            "baseline_bbox_mm": list(result.baseline_bbox_mm),
            "morphed_bbox_mm": list(result.morphed_bbox_mm),
            "vascular_components": result.vascular_components,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(
    result: StatisticalAnatomyResult,
    registrations: tuple[PopulationRegistrationStats, ...],
    mode_weights: tuple[float, ...],
) -> str:
    lines = [
        "# Statistical Registration Anatomy Morph Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Target",
        "",
        f"- Target height: {'not supplied' if result.target_height_cm is None else f'{result.target_height_cm:.1f} cm'}",
        f"- Target BMI: {'not supplied' if result.target_bmi is None else f'{result.target_bmi:.2f} kg/m2'}",
        f"- Target waist: {'not supplied' if result.target_waist_cm is None else f'{result.target_waist_cm:.1f} cm'}",
        f"- Achieved waist proxy: {result.achieved_waist_cm:.1f} cm",
        "",
        "## Population Model",
        "",
        f"- Registered population cases: {result.population_case_count}",
        f"- PCA shape modes available: {result.shape_mode_count}",
        f"- Applied mode weights: {', '.join(f'{value:.3f}' for value in mode_weights) if mode_weights else 'none'}",
        f"- Shape model NPZ: `{Path(result.shape_model_npz_path).name}`",
        f"- Registration CSV: `{Path(result.registration_csv_path).name}`",
        f"- Deformation transforms CSV: `{Path(result.deformation_transforms_csv_path).name}`",
        "",
        "## Registered Cases",
        "",
        "| case | waist cm | body L | bbox mm | registration scale |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in registrations:
        lines.append(
            f"| {item.case_id} | {item.registered_waist_cm:.1f} | "
            f"{item.registered_body_volume_cm3 / 1000.0:.2f} | "
            f"{', '.join(f'{value:.1f}' for value in item.registered_bbox_mm)} | "
            f"{', '.join(f'{value:.3f}' for value in item.registration_scale)} |"
        )
    lines.extend(
        [
            "",
            "## Volume QA",
            "",
            f"- Body volume: {result.baseline_body_volume_cm3 / 1000.0:.3f} L -> {result.morphed_body_volume_cm3 / 1000.0:.3f} L",
            f"- Body volume change: {result.body_volume_change_percent:.2f}%",
            f"- Baseline bbox mm: {', '.join(f'{value:.1f}' for value in result.baseline_bbox_mm)}",
            f"- Morphed bbox mm: {', '.join(f'{value:.1f}' for value in result.morphed_bbox_mm)}",
            f"- Vascular connected components after morph: {result.vascular_components}",
            "",
            "## Outputs",
            "",
            f"- Blood material labels: `{Path(result.morphed_blood_material_labels_path).name}`",
            f"- Blood HU/density/RED: `{Path(result.morphed_blood_synthetic_hu_path).name}`, `{Path(result.morphed_blood_density_path).name}`, `{Path(result.morphed_blood_relative_electron_density_path).name}`",
            f"- Contrast material labels: `{Path(result.morphed_contrast_material_labels_path).name}`",
            f"- Body mask: `{Path(result.morphed_body_mask_path).name}`",
            f"- Vascular fluid mask: `{Path(result.morphed_vascular_fluid_mask_path).name}`",
            f"- Vessel wall mask: `{Path(result.morphed_vessel_wall_mask_path).name}`",
            f"- Preview PNG: `{Path(result.preview_png_path).name}`",
            f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
            "",
            "## Interpretation",
            "",
            "- This module is a population-registration scaffold: cases are affine body-registered, summarized as organ/body shape descriptors, and used to synthesize a new anatomy variant.",
            "- PCA modes become meaningful only after several independent real segmented CT cases are staged.",
            "- Stage 001 uses compact group-level transforms rather than full dense diffeomorphic registration.",
            "- Regenerate 3D meshes, RT packages, and flow packages from this morphed volume before comparing downstream behavior.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_statistical_anatomy_morph(
    combined_spec_path: str | Path,
    population_label_paths: tuple[str | Path, ...],
    output_dir: str | Path = "outputs/digital/statistical_anatomy_morph",
    case_id: str = "ct_org_population_statistical_morph",
    population_case_ids: tuple[str, ...] | None = None,
    target_height_cm: float | None = None,
    target_weight_kg: float | None = None,
    target_bmi: float | None = None,
    target_waist_cm: float | None = None,
    baseline_height_cm: float = 170.0,
    baseline_bmi: float = 24.0,
    mode_weights: tuple[float, ...] = (),
    max_modes: int = 3,
    adipose_layer_mm: float = 18.0,
    report_path: str | Path | None = "outputs/reports/statistical_anatomy_morph_stage001.md",
) -> StatisticalAnatomyResult:
    if not population_label_paths:
        raise ValueError("At least one population label map is required")
    _, _, _, nib, ndimage, _ = _import_dependencies()
    spec_path = Path(combined_spec_path)
    spec = _load_yaml(spec_path)
    outputs = spec.get("outputs", {})
    regions = list(spec.get("regions", []))
    if not isinstance(outputs, dict) or not isinstance(regions, list):
        raise ValueError("Combined spec must contain outputs and regions")

    blood_labels_path = _resolve_path(str(outputs["blood_material_labels"]), spec_path)
    contrast_labels_path = _resolve_path(str(outputs.get("contrast_material_labels", outputs["blood_material_labels"])), spec_path)
    blood_image = nib.load(str(blood_labels_path))
    contrast_image = nib.load(str(contrast_labels_path))
    reference_blood = np.rint(np.asanyarray(blood_image.dataobj)).astype(np.int16)
    reference_contrast = np.rint(np.asanyarray(contrast_image.dataobj)).astype(np.int16)
    if reference_blood.shape != reference_contrast.shape:
        raise ValueError("Blood and contrast material-label maps must have the same shape")

    spacing_mm = tuple(float(value) for value in blood_image.header.get_zooms()[:3])
    voxel_volume = _voxel_volume_cm3(spacing_mm)
    reference_body = reference_blood > 0
    if not np.any(reference_body):
        raise ValueError("Reference combined phantom has an empty body")
    reference_stats = _all_group_stats(reference_blood, spacing_mm, voxel_volume)

    case_ids = _population_case_ids(tuple(population_label_paths), population_case_ids)
    feature_names: list[str] | None = None
    feature_rows: list[np.ndarray] = []
    registrations: list[PopulationRegistrationStats] = []
    for case_id_value, label_path in zip(case_ids, population_label_paths, strict=True):
        loaded_labels = _load_label_like_reference(label_path, blood_image, nib, ndimage)
        registered_labels, scale, translation = _register_labels_to_reference(loaded_labels, reference_body, ndimage)
        stats = _all_group_stats(registered_labels, spacing_mm, voxel_volume)
        names, row = _case_feature_vector(stats)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Population feature schema changed unexpectedly")
        feature_rows.append(row)
        body_stats = stats["body"]
        registrations.append(
            PopulationRegistrationStats(
                case_id=case_id_value,
                source_path=str(label_path),
                registered_body_volume_cm3=body_stats.volume_cm3,
                registered_waist_cm=float(body_stats.waist_cm or 0.0),
                registered_bbox_mm=body_stats.bbox_mm,
                registration_scale=scale,
                registration_translation_voxels=translation,
                groups=tuple(stats[group_id] for group_id, *_ in GROUP_DEFINITIONS),
            )
        )

    assert feature_names is not None
    feature_matrix = np.vstack(feature_rows)
    mean, std, components, singular_values = _fit_feature_pca(feature_matrix, max_modes=max_modes)
    target_features = _target_feature_map(feature_names, mean, std, components, mode_weights)
    target_stats = _stats_from_feature_map(target_features, reference_stats, spacing_mm)
    resolved_bmi = _derive_target_bmi(target_height_cm, target_weight_kg, target_bmi)
    population_waist = float(target_stats["body"].waist_cm or reference_stats["body"].waist_cm or 0.0)
    resolved_waist = _derive_target_waist_cm(target_waist_cm, population_waist, resolved_bmi, baseline_bmi)
    target_stats = _apply_anthropometry_to_target_stats(
        target_stats,
        reference_stats=reference_stats,
        target_height_cm=target_height_cm,
        target_waist_cm=resolved_waist,
        baseline_height_cm=baseline_height_cm,
    )
    if resolved_waist is not None:
        for _ in range(5):
            probe_body = _warp_binary_mask_to_stats(
                reference_body,
                source_stats=reference_stats["body"],
                target_stats=target_stats["body"],
                output_shape=reference_blood.shape,
                ndimage=ndimage,
            )
            probe_body = _clean_body(probe_body, ndimage)
            probe_waist = _estimate_waist_cm(probe_body, spacing_mm)
            if probe_waist <= 0.0:
                break
            correction = float(resolved_waist / probe_waist)
            if abs(correction - 1.0) < 0.015:
                break
            target_stats = _scale_target_stats_xy(target_stats, correction)

    morphed_body = _warp_binary_mask_to_stats(
        reference_body,
        source_stats=reference_stats["body"],
        target_stats=target_stats["body"],
        output_shape=reference_blood.shape,
        ndimage=ndimage,
    )
    morphed_body = _clean_body(morphed_body, ndimage)

    globally_morphed_blood = _warp_labels_to_stats(
        reference_blood,
        source_stats=reference_stats["body"],
        target_stats=target_stats["body"],
        output_shape=reference_blood.shape,
        ndimage=ndimage,
    )
    globally_morphed_contrast = _warp_labels_to_stats(
        reference_contrast,
        source_stats=reference_stats["body"],
        target_stats=target_stats["body"],
        output_shape=reference_contrast.shape,
        ndimage=ndimage,
    )
    morphed_blood = _fill_body_background(globally_morphed_blood, morphed_body, spacing_mm, adipose_layer_mm, ndimage)
    morphed_contrast = _fill_body_background(globally_morphed_contrast, morphed_body, spacing_mm, adipose_layer_mm, ndimage)
    morphed_blood = _overlay_group_labels(morphed_blood, reference_blood, reference_stats, target_stats, morphed_body, ndimage)
    morphed_contrast = _overlay_group_labels(morphed_contrast, reference_contrast, reference_stats, target_stats, morphed_body, ndimage)

    vascular_fluid = np.isin(morphed_blood, (14, 15))
    connected, vascular_components = ndimage.label(vascular_fluid, structure=np.ones((3, 3, 3), dtype=bool))
    if vascular_components > 1:
        sizes = np.bincount(connected.ravel(), minlength=vascular_components + 1)
        sizes[0] = 0
        largest = int(sizes.argmax())
        keep = connected == largest
        morphed_blood[vascular_fluid & ~keep] = 4
        morphed_contrast[np.isin(morphed_contrast, (14, 15)) & ~keep] = 4
        vascular_fluid = keep
        vascular_components = 1
    vessel_wall = morphed_blood == 13

    blood_density = _property_volume(morphed_blood, regions, "mass_density_g_cm3")
    blood_red = _property_volume(morphed_blood, regions, "relative_electron_density")
    blood_hu = _property_volume(morphed_blood, regions, "target_hu_midpoint")
    contrast_density = _property_volume(morphed_contrast, regions, "mass_density_g_cm3")
    contrast_red = _property_volume(morphed_contrast, regions, "relative_electron_density")
    contrast_hu = _property_volume(morphed_contrast, regions, "target_hu_midpoint")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blood_labels_out = output / f"{case_id}_stat_morphed_material_labels_blood_v001.nii.gz"
    blood_density_out = output / f"{case_id}_stat_morphed_mass_density_blood_v001.nii.gz"
    blood_red_out = output / f"{case_id}_stat_morphed_relative_electron_density_blood_v001.nii.gz"
    blood_hu_out = output / f"{case_id}_stat_morphed_synthetic_hu_blood_v001.nii.gz"
    contrast_labels_out = output / f"{case_id}_stat_morphed_material_labels_contrast_v001.nii.gz"
    contrast_density_out = output / f"{case_id}_stat_morphed_mass_density_contrast_v001.nii.gz"
    contrast_red_out = output / f"{case_id}_stat_morphed_relative_electron_density_contrast_v001.nii.gz"
    contrast_hu_out = output / f"{case_id}_stat_morphed_synthetic_hu_contrast_v001.nii.gz"
    body_mask_out = output / f"{case_id}_stat_morphed_body_mask_v001.nii.gz"
    vascular_fluid_out = output / f"{case_id}_stat_morphed_vascular_fluid_mask_v001.nii.gz"
    vessel_wall_out = output / f"{case_id}_stat_morphed_vessel_wall_mask_v001.nii.gz"
    shape_model_out = output / f"{case_id}_statistical_shape_model_v001.npz"
    registration_csv_out = output / f"{case_id}_population_registration_summary_v001.csv"
    transforms_csv_out = output / f"{case_id}_deformation_transforms_v001.csv"
    preview_out = output / f"{case_id}_statistical_anatomy_morph_preview_v001.png"
    spec_out = output / f"{case_id}_statistical_anatomy_morph_spec_v001.yaml"

    _write_nifti(blood_labels_out, morphed_blood.astype(np.int16), blood_image, nib)
    _write_nifti(blood_density_out, blood_density.astype(np.float32), blood_image, nib)
    _write_nifti(blood_red_out, blood_red.astype(np.float32), blood_image, nib)
    _write_nifti(blood_hu_out, blood_hu.astype(np.float32), blood_image, nib)
    _write_nifti(contrast_labels_out, morphed_contrast.astype(np.int16), blood_image, nib)
    _write_nifti(contrast_density_out, contrast_density.astype(np.float32), blood_image, nib)
    _write_nifti(contrast_red_out, contrast_red.astype(np.float32), blood_image, nib)
    _write_nifti(contrast_hu_out, contrast_hu.astype(np.float32), blood_image, nib)
    _write_nifti(body_mask_out, morphed_body.astype(np.uint8), blood_image, nib)
    _write_nifti(vascular_fluid_out, vascular_fluid.astype(np.uint8), blood_image, nib)
    _write_nifti(vessel_wall_out, vessel_wall.astype(np.uint8), blood_image, nib)

    np.savez_compressed(
        shape_model_out,
        feature_names=np.array(feature_names, dtype=object),
        feature_matrix=feature_matrix,
        feature_mean=mean,
        feature_std=std,
        components=components,
        singular_values=singular_values,
        mode_weights=np.array(mode_weights, dtype=np.float64),
        population_case_ids=np.array(case_ids, dtype=object),
    )
    _write_registration_csv(registration_csv_out, tuple(registrations))
    _write_deformation_transforms_csv(transforms_csv_out, reference_stats, target_stats)

    baseline_body_volume = float(reference_body.sum() * voxel_volume)
    morphed_body_volume = float(morphed_body.sum() * voxel_volume)
    body_volume_change = (
        100.0 * (morphed_body_volume - baseline_body_volume) / baseline_body_volume
        if baseline_body_volume > 0.0
        else 0.0
    )
    achieved_waist = _estimate_waist_cm(morphed_body, spacing_mm)
    baseline_waist = float(reference_stats["body"].waist_cm or 0.0)
    notes = [
        "population_registration_stage001_group_affine_descriptors",
        "pca_modes_require_multiple_independent_segmented_ct_cases",
        "dense_diffeomorphic_registration_not_yet_enabled",
    ]
    if len(population_label_paths) < 3:
        notes.append("limited_population_size_use_three_or_more_real_cases_for_stable_statistics")
    if resolved_waist is not None:
        notes.append("target_waist_used_to_condition_body_envelope")
    if resolved_bmi is not None:
        notes.append("target_bmi_recorded_for_anthropometric_conditioning")

    result = StatisticalAnatomyResult(
        case_id=case_id,
        output_dir=str(output),
        morphed_blood_material_labels_path=str(blood_labels_out),
        morphed_blood_density_path=str(blood_density_out),
        morphed_blood_relative_electron_density_path=str(blood_red_out),
        morphed_blood_synthetic_hu_path=str(blood_hu_out),
        morphed_contrast_material_labels_path=str(contrast_labels_out),
        morphed_contrast_density_path=str(contrast_density_out),
        morphed_contrast_relative_electron_density_path=str(contrast_red_out),
        morphed_contrast_synthetic_hu_path=str(contrast_hu_out),
        morphed_body_mask_path=str(body_mask_out),
        morphed_vascular_fluid_mask_path=str(vascular_fluid_out),
        morphed_vessel_wall_mask_path=str(vessel_wall_out),
        shape_model_npz_path=str(shape_model_out),
        registration_csv_path=str(registration_csv_out),
        deformation_transforms_csv_path=str(transforms_csv_out),
        preview_png_path=str(preview_out),
        spec_yaml_path=str(spec_out),
        report_path=str(report_path) if report_path is not None else str(output / f"{case_id}_statistical_anatomy_morph_report_v001.md"),
        population_case_count=len(population_label_paths),
        shape_mode_count=int(components.shape[0]),
        target_height_cm=target_height_cm,
        target_bmi=resolved_bmi,
        target_waist_cm=resolved_waist,
        achieved_waist_cm=achieved_waist,
        baseline_body_volume_cm3=baseline_body_volume,
        morphed_body_volume_cm3=morphed_body_volume,
        body_volume_change_percent=body_volume_change,
        baseline_bbox_mm=reference_stats["body"].bbox_mm,
        morphed_bbox_mm=_bbox_mm(morphed_body, spacing_mm),
        vascular_components=int(vascular_components),
        notes=tuple(notes),
    )

    _render_preview(
        preview_out,
        reference_labels=reference_blood,
        morphed_labels=morphed_blood,
        reference_body=reference_body,
        morphed_body=morphed_body,
        regions=regions,
        summary={
            "population_case_count": float(result.population_case_count),
            "shape_mode_count": float(result.shape_mode_count),
            "baseline_waist_cm": baseline_waist,
            "achieved_waist_cm": result.achieved_waist_cm,
            "baseline_body_volume_l": result.baseline_body_volume_cm3 / 1000.0,
            "morphed_body_volume_l": result.morphed_body_volume_cm3 / 1000.0,
        },
    )
    _write_spec(spec_out, spec_path, tuple(population_label_paths), mode_weights, result)
    report = _format_report(result, tuple(registrations), mode_weights)
    Path(result.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result.report_path).write_text(report)
    return result


def format_statistical_anatomy_morph_result(result: StatisticalAnatomyResult) -> str:
    lines = [
        "# Statistical Registration Anatomy Morph",
        "",
        f"Case ID: `{result.case_id}`",
        f"Population cases: {result.population_case_count}",
        f"PCA shape modes: {result.shape_mode_count}",
        f"Achieved waist proxy: {result.achieved_waist_cm:.1f} cm",
        f"Body volume: {result.baseline_body_volume_cm3 / 1000.0:.3f} L -> {result.morphed_body_volume_cm3 / 1000.0:.3f} L",
        "",
        "## Outputs",
        "",
        f"- Blood labels: `{result.morphed_blood_material_labels_path}`",
        f"- Contrast labels: `{result.morphed_contrast_material_labels_path}`",
        f"- Body mask: `{result.morphed_body_mask_path}`",
        f"- Vascular fluid mask: `{result.morphed_vascular_fluid_mask_path}`",
        f"- Shape model: `{result.shape_model_npz_path}`",
        f"- Registration summary: `{result.registration_csv_path}`",
        f"- Deformation transforms: `{result.deformation_transforms_csv_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        f"- Spec YAML: `{result.spec_yaml_path}`",
    ]
    return "\n".join(lines)
