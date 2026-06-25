from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np
import yaml


ABDOMINAL_REGION_SPECS: tuple[tuple[int, str, str, str], ...] = (
    (16, "spleen", "spleen", "#7b2cbf"),
    (17, "stomach_bowel_wall", "stomach_bowel_wall", "#ffb703"),
    (18, "gallbladder_bile_fluid", "gallbladder_bile_fluid", "#84a59d"),
    (19, "esophagus_wall", "stomach_bowel_wall", "#f77f00"),
    (20, "pancreas", "pancreas", "#f28482"),
    (21, "adrenal_glands", "adrenal_soft_tissue", "#b08968"),
    (22, "gi_lumen_gas", "gi_lumen_gas", "#22223b"),
    (23, "gi_lumen_fluid", "gi_lumen_fluid", "#80ffdb"),
    (24, "duodenum_wall_placeholder", "duodenum_wall", "#f4a261"),
    (25, "small_bowel_wall_placeholder", "small_bowel_wall", "#e9c46a"),
    (26, "colon_wall_placeholder", "colon_wall", "#2a9d8f"),
    (27, "rectum_wall_placeholder", "rectum_wall", "#8d6a9f"),
    (28, "small_bowel_lumen_fluid_placeholder", "gi_lumen_fluid", "#64dfdf"),
    (29, "colon_lumen_gas_placeholder", "gi_lumen_gas", "#1d3557"),
    (30, "colon_lumen_fluid_placeholder", "gi_lumen_fluid", "#4cc9f0"),
    (31, "rectum_lumen_fluid_placeholder", "gi_lumen_fluid", "#90be6d"),
)


@dataclass(frozen=True)
class BTCVAbdominalOrganDefinition:
    organ_id: str
    label: str
    source_label_ids: tuple[int, ...]
    material_label_id: int
    notes: tuple[str, ...] = ()


BTCV_ORGAN_DEFINITIONS: tuple[BTCVAbdominalOrganDefinition, ...] = (
    BTCVAbdominalOrganDefinition("spleen", "Spleen", (1,), 16),
    BTCVAbdominalOrganDefinition("stomach_wall", "Stomach/bowel wall", (7,), 17),
    BTCVAbdominalOrganDefinition("gallbladder", "Gallbladder / bile fluid", (4,), 18),
    BTCVAbdominalOrganDefinition("esophagus", "Esophagus wall", (5,), 19),
    BTCVAbdominalOrganDefinition("pancreas", "Pancreas", (11,), 20),
    BTCVAbdominalOrganDefinition("adrenal_glands", "Adrenal glands", (12, 13), 21),
)


GI_TARGETS: tuple[str, ...] = ("stomach", "duodenum", "small_bowel", "colon", "rectum")
GI_TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "stomach": ("stomach", "gastric"),
    "duodenum": ("duodenum", "duodenal"),
    "small_bowel": ("small_bowel", "small bowel", "small_intestine", "small intestine", "jejunum", "ileum", "jejunum_ileum"),
    "colon": ("colon", "large_bowel", "large bowel", "large_intestine", "large intestine"),
    "rectum": ("rectum", "rectal"),
}
GI_DIRECTORY_FILES: dict[str, tuple[str, ...]] = {
    "stomach": ("stomach.nii.gz", "stomach.nii", "stomach_label.nii.gz"),
    "duodenum": ("duodenum.nii.gz", "duodenum.nii", "duodenum_label.nii.gz"),
    "small_bowel": ("small_bowel.nii.gz", "small_bowel.nii", "small_intestine.nii.gz", "small_intestine.nii", "jejunum_ileum.nii.gz"),
    "colon": ("colon.nii.gz", "colon.nii", "large_bowel.nii.gz", "large_intestine.nii.gz"),
    "rectum": ("rectum.nii.gz", "rectum.nii", "rectum_label.nii.gz"),
}
GI_WALL_LABELS: dict[str, int] = {
    "stomach": 17,
    "duodenum": 24,
    "small_bowel": 25,
    "colon": 26,
    "rectum": 27,
}


@dataclass(frozen=True)
class AbdominalOrganMetric:
    organ_id: str
    label: str
    source_label_ids: tuple[int, ...]
    material_label_id: int
    source_voxels: int
    material_voxels: int
    volume_cm3: float
    centroid_ijk: tuple[float, float, float] | None
    centroid_mm: tuple[float, float, float] | None
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class AbdominalOrganAugmentationResult:
    case_id: str
    material_label_path: str
    mass_density_path: str
    relative_electron_density_path: str
    synthetic_hu_path: str
    gi_lumen_mask_path: str
    gi_gas_mask_path: str
    gi_fluid_mask_path: str
    gi_tract_placeholder_labels_path: str
    gi_real_segmentation_source_path: str | None
    gi_real_segmentation_labelmap_path: str | None
    gi_real_segmentation_applied_targets: tuple[str, ...]
    organ_metrics_csv_path: str
    qa_yaml_path: str
    preview_png_path: str
    report_path: str
    pass_count: int
    review_count: int
    fail_count: int
    metrics: tuple[AbdominalOrganMetric, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RealGILoadResult:
    masks: dict[str, np.ndarray]
    source_path: str | None
    labelmap_path: str | None
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from matplotlib.colors import ListedColormap  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
        from scipy import ndimage  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Abdominal organ preservation requires matplotlib, nibabel, and scipy.") from exc
    return plt, ListedColormap, Patch, nib, ndimage


def _write_nifti(path: Path, data: np.ndarray, reference_image, nib) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _load_yaml_if_exists(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    data = yaml.safe_load(yaml_path.read_text())
    return data if isinstance(data, dict) else {}


def _normalise_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _target_from_name(value: str) -> str | None:
    normalised = _normalise_name(value)
    for target, aliases in GI_TARGET_ALIASES.items():
        if normalised == target:
            return target
        for alias in aliases:
            alias_norm = _normalise_name(alias)
            if normalised == alias_norm or alias_norm in normalised:
                return target
    return None


def _label_ids_by_target(labelmap: dict[str, Any]) -> dict[str, set[int]]:
    by_target: dict[str, set[int]] = {target: set() for target in GI_TARGETS}
    labels = labelmap.get("labels", {})
    if isinstance(labels, dict):
        for raw_label, payload in labels.items():
            try:
                label_id = int(raw_label)
            except (TypeError, ValueError):
                continue
            name = ""
            target = None
            if isinstance(payload, dict):
                name = str(payload.get("name", ""))
                raw_target = payload.get("target")
                target = _target_from_name(str(raw_target)) if raw_target is not None else None
            else:
                name = str(payload)
            target = target or _target_from_name(name)
            if target is not None:
                by_target[target].add(label_id)
    return by_target


def _directory_files_by_target(labelmap: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    files = {target: list(GI_DIRECTORY_FILES[target]) for target in GI_TARGETS}
    targets = labelmap.get("targets", {})
    if isinstance(targets, dict):
        for raw_target, payload in targets.items():
            target = _target_from_name(str(raw_target))
            if target is None or not isinstance(payload, dict):
                continue
            for raw_file in payload.get("files", []) or []:
                if raw_file is not None:
                    files[target].append(str(raw_file))
    return {target: tuple(dict.fromkeys(values)) for target, values in files.items()}


def _validate_mask_image(image, *, reference_shape: tuple[int, int, int], reference_spacing: tuple[float, float, float], path: Path) -> None:
    shape = tuple(int(value) for value in image.shape[:3])
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if shape != reference_shape:
        raise ValueError(f"GI segmentation shape differs from CT/material grid for {path}: {shape} vs {reference_shape}")
    if any(abs(spacing[index] - reference_spacing[index]) > 1e-5 for index in range(3)):
        raise ValueError(f"GI segmentation spacing differs from CT/material grid for {path}: {spacing} vs {reference_spacing}")


def _load_real_gi_segmentation(
    *,
    gi_segmentation_path: str | Path | None,
    gi_labelmap_path: str | Path | None,
    reference_shape: tuple[int, int, int],
    reference_spacing: tuple[float, float, float],
    nib,
) -> RealGILoadResult:
    if gi_segmentation_path is None or str(gi_segmentation_path) == "":
        return RealGILoadResult({}, None, None if gi_labelmap_path is None else str(gi_labelmap_path), ("real_gi_segmentation_not_supplied",))

    source = Path(gi_segmentation_path)
    if not source.exists():
        raise FileNotFoundError(f"GI segmentation path not found: {source}")

    labelmap = _load_yaml_if_exists(gi_labelmap_path)
    masks: dict[str, np.ndarray] = {}
    notes: list[str] = []
    if source.is_dir():
        files_by_target = _directory_files_by_target(labelmap)
        for target, names in files_by_target.items():
            mask = np.zeros(reference_shape, dtype=bool)
            matched: list[str] = []
            for name in names:
                candidate = source / name
                if not candidate.exists():
                    continue
                image = nib.load(str(candidate))
                _validate_mask_image(image, reference_shape=reference_shape, reference_spacing=reference_spacing, path=candidate)
                mask |= np.asanyarray(image.dataobj) > 0
                matched.append(str(candidate))
            if mask.any():
                masks[target] = mask
                notes.append(f"real_gi_directory_mask_loaded:{target}:{len(matched)}")
        notes.append("real_gi_source_format=directory_binary_masks")
    else:
        image = nib.load(str(source))
        _validate_mask_image(image, reference_shape=reference_shape, reference_spacing=reference_spacing, path=source)
        data = np.rint(np.asanyarray(image.dataobj)).astype(np.int32)
        ids_by_target = _label_ids_by_target(labelmap)
        for target, label_ids in ids_by_target.items():
            if not label_ids:
                continue
            mask = np.isin(data, np.asarray(sorted(label_ids), dtype=np.int32))
            if mask.any():
                masks[target] = mask
                notes.append(f"real_gi_multilabel_mask_loaded:{target}:{','.join(str(value) for value in sorted(label_ids))}")
        if not masks:
            notes.append("real_gi_multilabel_had_no_matching_configured_labels")
        notes.append("real_gi_source_format=single_multilabel_nifti")

    if masks:
        notes.append("real_gi_segmentation_replaces_matching_synthetic_placeholders")
    else:
        notes.append("real_gi_segmentation_supplied_but_no_supported_gi_labels_found")
    return RealGILoadResult(masks, str(source), None if gi_labelmap_path is None else str(gi_labelmap_path), tuple(notes))


def _property_volume(material_labels: np.ndarray, regions: tuple[Any, ...], attribute: str) -> np.ndarray:
    output = np.zeros(material_labels.shape, dtype=np.float32)
    for region in regions:
        output[material_labels == int(region.index)] = float(getattr(region, attribute))
    return output


def _centroid(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    centroid_ijk = tuple(float(value) for value in coords.mean(axis=0))
    centroid_mm = tuple(float(value) for value in (coords.mean(axis=0) * np.asarray(spacing, dtype=float)))
    return centroid_ijk, centroid_mm


def _metric_for_mask(
    definition: BTCVAbdominalOrganDefinition,
    source_mask: np.ndarray,
    material_mask: np.ndarray,
    spacing: tuple[float, float, float],
    voxel_volume_cm3: float,
) -> AbdominalOrganMetric:
    source_voxels = int(source_mask.sum())
    material_voxels = int(material_mask.sum())
    centroid = _centroid(material_mask, spacing)
    status = "pass"
    notes: list[str] = []
    if source_voxels == 0:
        status = "review"
        notes.append("source_label_absent_in_this_case")
    elif material_voxels == 0:
        status = "fail"
        notes.append("source_label_present_but_material_label_not_written")
    return AbdominalOrganMetric(
        organ_id=definition.organ_id,
        label=definition.label,
        source_label_ids=definition.source_label_ids,
        material_label_id=definition.material_label_id,
        source_voxels=source_voxels,
        material_voxels=material_voxels,
        volume_cm3=material_voxels * voxel_volume_cm3,
        centroid_ijk=None if centroid is None else centroid[0],
        centroid_mm=None if centroid is None else centroid[1],
        status=status,
        notes=tuple(notes),
    )


def _build_gi_lumen_masks(
    *,
    material_labels: np.ndarray,
    btcv_labels: np.ndarray,
    ct_hu: np.ndarray,
    ndimage,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stomach = btcv_labels == 7
    body = material_labels > 0
    internal_air = material_labels == 1
    gas = np.zeros(material_labels.shape, dtype=bool)
    fluid = np.zeros(material_labels.shape, dtype=bool)

    if stomach.any():
        stomach_eroded = ndimage.binary_erosion(
            stomach,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=1,
            border_value=0,
        )
        if int(stomach_eroded.sum()) < max(8, int(stomach.sum() * 0.02)):
            stomach_eroded = stomach & (ct_hu < -250.0)
        stomach_gas = stomach & (ct_hu < -250.0)
        gas |= stomach_gas
        fluid |= stomach_eroded & ~stomach_gas

    bowel_gas = internal_air & body & ~stomach
    if bowel_gas.any():
        gas |= bowel_gas
        shell = ndimage.binary_dilation(bowel_gas, structure=np.ones((3, 3, 3), dtype=bool), iterations=1) & ~bowel_gas
        fluid |= shell & body & np.isin(material_labels, (4, 5)) & (ct_hu > -80.0) & (ct_hu < 120.0)

    fluid &= ~gas
    return gas | fluid, gas, fluid


def _normalized_body_coordinates(body: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(body)
    if coords.size == 0:
        axes = np.indices(body.shape, dtype=float)
        denominators = np.maximum(np.asarray(body.shape, dtype=float) - 1.0, 1.0)
        return tuple(axes[index] / denominators[index] for index in range(3))  # type: ignore[return-value]
    mins = coords.min(axis=0).astype(float)
    maxs = coords.max(axis=0).astype(float)
    spans = np.maximum(maxs - mins, 1.0)
    axes = np.indices(body.shape, dtype=float)
    return tuple((axes[index] - mins[index]) / spans[index] for index in range(3))  # type: ignore[return-value]


def _ellipsoid(
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    *,
    center: tuple[float, float, float],
    radius: tuple[float, float, float],
) -> np.ndarray:
    return (
        ((nx - center[0]) / max(radius[0], 1e-6)) ** 2
        + ((ny - center[1]) / max(radius[1], 1e-6)) ** 2
        + ((nz - center[2]) / max(radius[2], 1e-6)) ** 2
    ) <= 1.0


def _colon_centerline_region(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray) -> np.ndarray:
    y_band = np.abs(ny - 0.56) <= 0.13
    ascending = (np.abs(nx - 0.25) <= 0.055) & (nz >= 0.24) & (nz <= 0.68)
    transverse = (np.abs(nz - 0.64) <= 0.055) & (nx >= 0.25) & (nx <= 0.76)
    descending = (np.abs(nx - 0.75) <= 0.055) & (nz >= 0.18) & (nz <= 0.64)
    sigmoid = _ellipsoid(nx, ny, nz, center=(0.58, 0.58, 0.22), radius=(0.18, 0.11, 0.075))
    return y_band & (ascending | transverse | descending | sigmoid)


def _candidate_or_fallback(
    candidate: np.ndarray,
    fallback: np.ndarray,
    *,
    min_voxels: int = 8,
) -> np.ndarray:
    if int(candidate.sum()) >= min_voxels:
        return candidate
    relaxed = fallback & (candidate | fallback)
    return relaxed if int(relaxed.sum()) >= min_voxels else candidate


def _wall_from_lumen(lumen: np.ndarray, available_wall: np.ndarray, ndimage, *, iterations: int = 1) -> np.ndarray:
    if not lumen.any():
        return np.zeros(lumen.shape, dtype=bool)
    dilated = ndimage.binary_dilation(
        lumen,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=iterations,
    )
    return dilated & available_wall & ~lumen


def _split_real_gi_mask(mask: np.ndarray, ct_hu: np.ndarray, ndimage) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not mask.any():
        empty = np.zeros(mask.shape, dtype=bool)
        return empty, empty, empty
    interior = ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=1,
        border_value=0,
    )
    if int(interior.sum()) < max(4, int(mask.sum() * 0.02)):
        interior = np.zeros(mask.shape, dtype=bool)
    gas = interior & (ct_hu < -250.0)
    fluid = interior & ~gas
    wall = mask & ~(gas | fluid)
    if int(wall.sum()) == 0:
        wall = mask & ~gas
        fluid &= ~wall
    return wall, gas, fluid


def _apply_real_gi_masks(
    *,
    augmented: np.ndarray,
    real_masks: dict[str, np.ndarray],
    ct_hu: np.ndarray,
    ndimage,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[str, ...]]:
    placeholder_labels = np.zeros(augmented.shape, dtype=np.int16)
    assigned_by_target: dict[str, np.ndarray] = {}
    notes: list[str] = []
    protected = np.isin(augmented, np.asarray([6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21], dtype=np.int16))
    for target in GI_TARGETS:
        source_mask = real_masks.get(target)
        if source_mask is None or not source_mask.any():
            continue
        allowed = source_mask & ~protected
        skipped = int(source_mask.sum() - allowed.sum())
        if not allowed.any():
            notes.append(f"real_gi_mask_skipped_by_protected_overlap:{target}:{skipped}")
            continue
        wall, gas, fluid = _split_real_gi_mask(allowed, ct_hu, ndimage)
        wall_label = GI_WALL_LABELS[target]
        if target == "colon":
            gas_label, fluid_label = 29, 30
        elif target == "rectum":
            gas_label, fluid_label = 29, 31
        elif target == "stomach":
            gas_label, fluid_label = 22, 23
        else:
            gas_label, fluid_label = 22, 28

        augmented[wall] = wall_label
        augmented[gas] = gas_label
        augmented[fluid] = fluid_label
        placeholder_labels[wall | gas | fluid] = wall_label
        assigned_by_target[target] = wall | gas | fluid
        notes.append(
            f"real_gi_applied:{target}:source={int(source_mask.sum())}:assigned={int((wall | gas | fluid).sum())}:protected_overlap={skipped}"
        )
    return augmented, placeholder_labels, assigned_by_target, tuple(notes)


def _build_remaining_gi_placeholder_masks(
    *,
    material_labels: np.ndarray,
    btcv_labels: np.ndarray,
    ct_hu: np.ndarray,
    ndimage,
    skip_targets: tuple[str, ...] = (),
) -> dict[int, np.ndarray]:
    body = material_labels > 0
    nx, ny, nz = _normalized_body_coordinates(body)
    real_btcv_organs = np.isin(btcv_labels, np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int16))
    protected_materials = np.isin(material_labels, np.asarray([6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21], dtype=np.int16))
    available_wall = body & ~real_btcv_organs & ~protected_materials
    available_lumen = body & ~np.isin(material_labels, np.asarray([6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21], dtype=np.int16))
    fallback_wall = body & ~protected_materials
    fallback_lumen = body & ~protected_materials

    duodenum_shape = _ellipsoid(nx, ny, nz, center=(0.45, 0.52, 0.58), radius=(0.11, 0.10, 0.12))
    small_lumen_shape = _ellipsoid(nx, ny, nz, center=(0.50, 0.56, 0.39), radius=(0.24, 0.17, 0.17))
    colon_lumen_shape = _colon_centerline_region(nx, ny, nz)
    rectum_lumen_shape = _ellipsoid(nx, ny, nz, center=(0.50, 0.69, 0.13), radius=(0.085, 0.075, 0.12))

    duodenum_wall = _candidate_or_fallback(duodenum_shape & available_wall, duodenum_shape & fallback_wall)
    small_lumen = _candidate_or_fallback(small_lumen_shape & available_lumen & ~duodenum_wall, small_lumen_shape & fallback_lumen)
    small_wall = _wall_from_lumen(small_lumen, available_wall & ~duodenum_wall, ndimage, iterations=1)
    if int(small_wall.sum()) < 8:
        small_wall = _candidate_or_fallback(
            ndimage.binary_dilation(small_lumen, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
            & fallback_wall
            & ~small_lumen
            & ~duodenum_wall,
            small_lumen_shape & fallback_wall & ~small_lumen,
        )

    colon_lumen_base = _candidate_or_fallback(
        colon_lumen_shape & available_lumen & ~small_lumen & ~duodenum_wall,
        colon_lumen_shape & fallback_lumen & ~small_lumen,
    )
    colon_gas = colon_lumen_base & ((ct_hu < -250.0) | (nz > 0.55) | (nx < 0.32))
    colon_fluid = colon_lumen_base & ~colon_gas
    if int(colon_fluid.sum()) == 0 and int(colon_lumen_base.sum()) > 0:
        coords = np.argwhere(colon_lumen_base)
        colon_fluid[tuple(coords[::2].T)] = True
        colon_gas &= ~colon_fluid
    colon_wall = _wall_from_lumen(colon_lumen_base, available_wall & ~small_wall & ~duodenum_wall, ndimage, iterations=1)
    if int(colon_wall.sum()) < 8:
        colon_wall = _candidate_or_fallback(
            ndimage.binary_dilation(colon_lumen_base, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
            & fallback_wall
            & ~colon_lumen_base
            & ~small_wall
            & ~duodenum_wall,
            colon_lumen_shape & fallback_wall & ~colon_lumen_base,
        )

    rectum_lumen = _candidate_or_fallback(
        rectum_lumen_shape & available_lumen & ~colon_lumen_base & ~small_lumen,
        rectum_lumen_shape & fallback_lumen & ~colon_lumen_base,
    )
    rectum_wall = _wall_from_lumen(rectum_lumen, available_wall & ~colon_wall & ~small_wall, ndimage, iterations=1)
    if int(rectum_wall.sum()) < 8:
        rectum_wall = _candidate_or_fallback(
            ndimage.binary_dilation(rectum_lumen, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
            & fallback_wall
            & ~rectum_lumen
            & ~colon_wall
            & ~small_wall,
            rectum_lumen_shape & fallback_wall & ~rectum_lumen,
        )

    skip = set(skip_targets)
    used = np.zeros(body.shape, dtype=bool)
    masks: dict[int, np.ndarray] = {}
    for target, label_id, mask in (
        ("duodenum", 24, duodenum_wall),
        ("small_bowel", 25, small_wall),
        ("colon", 26, colon_wall),
        ("rectum", 27, rectum_wall),
        ("small_bowel", 28, small_lumen),
        ("colon", 29, colon_gas),
        ("colon", 30, colon_fluid),
        ("rectum", 31, rectum_lumen),
    ):
        if target in skip:
            masks[label_id] = np.zeros(body.shape, dtype=bool)
            continue
        clean = mask & body & ~used
        masks[label_id] = clean
        used |= clean
    return masks


def _placeholder_metric(
    *,
    organ_id: str,
    label: str,
    material_label_id: int,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    voxel_volume_cm3: float,
    notes: tuple[str, ...],
) -> AbdominalOrganMetric:
    metric = _metric_for_mask(
        BTCVAbdominalOrganDefinition(organ_id, label, (), material_label_id),
        mask,
        mask,
        spacing,
        voxel_volume_cm3,
    )
    if metric.material_voxels == 0:
        return AbdominalOrganMetric(
            **{**metric.__dict__, "status": "review", "notes": (*notes, "placeholder_mask_empty")}
        )
    return AbdominalOrganMetric(**{**metric.__dict__, "notes": notes})


def _real_gi_metric(
    *,
    target: str,
    source_mask: np.ndarray,
    assigned_mask: np.ndarray,
    spacing: tuple[float, float, float],
    voxel_volume_cm3: float,
) -> AbdominalOrganMetric:
    label_by_target = {
        "stomach": "Real stomach segmentation",
        "duodenum": "Real duodenum segmentation",
        "small_bowel": "Real small bowel segmentation",
        "colon": "Real colon segmentation",
        "rectum": "Real rectum segmentation",
    }
    metric = _metric_for_mask(
        BTCVAbdominalOrganDefinition(f"{target}_real_segmentation", label_by_target[target], (), GI_WALL_LABELS[target]),
        source_mask,
        assigned_mask,
        spacing,
        voxel_volume_cm3,
    )
    notes = ("real_gi_segmentation_source", "solid_mask_split_to_wall_gas_fluid_by_ct_hu")
    if metric.material_voxels == 0:
        notes = (*notes, "real_gi_mask_present_but_no_voxels_assigned_after_protection")
    return AbdominalOrganMetric(**{**metric.__dict__, "notes": notes})


def _write_metrics_csv(path: Path, metrics: tuple[AbdominalOrganMetric, ...]) -> None:
    fields = [
        "organ_id",
        "label",
        "source_label_ids",
        "material_label_id",
        "source_voxels",
        "material_voxels",
        "volume_cm3",
        "centroid_ijk",
        "centroid_mm",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "organ_id": metric.organ_id,
                    "label": metric.label,
                    "source_label_ids": " ".join(str(value) for value in metric.source_label_ids),
                    "material_label_id": metric.material_label_id,
                    "source_voxels": metric.source_voxels,
                    "material_voxels": metric.material_voxels,
                    "volume_cm3": f"{metric.volume_cm3:.6f}",
                    "centroid_ijk": "" if metric.centroid_ijk is None else " ".join(f"{value:.3f}" for value in metric.centroid_ijk),
                    "centroid_mm": "" if metric.centroid_mm is None else " ".join(f"{value:.3f}" for value in metric.centroid_mm),
                    "status": metric.status,
                    "notes": ";".join(metric.notes),
                }
            )


def _write_preview(
    path: Path,
    ct_hu: np.ndarray,
    labels: np.ndarray,
    organ_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> None:
    plt, ListedColormap, Patch, *_ = _import_dependencies()
    coords = np.argwhere(organ_mask)
    if coords.size:
        x_index, y_index, z_index = (int(round(float(np.median(coords[:, axis])))) for axis in range(3))
    else:
        x_index, y_index, z_index = (value // 2 for value in labels.shape)
    organ_ids = [spec[0] for spec in ABDOMINAL_REGION_SPECS]
    colors = ["#000000"] * (max(organ_ids) + 1)
    for index, _, _, color in ABDOMINAL_REGION_SPECS:
        colors[index] = color
    cmap = ListedColormap(colors)
    views = [
        (ct_hu[:, :, z_index], labels[:, :, z_index], f"Axial z={z_index}"),
        (ct_hu[:, y_index, :], labels[:, y_index, :], f"Coronal y={y_index}"),
        (ct_hu[x_index, :, :], labels[x_index, :, :], f"Sagittal x={x_index}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=170)
    for ax, (ct_view, label_view, title) in zip(axes, views):
        overlay = np.ma.masked_where(~np.isin(label_view, organ_ids), label_view)
        ax.imshow(np.rot90(np.clip(ct_view, -1000, 1000)), cmap="gray", vmin=-1000, vmax=1000)
        ax.imshow(np.rot90(overlay), cmap=cmap, vmin=0, vmax=max(organ_ids), alpha=0.72, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    handles = [Patch(facecolor=color, label=f"{index}: {name}") for index, name, _, color in ABDOMINAL_REGION_SPECS]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.5), fontsize=6)
    fig.suptitle("BTCV Organ-Preserving Abdomen + GI Placeholder", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, result: AbdominalOrganAugmentationResult) -> None:
    lines = [
        "# Organ-Preserving Abdominal Module",
        "",
        f"Case ID: `{result.case_id}`",
        f"Pass / review / fail: {result.pass_count} / {result.review_count} / {result.fail_count}",
        "",
        "## Preserved BTCV Organs And GI Placeholders",
        "",
        "| organ | source labels | material label | volume cm3 | centroid mm | status |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for metric in result.metrics:
        centroid = "n/a" if metric.centroid_mm is None else ", ".join(f"{value:.1f}" for value in metric.centroid_mm)
        source = ", ".join(str(value) for value in metric.source_label_ids)
        lines.append(
            f"| {metric.label} | {source} | {metric.material_label_id} | {metric.volume_cm3:.2f} | {centroid} | `{metric.status}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- QA YAML: `{result.qa_yaml_path}`",
            f"- Metrics CSV: `{result.organ_metrics_csv_path}`",
            f"- Preview PNG: `{result.preview_png_path}`",
            f"- GI lumen mask: `{result.gi_lumen_mask_path}`",
            f"- GI tract placeholder labels: `{result.gi_tract_placeholder_labels_path}`",
            f"- Real GI segmentation source: `{result.gi_real_segmentation_source_path or 'not_supplied'}`",
            f"- Real GI applied targets: `{', '.join(result.gi_real_segmentation_applied_targets) or 'none'}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_qa_yaml(path: Path, result: AbdominalOrganAugmentationResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "btcv_organ_preserving_abdominal_module",
        "summary": {
            "pass_count": result.pass_count,
            "review_count": result.review_count,
            "fail_count": result.fail_count,
            "organ_count": len(result.metrics),
        },
        "outputs": {
            "material_label_map": result.material_label_path,
            "gi_lumen_mask": result.gi_lumen_mask_path,
            "gi_gas_mask": result.gi_gas_mask_path,
            "gi_fluid_mask": result.gi_fluid_mask_path,
            "gi_tract_placeholder_labels": result.gi_tract_placeholder_labels_path,
            "real_gi_segmentation_source": result.gi_real_segmentation_source_path,
            "real_gi_segmentation_labelmap": result.gi_real_segmentation_labelmap_path,
            "organ_metrics_csv": result.organ_metrics_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "real_gi_segmentation": {
            "source_path": result.gi_real_segmentation_source_path,
            "labelmap_path": result.gi_real_segmentation_labelmap_path,
            "applied_targets": list(result.gi_real_segmentation_applied_targets),
            "fallback_placeholder_targets": [
                target for target in ("duodenum", "small_bowel", "colon", "rectum") if target not in result.gi_real_segmentation_applied_targets
            ],
        },
        "metrics": [
            {
                "organ_id": metric.organ_id,
                "label": metric.label,
                "source_label_ids": list(metric.source_label_ids),
                "material_label_id": metric.material_label_id,
                "source_voxels": metric.source_voxels,
                "material_voxels": metric.material_voxels,
                "volume_cm3": metric.volume_cm3,
                "centroid_ijk": None if metric.centroid_ijk is None else list(metric.centroid_ijk),
                "centroid_mm": None if metric.centroid_mm is None else list(metric.centroid_mm),
                "status": metric.status,
                "notes": list(metric.notes),
            }
            for metric in result.metrics
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _update_torso_spec(
    spec_path: Path,
    *,
    regions: tuple[Any, ...],
    labels: np.ndarray,
    ct_hu: np.ndarray,
    voxel_volume_cm3: float,
    result: AbdominalOrganAugmentationResult,
) -> None:
    spec = yaml.safe_load(spec_path.read_text()) if spec_path.exists() else {}
    if not isinstance(spec, dict):
        spec = {}
    outputs = spec.setdefault("outputs", {})
    if isinstance(outputs, dict):
        outputs.update(
            {
                "abdominal_organ_qa_yaml": result.qa_yaml_path,
                "abdominal_organ_metrics_csv": result.organ_metrics_csv_path,
                "abdominal_organ_preview_png": result.preview_png_path,
                "gi_lumen_mask": result.gi_lumen_mask_path,
                "gi_gas_lumen_mask": result.gi_gas_mask_path,
                "gi_fluid_lumen_mask": result.gi_fluid_mask_path,
                "gi_tract_placeholder_labels": result.gi_tract_placeholder_labels_path,
                "real_gi_segmentation_source": result.gi_real_segmentation_source_path,
                "real_gi_segmentation_labelmap": result.gi_real_segmentation_labelmap_path,
                "real_gi_segmentation_applied_targets": list(result.gi_real_segmentation_applied_targets),
            }
        )
    spec["regions"] = [
        {
            "index": int(region.index),
            "name": str(region.name),
            "material_id": str(region.material_id),
            "target_hu_midpoint": float(region.target_hu_midpoint),
            "mass_density_g_cm3": float(region.mass_density_g_cm3),
            "relative_electron_density": float(region.relative_electron_density),
            "color": str(region.color),
        }
        for region in regions
    ]
    stats = []
    for region in regions:
        mask = labels == int(region.index)
        count = int(mask.sum())
        if count == 0:
            continue
        stats.append(
            {
                "index": int(region.index),
                "name": str(region.name),
                "material_id": str(region.material_id),
                "voxel_count": count,
                "volume_cm3": count * voxel_volume_cm3,
                "mean_source_hu": float(np.mean(ct_hu[mask])),
                "target_hu_midpoint": float(region.target_hu_midpoint),
                "mass_density_g_cm3": float(region.mass_density_g_cm3),
                "relative_electron_density": float(region.relative_electron_density),
            }
        )
    spec["stats"] = stats
    notes = spec.setdefault("notes", [])
    if isinstance(notes, list):
        notes.extend(
            [
                "btcv_abdominal_organs_preserved_as_explicit_material_labels",
                "gi_lumen_gas_fluid_placeholder_added_from_stomach_and_internal_air_cavities",
                "remaining_gi_placeholders_added_for_duodenum_small_bowel_colon_and_rectum",
                "real_gi_segmentations_replace_matching_placeholders_when_supplied",
            ]
        )
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))


def augment_btcv_abdominal_organs(
    *,
    case_id: str,
    ct_path: str | Path,
    btcv_labels_path: str | Path,
    gi_segmentation_path: str | Path | None = None,
    gi_labelmap_path: str | Path | None = None,
    material_label_path: str | Path,
    mass_density_path: str | Path,
    relative_electron_density_path: str | Path,
    synthetic_hu_path: str | Path,
    torso_spec_path: str | Path,
    regions: tuple[Any, ...],
    output_dir: str | Path,
    report_path: str | Path,
) -> AbdominalOrganAugmentationResult:
    _, _, _, nib, ndimage = _import_dependencies()
    ct_image = nib.load(str(ct_path))
    btcv_image = nib.load(str(btcv_labels_path))
    material_image = nib.load(str(material_label_path))
    ct_hu = np.asanyarray(ct_image.dataobj).astype(np.float32)
    btcv_labels = np.rint(np.asanyarray(btcv_image.dataobj)).astype(np.int16)
    material_labels = np.rint(np.asanyarray(material_image.dataobj)).astype(np.int16)
    if ct_hu.shape != material_labels.shape or btcv_labels.shape != material_labels.shape:
        raise ValueError(
            f"CT, BTCV labels, and material labels must share shape: {ct_hu.shape}, {btcv_labels.shape}, {material_labels.shape}"
        )

    spacing = tuple(float(value) for value in material_image.header.get_zooms()[:3])
    reference_shape = tuple(int(value) for value in material_labels.shape)
    voxel_volume_cm3 = float(np.prod(np.asarray(spacing, dtype=float)) / 1000.0)
    augmented = material_labels.copy()

    for definition in BTCV_ORGAN_DEFINITIONS:
        if definition.organ_id == "stomach_wall":
            continue
        source = np.isin(btcv_labels, np.asarray(definition.source_label_ids, dtype=np.int16))
        augmented[source] = definition.material_label_id

    stomach = btcv_labels == 7
    gi_lumen, gi_gas, gi_fluid = _build_gi_lumen_masks(
        material_labels=material_labels,
        btcv_labels=btcv_labels,
        ct_hu=ct_hu,
        ndimage=ndimage,
    )
    stomach_lumen = stomach & gi_lumen
    augmented[stomach & ~stomach_lumen] = 17
    augmented[gi_gas] = 22
    augmented[gi_fluid] = 23
    real_gi = _load_real_gi_segmentation(
        gi_segmentation_path=gi_segmentation_path,
        gi_labelmap_path=gi_labelmap_path,
        reference_shape=reference_shape,
        reference_spacing=spacing,
        nib=nib,
    )
    augmented, real_gi_labels, real_gi_assigned, real_gi_apply_notes = _apply_real_gi_masks(
        augmented=augmented,
        real_masks=real_gi.masks,
        ct_hu=ct_hu,
        ndimage=ndimage,
    )
    gi_placeholder_masks = _build_remaining_gi_placeholder_masks(
        material_labels=augmented,
        btcv_labels=btcv_labels,
        ct_hu=ct_hu,
        ndimage=ndimage,
        skip_targets=tuple(real_gi_assigned),
    )
    gi_placeholder_labels = np.zeros(augmented.shape, dtype=np.int16)
    for label_id, mask in gi_placeholder_masks.items():
        augmented[mask] = label_id
        gi_placeholder_labels[mask] = label_id
    gi_placeholder_labels[real_gi_labels > 0] = real_gi_labels[real_gi_labels > 0]

    output = Path(output_dir)
    gi_lumen_path = output / f"{case_id}_gi_lumen_mask_v001.nii.gz"
    gi_gas_path = output / f"{case_id}_gi_gas_lumen_mask_v001.nii.gz"
    gi_fluid_path = output / f"{case_id}_gi_fluid_lumen_mask_v001.nii.gz"
    gi_placeholder_path = output / f"{case_id}_gi_tract_placeholder_labels_v001.nii.gz"
    metrics_csv = output / f"{case_id}_abdominal_organ_metrics_v001.csv"
    qa_yaml = output / f"{case_id}_abdominal_organ_qa_v001.yaml"
    preview = output / f"{case_id}_abdominal_organ_preview_v001.png"
    report = Path(report_path)

    _write_nifti(Path(material_label_path), augmented.astype(np.int16), material_image, nib)
    _write_nifti(Path(mass_density_path), _property_volume(augmented, regions, "mass_density_g_cm3"), material_image, nib)
    _write_nifti(
        Path(relative_electron_density_path),
        _property_volume(augmented, regions, "relative_electron_density"),
        material_image,
        nib,
    )
    _write_nifti(Path(synthetic_hu_path), _property_volume(augmented, regions, "target_hu_midpoint"), material_image, nib)
    _write_nifti(gi_lumen_path, gi_lumen.astype(np.uint8), material_image, nib)
    _write_nifti(gi_gas_path, gi_gas.astype(np.uint8), material_image, nib)
    _write_nifti(gi_fluid_path, gi_fluid.astype(np.uint8), material_image, nib)
    _write_nifti(gi_placeholder_path, gi_placeholder_labels.astype(np.int16), material_image, nib)

    metrics = tuple(
        _metric_for_mask(
            definition,
            np.isin(btcv_labels, np.asarray(definition.source_label_ids, dtype=np.int16)),
            augmented == definition.material_label_id,
            spacing,
            voxel_volume_cm3,
        )
        for definition in BTCV_ORGAN_DEFINITIONS
    )
    gi_definition = BTCVAbdominalOrganDefinition("gi_lumen_placeholder", "GI lumen gas/fluid placeholder", (), 22)
    gi_metric = _metric_for_mask(
        gi_definition,
        gi_lumen,
        gi_lumen,
        spacing,
        voxel_volume_cm3,
    )
    if gi_metric.material_voxels == 0:
        gi_metric = AbdominalOrganMetric(
            **{**gi_metric.__dict__, "status": "review", "notes": ("gi_lumen_placeholder_empty",)}
        )
    real_gi_metrics = tuple(
        _real_gi_metric(
            target=target,
            source_mask=real_gi.masks[target],
            assigned_mask=real_gi_assigned.get(target, np.zeros(augmented.shape, dtype=bool)),
            spacing=spacing,
            voxel_volume_cm3=voxel_volume_cm3,
        )
        for target in GI_TARGETS
        if target in real_gi.masks
    )
    placeholder_notes = (
        "synthetic_gi_placeholder_not_btcv_ground_truth",
        "replace_with_bowel_colon_small_intestine_segmentation_when_available",
    )
    placeholder_metric_items = []
    if "duodenum" not in real_gi_assigned:
        placeholder_metric_items.append(_placeholder_metric(
            organ_id="duodenum_wall_placeholder",
            label="Duodenum wall placeholder",
            material_label_id=24,
            mask=augmented == 24,
            spacing=spacing,
            voxel_volume_cm3=voxel_volume_cm3,
            notes=placeholder_notes,
        ))
    if "small_bowel" not in real_gi_assigned:
        placeholder_metric_items.append(_placeholder_metric(
            organ_id="small_bowel_wall_placeholder",
            label="Small bowel wall placeholder",
            material_label_id=25,
            mask=augmented == 25,
            spacing=spacing,
            voxel_volume_cm3=voxel_volume_cm3,
            notes=placeholder_notes,
        ))
    if "colon" not in real_gi_assigned:
        placeholder_metric_items.append(_placeholder_metric(
            organ_id="colon_wall_placeholder",
            label="Colon wall placeholder",
            material_label_id=26,
            mask=augmented == 26,
            spacing=spacing,
            voxel_volume_cm3=voxel_volume_cm3,
            notes=placeholder_notes,
        ))
    if "rectum" not in real_gi_assigned:
        placeholder_metric_items.append(_placeholder_metric(
            organ_id="rectum_wall_placeholder",
            label="Rectum wall placeholder",
            material_label_id=27,
            mask=augmented == 27,
            spacing=spacing,
            voxel_volume_cm3=voxel_volume_cm3,
            notes=placeholder_notes,
        ))
    if "small_bowel" not in real_gi_assigned:
        placeholder_metric_items.append(_placeholder_metric(
            organ_id="small_bowel_lumen_placeholder",
            label="Small bowel lumen placeholder",
            material_label_id=28,
            mask=augmented == 28,
            spacing=spacing,
            voxel_volume_cm3=voxel_volume_cm3,
            notes=placeholder_notes,
        ))
    if "colon" not in real_gi_assigned:
        placeholder_metric_items.append(_placeholder_metric(
            organ_id="colon_lumen_placeholder",
            label="Colon lumen gas/fluid placeholder",
            material_label_id=29,
            mask=np.isin(augmented, (29, 30)),
            spacing=spacing,
            voxel_volume_cm3=voxel_volume_cm3,
            notes=placeholder_notes,
        ))
    if "rectum" not in real_gi_assigned:
        placeholder_metric_items.append(_placeholder_metric(
            organ_id="rectum_lumen_placeholder",
            label="Rectum lumen placeholder",
            material_label_id=31,
            mask=augmented == 31,
            spacing=spacing,
            voxel_volume_cm3=voxel_volume_cm3,
            notes=placeholder_notes,
        ))
    placeholder_metrics = tuple(placeholder_metric_items)
    metrics = (*metrics, gi_metric, *real_gi_metrics, *placeholder_metrics)
    pass_count = sum(1 for metric in metrics if metric.status == "pass")
    review_count = sum(1 for metric in metrics if metric.status == "review")
    fail_count = sum(1 for metric in metrics if metric.status == "fail")
    notes = (
        "btcv_labels_preserved_for_spleen_stomach_gallbladder_esophagus_pancreas_and_adrenals",
        "stomach_lumen_split_uses_ct_air_threshold_and_eroded_stomach_interior",
        "bowel_gas_fluid_lumen_is_placeholder_from_internal_air_cavities_until_bowel_specific_segmentations_are_staged",
        "remaining_gi_placeholder_labels_added_for_duodenum_small_bowel_colon_and_rectum",
        "real_gi_segmentations_replace_matching_placeholders_when_supplied",
        *real_gi.notes,
        *real_gi_apply_notes,
    )
    result = AbdominalOrganAugmentationResult(
        case_id=case_id,
        material_label_path=str(material_label_path),
        mass_density_path=str(mass_density_path),
        relative_electron_density_path=str(relative_electron_density_path),
        synthetic_hu_path=str(synthetic_hu_path),
        gi_lumen_mask_path=str(gi_lumen_path),
        gi_gas_mask_path=str(gi_gas_path),
        gi_fluid_mask_path=str(gi_fluid_path),
        gi_tract_placeholder_labels_path=str(gi_placeholder_path),
        gi_real_segmentation_source_path=real_gi.source_path,
        gi_real_segmentation_labelmap_path=real_gi.labelmap_path,
        gi_real_segmentation_applied_targets=tuple(sorted(real_gi_assigned)),
        organ_metrics_csv_path=str(metrics_csv),
        qa_yaml_path=str(qa_yaml),
        preview_png_path=str(preview),
        report_path=str(report),
        pass_count=pass_count,
        review_count=review_count,
        fail_count=fail_count,
        metrics=metrics,
        notes=notes,
    )
    _write_metrics_csv(metrics_csv, metrics)
    _write_preview(preview, ct_hu, augmented, np.isin(augmented, [spec[0] for spec in ABDOMINAL_REGION_SPECS]), spacing)
    _write_qa_yaml(qa_yaml, result)
    _write_report(report, result)
    _update_torso_spec(
        Path(torso_spec_path),
        regions=regions,
        labels=augmented,
        ct_hu=ct_hu,
        voxel_volume_cm3=voxel_volume_cm3,
        result=result,
    )
    return result
