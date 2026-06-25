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
            "Anthropometric torso morphing requires matplotlib, nibabel, scipy, and PyYAML."
        ) from exc
    return plt, ListedColormap, Patch, nib, ndimage, yaml


@dataclass(frozen=True)
class MorphRegionStats:
    index: int
    name: str
    baseline_voxels: int
    morphed_voxels: int
    baseline_volume_cm3: float
    morphed_volume_cm3: float
    volume_change_percent: float


@dataclass(frozen=True)
class AnthropometricMorphResult:
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
    morphed_flow_boundary_labels_path: str | None
    scale_profile_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    morph_mode: str
    xy_padding_voxels: int
    output_shape_voxels: tuple[int, int, int]
    target_height_cm: float
    target_weight_kg: float | None
    target_bmi: float
    target_waist_cm: float
    baseline_height_cm: float
    baseline_bmi: float
    baseline_waist_cm: float
    achieved_waist_cm: float
    target_body_radial_scale: float
    target_height_scale: float
    baseline_body_volume_cm3: float
    morphed_body_volume_cm3: float
    body_volume_change_percent: float
    baseline_bbox_mm: tuple[float, float, float]
    morphed_bbox_mm: tuple[float, float, float]
    vascular_components: int
    region_stats: tuple[MorphRegionStats, ...]
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


def _pad_xy(data: np.ndarray, xy_padding_voxels: int) -> np.ndarray:
    if xy_padding_voxels <= 0:
        return data
    pad = int(xy_padding_voxels)
    return np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)


def _reference_with_shape(reference_image, shape: tuple[int, int, int], nib):
    if tuple(int(value) for value in reference_image.shape[:3]) == tuple(shape):
        return reference_image
    header = reference_image.header.copy()
    header.set_data_shape(shape)
    return nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), reference_image.affine, header)


def _voxel_volume_cm3(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(spacing_mm) / 1000.0)


def _bbox_mm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0.0, 0.0, 0.0)
    extent_voxels = coords.max(axis=0) - coords.min(axis=0) + 1
    return tuple(float(value) for value in extent_voxels * np.array(spacing_mm, dtype=float))


def _body_center_ijk(mask: np.ndarray) -> tuple[float, float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(float((value - 1) / 2.0) for value in mask.shape)
    return tuple(float(value) for value in np.median(coords, axis=0))


def _waist_slice(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask.shape[2] // 2
    min_z = int(coords[:, 2].min())
    max_z = int(coords[:, 2].max())
    return int(round(min_z + 0.46 * (max_z - min_z)))


def _ellipse_circumference_cm(width_mm: float, depth_mm: float) -> float:
    a = max(width_mm / 2.0, 1e-6)
    b = max(depth_mm / 2.0, 1e-6)
    circumference_mm = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
    return float(circumference_mm / 10.0)


def _estimate_waist_cm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    z_index = _waist_slice(mask)
    slice_mask = mask[:, :, z_index]
    coords = np.argwhere(slice_mask)
    if coords.size == 0:
        bbox = _bbox_mm(mask, spacing_mm)
        return _ellipse_circumference_cm(bbox[0], bbox[1])
    width_mm = float((coords[:, 0].max() - coords[:, 0].min() + 1) * spacing_mm[0])
    depth_mm = float((coords[:, 1].max() - coords[:, 1].min() + 1) * spacing_mm[1])
    return _ellipse_circumference_cm(width_mm, depth_mm)


def _derive_target_bmi(target_height_cm: float, target_weight_kg: float | None, target_bmi: float | None) -> float:
    if target_bmi is not None:
        return float(target_bmi)
    if target_weight_kg is None:
        raise ValueError("Provide either target_bmi or target_weight_kg")
    height_m = target_height_cm / 100.0
    if height_m <= 0.0:
        raise ValueError("target_height_cm must be positive")
    return float(target_weight_kg / (height_m**2))


def _derive_target_waist_cm(
    target_waist_cm: float | None,
    baseline_waist_cm: float,
    target_bmi: float,
    baseline_bmi: float,
) -> float:
    if target_waist_cm is not None:
        return float(target_waist_cm)
    if baseline_bmi <= 0.0:
        raise ValueError("baseline_bmi must be positive")
    return float(baseline_waist_cm * math.sqrt(max(target_bmi, 1e-6) / baseline_bmi))


def _z_bounds(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, mask.shape[2] - 1
    return int(coords[:, 2].min()), int(coords[:, 2].max())


def _profile_weight(z_index: int, z_min: int, z_max: int, center_fraction: float, width_fraction: float) -> float:
    if z_max <= z_min:
        return 1.0
    z_norm = (z_index - z_min) / float(z_max - z_min)
    width = max(width_fraction, 1e-3)
    abdomen = math.exp(-((z_norm - center_fraction) ** 2) / (2.0 * width**2))
    return float(0.35 + 0.65 * abdomen)


def _scale_at_slice(
    z_index: int,
    z_min: int,
    z_max: int,
    radial_scale: float,
    center_fraction: float,
    width_fraction: float,
) -> float:
    return float(1.0 + (radial_scale - 1.0) * _profile_weight(z_index, z_min, z_max, center_fraction, width_fraction))


def _morph_label_volume(
    labels: np.ndarray,
    body_mask: np.ndarray,
    radial_scale: float,
    height_scale: float,
    center_ijk: tuple[float, float, float],
    abdomen_center_fraction: float,
    abdomen_width_fraction: float,
    ndimage,
) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=labels.dtype)
    z_min, z_max = _z_bounds(body_mask)
    center_xy = np.array(center_ijk[:2], dtype=float)
    center_z = float(center_ijk[2])
    safe_height_scale = max(height_scale, 1e-6)

    for out_z in range(labels.shape[2]):
        source_z = int(round(center_z + (out_z - center_z) / safe_height_scale))
        if source_z < 0 or source_z >= labels.shape[2]:
            continue
        slice_scale = _scale_at_slice(
            out_z,
            z_min,
            z_max,
            radial_scale=radial_scale,
            center_fraction=abdomen_center_fraction,
            width_fraction=abdomen_width_fraction,
        )
        matrix = np.diag([1.0 / max(slice_scale, 1e-6), 1.0 / max(slice_scale, 1e-6)])
        offset = center_xy - matrix @ center_xy
        output[:, :, out_z] = ndimage.affine_transform(
            labels[:, :, source_z],
            matrix=matrix,
            offset=offset,
            output_shape=labels.shape[:2],
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(labels.dtype)
    return output


def _morph_binary_mask(
    mask: np.ndarray,
    body_mask: np.ndarray,
    radial_scale: float,
    height_scale: float,
    center_ijk: tuple[float, float, float],
    abdomen_center_fraction: float,
    abdomen_width_fraction: float,
    ndimage,
) -> np.ndarray:
    morphed = _morph_label_volume(
        mask.astype(np.uint8),
        body_mask=body_mask,
        radial_scale=radial_scale,
        height_scale=height_scale,
        center_ijk=center_ijk,
        abdomen_center_fraction=abdomen_center_fraction,
        abdomen_width_fraction=abdomen_width_fraction,
        ndimage=ndimage,
    )
    return morphed > 0


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


def _property_volume(material_labels: np.ndarray, regions: list[dict[str, Any]], key: str) -> np.ndarray:
    output = np.zeros(material_labels.shape, dtype=np.float32)
    for region in regions:
        output[material_labels == int(region["index"])] = float(region[key])
    return output


def _region_name_by_index(regions: list[dict[str, Any]]) -> dict[int, str]:
    return {int(region["index"]): str(region["name"]) for region in regions}


def _region_stats(
    baseline_labels: np.ndarray,
    morphed_labels: np.ndarray,
    regions: list[dict[str, Any]],
    voxel_volume_cm3: float,
) -> tuple[MorphRegionStats, ...]:
    names = _region_name_by_index(regions)
    stats: list[MorphRegionStats] = []
    for index in sorted(set(int(value) for value in np.unique(baseline_labels)) | set(int(value) for value in np.unique(morphed_labels))):
        if index == 0:
            continue
        baseline_voxels = int((baseline_labels == index).sum())
        morphed_voxels = int((morphed_labels == index).sum())
        baseline_volume = baseline_voxels * voxel_volume_cm3
        morphed_volume = morphed_voxels * voxel_volume_cm3
        if baseline_volume > 0.0:
            change = 100.0 * (morphed_volume - baseline_volume) / baseline_volume
        else:
            change = 0.0
        stats.append(
            MorphRegionStats(
                index=index,
                name=names.get(index, f"label_{index}"),
                baseline_voxels=baseline_voxels,
                morphed_voxels=morphed_voxels,
                baseline_volume_cm3=float(baseline_volume),
                morphed_volume_cm3=float(morphed_volume),
                volume_change_percent=float(change),
            )
        )
    return tuple(stats)


def _write_scale_profile(
    path: Path,
    shape_z: int,
    body_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    body_radial_scale: float,
    organ_radial_scale: float,
    bone_radial_scale: float,
    vascular_radial_scale: float,
    abdomen_center_fraction: float,
    abdomen_width_fraction: float,
) -> None:
    z_min, z_max = _z_bounds(body_mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "z_index",
                "z_mm",
                "abdomen_profile_weight",
                "body_radial_scale",
                "organ_radial_scale",
                "bone_radial_scale",
                "vascular_radial_scale",
            ]
        )
        for z_index in range(shape_z):
            weight = _profile_weight(z_index, z_min, z_max, abdomen_center_fraction, abdomen_width_fraction)
            writer.writerow(
                [
                    z_index,
                    f"{z_index * spacing_mm[2]:.6f}",
                    f"{weight:.6f}",
                    f"{1.0 + (body_radial_scale - 1.0) * weight:.6f}",
                    f"{1.0 + (organ_radial_scale - 1.0) * weight:.6f}",
                    f"{1.0 + (bone_radial_scale - 1.0) * weight:.6f}",
                    f"{1.0 + (vascular_radial_scale - 1.0) * weight:.6f}",
                ]
            )


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(value // 2 for value in mask.shape)
    return tuple(int(round(float(np.median(coords[:, axis])))) for axis in range(3))


def _render_preview(
    path: Path,
    baseline_labels: np.ndarray,
    morphed_labels: np.ndarray,
    baseline_body: np.ndarray,
    morphed_body: np.ndarray,
    regions: list[dict[str, Any]],
    spacing_mm: tuple[float, float, float],
    result_summary: dict[str, float],
) -> None:
    plt, ListedColormap, Patch, *_ = _import_dependencies()
    colors = [str(region.get("color", "#000000")) for region in sorted(regions, key=lambda item: int(item["index"]))]
    cmap = ListedColormap(colors)
    vmax = max(int(region["index"]) for region in regions)
    _, _, z_index = _slice_indices(baseline_body | morphed_body)
    x_index, y_index, _ = _slice_indices(morphed_body)
    views = [
        ("Axial", baseline_labels[:, :, z_index], morphed_labels[:, :, z_index]),
        ("Coronal", baseline_labels[:, y_index, :], morphed_labels[:, y_index, :]),
        ("Sagittal", baseline_labels[x_index, :, :], morphed_labels[x_index, :, :]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    for ax in axes.ravel():
        ax.set_facecolor("#f6f1e8")
        ax.axis("off")
    for col, (title, baseline, morphed) in enumerate(views):
        axes[0, col].imshow(np.rot90(baseline), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[0, col].set_title(f"Baseline {title}", fontsize=10, color="#13202a")
        axes[1, col].imshow(np.rot90(morphed), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[1, col].set_title(f"Morphed {title}", fontsize=10, color="#13202a")
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
    fig.suptitle(
        "Anthropometric Torso Morph "
        f"(BMI {result_summary['target_bmi']:.1f}, waist {result_summary['target_waist_cm']:.1f} cm, "
        f"height {result_summary['target_height_cm']:.1f} cm, {result_summary.get('morph_mode', 'standard')})",
        fontsize=15,
        color="#13202a",
    )
    fig.text(
        0.08,
        0.02,
        "Waist proxy: "
        f"{result_summary['baseline_waist_cm']:.1f} -> {result_summary['achieved_waist_cm']:.1f} cm; "
        "body volume: "
        f"{result_summary['baseline_body_volume_cm3'] / 1000.0:.2f} -> "
        f"{result_summary['morphed_body_volume_cm3'] / 1000.0:.2f} L; "
        f"padding {result_summary.get('xy_padding_voxels', 0)} vox/side; "
        f"spacing {spacing_mm[0]:.3g} x {spacing_mm[1]:.3g} x {spacing_mm[2]:.3g} mm",
        fontsize=9,
        color="#1e2a32",
    )
    fig.tight_layout(rect=(0, 0.04, 0.90, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_spec(path: Path, source_spec_path: Path, result: AnthropometricMorphResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "package_type": "anthropometric_torso_morph",
        "source_combined_spec": str(source_spec_path),
        "anthropometry": {
            "morph_mode": result.morph_mode,
            "xy_padding_voxels": result.xy_padding_voxels,
            "target_height_cm": result.target_height_cm,
            "target_weight_kg": result.target_weight_kg,
            "target_bmi": result.target_bmi,
            "target_waist_cm": result.target_waist_cm,
            "baseline_height_cm": result.baseline_height_cm,
            "baseline_bmi": result.baseline_bmi,
            "baseline_waist_cm": result.baseline_waist_cm,
            "achieved_waist_cm": result.achieved_waist_cm,
            "body_radial_scale": result.target_body_radial_scale,
            "height_scale": result.target_height_scale,
            "output_shape_voxels": list(result.output_shape_voxels),
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
            "flow_boundary_labels": result.morphed_flow_boundary_labels_path,
            "scale_profile_csv": result.scale_profile_csv_path,
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
        "region_stats": [
            {
                "index": stat.index,
                "name": stat.name,
                "baseline_volume_cm3": stat.baseline_volume_cm3,
                "morphed_volume_cm3": stat.morphed_volume_cm3,
                "volume_change_percent": stat.volume_change_percent,
            }
            for stat in result.region_stats
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: AnthropometricMorphResult) -> str:
    lines = [
        "# Anthropometric Torso Morph Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Target Anthropometry",
        "",
        f"- Morph mode: {result.morph_mode}",
        f"- XY grid padding: {result.xy_padding_voxels} voxels per side",
        f"- Target height: {result.target_height_cm:.1f} cm",
        f"- Target BMI: {result.target_bmi:.2f} kg/m2",
        f"- Target weight: {'not supplied' if result.target_weight_kg is None else f'{result.target_weight_kg:.1f} kg'}",
        f"- Target waist: {result.target_waist_cm:.1f} cm",
        f"- Baseline height assumption: {result.baseline_height_cm:.1f} cm",
        f"- Baseline BMI assumption: {result.baseline_bmi:.2f} kg/m2",
        f"- Baseline waist proxy: {result.baseline_waist_cm:.1f} cm",
        f"- Achieved waist proxy: {result.achieved_waist_cm:.1f} cm",
        "",
        "## Morph Scales",
        "",
        f"- Body radial scale at abdominal peak: {result.target_body_radial_scale:.3f}",
        f"- Height scale: {result.target_height_scale:.3f}",
        "",
        "## Volume QA",
        "",
        f"- Body volume: {result.baseline_body_volume_cm3 / 1000.0:.3f} L -> {result.morphed_body_volume_cm3 / 1000.0:.3f} L",
        f"- Body volume change: {result.body_volume_change_percent:.2f}%",
        f"- Baseline bbox mm: {', '.join(f'{value:.1f}' for value in result.baseline_bbox_mm)}",
        f"- Morphed bbox mm: {', '.join(f'{value:.1f}' for value in result.morphed_bbox_mm)}",
        f"- Vascular connected components after morph: {result.vascular_components}",
        f"- Output shape voxels: {', '.join(str(value) for value in result.output_shape_voxels)}",
        "",
        "## Outputs",
        "",
        f"- Blood material labels: `{Path(result.morphed_blood_material_labels_path).name}`",
        f"- Blood HU/density/RED: `{Path(result.morphed_blood_synthetic_hu_path).name}`, `{Path(result.morphed_blood_density_path).name}`, `{Path(result.morphed_blood_relative_electron_density_path).name}`",
        f"- Contrast material labels: `{Path(result.morphed_contrast_material_labels_path).name}`",
        f"- Body mask: `{Path(result.morphed_body_mask_path).name}`",
        f"- Vascular fluid mask: `{Path(result.morphed_vascular_fluid_mask_path).name}`",
        f"- Vessel wall mask: `{Path(result.morphed_vessel_wall_mask_path).name}`",
        f"- Flow boundary labels: `{Path(result.morphed_flow_boundary_labels_path).name if result.morphed_flow_boundary_labels_path else 'not available'}`",
        f"- Scale profile CSV: `{Path(result.scale_profile_csv_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Region Volume Changes",
        "",
        "| label | region | baseline cm3 | morphed cm3 | change % |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for stat in result.region_stats:
        lines.append(
            f"| {stat.index} | {stat.name} | {stat.baseline_volume_cm3:.2f} | "
            f"{stat.morphed_volume_cm3:.2f} | {stat.volume_change_percent:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an anthropometrically parameterized engineering morph, not a subject-specific anatomical registration.",
            "- Waist circumference controls transverse body-envelope deformation; BMI is used when waist is not supplied.",
            "- Bone and organ structures are intentionally deformed less than the soft-tissue envelope.",
            "- Downstream mesh, RT, and flow packages should be regenerated from this morphed volume before using it for comparisons.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_anthropometric_torso_morph(
    combined_spec_path: str | Path,
    output_dir: str | Path = "outputs/digital/anthropometric_morph",
    case_id: str = "ct_org_case0_imagetbad_case125_bmi32_waist110",
    target_height_cm: float = 175.0,
    target_weight_kg: float | None = None,
    target_bmi: float | None = 32.0,
    target_waist_cm: float | None = 110.0,
    baseline_height_cm: float = 170.0,
    baseline_bmi: float = 24.0,
    baseline_waist_cm: float | None = None,
    abdomen_center_fraction: float = 0.46,
    abdomen_width_fraction: float = 0.24,
    morph_mode: str = "standard",
    xy_padding_voxels: int = 0,
    report_path: str | Path | None = "outputs/reports/anthropometric_torso_morph_stage001.md",
) -> AnthropometricMorphResult:
    _, _, _, nib, ndimage, _ = _import_dependencies()
    spec_path = Path(combined_spec_path)
    spec = _load_yaml(spec_path)
    outputs = spec.get("outputs", {})
    regions = list(spec.get("regions", []))
    if not isinstance(outputs, dict) or not isinstance(regions, list):
        raise ValueError("Combined spec must contain outputs and regions")
    normalized_morph_mode = morph_mode.lower().replace("_", "-")
    if normalized_morph_mode not in {"standard", "high-bmi"}:
        raise ValueError("morph_mode must be either 'standard' or 'high-bmi'")
    padding_voxels = max(0, int(xy_padding_voxels))
    if normalized_morph_mode == "high-bmi" and padding_voxels == 0:
        padding_voxels = 96

    blood_labels_path = _resolve_path(str(outputs["blood_material_labels"]), spec_path)
    contrast_labels_path = _resolve_path(str(outputs.get("contrast_material_labels", outputs["blood_material_labels"])), spec_path)
    flow_boundary_path = outputs.get("flow_boundary_labels")

    blood_image = nib.load(str(blood_labels_path))
    contrast_image = nib.load(str(contrast_labels_path))
    blood_labels = np.rint(np.asanyarray(blood_image.dataobj)).astype(np.int16)
    contrast_labels = np.rint(np.asanyarray(contrast_image.dataobj)).astype(np.int16)
    if contrast_labels.shape != blood_labels.shape:
        raise ValueError("Blood and contrast material-label maps must have the same shape")
    source_shape = blood_labels.shape
    spacing_mm = tuple(float(value) for value in blood_image.header.get_zooms()[:3])
    blood_labels = _pad_xy(blood_labels, padding_voxels)
    contrast_labels = _pad_xy(contrast_labels, padding_voxels)
    blood_image = _reference_with_shape(blood_image, tuple(int(value) for value in blood_labels.shape), nib)
    voxel_volume = _voxel_volume_cm3(spacing_mm)
    body = blood_labels > 0
    if not np.any(body):
        raise ValueError("Source combined phantom has an empty body mask")

    baseline_waist = float(baseline_waist_cm) if baseline_waist_cm is not None else _estimate_waist_cm(body, spacing_mm)
    bmi = _derive_target_bmi(target_height_cm=target_height_cm, target_weight_kg=target_weight_kg, target_bmi=target_bmi)
    waist = _derive_target_waist_cm(
        target_waist_cm=target_waist_cm,
        baseline_waist_cm=baseline_waist,
        target_bmi=bmi,
        baseline_bmi=baseline_bmi,
    )
    body_radial_scale = float(waist / max(baseline_waist, 1e-6))
    height_scale = float(target_height_cm / max(baseline_height_cm, 1e-6))
    initial_body_radial_max = 2.05 if normalized_morph_mode == "high-bmi" else 1.45
    calibrated_body_radial_max = 2.35 if normalized_morph_mode == "high-bmi" else 1.60
    calibration_iterations = 8 if normalized_morph_mode == "high-bmi" else 4
    body_radial_scale = float(np.clip(body_radial_scale, 0.70, initial_body_radial_max))
    height_scale = float(np.clip(height_scale, 0.82, 1.18))

    center_ijk = _body_center_ijk(body)
    for _ in range(calibration_iterations):
        calibration_body = _morph_binary_mask(
            body,
            body_mask=body,
            radial_scale=body_radial_scale,
            height_scale=height_scale,
            center_ijk=center_ijk,
            abdomen_center_fraction=abdomen_center_fraction,
            abdomen_width_fraction=abdomen_width_fraction,
            ndimage=ndimage,
        )
        calibration_body = _clean_body(calibration_body, ndimage)
        calibration_waist = _estimate_waist_cm(calibration_body, spacing_mm)
        if calibration_waist <= 0.0:
            break
        correction = waist / calibration_waist
        if abs(correction - 1.0) < 0.015:
            break
        body_radial_scale = float(np.clip(body_radial_scale * correction, 0.70, calibrated_body_radial_max))

    organ_radial_scale = 1.0 + 0.50 * (body_radial_scale - 1.0)
    bone_radial_scale = 1.0 + 0.18 * (body_radial_scale - 1.0)
    vascular_radial_scale = 1.0 + 0.58 * (body_radial_scale - 1.0)
    organ_height_scale = 1.0 + 0.70 * (height_scale - 1.0)
    bone_height_scale = 1.0 + 0.45 * (height_scale - 1.0)
    vascular_height_scale = 1.0 + 0.80 * (height_scale - 1.0)

    morphed_body = _morph_binary_mask(
        body,
        body_mask=body,
        radial_scale=body_radial_scale,
        height_scale=height_scale,
        center_ijk=center_ijk,
        abdomen_center_fraction=abdomen_center_fraction,
        abdomen_width_fraction=abdomen_width_fraction,
        ndimage=ndimage,
    )
    morphed_body = _clean_body(morphed_body, ndimage)
    morphed_blood = _morph_label_volume(
        blood_labels,
        body_mask=body,
        radial_scale=body_radial_scale,
        height_scale=height_scale,
        center_ijk=center_ijk,
        abdomen_center_fraction=abdomen_center_fraction,
        abdomen_width_fraction=abdomen_width_fraction,
        ndimage=ndimage,
    )
    morphed_contrast = _morph_label_volume(
        contrast_labels,
        body_mask=body,
        radial_scale=body_radial_scale,
        height_scale=height_scale,
        center_ijk=center_ijk,
        abdomen_center_fraction=abdomen_center_fraction,
        abdomen_width_fraction=abdomen_width_fraction,
        ndimage=ndimage,
    )
    morphed_blood[morphed_body & (morphed_blood == 0)] = 3
    morphed_contrast[morphed_body & (morphed_contrast == 0)] = 3
    deep_structure_labels = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    morphed_blood[np.isin(morphed_blood, deep_structure_labels)] = 4
    morphed_contrast[np.isin(morphed_contrast, deep_structure_labels)] = 4

    overlay_groups = [
        ((10, 11), bone_radial_scale, bone_height_scale),
        ((8, 6, 7, 9, 12), organ_radial_scale, organ_height_scale),
        ((13, 14, 15), vascular_radial_scale, vascular_height_scale),
    ]
    for labels_to_overlay, radial, zscale in overlay_groups:
        for label_value in labels_to_overlay:
            blood_mask = _morph_binary_mask(
                blood_labels == label_value,
                body_mask=body,
                radial_scale=radial,
                height_scale=zscale,
                center_ijk=center_ijk,
                abdomen_center_fraction=abdomen_center_fraction,
                abdomen_width_fraction=abdomen_width_fraction,
                ndimage=ndimage,
            ) & morphed_body
            contrast_mask = _morph_binary_mask(
                contrast_labels == label_value,
                body_mask=body,
                radial_scale=radial,
                height_scale=zscale,
                center_ijk=center_ijk,
                abdomen_center_fraction=abdomen_center_fraction,
                abdomen_width_fraction=abdomen_width_fraction,
                ndimage=ndimage,
            ) & morphed_body
            morphed_blood[blood_mask] = label_value
            morphed_contrast[contrast_mask] = label_value

    morphed_blood[~morphed_body] = 0
    morphed_contrast[~morphed_body] = 0
    vascular_fluid = np.isin(morphed_blood, (14, 15))
    vessel_wall = morphed_blood == 13
    connected, vascular_count = ndimage.label(vascular_fluid, structure=np.ones((3, 3, 3), dtype=bool))
    if vascular_count > 1:
        sizes = np.bincount(connected.ravel(), minlength=vascular_count + 1)
        sizes[0] = 0
        largest = int(sizes.argmax())
        vascular_fluid = connected == largest
        morphed_blood[np.isin(morphed_blood, (14, 15)) & ~vascular_fluid] = 4
        morphed_contrast[np.isin(morphed_contrast, (14, 15)) & ~vascular_fluid] = 4
        vascular_count = 1

    flow_boundary_morphed: np.ndarray | None = None
    if flow_boundary_path:
        flow_image = nib.load(str(_resolve_path(str(flow_boundary_path), spec_path)))
        flow_labels = np.rint(np.asanyarray(flow_image.dataobj)).astype(np.int16)
        if flow_labels.shape != source_shape:
            raise ValueError(f"Flow boundary labels shape {flow_labels.shape} does not match material labels shape {source_shape}")
        flow_labels = _pad_xy(flow_labels, padding_voxels)
        flow_boundary_morphed = _morph_label_volume(
            flow_labels,
            body_mask=body,
            radial_scale=vascular_radial_scale,
            height_scale=vascular_height_scale,
            center_ijk=center_ijk,
            abdomen_center_fraction=abdomen_center_fraction,
            abdomen_width_fraction=abdomen_width_fraction,
            ndimage=ndimage,
        )
        flow_boundary_morphed[~morphed_body] = 0

    blood_density = _property_volume(morphed_blood, regions, "mass_density_g_cm3")
    blood_red = _property_volume(morphed_blood, regions, "relative_electron_density")
    blood_hu = _property_volume(morphed_blood, regions, "target_hu_midpoint")
    contrast_density = _property_volume(morphed_contrast, regions, "mass_density_g_cm3")
    contrast_red = _property_volume(morphed_contrast, regions, "relative_electron_density")
    contrast_hu = _property_volume(morphed_contrast, regions, "target_hu_midpoint")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blood_labels_out = output / f"{case_id}_anthro_morphed_material_labels_blood_v001.nii.gz"
    blood_density_out = output / f"{case_id}_anthro_morphed_mass_density_blood_v001.nii.gz"
    blood_red_out = output / f"{case_id}_anthro_morphed_relative_electron_density_blood_v001.nii.gz"
    blood_hu_out = output / f"{case_id}_anthro_morphed_synthetic_hu_blood_v001.nii.gz"
    contrast_labels_out = output / f"{case_id}_anthro_morphed_material_labels_contrast_v001.nii.gz"
    contrast_density_out = output / f"{case_id}_anthro_morphed_mass_density_contrast_v001.nii.gz"
    contrast_red_out = output / f"{case_id}_anthro_morphed_relative_electron_density_contrast_v001.nii.gz"
    contrast_hu_out = output / f"{case_id}_anthro_morphed_synthetic_hu_contrast_v001.nii.gz"
    body_out = output / f"{case_id}_anthro_morphed_body_mask_v001.nii.gz"
    vascular_fluid_out = output / f"{case_id}_anthro_morphed_vascular_fluid_mask_v001.nii.gz"
    vessel_wall_out = output / f"{case_id}_anthro_morphed_vessel_wall_mask_v001.nii.gz"
    flow_boundary_out = output / f"{case_id}_anthro_morphed_flow_boundary_labels_v001.nii.gz"
    scale_profile = output / f"{case_id}_anthro_morph_scale_profile_v001.csv"
    preview = output / f"{case_id}_anthro_morph_preview_v001.png"
    spec_yaml = output / f"{case_id}_anthro_morph_spec_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_anthro_morph_report_v001.md"

    _write_nifti(blood_labels_out, morphed_blood.astype(np.int16), blood_image, nib)
    _write_nifti(blood_density_out, blood_density.astype(np.float32), blood_image, nib)
    _write_nifti(blood_red_out, blood_red.astype(np.float32), blood_image, nib)
    _write_nifti(blood_hu_out, blood_hu.astype(np.float32), blood_image, nib)
    _write_nifti(contrast_labels_out, morphed_contrast.astype(np.int16), blood_image, nib)
    _write_nifti(contrast_density_out, contrast_density.astype(np.float32), blood_image, nib)
    _write_nifti(contrast_red_out, contrast_red.astype(np.float32), blood_image, nib)
    _write_nifti(contrast_hu_out, contrast_hu.astype(np.float32), blood_image, nib)
    _write_nifti(body_out, morphed_body.astype(np.uint8), blood_image, nib)
    _write_nifti(vascular_fluid_out, vascular_fluid.astype(np.uint8), blood_image, nib)
    _write_nifti(vessel_wall_out, vessel_wall.astype(np.uint8), blood_image, nib)
    flow_boundary_path_out: str | None = None
    if flow_boundary_morphed is not None:
        _write_nifti(flow_boundary_out, flow_boundary_morphed.astype(np.int16), blood_image, nib)
        flow_boundary_path_out = str(flow_boundary_out)

    _write_scale_profile(
        scale_profile,
        shape_z=blood_labels.shape[2],
        body_mask=body,
        spacing_mm=spacing_mm,
        body_radial_scale=body_radial_scale,
        organ_radial_scale=organ_radial_scale,
        bone_radial_scale=bone_radial_scale,
        vascular_radial_scale=vascular_radial_scale,
        abdomen_center_fraction=abdomen_center_fraction,
        abdomen_width_fraction=abdomen_width_fraction,
    )

    baseline_body_volume = float(body.sum() * voxel_volume)
    morphed_body_volume = float(morphed_body.sum() * voxel_volume)
    body_change = 0.0 if baseline_body_volume == 0.0 else 100.0 * (morphed_body_volume - baseline_body_volume) / baseline_body_volume
    region_stats = _region_stats(blood_labels, morphed_blood, regions, voxel_volume)
    achieved_waist = _estimate_waist_cm(morphed_body, spacing_mm)
    notes = [
        "anthropometric_morph_not_subject_specific_registration",
        "waist_drives_transverse_body_envelope_scaling",
        "bmi_used_to_infer_waist_only_when_target_waist_is_not_supplied",
        "bone_and_organs_are_deformed_less_than_soft_tissue",
        "downstream_3d_rt_and_flow_outputs_should_be_regenerated_from_morphed_labels",
    ]
    if normalized_morph_mode == "high-bmi":
        notes.append("high_bmi_mode_uses_expanded_xy_canvas_and_radial_calibration_limits_without_abrupt_abdomen_width_change")
    if padding_voxels > 0:
        notes.append("xy_voxel_grid_was_padded_to_avoid_large_body_envelope_clipping")
        notes.append("affine_spacing_is_preserved_while_voxel_grid_extent_expands_for_digital_pipeline_compatibility")
    else:
        notes.append("same_voxel_grid_and_affine_are_preserved_for_pipeline_compatibility")
    result = AnthropometricMorphResult(
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
        morphed_body_mask_path=str(body_out),
        morphed_vascular_fluid_mask_path=str(vascular_fluid_out),
        morphed_vessel_wall_mask_path=str(vessel_wall_out),
        morphed_flow_boundary_labels_path=flow_boundary_path_out,
        scale_profile_csv_path=str(scale_profile),
        preview_png_path=str(preview),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        morph_mode=normalized_morph_mode,
        xy_padding_voxels=padding_voxels,
        output_shape_voxels=tuple(int(value) for value in morphed_blood.shape),
        target_height_cm=float(target_height_cm),
        target_weight_kg=None if target_weight_kg is None else float(target_weight_kg),
        target_bmi=bmi,
        target_waist_cm=waist,
        baseline_height_cm=float(baseline_height_cm),
        baseline_bmi=float(baseline_bmi),
        baseline_waist_cm=baseline_waist,
        achieved_waist_cm=achieved_waist,
        target_body_radial_scale=body_radial_scale,
        target_height_scale=height_scale,
        baseline_body_volume_cm3=baseline_body_volume,
        morphed_body_volume_cm3=morphed_body_volume,
        body_volume_change_percent=body_change,
        baseline_bbox_mm=_bbox_mm(body, spacing_mm),
        morphed_bbox_mm=_bbox_mm(morphed_body, spacing_mm),
        vascular_components=int(vascular_count),
        region_stats=region_stats,
        notes=tuple(notes),
    )
    _render_preview(
        preview,
        baseline_labels=blood_labels,
        morphed_labels=morphed_blood,
        baseline_body=body,
        morphed_body=morphed_body,
        regions=regions,
        spacing_mm=spacing_mm,
        result_summary={
            "target_bmi": result.target_bmi,
            "target_waist_cm": result.target_waist_cm,
            "target_height_cm": result.target_height_cm,
            "baseline_waist_cm": result.baseline_waist_cm,
            "achieved_waist_cm": result.achieved_waist_cm,
            "baseline_body_volume_cm3": result.baseline_body_volume_cm3,
            "morphed_body_volume_cm3": result.morphed_body_volume_cm3,
            "morph_mode": result.morph_mode,
            "xy_padding_voxels": result.xy_padding_voxels,
        },
    )
    _write_spec(spec_yaml, spec_path, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_anthropometric_torso_morph_result(result: AnthropometricMorphResult) -> str:
    lines = [
        "Anthropometric torso morph created",
        f"Case ID: {result.case_id}",
        f"Target BMI: {result.target_bmi:.2f}",
        f"Target waist: {result.target_waist_cm:.1f} cm",
        f"Achieved waist proxy: {result.achieved_waist_cm:.1f} cm",
        f"Morph mode: {result.morph_mode}",
        f"XY padding voxels/side: {result.xy_padding_voxels}",
        f"Body radial scale: {result.target_body_radial_scale:.3f}",
        f"Height scale: {result.target_height_scale:.3f}",
        f"Body volume: {result.baseline_body_volume_cm3 / 1000.0:.3f} -> {result.morphed_body_volume_cm3 / 1000.0:.3f} L",
        f"Vascular components: {result.vascular_components}",
        f"Blood material labels: {result.morphed_blood_material_labels_path}",
        f"Contrast material labels: {result.morphed_contrast_material_labels_path}",
        f"Preview PNG: {result.preview_png_path}",
        f"Spec YAML: {result.spec_yaml_path}",
        f"Report: {result.report_path}",
    ]
    return "\n".join(lines)
