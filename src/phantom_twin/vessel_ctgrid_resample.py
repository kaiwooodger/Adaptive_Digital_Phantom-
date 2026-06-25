from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import numpy as np
import yaml

from .validation_intake import DEFAULT_REQUIRED_VESSEL_LABELS


@dataclass(frozen=True)
class VesselCtGridResampleResult:
    case_id: str
    ct_path: str
    source_vessel_path: str
    target_mask_path: str
    output_dir: str
    resampled_vessel_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    alignment_mode: str
    source_geometry_status: str
    output_geometry_status: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    source_nonzero_voxels: int
    output_nonzero_voxels: int
    source_label_count: int
    output_label_count: int
    present_required_vessel_label_count: int
    required_vessel_label_count: int
    vessel_label_coverage_percent: float
    missing_required_vessel_labels: tuple[int, ...]
    recommended_commands: tuple[str, ...]
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from scipy import ndimage  # type: ignore
    except ImportError as exc:
        raise RuntimeError("CT-grid vessel resampling requires matplotlib, nibabel, and scipy.") from exc
    return plt, nib, ndimage


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def _input_path(manifest: dict[str, Any], role: str) -> str:
    for item in manifest.get("inputs", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")) == role:
            return str(item.get("staged_path", "") or item.get("source_path", "") or "")
    return ""


def _paths_from_staged_manifest(path: str | Path | None) -> tuple[str, str, str]:
    if path is None or str(path) == "":
        return "", "", ""
    manifest = _load_yaml(path)
    if manifest.get("package_type") != "p1_validation_case_staging":
        raise ValueError("staged case manifest must have package_type=p1_validation_case_staging")
    return _input_path(manifest, "ct"), _input_path(manifest, "vessel_seg"), _input_path(manifest, "organ_seg")


def _geometry_status(reference_image: Any, candidate_image: Any) -> str:
    ref_shape = tuple(int(value) for value in reference_image.shape)
    cand_shape = tuple(int(value) for value in candidate_image.shape)
    ref_spacing = tuple(float(value) for value in reference_image.header.get_zooms()[: len(ref_shape)])
    cand_spacing = tuple(float(value) for value in candidate_image.header.get_zooms()[: len(cand_shape)])
    shape_match = ref_shape == cand_shape
    spacing_match = np.allclose(np.asarray(ref_spacing), np.asarray(cand_spacing), atol=1e-3)
    affine_match = np.allclose(np.asarray(reference_image.affine), np.asarray(candidate_image.affine), atol=1e-3)
    if shape_match and spacing_match and affine_match:
        return "co_registered_to_ct_grid"
    if shape_match and spacing_match:
        return "same_shape_spacing_but_affine_differs"
    return "registration_required_to_ct_grid"


def _label_stats(data: np.ndarray) -> tuple[int, int, set[int]]:
    labels = set(int(value) for value in np.unique(data.astype(np.int64)).tolist())
    labels.discard(0)
    return int(np.count_nonzero(data)), len(labels), labels


def _coverage(labels: set[int], required: tuple[int, ...]) -> tuple[int, tuple[int, ...], float]:
    missing = tuple(label for label in required if label not in labels)
    present = len(required) - len(missing)
    return present, missing, 100.0 * present / max(len(required), 1)


def _bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        shape = np.asarray(mask.shape, dtype=float)
        return shape * 0.5, np.maximum(shape, 1.0)
    mins = coords.min(axis=0).astype(float)
    maxs = coords.max(axis=0).astype(float)
    center = (mins + maxs) * 0.5
    extent = np.maximum(maxs - mins + 1.0, 1.0)
    return center, extent


def _target_mask_from_inputs(ct_data: np.ndarray, target_mask_path: str, nib: Any) -> np.ndarray:
    if target_mask_path:
        mask_path = Path(target_mask_path)
        if mask_path.exists() and not mask_path.is_dir():
            mask_data = np.asanyarray(nib.load(str(mask_path)).dataobj)
            if mask_data.shape == ct_data.shape:
                return mask_data != 0
    finite = np.isfinite(ct_data)
    if np.any(finite):
        return finite & (ct_data > -900.0)
    return np.ones(ct_data.shape, dtype=bool)


def _header_affine_matrix(source_image: Any, target_image: Any) -> tuple[np.ndarray, np.ndarray]:
    transform = np.linalg.inv(source_image.affine) @ target_image.affine
    return transform[:3, :3], transform[:3, 3]


def _centered_bbox_matrix(source_data: np.ndarray, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center, source_extent = _bbox(source_data != 0)
    target_center, target_extent = _bbox(target_mask)
    scale = source_extent / np.maximum(target_extent, 1.0)
    matrix = np.diag(scale)
    offset = source_center - matrix @ target_center
    return matrix, offset


def _resample(
    *,
    source_data: np.ndarray,
    source_image: Any,
    target_image: Any,
    target_mask: np.ndarray,
    alignment_mode: str,
    ndimage: Any,
) -> np.ndarray:
    if alignment_mode == "header-affine":
        matrix, offset = _header_affine_matrix(source_image, target_image)
    elif alignment_mode == "centered-bbox":
        matrix, offset = _centered_bbox_matrix(source_data, target_mask)
    else:
        raise ValueError("alignment_mode must be 'header-affine' or 'centered-bbox'")
    output = ndimage.affine_transform(
        source_data.astype(np.int16),
        matrix=matrix,
        offset=offset,
        output_shape=tuple(int(value) for value in target_image.shape),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return np.rint(output).astype(np.int16)


def _write_nifti(path: Path, data: np.ndarray, reference_image: Any, nib: Any) -> None:
    header = reference_image.header.copy()
    header.set_data_dtype(np.int16)
    image = nib.Nifti1Image(data.astype(np.int16), reference_image.affine, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _write_preview(path: Path, result: VesselCtGridResampleResult, output_data: np.ndarray) -> None:
    plt, _, _ = _import_dependencies()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if output_data.ndim == 3 and np.any(output_data):
        counts = np.count_nonzero(output_data, axis=(0, 1))
        z_index = int(np.argmax(counts))
        ax.imshow(np.rot90(output_data[:, :, z_index]), cmap="turbo", interpolation="nearest")
        ax.set_title(f"Resampled Vessel Labels, z={z_index}")
        ax.axis("off")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No output vessel voxels", ha="center", va="center", fontsize=12)

    axes[1].axis("off")
    text = "\n".join(
        [
            "CT-Grid Vessel Resample",
            "",
            f"Case: {result.case_id}",
            f"Mode: {result.alignment_mode}",
            f"Source geometry: {result.source_geometry_status}",
            f"Output geometry: {result.output_geometry_status}",
            f"Output voxels: {result.output_nonzero_voxels}",
            f"Required coverage: {result.vessel_label_coverage_percent:.1f}%",
        ]
    )
    axes[1].text(
        0.02,
        0.98,
        text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        bbox={"facecolor": "#f7f9fb", "edgecolor": "#ccd6dd", "boxstyle": "round,pad=0.6"},
    )
    fig.suptitle("Vessel Labels Resampled To CT Grid", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_manifest(path: Path, result: VesselCtGridResampleResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "p1_vessel_ct_grid_resample",
        "alignment_mode": result.alignment_mode,
        "ct_path": result.ct_path,
        "source_vessel_path": result.source_vessel_path,
        "target_mask_path": result.target_mask_path,
        "summary": {
            "source_geometry_status": result.source_geometry_status,
            "output_geometry_status": result.output_geometry_status,
            "source_shape": list(result.source_shape),
            "target_shape": list(result.target_shape),
            "source_nonzero_voxels": result.source_nonzero_voxels,
            "output_nonzero_voxels": result.output_nonzero_voxels,
            "source_label_count": result.source_label_count,
            "output_label_count": result.output_label_count,
            "required_vessel_label_count": result.required_vessel_label_count,
            "present_required_vessel_label_count": result.present_required_vessel_label_count,
            "vessel_label_coverage_percent": result.vessel_label_coverage_percent,
            "missing_required_vessel_labels": list(result.missing_required_vessel_labels),
        },
        "outputs": {
            "resampled_vessel": result.resampled_vessel_path,
            "manifest_yaml": result.manifest_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "recommended_commands": list(result.recommended_commands),
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VesselCtGridResampleResult) -> str:
    image_rel = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    lines = [
        "# P1 Vessel CT-Grid Resampling",
        "",
        f"Case ID: `{result.case_id}`",
        f"Alignment mode: `{result.alignment_mode}`",
        f"Source geometry: `{result.source_geometry_status}`",
        f"Output geometry: `{result.output_geometry_status}`",
        "",
        f"![CT-grid vessel resample]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Source shape: `{result.source_shape}`",
        f"- Target CT shape: `{result.target_shape}`",
        f"- Source / output nonzero voxels: {result.source_nonzero_voxels} / {result.output_nonzero_voxels}",
        f"- Source / output label count: {result.source_label_count} / {result.output_label_count}",
        f"- Required label coverage: {result.present_required_vessel_label_count}/{result.required_vessel_label_count} ({result.vessel_label_coverage_percent:.1f}%)",
        f"- Missing required labels: `{', '.join(str(label) for label in result.missing_required_vessel_labels) or 'none'}`",
        "",
        "## Outputs",
        "",
        f"- Resampled vessel NIfTI: `{result.resampled_vessel_path}`",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        "",
        "## Recommended Commands",
        "",
        "```bash",
    ]
    lines.extend(result.recommended_commands)
    lines.extend(["```", "", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_vessel_ct_grid_resample_result(result: VesselCtGridResampleResult) -> str:
    return _format_report(result)


def resample_vessel_to_ct_grid(
    *,
    ct_path: str | Path | None = None,
    vessel_seg_path: str | Path | None = None,
    staged_case_manifest_path: str | Path | None = None,
    target_mask_path: str | Path | None = None,
    case_id: str = "vessel_ct_grid_resample",
    output_dir: str | Path = "outputs/digital/vessel_ct_grid_resample",
    alignment_mode: str = "header-affine",
    required_vessel_labels: tuple[int, ...] | None = DEFAULT_REQUIRED_VESSEL_LABELS,
    report_path: str | Path | None = None,
) -> VesselCtGridResampleResult:
    plt, nib, ndimage = _import_dependencies()
    del plt
    manifest_ct, manifest_vessel, manifest_target_mask = _paths_from_staged_manifest(staged_case_manifest_path)
    resolved_ct = str(ct_path or manifest_ct)
    resolved_vessel = str(vessel_seg_path or manifest_vessel)
    resolved_target_mask = str(target_mask_path or manifest_target_mask or "")
    if not resolved_ct or not resolved_vessel:
        raise ValueError("Provide --ct and --vessel-seg, or a staged case manifest with ct and vessel_seg inputs")
    ct_image = nib.load(resolved_ct)
    vessel_image = nib.load(resolved_vessel)
    ct_data = np.asanyarray(ct_image.dataobj)
    source_data = np.rint(np.asanyarray(vessel_image.dataobj)).astype(np.int16)
    target_mask = _target_mask_from_inputs(ct_data, resolved_target_mask, nib)
    resampled = _resample(
        source_data=source_data,
        source_image=vessel_image,
        target_image=ct_image,
        target_mask=target_mask,
        alignment_mode=alignment_mode,
        ndimage=ndimage,
    )
    output = Path(output_dir)
    resampled_path = output / f"{case_id}_ct_grid_vessel_labels_v001.nii.gz"
    manifest_yaml = output / f"{case_id}_ct_grid_vessel_resample_manifest_v001.yaml"
    preview_png = output / f"{case_id}_ct_grid_vessel_resample_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_ct_grid_vessel_resample_report_v001.md"
    _write_nifti(resampled_path, resampled, ct_image, nib)

    source_nonzero, source_label_count, source_labels = _label_stats(source_data)
    output_nonzero, output_label_count, output_labels = _label_stats(resampled)
    resolved_required = tuple(required_vessel_labels or DEFAULT_REQUIRED_VESSEL_LABELS)
    present_required, missing_required, coverage = _coverage(output_labels, resolved_required)
    output_image = nib.load(str(resampled_path))
    source_geometry = _geometry_status(ct_image, vessel_image)
    output_geometry = _geometry_status(ct_image, output_image)
    recommended_commands = (
        f"python -m phantom_twin.cli stage-validation-case --case-id {case_id}_ct_grid_case --source-dataset ct_grid_resampled_vessel --ct {resolved_ct} --organ-seg {resolved_target_mask or '<organ_seg_path>'} --vessel-seg {resampled_path}",
        f"python -m phantom_twin.cli build-validation-intake-package --cases-csv data/validation/p1_cases/{case_id}_ct_grid_case/{case_id}_ct_grid_case_p1_intake_candidate_v001.csv",
    )
    notes = [
        "nearest_neighbor_resampling_preserves_integer_vessel_labels",
        "output_is_on_the_CT_grid_but_this_is_not_a_validated_deformable_patient_specific_registration",
        "header-affine_mode_is_appropriate_only_when_source_and_CT_share_a_world_coordinate_frame",
    ]
    if alignment_mode == "centered-bbox":
        notes.append("centered-bbox_mode_is_template_placement_for_research_QA_not_anatomical_registration")
    if output_nonzero == 0:
        notes.append("resampled_output_has_zero_vessel_voxels_check_source_target_overlap_or_use_centered-bbox_for_template_smoke_tests")
    result = VesselCtGridResampleResult(
        case_id=case_id,
        ct_path=resolved_ct,
        source_vessel_path=resolved_vessel,
        target_mask_path=resolved_target_mask,
        output_dir=str(output),
        resampled_vessel_path=str(resampled_path),
        manifest_yaml_path=str(manifest_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        alignment_mode=alignment_mode,
        source_geometry_status=source_geometry,
        output_geometry_status=output_geometry,
        source_shape=tuple(int(value) for value in vessel_image.shape),
        target_shape=tuple(int(value) for value in ct_image.shape),
        source_nonzero_voxels=source_nonzero,
        output_nonzero_voxels=output_nonzero,
        source_label_count=source_label_count,
        output_label_count=output_label_count,
        present_required_vessel_label_count=present_required,
        required_vessel_label_count=len(resolved_required),
        vessel_label_coverage_percent=coverage,
        missing_required_vessel_labels=missing_required,
        recommended_commands=recommended_commands,
        notes=tuple(notes),
    )
    _write_preview(preview_png, result, resampled)
    _write_manifest(manifest_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
