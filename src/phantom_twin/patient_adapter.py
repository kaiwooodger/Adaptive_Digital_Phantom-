from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import shutil
from typing import Any

import numpy as np
import yaml

from .stage007_baseline import resolve_stage007_active_baseline


NIFTI_EXTENSIONS = (".nii", ".nii.gz")


@dataclass(frozen=True)
class PatientInputSummary:
    role: str
    source_path: str
    staged_path: str
    exists: bool
    input_kind: str
    status: str
    shape: tuple[int, ...]
    spacing_mm: tuple[float, ...]
    dtype: str
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    nonzero_voxels: int | None
    nonzero_fraction: float | None
    unique_label_count: int | None
    unique_labels_sample: tuple[float, ...]
    geometry_status: str
    affine: tuple[tuple[float, ...], ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PatientPhantomAdapterResult:
    case_id: str
    patient_id: str
    adaptation_mode: str
    output_dir: str
    manifest_yaml_path: str
    input_qa_csv_path: str
    preview_png_path: str
    report_path: str
    input_count: int
    primary_ct_status: str
    anatomy_adaptation_status: str
    vascular_adaptation_status: str
    rt_readiness_status: str
    overall_status: str
    inputs: tuple[PatientInputSummary, ...]
    recommended_commands: tuple[str, ...]
    notes: tuple[str, ...]


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Patient input adapter requires nibabel for NIfTI input QA.") from exc
    return nib


def _compound_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "patient"


def _detect_input_kind(path: Path) -> str:
    if path.is_dir():
        if _nifti_files_in_directory(path):
            return "nifti_mask_directory"
        return "dicom_directory"
    if path.name.endswith(NIFTI_EXTENSIONS):
        return "nifti"
    return "file"


def _nifti_files_in_directory(path: Path) -> tuple[Path, ...]:
    if not path.exists() or not path.is_dir():
        return ()
    return tuple(sorted(item for item in path.rglob("*") if item.is_file() and item.name.endswith(NIFTI_EXTENSIONS)))


def _stage_path(path: Path, role: str, output_dir: Path, copy_inputs: bool) -> Path:
    if not copy_inputs:
        return path
    staged_dir = output_dir / "inputs"
    staged_dir.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        destination = staged_dir / f"{role}_{path.name}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(path, destination)
        return destination
    destination = staged_dir / f"{role}{_compound_suffix(path)}"
    shutil.copy2(path, destination)
    return destination


def _load_nifti_summary(path: Path, *, label_like: bool) -> dict[str, Any]:
    nib = _import_nibabel()
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        min_value = max_value = mean_value = None
    else:
        min_value = float(np.nanmin(finite))
        max_value = float(np.nanmax(finite))
        mean_value = float(np.nanmean(finite))
    nonzero = int(np.count_nonzero(data))
    unique_count = None
    unique_sample: tuple[float, ...] = ()
    if label_like:
        labels = np.unique(data)
        unique_count = int(len(labels))
        if len(labels) <= 32:
            sample = labels
        else:
            sample = np.concatenate([labels[:24], labels[-8:]])
        unique_sample = tuple(float(value) for value in sample.tolist())
    return {
        "shape": tuple(int(value) for value in data.shape),
        "spacing_mm": tuple(float(value) for value in image.header.get_zooms()[: data.ndim]),
        "dtype": str(data.dtype),
        "min_value": min_value,
        "max_value": max_value,
        "mean_value": mean_value,
        "nonzero_voxels": nonzero,
        "nonzero_fraction": float(nonzero / max(data.size, 1)),
        "unique_label_count": unique_count,
        "unique_labels_sample": unique_sample,
        "affine": tuple(tuple(float(value) for value in row) for row in image.affine.tolist()),
    }


def _load_nifti_mask_directory_summary(path: Path, *, label_like: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    files = _nifti_files_in_directory(path)
    if not files:
        return _empty_summary(), ("nifti_mask_directory_contains_no_nifti_files",)
    nib = _import_nibabel()
    reference = nib.load(str(files[0]))
    reference_shape = tuple(int(value) for value in reference.shape)
    reference_spacing = tuple(float(value) for value in reference.header.get_zooms()[: len(reference_shape)])
    reference_affine = tuple(tuple(float(value) for value in row) for row in reference.affine.tolist())
    union_mask = np.zeros(reference_shape, dtype=bool)
    min_value = np.inf
    max_value = -np.inf
    total_sum = 0.0
    total_voxels = 0
    unique_values: set[float] = set()
    mismatched_files = 0
    for file_path in files:
        image = nib.load(str(file_path))
        shape = tuple(int(value) for value in image.shape)
        spacing = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
        affine = tuple(tuple(float(value) for value in row) for row in image.affine.tolist())
        if shape != reference_shape or not np.allclose(spacing, reference_spacing, atol=1e-3) or not np.allclose(affine, reference_affine, atol=1e-3):
            mismatched_files += 1
            continue
        data = np.asanyarray(image.dataobj)
        finite = data[np.isfinite(data)]
        if finite.size:
            min_value = min(min_value, float(np.nanmin(finite)))
            max_value = max(max_value, float(np.nanmax(finite)))
            total_sum += float(np.nansum(finite))
            total_voxels += int(finite.size)
        union_mask |= data != 0
        if label_like:
            for value in np.unique(data)[:64].tolist():
                unique_values.add(float(value))
    nonzero = int(np.count_nonzero(union_mask))
    notes = [f"nifti_mask_directory_files={len(files)}"]
    if mismatched_files:
        notes.append(f"nifti_mask_directory_skipped_geometry_mismatch_files={mismatched_files}")
    if not np.isfinite(min_value):
        min_value = max_value = None
    return (
        {
            "shape": reference_shape,
            "spacing_mm": reference_spacing,
            "dtype": "nifti_mask_directory",
            "min_value": None if min_value is None else float(min_value),
            "max_value": None if max_value is None else float(max_value),
            "mean_value": None if total_voxels == 0 else float(total_sum / total_voxels),
            "nonzero_voxels": nonzero,
            "nonzero_fraction": float(nonzero / max(union_mask.size, 1)),
            "unique_label_count": None if not label_like else len(unique_values),
            "unique_labels_sample": tuple(sorted(unique_values)[:32]) if label_like else (),
            "affine": reference_affine,
        },
        tuple(notes),
    )


def _empty_summary() -> dict[str, Any]:
    return {
        "shape": (),
        "spacing_mm": (),
        "dtype": "",
        "min_value": None,
        "max_value": None,
        "mean_value": None,
        "nonzero_voxels": None,
        "nonzero_fraction": None,
        "unique_label_count": None,
        "unique_labels_sample": (),
        "affine": (),
    }


def _geometry_status(summary: dict[str, Any], ct_summary: dict[str, Any] | None, role: str) -> str:
    if summary["shape"] == ():
        return "not_evaluated"
    if role == "ct":
        return "primary_reference_grid"
    if ct_summary is None or ct_summary["shape"] == ():
        return "no_primary_ct_reference"
    shape_match = tuple(summary["shape"]) == tuple(ct_summary["shape"])
    spacing_match = np.allclose(np.asarray(summary["spacing_mm"], dtype=float), np.asarray(ct_summary["spacing_mm"], dtype=float), atol=1e-3)
    affine_match = np.allclose(np.asarray(summary["affine"], dtype=float), np.asarray(ct_summary["affine"], dtype=float), atol=1e-3)
    if shape_match and spacing_match and affine_match:
        return "co_registered_to_ct_grid"
    if shape_match and spacing_match:
        return "same_shape_spacing_but_affine_differs"
    return "registration_required_to_ct_grid"


def _summarize_input(
    *,
    role: str,
    source_path: str | Path | None,
    output_dir: Path,
    copy_inputs: bool,
    label_like: bool,
    ct_summary: dict[str, Any] | None,
) -> PatientInputSummary | None:
    if source_path is None or str(source_path) == "":
        return None
    source = Path(source_path)
    exists = source.exists()
    notes: list[str] = []
    if not exists:
        summary = _empty_summary()
        return PatientInputSummary(
            role=role,
            source_path=str(source),
            staged_path=str(source),
            exists=False,
            input_kind="missing",
            status="missing",
            geometry_status="not_evaluated",
            notes=("input_path_does_not_exist",),
            **summary,
        )

    kind = _detect_input_kind(source)
    staged = _stage_path(source, role, output_dir, copy_inputs)
    if kind == "nifti":
        summary = _load_nifti_summary(staged, label_like=label_like)
        status = "accepted"
        if label_like and summary["unique_label_count"] in (None, 0, 1):
            status = "review"
            notes.append("label_like_input_has_one_or_zero_unique_values")
        if summary["nonzero_voxels"] == 0:
            status = "review"
            notes.append("input_volume_is_empty")
    elif kind == "nifti_mask_directory":
        summary, directory_notes = _load_nifti_mask_directory_summary(staged, label_like=label_like)
        status = "accepted" if summary["shape"] else "review"
        notes.extend(directory_notes)
        if summary["nonzero_voxels"] == 0:
            status = "review"
            notes.append("input_mask_directory_is_empty")
    elif kind == "dicom_directory":
        summary = _empty_summary()
        status = "accepted_metadata_only"
        notes.append("dicom_directory_staged_without_series_geometry_QA_convert_to_nifti_for_full_pipeline")
    else:
        summary = _empty_summary()
        status = "review"
        notes.append("unsupported_or_unclassified_file_type_convert_to_nifti_for_full_pipeline")

    geometry = _geometry_status(summary, ct_summary, role)
    if geometry in {"same_shape_spacing_but_affine_differs", "registration_required_to_ct_grid"}:
        notes.append(geometry)
    return PatientInputSummary(
        role=role,
        source_path=str(source),
        staged_path=str(staged),
        exists=exists,
        input_kind=kind,
        status=status,
        geometry_status=geometry,
        notes=tuple(notes),
        **summary,
    )


def _input_by_role(inputs: tuple[PatientInputSummary, ...], role: str) -> PatientInputSummary | None:
    for item in inputs:
        if item.role == role:
            return item
    return None


def _path_for_role(inputs: tuple[PatientInputSummary, ...], role: str) -> str | None:
    item = _input_by_role(inputs, role)
    if item is None or not item.exists:
        return None
    return item.staged_path


def _primary_ct_status(inputs: tuple[PatientInputSummary, ...]) -> str:
    ct = _input_by_role(inputs, "ct")
    if ct is None:
        return "missing_primary_ct"
    if not ct.exists:
        return "missing_primary_ct_path"
    if ct.input_kind == "nifti":
        return "ct_nifti_ready"
    if ct.input_kind == "dicom_directory":
        return "ct_dicom_needs_conversion"
    return "ct_file_needs_conversion"


def _anatomy_status(inputs: tuple[PatientInputSummary, ...]) -> str:
    ct_status = _primary_ct_status(inputs)
    organ = _input_by_role(inputs, "organ_seg")
    if ct_status.startswith("missing"):
        return "blocked_missing_ct"
    if organ is None:
        return "ct_available_needs_organ_segmentation"
    if not organ.exists:
        return "blocked_missing_organ_segmentation"
    if organ.geometry_status == "co_registered_to_ct_grid":
        return "ready_for_ct_registered_anatomy_build"
    return "organ_segmentation_registration_required"


def _vascular_status(inputs: tuple[PatientInputSummary, ...]) -> str:
    vessel = _input_by_role(inputs, "vessel_seg")
    cta = _input_by_role(inputs, "cta")
    ctv = _input_by_role(inputs, "ctv")
    vascular_scans = [item for item in (cta, ctv) if item is not None and item.exists]
    if vessel is not None and vessel.exists:
        if vessel.geometry_status == "co_registered_to_ct_grid":
            return "ready_for_patient_vessel_replacement"
        return "vessel_segmentation_registration_required"
    if vascular_scans:
        return "cta_or_ctv_available_needs_vessel_segmentation"
    return "template_vascular_graph_only"


def _rt_status(primary_ct_status: str, anatomy_status: str) -> str:
    if primary_ct_status != "ct_nifti_ready":
        return "not_ready_ct_conversion_required"
    if anatomy_status == "ready_for_ct_registered_anatomy_build":
        return "rt_density_mapping_ready_after_labelmap_review"
    return "not_ready_requires_organ_segmentation_or_registration"


def _overall_status(primary_ct_status: str, anatomy_status: str, vascular_status: str) -> str:
    if primary_ct_status.startswith("missing"):
        return "blocked_missing_primary_ct"
    if primary_ct_status != "ct_nifti_ready":
        return "staged_needs_nifti_conversion"
    if anatomy_status == "ready_for_ct_registered_anatomy_build" and vascular_status == "ready_for_patient_vessel_replacement":
        return "ready_for_patient_specific_anatomy_and_vascular_build"
    if anatomy_status == "ready_for_ct_registered_anatomy_build":
        return "ready_for_patient_anatomy_build_vascular_template_or_pending"
    return "intake_staged_needs_segmentation_or_registration"


def _recommended_commands(
    *,
    case_id: str,
    patient_id: str,
    inputs: tuple[PatientInputSummary, ...],
    organ_labelmap_path: str | Path,
    gi_labelmap_path: str | Path,
    materials_path: str | Path,
    approved_set_manifest_path: str | Path | None,
    baseline_graph_path: str | Path | None,
    baseline_combined_spec_path: str | Path | None,
    target_height_cm: float | None,
    target_weight_kg: float | None,
    target_bmi: float | None,
    target_waist_cm: float | None,
) -> tuple[str, ...]:
    commands: list[str] = []
    ct = _path_for_role(inputs, "ct")
    organ = _path_for_role(inputs, "organ_seg")
    gi = _path_for_role(inputs, "gi_seg")
    vessel = _path_for_role(inputs, "vessel_seg")

    if approved_set_manifest_path is not None:
        command = (
            "python -m phantom_twin.cli build-user-profile-adapter "
            f"--approved-set-manifest {approved_set_manifest_path} "
            f"--profile-id {patient_id} "
            f"--case-id {case_id}_profile_adapter"
        )
        if target_height_cm is not None:
            command += f" --target-height-cm {target_height_cm:.3f}"
        if target_weight_kg is not None:
            command += f" --target-weight-kg {target_weight_kg:.3f}"
        if target_bmi is not None:
            command += f" --target-bmi {target_bmi:.3f}"
        if target_waist_cm is not None:
            command += f" --target-waist-cm {target_waist_cm:.3f}"
        commands.append(command)

    organ_summary = _input_by_role(inputs, "organ_seg")
    if ct and organ and organ_summary is not None and organ_summary.geometry_status == "co_registered_to_ct_grid":
        commands.append(
            "python -m phantom_twin.cli build-digital-torso "
            f"--ct {ct} "
            f"--labels {organ} "
            f"--labelmap {organ_labelmap_path} "
            f"--materials {materials_path} "
            f"--case-id {case_id}_patient_torso "
            f"--output-dir outputs/digital/patient_builds/{case_id}/torso "
            f"--report outputs/reports/{case_id}_patient_digital_torso.md"
        )
    elif ct and organ:
        commands.append(
            "# Register/resample organ segmentation to the CT grid before running build-digital-torso."
        )
    elif ct:
        commands.append("# Segment CT into organ/material labels before running build-digital-torso.")

    gi_summary = _input_by_role(inputs, "gi_seg")
    if gi and gi_summary is not None and gi_summary.geometry_status == "co_registered_to_ct_grid":
        commands.append(
            "# Optional GI replacement is ready: pass the staged patient manifest into run-patient-phantom-build; "
            f"GI labels will use {gi_labelmap_path}."
        )
    elif gi:
        commands.append("# Register/resample GI bowel/colon/small-intestine segmentation to the CT grid before replacing GI placeholders.")

    vessel_summary = _input_by_role(inputs, "vessel_seg")
    if vessel and baseline_graph_path is not None:
        if vessel_summary is not None and vessel_summary.geometry_status != "co_registered_to_ct_grid":
            commands.append("# Register/resample vessel segmentation to the patient CT grid before graph replacement.")
        commands.append(
            "python -m phantom_twin.cli build-cta-derived-vascular-graph "
            f"--baseline-graph {baseline_graph_path} "
            f"--vascular-mask {vessel} "
            f"--case-id {case_id}_patient_vessels "
            f"--output-dir outputs/digital/patient_builds/{case_id}/vascular_graph "
            f"--report outputs/reports/{case_id}_patient_vascular_graph.md"
        )
    elif vessel:
        commands.append("# Provide --baseline-graph to replace/deform the template vascular graph from the vessel segmentation.")

    if baseline_combined_spec_path is not None:
        commands.append(
            "# After patient torso + vessels are generated, voxelize vessels, build flow boundaries, rerun coupled flow, and rerun RT/RT-flow QA using "
            f"{baseline_combined_spec_path} as the current baseline reference."
        )
    else:
        commands.append("# After patient torso + vessels are generated, voxelize vessels, build flow boundaries, rerun coupled flow, and rerun RT/RT-flow QA.")
    return tuple(commands)


def _write_input_csv(path: Path, inputs: tuple[PatientInputSummary, ...]) -> None:
    fieldnames = [
        "role",
        "source_path",
        "staged_path",
        "exists",
        "input_kind",
        "status",
        "geometry_status",
        "shape",
        "spacing_mm",
        "dtype",
        "min_value",
        "max_value",
        "mean_value",
        "nonzero_voxels",
        "nonzero_fraction",
        "unique_label_count",
        "unique_labels_sample",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in inputs:
            writer.writerow(
                {
                    "role": item.role,
                    "source_path": item.source_path,
                    "staged_path": item.staged_path,
                    "exists": item.exists,
                    "input_kind": item.input_kind,
                    "status": item.status,
                    "geometry_status": item.geometry_status,
                    "shape": "x".join(str(value) for value in item.shape),
                    "spacing_mm": "x".join(f"{value:.6g}" for value in item.spacing_mm),
                    "dtype": item.dtype,
                    "min_value": "" if item.min_value is None else f"{item.min_value:.6g}",
                    "max_value": "" if item.max_value is None else f"{item.max_value:.6g}",
                    "mean_value": "" if item.mean_value is None else f"{item.mean_value:.6g}",
                    "nonzero_voxels": "" if item.nonzero_voxels is None else item.nonzero_voxels,
                    "nonzero_fraction": "" if item.nonzero_fraction is None else f"{item.nonzero_fraction:.8f}",
                    "unique_label_count": "" if item.unique_label_count is None else item.unique_label_count,
                    "unique_labels_sample": ";".join(f"{value:.6g}" for value in item.unique_labels_sample),
                    "notes": ";".join(item.notes),
                }
            )


def _serialise_input(item: PatientInputSummary) -> dict[str, Any]:
    return {
        "role": item.role,
        "source_path": item.source_path,
        "staged_path": item.staged_path,
        "exists": item.exists,
        "input_kind": item.input_kind,
        "status": item.status,
        "geometry_status": item.geometry_status,
        "shape": list(item.shape),
        "spacing_mm": list(item.spacing_mm),
        "dtype": item.dtype,
        "min_value": item.min_value,
        "max_value": item.max_value,
        "mean_value": item.mean_value,
        "nonzero_voxels": item.nonzero_voxels,
        "nonzero_fraction": item.nonzero_fraction,
        "unique_label_count": item.unique_label_count,
        "unique_labels_sample": list(item.unique_labels_sample),
        "affine": [list(row) for row in item.affine],
        "notes": list(item.notes),
    }


def _write_manifest(
    path: Path,
    result: PatientPhantomAdapterResult,
    *,
    organ_labelmap_path: str | Path,
    gi_labelmap_path: str | Path,
    materials_path: str | Path,
    approved_set_manifest_path: str | Path | None,
    baseline_graph_path: str | Path | None,
    baseline_combined_spec_path: str | Path | None,
    target_height_cm: float | None,
    target_weight_kg: float | None,
    target_bmi: float | None,
    target_waist_cm: float | None,
    copy_inputs: bool,
) -> None:
    payload = {
        "case_id": result.case_id,
        "patient_id": result.patient_id,
        "package_type": "patient_phantom_input_adapter",
        "adaptation_mode": result.adaptation_mode,
        "copy_inputs": copy_inputs,
        "status": {
            "primary_ct": result.primary_ct_status,
            "anatomy_adaptation": result.anatomy_adaptation_status,
            "vascular_adaptation": result.vascular_adaptation_status,
            "rt_readiness": result.rt_readiness_status,
            "overall": result.overall_status,
        },
        "configuration": {
            "organ_labelmap": str(organ_labelmap_path),
            "gi_labelmap": str(gi_labelmap_path),
            "materials": str(materials_path),
            "approved_set_manifest": None if approved_set_manifest_path is None else str(approved_set_manifest_path),
            "baseline_graph": None if baseline_graph_path is None else str(baseline_graph_path),
            "baseline_combined_spec": None if baseline_combined_spec_path is None else str(baseline_combined_spec_path),
        },
        "target_profile": {
            "height_cm": target_height_cm,
            "weight_kg": target_weight_kg,
            "bmi": target_bmi,
            "waist_cm": target_waist_cm,
        },
        "inputs": [_serialise_input(item) for item in result.inputs],
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "input_qa_csv": result.input_qa_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "recommended_commands": list(result.recommended_commands),
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _load_slice(path: Path, label_like: bool) -> tuple[np.ndarray, str] | None:
    if path.is_dir():
        files = _nifti_files_in_directory(path)
        if not files:
            return None
        path = files[0]
    if not path.exists() or not path.name.endswith(NIFTI_EXTENSIONS):
        return None
    nib = _import_nibabel()
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim < 3:
        return None
    z = data.shape[2] // 2
    slice_data = np.asarray(data[:, :, z], dtype=float)
    if label_like:
        return slice_data, "label"
    finite = slice_data[np.isfinite(slice_data)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        if high > low:
            slice_data = np.clip((slice_data - low) / (high - low), 0, 1)
    return slice_data, "image"


def _write_preview(path: Path, inputs: tuple[PatientInputSummary, ...], result: PatientPhantomAdapterResult) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Patient input adapter preview generation requires matplotlib.") from exc

    display = [item for item in inputs if item.exists and item.input_kind in {"nifti", "nifti_mask_directory"}]
    if not display:
        display = list(inputs)
    columns = max(1, min(5, len(display)))
    fig, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.8))
    if columns == 1:
        axes = [axes]
    fig.suptitle(
        f"Patient Input Adapter: {result.patient_id}\n{result.overall_status.replace('_', ' ')}",
        fontsize=13,
        fontweight="bold",
    )
    for ax, item in zip(axes, display):
        ax.axis("off")
        label_like = item.role in {"organ_seg", "gi_seg", "vessel_seg"}
        loaded = _load_slice(Path(item.staged_path), label_like)
        if loaded is None:
            ax.text(0.5, 0.55, item.role, ha="center", va="center", fontsize=12, fontweight="bold")
            ax.text(0.5, 0.42, item.input_kind, ha="center", va="center", fontsize=9)
        else:
            array, mode = loaded
            cmap = "tab20" if mode == "label" else "gray"
            ax.imshow(np.rot90(array), cmap=cmap, interpolation="nearest")
        ax.set_title(f"{item.role}\n{item.geometry_status}", fontsize=9)
    for ax in axes[len(display) :]:
        ax.axis("off")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, result: PatientPhantomAdapterResult) -> None:
    lines = [
        "# Patient Phantom Input Adapter",
        "",
        f"Case ID: `{result.case_id}`",
        f"Patient/profile ID: `{result.patient_id}`",
        f"Adaptation mode: `{result.adaptation_mode}`",
        "",
        "## Readiness",
        "",
        f"- Primary CT: `{result.primary_ct_status}`",
        f"- Anatomy adaptation: `{result.anatomy_adaptation_status}`",
        f"- Vascular adaptation: `{result.vascular_adaptation_status}`",
        f"- RT readiness: `{result.rt_readiness_status}`",
        f"- Overall: `{result.overall_status}`",
        "",
        "## Input QA",
        "",
        "| role | kind | status | geometry | shape | spacing mm | labels | nonzero % |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in result.inputs:
        shape = "x".join(str(value) for value in item.shape) or "n/a"
        spacing = "x".join(f"{value:.3g}" for value in item.spacing_mm) or "n/a"
        labels = "" if item.unique_label_count is None else str(item.unique_label_count)
        nonzero = "" if item.nonzero_fraction is None else f"{100.0 * item.nonzero_fraction:.2f}"
        lines.append(
            f"| {item.role} | {item.input_kind} | {item.status} | {item.geometry_status} | "
            f"{shape} | {spacing} | {labels} | {nonzero} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- CT drives patient body shape, density/HU handling, RT grid geometry, and registration reference.",
            "- Organ segmentation lets the phantom build patient-adapted material labels; if it is not on the CT grid, it must be registered/resampled first.",
            "- CTA/CTV scans are accepted as vascular source images, but vessel masks or a segmentation step are still needed before graph replacement.",
            "- Vessel segmentations can replace or deform the vascular scaffold once they are registered to the CT grid.",
            "",
            "## Recommended Commands",
            "",
        ]
    )
    for command in result.recommended_commands:
        lines.append(f"```bash\n{command}\n```")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Manifest YAML: `{result.manifest_yaml_path}`",
            f"- Input QA CSV: `{result.input_qa_csv_path}`",
            f"- Preview PNG: `{result.preview_png_path}`",
            f"- Report: `{result.report_path}`",
            "",
            "## Limitations",
            "",
            "- This intake package is research software and is not validated for patient care.",
            "- DICOM directories are recorded but should be converted to NIfTI before the current build pipeline can run full image QA.",
            "- Patient-specific vascular physiology still requires calibrated boundary conditions and validation against measured flow/pressure data.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_patient_phantom_adapter(
    *,
    input_ct_path: str | Path | None = None,
    input_cta_path: str | Path | None = None,
    input_ctv_path: str | Path | None = None,
    organ_seg_path: str | Path | None = None,
    gi_seg_path: str | Path | None = None,
    vessel_seg_path: str | Path | None = None,
    patient_id: str = "patient_demo",
    case_id: str | None = None,
    adaptation_mode: str = "hybrid",
    output_dir: str | Path = "outputs/digital/patient_input_adapter",
    organ_labelmap_path: str | Path = "configs/labelmaps/ct_org.yaml",
    gi_labelmap_path: str | Path = "configs/labelmaps/gi_tract.yaml",
    materials_path: str | Path = "configs/materials.yaml",
    approved_set_manifest_path: str | Path | None = None,
    baseline_graph_path: str | Path | None = None,
    baseline_combined_spec_path: str | Path | None = None,
    target_height_cm: float | None = None,
    target_weight_kg: float | None = None,
    target_bmi: float | None = None,
    target_waist_cm: float | None = None,
    copy_inputs: bool = False,
    report_path: str | Path | None = "outputs/reports/patient_phantom_input_adapter_stage001.md",
) -> PatientPhantomAdapterResult:
    output = Path(output_dir)
    resolved_case_id = case_id or f"{_slug(patient_id)}_patient_adapter"
    supplied_inputs = [input_ct_path, input_cta_path, input_ctv_path, organ_seg_path, gi_seg_path, vessel_seg_path]
    if not any(path is not None and str(path) != "" for path in supplied_inputs):
        raise ValueError("At least one patient input path must be supplied")

    ct = _summarize_input(
        role="ct",
        source_path=input_ct_path,
        output_dir=output,
        copy_inputs=copy_inputs,
        label_like=False,
        ct_summary=None,
    )
    ct_summary = None if ct is None else _serialise_input(ct)
    # Keep only the geometry keys needed by _geometry_status.
    ct_geometry = None
    if ct is not None:
        ct_geometry = {
            "shape": ct.shape,
            "spacing_mm": ct.spacing_mm,
            "affine": ct.affine,
        }

    inputs_raw = [
        ct,
        _summarize_input(
            role="cta",
            source_path=input_cta_path,
            output_dir=output,
            copy_inputs=copy_inputs,
            label_like=False,
            ct_summary=ct_geometry,
        ),
        _summarize_input(
            role="ctv",
            source_path=input_ctv_path,
            output_dir=output,
            copy_inputs=copy_inputs,
            label_like=False,
            ct_summary=ct_geometry,
        ),
        _summarize_input(
            role="organ_seg",
            source_path=organ_seg_path,
            output_dir=output,
            copy_inputs=copy_inputs,
            label_like=True,
            ct_summary=ct_geometry,
        ),
        _summarize_input(
            role="gi_seg",
            source_path=gi_seg_path,
            output_dir=output,
            copy_inputs=copy_inputs,
            label_like=True,
            ct_summary=ct_geometry,
        ),
        _summarize_input(
            role="vessel_seg",
            source_path=vessel_seg_path,
            output_dir=output,
            copy_inputs=copy_inputs,
            label_like=True,
            ct_summary=ct_geometry,
        ),
    ]
    inputs = tuple(item for item in inputs_raw if item is not None)

    primary = _primary_ct_status(inputs)
    anatomy = _anatomy_status(inputs)
    vascular = _vascular_status(inputs)
    rt = _rt_status(primary, anatomy)
    overall = _overall_status(primary, anatomy, vascular)
    active_baseline = resolve_stage007_active_baseline()
    resolved_baseline_graph = (
        str(baseline_graph_path)
        if baseline_graph_path is not None and str(baseline_graph_path) != ""
        else active_baseline.graph_path
    )
    resolved_baseline_combined_spec = (
        str(baseline_combined_spec_path)
        if baseline_combined_spec_path is not None and str(baseline_combined_spec_path) != ""
        else active_baseline.voxelized_spec_path
    )
    commands = _recommended_commands(
        case_id=resolved_case_id,
        patient_id=patient_id,
        inputs=inputs,
        organ_labelmap_path=organ_labelmap_path,
        gi_labelmap_path=gi_labelmap_path,
        materials_path=materials_path,
        approved_set_manifest_path=approved_set_manifest_path,
        baseline_graph_path=resolved_baseline_graph,
        baseline_combined_spec_path=resolved_baseline_combined_spec,
        target_height_cm=target_height_cm,
        target_weight_kg=target_weight_kg,
        target_bmi=target_bmi,
        target_waist_cm=target_waist_cm,
    )

    slug = _slug(patient_id)
    manifest = output / f"{resolved_case_id}_{slug}_patient_input_manifest_v001.yaml"
    input_csv = output / f"{resolved_case_id}_{slug}_patient_input_qa_v001.csv"
    preview = output / f"{resolved_case_id}_{slug}_patient_input_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{resolved_case_id}_{slug}_patient_input_adapter_report_v001.md"

    notes = [
        "patient_adapter_accepts_ct_cta_ctv_organ_seg_and_vessel_seg_inputs",
        "optional_gi_seg_replaces_synthetic_bowel_colon_small_intestine_placeholders_when_co_registered",
        "adapter_performs_intake_QA_and_build_planning_not_full_clinical_registration",
        "all_patient_specific_outputs_require_registration_and_segmentation_QA_before_research_use",
    ]
    if any(item.geometry_status == "registration_required_to_ct_grid" for item in inputs):
        notes.append("one_or_more_inputs_require_registration_to_primary_ct_grid")
    if vascular == "template_vascular_graph_only":
        notes.append("no_patient_vascular_input_supplied_template_vascular_scaffold_would_be_used")
    if baseline_graph_path is None and active_baseline.graph_path is not None:
        notes.append("baseline_graph_auto_resolved_from_stage007_active_baseline")
    if baseline_combined_spec_path is None and active_baseline.voxelized_spec_path is not None:
        notes.append("baseline_reference_spec_auto_resolved_from_stage007_active_voxelized_spec")

    result = PatientPhantomAdapterResult(
        case_id=resolved_case_id,
        patient_id=patient_id,
        adaptation_mode=adaptation_mode,
        output_dir=str(output),
        manifest_yaml_path=str(manifest),
        input_qa_csv_path=str(input_csv),
        preview_png_path=str(preview),
        report_path=str(report),
        input_count=len(inputs),
        primary_ct_status=primary,
        anatomy_adaptation_status=anatomy,
        vascular_adaptation_status=vascular,
        rt_readiness_status=rt,
        overall_status=overall,
        inputs=inputs,
        recommended_commands=commands,
        notes=tuple(notes),
    )
    _write_input_csv(input_csv, inputs)
    _write_preview(preview, inputs, result)
    _write_manifest(
        manifest,
        result,
        organ_labelmap_path=organ_labelmap_path,
        gi_labelmap_path=gi_labelmap_path,
        materials_path=materials_path,
        approved_set_manifest_path=approved_set_manifest_path,
        baseline_graph_path=resolved_baseline_graph,
        baseline_combined_spec_path=resolved_baseline_combined_spec,
        target_height_cm=target_height_cm,
        target_weight_kg=target_weight_kg,
        target_bmi=target_bmi,
        target_waist_cm=target_waist_cm,
        copy_inputs=copy_inputs,
    )
    _write_report(report, result)
    return result


def format_patient_phantom_adapter_result(result: PatientPhantomAdapterResult) -> str:
    return "\n".join(
        [
            "Patient Phantom Input Adapter",
            f"Case ID: {result.case_id}",
            f"Patient/profile ID: {result.patient_id}",
            f"Inputs accepted: {result.input_count}",
            f"Primary CT: {result.primary_ct_status}",
            f"Anatomy adaptation: {result.anatomy_adaptation_status}",
            f"Vascular adaptation: {result.vascular_adaptation_status}",
            f"RT readiness: {result.rt_readiness_status}",
            f"Overall: {result.overall_status}",
            f"Manifest YAML: {result.manifest_yaml_path}",
            f"Input QA CSV: {result.input_qa_csv_path}",
            f"Preview PNG: {result.preview_png_path}",
        ]
    )
