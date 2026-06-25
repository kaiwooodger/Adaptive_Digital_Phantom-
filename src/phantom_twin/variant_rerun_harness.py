from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import stat
from typing import Any

import numpy as np
import yaml


GROUP_LABELS = {
    "body": tuple(range(1, 16)),
    "lungs": (8,),
    "liver": (6,),
    "kidneys": (7,),
    "bone": (10, 11),
    "vessel_wall": (13,),
    "vascular_fluid": (14, 15),
}


@dataclass(frozen=True)
class VariantPreflightMetric:
    metric: str
    value: str
    status: str
    notes: str


@dataclass(frozen=True)
class VariantRerunHarnessResult:
    case_id: str
    variant_id: str
    output_dir: str
    variant_labels_path: str
    baseline_combined_spec_path: str
    variant_combined_spec_path: str
    harness_yaml_path: str
    preflight_csv_path: str
    commands_script_path: str
    report_path: str
    material_maps_staged: bool
    material_map_paths: tuple[str, ...]
    rt_ready: bool
    flow_ready: bool
    preflight_metrics: tuple[VariantPreflightMetric, ...]
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import nibabel as nib  # type: ignore
        from scipy import ndimage  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Variant rerun harness requires nibabel and scipy.") from exc
    return nib, ndimage


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(raw_path: str | Path, reference_path: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    if reference_path is not None:
        return reference_path.parent / path
    return path


def _find_variant(approved_set: dict[str, Any], variant_id: str) -> dict[str, Any]:
    variants = approved_set.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Approved set manifest must contain variants")
    for item in variants:
        if isinstance(item, dict) and str(item.get("variant_id")) == variant_id:
            return item
    raise ValueError(f"Variant {variant_id!r} was not found in approved set manifest")


def _source_combined_spec(approved_set: dict[str, Any], approved_set_path: Path, override: str | Path | None) -> Path:
    if override is not None:
        return _resolve_path(override)
    atlas_path_raw = approved_set.get("source_atlas_spec")
    if not atlas_path_raw:
        raise ValueError("baseline_combined_spec_path is required when source_atlas_spec is missing")
    atlas_path = _resolve_path(str(atlas_path_raw), approved_set_path)
    atlas = _load_yaml(atlas_path)
    combined = atlas.get("source_combined_spec")
    if not combined:
        raise ValueError("Atlas spec does not contain source_combined_spec")
    return _resolve_path(str(combined), atlas_path)


def _spacing(image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _voxel_volume_cm3(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(spacing_mm) / 1000.0)


def _group_volume_cm3(labels: np.ndarray, group_id: str, voxel_volume_cm3: float) -> float:
    label_ids = GROUP_LABELS[group_id]
    mask = np.isin(labels, label_ids) if len(label_ids) > 1 else labels == label_ids[0]
    return float(mask.sum() * voxel_volume_cm3)


def _component_count(labels: np.ndarray, group_id: str) -> int:
    _, ndimage = _import_dependencies()
    label_ids = GROUP_LABELS[group_id]
    mask = np.isin(labels, label_ids) if len(label_ids) > 1 else labels == label_ids[0]
    _, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    return int(count)


def _write_nifti(path: Path, data: np.ndarray, reference_image) -> None:
    nib, _ = _import_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _stage_material_maps(
    output_dir: Path,
    case_id: str,
    labels: np.ndarray,
    label_image,
    regions: list[dict[str, Any]],
) -> tuple[str, str, str]:
    hu = np.zeros(labels.shape, dtype=np.float32)
    density = np.zeros(labels.shape, dtype=np.float32)
    red = np.zeros(labels.shape, dtype=np.float32)
    for region in regions:
        label_index = int(region["index"])
        mask = labels == label_index
        if not np.any(mask):
            continue
        hu[mask] = float(region["target_hu_midpoint"])
        density[mask] = float(region["mass_density_g_cm3"])
        red[mask] = float(region["relative_electron_density"])
    material_dir = output_dir / "material_maps"
    hu_path = material_dir / f"{case_id}_variant_synthetic_hu_blood_v001.nii.gz"
    density_path = material_dir / f"{case_id}_variant_mass_density_blood_v001.nii.gz"
    red_path = material_dir / f"{case_id}_variant_relative_electron_density_blood_v001.nii.gz"
    _write_nifti(hu_path, hu, label_image)
    _write_nifti(density_path, density, label_image)
    _write_nifti(red_path, red, label_image)
    return str(hu_path), str(density_path), str(red_path)


def _baseline_shape_spacing(combined_spec: dict[str, Any], combined_spec_path: Path) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None]:
    outputs = combined_spec.get("outputs", {})
    if not isinstance(outputs, dict) or not outputs.get("blood_material_labels"):
        return None, None
    labels_path = _resolve_path(str(outputs["blood_material_labels"]), combined_spec_path)
    if not labels_path.exists():
        return None, None
    nib, _ = _import_dependencies()
    image = nib.load(str(labels_path))
    return tuple(int(value) for value in image.shape), _spacing(image)


def _metric(metric: str, value: Any, status: str = "ok", notes: str = "") -> VariantPreflightMetric:
    return VariantPreflightMetric(metric=metric, value=str(value), status=status, notes=notes)


def _write_preflight_csv(path: Path, metrics: tuple[VariantPreflightMetric, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value", "status", "notes"])
        for item in metrics:
            writer.writerow([item.metric, item.value, item.status, item.notes])


def _variant_combined_spec(
    baseline_spec: dict[str, Any],
    case_id: str,
    variant_id: str,
    variant_labels_path: str,
    staged_maps: tuple[str, str, str] | None,
) -> dict[str, Any]:
    payload = dict(baseline_spec)
    outputs = dict(baseline_spec.get("outputs", {}))
    outputs["blood_material_labels"] = variant_labels_path
    if staged_maps is not None:
        hu_path, density_path, red_path = staged_maps
        outputs["blood_synthetic_hu"] = hu_path
        outputs["blood_mass_density_g_cm3"] = density_path
        outputs["blood_relative_electron_density"] = red_path
        outputs["contrast_material_labels"] = variant_labels_path
        outputs["contrast_synthetic_hu"] = hu_path
        outputs["contrast_mass_density_g_cm3"] = density_path
        outputs["contrast_relative_electron_density"] = red_path
    else:
        for key in (
            "blood_synthetic_hu",
            "blood_mass_density_g_cm3",
            "blood_relative_electron_density",
            "contrast_material_labels",
            "contrast_synthetic_hu",
            "contrast_mass_density_g_cm3",
            "contrast_relative_electron_density",
        ):
            outputs.pop(key, None)
    payload["case_id"] = case_id
    payload["outputs"] = outputs
    notes = list(payload.get("notes", []))
    notes.extend(
        [
            f"variant_rerun_harness_variant_id={variant_id}",
            "variant_combined_spec_references_existing_approved_pca_material_labels",
        ]
    )
    if staged_maps is not None:
        notes.append("contrast_outputs_alias_blood_maps_for_stage001_variant_rerun_harness")
    else:
        notes.append("material_maps_not_staged_RT_package_can_synthesize_maps_from_material_labels")
    payload["notes"] = notes
    return payload


def _write_commands(
    path: Path,
    case_id: str,
    variant_combined_spec_path: str,
    flow_model_spec_path: str | None,
) -> tuple[str, ...]:
    rt_qa_dir = f"outputs/radiotherapy/variant_reruns/{case_id}/qa_package"
    rt_plan_dir = f"outputs/radiotherapy/variant_reruns/{case_id}/planning_bundle"
    gamma_dir = f"outputs/radiotherapy/variant_reruns/{case_id}/dose_gamma_qa"
    rt_package_spec = f"{rt_qa_dir}/{case_id}_radiotherapy_qa_package_spec_v001.yaml"
    planning_spec = f"{rt_plan_dir}/{case_id}_rt_planning_bundle_spec_v001.yaml"
    pymedphys_config = f"{rt_plan_dir}/{case_id}_pymedphys_dose_eval_config_v001.yaml"
    flow_arg = f" --coupled-flow-model {flow_model_spec_path}" if flow_model_spec_path else ""
    commands = (
        f"PYTHONPATH=src /opt/anaconda3/bin/python -m phantom_twin.cli build-radiotherapy-qa-package --combined-spec {variant_combined_spec_path} --case-id {case_id} --output-dir {rt_qa_dir} --scenario blood --report outputs/reports/{case_id}_radiotherapy_qa_package_stage001.md",
        f"PYTHONPATH=src /opt/anaconda3/bin/python -m phantom_twin.cli build-rt-planning-bundle --rt-package-spec {rt_package_spec}{flow_arg} --case-id {case_id} --output-dir {rt_plan_dir} --skip-dicom --report outputs/reports/{case_id}_rt_planning_bundle_stage001.md",
        f"PYTHONPATH=src /opt/anaconda3/bin/python -m phantom_twin.cli build-dose-gamma-qa --pymedphys-eval-config {pymedphys_config} --case-id {case_id} --output-dir {gamma_dir} --random-subset 10000 --report outputs/reports/{case_id}_dose_gamma_qa_stage001.md",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {Path.cwd()}",
        "",
        "# Variant-specific RT rerun commands. DICOM export is skipped to conserve disk space.",
        "# If this harness was generated with --skip-material-maps, the RT package will synthesize",
        "# HU, density, and relative-electron-density maps from material labels and region metadata.",
        *commands,
        "",
        "# Flow note: this harness attaches the existing coupled flow model as a baseline reference.",
        "# A true variant-specific flow rerun will require vascular graph deformation/re-voxelization.",
    ]
    path.write_text("\n".join(body) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return commands


def _write_harness_yaml(path: Path, result: VariantRerunHarnessResult, commands: tuple[str, ...]) -> None:
    payload = {
        "case_id": result.case_id,
        "variant_id": result.variant_id,
        "package_type": "variant_specific_rerun_harness",
        "variant_labels": result.variant_labels_path,
        "baseline_combined_spec": result.baseline_combined_spec_path,
        "variant_combined_spec": result.variant_combined_spec_path,
        "material_maps_staged": result.material_maps_staged,
        "material_map_paths": list(result.material_map_paths),
        "rt_ready": result.rt_ready,
        "flow_ready": result.flow_ready,
        "outputs": {
            "preflight_csv": result.preflight_csv_path,
            "commands_script": result.commands_script_path,
            "report": result.report_path,
        },
        "commands": list(commands),
        "preflight": [
            {"metric": item.metric, "value": item.value, "status": item.status, "notes": item.notes}
            for item in result.preflight_metrics
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VariantRerunHarnessResult, commands: tuple[str, ...]) -> str:
    lines = [
        "# Variant-Specific Rerun Harness Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        f"Variant ID: `{result.variant_id}`",
        "",
        "## Summary",
        "",
        f"- RT-ready material maps staged: {'yes' if result.material_maps_staged else 'no'}",
        f"- RT rerun ready: {'yes' if result.rt_ready else 'no'}",
        f"- Flow rerun ready: {'yes' if result.flow_ready else 'no'}",
        "- DICOM export is intentionally skipped in generated commands to conserve disk space.",
        "",
        "## Outputs",
        "",
        f"- Variant combined spec: `{Path(result.variant_combined_spec_path).name}`",
        f"- Harness YAML: `{Path(result.harness_yaml_path).name}`",
        f"- Preflight CSV: `{Path(result.preflight_csv_path).name}`",
        f"- Commands script: `{Path(result.commands_script_path).name}`",
        "",
        "## Preflight",
        "",
        "| metric | value | status | notes |",
        "| --- | --- | --- | --- |",
    ]
    for item in result.preflight_metrics:
        lines.append(f"| {item.metric} | {item.value} | {item.status} | {item.notes} |")
    lines.extend(["", "## Rerun Commands", ""])
    for command in commands:
        lines.append(f"```bash\n{command}\n```")
    lines.extend(["", "## Notes"])
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_variant_rerun_harness(
    approved_set_manifest_path: str | Path,
    variant_id: str = "mode01_neg",
    baseline_combined_spec_path: str | Path | None = None,
    flow_model_spec_path: str | Path | None = None,
    output_dir: str | Path = "outputs/experiments/variant_rerun_harness",
    case_id: str | None = None,
    stage_material_maps: bool = True,
    report_path: str | Path | None = None,
) -> VariantRerunHarnessResult:
    approved_path = Path(approved_set_manifest_path)
    approved_set = _load_yaml(approved_path)
    variant = _find_variant(approved_set, variant_id)
    resolved_case_id = case_id or f"{approved_set.get('case_id', 'approved_pca')}_{variant_id}"
    output = Path(output_dir) / variant_id
    variant_labels = _resolve_path(str(variant["material_labels"]), approved_path)
    if not variant_labels.exists():
        raise FileNotFoundError(f"Variant material-label NIfTI is missing: {variant_labels}")

    combined_path = _source_combined_spec(approved_set, approved_path, baseline_combined_spec_path)
    combined_spec = _load_yaml(combined_path)
    regions = list(combined_spec.get("regions", []))
    if not regions:
        raise ValueError("Baseline combined spec must contain regions")

    nib, _ = _import_dependencies()
    label_image = nib.load(str(variant_labels))
    labels = np.rint(np.asanyarray(label_image.dataobj)).astype(np.int16)
    spacing_mm = _spacing(label_image)
    voxel_volume = _voxel_volume_cm3(spacing_mm)
    baseline_shape, baseline_spacing = _baseline_shape_spacing(combined_spec, combined_path)
    shape_match = baseline_shape == tuple(labels.shape) if baseline_shape is not None else False
    spacing_match = baseline_spacing == spacing_mm if baseline_spacing is not None else False

    vascular_components = _component_count(labels, "vascular_fluid")
    staged_maps = _stage_material_maps(output, resolved_case_id, labels, label_image, regions) if stage_material_maps else None
    variant_spec = _variant_combined_spec(
        baseline_spec=combined_spec,
        case_id=resolved_case_id,
        variant_id=variant_id,
        variant_labels_path=str(variant_labels),
        staged_maps=staged_maps,
    )
    variant_spec_path = output / f"{resolved_case_id}_variant_combined_spec_v001.yaml"
    variant_spec_path.parent.mkdir(parents=True, exist_ok=True)
    variant_spec_path.write_text(yaml.safe_dump(variant_spec, sort_keys=False))

    rt_ready = vascular_components == 1 and _group_volume_cm3(labels, "bone", voxel_volume) > 0.0
    flow_ready = vascular_components == 1 and shape_match and spacing_match
    preflight = (
        _metric("variant_labels_path", str(variant_labels), "ok", "existing approved PCA variant label NIfTI"),
        _metric("shape", "x".join(str(value) for value in labels.shape), "ok"),
        _metric("spacing_mm", ",".join(f"{value:.6g}" for value in spacing_mm), "ok"),
        _metric("baseline_shape_match", shape_match, "ok" if shape_match else "warn", f"baseline_shape={baseline_shape}"),
        _metric("baseline_spacing_match", spacing_match, "ok" if spacing_match else "warn", f"baseline_spacing={baseline_spacing}"),
        _metric("body_volume_cm3", f"{_group_volume_cm3(labels, 'body', voxel_volume):.6f}", "ok"),
        _metric("lungs_volume_cm3", f"{_group_volume_cm3(labels, 'lungs', voxel_volume):.6f}", "ok"),
        _metric("liver_volume_cm3", f"{_group_volume_cm3(labels, 'liver', voxel_volume):.6f}", "ok"),
        _metric("kidneys_volume_cm3", f"{_group_volume_cm3(labels, 'kidneys', voxel_volume):.6f}", "ok"),
        _metric("bone_volume_cm3", f"{_group_volume_cm3(labels, 'bone', voxel_volume):.6f}", "ok"),
        _metric("vessel_wall_volume_cm3", f"{_group_volume_cm3(labels, 'vessel_wall', voxel_volume):.6f}", "ok"),
        _metric("vascular_fluid_volume_cm3", f"{_group_volume_cm3(labels, 'vascular_fluid', voxel_volume):.6f}", "ok"),
        _metric("vascular_fluid_components", vascular_components, "ok" if vascular_components == 1 else "warn"),
        _metric("material_maps_staged", staged_maps is not None, "ok" if staged_maps is not None else "warn"),
        _metric("rt_ready", rt_ready, "ok" if rt_ready else "warn", "RT commands require bone/vascular labels; maps can be synthesized from labels"),
        _metric("flow_ready", flow_ready, "warn", "baseline flow graph can be attached, but true variant flow needs graph deformation/re-voxelization"),
    )
    preflight_path = output / f"{resolved_case_id}_variant_preflight_metrics_v001.csv"
    commands_path = output / f"{resolved_case_id}_variant_rerun_commands_v001.sh"
    harness_path = output / f"{resolved_case_id}_variant_rerun_harness_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{resolved_case_id}_variant_rerun_harness_report_v001.md"
    flow_model = None if flow_model_spec_path is None else str(_resolve_path(flow_model_spec_path))
    commands = _write_commands(commands_path, resolved_case_id, str(variant_spec_path), flow_model)
    notes = (
        "harness_can_stage_small_material_maps_or_run_disk_light_with_label_derived_rt_maps",
        "rt_planning_command_uses_skip_dicom_to_avoid_regenerating_large_DICOM_exports",
        "flow_model_is_attached_as_baseline_reference_until_variant_specific_graph_deformation_is_implemented",
        "mode01_neg_was_selected_because_it_ranked_highest_in_anatomy_impact_triage",
    )
    result = VariantRerunHarnessResult(
        case_id=resolved_case_id,
        variant_id=variant_id,
        output_dir=str(output),
        variant_labels_path=str(variant_labels),
        baseline_combined_spec_path=str(combined_path),
        variant_combined_spec_path=str(variant_spec_path),
        harness_yaml_path=str(harness_path),
        preflight_csv_path=str(preflight_path),
        commands_script_path=str(commands_path),
        report_path=str(report),
        material_maps_staged=staged_maps is not None,
        material_map_paths=tuple(staged_maps or ()),
        rt_ready=rt_ready,
        flow_ready=flow_ready,
        preflight_metrics=preflight,
        notes=notes,
    )
    _write_preflight_csv(preflight_path, preflight)
    _write_harness_yaml(harness_path, result, commands)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, commands))
    return result


def format_variant_rerun_harness_result(result: VariantRerunHarnessResult) -> str:
    lines = [
        "# Variant Rerun Harness",
        "",
        f"Case ID: `{result.case_id}`",
        f"Variant ID: `{result.variant_id}`",
        f"RT ready: {'yes' if result.rt_ready else 'no'}",
        f"Flow baseline-compatible: {'yes' if result.flow_ready else 'no'}",
        "",
        "## Outputs",
        "",
        f"- Variant combined spec: `{result.variant_combined_spec_path}`",
        f"- Harness YAML: `{result.harness_yaml_path}`",
        f"- Preflight CSV: `{result.preflight_csv_path}`",
        f"- Commands script: `{result.commands_script_path}`",
        f"- Report: `{result.report_path}`",
    ]
    return "\n".join(lines)
