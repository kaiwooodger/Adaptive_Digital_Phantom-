from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .abdominal_organs import augment_btcv_abdominal_organs
from .combined import _combined_regions, _property_volume, _write_nifti
from .cta_vascular_graph import build_cta_derived_vascular_graph
from .flow_1d import build_flow_1d_model
from .flow_boundary import build_flow_boundary_package
from .flow_coupled import build_coupled_pulsatile_flow_model
from .materials import load_material_library
from .radiotherapy import build_radiotherapy_qa_package
from .rt_planning import build_rt_planning_bundle
from .stage007_baseline import resolve_stage007_active_baseline
from .torso import build_digital_torso_phantom
from .vascular_voxelize import voxelize_vascular_network
from .vessel_anatomy_correction import correct_vessel_bone_conflicts
from .vessel_anatomy_validation import validate_vessel_organ_anatomy
from .vessel_radius_validation import validate_vessel_radius_anatomy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CT_ORG_LABELMAP = PROJECT_ROOT / "configs" / "labelmaps" / "ct_org.yaml"
DEFAULT_GI_LABELMAP = PROJECT_ROOT / "configs" / "labelmaps" / "gi_tract.yaml"


@dataclass(frozen=True)
class PatientBuildStep:
    step_id: str
    status: str
    primary_output_path: str | None
    report_path: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PatientTorsoBuildResult:
    case_id: str
    output_dir: str
    material_label_path: str
    density_path: str
    relative_electron_density_path: str
    synthetic_hu_path: str
    body_mask_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    body_volume_cm3: float
    label_mode: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PatientPhantomBuildResult:
    case_id: str
    patient_id: str
    output_dir: str
    build_manifest_yaml_path: str
    report_path: str
    source_patient_manifest_path: str
    overall_status: str
    dry_run: bool
    torso_spec_path: str | None
    vascular_graph_path: str | None
    voxelized_spec_path: str | None
    flow_boundary_config_path: str | None
    flow_1d_model_path: str | None
    coupled_flow_model_path: str | None
    rt_package_spec_path: str | None
    rt_planning_spec_path: str | None
    steps: tuple[PatientBuildStep, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "patient"


def _input_by_role(manifest: dict[str, Any], role: str) -> dict[str, Any] | None:
    for item in manifest.get("inputs", []):
        if isinstance(item, dict) and str(item.get("role")) == role:
            return item
    return None


def _input_path(manifest: dict[str, Any], role: str) -> str | None:
    item = _input_by_role(manifest, role)
    if item is None or not bool(item.get("exists", False)):
        return None
    path = item.get("staged_path") or item.get("source_path")
    return None if path is None or str(path) == "" else str(path)


def _input_geometry(manifest: dict[str, Any], role: str) -> str | None:
    item = _input_by_role(manifest, role)
    if item is None:
        return None
    return str(item.get("geometry_status", ""))


def _config_path(manifest: dict[str, Any], key: str, override: str | Path | None = None) -> str | None:
    if override is not None and str(override) != "":
        return str(override)
    config = manifest.get("configuration", {})
    if not isinstance(config, dict):
        return None
    value = config.get(key)
    return None if value is None or str(value) == "" else str(value)


def _status(manifest: dict[str, Any], key: str) -> str:
    status = manifest.get("status", {})
    if not isinstance(status, dict):
        return ""
    return str(status.get(key, ""))


def _import_image_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Patient material-label torso staging requires matplotlib and nibabel.") from exc
    return plt, nib


def _organ_label_mode(manifest: dict[str, Any], requested_mode: str) -> str:
    if requested_mode not in {"auto", "ct-org", "material", "btcv"}:
        raise ValueError("organ_label_mode must be one of: auto, ct-org, material, btcv")
    if requested_mode != "auto":
        return requested_mode
    configuration = manifest.get("configuration", {})
    if isinstance(configuration, dict) and "btcv" in str(configuration.get("organ_labelmap", "")).lower():
        return "btcv"
    organ = _input_by_role(manifest, "organ_seg")
    if organ is None:
        return "ct-org"
    max_value = organ.get("max_value")
    unique = organ.get("unique_labels_sample", [])
    try:
        values = {int(round(float(value))) for value in unique}
    except (TypeError, ValueError):
        values = set()
    if max_value is not None and float(max_value) > 6.0 and values.issubset(set(range(16))):
        return "material"
    return "ct-org"


def _render_material_label_preview(path: Path, labels: np.ndarray, source_ct: np.ndarray, body: np.ndarray) -> None:
    plt, _ = _import_image_dependencies()
    coords = np.argwhere(body)
    if coords.size:
        x_index, y_index, z_index = (int(round(float(np.median(coords[:, axis])))) for axis in range(3))
    else:
        x_index, y_index, z_index = (value // 2 for value in labels.shape)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=160)
    for ax in axes.ravel():
        ax.axis("off")
    ct_views = [
        (source_ct[:, :, z_index], f"CT axial z={z_index}"),
        (source_ct[:, y_index, :], f"CT coronal y={y_index}"),
        (source_ct[x_index, :, :], f"CT sagittal x={x_index}"),
    ]
    label_views = [
        (labels[:, :, z_index], "Material axial"),
        (labels[:, y_index, :], "Material coronal"),
        (labels[x_index, :, :], "Material sagittal"),
    ]
    for ax, (view, title) in zip(axes[0], ct_views):
        ax.imshow(np.rot90(np.clip(view, -1000, 1000)), cmap="gray", vmin=-1000, vmax=1000)
        ax.set_title(title, fontsize=9)
    for ax, (view, title) in zip(axes[1], label_views):
        ax.imshow(np.rot90(view), cmap="tab20", vmin=0, vmax=15, interpolation="nearest")
        ax.set_title(title, fontsize=9)
    fig.suptitle("Patient Torso From Project Material Labels", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_material_torso_spec(path: Path, result: PatientTorsoBuildResult, regions: tuple[Any, ...]) -> None:
    payload = {
        "case_id": result.case_id,
        "coordinate_units": "mm",
        "volume_units": "cm3",
        "label_mode": result.label_mode,
        "outputs": {
            "material_label_map": result.material_label_path,
            "mass_density_g_cm3": result.density_path,
            "relative_electron_density": result.relative_electron_density_path,
            "synthetic_hu": result.synthetic_hu_path,
            "body_mask": result.body_mask_path,
            "preview_png": result.preview_png_path,
        },
        "regions": [
            {
                "index": region.index,
                "name": region.name,
                "material_id": region.material_id,
                "target_hu_midpoint": region.target_hu_midpoint,
                "mass_density_g_cm3": region.mass_density_g_cm3,
                "relative_electron_density": region.relative_electron_density,
                "color": region.color,
            }
            for region in regions
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_material_torso_report(path: Path, result: PatientTorsoBuildResult) -> None:
    lines = [
        "# Patient Torso Material-Label Stage",
        "",
        f"Case ID: `{result.case_id}`",
        f"Label mode: `{result.label_mode}`",
        f"Body volume: {result.body_volume_cm3:.2f} cm3",
        "",
        "## Outputs",
        "",
        f"- Material labels: `{result.material_label_path}`",
        f"- Mass density: `{result.density_path}`",
        f"- Relative electron density: `{result.relative_electron_density_path}`",
        f"- Synthetic HU: `{result.synthetic_hu_path}`",
        f"- Body mask: `{result.body_mask_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        f"- Spec YAML: `{result.spec_yaml_path}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _build_material_label_torso(
    ct_path: str | Path,
    material_labels_path: str | Path,
    materials_path: str | Path,
    output_dir: str | Path,
    case_id: str,
    report_path: str | Path,
) -> PatientTorsoBuildResult:
    _, nib = _import_image_dependencies()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ct_image = nib.load(str(ct_path))
    label_image = nib.load(str(material_labels_path))
    source_ct = np.asanyarray(ct_image.dataobj).astype(np.float32)
    labels = np.rint(np.asanyarray(label_image.dataobj)).astype(np.int16)
    if labels.shape != source_ct.shape:
        raise ValueError(f"CT and material-label shapes differ: {source_ct.shape} vs {labels.shape}")
    labels = labels.copy()
    labels[np.isin(labels, (13, 14, 15))] = 4
    body = labels > 0
    spacing = tuple(float(value) for value in label_image.header.get_zooms()[:3])
    voxel_volume_cm3 = float(np.prod(np.asarray(spacing, dtype=float)) / 1000.0)
    library = load_material_library(materials_path)
    regions = _combined_regions(library)
    density = _property_volume(labels, regions, "mass_density_g_cm3")
    red = _property_volume(labels, regions, "relative_electron_density")
    synthetic_hu = _property_volume(labels, regions, "target_hu_midpoint")
    material_path = output / f"{case_id}_torso_material_labels_v001.nii.gz"
    density_path = output / f"{case_id}_torso_mass_density_g_cm3_v001.nii.gz"
    red_path = output / f"{case_id}_torso_relative_electron_density_v001.nii.gz"
    hu_path = output / f"{case_id}_torso_synthetic_hu_v001.nii.gz"
    body_path = output / f"{case_id}_torso_body_mask_v001.nii.gz"
    preview = output / f"{case_id}_torso_material_preview_v001.png"
    spec = output / f"{case_id}_torso_material_spec_v001.yaml"
    report = Path(report_path)
    _write_nifti(material_path, labels, label_image, nib)
    _write_nifti(density_path, density, label_image, nib)
    _write_nifti(red_path, red, label_image, nib)
    _write_nifti(hu_path, synthetic_hu, label_image, nib)
    _write_nifti(body_path, body.astype(np.uint8), label_image, nib)
    _render_material_label_preview(preview, labels, source_ct, body)
    result = PatientTorsoBuildResult(
        case_id=case_id,
        output_dir=str(output),
        material_label_path=str(material_path),
        density_path=str(density_path),
        relative_electron_density_path=str(red_path),
        synthetic_hu_path=str(hu_path),
        body_mask_path=str(body_path),
        preview_png_path=str(preview),
        spec_yaml_path=str(spec),
        report_path=str(report),
        body_volume_cm3=float(body.sum() * voxel_volume_cm3),
        label_mode="material",
        notes=(
            "input_segmentation_detected_as_project_material_label_volume",
            "pre_existing_vascular_material_labels_13_14_15_recast_to_soft_tissue_before_patient_vessel_voxelization",
        ),
    )
    _write_material_torso_spec(spec, result, regions)
    _write_material_torso_report(report, result)
    return result


def _build_btcv_label_torso(
    ct_path: str | Path,
    btcv_labels_path: str | Path,
    materials_path: str | Path,
    output_dir: str | Path,
    case_id: str,
    report_path: str | Path,
    gi_segmentation_path: str | Path | None = None,
    gi_labelmap_path: str | Path | None = None,
) -> Any:
    _, nib = _import_image_dependencies()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_image = nib.load(str(btcv_labels_path))
    labels = np.rint(np.asanyarray(label_image.dataobj)).astype(np.int16)
    converted = np.zeros(labels.shape, dtype=np.int16)
    converted[labels == 6] = 1
    converted[np.isin(labels, (2, 3))] = 4
    converted_path = output / f"{case_id}_btcv_to_ct_org_labels_v001.nii.gz"
    _write_nifti(converted_path, converted, label_image, nib)
    result = build_digital_torso_phantom(
        ct_path=str(ct_path),
        labels_path=str(converted_path),
        labelmap_path=DEFAULT_CT_ORG_LABELMAP,
        materials_path=str(materials_path),
        output_dir=output,
        case_id=case_id,
        report_path=report_path,
    )
    regions = _combined_regions(load_material_library(materials_path))
    abdominal = augment_btcv_abdominal_organs(
        case_id=case_id,
        ct_path=ct_path,
        btcv_labels_path=btcv_labels_path,
        gi_segmentation_path=gi_segmentation_path,
        gi_labelmap_path=gi_labelmap_path,
        material_label_path=result.material_label_path,
        mass_density_path=result.density_path,
        relative_electron_density_path=result.relative_electron_density_path,
        synthetic_hu_path=result.synthetic_hu_path,
        torso_spec_path=result.spec_yaml_path,
        regions=regions,
        output_dir=output / "abdominal_organs",
        report_path=Path("outputs/reports") / f"{case_id}_abdominal_organ_preservation.md",
    )
    Path(result.report_path).write_text(
        Path(result.report_path).read_text()
        + "\n## Organ-Preserving Abdomen\n\n"
        + f"- Abdominal organ QA: `{abdominal.qa_yaml_path}`\n"
        + f"- Preserved organ pass/review/fail: {abdominal.pass_count}/{abdominal.review_count}/{abdominal.fail_count}\n"
        + "- BTCV spleen, stomach, gallbladder, esophagus, pancreas, and adrenal labels were preserved as explicit material labels.\n"
        + f"- Real GI applied targets: `{', '.join(abdominal.gi_real_segmentation_applied_targets) or 'none'}`.\n"
        + "- Missing duodenum, small bowel, colon, rectum, and specific bowel-lumen labels fall back to explicit placeholders.\n"
    )
    return result


def _planned_steps(run_rt: bool, use_patient_vessels: bool, use_template_vessels: bool) -> tuple[PatientBuildStep, ...]:
    steps = [
        PatientBuildStep("torso", "planned", None, None, ("build_patient_torso_from_ct_and_organ_segmentation",)),
        PatientBuildStep(
            "vascular_graph",
            "planned",
            None,
            None,
            (
                "replace_graph_from_patient_vessel_segmentation"
                if use_patient_vessels
                else "use_template_vascular_graph_with_patient_anatomy"
                if use_template_vessels
                else "blocked_until_vessel_input_or_template_permission",
            ),
        ),
        PatientBuildStep("voxelization", "planned", None, None, ("voxelize_vascular_graph_into_patient_torso",)),
        PatientBuildStep("vessel_qa", "planned", None, None, ("run_centerline_and_radius_anatomy_QA",)),
        PatientBuildStep("flow", "planned", None, None, ("run_boundary_steady_and_coupled_pulsatile_flow_models",)),
    ]
    if run_rt:
        steps.append(PatientBuildStep("radiotherapy", "planned", None, None, ("build_RT_QA_package_and_planning_bundle",)))
    return tuple(steps)


def _write_manifest(path: Path, result: PatientPhantomBuildResult) -> None:
    abdominal_organ_qa = None
    abdominal_organ_metrics = None
    gi_lumen_mask = None
    gi_tract_placeholder_labels = None
    real_gi_segmentation_source = None
    real_gi_segmentation_applied_targets = None
    if result.torso_spec_path is not None and Path(result.torso_spec_path).exists():
        try:
            torso_spec = _load_yaml(result.torso_spec_path)
            torso_outputs = torso_spec.get("outputs", {})
            if isinstance(torso_outputs, dict):
                abdominal_organ_qa = torso_outputs.get("abdominal_organ_qa_yaml")
                abdominal_organ_metrics = torso_outputs.get("abdominal_organ_metrics_csv")
                gi_lumen_mask = torso_outputs.get("gi_lumen_mask")
                gi_tract_placeholder_labels = torso_outputs.get("gi_tract_placeholder_labels")
                real_gi_segmentation_source = torso_outputs.get("real_gi_segmentation_source")
                real_gi_segmentation_applied_targets = torso_outputs.get("real_gi_segmentation_applied_targets")
        except Exception:
            abdominal_organ_qa = None
    payload = {
        "case_id": result.case_id,
        "patient_id": result.patient_id,
        "package_type": "patient_phantom_build_executor",
        "source_patient_manifest": result.source_patient_manifest_path,
        "overall_status": result.overall_status,
        "dry_run": result.dry_run,
        "outputs": {
            "build_manifest_yaml": result.build_manifest_yaml_path,
            "report": result.report_path,
            "torso_spec": result.torso_spec_path,
            "vascular_graph": result.vascular_graph_path,
            "voxelized_spec": result.voxelized_spec_path,
            "flow_boundary_config": result.flow_boundary_config_path,
            "flow_1d_model": result.flow_1d_model_path,
            "coupled_flow_model": result.coupled_flow_model_path,
            "rt_package_spec": result.rt_package_spec_path,
            "rt_planning_spec": result.rt_planning_spec_path,
            "abdominal_organ_qa": abdominal_organ_qa,
            "abdominal_organ_metrics": abdominal_organ_metrics,
            "gi_lumen_mask": gi_lumen_mask,
            "gi_tract_placeholder_labels": gi_tract_placeholder_labels,
            "real_gi_segmentation_source": real_gi_segmentation_source,
            "real_gi_segmentation_applied_targets": real_gi_segmentation_applied_targets,
        },
        "steps": [
            {
                "step_id": step.step_id,
                "status": step.status,
                "primary_output_path": step.primary_output_path,
                "report_path": step.report_path,
                "notes": list(step.notes),
            }
            for step in result.steps
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: PatientPhantomBuildResult) -> None:
    lines = [
        "# Patient Phantom Build Executor",
        "",
        f"Case ID: `{result.case_id}`",
        f"Patient/profile ID: `{result.patient_id}`",
        f"Overall status: `{result.overall_status}`",
        f"Dry run: `{result.dry_run}`",
        "",
        "## Step Status",
        "",
        "| step | status | output | report |",
        "| --- | --- | --- | --- |",
    ]
    for step in result.steps:
        output = "" if step.primary_output_path is None else f"`{step.primary_output_path}`"
        report = "" if step.report_path is None else f"`{step.report_path}`"
        lines.append(f"| {step.step_id} | {step.status} | {output} | {report} |")
    lines.extend(
        [
            "",
            "## Build Outputs",
            "",
            f"- Torso spec: `{result.torso_spec_path or 'not_written'}`",
            f"- Vascular graph: `{result.vascular_graph_path or 'not_written'}`",
            f"- Voxelized vascular spec: `{result.voxelized_spec_path or 'not_written'}`",
            f"- Flow boundary config: `{result.flow_boundary_config_path or 'not_written'}`",
            f"- 1D flow model: `{result.flow_1d_model_path or 'not_written'}`",
            f"- Coupled pulsatile flow model: `{result.coupled_flow_model_path or 'not_written'}`",
            f"- RT package spec: `{result.rt_package_spec_path or 'not_written'}`",
            f"- RT planning spec: `{result.rt_planning_spec_path or 'not_written'}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if result.overall_status == "completed":
        lines.append("- The patient-input package was executed through torso, vascular, flow, and RT planning stages.")
    elif result.overall_status == "planned_only":
        lines.append("- The build was validated in dry-run mode; no downstream phantom volumes were generated.")
    elif result.overall_status.startswith("blocked"):
        lines.append("- The build stopped before generating derived anatomy because one or more required intake conditions were not satisfied.")
    else:
        lines.append("- The build did not complete; inspect the failed step notes before rerunning.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _finalize(result: PatientPhantomBuildResult) -> PatientPhantomBuildResult:
    _write_manifest(Path(result.build_manifest_yaml_path), result)
    _write_report(Path(result.report_path), result)
    return result


def run_patient_phantom_build(
    patient_manifest_path: str | Path,
    output_dir: str | Path = "outputs/digital/patient_builds",
    case_id: str | None = None,
    report_path: str | Path | None = "outputs/reports/patient_phantom_build_executor_stage001.md",
    organ_labelmap_path: str | Path | None = None,
    materials_path: str | Path | None = None,
    baseline_graph_path: str | Path | None = None,
    allow_template_vessels: bool = False,
    dry_run: bool = False,
    run_rt: bool = True,
    export_dicom: bool = False,
    sample_step_mm: float = 0.75,
    vessel_wall_thickness_mm: float = 2.0,
    arterial_inlet_flow_ml_s: float = 80.0,
    heart_rate_bpm: float = 60.0,
    organ_label_mode: str = "auto",
    correct_bone_conflicts: bool = False,
    bone_clearance_mm: float = 8.0,
) -> PatientPhantomBuildResult:
    manifest_path = Path(patient_manifest_path)
    manifest = _load_yaml(manifest_path)
    patient_id = str(manifest.get("patient_id", "patient"))
    resolved_case_id = case_id or str(manifest.get("case_id", f"{_slug(patient_id)}_patient_build"))
    output_root = Path(output_dir) / resolved_case_id
    report = Path(report_path) if report_path is not None else output_root / f"{resolved_case_id}_patient_build_executor_report_v001.md"
    build_manifest = output_root / f"{resolved_case_id}_patient_build_manifest_v001.yaml"

    ct_path = _input_path(manifest, "ct")
    organ_path = _input_path(manifest, "organ_seg")
    gi_path = _input_path(manifest, "gi_seg")
    vessel_path = _input_path(manifest, "vessel_seg")
    labelmap = _config_path(manifest, "organ_labelmap", organ_labelmap_path)
    gi_labelmap = _config_path(manifest, "gi_labelmap", None) or str(DEFAULT_GI_LABELMAP)
    materials = _config_path(manifest, "materials", materials_path)
    configured_baseline_graph = _config_path(manifest, "baseline_graph", baseline_graph_path)
    active_baseline = resolve_stage007_active_baseline()
    baseline_graph = configured_baseline_graph or active_baseline.graph_path
    resolved_organ_label_mode = _organ_label_mode(manifest, organ_label_mode)

    use_patient_vessels = vessel_path is not None and _status(manifest, "vascular_adaptation") == "ready_for_patient_vessel_replacement"
    use_template_vessels = not use_patient_vessels and allow_template_vessels and baseline_graph is not None
    steps: list[PatientBuildStep] = []
    notes = [
        "patient_build_executor_consumes_patient_input_adapter_manifest",
        "registration_required_inputs_are_blocked_before_downstream_generation",
        f"organ_label_mode={resolved_organ_label_mode}",
    ]
    if configured_baseline_graph is None and active_baseline.graph_path is not None:
        notes.append("baseline_graph_auto_resolved_from_stage007_active_baseline")

    blocked_reasons: list[str] = []
    if _status(manifest, "primary_ct") != "ct_nifti_ready" or ct_path is None:
        blocked_reasons.append("primary_ct_not_ready_or_missing")
    if _status(manifest, "anatomy_adaptation") != "ready_for_ct_registered_anatomy_build" or organ_path is None:
        blocked_reasons.append("organ_segmentation_not_ready_on_ct_grid")
    if _input_geometry(manifest, "organ_seg") not in {None, "co_registered_to_ct_grid"}:
        blocked_reasons.append("organ_segmentation_registration_required")
    if vessel_path is not None and _input_geometry(manifest, "vessel_seg") != "co_registered_to_ct_grid":
        blocked_reasons.append("vessel_segmentation_registration_required")
    if gi_path is not None and _input_geometry(manifest, "gi_seg") != "co_registered_to_ct_grid":
        blocked_reasons.append("gi_segmentation_registration_required")
    if baseline_graph is None:
        blocked_reasons.append("baseline_graph_required_for_vascular_build")
    if not use_patient_vessels and not use_template_vessels:
        blocked_reasons.append("patient_vessel_segmentation_missing_or_template_vessels_not_allowed")
    if labelmap is None:
        blocked_reasons.append("organ_labelmap_missing")
    if materials is None:
        blocked_reasons.append("materials_config_missing")

    if blocked_reasons:
        status = "blocked_registration_required" if any("registration_required" in reason for reason in blocked_reasons) else "blocked_intake_not_ready"
        steps = list(_planned_steps(run_rt=run_rt, use_patient_vessels=use_patient_vessels, use_template_vessels=use_template_vessels))
        notes.extend(blocked_reasons)
        return _finalize(
            PatientPhantomBuildResult(
                case_id=resolved_case_id,
                patient_id=patient_id,
                output_dir=str(output_root),
                build_manifest_yaml_path=str(build_manifest),
                report_path=str(report),
                source_patient_manifest_path=str(manifest_path),
                overall_status=status,
                dry_run=dry_run,
                torso_spec_path=None,
                vascular_graph_path=None,
                voxelized_spec_path=None,
                flow_boundary_config_path=None,
                flow_1d_model_path=None,
                coupled_flow_model_path=None,
                rt_package_spec_path=None,
                rt_planning_spec_path=None,
                steps=tuple(steps),
                notes=tuple(notes),
            )
        )

    if dry_run:
        notes.append("dry_run_no_downstream_outputs_generated")
        return _finalize(
            PatientPhantomBuildResult(
                case_id=resolved_case_id,
                patient_id=patient_id,
                output_dir=str(output_root),
                build_manifest_yaml_path=str(build_manifest),
                report_path=str(report),
                source_patient_manifest_path=str(manifest_path),
                overall_status="planned_only",
                dry_run=True,
                torso_spec_path=None,
                vascular_graph_path=None,
                voxelized_spec_path=None,
                flow_boundary_config_path=None,
                flow_1d_model_path=None,
                coupled_flow_model_path=None,
                rt_package_spec_path=None,
                rt_planning_spec_path=None,
                steps=_planned_steps(run_rt=run_rt, use_patient_vessels=use_patient_vessels, use_template_vessels=use_template_vessels),
                notes=tuple(notes),
            )
        )

    torso_spec = vascular_graph = voxelized_spec = None
    boundary_config = flow_1d_model = coupled_flow_model = None
    rt_package_spec = rt_planning_spec = None

    try:
        if resolved_organ_label_mode == "material":
            torso = _build_material_label_torso(
                ct_path=str(ct_path),
                material_labels_path=str(organ_path),
                materials_path=str(materials),
                output_dir=output_root / "torso",
                case_id=f"{resolved_case_id}_torso",
                report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_torso.md",
            )
        elif resolved_organ_label_mode == "btcv":
            torso = _build_btcv_label_torso(
                ct_path=str(ct_path),
                btcv_labels_path=str(organ_path),
                gi_segmentation_path=gi_path,
                gi_labelmap_path=gi_labelmap,
                materials_path=str(materials),
                output_dir=output_root / "torso",
                case_id=f"{resolved_case_id}_torso",
                report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_torso.md",
            )
        else:
            torso = build_digital_torso_phantom(
                ct_path=str(ct_path),
                labels_path=str(organ_path),
                labelmap_path=str(labelmap),
                materials_path=str(materials),
                output_dir=output_root / "torso",
                case_id=f"{resolved_case_id}_torso",
                report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_torso.md",
            )
        torso_spec = torso.spec_yaml_path
        steps.append(PatientBuildStep("torso", "completed", torso.spec_yaml_path, torso.report_path, (f"body_volume_cm3={torso.body_volume_cm3:.3f}",)))

        if use_patient_vessels:
            graph = build_cta_derived_vascular_graph(
                baseline_graph_path=str(baseline_graph),
                vascular_mask_path=str(vessel_path),
                output_dir=output_root / "vascular_graph",
                case_id=f"{resolved_case_id}_patient_vessels",
                report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_vascular_graph.md",
            )
            vascular_graph = graph.graph_yaml_path
            steps.append(
                PatientBuildStep(
                    "vascular_graph",
                    "completed",
                    graph.graph_yaml_path,
                    graph.report_path,
                    (
                        f"replaced_edges={graph.replaced_edge_count}",
                        f"promoted_branches={graph.promoted_branch_count}",
                    ),
                )
            )
        else:
            vascular_graph = str(baseline_graph)
            steps.append(
                PatientBuildStep(
                    "vascular_graph",
                    "completed_template_vessels",
                    vascular_graph,
                    None,
                    ("template_vascular_graph_used_by_explicit_permission",),
                )
            )

        if correct_bone_conflicts:
            correction = correct_vessel_bone_conflicts(
                graph_yaml_path=vascular_graph,
                anatomy_labels_path=torso.material_label_path,
                edge_metrics_csv_path=None,
                output_dir=output_root / "vascular_graph_bone_corrected",
                case_id=f"{resolved_case_id}_patient_vessels_bone_corrected",
                clearance_mm=bone_clearance_mm,
                edge_bone_review_threshold=0.05,
                max_node_shift_mm=24.0,
                max_point_shift_mm=24.0,
                smooth_iterations=1,
                report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_vascular_bone_correction.md",
            )
            vascular_graph = correction.corrected_graph_yaml_path
            steps.append(
                PatientBuildStep(
                    "vascular_bone_correction",
                    "completed",
                    correction.corrected_graph_yaml_path,
                    correction.report_path,
                    (
                        f"corrected_nodes={correction.corrected_node_count}",
                        f"corrected_edges={correction.corrected_edge_count}",
                        f"bone_clearance_mm={bone_clearance_mm:.3f}",
                    ),
                )
            )

        voxelized = voxelize_vascular_network(
            graph_yaml_path=vascular_graph,
            combined_labels_path=torso.material_label_path,
            materials_path=str(materials),
            output_dir=output_root / "vascular_network_voxelized",
            case_id=f"{resolved_case_id}_vascular",
            source_ct_path=str(ct_path),
            body_mask_path=torso.body_mask_path,
            sample_step_mm=sample_step_mm,
            vessel_wall_thickness_mm=vessel_wall_thickness_mm,
            contrast_mode="arterial",
            collision_cleanup="nearest-centerline",
            clip_to_body=True,
            write_material_volumes=True,
            report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_vascular_voxelized.md",
        )
        voxelized_spec = voxelized.spec_yaml_path
        steps.append(
            PatientBuildStep(
                "voxelization",
                "completed",
                voxelized.spec_yaml_path,
                voxelized.report_path,
                (
                    f"connected_components={voxelized.connected_components}",
                    f"overlap_after_cleanup={voxelized.overlap_voxels_after_cleanup}",
                ),
            )
        )

        centerline_qa = validate_vessel_organ_anatomy(
            voxelized_spec_path=voxelized.spec_yaml_path,
            graph_yaml_path=vascular_graph,
            anatomy_labels_path=torso.material_label_path,
            output_dir=output_root / "validation" / "vessel_organ_anatomy",
            case_id=f"{resolved_case_id}_vessel_organ",
            sample_step_mm=2.0,
            report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_vessel_organ_anatomy.md",
        )
        radius_qa = validate_vessel_radius_anatomy(
            voxelized_spec_path=voxelized.spec_yaml_path,
            graph_yaml_path=vascular_graph,
            anatomy_labels_path=torso.material_label_path,
            output_dir=output_root / "validation" / "vessel_radius_anatomy",
            case_id=f"{resolved_case_id}_vessel_radius",
            sample_step_mm=2.0,
            scaled_radius_factor=0.75,
            review_lumen_bone_fraction=0.10,
            fail_lumen_bone_fraction=0.35,
            report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_vessel_radius_anatomy.md",
        )
        steps.append(
            PatientBuildStep(
                "vessel_qa",
                "completed",
                radius_qa.edge_metrics_csv_path,
                radius_qa.report_path,
                (
                    f"organ_aware_pass_review_fail={centerline_qa.pass_count}/{centerline_qa.review_count}/{centerline_qa.fail_count}",
                    f"radius_pass_review_fail={radius_qa.pass_count}/{radius_qa.review_count}/{radius_qa.fail_count}",
                ),
            )
        )

        flow_boundary = build_flow_boundary_package(
            voxelized_spec_path=voxelized.spec_yaml_path,
            graph_yaml_path=vascular_graph,
            output_dir=output_root / "flow_boundary_conditions",
            case_id=f"{resolved_case_id}_flow",
            arterial_inlet_flow_ml_s=arterial_inlet_flow_ml_s,
            report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_flow_boundary_conditions.md",
        )
        boundary_config = flow_boundary.config_yaml_path
        steady_flow = build_flow_1d_model(
            graph_yaml_path=vascular_graph,
            boundary_config_path=flow_boundary.config_yaml_path,
            output_dir=output_root / "flow_1d",
            case_id=f"{resolved_case_id}_flow",
            report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_flow_1d.md",
        )
        flow_1d_model = steady_flow.model_yaml_path
        coupled_flow = build_coupled_pulsatile_flow_model(
            flow_1d_model_path=steady_flow.model_yaml_path,
            boundary_config_path=flow_boundary.config_yaml_path,
            output_dir=output_root / "flow_coupled_pulsatile",
            case_id=f"{resolved_case_id}_flow",
            heart_rate_bpm=heart_rate_bpm,
            report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_flow_coupled_pulsatile.md",
        )
        coupled_flow_model = coupled_flow.model_yaml_path
        steps.append(
            PatientBuildStep(
                "flow",
                "completed",
                coupled_flow.model_yaml_path,
                coupled_flow.report_path,
                (
                    f"mapped_boundaries={flow_boundary.mapped_boundary_count}/{flow_boundary.boundary_count}",
                    f"max_mass_residual_ml_s={coupled_flow.max_abs_mass_balance_residual_ml_s:.9f}",
                ),
            )
        )

        if run_rt:
            rt_package = build_radiotherapy_qa_package(
                combined_spec_path=voxelized.spec_yaml_path,
                output_dir=output_root / "radiotherapy_qa_package",
                case_id=f"{resolved_case_id}_rt",
                scenario="blood",
                report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_radiotherapy_qa.md",
            )
            rt_package_spec = rt_package.package_spec_yaml_path
            rt_plan = build_rt_planning_bundle(
                rt_package_spec_path=rt_package.package_spec_yaml_path,
                coupled_flow_model_path=coupled_flow.model_yaml_path,
                output_dir=output_root / "rt_planning_bundle",
                case_id=f"{resolved_case_id}_rt",
                export_dicom=export_dicom,
                report_path=Path("outputs/reports") / f"{resolved_case_id}_patient_rt_planning_bundle.md",
            )
            rt_planning_spec = rt_plan.bundle_spec_yaml_path
            steps.append(
                PatientBuildStep(
                    "radiotherapy",
                    "completed",
                    rt_plan.bundle_spec_yaml_path,
                    rt_plan.report_path,
                    (
                        f"gtv_volume_cm3={rt_package.target_gtv_volume_cm3:.3f}",
                        f"flow_amplitude_fraction={rt_plan.flow_amplitude_fraction:.6f}",
                    ),
                )
            )
        else:
            notes.append("radiotherapy_stage_skipped_by_user_option")

        status = "completed"
    except Exception as exc:
        steps.append(PatientBuildStep("failed_stage", "failed", None, None, (f"{type(exc).__name__}: {exc}",)))
        notes.append("patient_build_failed_inspect_failed_stage_note")
        status = "failed"

    return _finalize(
        PatientPhantomBuildResult(
            case_id=resolved_case_id,
            patient_id=patient_id,
            output_dir=str(output_root),
            build_manifest_yaml_path=str(build_manifest),
            report_path=str(report),
            source_patient_manifest_path=str(manifest_path),
            overall_status=status,
            dry_run=False,
            torso_spec_path=torso_spec,
            vascular_graph_path=vascular_graph,
            voxelized_spec_path=voxelized_spec,
            flow_boundary_config_path=boundary_config,
            flow_1d_model_path=flow_1d_model,
            coupled_flow_model_path=coupled_flow_model,
            rt_package_spec_path=rt_package_spec,
            rt_planning_spec_path=rt_planning_spec,
            steps=tuple(steps),
            notes=tuple(notes),
        )
    )


def format_patient_phantom_build_result(result: PatientPhantomBuildResult) -> str:
    lines = [
        "Patient Phantom Build Executor",
        f"Case ID: {result.case_id}",
        f"Patient/profile ID: {result.patient_id}",
        f"Overall status: {result.overall_status}",
        f"Dry run: {result.dry_run}",
        f"Build manifest: {result.build_manifest_yaml_path}",
        f"Report: {result.report_path}",
    ]
    for step in result.steps:
        lines.append(f"- {step.step_id}: {step.status}")
    return "\n".join(lines)
