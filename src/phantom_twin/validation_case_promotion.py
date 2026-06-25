from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .validation_case_staging import ValidationCaseStagingResult, stage_validation_case


@dataclass(frozen=True)
class ValidationCasePromotionResult:
    base_case_id: str
    promoted_case_id: str
    source_dataset: str
    staged_case_manifest_path: str
    vessel_harmonization_manifest_path: str
    promoted_manifest_yaml_path: str
    promoted_intake_case_csv_path: str
    report_path: str
    harmonized_vessel_path: str
    harmonization_status: str
    geometry_status: str
    promoted_completeness_status: str
    recommended_commands: tuple[str, ...]
    notes: tuple[str, ...]


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Validation case promotion requires nibabel for geometry checks.") from exc
    return nib


def _load_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def _input_path(manifest: dict, role: str) -> str:
    for item in manifest.get("inputs", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")) == role:
            return str(item.get("staged_path", "") or item.get("source_path", "") or "")
    return ""


def _geometry_status(ct_path: str, vessel_path: str) -> str:
    if not ct_path or not vessel_path:
        return "missing_geometry_inputs"
    ct = Path(ct_path)
    vessel = Path(vessel_path)
    if not ct.exists() or not vessel.exists():
        return "path_missing"
    if ct.is_dir() or vessel.is_dir():
        return "dicom_or_directory_geometry_not_evaluated"
    nib = _import_nibabel()
    ct_image = nib.load(str(ct))
    vessel_image = nib.load(str(vessel))
    ct_shape = tuple(int(value) for value in ct_image.shape)
    vessel_shape = tuple(int(value) for value in vessel_image.shape)
    ct_spacing = tuple(float(value) for value in ct_image.header.get_zooms()[: len(ct_shape)])
    vessel_spacing = tuple(float(value) for value in vessel_image.header.get_zooms()[: len(vessel_shape)])
    shape_match = ct_shape == vessel_shape
    spacing_match = np.allclose(np.asarray(ct_spacing, dtype=float), np.asarray(vessel_spacing, dtype=float), atol=1e-3)
    affine_match = np.allclose(np.asarray(ct_image.affine, dtype=float), np.asarray(vessel_image.affine, dtype=float), atol=1e-3)
    if shape_match and spacing_match and affine_match:
        return "co_registered_to_ct_grid"
    if shape_match and spacing_match:
        return "same_shape_spacing_but_affine_differs"
    return "registration_required_to_ct_grid"


def _format_report(result: ValidationCasePromotionResult) -> str:
    lines = [
        "# P1 Harmonized Vessel Case Promotion",
        "",
        f"Base case: `{result.base_case_id}`",
        f"Promoted case: `{result.promoted_case_id}`",
        f"Harmonization status: `{result.harmonization_status}`",
        f"Geometry status: `{result.geometry_status}`",
        f"Promoted completeness: `{result.promoted_completeness_status}`",
        "",
        "## Outputs",
        "",
        f"- Promoted manifest: `{result.promoted_manifest_yaml_path}`",
        f"- Promoted intake CSV: `{result.promoted_intake_case_csv_path}`",
        f"- Harmonized vessel mask: `{result.harmonized_vessel_path}`",
        f"- Report: `{result.report_path}`",
        "",
        "## Recommended Commands",
        "",
        "```bash",
    ]
    lines.extend(result.recommended_commands)
    lines.extend(["```", "", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_validation_case_promotion_result(result: ValidationCasePromotionResult) -> str:
    return _format_report(result)


def promote_harmonized_vessel_case(
    *,
    staged_case_manifest_path: str | Path,
    vessel_harmonization_manifest_path: str | Path,
    promoted_case_id: str | None = None,
    output_dir: str | Path = "data/validation/p1_cases",
    report_path: str | Path | None = None,
) -> ValidationCasePromotionResult:
    staged_manifest = _load_yaml(staged_case_manifest_path)
    harmonization_manifest = _load_yaml(vessel_harmonization_manifest_path)
    if staged_manifest.get("package_type") != "p1_validation_case_staging":
        raise ValueError("staged_case_manifest_path must point to a p1_validation_case_staging manifest")
    if harmonization_manifest.get("package_type") != "p1_vessel_label_harmonization":
        raise ValueError("vessel_harmonization_manifest_path must point to a p1_vessel_label_harmonization manifest")

    base_case_id = str(staged_manifest.get("case_id", "validation_case"))
    final_case_id = promoted_case_id or f"{base_case_id}_harmonized_vessel"
    outputs = harmonization_manifest.get("outputs", {})
    harmonized_vessel = str(outputs.get("harmonized_nifti", "") or "")
    if not harmonized_vessel:
        raise ValueError("Harmonization manifest does not include a harmonized_nifti output")

    source_dataset = f"{staged_manifest.get('source_dataset', 'unknown')}_with_harmonized_vessel"
    notes = (
        f"base_case_manifest={staged_case_manifest_path}",
        f"vessel_harmonization_manifest={vessel_harmonization_manifest_path}",
        "promotion_replaces_only_the_vessel_segmentation_path",
        "geometry_status_must_be_co_registered_before_patient_specific_build_or_flow_QA",
    )
    staged_result: ValidationCaseStagingResult = stage_validation_case(
        case_id=final_case_id,
        source_dataset=source_dataset,
        ct_path=_input_path(staged_manifest, "ct"),
        cta_path=_input_path(staged_manifest, "cta"),
        ctv_path=_input_path(staged_manifest, "ctv"),
        organ_seg_path=_input_path(staged_manifest, "organ_seg"),
        vessel_seg_path=harmonized_vessel,
        vessel_label_config=str(harmonization_manifest.get("target_label_config", staged_manifest.get("vessel_label_config", ""))),
        output_dir=output_dir,
        required_vessel_labels=tuple(int(value) for value in staged_manifest.get("required_vessel_labels", [])) or None,
        access_status=str(staged_manifest.get("access_status", "local_review_required")),
        notes="; ".join(notes),
        copy_inputs=False,
        report_path=None,
    )
    geometry_status = _geometry_status(_input_path(staged_manifest, "ct"), harmonized_vessel)
    report = Path(report_path) if report_path is not None else Path("outputs/reports") / f"{staged_result.case_id}_harmonized_vessel_promotion.md"
    recommended_commands = (
        f"python -m phantom_twin.cli build-validation-intake-package --cases-csv {staged_result.intake_case_csv_path} --output-dir outputs/releases/mode03_neg_stage007_rc1/validation_intake_{staged_result.case_id} --intake-id {staged_result.case_id}_p1_intake --report outputs/reports/{staged_result.case_id}_p1_intake.md",
        f"python -m phantom_twin.cli discover-validation-candidates --search-root {staged_result.output_dir} --output-dir outputs/releases/mode03_neg_stage007_rc1/validation_discovery_{staged_result.case_id} --discovery-id {staged_result.case_id}_p1_discovery --report outputs/reports/{staged_result.case_id}_p1_discovery.md",
    )
    result = ValidationCasePromotionResult(
        base_case_id=base_case_id,
        promoted_case_id=staged_result.case_id,
        source_dataset=source_dataset,
        staged_case_manifest_path=str(staged_case_manifest_path),
        vessel_harmonization_manifest_path=str(vessel_harmonization_manifest_path),
        promoted_manifest_yaml_path=staged_result.manifest_yaml_path,
        promoted_intake_case_csv_path=staged_result.intake_case_csv_path,
        report_path=str(report),
        harmonized_vessel_path=harmonized_vessel,
        harmonization_status=str(harmonization_manifest.get("status", "unknown")),
        geometry_status=geometry_status,
        promoted_completeness_status=staged_result.completeness_status,
        recommended_commands=recommended_commands,
        notes=notes,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
