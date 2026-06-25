from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import shutil

import yaml

from .validation_intake import DEFAULT_REQUIRED_VESSEL_LABELS, INTAKE_FIELDS


@dataclass(frozen=True)
class StagedValidationInput:
    role: str
    source_path: str
    staged_path: str
    exists: bool
    staging_mode: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ValidationCaseStagingResult:
    case_id: str
    source_dataset: str
    output_dir: str
    manifest_yaml_path: str
    intake_case_csv_path: str
    report_path: str
    completeness_status: str
    present_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]
    required_vessel_labels: tuple[int, ...]
    inputs: tuple[StagedValidationInput, ...]
    recommended_commands: tuple[str, ...]
    notes: tuple[str, ...]


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "case"


def _compound_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix


def _stage_input(path_value: str | Path | None, *, role: str, case_dir: Path, copy_inputs: bool) -> StagedValidationInput:
    if path_value is None or str(path_value).strip() == "":
        return StagedValidationInput(
            role=role,
            source_path="",
            staged_path="",
            exists=False,
            staging_mode="missing",
            notes=("path_not_supplied",),
        )
    source = Path(path_value)
    exists = source.exists()
    if not exists:
        return StagedValidationInput(
            role=role,
            source_path=str(source),
            staged_path=str(source),
            exists=False,
            staging_mode="missing",
            notes=("path_does_not_exist",),
        )
    if not copy_inputs:
        return StagedValidationInput(
            role=role,
            source_path=str(source),
            staged_path=str(source),
            exists=True,
            staging_mode="referenced",
            notes=("input_referenced_without_copying",),
        )

    input_dir = case_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        destination = input_dir / f"{role}_{source.name}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        destination = input_dir / f"{role}{_compound_suffix(source)}"
        shutil.copy2(source, destination)
    return StagedValidationInput(
        role=role,
        source_path=str(source),
        staged_path=str(destination),
        exists=True,
        staging_mode="copied",
        notes=("input_copied_into_validation_case_folder",),
    )


def _role_status(inputs: tuple[StagedValidationInput, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_role = {item.role: item.exists for item in inputs}
    present: list[str] = []
    missing: list[str] = []
    for role in ("ct", "cta_or_ctv", "organ_seg", "vessel_seg"):
        if role == "cta_or_ctv":
            exists = by_role.get("cta", False) or by_role.get("ctv", False)
        else:
            exists = by_role.get(role, False)
        if exists:
            present.append(role)
        else:
            missing.append(role)
    return tuple(present), tuple(missing)


def _completeness_status(present_roles: tuple[str, ...]) -> str:
    if set(present_roles) == {"ct", "cta_or_ctv", "organ_seg", "vessel_seg"}:
        return "complete_ready_for_intake_qa"
    if present_roles:
        return "partial_missing_required_inputs"
    return "empty_case_template"


def _input_path(inputs: tuple[StagedValidationInput, ...], role: str) -> str:
    for item in inputs:
        if item.role == role:
            return item.staged_path
    return ""


def _write_case_csv(
    path: Path,
    *,
    case_id: str,
    source_dataset: str,
    inputs: tuple[StagedValidationInput, ...],
    vessel_label_config: str,
    required_vessel_labels: tuple[int, ...],
    access_status: str,
    notes: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=INTAKE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": case_id,
                "source_dataset": source_dataset,
                "ct_path": _input_path(inputs, "ct"),
                "cta_path": _input_path(inputs, "cta"),
                "ctv_path": _input_path(inputs, "ctv"),
                "organ_seg_path": _input_path(inputs, "organ_seg"),
                "vessel_seg_path": _input_path(inputs, "vessel_seg"),
                "vessel_label_config": vessel_label_config,
                "required_vessel_labels": ",".join(str(label) for label in required_vessel_labels),
                "access_status": access_status,
                "notes": notes,
            }
        )


def _write_manifest(path: Path, result: ValidationCaseStagingResult, *, vessel_label_config: str, access_status: str) -> None:
    payload = {
        "case_id": result.case_id,
        "source_dataset": result.source_dataset,
        "package_type": "p1_validation_case_staging",
        "completeness_status": result.completeness_status,
        "present_roles": list(result.present_roles),
        "missing_roles": list(result.missing_roles),
        "required_vessel_labels": list(result.required_vessel_labels),
        "vessel_label_config": vessel_label_config,
        "access_status": access_status,
        "copy_inputs": any(item.staging_mode == "copied" for item in result.inputs),
        "inputs": [
            {
                "role": item.role,
                "source_path": item.source_path,
                "staged_path": item.staged_path,
                "exists": item.exists,
                "staging_mode": item.staging_mode,
                "notes": list(item.notes),
            }
            for item in result.inputs
        ],
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "intake_case_csv": result.intake_case_csv_path,
            "report": result.report_path,
        },
        "recommended_commands": list(result.recommended_commands),
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: ValidationCaseStagingResult) -> str:
    lines = [
        "# P1 Validation Case Staging",
        "",
        f"Case ID: `{result.case_id}`",
        f"Source dataset: `{result.source_dataset}`",
        f"Completeness: `{result.completeness_status}`",
        "",
        "## Role Status",
        "",
        f"- Present roles: `{', '.join(result.present_roles) or 'none'}`",
        f"- Missing roles: `{', '.join(result.missing_roles) or 'none'}`",
        "",
        "## Outputs",
        "",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Intake case CSV: `{result.intake_case_csv_path}`",
        f"- Report: `{result.report_path}`",
        "",
        "## Inputs",
        "",
        "| role | status | staging | staged path |",
        "| --- | --- | --- | --- |",
    ]
    for item in result.inputs:
        status = "present" if item.exists else "missing"
        lines.append(f"| `{item.role}` | `{status}` | `{item.staging_mode}` | `{item.staged_path}` |")
    lines.extend(["", "## Recommended Commands", "", "```bash"])
    lines.extend(result.recommended_commands)
    lines.extend(["```", "", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_validation_case_staging_result(result: ValidationCaseStagingResult) -> str:
    return _format_report(result)


def stage_validation_case(
    *,
    case_id: str,
    source_dataset: str,
    ct_path: str | Path | None = None,
    cta_path: str | Path | None = None,
    ctv_path: str | Path | None = None,
    organ_seg_path: str | Path | None = None,
    vessel_seg_path: str | Path | None = None,
    vessel_label_config: str | Path | None = "configs/labelmaps/medseg_abdominal_vasculature.yaml",
    output_dir: str | Path = "data/validation/p1_cases",
    required_vessel_labels: tuple[int, ...] | None = DEFAULT_REQUIRED_VESSEL_LABELS,
    access_status: str = "local_review_required",
    notes: str = "",
    copy_inputs: bool = False,
    report_path: str | Path | None = None,
) -> ValidationCaseStagingResult:
    clean_case_id = _slug(case_id)
    root = Path(output_dir)
    case_dir = root / clean_case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    report = Path(report_path) if report_path is not None else Path("outputs/reports") / f"{clean_case_id}_p1_validation_case_staging.md"
    manifest_yaml = case_dir / f"{clean_case_id}_p1_validation_case_manifest_v001.yaml"
    intake_csv = case_dir / f"{clean_case_id}_p1_intake_candidate_v001.csv"
    resolved_required = tuple(required_vessel_labels or DEFAULT_REQUIRED_VESSEL_LABELS)
    inputs = (
        _stage_input(ct_path, role="ct", case_dir=case_dir, copy_inputs=copy_inputs),
        _stage_input(cta_path, role="cta", case_dir=case_dir, copy_inputs=copy_inputs),
        _stage_input(ctv_path, role="ctv", case_dir=case_dir, copy_inputs=copy_inputs),
        _stage_input(organ_seg_path, role="organ_seg", case_dir=case_dir, copy_inputs=copy_inputs),
        _stage_input(vessel_seg_path, role="vessel_seg", case_dir=case_dir, copy_inputs=copy_inputs),
    )
    present_roles, missing_roles = _role_status(inputs)
    intake_output = f"outputs/releases/mode03_neg_stage007_rc1/validation_intake_{clean_case_id}"
    recommended_commands = (
        f"python -m phantom_twin.cli build-validation-intake-package --cases-csv {intake_csv} --output-dir {intake_output} --intake-id {clean_case_id}_p1_intake --report outputs/reports/{clean_case_id}_p1_intake.md",
        f"python -m phantom_twin.cli discover-validation-candidates --search-root {case_dir} --output-dir outputs/releases/mode03_neg_stage007_rc1/validation_discovery_{clean_case_id} --discovery-id {clean_case_id}_p1_discovery --report outputs/reports/{clean_case_id}_p1_discovery.md",
    )
    result = ValidationCaseStagingResult(
        case_id=clean_case_id,
        source_dataset=source_dataset,
        output_dir=str(case_dir),
        manifest_yaml_path=str(manifest_yaml),
        intake_case_csv_path=str(intake_csv),
        report_path=str(report),
        completeness_status=_completeness_status(present_roles),
        present_roles=present_roles,
        missing_roles=missing_roles,
        required_vessel_labels=resolved_required,
        inputs=inputs,
        recommended_commands=recommended_commands,
        notes=(
            "staging_references_inputs_by_default_to_avoid_large_data_duplication",
            "run_intake_QA_before_using_case_for_patient_specific_phantom_builds",
            "complete_status_requires_CT_plus_CTA_or_CTV_plus_organ_segmentation_plus_branch_vessel_segmentation",
            notes or "no_extra_notes",
        ),
    )
    _write_case_csv(
        intake_csv,
        case_id=clean_case_id,
        source_dataset=source_dataset,
        inputs=inputs,
        vessel_label_config=str(vessel_label_config or ""),
        required_vessel_labels=resolved_required,
        access_status=access_status,
        notes=notes,
    )
    _write_manifest(manifest_yaml, result, vessel_label_config=str(vessel_label_config or ""), access_status=access_status)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
