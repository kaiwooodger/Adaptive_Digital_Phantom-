from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import numpy as np
import yaml


DEFAULT_REQUIRED_VESSEL_LABELS = (1, 2, 3, 4, 5, 6, 13, 14, 21, 24, 25, 27, 28, 33, 34, 35, 43)
INTAKE_FIELDS = [
    "case_id",
    "source_dataset",
    "ct_path",
    "cta_path",
    "ctv_path",
    "organ_seg_path",
    "vessel_seg_path",
    "vessel_label_config",
    "required_vessel_labels",
    "access_status",
    "notes",
]


@dataclass(frozen=True)
class NiftiGeometrySummary:
    path: str
    exists: bool
    kind: str
    shape: tuple[int, ...]
    spacing_mm: tuple[float, ...]
    affine: tuple[tuple[float, ...], ...]
    dtype: str
    nonzero_fraction: float | None
    unique_labels_sample: tuple[int, ...]
    unique_label_count: int | None
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ValidationIntakeCase:
    case_id: str
    source_dataset: str
    readiness_status: str
    score_percent: float
    ct_status: str
    vascular_image_status: str
    organ_seg_status: str
    vessel_seg_status: str
    geometry_status: str
    required_vessel_label_count: int
    present_required_vessel_label_count: int
    vessel_label_coverage_percent: float
    missing_required_vessel_labels: tuple[int, ...]
    recommended_action: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ValidationIntakeResult:
    intake_id: str
    output_dir: str
    template_csv_path: str
    case_summary_csv_path: str
    dataset_requirements_csv_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    case_count: int
    ready_case_count: int
    review_case_count: int
    missing_case_count: int
    required_vessel_labels: tuple[int, ...]
    cases: tuple[ValidationIntakeCase, ...]
    notes: tuple[str, ...]


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Validation intake requires nibabel for NIfTI case QA.") from exc
    return nib


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Validation intake preview requires matplotlib.") from exc
    return plt


def _is_nifti(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def _split_labels(raw: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None or str(raw).strip() == "":
        return fallback
    values: list[int] = []
    for token in str(raw).replace(";", ",").replace("|", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            continue
    return tuple(dict.fromkeys(values)) or fallback


def _load_label_config_required_labels(path: str | Path | None) -> tuple[int, ...]:
    if path is None or str(path) == "":
        return ()
    resolved = Path(path)
    if not resolved.exists():
        return ()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        return ()
    mapping = data.get("graph_edge_mapping", {})
    if not isinstance(mapping, dict):
        return ()
    labels: list[int] = []
    for item in mapping.values():
        if not isinstance(item, dict):
            continue
        for label in item.get("labels", []):
            try:
                labels.append(int(label))
            except (TypeError, ValueError):
                continue
    return tuple(dict.fromkeys(labels))


def _load_nifti_summary(path: str | Path | None, *, label_like: bool = False) -> NiftiGeometrySummary:
    if path is None or str(path) == "":
        return NiftiGeometrySummary(
            path="",
            exists=False,
            kind="missing",
            shape=(),
            spacing_mm=(),
            affine=(),
            dtype="",
            nonzero_fraction=None,
            unique_labels_sample=(),
            unique_label_count=None,
            status="missing",
            notes=("path_not_supplied",),
        )
    resolved = Path(path)
    if not resolved.exists():
        return NiftiGeometrySummary(
            path=str(resolved),
            exists=False,
            kind="missing",
            shape=(),
            spacing_mm=(),
            affine=(),
            dtype="",
            nonzero_fraction=None,
            unique_labels_sample=(),
            unique_label_count=None,
            status="missing",
            notes=("path_does_not_exist",),
        )
    if resolved.is_dir():
        return NiftiGeometrySummary(
            path=str(resolved),
            exists=True,
            kind="dicom_directory",
            shape=(),
            spacing_mm=(),
            affine=(),
            dtype="",
            nonzero_fraction=None,
            unique_labels_sample=(),
            unique_label_count=None,
            status="metadata_only",
            notes=("dicom_directory_requires_conversion_or_adapter_qa_for_geometry",),
        )
    if not _is_nifti(resolved):
        return NiftiGeometrySummary(
            path=str(resolved),
            exists=True,
            kind="file",
            shape=(),
            spacing_mm=(),
            affine=(),
            dtype="",
            nonzero_fraction=None,
            unique_labels_sample=(),
            unique_label_count=None,
            status="unsupported_file_type",
            notes=("convert_to_nifti_or_dicom_directory_for_intake_qa",),
        )
    nib = _import_nibabel()
    image = nib.load(str(resolved))
    data = np.asanyarray(image.dataobj)
    nonzero = int(np.count_nonzero(data))
    unique_count = None
    unique_sample: tuple[int, ...] = ()
    notes: list[str] = []
    if label_like:
        labels = np.unique(data.astype(np.int64))
        unique_count = int(labels.size)
        if labels.size <= 64:
            sample = labels
        else:
            sample = np.concatenate([labels[:48], labels[-16:]])
        unique_sample = tuple(int(value) for value in sample.tolist())
        if unique_count <= 1:
            notes.append("label_volume_has_one_or_zero_unique_values")
    if nonzero == 0:
        notes.append("volume_has_zero_nonzero_voxels")
    return NiftiGeometrySummary(
        path=str(resolved),
        exists=True,
        kind="nifti",
        shape=tuple(int(value) for value in data.shape),
        spacing_mm=tuple(float(value) for value in image.header.get_zooms()[: data.ndim]),
        affine=tuple(tuple(float(value) for value in row) for row in image.affine.tolist()),
        dtype=str(data.dtype),
        nonzero_fraction=float(nonzero / max(data.size, 1)),
        unique_labels_sample=unique_sample,
        unique_label_count=unique_count,
        status="accepted",
        notes=tuple(notes),
    )


def _deferred_path_summary(path: str | Path | None, *, reason: str) -> NiftiGeometrySummary:
    if path is None or str(path) == "":
        return _load_nifti_summary(path, label_like=False)
    resolved = Path(path)
    if not resolved.exists():
        return _load_nifti_summary(path, label_like=False)
    return NiftiGeometrySummary(
        path=str(resolved),
        exists=True,
        kind="nifti_deferred",
        shape=(),
        spacing_mm=(),
        affine=(),
        dtype="",
        nonzero_fraction=None,
        unique_labels_sample=(),
        unique_label_count=None,
        status="metadata_deferred",
        notes=(reason,),
    )


def _geometry_match(reference: NiftiGeometrySummary, candidate: NiftiGeometrySummary) -> str:
    if not candidate.exists:
        return "missing"
    if reference.kind != "nifti" or candidate.kind != "nifti":
        return "not_evaluated"
    shape_match = tuple(reference.shape) == tuple(candidate.shape)
    spacing_match = np.allclose(
        np.asarray(reference.spacing_mm, dtype=float),
        np.asarray(candidate.spacing_mm, dtype=float),
        atol=1e-3,
    )
    affine_match = np.allclose(
        np.asarray(reference.affine, dtype=float),
        np.asarray(candidate.affine, dtype=float),
        atol=1e-3,
    )
    if shape_match and spacing_match and affine_match:
        return "co_registered_to_ct_grid"
    if shape_match and spacing_match:
        return "same_shape_spacing_but_affine_differs"
    return "registration_required_to_ct_grid"


def _vessel_label_coverage(summary: NiftiGeometrySummary, required_labels: tuple[int, ...]) -> tuple[int, tuple[int, ...], float]:
    if not required_labels:
        return 0, (), 0.0
    present = set(summary.unique_labels_sample)
    missing = tuple(label for label in required_labels if label not in present)
    present_count = len(required_labels) - len(missing)
    coverage = 100.0 * present_count / max(len(required_labels), 1)
    return present_count, missing, coverage


def _status_for_path(summary: NiftiGeometrySummary) -> str:
    if not summary.exists:
        return "missing"
    if summary.status == "accepted":
        return "ready"
    return summary.status


def _case_score(
    *,
    ct: NiftiGeometrySummary,
    cta: NiftiGeometrySummary,
    ctv: NiftiGeometrySummary,
    organ: NiftiGeometrySummary,
    vessel: NiftiGeometrySummary,
    organ_geometry: str,
    vessel_geometry: str,
    label_coverage_percent: float,
) -> float:
    score = 0.0
    if ct.exists:
        score += 20.0
    if organ.exists:
        score += 15.0
    if vessel.exists:
        score += 20.0
    if cta.exists or ctv.exists:
        score += 10.0
    if organ_geometry == "co_registered_to_ct_grid":
        score += 7.5
    elif organ_geometry in {"same_shape_spacing_but_affine_differs", "not_evaluated"} and organ.exists:
        score += 3.0
    if vessel_geometry == "co_registered_to_ct_grid":
        score += 7.5
    elif vessel_geometry in {"same_shape_spacing_but_affine_differs", "not_evaluated"} and vessel.exists:
        score += 3.0
    score += 20.0 * label_coverage_percent / 100.0
    return min(score, 100.0)


def _readiness_status(
    score: float,
    *,
    vascular_image_ready: bool,
    organ_geometry: str,
    vessel_geometry: str,
    missing_labels: tuple[int, ...],
) -> str:
    if (
        score >= 90.0
        and vascular_image_ready
        and not missing_labels
        and organ_geometry == "co_registered_to_ct_grid"
        and vessel_geometry == "co_registered_to_ct_grid"
    ):
        return "ready_for_p1_patient_specific_validation"
    if score >= 60.0:
        return "registration_or_label_review_required"
    return "missing_required_data"


def _recommended_action(
    *,
    ct: NiftiGeometrySummary,
    cta: NiftiGeometrySummary,
    ctv: NiftiGeometrySummary,
    organ: NiftiGeometrySummary,
    vessel: NiftiGeometrySummary,
    organ_geometry: str,
    vessel_geometry: str,
    missing_labels: tuple[int, ...],
) -> str:
    if not ct.exists:
        return "Stage primary CT NIfTI or DICOM directory first."
    if not organ.exists:
        return "Stage organ/material segmentation on the CT grid."
    if not vessel.exists:
        return "Stage arterial/venous vessel segmentation with branch labels."
    if not (cta.exists or ctv.exists):
        return "Stage CTA and/or CTV image data for vascular provenance."
    if organ_geometry != "co_registered_to_ct_grid" or vessel_geometry != "co_registered_to_ct_grid":
        return "Register/resample organ and vessel masks to the CT grid before running patient-specific build."
    if missing_labels:
        return f"Add or remap missing required vessel labels: {', '.join(str(label) for label in missing_labels)}."
    return "Run build-patient-phantom-adapter, then run-patient-phantom-build with template vessels disabled."


def _case_from_row(row: dict[str, str], default_required_labels: tuple[int, ...]) -> ValidationIntakeCase:
    label_config_required = _load_label_config_required_labels(row.get("vessel_label_config"))
    required = _split_labels(row.get("required_vessel_labels"), label_config_required or default_required_labels)
    ct_path = row.get("ct_path")
    cta_path = row.get("cta_path")
    ctv_path = row.get("ctv_path")
    organ_path = row.get("organ_seg_path")
    vessel_path = row.get("vessel_seg_path")
    anatomy_only_without_ct = not ct_path and bool(organ_path) and not vessel_path and not cta_path and not ctv_path
    if anatomy_only_without_ct:
        ct = _load_nifti_summary(ct_path, label_like=False)
        cta = _load_nifti_summary(cta_path, label_like=False)
        ctv = _load_nifti_summary(ctv_path, label_like=False)
        organ = _deferred_path_summary(
            organ_path,
            reason="organ_label_volume_QA_deferred_until_primary_CT_is_staged",
        )
        vessel = _load_nifti_summary(vessel_path, label_like=True)
    else:
        ct = _load_nifti_summary(ct_path, label_like=False)
        cta = _load_nifti_summary(cta_path, label_like=False)
        ctv = _load_nifti_summary(ctv_path, label_like=False)
        organ = _load_nifti_summary(organ_path, label_like=True)
        vessel = _load_nifti_summary(vessel_path, label_like=True)
    organ_geometry = _geometry_match(ct, organ)
    vessel_geometry = _geometry_match(ct, vessel)
    present_count, missing_labels, label_coverage = _vessel_label_coverage(vessel, required)
    score = _case_score(
        ct=ct,
        cta=cta,
        ctv=ctv,
        organ=organ,
        vessel=vessel,
        organ_geometry=organ_geometry,
        vessel_geometry=vessel_geometry,
        label_coverage_percent=label_coverage,
    )
    status = _readiness_status(
        score,
        vascular_image_ready=cta.exists or ctv.exists,
        organ_geometry=organ_geometry,
        vessel_geometry=vessel_geometry,
        missing_labels=missing_labels,
    )
    notes = [
        note
        for summary in (ct, cta, ctv, organ, vessel)
        for note in summary.notes
    ]
    if str(row.get("access_status", "")).lower() not in {"", "approved", "open", "available"}:
        notes.append(f"access_status={row.get('access_status')}")
    return ValidationIntakeCase(
        case_id=row.get("case_id", "").strip() or "candidate_case",
        source_dataset=row.get("source_dataset", "").strip(),
        readiness_status=status,
        score_percent=score,
        ct_status=_status_for_path(ct),
        vascular_image_status="ready" if cta.exists or ctv.exists else "missing",
        organ_seg_status=_status_for_path(organ),
        vessel_seg_status=_status_for_path(vessel),
        geometry_status=f"organ={organ_geometry};vessel={vessel_geometry}",
        required_vessel_label_count=len(required),
        present_required_vessel_label_count=present_count,
        vessel_label_coverage_percent=label_coverage,
        missing_required_vessel_labels=missing_labels,
        recommended_action=_recommended_action(
            ct=ct,
            cta=cta,
            ctv=ctv,
            organ=organ,
            vessel=vessel,
            organ_geometry=organ_geometry,
            vessel_geometry=vessel_geometry,
            missing_labels=missing_labels,
        ),
        notes=tuple(notes),
    )


def _read_cases_csv(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or str(path) == "":
        return []
    resolved = Path(path)
    if not resolved.exists():
        return []
    with resolved.open(newline="") as csvfile:
        return [dict(row) for row in csv.DictReader(csvfile)]


def _write_template_csv(path: Path, required_labels: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=INTAKE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "candidate_case_001",
                "source_dataset": "local_or_public_cta_ctv_source",
                "ct_path": "",
                "cta_path": "",
                "ctv_path": "",
                "organ_seg_path": "",
                "vessel_seg_path": "",
                "vessel_label_config": "configs/labelmaps/medseg_abdominal_vasculature.yaml",
                "required_vessel_labels": ",".join(str(label) for label in required_labels),
                "access_status": "approved_or_open",
                "notes": "Fill paths, then rerun build-validation-intake-package with --cases-csv this template.",
            }
        )


def _write_case_summary_csv(path: Path, cases: tuple[ValidationIntakeCase, ...]) -> None:
    fields = [
        "case_id",
        "source_dataset",
        "readiness_status",
        "score_percent",
        "ct_status",
        "vascular_image_status",
        "organ_seg_status",
        "vessel_seg_status",
        "geometry_status",
        "required_vessel_label_count",
        "present_required_vessel_label_count",
        "vessel_label_coverage_percent",
        "missing_required_vessel_labels",
        "recommended_action",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "source_dataset": case.source_dataset,
                    "readiness_status": case.readiness_status,
                    "score_percent": f"{case.score_percent:.3f}",
                    "ct_status": case.ct_status,
                    "vascular_image_status": case.vascular_image_status,
                    "organ_seg_status": case.organ_seg_status,
                    "vessel_seg_status": case.vessel_seg_status,
                    "geometry_status": case.geometry_status,
                    "required_vessel_label_count": case.required_vessel_label_count,
                    "present_required_vessel_label_count": case.present_required_vessel_label_count,
                    "vessel_label_coverage_percent": f"{case.vessel_label_coverage_percent:.3f}",
                    "missing_required_vessel_labels": ";".join(str(label) for label in case.missing_required_vessel_labels),
                    "recommended_action": case.recommended_action,
                    "notes": ";".join(case.notes),
                }
            )


def _write_requirements_csv(path: Path, required_labels: tuple[int, ...]) -> None:
    rows = [
        {
            "requirement_id": "primary_ct",
            "description": "Primary CT NIfTI or DICOM directory for anatomy, density, RT grid, and registration reference.",
            "minimum": "1 per case",
        },
        {
            "requirement_id": "vascular_image",
            "description": "CTA and/or CTV image for vessel provenance; paired CTA+CTV preferred for arterial/venous separation.",
            "minimum": "CTA or CTV per case; both preferred",
        },
        {
            "requirement_id": "organ_segmentation",
            "description": "Body/organ/material segmentation on the CT grid, including body, bone, lungs, liver, and kidneys.",
            "minimum": "1 co-registered mask per case",
        },
        {
            "requirement_id": "vessel_segmentation",
            "description": f"Branch-rich arterial/venous vessel segmentation with required labels: {', '.join(str(label) for label in required_labels)}.",
            "minimum": "1 co-registered labelled mask per case",
        },
        {
            "requirement_id": "access_and_provenance",
            "description": "Anonymized, access-approved data with source dataset ID, license/access status, and version.",
            "minimum": "documented per case",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["requirement_id", "description", "minimum"])
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, result: ValidationIntakeResult, cases_csv_path: str | Path | None) -> None:
    payload = {
        "intake_id": result.intake_id,
        "package_type": "p1_cta_ctv_validation_intake",
        "source_cases_csv": "" if cases_csv_path is None else str(cases_csv_path),
        "summary": {
            "case_count": result.case_count,
            "ready_case_count": result.ready_case_count,
            "review_case_count": result.review_case_count,
            "missing_case_count": result.missing_case_count,
            "required_vessel_labels": list(result.required_vessel_labels),
        },
        "outputs": {
            "template_csv": result.template_csv_path,
            "case_summary_csv": result.case_summary_csv_path,
            "dataset_requirements_csv": result.dataset_requirements_csv_path,
            "manifest_yaml": result.manifest_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "source_dataset": case.source_dataset,
                "readiness_status": case.readiness_status,
                "score_percent": case.score_percent,
                "vessel_label_coverage_percent": case.vessel_label_coverage_percent,
                "missing_required_vessel_labels": list(case.missing_required_vessel_labels),
                "recommended_action": case.recommended_action,
            }
            for case in result.cases
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_preview(path: Path, result: ValidationIntakeResult) -> None:
    plt = _import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    cases = list(result.cases)
    if cases:
        y = list(range(len(cases)))
        scores = [case.score_percent for case in cases]
        colors = [
            "#15803d" if case.readiness_status == "ready_for_p1_patient_specific_validation" else "#d97706" if case.score_percent >= 60 else "#b91c1c"
            for case in cases
        ]
        ax.barh(y, scores, color=colors)
        ax.set_yticks(y, labels=[case.case_id for case in cases])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("P1 intake readiness score (%)")
        ax.grid(axis="x", alpha=0.25)
        for index, case in enumerate(cases):
            ax.text(min(case.score_percent + 1.0, 98.0), index, case.readiness_status.replace("_", " "), va="center", fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No candidate cases supplied yet", ha="center", va="center", fontsize=12)
    ax.set_title("CTA/CTV Validation Intake Readiness")

    axes[1].axis("off")
    text = "\n".join(
        [
            "P1 Intake Summary",
            "",
            f"Intake ID: {result.intake_id}",
            f"Cases: {result.case_count}",
            f"Ready: {result.ready_case_count}",
            f"Review: {result.review_case_count}",
            f"Missing: {result.missing_case_count}",
            "",
            "Required data",
            "CT + organ seg",
            "CTA and/or CTV",
            "Branch-rich vessel labels",
            "Co-registered masks",
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
    fig.suptitle("P1 CTA/CTV Validation Cohort Intake", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_report(result: ValidationIntakeResult) -> str:
    image_rel = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    lines = [
        "# P1 CTA/CTV Validation Intake",
        "",
        f"Intake ID: `{result.intake_id}`",
        "",
        f"![P1 validation intake]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Candidate cases: {result.case_count}",
        f"- Ready / review / missing: {result.ready_case_count} / {result.review_case_count} / {result.missing_case_count}",
        f"- Required vessel labels: `{', '.join(str(label) for label in result.required_vessel_labels)}`",
        "",
        "## Outputs",
        "",
        f"- Case template CSV: `{result.template_csv_path}`",
        f"- Case summary CSV: `{result.case_summary_csv_path}`",
        f"- Dataset requirements CSV: `{result.dataset_requirements_csv_path}`",
        f"- Intake manifest: `{result.manifest_yaml_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        "",
    ]
    if result.cases:
        lines.extend(["## Case Readiness", "", "| case | status | score | label coverage | next action |", "| --- | --- | ---: | ---: | --- |"])
        for case in result.cases:
            lines.append(
                f"| `{case.case_id}` | `{case.readiness_status}` | {case.score_percent:.1f}% | "
                f"{case.vessel_label_coverage_percent:.1f}% | {case.recommended_action} |"
            )
    else:
        lines.extend(
            [
                "## Case Readiness",
                "",
                "- No candidate cases were supplied. Fill the template CSV with CT/CTA/CTV, organ segmentation, and vessel segmentation paths, then rerun this command.",
            ]
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_validation_intake_result(result: ValidationIntakeResult) -> str:
    return _format_report(result)


def build_validation_intake_package(
    cases_csv_path: str | Path | None = None,
    output_dir: str | Path = "outputs/releases/mode03_neg_stage007_rc1/validation_intake",
    intake_id: str = "mode03_neg_stage007_p1_cta_ctv_intake",
    required_vessel_labels: tuple[int, ...] | None = DEFAULT_REQUIRED_VESSEL_LABELS,
    report_path: str | Path | None = "outputs/reports/mode03_neg_stage007_p1_validation_intake.md",
) -> ValidationIntakeResult:
    resolved_required_labels = tuple(required_vessel_labels or DEFAULT_REQUIRED_VESSEL_LABELS)
    output = Path(output_dir)
    template_csv = output / f"{intake_id}_case_template_v001.csv"
    case_summary_csv = output / f"{intake_id}_case_summary_v001.csv"
    requirements_csv = output / f"{intake_id}_dataset_requirements_v001.csv"
    manifest_yaml = output / f"{intake_id}_manifest_v001.yaml"
    preview_png = output / f"{intake_id}_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{intake_id}_report_v001.md"

    rows = _read_cases_csv(cases_csv_path)
    cases = tuple(_case_from_row(row, resolved_required_labels) for row in rows)
    ready = sum(case.readiness_status == "ready_for_p1_patient_specific_validation" for case in cases)
    review = sum(case.readiness_status == "registration_or_label_review_required" for case in cases)
    missing = sum(case.readiness_status == "missing_required_data" for case in cases)
    notes = (
        "intake_does_not_copy_or_modify_candidate_medical_data",
        "ready_cases_can_feed_build-patient-phantom-adapter_and_run-patient-phantom-build",
        "p1_goal_is_patient_or_cohort_specific_vascular_grounding_not_clinical_clearance",
    )
    result = ValidationIntakeResult(
        intake_id=intake_id,
        output_dir=str(output),
        template_csv_path=str(template_csv),
        case_summary_csv_path=str(case_summary_csv),
        dataset_requirements_csv_path=str(requirements_csv),
        manifest_yaml_path=str(manifest_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        case_count=len(cases),
        ready_case_count=ready,
        review_case_count=review,
        missing_case_count=missing,
        required_vessel_labels=resolved_required_labels,
        cases=cases,
        notes=notes,
    )
    _write_template_csv(template_csv, resolved_required_labels)
    _write_case_summary_csv(case_summary_csv, cases)
    _write_requirements_csv(requirements_csv, resolved_required_labels)
    _write_manifest(manifest_yaml, result, cases_csv_path)
    _write_preview(preview_png, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
