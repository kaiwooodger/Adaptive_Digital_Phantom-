from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import yaml

from .validation_intake import DEFAULT_REQUIRED_VESSEL_LABELS, INTAKE_FIELDS


ROLE_FIELDS = ("ct_path", "vascular_image", "organ_seg_path", "vessel_seg_path")
DISCOVERY_FIELDS = [
    *INTAKE_FIELDS,
    "discovery_source",
    "discovery_status",
    "present_roles",
    "missing_roles",
    "discovery_notes",
]


@dataclass(frozen=True)
class ValidationCandidate:
    case_id: str
    source_dataset: str
    ct_path: str
    cta_path: str
    ctv_path: str
    organ_seg_path: str
    vessel_seg_path: str
    vessel_label_config: str
    required_vessel_labels: str
    access_status: str
    notes: str
    discovery_source: str
    discovery_status: str
    present_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]
    discovery_notes: tuple[str, ...]


@dataclass(frozen=True)
class ValidationDiscoveryResult:
    discovery_id: str
    search_roots: tuple[str, ...]
    output_dir: str
    candidates_csv_path: str
    summary_csv_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    candidate_count: int
    complete_candidate_count: int
    partial_candidate_count: int
    candidates: tuple[ValidationCandidate, ...]
    notes: tuple[str, ...]


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Validation candidate discovery preview requires matplotlib.") from exc
    return plt


def _is_nifti(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def _safe_yaml_load(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _iter_manifest_files(search_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for pattern in ("*.yaml", "*.yml", "*.csv"):
            files.extend(root.rglob(pattern))
    return tuple(sorted(dict.fromkeys(files)))


def _iter_nifti_files(search_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and _is_nifti(root):
            files.append(root)
            continue
        if root.is_dir():
            files.extend(path for path in root.rglob("*.nii*") if _is_nifti(path))
    return tuple(sorted(dict.fromkeys(files)))


def _path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def _present_and_missing_roles(candidate: dict[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present: list[str] = []
    missing: list[str] = []
    if _path_exists(candidate.get("ct_path", "")):
        present.append("ct_path")
    else:
        missing.append("ct_path")
    if _path_exists(candidate.get("cta_path", "")) or _path_exists(candidate.get("ctv_path", "")):
        present.append("vascular_image")
    else:
        missing.append("vascular_image")
    for role in ("organ_seg_path", "vessel_seg_path"):
        if _path_exists(candidate.get(role, "")):
            present.append(role)
        else:
            missing.append(role)
    return tuple(present), tuple(missing)


def _status_from_roles(present_roles: tuple[str, ...]) -> str:
    if set(present_roles) == set(ROLE_FIELDS):
        return "complete_local_candidate"
    if present_roles:
        return "partial_local_candidate"
    return "metadata_only_candidate"


def _required_label_string(required_labels: tuple[int, ...]) -> str:
    return ",".join(str(label) for label in required_labels)


def _candidate_from_row(
    row: dict[str, str],
    *,
    discovery_source: str,
    discovery_notes: tuple[str, ...],
    required_labels: tuple[int, ...],
) -> ValidationCandidate:
    base = {field: str(row.get(field, "") or "") for field in INTAKE_FIELDS}
    if not base["required_vessel_labels"]:
        base["required_vessel_labels"] = _required_label_string(required_labels)
    present, missing = _present_and_missing_roles(base)
    return ValidationCandidate(
        case_id=base["case_id"] or "candidate_case",
        source_dataset=base["source_dataset"],
        ct_path=base["ct_path"],
        cta_path=base["cta_path"],
        ctv_path=base["ctv_path"],
        organ_seg_path=base["organ_seg_path"],
        vessel_seg_path=base["vessel_seg_path"],
        vessel_label_config=base["vessel_label_config"],
        required_vessel_labels=base["required_vessel_labels"],
        access_status=base["access_status"] or "open_or_project_internal",
        notes=base["notes"],
        discovery_source=discovery_source,
        discovery_status=_status_from_roles(present),
        present_roles=present,
        missing_roles=missing,
        discovery_notes=discovery_notes,
    )


def _ct_org_candidates(path: Path, required_labels: tuple[int, ...], max_rows: int) -> tuple[ValidationCandidate, ...]:
    if path.name != "ct_org_label_population_manifest_v001.csv":
        return ()
    candidates: list[ValidationCandidate] = []
    try:
        with path.open(newline="") as csvfile:
            rows = list(csv.DictReader(csvfile))
    except Exception:
        return ()
    for row in rows[:max_rows]:
        material_path = str(row.get("material_label_path", "") or row.get("raw_label_path", "") or "")
        candidates.append(
            _candidate_from_row(
                {
                    "case_id": f"{row.get('case_id', 'ct_org_case')}_label_only",
                    "source_dataset": "ct_org_label_population",
                    "organ_seg_path": material_path,
                    "required_vessel_labels": _required_label_string(required_labels),
                    "access_status": "open",
                    "notes": "CT-ORG label-only staged anatomy candidate; primary CT and vessel segmentation are not present locally.",
                },
                discovery_source=str(path),
                discovery_notes=("known_ct_org_label_population_manifest", "anatomy_only_no_local_ct_or_vascular_case"),
                required_labels=required_labels,
            )
        )
    return tuple(candidates)


def _medseg_candidate(path: Path, required_labels: tuple[int, ...]) -> tuple[ValidationCandidate, ...]:
    if path.name != "medseg_abdominal_vasculature_case001_manifest_v001.yaml":
        return ()
    data = _safe_yaml_load(path)
    if not data:
        return ()
    return (
        _candidate_from_row(
            {
                "case_id": str(data.get("case_id", "medseg_abdominal_vasculature_case001")),
                "source_dataset": str(data.get("dataset", "medseg_abdominal_vasculature")),
                "cta_path": str(data.get("image_path", "")),
                "vessel_seg_path": str(data.get("mask_path", "")),
                "vessel_label_config": str(data.get("label_config_path", "")),
                "required_vessel_labels": _required_label_string(required_labels),
                "access_status": "open",
                "notes": "Branch-rich abdominal vessel template; no paired primary CT/material organ segmentation staged with this case.",
            },
            discovery_source=str(path),
            discovery_notes=("known_medseg_abdominal_vascular_manifest", "branch_rich_template_not_patient_specific_ct_ctv_pair"),
            required_labels=required_labels,
        ),
    )


def _patient_input_candidate(path: Path, required_labels: tuple[int, ...]) -> tuple[ValidationCandidate, ...]:
    if not path.name.endswith("_patient_input_manifest_v001.yaml"):
        return ()
    data = _safe_yaml_load(path)
    if data.get("package_type") != "patient_phantom_input_adapter":
        return ()
    role_paths = {"ct_path": "", "cta_path": "", "ctv_path": "", "organ_seg_path": "", "vessel_seg_path": ""}
    for item in data.get("inputs", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        staged = str(item.get("staged_path", "") or item.get("source_path", "") or "")
        if role == "ct":
            role_paths["ct_path"] = staged
        elif role == "cta":
            role_paths["cta_path"] = staged
        elif role == "ctv":
            role_paths["ctv_path"] = staged
        elif role == "organ_seg":
            role_paths["organ_seg_path"] = staged
        elif role == "vessel_seg":
            role_paths["vessel_seg_path"] = staged
    return (
        _candidate_from_row(
            {
                "case_id": str(data.get("case_id", path.stem)),
                "source_dataset": "patient_input_adapter_manifest",
                **role_paths,
                "required_vessel_labels": _required_label_string(required_labels),
                "access_status": "project_internal",
                "notes": "Existing adapter candidate; may be synthetic/project-derived and must be provenance-reviewed before P1 validation claims.",
            },
            discovery_source=str(path),
            discovery_notes=("known_patient_input_adapter_manifest", "review_provenance_before_using_as_real_validation_case"),
            required_labels=required_labels,
        ),
    )


def _staged_validation_case_candidate(path: Path, required_labels: tuple[int, ...]) -> tuple[ValidationCandidate, ...]:
    data = _safe_yaml_load(path)
    if data.get("package_type") != "p1_validation_case_staging":
        return ()
    role_paths = {"ct_path": "", "cta_path": "", "ctv_path": "", "organ_seg_path": "", "vessel_seg_path": ""}
    for item in data.get("inputs", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        staged = str(item.get("staged_path", "") or item.get("source_path", "") or "")
        if role == "ct":
            role_paths["ct_path"] = staged
        elif role == "cta":
            role_paths["cta_path"] = staged
        elif role == "ctv":
            role_paths["ctv_path"] = staged
        elif role == "organ_seg":
            role_paths["organ_seg_path"] = staged
        elif role == "vessel_seg":
            role_paths["vessel_seg_path"] = staged
    parsed_labels: list[int] = []
    for label in data.get("required_vessel_labels", []):
        try:
            parsed_labels.append(int(label))
        except (TypeError, ValueError):
            continue
    labels = tuple(dict.fromkeys(parsed_labels)) or required_labels
    return (
        _candidate_from_row(
            {
                "case_id": str(data.get("case_id", path.stem)),
                "source_dataset": str(data.get("source_dataset", "staged_validation_case")),
                **role_paths,
                "vessel_label_config": str(data.get("vessel_label_config", "")),
                "required_vessel_labels": _required_label_string(labels),
                "access_status": str(data.get("access_status", "local_review_required")),
                "notes": "Staged P1 validation case manifest; run intake QA before patient-specific build.",
            },
            discovery_source=str(path),
            discovery_notes=("known_p1_validation_case_staging_manifest", "manifest_generated_by_stage-validation-case"),
            required_labels=labels,
        ),
    )


def _role_from_nifti_name(path: Path) -> str | None:
    stem = path.name.lower().replace(".nii.gz", "").replace(".nii", "")
    parent = path.parent.name.lower()
    haystack = f"{parent}_{stem}"
    if any(token in haystack for token in ("cta", "arterial", "angio")):
        return "cta_path"
    if any(token in haystack for token in ("ctv", "venous")):
        return "ctv_path"
    if any(token in haystack for token in ("vessel", "vascular", "lumen")) or stem in {"msk", "mask"}:
        return "vessel_seg_path"
    if any(token in haystack for token in ("organ", "seg", "label", "material")):
        return "organ_seg_path"
    if any(token in haystack for token in ("ct", "image", "img", "hu")):
        return "ct_path"
    return None


def _loose_nifti_candidates(
    nifti_files: tuple[Path, ...],
    *,
    used_paths: set[str],
    required_labels: tuple[int, ...],
    max_cases: int,
) -> tuple[ValidationCandidate, ...]:
    if max_cases <= 0:
        return ()
    grouped: dict[Path, dict[str, str]] = {}
    for path in nifti_files:
        path_string = str(path)
        if path_string in used_paths:
            continue
        role = _role_from_nifti_name(path)
        if role is None:
            continue
        row = grouped.setdefault(
            path.parent,
            {
                "case_id": f"{path.parent.name}_loose_nifti",
                "source_dataset": "loose_nifti_folder",
                "required_vessel_labels": _required_label_string(required_labels),
                "access_status": "local_review_required",
                "notes": "Auto-grouped by folder and filename roles; verify provenance, anatomy region, and co-registration before use.",
            },
        )
        row.setdefault(role, path_string)
        if not row.get(role):
            row[role] = path_string

    candidates: list[ValidationCandidate] = []
    for folder, row in sorted(grouped.items(), key=lambda item: str(item[0])):
        role_count = sum(bool(row.get(role, "")) for role in ("ct_path", "cta_path", "ctv_path", "organ_seg_path", "vessel_seg_path"))
        if role_count < 2:
            continue
        candidates.append(
            _candidate_from_row(
                row,
                discovery_source=str(folder),
                discovery_notes=("loose_nifti_filename_grouping", "manual_review_required"),
                required_labels=required_labels,
            )
        )
        if len(candidates) >= max_cases:
            break
    return tuple(candidates)


def _dedupe_candidates(candidates: tuple[ValidationCandidate, ...]) -> tuple[ValidationCandidate, ...]:
    seen: dict[str, int] = {}
    output: list[ValidationCandidate] = []
    for candidate in candidates:
        key = candidate.case_id
        count = seen.get(key, 0)
        seen[key] = count + 1
        if count == 0:
            output.append(candidate)
            continue
        output.append(
            ValidationCandidate(
                **{
                    **candidate.__dict__,
                    "case_id": f"{candidate.case_id}_{count + 1:02d}",
                    "discovery_notes": (*candidate.discovery_notes, "case_id_deduplicated"),
                }
            )
        )
    return tuple(output)


def _candidate_to_csv_row(candidate: ValidationCandidate) -> dict[str, str]:
    return {
        "case_id": candidate.case_id,
        "source_dataset": candidate.source_dataset,
        "ct_path": candidate.ct_path,
        "cta_path": candidate.cta_path,
        "ctv_path": candidate.ctv_path,
        "organ_seg_path": candidate.organ_seg_path,
        "vessel_seg_path": candidate.vessel_seg_path,
        "vessel_label_config": candidate.vessel_label_config,
        "required_vessel_labels": candidate.required_vessel_labels,
        "access_status": candidate.access_status,
        "notes": candidate.notes,
        "discovery_source": candidate.discovery_source,
        "discovery_status": candidate.discovery_status,
        "present_roles": ";".join(candidate.present_roles),
        "missing_roles": ";".join(candidate.missing_roles),
        "discovery_notes": ";".join(candidate.discovery_notes),
    }


def _write_candidates_csv(path: Path, candidates: tuple[ValidationCandidate, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=DISCOVERY_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(_candidate_to_csv_row(candidate))


def _write_summary_csv(path: Path, candidates: tuple[ValidationCandidate, ...]) -> None:
    fields = [
        "case_id",
        "source_dataset",
        "discovery_status",
        "present_role_count",
        "missing_roles",
        "recommended_intake_action",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            if candidate.discovery_status == "complete_local_candidate":
                action = "Run build-validation-intake-package and patient-specific registration checks."
            elif candidate.vessel_seg_path and (candidate.cta_path or candidate.ctv_path):
                action = "Pair this vessel case with co-registered CT and organ/material labels."
            elif candidate.organ_seg_path:
                action = "Pair this anatomy case with primary CT and branch-rich CTA/CTV vessel labels."
            else:
                action = "Manual review required before intake scoring."
            writer.writerow(
                {
                    "case_id": candidate.case_id,
                    "source_dataset": candidate.source_dataset,
                    "discovery_status": candidate.discovery_status,
                    "present_role_count": len(candidate.present_roles),
                    "missing_roles": ";".join(candidate.missing_roles),
                    "recommended_intake_action": action,
                }
            )


def _write_manifest(path: Path, result: ValidationDiscoveryResult) -> None:
    payload = {
        "discovery_id": result.discovery_id,
        "package_type": "p1_validation_candidate_discovery",
        "search_roots": list(result.search_roots),
        "summary": {
            "candidate_count": result.candidate_count,
            "complete_candidate_count": result.complete_candidate_count,
            "partial_candidate_count": result.partial_candidate_count,
        },
        "outputs": {
            "candidates_csv": result.candidates_csv_path,
            "summary_csv": result.summary_csv_path,
            "manifest_yaml": result.manifest_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "candidates": [
            {
                "case_id": candidate.case_id,
                "source_dataset": candidate.source_dataset,
                "discovery_status": candidate.discovery_status,
                "present_roles": list(candidate.present_roles),
                "missing_roles": list(candidate.missing_roles),
                "discovery_source": candidate.discovery_source,
            }
            for candidate in result.candidates
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_preview(path: Path, result: ValidationDiscoveryResult) -> None:
    plt = _import_plotting()
    candidates = list(result.candidates)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * max(len(candidates), 1) + 2)))
    ax = axes[0]
    if candidates:
        y = list(range(len(candidates)))
        counts = [len(candidate.present_roles) for candidate in candidates]
        colors = ["#166534" if count == len(ROLE_FIELDS) else "#ca8a04" if count >= 2 else "#991b1b" for count in counts]
        ax.barh(y, counts, color=colors)
        ax.set_yticks(y, labels=[candidate.case_id for candidate in candidates], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, len(ROLE_FIELDS))
        ax.set_xticks(range(len(ROLE_FIELDS) + 1))
        ax.set_xlabel("Existing required role count")
        ax.set_title("Discovered Candidate Completeness")
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No local candidates discovered", ha="center", va="center", fontsize=12)

    axes[1].axis("off")
    text = "\n".join(
        [
            "P1 Candidate Discovery",
            "",
            f"Discovery ID: {result.discovery_id}",
            f"Candidates: {result.candidate_count}",
            f"Complete: {result.complete_candidate_count}",
            f"Partial: {result.partial_candidate_count}",
            "",
            "Required roles",
            "CT",
            "CTA or CTV",
            "Organ/material labels",
            "Branch vessel labels",
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
    fig.suptitle("Local CTA/CTV Validation Candidate Discovery", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_report(result: ValidationDiscoveryResult) -> str:
    image_rel = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    lines = [
        "# P1 Validation Candidate Discovery",
        "",
        f"Discovery ID: `{result.discovery_id}`",
        "",
        f"![P1 candidate discovery]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Candidate cases: {result.candidate_count}",
        f"- Complete / partial: {result.complete_candidate_count} / {result.partial_candidate_count}",
        f"- Search roots: `{', '.join(result.search_roots)}`",
        "",
        "## Outputs",
        "",
        f"- Candidates CSV: `{result.candidates_csv_path}`",
        f"- Summary CSV: `{result.summary_csv_path}`",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        "",
    ]
    if result.candidates:
        lines.extend(["## Discovered Cases", "", "| case | source | status | present roles | missing roles |", "| --- | --- | --- | --- | --- |"])
        for candidate in result.candidates:
            lines.append(
                f"| `{candidate.case_id}` | `{candidate.source_dataset}` | `{candidate.discovery_status}` | "
                f"{', '.join(candidate.present_roles) or 'none'} | {', '.join(candidate.missing_roles) or 'none'} |"
            )
    else:
        lines.extend(["## Discovered Cases", "", "- No CT/CTA/CTV validation candidates were found in the selected roots."])
    lines.extend(
        [
            "",
            "## Recommended Next Command",
            "",
            "```bash",
            f"python -m phantom_twin.cli build-validation-intake-package --cases-csv {result.candidates_csv_path}",
            "```",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_validation_discovery_result(result: ValidationDiscoveryResult) -> str:
    return _format_report(result)


def discover_validation_candidates(
    search_roots: tuple[str | Path, ...] = (
        "data",
        "outputs/digital/patient_input_adapter_stage001",
    ),
    output_dir: str | Path = "outputs/releases/mode03_neg_stage007_rc1/validation_discovery",
    discovery_id: str = "mode03_neg_stage007_p1_candidate_discovery",
    required_vessel_labels: tuple[int, ...] | None = DEFAULT_REQUIRED_VESSEL_LABELS,
    max_ct_org_cases: int = 10,
    max_loose_nifti_cases: int = 20,
    report_path: str | Path | None = "outputs/reports/mode03_neg_stage007_p1_candidate_discovery.md",
) -> ValidationDiscoveryResult:
    resolved_roots = tuple(Path(root) for root in search_roots)
    resolved_required = tuple(required_vessel_labels or DEFAULT_REQUIRED_VESSEL_LABELS)
    output = Path(output_dir)
    candidates_csv = output / f"{discovery_id}_candidates_v001.csv"
    summary_csv = output / f"{discovery_id}_summary_v001.csv"
    manifest_yaml = output / f"{discovery_id}_manifest_v001.yaml"
    preview_png = output / f"{discovery_id}_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{discovery_id}_report_v001.md"

    manifest_files = _iter_manifest_files(resolved_roots)
    candidates: list[ValidationCandidate] = []
    for path in manifest_files:
        candidates.extend(_ct_org_candidates(path, resolved_required, max_ct_org_cases))
        candidates.extend(_medseg_candidate(path, resolved_required))
        candidates.extend(_patient_input_candidate(path, resolved_required))
        candidates.extend(_staged_validation_case_candidate(path, resolved_required))

    used_paths = {
        path
        for candidate in candidates
        for path in (candidate.ct_path, candidate.cta_path, candidate.ctv_path, candidate.organ_seg_path, candidate.vessel_seg_path)
        if path
    }
    candidates.extend(
        _loose_nifti_candidates(
            _iter_nifti_files(resolved_roots),
            used_paths=used_paths,
            required_labels=resolved_required,
            max_cases=max_loose_nifti_cases,
        )
    )
    discovered = _dedupe_candidates(tuple(candidates))
    complete = sum(candidate.discovery_status == "complete_local_candidate" for candidate in discovered)
    partial = sum(candidate.discovery_status == "partial_local_candidate" for candidate in discovered)
    notes = (
        "discovery_indexes_paths_only_and_does_not_copy_medical_data",
        "complete_candidates_still_require_intake_geometry_and_label_QA",
        "partial_candidates_identify_which_CT_CTA_CTV_or_segmentation_inputs_must_be_staged_next",
    )
    result = ValidationDiscoveryResult(
        discovery_id=discovery_id,
        search_roots=tuple(str(root) for root in resolved_roots),
        output_dir=str(output),
        candidates_csv_path=str(candidates_csv),
        summary_csv_path=str(summary_csv),
        manifest_yaml_path=str(manifest_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        candidate_count=len(discovered),
        complete_candidate_count=complete,
        partial_candidate_count=partial,
        candidates=discovered,
        notes=notes,
    )
    _write_candidates_csv(candidates_csv, discovered)
    _write_summary_csv(summary_csv, discovered)
    _write_manifest(manifest_yaml, result)
    _write_preview(preview_png, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
