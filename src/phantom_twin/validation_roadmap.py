from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import yaml


@dataclass(frozen=True)
class ValidationRoadmapTask:
    phase_id: str
    domain: str
    blocker_id: str
    priority: int
    objective: str
    dataset_requirement: str
    acceptance_gate: str
    deliverable: str
    recommended_command: str
    status: str


@dataclass(frozen=True)
class ValidationRoadmapResult:
    roadmap_id: str
    release_id: str
    case_id: str
    output_dir: str
    protocol_markdown_path: str
    tasks_csv_path: str
    acceptance_criteria_csv_path: str
    dataset_requirements_csv_path: str
    roadmap_yaml_path: str
    roadmap_png_path: str
    report_path: str
    task_count: int
    clinical_blocker_count: int
    high_priority_task_count: int
    tasks: tuple[ValidationRoadmapTask, ...]
    notes: tuple[str, ...]


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Validation roadmap plotting requires matplotlib.") from exc
    return plt


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    return data if isinstance(data, dict) else {}


def _read_csv_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or str(path) == "":
        return []
    resolved = Path(path)
    if not resolved.exists():
        return []
    with resolved.open(newline="") as csvfile:
        return [dict(row) for row in csv.DictReader(csvfile)]


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_path(raw_path: Any, reference_path: Path) -> Path:
    if raw_path is None or str(raw_path) == "":
        return Path("__missing_validation_roadmap_input__")
    path = Path(str(raw_path))
    if path.is_absolute() or path.exists():
        return path
    candidate = reference_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _template_for_blocker(domain: str, check_id: str, recommended_action: str) -> dict[str, str]:
    key = f"{domain}/{check_id}"
    templates = {
        "vascular_anatomy/patient_specific_vascular_template_limit": {
            "phase_id": "P1",
            "objective": "Replace remaining template-derived vessels with patient-specific or cohort-specific CTA/CTV centerlines.",
            "dataset_requirement": "At least 3 branch-rich abdominal CTA/CTV cases with arterial, venous, renal, iliac, celiac/hepatic/splenic labels and matching organ masks.",
            "acceptance_gate": ">= 95% required branch coverage, 0 radius-aware bone fail edges, 0 arterial/venous overlap voxels, connected arterial and venous domains.",
            "deliverable": "Patient-specific CTA/CTV vascular graph replacement report plus vessel-organ QA for every case.",
            "recommended_command": "build-patient-phantom-adapter -> run-patient-phantom-build --allow-template-vessels false -> qa-patient-phantom-build",
        },
        "flow_model/flow_physics_validation_gap": {
            "phase_id": "P2",
            "objective": "Calibrate and validate the vascular flow model beyond the current graph-coupled 1D research scaffold.",
            "dataset_requirement": "Per-case inlet/outlet flow targets, pressure targets, vessel radii, and at least one CFD or measured pressure-flow reference loop.",
            "acceptance_gate": "Mass residual <= 1e-4 mL/s, mean outlet split error <= 10%, pressure-flow curve error <= 15%, documented waveform phase and amplitude uncertainty.",
            "deliverable": "0D/1D calibration table, optional CFD comparison, pressure-flow plots, and uncertainty report.",
            "recommended_command": "build-flow-boundary-package -> build-flow-1d-model -> build-coupled-pulsatile-flow-model -> external CFD/bench comparison",
        },
        "radiotherapy/clinical_dose_engine_gap": {
            "phase_id": "P3",
            "objective": "Connect the RT bundle to a real TPS or Monte Carlo dose engine instead of synthetic dose patterns.",
            "dataset_requirement": "Exportable CT/material maps, RT structures, prescription, beam geometry, TPS or Monte Carlo dose outputs, and PyMedPhys-compatible dose grids.",
            "acceptance_gate": "3%/3mm gamma pass >= 95%, PTV D95 delta <= 2%, OAR mean-dose delta <= 3%, static-vs-pulsatile dose comparison reproduced from imported dose grids.",
            "deliverable": "TPS/MC dose import package, DVH comparison CSV, gamma QA report, and DICOM-RT handoff notes.",
            "recommended_command": "build-radiotherapy-qa-package -> build-rt-planning-bundle --export-dicom -> build-dose-gamma-qa on imported TPS/MC dose",
        },
        "population_validation/single_release_case_validation_gap": {
            "phase_id": "P4",
            "objective": "Scale the release/audit workflow from one case to a population validation cohort.",
            "dataset_requirement": "Minimum 10 segmented CT cases for anatomy; ideally paired CTA/CTV in a representative subset spanning BMI/waist/body habitus.",
            "acceptance_gate": ">= 90% cases research-ready, no unresolved fail checks, organ volume/centroid outliers reviewed, vascular/flow/RT metrics summarized by cohort.",
            "deliverable": "Multi-case release manifest index, cohort readiness table, population validation atlas, and outlier review report.",
            "recommended_command": "build-population-cohort -> generate-pca-mode-variants -> build-research-release-package per case -> audit-research-release-package per case",
        },
        "population_validation/anatomical_equivalence_gap": {
            "phase_id": "P5",
            "objective": "Quantify how anatomically equivalent the phantom is to real human CT/CTA anatomy.",
            "dataset_requirement": "Population CT/CTA segmentations with body, bone, lungs, liver, kidneys, major vessels, centroids, volumes, and landmark annotations.",
            "acceptance_gate": "Organ volume z-scores within target cohort envelope, centroid errors <= 20 mm for major organs, body-shape metrics within target anthropometric band, vessel-organ distances within validated bounds.",
            "deliverable": "Anatomical equivalence validation report with organ/body/vessel metrics, PCA coverage, and residual limitations.",
            "recommended_command": "build-population-cohort -> build-profile-envelope -> audit-research-release-package -> validation-roadmap cohort summary",
        },
    }
    fallback = {
        "phase_id": "PX",
        "objective": f"Resolve blocker `{key}`.",
        "dataset_requirement": "Evidence dataset required by blocker-specific investigation.",
        "acceptance_gate": "Blocker status changes from review/fail to pass with traceable evidence.",
        "deliverable": "Updated QA evidence and release-readiness audit.",
        "recommended_command": recommended_action or "Regenerate relevant package and rerun audit-research-release-package.",
    }
    return templates.get(key, fallback)


def _task_from_row(row: dict[str, str]) -> ValidationRoadmapTask:
    domain = row.get("domain", "unknown")
    check_id = row.get("check_id", "unknown")
    template = _template_for_blocker(domain, check_id, row.get("recommended_action", ""))
    return ValidationRoadmapTask(
        phase_id=template["phase_id"],
        domain=domain,
        blocker_id=check_id,
        priority=_safe_int(row.get("priority"), 9),
        objective=template["objective"],
        dataset_requirement=template["dataset_requirement"],
        acceptance_gate=template["acceptance_gate"],
        deliverable=template["deliverable"],
        recommended_command=template["recommended_command"],
        status="planned",
    )


def _write_tasks_csv(path: Path, tasks: tuple[ValidationRoadmapTask, ...]) -> None:
    fields = [
        "phase_id",
        "priority",
        "domain",
        "blocker_id",
        "objective",
        "dataset_requirement",
        "acceptance_gate",
        "deliverable",
        "recommended_command",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow({field: getattr(task, field) for field in fields})


def _write_acceptance_csv(path: Path, tasks: tuple[ValidationRoadmapTask, ...]) -> None:
    fields = ["phase_id", "domain", "blocker_id", "acceptance_gate", "evidence_required", "pass_condition"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "phase_id": task.phase_id,
                    "domain": task.domain,
                    "blocker_id": task.blocker_id,
                    "acceptance_gate": task.acceptance_gate,
                    "evidence_required": task.deliverable,
                    "pass_condition": "Release-readiness audit blocker can be downgraded or removed with documented evidence.",
                }
            )


def _write_dataset_csv(path: Path, tasks: tuple[ValidationRoadmapTask, ...]) -> None:
    fields = ["phase_id", "domain", "blocker_id", "dataset_requirement", "minimum_cases", "notes"]
    minimum_cases_by_phase = {"P1": "3", "P2": "1+", "P3": "1+", "P4": "10", "P5": "10-30"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "phase_id": task.phase_id,
                    "domain": task.domain,
                    "blocker_id": task.blocker_id,
                    "dataset_requirement": task.dataset_requirement,
                    "minimum_cases": minimum_cases_by_phase.get(task.phase_id, "TBD"),
                    "notes": "Use anonymized, access-approved data only; keep source IDs/versioning in every report.",
                }
            )


def _write_yaml(path: Path, result: ValidationRoadmapResult, readiness_audit_path: Path, roadmap_csv_path: Path) -> None:
    payload = {
        "roadmap_id": result.roadmap_id,
        "release_id": result.release_id,
        "case_id": result.case_id,
        "package_type": "clinical_validation_gap_closure_roadmap",
        "source_readiness_audit": str(readiness_audit_path),
        "source_blocker_roadmap_csv": str(roadmap_csv_path),
        "summary": {
            "task_count": result.task_count,
            "clinical_blocker_count": result.clinical_blocker_count,
            "high_priority_task_count": result.high_priority_task_count,
        },
        "outputs": {
            "protocol_markdown": result.protocol_markdown_path,
            "tasks_csv": result.tasks_csv_path,
            "acceptance_criteria_csv": result.acceptance_criteria_csv_path,
            "dataset_requirements_csv": result.dataset_requirements_csv_path,
            "roadmap_yaml": result.roadmap_yaml_path,
            "roadmap_png": result.roadmap_png_path,
            "report": result.report_path,
        },
        "tasks": [
            {
                "phase_id": task.phase_id,
                "priority": task.priority,
                "domain": task.domain,
                "blocker_id": task.blocker_id,
                "objective": task.objective,
                "dataset_requirement": task.dataset_requirement,
                "acceptance_gate": task.acceptance_gate,
                "deliverable": task.deliverable,
                "recommended_command": task.recommended_command,
                "status": task.status,
            }
            for task in result.tasks
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_protocol(path: Path, result: ValidationRoadmapResult) -> None:
    lines = [
        "# Clinical Validation Gap-Closure Protocol",
        "",
        f"Roadmap ID: `{result.roadmap_id}`",
        f"Release ID: `{result.release_id}`",
        f"Case ID: `{result.case_id}`",
        "",
        "## Purpose",
        "",
        "This protocol converts the release-readiness audit blockers into executable validation work. It is a research and engineering protocol, not a regulatory submission or clinical-use clearance.",
        "",
        "## Phase Plan",
        "",
        "| phase | domain | objective | acceptance gate | deliverable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for task in result.tasks:
        lines.append(
            f"| {task.phase_id} | {task.domain} | {task.objective} | {task.acceptance_gate} | {task.deliverable} |"
        )
    lines.extend(
        [
            "",
            "## Execution Rules",
            "",
            "- Every dataset must be anonymized, source-versioned, and traceable to an access-approved location.",
            "- Every validation rerun must produce a release package and a release-readiness audit before being treated as evidence.",
            "- Clinical language remains blocked until all clinical-blocker tasks are supported by external evidence, not just internal synthetic QA.",
            "- Failures should create a new corrective task rather than silently changing thresholds.",
            "",
            "## Recommended Order",
            "",
        ]
    )
    for task in sorted(result.tasks, key=lambda item: (item.priority, item.phase_id)):
        lines.append(f"- `{task.phase_id}`: {task.objective}")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_png(path: Path, result: ValidationRoadmapResult) -> None:
    plt = _import_plotting()
    tasks = sorted(result.tasks, key=lambda item: (item.priority, item.phase_id))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 1.0 + 0.85 * len(tasks))), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    y_positions = list(range(len(tasks)))
    colors = {
        "vascular_anatomy": "#0f766e",
        "flow_model": "#2563eb",
        "radiotherapy": "#b45309",
        "population_validation": "#7c3aed",
    }
    ax.barh(y_positions, [1] * len(tasks), color=[colors.get(task.domain, "#64748b") for task in tasks])
    ax.set_yticks(y_positions, labels=[f"{task.phase_id}  {task.domain.replace('_', ' ')}" for task in tasks])
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Clinical Blocker Closure Phases")
    for index, task in enumerate(tasks):
        ax.text(0.03, index, task.blocker_id.replace("_", " "), va="center", ha="left", color="white", fontsize=9)

    axes[1].axis("off")
    text = "\n".join(
        [
            "Roadmap Summary",
            "",
            f"Release: {result.release_id}",
            f"Tasks: {result.task_count}",
            f"Clinical blockers: {result.clinical_blocker_count}",
            f"High priority: {result.high_priority_task_count}",
            "",
            "First moves",
            "1. Patient-specific CTA/CTV vessels",
            "2. Flow calibration/CFD reference",
            "3. TPS or Monte Carlo dose import",
            "4. Multi-case cohort audit",
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
    fig.suptitle("Validation Roadmap for Clinical-Claim Gap Closure", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_report(result: ValidationRoadmapResult) -> str:
    image_rel = os.path.relpath(result.roadmap_png_path, start=Path(result.report_path).parent)
    lines = [
        "# Clinical Validation Roadmap",
        "",
        f"Roadmap ID: `{result.roadmap_id}`",
        f"Release ID: `{result.release_id}`",
        f"Case ID: `{result.case_id}`",
        "",
        f"![Validation roadmap]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Clinical blocker tasks: {result.clinical_blocker_count}",
        f"- Total planned tasks: {result.task_count}",
        f"- High-priority tasks: {result.high_priority_task_count}",
        "",
        "## Planned Phases",
        "",
    ]
    for task in result.tasks:
        lines.append(f"- `{task.phase_id}` `{task.domain}`: {task.objective}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Protocol: `{result.protocol_markdown_path}`",
            f"- Tasks CSV: `{result.tasks_csv_path}`",
            f"- Acceptance criteria CSV: `{result.acceptance_criteria_csv_path}`",
            f"- Dataset requirements CSV: `{result.dataset_requirements_csv_path}`",
            f"- Roadmap YAML: `{result.roadmap_yaml_path}`",
            f"- Roadmap PNG: `{result.roadmap_png_path}`",
            "",
            "## Interpretation",
            "",
            "- The release is research-ready, but these tasks define what must be validated before making stronger clinical-equivalence claims.",
            "- The first highest-leverage data step is patient-specific or cohort-specific CTA/CTV vessel segmentation with organ labels.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_validation_roadmap_result(result: ValidationRoadmapResult) -> str:
    return _format_report(result)


def build_validation_roadmap(
    readiness_audit_yaml_path: str | Path,
    roadmap_csv_path: str | Path | None = None,
    output_dir: str | Path = "outputs/releases/mode03_neg_stage007_rc1/validation_roadmap",
    roadmap_id: str | None = None,
    report_path: str | Path | None = "outputs/reports/mode03_neg_stage007_validation_roadmap.md",
) -> ValidationRoadmapResult:
    audit_path = Path(readiness_audit_yaml_path)
    if not audit_path.exists():
        raise FileNotFoundError(f"Readiness audit YAML does not exist: {audit_path}")
    audit = _load_yaml(audit_path)
    release_id = str(audit.get("release_id", "unknown_release"))
    case_id = str(audit.get("case_id", "unknown_case"))
    roadmap = roadmap_id or f"{release_id}_validation_roadmap"
    outputs = _as_mapping(audit.get("outputs"))
    source_roadmap = _resolve_path(roadmap_csv_path or outputs.get("roadmap_csv"), audit_path)
    rows = _read_csv_rows(source_roadmap)
    if not rows:
        rows = [
            {
                "priority": "1",
                "domain": str(check.get("domain", "unknown")),
                "check_id": str(check.get("check_id", "unknown")),
                "clinical_blocker": str(check.get("clinical_blocker", False)),
                "recommended_action": str(check.get("recommended_action", "")),
            }
            for check in audit.get("checks", [])
            if isinstance(check, dict) and bool(check.get("clinical_blocker", False))
        ]
    tasks = tuple(sorted((_task_from_row(row) for row in rows), key=lambda item: (item.priority, item.phase_id)))

    output = Path(output_dir)
    protocol = output / f"{roadmap}_protocol_v001.md"
    tasks_csv = output / f"{roadmap}_tasks_v001.csv"
    acceptance_csv = output / f"{roadmap}_acceptance_criteria_v001.csv"
    datasets_csv = output / f"{roadmap}_dataset_requirements_v001.csv"
    roadmap_yaml = output / f"{roadmap}_roadmap_v001.yaml"
    roadmap_png = output / f"{roadmap}_roadmap_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{roadmap}_report_v001.md"
    notes = (
        "roadmap_closes_clinical_claim_blockers_not_research_build_failures",
        "dataset_requirements_are_minimum_engineering_targets_not_regulatory_sample_size_claims",
        "all_validation_outputs_should_feed_back_into_release_package_and_readiness_audit",
    )
    result = ValidationRoadmapResult(
        roadmap_id=roadmap,
        release_id=release_id,
        case_id=case_id,
        output_dir=str(output),
        protocol_markdown_path=str(protocol),
        tasks_csv_path=str(tasks_csv),
        acceptance_criteria_csv_path=str(acceptance_csv),
        dataset_requirements_csv_path=str(datasets_csv),
        roadmap_yaml_path=str(roadmap_yaml),
        roadmap_png_path=str(roadmap_png),
        report_path=str(report),
        task_count=len(tasks),
        clinical_blocker_count=len(tasks),
        high_priority_task_count=sum(task.priority <= 1 for task in tasks),
        tasks=tasks,
        notes=notes,
    )
    _write_tasks_csv(tasks_csv, tasks)
    _write_acceptance_csv(acceptance_csv, tasks)
    _write_dataset_csv(datasets_csv, tasks)
    _write_protocol(protocol, result)
    _write_yaml(roadmap_yaml, result, audit_path, source_roadmap)
    _write_png(roadmap_png, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
