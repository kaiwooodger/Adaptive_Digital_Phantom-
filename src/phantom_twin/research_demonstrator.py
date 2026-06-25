from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import shutil
from typing import Any

import yaml


@dataclass(frozen=True)
class DemonstratorFigure:
    figure_id: str
    title: str
    caption: str
    source_path: str
    packaged_path: str
    status: str


@dataclass(frozen=True)
class DemonstratorMetric:
    category: str
    metric: str
    value: str
    unit: str
    interpretation: str
    source: str


@dataclass(frozen=True)
class ResearchDemonstratorResult:
    package_id: str
    case_id: str
    release_id: str
    readiness_status: str
    output_dir: str
    manifest_yaml_path: str
    report_path: str
    manuscript_outline_path: str
    figure_atlas_path: str
    figure_index_csv_path: str
    metrics_csv_path: str
    command_log_path: str
    limitations_path: str
    summary: dict[str, Any]
    figures: tuple[DemonstratorFigure, ...]
    metrics: tuple[DemonstratorMetric, ...]
    notes: tuple[str, ...]


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Research demonstrator figure atlas generation requires matplotlib.") from exc
    return plt


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text())
    return data if isinstance(data, dict) else {}


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, default)))


def _resolve_path(raw_path: Any, base_path: str | Path | None = None) -> Path:
    if raw_path is None or str(raw_path) == "":
        return Path("__missing__")
    path = Path(str(raw_path))
    if path.is_absolute() or path.exists():
        return path
    if base_path is not None:
        candidate = Path(base_path).parent / path
        if candidate.exists():
            return candidate
    return path


def _copy_artifact(source: Path, destination_dir: Path, figure_id: str) -> str:
    if not source.exists() or source.is_dir():
        return ""
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if source.name.endswith(".nii.gz") else source.suffix
    destination = destination_dir / f"{figure_id}{suffix}"
    shutil.copy2(source, destination)
    return str(destination)


def _format_number(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    if 0 < abs(number) < 1e-3:
        return f"{number:.3e}"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.{digits}f}"


def _metric(
    metrics: list[DemonstratorMetric],
    category: str,
    metric: str,
    value: Any,
    unit: str,
    interpretation: str,
    source: str | Path,
    *,
    digits: int = 3,
) -> None:
    if isinstance(value, str):
        formatted = value
    elif isinstance(value, int):
        formatted = str(value)
    else:
        formatted = _format_number(value, digits=digits)
    metrics.append(
        DemonstratorMetric(
            category=category,
            metric=metric,
            value=formatted,
            unit=unit,
            interpretation=interpretation,
            source=str(source),
        )
    )


def _collect_metrics(
    *,
    release_manifest: dict[str, Any],
    release_manifest_path: Path,
    audit: dict[str, Any],
    audit_path: Path,
    roadmap: dict[str, Any],
    roadmap_path: Path,
    validation_intake: dict[str, Any],
    validation_intake_path: Path,
    vessel_harmonization: dict[str, Any],
    vessel_harmonization_path: Path,
) -> tuple[DemonstratorMetric, ...]:
    metrics: list[DemonstratorMetric] = []
    release_summary = _as_mapping(release_manifest.get("summary"))
    status = _as_mapping(release_summary.get("status_summary"))
    vascular = _as_mapping(status.get("vascular_domain"))
    rt_material = _as_mapping(status.get("rt_material_package"))
    flow = _as_mapping(status.get("flow"))
    flow4d = _as_mapping(status.get("flow4d"))
    rt_flow = _as_mapping(status.get("rt_flow"))
    gamma = _as_mapping(status.get("gamma"))
    audit_scores = _as_mapping(audit.get("scores"))
    roadmap_summary = _as_mapping(roadmap.get("summary"))
    intake_summary = _as_mapping(validation_intake.get("summary"))
    vessel_summary = _as_mapping(vessel_harmonization.get("summary"))
    checks = _as_list(audit.get("checks"))
    pass_count = sum(check.get("status") == "pass" for check in checks if isinstance(check, dict))
    review_count = sum(check.get("status") == "review" for check in checks if isinstance(check, dict))
    fail_count = sum(check.get("status") == "fail" for check in checks if isinstance(check, dict))

    _metric(metrics, "release", "indexed artifacts", release_summary.get("artifact_count"), "count", "Disk-light demonstrator artifact index.", release_manifest_path)
    _metric(metrics, "release", "copied small artifacts", release_summary.get("copied_artifact_count"), "count", "Small reports, tables, and figures copied into release.", release_manifest_path)
    _metric(metrics, "release", "missing artifacts", release_summary.get("missing_artifact_count"), "count", "Should remain zero for a publishable package.", release_manifest_path)
    _metric(metrics, "readiness", "overall score", audit_scores.get("overall_score_percent"), "%", "Engineering readiness including clinical blockers.", audit_path)
    _metric(metrics, "readiness", "research readiness score", audit_scores.get("research_score_percent"), "%", "Research-demonstrator score excluding clinical-claim blockers.", audit_path)
    _metric(metrics, "readiness", "clinical blocker count", audit_scores.get("clinical_blocker_count"), "count", "Remaining blockers before clinical-equivalence claims.", audit_path)
    _metric(metrics, "readiness", "audit pass/review/fail", f"{pass_count}/{review_count}/{fail_count}", "count", "Readiness audit status distribution.", audit_path)

    _metric(metrics, "vascular", "arterial voxels", vascular.get("arterial_voxels"), "voxels", "Corrected arterial lumen domain.", release_manifest_path)
    _metric(metrics, "vascular", "venous voxels", vascular.get("venous_voxels"), "voxels", "Corrected venous lumen domain.", release_manifest_path)
    _metric(metrics, "vascular", "vessel wall voxels", vascular.get("vessel_wall_voxels"), "voxels", "Derived vessel wall/material shell.", release_manifest_path)
    _metric(metrics, "vascular", "snapped boundary nodes", vascular.get("snapped_boundary_nodes"), "nodes", "Inlet/outlet nodes snapped to branch-labelled lumen.", release_manifest_path)
    _metric(metrics, "vascular", "unclassified labels", len(_as_list(vascular.get("unclassified_labels"))), "count", "Should be zero for the corrected branch label map.", release_manifest_path)

    _metric(metrics, "flow", "graph nodes", flow.get("node_count"), "nodes", "Coupled 1D vascular graph size.", release_manifest_path)
    _metric(metrics, "flow", "graph edges", flow.get("edge_count"), "edges", "Coupled 1D vascular graph size.", release_manifest_path)
    _metric(metrics, "flow", "boundary conditions", flow.get("boundary_count"), "boundaries", "Inlet/outlet boundary-condition count.", release_manifest_path)
    _metric(metrics, "flow", "aorta flow mean", flow.get("aorta_flow_mean_ml_s"), "mL/s", "Pulsatile inlet mean flow.", release_manifest_path)
    _metric(metrics, "flow", "aorta flow min/max", f"{_format_number(flow.get('aorta_flow_min_ml_s'))}/{_format_number(flow.get('aorta_flow_max_ml_s'))}", "mL/s", "Pulsatile waveform range.", release_manifest_path)
    _metric(metrics, "flow", "mass balance residual", flow.get("max_mass_balance_residual_ml_s"), "mL/s", "Numerical conservation check.", release_manifest_path)
    _metric(metrics, "flow", "4D flow frames", flow4d.get("frame_count"), "frames", "Temporal flow visualization states.", release_manifest_path)

    _metric(metrics, "radiotherapy", "vascular fluid volume", rt_material.get("vascular_fluid_volume_cm3"), "cm3", "RT material volume for blood/contrast-fluid region.", release_manifest_path)
    _metric(metrics, "radiotherapy", "vessel wall volume", rt_material.get("vessel_wall_volume_cm3"), "cm3", "RT material volume for vessel wall.", release_manifest_path)
    _metric(metrics, "radiotherapy", "PTV volume", rt_material.get("ptv_volume_cm3"), "cm3", "Synthetic RT target volume used for QA.", release_manifest_path)
    _metric(metrics, "radiotherapy", "spatial RT-flow edges", rt_flow.get("selected_edge_count"), "edges", "Vascular edges coupled into spatial dose perturbation.", release_manifest_path)
    _metric(metrics, "radiotherapy", "peak/trough dose delta", f"{_format_number(rt_flow.get('max_peak_delta_mgy'))}/{_format_number(rt_flow.get('max_trough_delta_mgy'))}", "mGy", "Maximum spatial flow-dose perturbation range.", release_manifest_path)
    _metric(metrics, "radiotherapy", "gamma min pass rate", gamma.get("min_pass_rate_percent"), "%", "Spatial dose gamma QA lower-bound pass rate.", release_manifest_path)

    _metric(metrics, "validation", "roadmap tasks", roadmap_summary.get("task_count"), "tasks", "Planned blocker-closure tasks.", roadmap_path)
    _metric(metrics, "validation", "validation intake ready/review/missing", f"{intake_summary.get('ready_case_count', 0)}/{intake_summary.get('review_case_count', 0)}/{intake_summary.get('missing_case_count', 0)}", "cases", "Local case readiness status.", validation_intake_path)
    _metric(metrics, "validation", "harmonized vessel label coverage", vessel_summary.get("vessel_label_coverage_percent"), "%", "Coverage for required abdominal vessel labels.", vessel_harmonization_path)
    _metric(metrics, "validation", "mapped vessel labels", f"{vessel_summary.get('mapped_source_label_count', 0)}/{vessel_summary.get('source_label_count', 0)}", "labels", "Identity harmonization status for MedSeg branch-rich case.", vessel_harmonization_path)
    return tuple(metrics)


def _figure_sources(
    *,
    status_manifest: dict[str, Any],
    status_manifest_path: Path,
    audit: dict[str, Any],
    audit_path: Path,
    roadmap: dict[str, Any],
    roadmap_path: Path,
    validation_intake: dict[str, Any],
    validation_intake_path: Path,
    vessel_harmonization: dict[str, Any],
    vessel_harmonization_path: Path,
) -> tuple[tuple[str, str, str, Path], ...]:
    status_outputs = _as_mapping(status_manifest.get("outputs"))
    status_artifacts = _as_mapping(status_manifest.get("artifacts"))
    audit_outputs = _as_mapping(audit.get("outputs"))
    roadmap_outputs = _as_mapping(roadmap.get("outputs"))
    intake_outputs = _as_mapping(validation_intake.get("outputs"))
    vessel_outputs = _as_mapping(vessel_harmonization.get("outputs"))

    def artifact_path(key: str) -> Path:
        payload = _as_mapping(status_artifacts.get(key))
        return _resolve_path(payload.get("path"), status_manifest_path)

    return (
        (
            "fig01_status_atlas",
            "Current Corrected Phantom",
            "Consolidated corrected-branch phantom status atlas showing flow domain, RT geometry, flow, and QA outputs.",
            _resolve_path(status_outputs.get("atlas_png"), status_manifest_path),
        ),
        (
            "fig02_flow_domain",
            "Branch-Labelled Flow Domain",
            "Corrected CT-grid vessel-label flow domain with arterial, venous, wall, and boundary information.",
            artifact_path("flow_domain_preview"),
        ),
        (
            "fig03_pulsatile_flow",
            "Coupled Pulsatile Flow",
            "Pressure-flow preview for the graph-coupled pulsatile vascular solver.",
            artifact_path("coupled_flow_preview"),
        ),
        (
            "fig04_flow4d",
            "4D Flow Visualization",
            "Temporal contact sheet of the velocity-colored vascular flow state.",
            artifact_path("flow4d_contact_sheet"),
        ),
        (
            "fig05_rt_materials",
            "RT Material Package",
            "Radiotherapy QA material preview with corrected vascular fluid and vessel wall regions.",
            artifact_path("rt_qa_preview"),
        ),
        (
            "fig06_spatial_rt_flow",
            "Spatial RT-Flow Coupling",
            "Spatial coupling between RT target geometry and nearby vascular graph edges.",
            artifact_path("spatial_coupling_preview"),
        ),
        (
            "fig07_spatial_dose",
            "Flow-Modulated Dose",
            "Synthetic spatial flow-dose perturbation preview for static, peak, and trough states.",
            artifact_path("spatial_dose_preview"),
        ),
        (
            "fig08_gamma_qa",
            "Dose Gamma QA",
            "Gamma QA preview comparing static and pulsatile spatial dose states.",
            artifact_path("gamma_qa_preview"),
        ),
        (
            "fig09_readiness",
            "Release Readiness",
            "Corrected branch release scorecard separating research readiness from clinical blockers.",
            _resolve_path(audit_outputs.get("scorecard_png"), audit_path),
        ),
        (
            "fig10_validation_roadmap",
            "Validation Roadmap",
            "Clinical-claim blocker closure roadmap for patient-specific validation, flow validation, and RT dose validation.",
            _resolve_path(roadmap_outputs.get("roadmap_png"), roadmap_path),
        ),
        (
            "fig11_validation_intake",
            "Validation Intake",
            "Current P1 validation intake status for the harmonized branch-rich MedSeg partial validation case.",
            _resolve_path(intake_outputs.get("preview_png"), validation_intake_path),
        ),
        (
            "fig12_vessel_harmonization",
            "Vessel Label Harmonization",
            "Identity harmonization of the branch-rich MedSeg vessel labels into the P1 abdominal vessel label schema.",
            _resolve_path(vessel_outputs.get("preview_png"), vessel_harmonization_path),
        ),
    )


def _stage_figures(
    sources: tuple[tuple[str, str, str, Path], ...],
    figures_dir: Path,
) -> tuple[DemonstratorFigure, ...]:
    figures: list[DemonstratorFigure] = []
    for figure_id, title, caption, source in sources:
        packaged = _copy_artifact(source, figures_dir, figure_id)
        status = "present" if source.exists() and packaged else "missing"
        figures.append(
            DemonstratorFigure(
                figure_id=figure_id,
                title=title,
                caption=caption,
                source_path=str(source),
                packaged_path=packaged,
                status=status,
            )
        )
    return tuple(figures)


def _write_figure_index(path: Path, figures: tuple[DemonstratorFigure, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("figure_id", "title", "caption", "source_path", "packaged_path", "status")
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for figure in figures:
            writer.writerow(
                {
                    "figure_id": figure.figure_id,
                    "title": figure.title,
                    "caption": figure.caption,
                    "source_path": figure.source_path,
                    "packaged_path": figure.packaged_path,
                    "status": figure.status,
                }
            )


def _write_metrics_csv(path: Path, metrics: tuple[DemonstratorMetric, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("category", "metric", "value", "unit", "interpretation", "source")
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "category": metric.category,
                    "metric": metric.metric,
                    "value": metric.value,
                    "unit": metric.unit,
                    "interpretation": metric.interpretation,
                    "source": metric.source,
                }
            )


def _write_figure_atlas(path: Path, result: ResearchDemonstratorResult) -> None:
    plt = _import_plotting()
    fig = plt.figure(figsize=(19, 16))
    grid = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 0.95])
    axes = [fig.add_subplot(grid[row, col]) for row in range(4) for col in range(4)]
    fig.suptitle(
        f"Publishable Research Demonstrator\n{result.case_id}",
        fontsize=18,
        fontweight="bold",
    )
    for ax, figure in zip(axes[:12], result.figures[:12], strict=False):
        ax.axis("off")
        ax.set_title(figure.title, fontsize=10, fontweight="bold")
        image_path = Path(figure.packaged_path)
        if figure.status == "present" and image_path.exists():
            try:
                ax.imshow(plt.imread(image_path))
            except Exception:
                ax.text(0.5, 0.5, "unreadable image", ha="center", va="center", color="#9f1239")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", color="#9f1239")

    summary_ax = axes[12]
    summary_ax.axis("off")
    summary = result.summary
    summary_text = "\n".join(
        [
            "Demonstrator Readiness",
            "",
            f"Status: {result.readiness_status}",
            f"Research score: {summary.get('research_score_percent', 0.0):.1f}%",
            f"Overall score: {summary.get('overall_score_percent', 0.0):.1f}%",
            f"Audit pass/review/fail: {summary.get('audit_pass_count', 0)} / {summary.get('audit_review_count', 0)} / {summary.get('audit_fail_count', 0)}",
            f"Figures present: {summary.get('present_figure_count', 0)} / {summary.get('figure_count', 0)}",
            f"Metrics table rows: {summary.get('metric_count', 0)}",
            "",
            "Included Evidence",
            "CT/material phantom",
            "branch-labelled arteries/veins",
            "pulsatile 1D/RCR flow",
            "4D flow visualization",
            "RT-flow dose QA",
            "validation roadmap/intake",
        ]
    )
    summary_ax.text(
        0.02,
        0.98,
        summary_text,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "#f8fafc", "edgecolor": "#94a3b8", "boxstyle": "round,pad=0.6"},
    )

    limits_ax = axes[13]
    limits_ax.axis("off")
    limits_ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                "Scope Guardrails",
                "",
                "Research demonstrator only.",
                "Not a clinical device.",
                "Not a patient-specific twin.",
                "No TPS/Monte Carlo dose yet.",
                "No calibrated 3D CFD/FSI yet.",
                "Complete paired CT+CTA/CTV validation",
                "case still required.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "#fff7ed", "edgecolor": "#fdba74", "boxstyle": "round,pad=0.6"},
    )

    flow_ax = axes[14]
    flow_ax.axis("off")
    flow_ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                "Reproducible Pipeline",
                "",
                "1. Correct vessel labels.",
                "2. Build flow domains.",
                "3. Run boundary conditions.",
                "4. Run coupled pulsatile flow.",
                "5. Generate 4D visualization.",
                "6. Build RT QA/planning bundle.",
                "7. Run spatial dose + gamma QA.",
                "8. Package, audit, roadmap.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "#ecfeff", "edgecolor": "#67e8f9", "boxstyle": "round,pad=0.6"},
    )

    next_ax = axes[15]
    next_ax.axis("off")
    next_ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                "Next Research Milestone",
                "",
                "Stage one complete paired case:",
                "CT + organ/material labels + CTA/CTV",
                "or branch-labelled vessel mask.",
                "",
                "Then rerun CT-grid resampling,",
                "vessel-organ QA, flow, RT-flow QA,",
                "release packaging, and this",
                "demonstrator package.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "#f0fdf4", "edgecolor": "#86efac", "boxstyle": "round,pad=0.6"},
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_command_log(path: Path, result: ResearchDemonstratorResult, inputs: dict[str, Path]) -> None:
    prefix = "TMPDIR=.tmp PYTHONPATH=.deps/python:src python -m phantom_twin.cli"
    lines = [
        "# Research Demonstrator Reproducibility Commands",
        "",
        "These commands reproduce the demonstrator package from the current corrected branch release artifacts.",
        "",
        "```bash",
        f"{prefix} build-corrected-branch-status-report",
        f"{prefix} build-corrected-branch-release-package",
        f"{prefix} audit-corrected-branch-release-package",
        (
            f"{prefix} build-validation-roadmap --readiness-audit {inputs['audit']} "
            f"--roadmap-csv outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/readiness_audit/"
            "mode03_neg_branch_ctgrid_corrected_flow_rc1_readiness_audit_roadmap_v001.csv "
            "--output-dir outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/validation_roadmap "
            "--roadmap-id mode03_neg_branch_ctgrid_corrected_flow_rc1_validation_roadmap "
            "--report outputs/reports/mode03_neg_branch_ctgrid_corrected_validation_roadmap.md"
        ),
        (
            f"{prefix} build-research-demonstrator-package --release-manifest {inputs['release']} "
            f"--audit-yaml {inputs['audit']} --status-manifest {inputs['status']} "
            f"--validation-roadmap {inputs['roadmap']} --validation-intake {inputs['validation_intake']} "
            f"--vessel-harmonization {inputs['vessel_harmonization']}"
        ),
        "```",
        "",
        "Large NIfTI volumes remain referenced by the release manifest and are not duplicated in this demonstrator package.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_limitations(path: Path) -> None:
    lines = [
        "# Research Demonstrator Limitations",
        "",
        "- This package supports engineering/research communication only; it is not a clinical device, treatment-planning result, or regulatory submission.",
        "- Vessel networks are corrected branch-labelled/template-derived volumes on a CT grid, not yet a fully patient-specific CTA/CTV deformable registration.",
        "- Blood-flow outputs use a graph-coupled 1D/RCR surrogate; they are not calibrated 3D CFD, fluid-structure interaction, or measured physiology.",
        "- RT outputs are synthetic engineering dose states and gamma comparisons, not TPS-commissioned or Monte Carlo clinical dose calculations.",
        "- Current validation evidence is partial: the harmonized MedSeg vessel case has complete vessel labels, but still lacks paired primary CT and organ/material segmentation.",
        "- Anatomical equivalence to a human subject remains unproven until organ volumes, body-shape metrics, vessel topology, landmarks, and vessel-organ relationships are validated across real paired cases.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _metric_lookup(metrics: tuple[DemonstratorMetric, ...], category: str, metric: str) -> str:
    for item in metrics:
        if item.category == category and item.metric == metric:
            return f"{item.value} {item.unit}".strip()
    return "not available"


def _write_manuscript_outline(path: Path, result: ResearchDemonstratorResult) -> None:
    lines = [
        "# Manuscript Outline: Digital Torso Phantom With Branch-Labelled Vascular Flow and RT-Flow QA",
        "",
        "## Working Title",
        "",
        "A reproducible digital torso phantom demonstrator integrating CT-derived material labels, branch-labelled vascular domains, pulsatile flow, and radiotherapy-flow quality assurance",
        "",
        "## Abstract Draft",
        "",
        (
            "We present a research-engineering digital phantom workflow that combines a CT/material torso model, "
            "branch-labelled arterial and venous vascular domains, graph-coupled pulsatile flow, 4D flow visualization, "
            "and radiotherapy-style spatial dose QA. The current corrected branch release contains "
            f"{_metric_lookup(result.metrics, 'vascular', 'arterial voxels')} arterial voxels, "
            f"{_metric_lookup(result.metrics, 'vascular', 'venous voxels')} venous voxels, and "
            f"{_metric_lookup(result.metrics, 'vascular', 'vessel wall voxels')} vessel-wall voxels. "
            f"The pulsatile flow model uses {_metric_lookup(result.metrics, 'flow', 'graph nodes')} and "
            f"{_metric_lookup(result.metrics, 'flow', 'graph edges')}, with a maximum mass-balance residual of "
            f"{_metric_lookup(result.metrics, 'flow', 'mass balance residual')}. "
            f"Radiotherapy-flow QA achieved a gamma minimum pass rate of {_metric_lookup(result.metrics, 'radiotherapy', 'gamma min pass rate')}. "
            "A release-readiness audit found no failed checks and separates research readiness from clinical validation blockers."
        ),
        "",
        "## Methods",
        "",
        "1. CT/material phantom construction: material-label anatomy, HU/density/relative-electron-density overlays, and RT region definitions were generated on a common CT grid.",
        "2. Vascular-domain construction: branch-labelled arterial and venous labels were harmonized, cleaned against anatomy, and converted into lumen/wall/flow-domain volumes.",
        "3. Flow model: inlet/outlet boundary conditions were snapped to branch-specific labels and solved using a graph-coupled pulsatile 1D/RCR surrogate.",
        "4. 4D visualization: temporal flow states were rendered as velocity-colored vascular volumes for qualitative review.",
        "5. RT-flow QA: a synthetic RT planning bundle, spatial vessel-target coupling, flow-modulated dose states, and gamma QA were generated.",
        "6. Release audit: artifacts, checksums, missing files, QA metrics, clinical blockers, and validation-roadmap requirements were scored.",
        "",
        "## Key Results",
        "",
    ]
    for metric in result.metrics:
        if metric.category in {"readiness", "vascular", "flow", "radiotherapy", "validation"}:
            lines.append(f"- {metric.category}/{metric.metric}: {metric.value} {metric.unit}. {metric.interpretation}")
    lines.extend(
        [
            "",
            "## Suggested Figure Set",
            "",
        ]
    )
    for figure in result.figures:
        lines.append(f"- {figure.figure_id}: {figure.title}. {figure.caption}")
    lines.extend(
        [
            "",
            "## Claims Supported Now",
            "",
            "- A reproducible engineering workflow exists for a digital torso phantom with CT/material labels, branch-labelled vessels, pulsatile flow, 4D visualization, and RT-flow QA.",
            "- The corrected branch release is internally complete for research demonstration, with no failed release-readiness checks.",
            "- The workflow has explicit validation gates and a staged partial branch-rich vessel case for future paired CT/organ validation.",
            "",
            "## Claims Not Yet Supported",
            "",
            "- Fully anatomically equivalent human subject phantom.",
            "- Patient-specific clinical digital twin.",
            "- Clinically commissioned dose-calculation workflow.",
            "- Physiologically validated 3D CFD/FSI blood-flow model.",
            "- Regulatory or treatment-planning use.",
            "",
            "## Immediate Paper/Poster Message",
            "",
            "The publishable contribution is the integrated, reproducible research-demonstrator pipeline and audit framework, not a clinical equivalence claim.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_manifest(path: Path, result: ResearchDemonstratorResult, inputs: dict[str, Path]) -> None:
    payload = {
        "package_id": result.package_id,
        "case_id": result.case_id,
        "release_id": result.release_id,
        "package_type": "publishable_research_demonstrator_package",
        "readiness_status": result.readiness_status,
        "inputs": {key: str(value) for key, value in inputs.items()},
        "summary": result.summary,
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "report": result.report_path,
            "manuscript_outline": result.manuscript_outline_path,
            "figure_atlas_png": result.figure_atlas_path,
            "figure_index_csv": result.figure_index_csv_path,
            "metrics_csv": result.metrics_csv_path,
            "command_log": result.command_log_path,
            "limitations": result.limitations_path,
        },
        "figures": [
            {
                "figure_id": figure.figure_id,
                "title": figure.title,
                "caption": figure.caption,
                "source_path": figure.source_path,
                "packaged_path": figure.packaged_path,
                "status": figure.status,
            }
            for figure in result.figures
        ],
        "metrics": [
            {
                "category": metric.category,
                "metric": metric.metric,
                "value": metric.value,
                "unit": metric.unit,
                "interpretation": metric.interpretation,
                "source": metric.source,
            }
            for metric in result.metrics
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: ResearchDemonstratorResult) -> str:
    image_rel = os.path.relpath(result.figure_atlas_path, start=Path(result.report_path).parent)
    lines = [
        "# Publishable Research Demonstrator Package",
        "",
        f"Package ID: `{result.package_id}`",
        f"Case ID: `{result.case_id}`",
        f"Release ID: `{result.release_id}`",
        f"Readiness status: `{result.readiness_status}`",
        "",
        f"![Research demonstrator atlas]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Research readiness score: {result.summary.get('research_score_percent', 0.0):.1f}%",
        f"- Overall readiness score: {result.summary.get('overall_score_percent', 0.0):.1f}%",
        f"- Audit pass/review/fail: {result.summary.get('audit_pass_count', 0)} / {result.summary.get('audit_review_count', 0)} / {result.summary.get('audit_fail_count', 0)}",
        f"- Figures present: {result.summary.get('present_figure_count', 0)} / {result.summary.get('figure_count', 0)}",
        f"- Metrics table rows: {result.summary.get('metric_count', 0)}",
        f"- Clinical blockers remaining: {result.summary.get('clinical_blocker_count', 0)}",
        "",
        "## Outputs",
        "",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Manuscript outline: `{result.manuscript_outline_path}`",
        f"- Figure atlas: `{result.figure_atlas_path}`",
        f"- Figure index: `{result.figure_index_csv_path}`",
        f"- Metrics CSV: `{result.metrics_csv_path}`",
        f"- Command log: `{result.command_log_path}`",
        f"- Limitations: `{result.limitations_path}`",
        "",
        "## Figure Set",
        "",
    ]
    for figure in result.figures:
        lines.append(f"- `{figure.figure_id}` `{figure.status}`: {figure.title}")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def build_research_demonstrator_package(
    *,
    release_manifest_path: str | Path = "outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/mode03_neg_branch_ctgrid_corrected_flow_rc1_release_manifest_v001.yaml",
    audit_yaml_path: str | Path = "outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/readiness_audit/mode03_neg_branch_ctgrid_corrected_flow_rc1_readiness_audit_audit_v001.yaml",
    status_manifest_path: str | Path = "outputs/reports/corrected_branch_status/mode03_neg_branch_ctgrid_corrected_flow_corrected_branch_status_manifest_v001.yaml",
    validation_roadmap_path: str | Path = "outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/validation_roadmap/mode03_neg_branch_ctgrid_corrected_flow_rc1_validation_roadmap_roadmap_v001.yaml",
    validation_intake_path: str | Path = "outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/validation_intake_medseg_harmonized_partial_p1/medseg_abdominal_vasculature_case001_harmonized_partial_p1_intake_manifest_v001.yaml",
    vessel_harmonization_path: str | Path = "outputs/digital/vessel_label_harmonization/medseg_abdominal_vasculature_case001_partial_p1/medseg_abdominal_vasculature_case001_partial_p1_vessel_label_harmonization_manifest_v001.yaml",
    output_dir: str | Path = "outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/research_demonstrator",
    package_id: str | None = None,
    report_path: str | Path | None = "outputs/reports/mode03_neg_branch_ctgrid_corrected_research_demonstrator.md",
) -> ResearchDemonstratorResult:
    release_path = Path(release_manifest_path)
    audit_path = Path(audit_yaml_path)
    status_path = Path(status_manifest_path)
    roadmap_path = Path(validation_roadmap_path)
    intake_path = Path(validation_intake_path)
    harmonization_path = Path(vessel_harmonization_path)
    for required in (release_path, audit_path, status_path, roadmap_path, intake_path, harmonization_path):
        if not required.exists():
            raise FileNotFoundError(f"Required demonstrator input does not exist: {required}")

    release = _load_yaml(release_path)
    audit = _load_yaml(audit_path)
    status_manifest = _load_yaml(status_path)
    roadmap = _load_yaml(roadmap_path)
    validation_intake = _load_yaml(intake_path)
    vessel_harmonization = _load_yaml(harmonization_path)

    case_id = str(release.get("case_id", "unknown_case"))
    release_id = str(release.get("release_id", "unknown_release"))
    package = package_id or f"{case_id}_research_demonstrator_v1"
    output = Path(output_dir)
    figures_dir = output / "figures"
    manifest = output / f"{package}_manifest_v001.yaml"
    figure_atlas = output / f"{package}_figure_atlas_v001.png"
    figure_index = output / f"{package}_figure_index_v001.csv"
    metrics_csv = output / f"{package}_metrics_v001.csv"
    manuscript_outline = output / f"{package}_manuscript_outline_v001.md"
    command_log = output / f"{package}_reproducibility_commands_v001.md"
    limitations = output / f"{package}_limitations_v001.md"
    report = Path(report_path) if report_path is not None else output / f"{package}_report_v001.md"

    figure_sources = _figure_sources(
        status_manifest=status_manifest,
        status_manifest_path=status_path,
        audit=audit,
        audit_path=audit_path,
        roadmap=roadmap,
        roadmap_path=roadmap_path,
        validation_intake=validation_intake,
        validation_intake_path=intake_path,
        vessel_harmonization=vessel_harmonization,
        vessel_harmonization_path=harmonization_path,
    )
    figures = _stage_figures(figure_sources, figures_dir)
    metrics = _collect_metrics(
        release_manifest=release,
        release_manifest_path=release_path,
        audit=audit,
        audit_path=audit_path,
        roadmap=roadmap,
        roadmap_path=roadmap_path,
        validation_intake=validation_intake,
        validation_intake_path=intake_path,
        vessel_harmonization=vessel_harmonization,
        vessel_harmonization_path=harmonization_path,
    )
    checks = [check for check in _as_list(audit.get("checks")) if isinstance(check, dict)]
    pass_count = sum(check.get("status") == "pass" for check in checks)
    review_count = sum(check.get("status") == "review" for check in checks)
    fail_count = sum(check.get("status") == "fail" for check in checks)
    scores = _as_mapping(audit.get("scores"))
    present_figures = sum(figure.status == "present" for figure in figures)
    research_score = _safe_float(scores.get("research_score_percent"))
    readiness_status = (
        "publishable_research_demonstrator_ready"
        if fail_count == 0 and present_figures >= 10 and research_score >= 95.0
        else "publishable_research_demonstrator_review_required"
    )
    summary = {
        "research_score_percent": research_score,
        "overall_score_percent": _safe_float(scores.get("overall_score_percent")),
        "clinical_blocker_count": _safe_int(scores.get("clinical_blocker_count")),
        "audit_pass_count": pass_count,
        "audit_review_count": review_count,
        "audit_fail_count": fail_count,
        "figure_count": len(figures),
        "present_figure_count": present_figures,
        "metric_count": len(metrics),
        "validation_case_ready_count": _safe_int(_as_mapping(validation_intake.get("summary")).get("ready_case_count")),
        "validation_case_missing_count": _safe_int(_as_mapping(validation_intake.get("summary")).get("missing_case_count")),
    }
    notes = (
        "package_is_publishable_engineering_research_demonstrator_not_clinical_claim",
        "figure_atlas_and_metrics_are_generated_from_current_corrected_branch_release_outputs",
        "complete_paired_ct_cta_ctv_validation_case_remains_the_next_major_milestone",
    )
    result = ResearchDemonstratorResult(
        package_id=package,
        case_id=case_id,
        release_id=release_id,
        readiness_status=readiness_status,
        output_dir=str(output),
        manifest_yaml_path=str(manifest),
        report_path=str(report),
        manuscript_outline_path=str(manuscript_outline),
        figure_atlas_path=str(figure_atlas),
        figure_index_csv_path=str(figure_index),
        metrics_csv_path=str(metrics_csv),
        command_log_path=str(command_log),
        limitations_path=str(limitations),
        summary=summary,
        figures=figures,
        metrics=metrics,
        notes=notes,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_figure_index(figure_index, figures)
    _write_metrics_csv(metrics_csv, metrics)
    _write_figure_atlas(figure_atlas, result)
    _write_manuscript_outline(manuscript_outline, result)
    inputs = {
        "release": release_path,
        "audit": audit_path,
        "status": status_path,
        "roadmap": roadmap_path,
        "validation_intake": intake_path,
        "vessel_harmonization": harmonization_path,
    }
    _write_command_log(command_log, result, inputs)
    _write_limitations(limitations)
    _write_manifest(manifest, result, inputs)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_research_demonstrator_result(result: ResearchDemonstratorResult) -> str:
    return _format_report(result)
