from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import yaml


@dataclass(frozen=True)
class ReleaseReadinessCheck:
    domain: str
    check_id: str
    status: str
    score: float
    max_score: float
    evidence_path: str
    finding: str
    recommended_action: str
    clinical_blocker: bool


@dataclass(frozen=True)
class ReleaseReadinessAuditResult:
    release_id: str
    case_id: str
    audit_id: str
    output_dir: str
    readiness_tier: str
    overall_score_percent: float
    research_score_percent: float
    clinical_blocker_count: int
    checks_csv_path: str
    roadmap_csv_path: str
    audit_yaml_path: str
    scorecard_png_path: str
    report_path: str
    checks: tuple[ReleaseReadinessCheck, ...]
    domain_scores: dict[str, dict[str, float]]
    notes: tuple[str, ...]


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Release readiness audit plotting requires matplotlib.") from exc
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


def _resolve_path(raw_path: Any, manifest_path: Path) -> Path:
    if raw_path is None or str(raw_path) == "":
        return Path("__missing_release_output__")
    path = Path(str(raw_path))
    if str(path) == "":
        return path
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _score_for(status: str, max_score: float) -> float:
    if status == "pass":
        return max_score
    if status == "review":
        return max_score * 0.5
    return 0.0


def _add_check(
    checks: list[ReleaseReadinessCheck],
    *,
    domain: str,
    check_id: str,
    status: str,
    max_score: float = 1.0,
    evidence_path: str | Path | None = None,
    finding: str,
    recommended_action: str,
    clinical_blocker: bool = False,
) -> None:
    checks.append(
        ReleaseReadinessCheck(
            domain=domain,
            check_id=check_id,
            status=status,
            score=_score_for(status, max_score),
            max_score=max_score,
            evidence_path="" if evidence_path is None else str(evidence_path),
            finding=finding,
            recommended_action=recommended_action,
            clinical_blocker=clinical_blocker,
        )
    )


def _qa_status_count(rows: list[dict[str, str]], status: str) -> int:
    return sum(row.get("status") == status for row in rows)


def _metric_row(rows: list[dict[str, str]], metric: str, category: str | None = None) -> dict[str, str]:
    for row in rows:
        if row.get("metric") != metric:
            continue
        if category is not None and row.get("category") != category:
            continue
        return row
    return {}


def _artifact_group_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        group = row.get("group", "unknown")
        counts[group] = counts.get(group, 0) + 1
    return counts


def _all_artifacts_exist(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        if str(row.get("exists", "")).lower() != "true":
            return False
        source_path = row.get("source_path", "")
        if source_path and not Path(source_path).exists():
            return False
    return bool(rows)


def _copied_checksums_complete(rows: list[dict[str, str]]) -> bool:
    copied = [row for row in rows if row.get("copy_policy") == "copied"]
    if not copied:
        return False
    return all(bool(row.get("sha256")) for row in copied)


def _domain_scores(checks: tuple[ReleaseReadinessCheck, ...]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for check in checks:
        domain = scores.setdefault(check.domain, {"score": 0.0, "max_score": 0.0, "check_count": 0.0})
        domain["score"] += check.score
        domain["max_score"] += check.max_score
        domain["check_count"] += 1.0
    for domain in scores.values():
        domain["percent"] = 100.0 * domain["score"] / max(domain["max_score"], 1e-9)
    return scores


def _readiness_tier(checks: tuple[ReleaseReadinessCheck, ...], clinical_blocker_count: int) -> str:
    if any(check.status == "fail" for check in checks):
        return "blocked_by_readiness_failure"
    if clinical_blocker_count:
        return "research_ready_clinical_validation_required"
    if any(check.status == "review" for check in checks):
        return "research_ready_with_review_items"
    return "research_ready"


def _write_checks_csv(path: Path, checks: tuple[ReleaseReadinessCheck, ...]) -> None:
    fields = [
        "domain",
        "check_id",
        "status",
        "score",
        "max_score",
        "evidence_path",
        "finding",
        "recommended_action",
        "clinical_blocker",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for check in checks:
            writer.writerow(
                {
                    "domain": check.domain,
                    "check_id": check.check_id,
                    "status": check.status,
                    "score": f"{check.score:.3f}",
                    "max_score": f"{check.max_score:.3f}",
                    "evidence_path": check.evidence_path,
                    "finding": check.finding,
                    "recommended_action": check.recommended_action,
                    "clinical_blocker": check.clinical_blocker,
                }
            )


def _write_roadmap_csv(path: Path, checks: tuple[ReleaseReadinessCheck, ...]) -> None:
    fields = ["priority", "domain", "check_id", "status", "clinical_blocker", "recommended_action", "evidence_path"]
    blockers = [check for check in checks if check.clinical_blocker]
    failures = [check for check in checks if check.status == "fail" and not check.clinical_blocker]
    reviews = [check for check in checks if check.status == "review" and not check.clinical_blocker]
    rows: list[tuple[int, ReleaseReadinessCheck]] = []
    rows.extend((1, check) for check in blockers)
    rows.extend((2, check) for check in failures)
    rows.extend((3, check) for check in reviews)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for priority, check in rows:
            writer.writerow(
                {
                    "priority": priority,
                    "domain": check.domain,
                    "check_id": check.check_id,
                    "status": check.status,
                    "clinical_blocker": check.clinical_blocker,
                    "recommended_action": check.recommended_action,
                    "evidence_path": check.evidence_path,
                }
            )


def _write_yaml(path: Path, result: ReleaseReadinessAuditResult, release_manifest_path: Path) -> None:
    payload = {
        "audit_id": result.audit_id,
        "release_id": result.release_id,
        "case_id": result.case_id,
        "package_type": "digital_phantom_release_readiness_audit",
        "source_release_manifest": str(release_manifest_path),
        "readiness_tier": result.readiness_tier,
        "scores": {
            "overall_score_percent": result.overall_score_percent,
            "research_score_percent": result.research_score_percent,
            "clinical_blocker_count": result.clinical_blocker_count,
            "domain_scores": result.domain_scores,
        },
        "outputs": {
            "checks_csv": result.checks_csv_path,
            "roadmap_csv": result.roadmap_csv_path,
            "audit_yaml": result.audit_yaml_path,
            "scorecard_png": result.scorecard_png_path,
            "report": result.report_path,
        },
        "checks": [
            {
                "domain": check.domain,
                "check_id": check.check_id,
                "status": check.status,
                "score": check.score,
                "max_score": check.max_score,
                "evidence_path": check.evidence_path,
                "finding": check.finding,
                "recommended_action": check.recommended_action,
                "clinical_blocker": check.clinical_blocker,
            }
            for check in result.checks
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_scorecard(path: Path, result: ReleaseReadinessAuditResult) -> None:
    plt = _import_plotting()
    domains = list(result.domain_scores)
    percents = [result.domain_scores[domain]["percent"] for domain in domains]
    colors = ["#2563eb" if value >= 90.0 else "#d97706" if value >= 70.0 else "#b91c1c" for value in percents]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    positions = range(len(domains))
    ax.barh(list(positions), percents, color=colors)
    ax.set_yticks(list(positions), labels=[domain.replace("_", " ").title() for domain in domains])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Domain readiness score (%)")
    ax.set_title("Release Readiness Domains")
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(percents):
        ax.text(min(value + 1.5, 98.0), index, f"{value:.1f}%", va="center", fontsize=9)

    counts = {
        "pass": sum(check.status == "pass" for check in result.checks),
        "review": sum(check.status == "review" for check in result.checks),
        "fail": sum(check.status == "fail" for check in result.checks),
        "clinical blockers": result.clinical_blocker_count,
    }
    axes[1].axis("off")
    text = "\n".join(
        [
            "Scorecard",
            "",
            f"Release: {result.release_id}",
            f"Tier: {result.readiness_tier}",
            f"Overall score: {result.overall_score_percent:.1f}%",
            f"Research score: {result.research_score_percent:.1f}%",
            "",
            f"Pass: {counts['pass']}",
            f"Review: {counts['review']}",
            f"Fail: {counts['fail']}",
            f"Clinical blockers: {counts['clinical blockers']}",
            "",
            "Interpretation",
            "Research package is usable when failures are zero.",
            "Clinical claims remain blocked until review items marked clinical blockers are resolved.",
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
    fig.suptitle("Digital Phantom Release Readiness Audit", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_report(result: ReleaseReadinessAuditResult) -> str:
    image_rel = os.path.relpath(result.scorecard_png_path, start=Path(result.report_path).parent)
    status_counts = {
        "pass": sum(check.status == "pass" for check in result.checks),
        "review": sum(check.status == "review" for check in result.checks),
        "fail": sum(check.status == "fail" for check in result.checks),
    }
    lines = [
        "# Digital Phantom Release Readiness Audit",
        "",
        f"Release ID: `{result.release_id}`",
        f"Case ID: `{result.case_id}`",
        f"Readiness tier: `{result.readiness_tier}`",
        "",
        f"![Release readiness scorecard]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Overall score: {result.overall_score_percent:.1f}%",
        f"- Research-readiness score: {result.research_score_percent:.1f}%",
        f"- Pass / review / fail: {status_counts['pass']} / {status_counts['review']} / {status_counts['fail']}",
        f"- Clinical blocker count: {result.clinical_blocker_count}",
        "",
        "## Domain Scores",
        "",
        "| domain | score | checks |",
        "| --- | ---: | ---: |",
    ]
    for domain, score in result.domain_scores.items():
        lines.append(f"| {domain.replace('_', ' ')} | {score['percent']:.1f}% | {int(score['check_count'])} |")
    lines.extend(
        [
            "",
            "## Clinical-Claim Blockers",
            "",
        ]
    )
    blockers = [check for check in result.checks if check.clinical_blocker]
    if blockers:
        for check in blockers:
            lines.append(f"- `{check.domain}/{check.check_id}`: {check.finding} Next: {check.recommended_action}")
    else:
        lines.append("- None flagged by this audit.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Checks CSV: `{result.checks_csv_path}`",
            f"- Roadmap CSV: `{result.roadmap_csv_path}`",
            f"- Audit YAML: `{result.audit_yaml_path}`",
            f"- Scorecard PNG: `{result.scorecard_png_path}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_release_readiness_audit_result(result: ReleaseReadinessAuditResult) -> str:
    return _format_report(result)


def audit_research_release_package(
    release_manifest_path: str | Path,
    output_dir: str | Path = "outputs/releases/mode03_neg_stage007_rc1/readiness_audit",
    audit_id: str | None = None,
    report_path: str | Path | None = "outputs/reports/mode03_neg_stage007_release_readiness_audit.md",
) -> ReleaseReadinessAuditResult:
    manifest_path = Path(release_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Release manifest does not exist: {manifest_path}")
    manifest = _load_yaml(manifest_path)
    release_id = str(manifest.get("release_id", manifest_path.stem))
    case_id = str(manifest.get("case_id", "unknown_case"))
    audit = audit_id or f"{release_id}_readiness_audit"
    output = Path(output_dir)
    checks_csv = output / f"{audit}_checks_v001.csv"
    roadmap_csv = output / f"{audit}_roadmap_v001.csv"
    audit_yaml = output / f"{audit}_audit_v001.yaml"
    scorecard_png = output / f"{audit}_scorecard_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{audit}_report_v001.md"

    outputs = _as_mapping(manifest.get("outputs"))
    artifact_index_path = _resolve_path(outputs.get("artifact_index_csv"), manifest_path)
    qa_summary_path = _resolve_path(outputs.get("qa_summary_csv"), manifest_path)
    command_log_path = _resolve_path(outputs.get("command_log"), manifest_path)
    limitations_path = _resolve_path(outputs.get("limitations_markdown"), manifest_path)
    release_report_path = _resolve_path(outputs.get("report"), manifest_path)
    atlas_path = _resolve_path(outputs.get("atlas_png"), manifest_path)
    artifact_rows = _read_csv_rows(artifact_index_path)
    qa_rows = _read_csv_rows(qa_summary_path)
    limitations_text = limitations_path.read_text().lower() if limitations_path.exists() else ""
    command_text = command_log_path.read_text().lower() if command_log_path.exists() else ""
    group_counts = _artifact_group_counts(artifact_rows)
    checks: list[ReleaseReadinessCheck] = []

    required_outputs = {
        "release_manifest": manifest_path,
        "artifact_index": artifact_index_path,
        "qa_summary": qa_summary_path,
        "command_log": command_log_path,
        "limitations": limitations_path,
        "release_report": release_report_path,
        "atlas_png": atlas_path,
    }
    for check_id, path in required_outputs.items():
        exists = bool(str(path)) and path.exists()
        _add_check(
            checks,
            domain="release_integrity",
            check_id=f"{check_id}_exists",
            status="pass" if exists else "fail",
            evidence_path=path,
            finding=f"{check_id} {'is present' if exists else 'is missing'}",
            recommended_action="Regenerate the release package output." if not exists else "No action needed.",
        )

    _add_check(
        checks,
        domain="release_integrity",
        check_id="all_indexed_artifacts_exist",
        status="pass" if _all_artifacts_exist(artifact_rows) else "fail",
        evidence_path=artifact_index_path,
        finding=f"{len(artifact_rows)} indexed artifacts checked for local existence.",
        recommended_action="Restage missing artifacts or rebuild the release package." if not _all_artifacts_exist(artifact_rows) else "No action needed.",
    )
    _add_check(
        checks,
        domain="release_integrity",
        check_id="copied_artifacts_have_checksums",
        status="pass" if _copied_checksums_complete(artifact_rows) else "review",
        evidence_path=artifact_index_path,
        finding="Copied small artifacts include SHA-256 checksums." if _copied_checksums_complete(artifact_rows) else "One or more copied artifacts lacks a checksum.",
        recommended_action="Regenerate artifact index checksums for every copied artifact.",
    )

    expected_groups = {
        "vascular_voxelization",
        "vessel_organ_validation",
        "vessel_radius_validation",
        "coupled_pulsatile_flow",
        "radiotherapy_qa",
        "rt_planning",
        "dose_gamma_qa",
        "reports",
    }
    missing_groups = sorted(group for group in expected_groups if group_counts.get(group, 0) == 0)
    _add_check(
        checks,
        domain="release_integrity",
        check_id="required_evidence_groups_present",
        status="pass" if not missing_groups else "fail",
        evidence_path=artifact_index_path,
        finding="All required evidence groups are present." if not missing_groups else f"Missing evidence groups: {', '.join(missing_groups)}",
        recommended_action="Run the missing build/QA stages and rebuild the release package." if missing_groups else "No action needed.",
    )
    indexed_volumes = sum(row.get("file_type") == "nifti_volume" and row.get("copy_policy") == "indexed_only_large_or_volume" for row in artifact_rows)
    _add_check(
        checks,
        domain="release_integrity",
        check_id="large_volumes_are_disk_light_indexed",
        status="pass" if indexed_volumes > 0 else "review",
        evidence_path=artifact_index_path,
        finding=f"{indexed_volumes} NIfTI/volume artifacts are indexed instead of duplicated.",
        recommended_action="Keep heavyweight volumes indexed unless preparing a dedicated archive.",
    )

    qa_fails = _qa_status_count(qa_rows, "fail")
    qa_reviews = _qa_status_count(qa_rows, "review")
    _add_check(
        checks,
        domain="release_qa",
        check_id="no_release_qa_failures",
        status="pass" if qa_fails == 0 else "fail",
        evidence_path=qa_summary_path,
        finding=f"Release QA fail count = {qa_fails}.",
        recommended_action="Resolve failing QA metrics before using this release candidate." if qa_fails else "No action needed.",
    )
    _add_check(
        checks,
        domain="release_qa",
        check_id="no_release_qa_reviews",
        status="pass" if qa_reviews == 0 else "review",
        evidence_path=qa_summary_path,
        finding=f"Release QA review count = {qa_reviews}.",
        recommended_action="Review non-blocking QA items before stronger claims." if qa_reviews else "No action needed.",
    )

    vascular_metrics = [
        "connected_lumen_components",
        "arterial_components",
        "venous_components",
        "arterial_venous_overlap_after_cleanup_voxels",
        "outside_body_fraction_before_clip",
    ]
    vascular_ok = all(_metric_row(qa_rows, metric, "vascular_domain").get("status") == "pass" for metric in vascular_metrics)
    _add_check(
        checks,
        domain="vascular_anatomy",
        check_id="vascular_domain_topology_passes",
        status="pass" if vascular_ok else "fail",
        evidence_path=qa_summary_path,
        finding="Arterial/venous domains are connected and separated by current QA gates." if vascular_ok else "One or more vascular-domain topology checks failed.",
        recommended_action="Rerun domain repair, voxelization, and vascular QA." if not vascular_ok else "No action needed.",
    )
    organ_ok = (
        _metric_row(qa_rows, "fail_count", "organ_aware_vascular_anatomy").get("status") == "pass"
        and _metric_row(qa_rows, "review_count", "organ_aware_vascular_anatomy").get("status") == "pass"
    )
    radius_ok = (
        _metric_row(qa_rows, "fail_count", "radius_aware_vascular_anatomy").get("status") == "pass"
        and _metric_row(qa_rows, "review_count", "radius_aware_vascular_anatomy").get("status") == "pass"
    )
    _add_check(
        checks,
        domain="vascular_anatomy",
        check_id="organ_and_radius_aware_vascular_qa_passes",
        status="pass" if organ_ok and radius_ok else "fail",
        evidence_path=qa_summary_path,
        finding="Organ-aware and radius-aware vascular checks report no review/fail edges." if organ_ok and radius_ok else "Vessel-organ or radius-aware QA still has non-pass edges.",
        recommended_action="Correct/reroute/tune flagged vascular edges, then rerun QA." if not (organ_ok and radius_ok) else "No action needed.",
    )
    real_vessel_labels = any(
        "registered_labeled" in row.get("source_path", "").lower()
        or "medseg" in row.get("source_path", "").lower()
        or "cta" in row.get("source_path", "").lower()
        for row in artifact_rows
    )
    _add_check(
        checks,
        domain="vascular_anatomy",
        check_id="patient_specific_vascular_template_limit",
        status="review",
        evidence_path=artifact_index_path,
        finding="The release contains a staged vascular graph and real-labelled/template-derived branches, but not a fully patient-specific CTA/CTV-registered network.",
        recommended_action="Stage a patient-specific CTA/CTV vessel segmentation and rerun labelled centerline replacement plus deformable vessel-organ registration.",
        clinical_blocker=True,
    )

    flow_residual = _metric_row(qa_rows, "max_abs_mass_balance_residual_ml_s", "coupled_pulsatile_flow")
    _add_check(
        checks,
        domain="flow_model",
        check_id="coupled_flow_mass_balance_passes",
        status="pass" if flow_residual.get("status") == "pass" else "fail",
        evidence_path=flow_residual.get("source_path", qa_summary_path),
        finding=f"Coupled pulsatile mass-balance status = {flow_residual.get('status', 'missing')}.",
        recommended_action="Debug boundary conditions and graph connectivity if mass balance fails." if flow_residual.get("status") != "pass" else "No action needed.",
    )
    limited_flow_physics = "no_wave_speed" in command_text or "rigid" in limitations_text or "3d cfd" in limitations_text
    _add_check(
        checks,
        domain="flow_model",
        check_id="flow_physics_validation_gap",
        status="review",
        evidence_path=limitations_path,
        finding="Flow is graph-coupled 1D and not yet 3D CFD, FSI, wall-compliant, or measurement-calibrated.",
        recommended_action="Add 3D CFD/0D-1D calibration cases, pressure/flow validation, wall compliance, and uncertainty bounds.",
        clinical_blocker=True,
    )

    gamma_pass = _metric_row(qa_rows, "minimum_gamma_pass_rate_percent", "dose_gamma_qa")
    ptv_d95 = _metric_row(qa_rows, "ptv_static_d95_gy", "radiotherapy_planning")
    _add_check(
        checks,
        domain="radiotherapy",
        check_id="rt_gamma_and_ptv_metrics_pass",
        status="pass" if gamma_pass.get("status") == "pass" and ptv_d95.get("status") == "pass" else "fail",
        evidence_path=qa_summary_path,
        finding="PyMedPhys gamma and placeholder PTV D95 checks pass." if gamma_pass.get("status") == "pass" and ptv_d95.get("status") == "pass" else "RT/gamma release checks did not fully pass.",
        recommended_action="Rerun RT planning and dose gamma QA." if gamma_pass.get("status") != "pass" or ptv_d95.get("status") != "pass" else "No action needed.",
    )
    synthetic_dose_gap = "synthetic dose" in limitations_text or "not tps" in limitations_text or "monte carlo" in limitations_text
    _add_check(
        checks,
        domain="radiotherapy",
        check_id="clinical_dose_engine_gap",
        status="review",
        evidence_path=limitations_path,
        finding="RT outputs are synthetic engineering dose patterns, not TPS or Monte Carlo dose calculations.",
        recommended_action="Connect the bundle to a TPS or Monte Carlo dose engine and compare DVH/gamma across static and pulsatile states.",
        clinical_blocker=True,
    )

    limitation_has_not_clinical = "not clinical" in limitations_text or "not a clinical" in limitations_text
    _add_check(
        checks,
        domain="documentation",
        check_id="limitations_explicitly_block_clinical_use",
        status="pass" if limitation_has_not_clinical else "fail",
        evidence_path=limitations_path,
        finding="Limitations explicitly state that the release is not for clinical use." if limitation_has_not_clinical else "Clinical-use limitation language is missing.",
        recommended_action="Keep explicit non-clinical-use language in every release package." if limitation_has_not_clinical else "Add explicit non-clinical-use limitations.",
    )
    command_complete = all(token in command_text for token in ("validate-vessel", "build-flow", "build-rt", "build-dose-gamma"))
    _add_check(
        checks,
        domain="documentation",
        check_id="reproducibility_commands_cover_major_stages",
        status="pass" if command_complete else "review",
        evidence_path=command_log_path,
        finding="Command log covers vascular QA, flow, RT planning, dose gamma, and release packaging." if command_complete else "Command log does not clearly cover every major stage.",
        recommended_action="Add missing command-line reproduction steps.",
    )

    _add_check(
        checks,
        domain="population_validation",
        check_id="single_release_case_validation_gap",
        status="review",
        evidence_path=manifest_path,
        finding="This audit covers one released case; cohort-level anatomical validity and robustness still need external validation.",
        recommended_action="Run the same release audit across multiple segmented CT/CTA/CTV cases and summarize organ volumes, centroids, vessel-organ distances, flow metrics, and RT sensitivity.",
        clinical_blocker=True,
    )
    _add_check(
        checks,
        domain="population_validation",
        check_id="anatomical_equivalence_gap",
        status="review",
        evidence_path=limitations_path,
        finding="The phantom is anatomically useful as a research surrogate, but not yet proven fully anatomically equivalent to a human subject.",
        recommended_action="Validate organ volumes, body-shape metrics, vessel branch topology, and landmark distances against real population CT/CTA segmentations.",
        clinical_blocker=True,
    )

    checks_tuple = tuple(checks)
    domain_scores = _domain_scores(checks_tuple)
    overall_score = 100.0 * sum(check.score for check in checks_tuple) / max(sum(check.max_score for check in checks_tuple), 1e-9)
    research_checks = tuple(check for check in checks_tuple if not check.clinical_blocker)
    research_score = 100.0 * sum(check.score for check in research_checks) / max(
        sum(check.max_score for check in research_checks), 1e-9
    )
    clinical_blockers = sum(check.clinical_blocker for check in checks_tuple)
    tier = _readiness_tier(checks_tuple, clinical_blockers)
    notes = (
        "audit_distinguishes_research_readiness_from_clinical_equivalence",
        "review_items_marked_clinical_blocker_must_be_resolved_before_clinical_claims",
        "scores_are_project_engineering_triage_not_regulatory_acceptance",
    )
    result = ReleaseReadinessAuditResult(
        release_id=release_id,
        case_id=case_id,
        audit_id=audit,
        output_dir=str(output),
        readiness_tier=tier,
        overall_score_percent=overall_score,
        research_score_percent=research_score,
        clinical_blocker_count=clinical_blockers,
        checks_csv_path=str(checks_csv),
        roadmap_csv_path=str(roadmap_csv),
        audit_yaml_path=str(audit_yaml),
        scorecard_png_path=str(scorecard_png),
        report_path=str(report),
        checks=checks_tuple,
        domain_scores=domain_scores,
        notes=notes,
    )

    _write_checks_csv(checks_csv, checks_tuple)
    _write_roadmap_csv(roadmap_csv, checks_tuple)
    _write_yaml(audit_yaml, result, manifest_path)
    _write_scorecard(scorecard_png, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
