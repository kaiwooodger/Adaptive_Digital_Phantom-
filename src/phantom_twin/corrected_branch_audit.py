from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from .release_readiness import (
    ReleaseReadinessAuditResult,
    ReleaseReadinessCheck,
    _add_check,
    _all_artifacts_exist,
    _artifact_group_counts,
    _as_mapping,
    _copied_checksums_complete,
    _domain_scores,
    _read_csv_rows,
    _readiness_tier,
    _resolve_path,
    _safe_float,
    _safe_int,
    _write_checks_csv,
    _write_roadmap_csv,
    _write_scorecard,
    _write_yaml,
)

import yaml


EXPECTED_EVIDENCE_GROUPS = (
    "corrected_vascular_domain",
    "flow_boundary_conditions",
    "steady_1d_flow",
    "coupled_pulsatile_flow",
    "flow_4d_visualization",
    "radiotherapy_qa",
    "rt_planning",
    "spatial_rt_flow_coupling",
    "spatial_rt_flow_dose",
    "dose_gamma_qa",
    "status_package",
    "reports",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _status_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    summary = _as_mapping(manifest.get("summary"))
    return _as_mapping(summary.get("status_summary"))


def _text_or_empty(path: Path) -> str:
    return path.read_text().lower() if path.exists() else ""


def _missing_artifact_count(rows: list[dict[str, str]]) -> int:
    return sum(row.get("copy_policy") == "missing" or row.get("exists", "").lower() != "true" for row in rows)


def _indexed_volume_count(rows: list[dict[str, str]]) -> int:
    return sum(
        row.get("file_type") == "nifti_volume" and row.get("copy_policy") == "indexed_only_large_or_volume"
        for row in rows
    )


def _copied_total_size(rows: list[dict[str, str]]) -> int:
    return sum(_safe_int(row.get("size_bytes")) for row in rows if row.get("copy_policy") == "copied")


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GiB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    return f"{size_bytes} B"


def _format_corrected_report(result: ReleaseReadinessAuditResult) -> str:
    image_rel = os.path.relpath(result.scorecard_png_path, start=Path(result.report_path).parent)
    status_counts = {
        "pass": sum(check.status == "pass" for check in result.checks),
        "review": sum(check.status == "review" for check in result.checks),
        "fail": sum(check.status == "fail" for check in result.checks),
    }
    lines = [
        "# Corrected Branch-Labelled Release Readiness Audit",
        "",
        f"Release ID: `{result.release_id}`",
        f"Case ID: `{result.case_id}`",
        f"Readiness tier: `{result.readiness_tier}`",
        "",
        f"![Corrected release readiness scorecard]({image_rel})",
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
    lines.extend(["", "## Clinical-Claim Blockers", ""])
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


def audit_corrected_branch_release_package(
    release_manifest_path: str | Path = "outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/mode03_neg_branch_ctgrid_corrected_flow_rc1_release_manifest_v001.yaml",
    output_dir: str | Path = "outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/readiness_audit",
    audit_id: str | None = None,
    report_path: str | Path | None = "outputs/reports/mode03_neg_branch_ctgrid_corrected_release_readiness_audit.md",
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
    readme_path = _resolve_path(outputs.get("readme_markdown"), manifest_path)
    command_log_path = _resolve_path(outputs.get("command_log"), manifest_path)
    limitations_path = _resolve_path(outputs.get("limitations_markdown"), manifest_path)
    release_report_path = _resolve_path(outputs.get("report"), manifest_path)
    artifact_rows = _read_csv_rows(artifact_index_path)
    group_counts = _artifact_group_counts(artifact_rows)
    status = _status_summary(manifest)
    vascular = _as_mapping(status.get("vascular_domain"))
    rt_material = _as_mapping(status.get("rt_material_package"))
    flow = _as_mapping(status.get("flow"))
    flow4d = _as_mapping(status.get("flow4d"))
    rt_flow = _as_mapping(status.get("rt_flow"))
    gamma = _as_mapping(status.get("gamma"))
    limitations_text = _text_or_empty(limitations_path)
    command_text = _text_or_empty(command_log_path)
    checks: list[ReleaseReadinessCheck] = []

    required_outputs = {
        "release_manifest": manifest_path,
        "artifact_index": artifact_index_path,
        "readme": readme_path,
        "command_log": command_log_path,
        "limitations": limitations_path,
        "release_report": release_report_path,
    }
    for check_id, path in required_outputs.items():
        exists = bool(str(path)) and path.exists()
        _add_check(
            checks,
            domain="release_integrity",
            check_id=f"{check_id}_exists",
            status="pass" if exists else "fail",
            evidence_path=path,
            finding=f"{check_id} {'is present' if exists else 'is missing'}.",
            recommended_action="Regenerate the corrected branch release package." if not exists else "No action needed.",
        )

    all_exist = _all_artifacts_exist(artifact_rows)
    _add_check(
        checks,
        domain="release_integrity",
        check_id="all_indexed_artifacts_exist",
        status="pass" if all_exist else "fail",
        evidence_path=artifact_index_path,
        finding=f"{len(artifact_rows)} indexed artifacts were checked for local existence.",
        recommended_action="Restage missing artifacts or rebuild the corrected branch package." if not all_exist else "No action needed.",
    )

    missing_count = _safe_int(_as_mapping(manifest.get("summary")).get("missing_artifact_count"))
    row_missing_count = _missing_artifact_count(artifact_rows)
    _add_check(
        checks,
        domain="release_integrity",
        check_id="missing_artifact_count_zero",
        status="pass" if missing_count == 0 and row_missing_count == 0 else "fail",
        evidence_path=artifact_index_path,
        finding=f"Manifest missing artifacts = {missing_count}; artifact-index missing rows = {row_missing_count}.",
        recommended_action="Restore missing files and rebuild the release manifest." if missing_count or row_missing_count else "No action needed.",
    )

    checksums_complete = _copied_checksums_complete(artifact_rows)
    _add_check(
        checks,
        domain="release_integrity",
        check_id="copied_artifacts_have_checksums",
        status="pass" if checksums_complete else "review",
        evidence_path=artifact_index_path,
        finding="Copied small artifacts include SHA-256 checksums."
        if checksums_complete
        else "One or more copied artifacts lacks a SHA-256 checksum.",
        recommended_action="Regenerate artifact checksums for all copied files." if not checksums_complete else "No action needed.",
    )

    missing_groups = sorted(group for group in EXPECTED_EVIDENCE_GROUPS if group_counts.get(group, 0) == 0)
    _add_check(
        checks,
        domain="release_integrity",
        check_id="required_corrected_evidence_groups_present",
        status="pass" if not missing_groups else "fail",
        evidence_path=artifact_index_path,
        finding="All corrected branch evidence groups are present."
        if not missing_groups
        else f"Missing evidence groups: {', '.join(missing_groups)}.",
        recommended_action="Rerun missing corrected branch stages and rebuild the release." if missing_groups else "No action needed.",
    )

    indexed_volumes = _indexed_volume_count(artifact_rows)
    copied_size = _copied_total_size(artifact_rows)
    _add_check(
        checks,
        domain="release_integrity",
        check_id="large_volumes_are_disk_light_indexed",
        status="pass" if indexed_volumes > 0 and copied_size < 250 * 1024 * 1024 else "review",
        evidence_path=artifact_index_path,
        finding=f"{indexed_volumes} NIfTI volumes are indexed; copied footprint is {_format_size(copied_size)}.",
        recommended_action="Keep large volumes indexed unless preparing a dedicated archive."
        if indexed_volumes > 0
        else "Confirm volume-copy policy before sharing this release.",
    )

    vascular_voxels_ok = (
        _safe_int(vascular.get("arterial_voxels")) > 0
        and _safe_int(vascular.get("venous_voxels")) > 0
        and _safe_int(vascular.get("vessel_wall_voxels")) > 0
    )
    _add_check(
        checks,
        domain="vascular_domain",
        check_id="arterial_venous_wall_voxels_present",
        status="pass" if vascular_voxels_ok else "fail",
        evidence_path=manifest_path,
        finding=(
            f"Arterial/venous/wall voxels = {vascular.get('arterial_voxels', 0)} / "
            f"{vascular.get('venous_voxels', 0)} / {vascular.get('vessel_wall_voxels', 0)}."
        ),
        recommended_action="Rebuild the corrected label-vessel flow domain." if not vascular_voxels_ok else "No action needed.",
    )

    unclassified = vascular.get("unclassified_labels", [])
    unclassified_count = len(unclassified) if isinstance(unclassified, list) else 1
    _add_check(
        checks,
        domain="vascular_domain",
        check_id="all_vessel_labels_classified",
        status="pass" if unclassified_count == 0 else "fail",
        evidence_path=manifest_path,
        finding=f"Unclassified vessel labels = {unclassified_count}.",
        recommended_action="Update label-to-flow-domain mappings for unclassified vessel labels."
        if unclassified_count
        else "No action needed.",
    )

    snapped_boundary_nodes = _safe_int(vascular.get("snapped_boundary_nodes"))
    boundary_count = _safe_int(flow.get("boundary_count"))
    snapped_ok = snapped_boundary_nodes > 0 and boundary_count > 0 and snapped_boundary_nodes >= boundary_count
    _add_check(
        checks,
        domain="vascular_domain",
        check_id="boundary_nodes_snapped_to_label_volume",
        status="pass" if snapped_ok else "fail",
        evidence_path=manifest_path,
        finding=f"Snapped boundary nodes = {snapped_boundary_nodes}; flow boundary count = {boundary_count}.",
        recommended_action="Rerun boundary snapping against label-specific vessel masks." if not snapped_ok else "No action needed.",
    )

    _add_check(
        checks,
        domain="vascular_domain",
        check_id="patient_specific_cta_ctv_registration_gap",
        status="review",
        evidence_path=limitations_path,
        finding=(
            "The current vessels are corrected branch-labelled/template-derived vessels on the CT grid, "
            "not a patient-specific CTA/CTV deformable registration."
        ),
        recommended_action="Stage patient-specific CTA/CTV segmentations and perform vessel-organ deformable registration.",
        clinical_blocker=True,
    )

    graph_ok = _safe_int(flow.get("node_count")) > 0 and _safe_int(flow.get("edge_count")) > 0 and boundary_count > 0
    _add_check(
        checks,
        domain="flow_model",
        check_id="flow_graph_has_nodes_edges_boundaries",
        status="pass" if graph_ok else "fail",
        evidence_path=manifest_path,
        finding=(
            f"Flow graph nodes/edges/boundaries = {flow.get('node_count', 0)} / "
            f"{flow.get('edge_count', 0)} / {flow.get('boundary_count', 0)}."
        ),
        recommended_action="Regenerate flow-domain graph and boundary package." if not graph_ok else "No action needed.",
    )

    residual = abs(_safe_float(flow.get("max_mass_balance_residual_ml_s")))
    _add_check(
        checks,
        domain="flow_model",
        check_id="coupled_flow_mass_balance_within_tolerance",
        status="pass" if residual <= 1e-4 else "fail",
        evidence_path=manifest_path,
        finding=f"Maximum mass-balance residual = {residual:.3e} mL/s.",
        recommended_action="Debug graph connectivity, outlet splits, or boundary conditions." if residual > 1e-4 else "No action needed.",
    )

    mean_flow = _safe_float(flow.get("aorta_flow_mean_ml_s"))
    min_flow = _safe_float(flow.get("aorta_flow_min_ml_s"))
    max_flow = _safe_float(flow.get("aorta_flow_max_ml_s"))
    pulsatile_ok = min_flow > 0.0 and max_flow > mean_flow > min_flow
    _add_check(
        checks,
        domain="flow_model",
        check_id="pulsatile_aorta_waveform_is_positive",
        status="pass" if pulsatile_ok else "fail",
        evidence_path=manifest_path,
        finding=f"Aorta flow mean/min/max = {mean_flow:.3f} / {min_flow:.3f} / {max_flow:.3f} mL/s.",
        recommended_action="Rerun pulsatile flow coupling and inspect inlet waveform." if not pulsatile_ok else "No action needed.",
    )

    split_range = _safe_float(flow.get("max_outlet_split_range_pp"))
    _add_check(
        checks,
        domain="flow_model",
        check_id="outlet_split_variation_is_stable",
        status="pass" if split_range <= 5.0 else "review",
        evidence_path=manifest_path,
        finding=f"Maximum outlet split variation = {split_range:.3f} percentage points.",
        recommended_action="Inspect outlet resistance/capacitance tuning if split variation grows." if split_range > 5.0 else "No action needed.",
    )

    frame_count = _safe_int(flow4d.get("frame_count"))
    color_min = _safe_float(flow4d.get("color_min"))
    color_max = _safe_float(flow4d.get("color_max"))
    flow4d_ok = frame_count >= 12 and color_max > color_min
    _add_check(
        checks,
        domain="flow_model",
        check_id="flow4d_visualization_frames_present",
        status="pass" if flow4d_ok else "review",
        evidence_path=flow4d.get("animation_gif", manifest_path),
        finding=f"4D flow frames = {frame_count}; velocity color range = {color_min:.3f} to {color_max:.3f}.",
        recommended_action="Regenerate the 4D flow visualization." if not flow4d_ok else "No action needed.",
    )

    _add_check(
        checks,
        domain="flow_model",
        check_id="flow_physics_validation_gap",
        status="review",
        evidence_path=limitations_path,
        finding="Flow remains a graph-coupled 1D/RCR surrogate, not calibrated 3D CFD, FSI, or measured physiology.",
        recommended_action="Add calibrated 0D/1D validation cases, 3D CFD comparison, pressure-flow data, and uncertainty bounds.",
        clinical_blocker=True,
    )

    rt_material_ok = (
        _safe_float(rt_material.get("vascular_fluid_volume_cm3")) > 0.0
        and _safe_float(rt_material.get("vessel_wall_volume_cm3")) > 0.0
        and _safe_float(rt_material.get("ptv_volume_cm3")) > 0.0
    )
    vascular_fluid_volume = _safe_float(rt_material.get("vascular_fluid_volume_cm3"))
    vessel_wall_volume = _safe_float(rt_material.get("vessel_wall_volume_cm3"))
    ptv_volume = _safe_float(rt_material.get("ptv_volume_cm3"))
    _add_check(
        checks,
        domain="radiotherapy",
        check_id="rt_material_regions_present",
        status="pass" if rt_material_ok else "fail",
        evidence_path=manifest_path,
        finding=(
            f"Vascular fluid/wall/PTV volumes = {vascular_fluid_volume:.3f} / "
            f"{vessel_wall_volume:.3f} / {ptv_volume:.3f} cm3."
        ),
        recommended_action="Rebuild the corrected RT QA material package." if not rt_material_ok else "No action needed.",
    )

    spatial_edges = _safe_int(rt_flow.get("selected_edge_count"))
    _add_check(
        checks,
        domain="radiotherapy",
        check_id="spatial_rt_flow_edges_selected",
        status="pass" if spatial_edges > 0 else "fail",
        evidence_path=manifest_path,
        finding=f"Spatial RT-flow selected edges = {spatial_edges}; top edge = {rt_flow.get('top_coupled_edge', 'unknown')}.",
        recommended_action="Rerun spatial RT-flow coupling against the corrected graph." if spatial_edges <= 0 else "No action needed.",
    )

    peak_delta = _safe_float(rt_flow.get("max_peak_delta_mgy"))
    trough_delta = _safe_float(rt_flow.get("max_trough_delta_mgy"))
    dose_states_ok = peak_delta > 0.0 and trough_delta > 0.0
    _add_check(
        checks,
        domain="radiotherapy",
        check_id="spatial_dose_states_have_flow_response",
        status="pass" if dose_states_ok else "review",
        evidence_path=manifest_path,
        finding=f"Max peak/trough dose deltas = {peak_delta:.3f} / {trough_delta:.3f} mGy.",
        recommended_action="Regenerate spatial flow-dose states if dose perturbations are absent." if not dose_states_ok else "No action needed.",
    )

    min_gamma_pass = _safe_float(gamma.get("min_pass_rate_percent"))
    max_p95_gamma = _safe_float(gamma.get("max_p95_gamma"))
    _add_check(
        checks,
        domain="radiotherapy",
        check_id="spatial_gamma_pass_rate_is_high",
        status="pass" if min_gamma_pass >= 95.0 and max_p95_gamma <= 1.0 else "fail",
        evidence_path=manifest_path,
        finding=f"Minimum gamma pass rate = {min_gamma_pass:.3f}%; max p95 gamma = {max_p95_gamma:.6f}.",
        recommended_action="Inspect dose perturbations and rerun gamma QA." if min_gamma_pass < 95.0 else "No action needed.",
    )

    peak_v95 = _safe_float(rt_flow.get("ptv_peak_v95_percent"))
    _add_check(
        checks,
        domain="radiotherapy",
        check_id="placeholder_ptv_peak_v95_review",
        status="pass" if peak_v95 >= 95.0 else "review",
        evidence_path=manifest_path,
        finding=f"Placeholder peak-state PTV V95 = {peak_v95:.3f}%.",
        recommended_action="Optimize or TPS-recalculate the plan before using PTV coverage as a radiotherapy claim."
        if peak_v95 < 95.0
        else "No action needed.",
    )

    dose_gap_flagged = (
        "synthetic" in limitations_text
        or "not tps" in limitations_text
        or "monte carlo" in limitations_text
        or "not a clinical" in limitations_text
    )
    limitations_ok = dose_gap_flagged and "not" in limitations_text and "clinical" in limitations_text
    _add_check(
        checks,
        domain="radiotherapy",
        check_id="clinical_dose_engine_validation_gap",
        status="review",
        evidence_path=limitations_path,
        finding="RT outputs are synthetic engineering dose states, not TPS-commissioned or Monte Carlo clinical dose calculations.",
        recommended_action="Connect the bundle to a TPS or Monte Carlo engine and compare DVH/gamma across static and pulsatile states.",
        clinical_blocker=True,
    )
    _add_check(
        checks,
        domain="documentation",
        check_id="limitations_explicitly_block_clinical_use",
        status="pass" if limitations_ok else "fail",
        evidence_path=limitations_path,
        finding="Limitations explicitly block clinical use and identify synthetic-dose constraints."
        if limitations_ok
        else "Clinical-use or synthetic-dose limitation language is incomplete.",
        recommended_action="Keep explicit non-clinical-use and synthetic-dose limitations in every release.",
    )

    command_ok = "build-corrected-branch-status-report" in command_text and "build-corrected-branch-release-package" in command_text
    _add_check(
        checks,
        domain="documentation",
        check_id="corrected_release_commands_documented",
        status="pass" if command_ok else "review",
        evidence_path=command_log_path,
        finding="Command log documents corrected status and release-package rebuild commands."
        if command_ok
        else "Command log does not clearly document corrected release rebuild commands.",
        recommended_action="Add corrected status/release rebuild commands to the reproducibility log.",
    )

    status_group_present = group_counts.get("status_package", 0) > 0 and group_counts.get("reports", 0) > 0
    _add_check(
        checks,
        domain="documentation",
        check_id="status_and_report_artifacts_packaged",
        status="pass" if status_group_present else "fail",
        evidence_path=artifact_index_path,
        finding=(
            f"Status-package artifacts = {group_counts.get('status_package', 0)}; "
            f"report artifacts = {group_counts.get('reports', 0)}."
        ),
        recommended_action="Regenerate the status package and release report artifacts." if not status_group_present else "No action needed.",
    )

    _add_check(
        checks,
        domain="population_validation",
        check_id="single_case_external_validation_gap",
        status="review",
        evidence_path=manifest_path,
        finding="This corrected release is one case/build state; cohort-scale anatomical and workflow robustness are not yet established.",
        recommended_action="Run this audit across multiple segmented CT/CTA/CTV cases and summarize organ/vessel/RT-flow variability.",
        clinical_blocker=True,
    )

    _add_check(
        checks,
        domain="population_validation",
        check_id="anatomical_equivalence_validation_gap",
        status="review",
        evidence_path=limitations_path,
        finding="The phantom is a useful research surrogate, but full anatomical equivalence to human subjects is not yet proven.",
        recommended_action="Validate organ volumes, body shape, vessel topology, branch landmarks, and vessel-organ distances against real cohorts.",
        clinical_blocker=True,
    )

    checks_tuple = tuple(checks)
    domain_scores = _domain_scores(checks_tuple)
    overall_score = 100.0 * sum(check.score for check in checks_tuple) / max(
        sum(check.max_score for check in checks_tuple), 1e-9
    )
    research_checks = tuple(check for check in checks_tuple if not check.clinical_blocker)
    research_score = 100.0 * sum(check.score for check in research_checks) / max(
        sum(check.max_score for check in research_checks), 1e-9
    )
    clinical_blockers = sum(check.clinical_blocker for check in checks_tuple)
    tier = _readiness_tier(checks_tuple, clinical_blockers)
    notes = (
        "corrected_branch_audit_scores_research_reproducibility_separately_from_clinical_claims",
        "clinical_blockers_are_expected_until_patient_specific_registration_flow_validation_and_tps_or_mc_dose_are_added",
        "scores_are_engineering_triage_not_regulatory_acceptance",
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
    report.write_text(_format_corrected_report(result))
    return result


def format_corrected_branch_release_audit_result(result: ReleaseReadinessAuditResult) -> str:
    return _format_corrected_report(result)
