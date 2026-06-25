from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import yaml


@dataclass(frozen=True)
class PatientBuildQACheck:
    check_id: str
    category: str
    status: str
    metric: str
    value: str
    threshold: str
    source_path: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PatientBuildQAResult:
    case_id: str
    patient_id: str
    output_dir: str
    source_build_manifest_path: str
    qa_yaml_path: str
    checks_csv_path: str
    report_path: str
    readiness_status: str
    pass_count: int
    review_count: int
    fail_count: int
    checks: tuple[PatientBuildQACheck, ...]
    recommended_actions: tuple[str, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _first_existing(paths: list[str | None]) -> Path | None:
    for raw in paths:
        if raw is None or str(raw) == "":
            continue
        path = Path(str(raw))
        if path.exists():
            return path
    return None


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _output_path(manifest: dict[str, Any], key: str) -> str | None:
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    value = outputs.get(key)
    return None if value is None or str(value) == "" else str(value)


def _step_by_id(manifest: dict[str, Any], step_id: str) -> dict[str, Any] | None:
    for step in manifest.get("steps", []):
        if isinstance(step, dict) and str(step.get("step_id")) == step_id:
            return step
    return None


def _check(
    checks: list[PatientBuildQACheck],
    *,
    check_id: str,
    category: str,
    status: str,
    metric: str,
    value: Any,
    threshold: Any,
    source_path: str | Path | None,
    notes: tuple[str, ...] = (),
) -> None:
    checks.append(
        PatientBuildQACheck(
            check_id=check_id,
            category=category,
            status=status,
            metric=metric,
            value=str(value),
            threshold=str(threshold),
            source_path=None if source_path is None else str(source_path),
            notes=notes,
        )
    )


def _read_failure_edges(edge_metrics_csv: str | Path | None, max_rows: int = 8) -> tuple[str, ...]:
    if edge_metrics_csv is None or not Path(edge_metrics_csv).exists():
        return ()
    rows: list[str] = []
    with Path(edge_metrics_csv).open(newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            status = str(row.get("status", ""))
            if status not in {"fail", "review"}:
                continue
            edge_id = str(row.get("edge_id", "unknown"))
            note = str(row.get("status_note", ""))
            outside = _safe_float(row.get("outside_body_fraction")) * 100.0
            bone = _safe_float(row.get("inside_bone_fraction")) * 100.0
            rows.append(f"{edge_id}:{status}:{note}:outside={outside:.2f}%:bone={bone:.2f}%")
            if len(rows) >= max_rows:
                break
    return tuple(rows)


def _write_checks_csv(path: Path, checks: tuple[PatientBuildQACheck, ...]) -> None:
    fieldnames = ["check_id", "category", "status", "metric", "value", "threshold", "source_path", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in checks:
            writer.writerow(
                {
                    "check_id": item.check_id,
                    "category": item.category,
                    "status": item.status,
                    "metric": item.metric,
                    "value": item.value,
                    "threshold": item.threshold,
                    "source_path": "" if item.source_path is None else item.source_path,
                    "notes": ";".join(item.notes),
                }
            )


def _write_yaml(path: Path, result: PatientBuildQAResult) -> None:
    payload = {
        "case_id": result.case_id,
        "patient_id": result.patient_id,
        "package_type": "patient_phantom_build_qa_gate",
        "source_build_manifest": result.source_build_manifest_path,
        "readiness_status": result.readiness_status,
        "summary": {
            "pass_count": result.pass_count,
            "review_count": result.review_count,
            "fail_count": result.fail_count,
        },
        "outputs": {
            "qa_yaml": result.qa_yaml_path,
            "checks_csv": result.checks_csv_path,
            "report": result.report_path,
        },
        "checks": [
            {
                "check_id": item.check_id,
                "category": item.category,
                "status": item.status,
                "metric": item.metric,
                "value": item.value,
                "threshold": item.threshold,
                "source_path": item.source_path,
                "notes": list(item.notes),
            }
            for item in result.checks
        ],
        "recommended_actions": list(result.recommended_actions),
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: PatientBuildQAResult) -> None:
    lines = [
        "# Patient Phantom Build QA Gate",
        "",
        f"Case ID: `{result.case_id}`",
        f"Patient/profile ID: `{result.patient_id}`",
        f"Readiness status: `{result.readiness_status}`",
        "",
        "## Summary",
        "",
        f"- Pass / review / fail: {result.pass_count} / {result.review_count} / {result.fail_count}",
        f"- Source build manifest: `{result.source_build_manifest_path}`",
        "",
        "## Checks",
        "",
        "| check | category | status | metric | value | threshold |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for item in result.checks:
        lines.append(
            f"| `{item.check_id}` | {item.category} | `{item.status}` | {item.metric} | {item.value} | {item.threshold} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Actions",
            "",
        ]
    )
    lines.extend(f"- {action}" for action in result.recommended_actions)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if result.readiness_status == "approved_research_use":
        lines.append("- The build passed the current research QA gate. This is still not a clinical device or patient-care artifact.")
    elif result.readiness_status == "review_required":
        lines.append("- The build completed but has non-blocking review items that should be resolved before claiming readiness.")
    else:
        lines.append("- The build is blocked by one or more QA failures and should not be used for clinical-style claims until corrected.")
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


def qa_patient_phantom_build(
    build_manifest_path: str | Path,
    output_dir: str | Path = "outputs/qa/patient_builds",
    case_id: str | None = None,
    report_path: str | Path | None = "outputs/reports/patient_phantom_build_qa_gate_stage001.md",
    max_organ_fail_edges: int = 0,
    max_organ_review_edges: int = 0,
    max_radius_fail_edges: int = 0,
    max_radius_review_edges: int = 0,
    expected_lumen_components: int = 1,
    max_overlap_after_cleanup: int = 0,
    max_outside_body_fraction: float = 0.0,
    max_flow_mass_residual_ml_s: float = 1e-4,
) -> PatientBuildQAResult:
    manifest_path = Path(build_manifest_path)
    manifest = _load_yaml(manifest_path)
    case = case_id or str(manifest.get("case_id", "patient_build"))
    patient_id = str(manifest.get("patient_id", "patient"))
    output = Path(output_dir)
    qa_yaml = output / f"{case}_patient_build_qa_gate_v001.yaml"
    checks_csv = output / f"{case}_patient_build_qa_checks_v001.csv"
    report = Path(report_path) if report_path is not None else output / f"{case}_patient_build_qa_gate_report_v001.md"
    build_root = manifest_path.parent
    checks: list[PatientBuildQACheck] = []
    notes = [
        "qa_gate_is_research_engineering_validation_not_patient_care_clearance",
        "readiness_requires_completed_build_plus_anatomy_vascular_flow_and_rt_output_checks",
    ]

    build_status = str(manifest.get("overall_status", ""))
    _check(
        checks,
        check_id="build_completed",
        category="build",
        status="pass" if build_status == "completed" else "fail",
        metric="overall_status",
        value=build_status,
        threshold="completed",
        source_path=manifest_path,
    )

    required_outputs = [
        "torso_spec",
        "vascular_graph",
        "voxelized_spec",
        "flow_boundary_config",
        "flow_1d_model",
        "coupled_flow_model",
        "rt_package_spec",
        "rt_planning_spec",
    ]
    for key in required_outputs:
        raw = _output_path(manifest, key)
        path = None if raw is None else Path(raw)
        _check(
            checks,
            check_id=f"output_exists_{key}",
            category="outputs",
            status="pass" if path is not None and path.exists() else "fail",
            metric=key,
            value="exists" if path is not None and path.exists() else "missing",
            threshold="exists",
            source_path=path,
        )

    abdominal_qa_path = _first_existing([_output_path(manifest, "abdominal_organ_qa")])
    if abdominal_qa_path is not None:
        abdominal_qa = _load_yaml(abdominal_qa_path)
        abdominal_summary = abdominal_qa.get("summary", {})
        abdominal_fail = _safe_int(abdominal_summary.get("fail_count"))
        abdominal_review = _safe_int(abdominal_summary.get("review_count"))
        abdominal_organs = _safe_int(abdominal_summary.get("organ_count"))
        _check(
            checks,
            check_id="abdominal_organ_preservation_fail_count",
            category="abdominal_organs",
            status="pass" if abdominal_fail == 0 else "fail",
            metric="fail_count",
            value=abdominal_fail,
            threshold="0",
            source_path=abdominal_qa_path,
        )
        _check(
            checks,
            check_id="abdominal_organ_preservation_review_count",
            category="abdominal_organs",
            status="pass" if abdominal_review == 0 else "review",
            metric="review_count",
            value=abdominal_review,
            threshold="0",
            source_path=abdominal_qa_path,
        )
        _check(
            checks,
            check_id="abdominal_organ_metric_count",
            category="abdominal_organs",
            status="pass" if abdominal_organs >= 14 else "review",
            metric="organ_count",
            value=abdominal_organs,
            threshold=">=14 including expanded GI placeholders",
            source_path=abdominal_qa_path,
        )
    else:
        notes.append("abdominal_organ_preservation_qa_not_present_for_this_build")

    voxel_path = _first_existing([_output_path(manifest, "voxelized_spec")])
    if voxel_path is not None:
        voxel_spec = _load_yaml(voxel_path)
        voxel_summary = voxel_spec.get("voxelization", {})
        connected = _safe_int(voxel_summary.get("connected_components"))
        overlap = _safe_int(voxel_summary.get("arterial_venous_overlap_voxels_after_cleanup"))
        outside = _safe_float(voxel_summary.get("outside_body_fraction_before_clip"))
        _check(
            checks,
            check_id="voxel_lumen_connectivity",
            category="voxelization",
            status="pass" if connected == expected_lumen_components else "review",
            metric="connected_components",
            value=connected,
            threshold=expected_lumen_components,
            source_path=voxel_path,
        )
        _check(
            checks,
            check_id="voxel_overlap_cleanup",
            category="voxelization",
            status="pass" if overlap <= max_overlap_after_cleanup else "fail",
            metric="arterial_venous_overlap_voxels_after_cleanup",
            value=overlap,
            threshold=f"<={max_overlap_after_cleanup}",
            source_path=voxel_path,
        )
        _check(
            checks,
            check_id="voxel_outside_body_fraction",
            category="voxelization",
            status="pass" if outside <= max_outside_body_fraction else "fail",
            metric="outside_body_fraction_before_clip",
            value=f"{outside:.6f}",
            threshold=f"<={max_outside_body_fraction:.6f}",
            source_path=voxel_path,
        )

    organ_spec_path = _find_first(build_root, "*vessel_organ_validation_spec_v001.yaml")
    if organ_spec_path is not None:
        organ_spec = _load_yaml(organ_spec_path)
        summary = organ_spec.get("summary", {})
        fail_count = _safe_int(summary.get("fail_count"))
        review_count = _safe_int(summary.get("review_count"))
        outside_count = _safe_int(summary.get("outside_body_edge_count"))
        edge_metrics = organ_spec.get("outputs", {}).get("edge_metrics_csv") if isinstance(organ_spec.get("outputs"), dict) else None
        edge_notes = _read_failure_edges(edge_metrics)
        _check(
            checks,
            check_id="organ_aware_vessel_fail_edges",
            category="vessel_anatomy",
            status="pass" if fail_count <= max_organ_fail_edges else "fail",
            metric="fail_count",
            value=fail_count,
            threshold=f"<={max_organ_fail_edges}",
            source_path=organ_spec_path,
            notes=edge_notes,
        )
        _check(
            checks,
            check_id="organ_aware_vessel_review_edges",
            category="vessel_anatomy",
            status="pass" if review_count <= max_organ_review_edges else "review",
            metric="review_count",
            value=review_count,
            threshold=f"<={max_organ_review_edges}",
            source_path=organ_spec_path,
        )
        _check(
            checks,
            check_id="organ_aware_outside_body_edges",
            category="vessel_anatomy",
            status="pass" if outside_count == 0 else "fail",
            metric="outside_body_edge_count",
            value=outside_count,
            threshold="0",
            source_path=organ_spec_path,
        )
    else:
        _check(
            checks,
            check_id="organ_aware_vessel_qa_present",
            category="vessel_anatomy",
            status="fail",
            metric="spec",
            value="missing",
            threshold="exists",
            source_path=build_root,
        )

    radius_spec_path = _find_first(build_root, "*vessel_radius_validation_spec_v001.yaml")
    if radius_spec_path is not None:
        radius_spec = _load_yaml(radius_spec_path)
        summary = radius_spec.get("summary", {})
        fail_count = _safe_int(summary.get("fail_count"))
        review_count = _safe_int(summary.get("review_count"))
        _check(
            checks,
            check_id="radius_aware_vessel_fail_edges",
            category="vessel_radius",
            status="pass" if fail_count <= max_radius_fail_edges else "fail",
            metric="fail_count",
            value=fail_count,
            threshold=f"<={max_radius_fail_edges}",
            source_path=radius_spec_path,
        )
        _check(
            checks,
            check_id="radius_aware_vessel_review_edges",
            category="vessel_radius",
            status="pass" if review_count <= max_radius_review_edges else "review",
            metric="review_count",
            value=review_count,
            threshold=f"<={max_radius_review_edges}",
            source_path=radius_spec_path,
        )
    else:
        _check(
            checks,
            check_id="radius_aware_vessel_qa_present",
            category="vessel_radius",
            status="fail",
            metric="spec",
            value="missing",
            threshold="exists",
            source_path=build_root,
        )

    flow_model_path = _first_existing([_output_path(manifest, "coupled_flow_model")])
    if flow_model_path is not None:
        flow_spec = _load_yaml(flow_model_path)
        summary = flow_spec.get("summary", {})
        residual = _safe_float(summary.get("max_abs_mass_balance_residual_ml_s"))
        _check(
            checks,
            check_id="flow_mass_balance_residual",
            category="flow",
            status="pass" if residual <= max_flow_mass_residual_ml_s else "fail",
            metric="max_abs_mass_balance_residual_ml_s",
            value=f"{residual:.9g}",
            threshold=f"<={max_flow_mass_residual_ml_s:.9g}",
            source_path=flow_model_path,
        )

    rt_spec_path = _first_existing([_output_path(manifest, "rt_planning_spec")])
    if rt_spec_path is not None:
        rt_spec = _load_yaml(rt_spec_path)
        outputs = rt_spec.get("outputs", {})
        dose_outputs = [
            "static_dose_nifti",
            "pulsatile_mean_dose_nifti",
            "pulsatile_peak_dose_nifti",
            "pulsatile_trough_dose_nifti",
            "pulsatile_delta_dose_nifti",
            "dose_metrics_csv",
            "dose_comparison_csv",
        ]
        missing = [key for key in dose_outputs if not (isinstance(outputs, dict) and outputs.get(key) and Path(str(outputs[key])).exists())]
        _check(
            checks,
            check_id="rt_planning_outputs_present",
            category="radiotherapy",
            status="pass" if not missing else "fail",
            metric="missing_rt_outputs",
            value="none" if not missing else ",".join(missing),
            threshold="none",
            source_path=rt_spec_path,
        )

    fail_count = sum(1 for item in checks if item.status == "fail")
    review_count = sum(1 for item in checks if item.status == "review")
    pass_count = sum(1 for item in checks if item.status == "pass")
    if fail_count:
        readiness = "blocked_anatomy_or_pipeline_qa_failed"
    elif review_count:
        readiness = "review_required"
    else:
        readiness = "approved_research_use"

    actions: list[str] = []
    if any(item.category == "abdominal_organs" and item.status in {"fail", "review"} for item in checks):
        actions.append("Review abdominal organ preservation QA; preserve BTCV organ labels or stage richer GI/bowel segmentations before making detailed abdominal-organ realism claims.")
    if any(item.check_id.startswith("organ_aware") and item.status == "fail" for item in checks):
        actions.append("Correct vessel-organ relationships: register/reroute patient vessel graph so centerlines remain inside body and near expected organs, then rerun patient build QA.")
    if any(item.category == "voxelization" and item.status == "fail" for item in checks):
        actions.append("Fix vascular voxelization overlap/outside-body issues before rerunning flow or RT claims.")
    if any(item.category == "flow" and item.status == "fail" for item in checks):
        actions.append("Review flow boundary conditions and graph connectivity; mass-balance residual exceeds the QA threshold.")
    if any(item.category == "radiotherapy" and item.status == "fail" for item in checks):
        actions.append("Regenerate the RT planning bundle and verify all static/pulsatile dose outputs exist.")
    if not actions:
        actions.append("Archive this QA gate with the build manifest; rerun when inputs, registration, vessel graph, flow, or RT outputs change.")

    result = PatientBuildQAResult(
        case_id=case,
        patient_id=patient_id,
        output_dir=str(output),
        source_build_manifest_path=str(manifest_path),
        qa_yaml_path=str(qa_yaml),
        checks_csv_path=str(checks_csv),
        report_path=str(report),
        readiness_status=readiness,
        pass_count=pass_count,
        review_count=review_count,
        fail_count=fail_count,
        checks=tuple(checks),
        recommended_actions=tuple(actions),
        notes=tuple(notes),
    )
    _write_checks_csv(checks_csv, result.checks)
    _write_yaml(qa_yaml, result)
    _write_report(report, result)
    return result


def format_patient_build_qa_result(result: PatientBuildQAResult) -> str:
    return "\n".join(
        [
            "Patient Phantom Build QA Gate",
            f"Case ID: {result.case_id}",
            f"Patient/profile ID: {result.patient_id}",
            f"Readiness status: {result.readiness_status}",
            f"Pass/review/fail: {result.pass_count}/{result.review_count}/{result.fail_count}",
            f"QA YAML: {result.qa_yaml_path}",
            f"Checks CSV: {result.checks_csv_path}",
            f"Report: {result.report_path}",
        ]
    )
