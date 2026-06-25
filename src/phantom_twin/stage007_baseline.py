from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
from typing import Any

import yaml


DEFAULT_STAGE007_ROOT = Path("outputs/patient_case_adapter/btcv_case0001_stage007_ivc_branch_reroute")
DEFAULT_STAGE007_CASE_ID = "btcv_case0001_stage007_left_iliac_radius_clean"
DEFAULT_STAGE007_RELEASE_ID = "stage007_left_iliac_radius_clean_rc1"
DEFAULT_STAGE007_RELEASE_MANIFEST = Path(
    "outputs/releases/stage007_left_iliac_radius_clean_rc1/"
    "stage007_left_iliac_radius_clean_rc1_release_manifest_v001.yaml"
)
DEFAULT_STAGE007_RELEASE_ARCHIVE = Path("stage007_left_iliac_radius_clean_rc1_compact_release.tar.gz")


@dataclass(frozen=True)
class Stage007BaselinePromotionResult:
    case_id: str
    baseline_id: str
    status: str
    stage_root: str
    active_manifest_path: str
    accepted_manifest_path: str
    report_path: str
    pointer_paths: tuple[str, ...]
    graph_path: str
    voxelized_spec_path: str
    release_manifest_path: str
    coupled_flow_model_path: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Stage007AcceptanceCheck:
    domain: str
    check_id: str
    status: str
    value: str
    threshold: str
    evidence_path: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Stage007AcceptanceSmokeResult:
    case_id: str
    baseline_id: str
    status: str
    output_dir: str
    checks_csv_path: str
    smoke_yaml_path: str
    report_path: str
    pass_count: int
    review_count: int
    fail_count: int
    flow_split_range_percentage_points: float
    checks: tuple[Stage007AcceptanceCheck, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Stage007ActiveBaselineResolution:
    status: str
    stage_root: str
    active_manifest_path: str | None
    graph_path: str | None
    voxelized_spec_path: str | None
    release_manifest_path: str | None
    flow_boundary_config_path: str | None
    flow_1d_model_path: str | None
    coupled_flow_model_path: str | None
    release_archive_path: str | None
    notes: tuple[str, ...]


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text())
    return data if isinstance(data, dict) else {}


def _read_csv_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or str(path) == "":
        return []
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return []
    with resolved.open(newline="") as csvfile:
        return [dict(row) for row in csv.DictReader(csvfile)]


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


def _resolve_output_path(raw_path: Any, manifest_path: Path | None = None) -> Path:
    if raw_path is None or str(raw_path) == "":
        return Path("")
    path = Path(str(raw_path))
    if path.is_absolute() or path.exists():
        return path
    if manifest_path is not None:
        candidate = manifest_path.parent / path
        if candidate.exists():
            return candidate
    return path


def _manifest_output_path(manifest: dict[str, Any], key: str, manifest_path: Path) -> Path:
    return _resolve_output_path(_as_mapping(manifest.get("outputs")).get(key), manifest_path)


def _key_artifact_path(manifest: dict[str, Any], *tokens: str, manifest_path: Path) -> Path:
    key_artifacts = _as_mapping(manifest.get("key_artifacts"))
    for key, value in key_artifacts.items():
        haystack = f"{key} {value}"
        if all(token in haystack for token in tokens):
            return _resolve_output_path(value, manifest_path)
    return Path("")


def _artifact_path(
    rows: list[dict[str, str]],
    *,
    group: str | None = None,
    file_type: str | None = None,
    tokens: tuple[str, ...] = (),
) -> Path:
    for row in rows:
        if group is not None and row.get("group") != group:
            continue
        if file_type is not None and row.get("file_type") != file_type:
            continue
        haystack = " ".join(str(value) for value in row.values())
        if all(token in haystack for token in tokens):
            raw_path = row.get("source_path") or row.get("packaged_path")
            if raw_path:
                return Path(raw_path)
    return Path("")


def _qa_metric(rows: list[dict[str, str]], category: str, metric: str) -> dict[str, str]:
    for row in rows:
        if row.get("category") == category and row.get("metric") == metric:
            return row
    return {}


def _qa_value(rows: list[dict[str, str]], category: str, metric: str, default: float = 0.0) -> float:
    return _safe_float(_qa_metric(rows, category, metric).get("value"), default)


def _qa_status_count(rows: list[dict[str, str]], status: str) -> int:
    return sum(1 for row in rows if row.get("status") == status)


def _write_text_pointer(path: Path, value: str | Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n")


def _existing_str(path: str | Path | None) -> str | None:
    if path is None or str(path) == "":
        return None
    resolved = Path(path)
    return str(resolved) if resolved.exists() else None


def _read_pointer(path: Path) -> Path | None:
    if not path.exists():
        return None
    value = path.read_text().strip()
    if not value:
        return None
    return Path(value)


def _write_acceptance_checks_csv(path: Path, checks: tuple[Stage007AcceptanceCheck, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["domain", "check_id", "status", "value", "threshold", "evidence_path", "notes"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for check in checks:
            writer.writerow(
                {
                    "domain": check.domain,
                    "check_id": check.check_id,
                    "status": check.status,
                    "value": check.value,
                    "threshold": check.threshold,
                    "evidence_path": check.evidence_path,
                    "notes": "; ".join(check.notes),
                }
            )


def _status_from_checks(checks: tuple[Stage007AcceptanceCheck, ...]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "review" for check in checks):
        return "review"
    return "pass"


def _format_count_status(pass_count: int, review_count: int, fail_count: int) -> str:
    return f"{pass_count} pass / {review_count} review / {fail_count} fail"


def _add_check(
    checks: list[Stage007AcceptanceCheck],
    *,
    domain: str,
    check_id: str,
    status: str,
    value: Any,
    threshold: str,
    evidence_path: str | Path | None = None,
    notes: tuple[str, ...] = (),
) -> None:
    checks.append(
        Stage007AcceptanceCheck(
            domain=domain,
            check_id=check_id,
            status=status,
            value=str(value),
            threshold=threshold,
            evidence_path="" if evidence_path is None else str(evidence_path),
            notes=notes,
        )
    )


def _write_promotion_report(path: Path, manifest: dict[str, Any], result: Stage007BaselinePromotionResult) -> None:
    summary = _as_mapping(manifest.get("summary"))
    qa_counts = _as_mapping(summary.get("qa_status_counts"))
    lines = [
        "# Stage 007 Active Baseline Promotion",
        "",
        f"- Case ID: `{result.case_id}`",
        f"- Baseline ID: `{result.baseline_id}`",
        f"- Status: `{result.status}`",
        f"- Release manifest: `{result.release_manifest_path}`",
        f"- Active graph: `{result.graph_path}`",
        f"- Voxelized vascular spec: `{result.voxelized_spec_path}`",
        f"- Coupled flow model: `{result.coupled_flow_model_path}`",
        f"- Release QA: `{qa_counts.get('pass', 0)} pass / {qa_counts.get('review', 0)} review / {qa_counts.get('fail', 0)} fail`",
        "",
        "## Pointer Files",
        "",
    ]
    lines.extend(f"- `{pointer}`" for pointer in result.pointer_paths)
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This promotion supersedes the older Stage 007 accepted baseline that still carried a radius-review item and outside-body clipping note.",
            "- The clean baseline remains a research/engineering demonstrator, not a clinical acceptance package.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def resolve_stage007_active_baseline(
    *,
    stage_root: str | Path = DEFAULT_STAGE007_ROOT,
    release_manifest_path: str | Path | None = None,
) -> Stage007ActiveBaselineResolution:
    stage_root_path = Path(stage_root)
    notes: list[str] = []
    active_manifest = stage_root_path / "stage007_active_baseline_manifest_v001.yaml"
    if not active_manifest.exists():
        accepted_v2 = stage_root_path / "stage007_accepted_baseline_manifest_v002.yaml"
        if accepted_v2.exists():
            active_manifest = accepted_v2
            notes.append("active_manifest_missing_using_accepted_v2_manifest")

    active = _load_yaml(active_manifest)
    active_artifacts = _as_mapping(active.get("active_artifacts"))

    def pointer_or_manifest(pointer_name: str, manifest_key: str) -> Path | None:
        pointer = _read_pointer(stage_root_path / pointer_name)
        if pointer is not None and pointer.exists():
            notes.append(f"{manifest_key}_resolved_from_pointer")
            return pointer
        manifest_value = active_artifacts.get(manifest_key)
        if manifest_value is not None and Path(str(manifest_value)).exists():
            notes.append(f"{manifest_key}_resolved_from_active_manifest")
            return Path(str(manifest_value))
        return None

    graph = pointer_or_manifest("latest_stage007_active_graph_path.txt", "graph")
    voxelized_spec = pointer_or_manifest("latest_stage007_active_voxelized_spec_path.txt", "voxelized_spec")
    flow_boundary = pointer_or_manifest("latest_stage007_active_flow_boundary_path.txt", "flow_boundary_conditions")
    flow_1d = pointer_or_manifest("latest_stage007_active_flow_1d_model_path.txt", "flow_1d_model")
    coupled_flow = pointer_or_manifest("latest_stage007_active_coupled_flow_model_path.txt", "coupled_pulsatile_flow_model")
    archive = _read_pointer(stage_root_path / "latest_stage007_active_release_archive_path.txt")
    if archive is None:
        archive_raw = _as_mapping(active.get("release")).get("archive")
        archive = Path(str(archive_raw)) if archive_raw else None
    release_manifest = (
        Path(release_manifest_path)
        if release_manifest_path is not None and str(release_manifest_path) != ""
        else _read_pointer(stage_root_path / "latest_stage007_active_release_manifest_path.txt")
    )
    if release_manifest is None:
        release_raw = _as_mapping(active.get("release")).get("manifest")
        release_manifest = Path(str(release_raw)) if release_raw else None
    if release_manifest is None and DEFAULT_STAGE007_RELEASE_MANIFEST.exists():
        release_manifest = DEFAULT_STAGE007_RELEASE_MANIFEST
        notes.append("release_manifest_resolved_from_default_clean_release")

    release = _load_yaml(release_manifest)
    artifact_rows = _read_csv_rows(_resolve_output_path(_as_mapping(release.get("outputs")).get("artifact_index_csv"), release_manifest))
    if graph is None:
        candidate = _artifact_path(
            artifact_rows,
            group="source_dependency",
            file_type="yaml",
            tokens=("radius_tuned_vascular_graph",),
        )
        graph = candidate if str(candidate) and candidate.exists() else None
        if graph is not None:
            notes.append("graph_resolved_from_release_artifact_index")
    if voxelized_spec is None:
        candidate = _key_artifact_path(
            release,
            "vascular_network_voxelized_spec",
            manifest_path=Path(release_manifest) if release_manifest is not None else DEFAULT_STAGE007_RELEASE_MANIFEST,
        )
        voxelized_spec = candidate if str(candidate) and candidate.exists() else None
        if voxelized_spec is not None:
            notes.append("voxelized_spec_resolved_from_release_key_artifacts")
    if flow_boundary is None:
        candidate = _artifact_path(
            artifact_rows,
            group="flow_boundary_conditions",
            file_type="yaml",
            tokens=("flow_boundary_conditions",),
        )
        flow_boundary = candidate if str(candidate) and candidate.exists() else None
    if flow_1d is None:
        candidate = _artifact_path(
            artifact_rows,
            group="steady_1d_flow",
            file_type="yaml",
            tokens=("flow_1d_model",),
        )
        flow_1d = candidate if str(candidate) and candidate.exists() else None
    if coupled_flow is None:
        candidate = _key_artifact_path(
            release,
            "coupled_pulsatile_flow_model",
            manifest_path=Path(release_manifest) if release_manifest is not None else DEFAULT_STAGE007_RELEASE_MANIFEST,
        )
        coupled_flow = candidate if str(candidate) and candidate.exists() else None

    status = "ready" if graph is not None and voxelized_spec is not None else "missing_required_artifacts"
    if status != "ready":
        notes.append("stage007_active_baseline_missing_graph_or_voxelized_spec")
    return Stage007ActiveBaselineResolution(
        status=status,
        stage_root=str(stage_root_path),
        active_manifest_path=_existing_str(active_manifest),
        graph_path=_existing_str(graph),
        voxelized_spec_path=_existing_str(voxelized_spec),
        release_manifest_path=_existing_str(release_manifest),
        flow_boundary_config_path=_existing_str(flow_boundary),
        flow_1d_model_path=_existing_str(flow_1d),
        coupled_flow_model_path=_existing_str(coupled_flow),
        release_archive_path=_existing_str(archive),
        notes=tuple(notes),
    )


def _write_acceptance_report(path: Path, result: Stage007AcceptanceSmokeResult) -> None:
    lines = [
        "# Stage 007 Acceptance Smoke Report",
        "",
        f"- Case ID: `{result.case_id}`",
        f"- Baseline ID: `{result.baseline_id}`",
        f"- Status: `{result.status}`",
        f"- Checks: `{_format_count_status(result.pass_count, result.review_count, result.fail_count)}`",
        f"- Flow outlet split range: `{result.flow_split_range_percentage_points:.3f}` percentage points",
        "",
        "## Checks",
        "",
    ]
    for check in result.checks:
        evidence = f" evidence=`{check.evidence_path}`" if check.evidence_path else ""
        lines.append(
            f"- `{check.status}` `{check.domain}/{check.check_id}`: {check.value} "
            f"against `{check.threshold}`.{evidence}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A `pass` here means the existing clean Stage 007 release can be used as the active research baseline without rerunning heavy volume or render generation.",
            "- This smoke test is an engineering reproducibility gate; patient-specific CTA registration, calibrated physiology, and clinical dose calculation remain future validation work.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def promote_stage007_clean_baseline(
    *,
    release_manifest_path: str | Path = DEFAULT_STAGE007_RELEASE_MANIFEST,
    stage_root: str | Path = DEFAULT_STAGE007_ROOT,
    case_id: str = DEFAULT_STAGE007_CASE_ID,
    baseline_id: str = DEFAULT_STAGE007_RELEASE_ID,
    graph_path: str | Path | None = None,
    voxelized_spec_path: str | Path | None = None,
    release_archive_path: str | Path | None = DEFAULT_STAGE007_RELEASE_ARCHIVE,
    report_path: str | Path = "outputs/reports/btcv_case0001_stage007_left_iliac_radius_clean_active_baseline_promotion.md",
    write_accepted_aliases: bool = True,
) -> Stage007BaselinePromotionResult:
    release_manifest = Path(release_manifest_path)
    manifest = _load_yaml(release_manifest)
    outputs = _as_mapping(manifest.get("outputs"))
    artifact_rows = _read_csv_rows(_resolve_output_path(outputs.get("artifact_index_csv"), release_manifest))

    graph = Path(graph_path) if graph_path is not None else _artifact_path(
        artifact_rows,
        group="source_dependency",
        file_type="yaml",
        tokens=("radius_tuned_vascular_graph",),
    )
    if str(graph) == "":
        graph = _key_artifact_path(manifest, "radius_tuned_vascular_graph", manifest_path=release_manifest)

    voxel_spec = Path(voxelized_spec_path) if voxelized_spec_path is not None else _key_artifact_path(
        manifest,
        "vascular_network_voxelized_spec",
        manifest_path=release_manifest,
    )
    if str(voxel_spec) == "":
        voxel_spec = _artifact_path(
            artifact_rows,
            group="vascular_voxelization",
            file_type="yaml",
            tokens=("voxelized_spec",),
        )

    flow_boundary = _artifact_path(
        artifact_rows,
        group="flow_boundary_conditions",
        file_type="yaml",
        tokens=("flow_boundary_conditions",),
    )
    flow_1d = _artifact_path(
        artifact_rows,
        group="steady_1d_flow",
        file_type="yaml",
        tokens=("flow_1d_model",),
    )
    coupled_flow = _key_artifact_path(
        manifest,
        "coupled_pulsatile_flow_model",
        manifest_path=release_manifest,
    )
    if str(coupled_flow) == "":
        coupled_flow = _artifact_path(
            artifact_rows,
            group="coupled_pulsatile_flow",
            file_type="yaml",
            tokens=("coupled_pulsatile_flow_model",),
        )

    stage_root_path = Path(stage_root)
    active_manifest = stage_root_path / "stage007_active_baseline_manifest_v001.yaml"
    accepted_manifest = stage_root_path / "stage007_accepted_baseline_manifest_v002.yaml"
    qa_summary = _resolve_output_path(outputs.get("qa_summary_csv"), release_manifest)
    artifact_index = _resolve_output_path(outputs.get("artifact_index_csv"), release_manifest)
    atlas = _resolve_output_path(outputs.get("atlas_png"), release_manifest)
    report = Path(report_path)
    archive = Path(release_archive_path) if release_archive_path is not None else Path("")

    pointer_payload: dict[str, Path] = {
        "latest_stage007_active_graph_path.txt": graph,
        "latest_stage007_active_voxelized_spec_path.txt": voxel_spec,
        "latest_stage007_active_release_manifest_path.txt": release_manifest,
        "latest_stage007_active_flow_boundary_path.txt": flow_boundary,
        "latest_stage007_active_flow_1d_model_path.txt": flow_1d,
        "latest_stage007_active_coupled_flow_model_path.txt": coupled_flow,
    }
    if str(archive) != "":
        pointer_payload["latest_stage007_active_release_archive_path.txt"] = archive
    if write_accepted_aliases:
        pointer_payload.update(
            {
                "latest_stage007_accepted_graph_path.txt": graph,
                "latest_stage007_accepted_voxelized_spec_path.txt": voxel_spec,
                "latest_stage007_accepted_release_manifest_path.txt": release_manifest,
            }
        )

    pointer_paths: list[str] = []
    for filename, target in pointer_payload.items():
        pointer = stage_root_path / filename
        _write_text_pointer(pointer, target)
        pointer_paths.append(str(pointer))

    qa_counts = _as_mapping(_as_mapping(manifest.get("summary")).get("qa_status_counts"))
    baseline_manifest = {
        "case_id": case_id,
        "baseline_id": baseline_id,
        "status": "active_research_release_candidate",
        "promoted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "stage007_clean_branch_rich_btcv_vascular_realism",
        "supersedes": {
            "previous_manifest": str(stage_root_path / "stage007_accepted_baseline_manifest_v001.yaml"),
            "reason": "left_iliac_radius_cleanup_removed_final_radius_review_and_outside_body_margin_repair_removed_clipping_review",
        },
        "release": {
            "manifest": str(release_manifest),
            "archive": str(archive) if str(archive) else "",
            "readiness_status": manifest.get("readiness_status", ""),
            "qa_status_counts": dict(qa_counts),
        },
        "active_artifacts": {
            "graph": str(graph),
            "voxelized_spec": str(voxel_spec),
            "flow_boundary_conditions": str(flow_boundary),
            "flow_1d_model": str(flow_1d),
            "coupled_pulsatile_flow_model": str(coupled_flow),
            "qa_summary_csv": str(qa_summary),
            "artifact_index_csv": str(artifact_index),
            "release_atlas_png": str(atlas),
        },
        "acceptance_gates": {
            "release_qa_status_counts": dict(qa_counts),
            "requires_acceptance_smoke": True,
            "recommended_command": (
                "python -m phantom_twin.cli run-stage007-acceptance-smoke "
                f"--release-manifest {release_manifest}"
            ),
        },
        "pointer_files": pointer_paths,
        "limitations": [
            "research_engineering_baseline_not_clinical_release",
            "flow_is_graph_coupled_1d_placeholder_not_3d_cfd",
            "remaining_branch_detail_depends_on_available_cta_ctv_segmentations",
        ],
    }
    active_manifest.parent.mkdir(parents=True, exist_ok=True)
    active_manifest.write_text(yaml.safe_dump(baseline_manifest, sort_keys=False))
    if write_accepted_aliases:
        accepted_manifest.write_text(yaml.safe_dump({**baseline_manifest, "status": "accepted_active_research_release_candidate"}, sort_keys=False))

    result = Stage007BaselinePromotionResult(
        case_id=case_id,
        baseline_id=baseline_id,
        status="active_research_release_candidate",
        stage_root=str(stage_root_path),
        active_manifest_path=str(active_manifest),
        accepted_manifest_path=str(accepted_manifest) if write_accepted_aliases else "",
        report_path=str(report),
        pointer_paths=tuple(pointer_paths),
        graph_path=str(graph),
        voxelized_spec_path=str(voxel_spec),
        release_manifest_path=str(release_manifest),
        coupled_flow_model_path=str(coupled_flow),
        notes=(
            "active pointers updated",
            "accepted pointer aliases updated" if write_accepted_aliases else "accepted pointer aliases preserved",
        ),
    )
    _write_promotion_report(report, manifest, result)
    return result


def run_stage007_acceptance_smoke(
    *,
    release_manifest_path: str | Path = DEFAULT_STAGE007_RELEASE_MANIFEST,
    baseline_manifest_path: str | Path | None = None,
    release_archive_path: str | Path | None = DEFAULT_STAGE007_RELEASE_ARCHIVE,
    output_dir: str | Path = "outputs/acceptance/stage007_left_iliac_radius_clean",
    report_path: str | Path = "outputs/reports/btcv_case0001_stage007_left_iliac_radius_clean_acceptance_smoke.md",
    case_id: str | None = None,
    baseline_id: str | None = None,
    flow_mass_residual_threshold_ml_s: float = 1e-4,
    flow_split_review_threshold_pp: float = 10.0,
    flow_split_fail_threshold_pp: float = 15.0,
    min_boundary_count: int = 10,
) -> Stage007AcceptanceSmokeResult:
    release_manifest = Path(release_manifest_path)
    release = _load_yaml(release_manifest)
    baseline = _load_yaml(baseline_manifest_path) if baseline_manifest_path is not None else {}
    outputs = _as_mapping(release.get("outputs"))
    qa_rows = _read_csv_rows(_resolve_output_path(outputs.get("qa_summary_csv"), release_manifest))
    artifact_rows = _read_csv_rows(_resolve_output_path(outputs.get("artifact_index_csv"), release_manifest))

    selected_case_id = case_id or str(release.get("case_id") or baseline.get("case_id") or DEFAULT_STAGE007_CASE_ID)
    selected_baseline_id = baseline_id or str(release.get("release_id") or baseline.get("baseline_id") or DEFAULT_STAGE007_RELEASE_ID)
    output_path = Path(output_dir)
    checks: list[Stage007AcceptanceCheck] = []

    readiness_status = str(release.get("readiness_status", ""))
    _add_check(
        checks,
        domain="release",
        check_id="readiness_status",
        status="pass" if readiness_status == "research_release_candidate" else "fail",
        value=readiness_status,
        threshold="research_release_candidate",
        evidence_path=release_manifest,
    )

    qa_fail_count = _qa_status_count(qa_rows, "fail")
    qa_review_count = _qa_status_count(qa_rows, "review")
    qa_pass_count = _qa_status_count(qa_rows, "pass")
    _add_check(
        checks,
        domain="release",
        check_id="qa_summary_no_failures",
        status="pass" if qa_fail_count == 0 and bool(qa_rows) else "fail",
        value=qa_fail_count,
        threshold="== 0",
        evidence_path=outputs.get("qa_summary_csv"),
        notes=(f"pass_count={qa_pass_count}", f"review_count={qa_review_count}"),
    )
    _add_check(
        checks,
        domain="release",
        check_id="qa_summary_no_reviews",
        status="pass" if qa_review_count == 0 and bool(qa_rows) else "review",
        value=qa_review_count,
        threshold="== 0",
        evidence_path=outputs.get("qa_summary_csv"),
    )

    missing_artifacts = [
        row.get("source_path", "")
        for row in artifact_rows
        if str(row.get("exists", "")).lower() != "true" or not Path(row.get("source_path", "")).exists()
    ]
    _add_check(
        checks,
        domain="release",
        check_id="artifact_index_sources_exist",
        status="pass" if artifact_rows and not missing_artifacts else "fail",
        value=f"{len(artifact_rows) - len(missing_artifacts)}/{len(artifact_rows)}",
        threshold="all indexed source artifacts exist",
        evidence_path=outputs.get("artifact_index_csv"),
        notes=tuple(missing_artifacts[:3]),
    )

    vascular_checks = (
        ("connected_lumen_components", _qa_value(qa_rows, "vascular_domain", "connected_lumen_components"), 1.0, "== 1"),
        ("arterial_components", _qa_value(qa_rows, "vascular_domain", "arterial_components"), 1.0, "== 1"),
        ("venous_components", _qa_value(qa_rows, "vascular_domain", "venous_components"), 1.0, "== 1"),
        (
            "arterial_venous_overlap_after_cleanup_voxels",
            _qa_value(qa_rows, "vascular_domain", "arterial_venous_overlap_after_cleanup_voxels"),
            0.0,
            "== 0",
        ),
        (
            "outside_body_fraction_before_clip",
            _qa_value(qa_rows, "vascular_domain", "outside_body_fraction_before_clip"),
            0.0,
            "<= 0",
        ),
    )
    for metric, value, expected, threshold in vascular_checks:
        if metric == "outside_body_fraction_before_clip":
            ok = value <= expected
        else:
            ok = abs(value - expected) <= 1e-9
        row = _qa_metric(qa_rows, "vascular_domain", metric)
        _add_check(
            checks,
            domain="vascular_domain",
            check_id=metric,
            status="pass" if ok else "fail",
            value=f"{value:.8g}",
            threshold=threshold,
            evidence_path=row.get("source_path", ""),
        )

    organ_review = _qa_value(qa_rows, "organ_aware_vascular_anatomy", "review_count")
    organ_fail = _qa_value(qa_rows, "organ_aware_vascular_anatomy", "fail_count")
    bone_edges = _qa_value(qa_rows, "organ_aware_vascular_anatomy", "bone_intersection_edge_count")
    _add_check(
        checks,
        domain="vascular_anatomy",
        check_id="organ_aware_review_fail_bone_intersections",
        status="pass" if organ_review == 0 and organ_fail == 0 and bone_edges == 0 else "fail",
        value=f"review={organ_review:.0f}; fail={organ_fail:.0f}; bone_edges={bone_edges:.0f}",
        threshold="all == 0",
        evidence_path=_qa_metric(qa_rows, "organ_aware_vascular_anatomy", "fail_count").get("source_path", ""),
    )

    radius_review = _qa_value(qa_rows, "radius_aware_vascular_anatomy", "review_count")
    radius_fail = _qa_value(qa_rows, "radius_aware_vascular_anatomy", "fail_count")
    radius_tune = _qa_value(qa_rows, "radius_aware_vascular_anatomy", "radius_tuning_candidate_count")
    reroute = _qa_value(qa_rows, "radius_aware_vascular_anatomy", "reroute_candidate_count")
    _add_check(
        checks,
        domain="vascular_anatomy",
        check_id="radius_aware_review_fail_candidates",
        status="pass" if radius_review == 0 and radius_fail == 0 and radius_tune == 0 and reroute == 0 else "fail",
        value=f"review={radius_review:.0f}; fail={radius_fail:.0f}; tune={radius_tune:.0f}; reroute={reroute:.0f}",
        threshold="all == 0",
        evidence_path=_qa_metric(qa_rows, "radius_aware_vascular_anatomy", "fail_count").get("source_path", ""),
    )

    boundary_config = _artifact_path(
        artifact_rows,
        group="flow_boundary_conditions",
        file_type="yaml",
        tokens=("flow_boundary_conditions",),
    )
    boundary_yaml = _load_yaml(boundary_config)
    boundaries = boundary_yaml.get("boundaries", [])
    boundary_count = len(boundaries) if isinstance(boundaries, list) else 0
    mapped_count = sum(1 for row in boundaries if isinstance(row, dict) and row.get("status") == "mapped")
    _add_check(
        checks,
        domain="flow",
        check_id="boundary_conditions_mapped",
        status="pass" if boundary_count >= min_boundary_count and mapped_count == boundary_count else "fail",
        value=f"{mapped_count}/{boundary_count}",
        threshold=f"all mapped and boundary_count >= {min_boundary_count}",
        evidence_path=boundary_config,
    )

    steady_residual = _qa_value(qa_rows, "steady_1d_flow", "max_abs_mass_balance_residual_ml_s")
    _add_check(
        checks,
        domain="flow",
        check_id="steady_1d_mass_balance",
        status="pass" if steady_residual <= flow_mass_residual_threshold_ml_s else "fail",
        value=f"{steady_residual:.12g}",
        threshold=f"<= {flow_mass_residual_threshold_ml_s:g} mL/s",
        evidence_path=_qa_metric(qa_rows, "steady_1d_flow", "max_abs_mass_balance_residual_ml_s").get("source_path", ""),
    )

    coupled_model = _key_artifact_path(release, "coupled_pulsatile_flow_model", manifest_path=release_manifest)
    if str(coupled_model) == "":
        coupled_model = _artifact_path(
            artifact_rows,
            group="coupled_pulsatile_flow",
            file_type="yaml",
            tokens=("coupled_pulsatile_flow_model",),
        )
    coupled_yaml = _load_yaml(coupled_model)
    coupled_summary = _as_mapping(coupled_yaml.get("summary"))
    coupled_residual = _safe_float(
        coupled_summary.get(
            "max_abs_mass_balance_residual_ml_s",
            _qa_value(qa_rows, "coupled_pulsatile_flow", "max_abs_mass_balance_residual_ml_s"),
        )
    )
    _add_check(
        checks,
        domain="flow",
        check_id="coupled_pulsatile_mass_balance",
        status="pass" if coupled_residual <= flow_mass_residual_threshold_ml_s else "fail",
        value=f"{coupled_residual:.12g}",
        threshold=f"<= {flow_mass_residual_threshold_ml_s:g} mL/s",
        evidence_path=coupled_model,
    )

    flow_split = _safe_float(coupled_summary.get("max_outlet_split_range_percentage_points"), -1.0)
    if flow_split < 0:
        flow_split_status = "fail"
    elif flow_split > flow_split_fail_threshold_pp:
        flow_split_status = "fail"
    elif flow_split > flow_split_review_threshold_pp:
        flow_split_status = "review"
    else:
        flow_split_status = "pass"
    _add_check(
        checks,
        domain="flow",
        check_id="outlet_split_range",
        status=flow_split_status,
        value=f"{flow_split:.6f} percentage points",
        threshold=f"pass <= {flow_split_review_threshold_pp:g}; fail > {flow_split_fail_threshold_pp:g}",
        evidence_path=coupled_model,
        notes=("tracks flow-distribution impact after radius cleanup",),
    )

    archive = Path(release_archive_path) if release_archive_path is not None else Path("")
    archive_status = "pass" if archive.exists() else "review"
    _add_check(
        checks,
        domain="release",
        check_id="compact_release_archive_available",
        status=archive_status,
        value=str(archive.exists()),
        threshold="archive exists, or review if not packaged",
        evidence_path=archive if str(archive) else None,
    )

    checks_tuple = tuple(checks)
    pass_count = sum(1 for check in checks_tuple if check.status == "pass")
    review_count = sum(1 for check in checks_tuple if check.status == "review")
    fail_count = sum(1 for check in checks_tuple if check.status == "fail")
    status = _status_from_checks(checks_tuple)
    checks_csv = output_path / f"{selected_case_id}_acceptance_smoke_checks_v001.csv"
    smoke_yaml = output_path / f"{selected_case_id}_acceptance_smoke_v001.yaml"
    report = Path(report_path)

    result = Stage007AcceptanceSmokeResult(
        case_id=selected_case_id,
        baseline_id=selected_baseline_id,
        status=status,
        output_dir=str(output_path),
        checks_csv_path=str(checks_csv),
        smoke_yaml_path=str(smoke_yaml),
        report_path=str(report),
        pass_count=pass_count,
        review_count=review_count,
        fail_count=fail_count,
        flow_split_range_percentage_points=flow_split,
        checks=checks_tuple,
        notes=(
            "acceptance smoke validates existing release artifacts only",
            "no heavy mesh or NIfTI regeneration performed",
        ),
    )
    _write_acceptance_checks_csv(checks_csv, checks_tuple)
    smoke_yaml.parent.mkdir(parents=True, exist_ok=True)
    smoke_yaml.write_text(
        yaml.safe_dump(
            {
                "case_id": selected_case_id,
                "baseline_id": selected_baseline_id,
                "status": status,
                "release_manifest": str(release_manifest),
                "baseline_manifest": "" if baseline_manifest_path is None else str(baseline_manifest_path),
                "release_archive": "" if release_archive_path is None else str(release_archive_path),
                "summary": {
                    "pass_count": pass_count,
                    "review_count": review_count,
                    "fail_count": fail_count,
                    "flow_split_range_percentage_points": flow_split,
                },
                "thresholds": {
                    "flow_mass_residual_threshold_ml_s": flow_mass_residual_threshold_ml_s,
                    "flow_split_review_threshold_pp": flow_split_review_threshold_pp,
                    "flow_split_fail_threshold_pp": flow_split_fail_threshold_pp,
                    "min_boundary_count": min_boundary_count,
                },
                "outputs": {
                    "checks_csv": str(checks_csv),
                    "report": str(report),
                },
                "checks": [
                    {
                        "domain": check.domain,
                        "check_id": check.check_id,
                        "status": check.status,
                        "value": check.value,
                        "threshold": check.threshold,
                        "evidence_path": check.evidence_path,
                        "notes": list(check.notes),
                    }
                    for check in checks_tuple
                ],
                "notes": list(result.notes),
            },
            sort_keys=False,
        )
    )
    _write_acceptance_report(report, result)
    return result


def format_stage007_baseline_promotion_result(result: Stage007BaselinePromotionResult) -> str:
    return "\n".join(
        [
            "Stage 007 clean baseline promoted.",
            f"- Case ID: {result.case_id}",
            f"- Baseline ID: {result.baseline_id}",
            f"- Status: {result.status}",
            f"- Active manifest: {result.active_manifest_path}",
            f"- Active graph: {result.graph_path}",
            f"- Voxelized spec: {result.voxelized_spec_path}",
            f"- Coupled flow model: {result.coupled_flow_model_path}",
            f"- Pointer files: {len(result.pointer_paths)}",
            f"- Report: {result.report_path}",
        ]
    )


def format_stage007_acceptance_smoke_result(result: Stage007AcceptanceSmokeResult) -> str:
    return "\n".join(
        [
            "Stage 007 acceptance smoke complete.",
            f"- Case ID: {result.case_id}",
            f"- Baseline ID: {result.baseline_id}",
            f"- Status: {result.status}",
            f"- Checks: {_format_count_status(result.pass_count, result.review_count, result.fail_count)}",
            f"- Flow outlet split range: {result.flow_split_range_percentage_points:.3f} percentage points",
            f"- Smoke YAML: {result.smoke_yaml_path}",
            f"- Checks CSV: {result.checks_csv_path}",
            f"- Report: {result.report_path}",
        ]
    )
