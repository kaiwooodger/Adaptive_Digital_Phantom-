from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import os
import shutil
from typing import Any

import yaml


VOLUME_SUFFIXES = (".nii.gz", ".nii", ".dcm", ".dicom", ".mha", ".mhd", ".nrrd")


@dataclass(frozen=True)
class ReleaseArtifact:
    group: str
    role: str
    file_type: str
    source_path: str
    exists: bool
    size_bytes: int
    sha256: str
    copy_policy: str
    packaged_path: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseMetric:
    category: str
    metric: str
    value: str
    threshold: str
    status: str
    source_path: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ResearchReleasePackageResult:
    case_id: str
    release_id: str
    output_dir: str
    manifest_yaml_path: str
    artifact_index_csv_path: str
    qa_summary_csv_path: str
    command_log_path: str
    limitations_markdown_path: str
    atlas_png_path: str
    report_path: str
    readiness_status: str
    summary: dict[str, Any]
    artifacts: tuple[ReleaseArtifact, ...]
    metrics: tuple[ReleaseMetric, ...]
    notes: tuple[str, ...]


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Release package atlas generation requires matplotlib.") from exc
    return plt


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
    if not resolved.exists():
        return []
    with resolved.open(newline="") as csvfile:
        return [dict(row) for row in csv.DictReader(csvfile)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _find_first(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    return matches[0] if matches else None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _file_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return "nifti_volume"
    if name.endswith((".yaml", ".yml")):
        return "yaml"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".png"):
        return "png"
    if name.endswith(".md"):
        return "markdown"
    if name.endswith((".stl", ".ply", ".obj")):
        return "mesh"
    return path.suffix.lower().lstrip(".") or "unknown"


def _artifact_group(path: Path) -> str:
    parts = set(path.parts)
    if "vascular_network_voxelized" in parts:
        return "vascular_voxelization"
    if "vessel_organ_anatomy" in parts:
        return "vessel_organ_validation"
    if "vessel_radius_anatomy" in parts:
        return "vessel_radius_validation"
    if "flow_boundary_conditions" in parts:
        return "flow_boundary_conditions"
    if "flow_1d" in parts:
        return "steady_1d_flow"
    if "flow_coupled_pulsatile" in parts:
        return "coupled_pulsatile_flow"
    if "radiotherapy_qa_package" in parts:
        return "radiotherapy_qa"
    if "rt_planning_bundle" in parts:
        return "rt_planning"
    if "dose_gamma_qa" in parts:
        return "dose_gamma_qa"
    if "reports" in parts:
        return "reports"
    return "source_dependency"


def _sha256(path: Path, max_size_bytes: int) -> str:
    if not path.exists() or path.stat().st_size > max_size_bytes:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_large_or_volume(path: Path, large_threshold_bytes: int) -> bool:
    lower = path.name.lower()
    return lower.endswith(VOLUME_SUFFIXES) or path.stat().st_size > large_threshold_bytes


def _relative_destination(path: Path, stage_root: Path, reports_dir: Path, output_dir: Path) -> Path:
    if path.is_relative_to(stage_root):
        return output_dir / "artifacts" / "stage" / path.relative_to(stage_root)
    if path.is_relative_to(reports_dir):
        return output_dir / "artifacts" / "reports" / path.name
    return output_dir / "artifacts" / "dependencies" / path.name


def _discover_artifact_paths(stage_root: Path, reports_dir: Path, case_id: str) -> list[Path]:
    paths: list[Path] = []
    if stage_root.exists():
        paths.extend(path for path in stage_root.rglob("*") if path.is_file())
    if reports_dir.exists():
        paths.extend(path for path in reports_dir.glob(f"{case_id}*.md") if path.is_file())

    dependencies: list[Path] = []
    for spec_path in paths:
        if _file_type(spec_path) != "yaml":
            continue
        spec = _load_yaml(spec_path)
        for section_name in ("inputs", "outputs", "voxelization"):
            section = _as_mapping(spec.get(section_name))
            for raw_value in section.values():
                if isinstance(raw_value, str) and raw_value:
                    candidate = Path(raw_value)
                    if candidate.exists() and candidate.is_file():
                        dependencies.append(candidate)
                elif isinstance(raw_value, list):
                    for item in raw_value:
                        if isinstance(item, str) and item:
                            candidate = Path(item)
                            if candidate.exists() and candidate.is_file():
                                dependencies.append(candidate)
        for key in ("source_graph", "source_combined_labels", "source_anatomy_labels", "source_flow_1d_model"):
            raw_value = spec.get(key)
            if isinstance(raw_value, str) and raw_value:
                candidate = Path(raw_value)
                if candidate.exists() and candidate.is_file():
                    dependencies.append(candidate)

    unique: dict[str, Path] = {}
    for path in [*paths, *dependencies]:
        unique[str(path)] = path
    return sorted(unique.values(), key=lambda item: str(item))


def _stage_artifacts(
    paths: list[Path],
    stage_root: Path,
    reports_dir: Path,
    output_dir: Path,
    copy_small_artifacts: bool,
    large_threshold_bytes: int,
) -> tuple[ReleaseArtifact, ...]:
    artifacts: list[ReleaseArtifact] = []
    for path in paths:
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        is_large = exists and _is_large_or_volume(path, large_threshold_bytes)
        packaged_path = ""
        copy_policy = "indexed_only_large_or_volume" if is_large else "copied"
        notes: list[str] = []
        if is_large:
            notes.append("not_copied_to_keep_release_disk_light")
        elif copy_small_artifacts and exists:
            destination = _relative_destination(path, stage_root, reports_dir, output_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            packaged_path = str(destination)
        elif not copy_small_artifacts:
            copy_policy = "indexed_only_copy_disabled"

        artifacts.append(
            ReleaseArtifact(
                group=_artifact_group(path),
                role=path.stem,
                file_type=_file_type(path),
                source_path=str(path),
                exists=exists,
                size_bytes=size_bytes,
                sha256=_sha256(path, large_threshold_bytes) if exists and not is_large else "",
                copy_policy=copy_policy,
                packaged_path=packaged_path,
                notes=tuple(notes),
            )
        )
    return tuple(artifacts)


def _metric(
    rows: list[ReleaseMetric],
    *,
    category: str,
    metric: str,
    value: Any,
    threshold: str,
    status: str,
    source_path: str | Path | None,
    notes: tuple[str, ...] = (),
) -> None:
    rows.append(
        ReleaseMetric(
            category=category,
            metric=metric,
            value=str(value),
            threshold=threshold,
            status=status,
            source_path="" if source_path is None else str(source_path),
            notes=notes,
        )
    )


def _count_statuses(metrics: tuple[ReleaseMetric, ...]) -> dict[str, int]:
    return {
        "pass": sum(item.status == "pass" for item in metrics),
        "review": sum(item.status == "review" for item in metrics),
        "fail": sum(item.status == "fail" for item in metrics),
    }


def _status_equal(value: int, expected: int) -> str:
    return "pass" if value == expected else "fail"


def _status_at_most(value: float, maximum: float) -> str:
    return "pass" if value <= maximum else "fail"


def _status_at_least(value: float, minimum: float) -> str:
    return "pass" if value >= minimum else "fail"


def _status_review_if_positive(value: int) -> str:
    return "review" if value > 0 else "pass"


def _status_fail_if_positive(value: int) -> str:
    return "fail" if value > 0 else "pass"


def _row_for(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if str(row.get(key, "")) == value:
            return row
    return {}


def _build_metrics(stage_root: Path) -> tuple[ReleaseMetric, ...]:
    metrics: list[ReleaseMetric] = []
    voxel_spec_path = _find_first(stage_root / "vascular_network_voxelized", "*vascular_network_voxelized_spec_v001.yaml")
    domain_csv_path = _find_first(stage_root / "vascular_network_voxelized", "*vascular_domain_connectivity_summary_v001.csv")
    organ_spec_path = _find_first(stage_root / "validation" / "vessel_organ_anatomy", "*vessel_organ_validation_spec_v001.yaml")
    radius_spec_path = _find_first(stage_root / "validation" / "vessel_radius_anatomy", "*vessel_radius_validation_spec_v001.yaml")
    flow_1d_path = _find_first(stage_root / "flow_1d", "*flow_1d_model_v001.yaml")
    coupled_flow_path = _find_first(stage_root / "flow_coupled_pulsatile", "*coupled_pulsatile_flow_model_v001.yaml")
    rt_qa_path = _find_first(stage_root / "radiotherapy_qa_package", "*radiotherapy_qa_package_spec_v001.yaml")
    rt_plan_path = _find_first(stage_root / "rt_planning_bundle", "*rt_planning_bundle_spec_v001.yaml")
    gamma_path = _find_first(stage_root / "dose_gamma_qa", "*dose_gamma_qa_spec_v001.yaml")

    voxel_spec = _load_yaml(voxel_spec_path)
    voxel = _as_mapping(voxel_spec.get("voxelization"))
    if voxel:
        connected = _safe_int(voxel.get("connected_components"))
        arterial = _safe_int(voxel.get("arterial_components"))
        venous = _safe_int(voxel.get("venous_components"))
        overlap = _safe_int(voxel.get("arterial_venous_overlap_voxels_after_cleanup"))
        outside = _safe_float(voxel.get("outside_body_fraction_before_clip"))
        _metric(
            metrics,
            category="vascular_domain",
            metric="connected_lumen_components",
            value=connected,
            threshold="== 1",
            status=_status_equal(connected, 1),
            source_path=voxel_spec_path,
        )
        _metric(
            metrics,
            category="vascular_domain",
            metric="arterial_components",
            value=arterial,
            threshold="== 1",
            status=_status_equal(arterial, 1),
            source_path=voxel_spec_path,
        )
        _metric(
            metrics,
            category="vascular_domain",
            metric="venous_components",
            value=venous,
            threshold="== 1",
            status=_status_equal(venous, 1),
            source_path=voxel_spec_path,
        )
        _metric(
            metrics,
            category="vascular_domain",
            metric="arterial_venous_overlap_after_cleanup_voxels",
            value=overlap,
            threshold="== 0",
            status=_status_equal(overlap, 0),
            source_path=voxel_spec_path,
        )
        _metric(
            metrics,
            category="vascular_domain",
            metric="outside_body_fraction_before_clip",
            value=f"{outside:.8f}",
            threshold="<= 0.000000",
            status=_status_at_most(outside, 0.0),
            source_path=voxel_spec_path,
        )

    for row in _read_csv_rows(domain_csv_path):
        domain = row.get("domain", "unknown")
        pruned_voxels = _safe_int(row.get("pruned_voxel_count"))
        connectors = _safe_int(row.get("connector_voxel_count"))
        components_after = _safe_int(row.get("components_after"))
        _metric(
            metrics,
            category="domain_repair",
            metric=f"{domain}_components_after_repair",
            value=components_after,
            threshold="== 1",
            status=_status_equal(components_after, 1),
            source_path=domain_csv_path,
            notes=(f"pruned_voxels={pruned_voxels}", f"connector_voxels={connectors}"),
        )

    organ_spec = _load_yaml(organ_spec_path)
    organ_summary = _as_mapping(organ_spec.get("summary"))
    if organ_summary:
        edge_count = _safe_int(organ_summary.get("edge_count"))
        organ_metric_specs = (
            ("pass_count", "record", "pass"),
            ("review_count", "== 0 for clean release; >0 requires review", _status_review_if_positive(_safe_int(organ_summary.get("review_count")))),
            ("fail_count", "== 0", _status_fail_if_positive(_safe_int(organ_summary.get("fail_count")))),
            (
                "bone_intersection_edge_count",
                "== 0 for clean release; >0 requires review",
                _status_review_if_positive(_safe_int(organ_summary.get("bone_intersection_edge_count"))),
            ),
        )
        for name, threshold, status in organ_metric_specs:
            value = _safe_int(organ_summary.get(name))
            notes = (f"edge_count={edge_count}",) if name == "pass_count" else ()
            _metric(
                metrics,
                category="organ_aware_vascular_anatomy",
                metric=name,
                value=value,
                threshold=threshold,
                status=status,
                source_path=organ_spec_path,
                notes=notes,
            )

    radius_spec = _load_yaml(radius_spec_path)
    radius_summary = _as_mapping(radius_spec.get("summary"))
    if radius_summary:
        edge_count = _safe_int(radius_summary.get("edge_count"))
        radius_metric_specs = (
            ("pass_count", "record", "pass"),
            ("review_count", "== 0 for clean release; >0 requires review", _status_review_if_positive(_safe_int(radius_summary.get("review_count")))),
            ("fail_count", "== 0", _status_fail_if_positive(_safe_int(radius_summary.get("fail_count")))),
            (
                "radius_tuning_candidate_count",
                "== 0 for clean release; >0 requires review",
                _status_review_if_positive(_safe_int(radius_summary.get("radius_tuning_candidate_count"))),
            ),
            (
                "reroute_candidate_count",
                "== 0 for clean release; >0 requires review",
                _status_review_if_positive(_safe_int(radius_summary.get("reroute_candidate_count"))),
            ),
        )
        for name, threshold, status in radius_metric_specs:
            value = _safe_int(radius_summary.get(name))
            notes = (f"edge_count={edge_count}",) if name == "pass_count" else ()
            _metric(
                metrics,
                category="radius_aware_vascular_anatomy",
                metric=name,
                value=value,
                threshold=threshold,
                status=status,
                source_path=radius_spec_path,
                notes=notes,
            )

    flow_1d = _load_yaml(flow_1d_path)
    flow_1d_summary = _as_mapping(flow_1d.get("summary"))
    if flow_1d_summary:
        residual = _safe_float(flow_1d_summary.get("max_abs_mass_balance_residual_ml_s"))
        _metric(
            metrics,
            category="steady_1d_flow",
            metric="max_abs_mass_balance_residual_ml_s",
            value=f"{residual:.12g}",
            threshold="<= 1e-4",
            status=_status_at_most(residual, 1e-4),
            source_path=flow_1d_path,
        )
        _metric(
            metrics,
            category="steady_1d_flow",
            metric="arterial_total_flow_ml_s",
            value=f"{_safe_float(flow_1d_summary.get('arterial_total_flow_ml_s')):.6f}",
            threshold="record",
            status="pass",
            source_path=flow_1d_path,
        )

    coupled_flow = _load_yaml(coupled_flow_path)
    coupled_summary = _as_mapping(coupled_flow.get("summary"))
    if coupled_summary:
        residual = _safe_float(coupled_summary.get("max_abs_mass_balance_residual_ml_s"))
        _metric(
            metrics,
            category="coupled_pulsatile_flow",
            metric="max_abs_mass_balance_residual_ml_s",
            value=f"{residual:.12g}",
            threshold="<= 1e-4",
            status=_status_at_most(residual, 1e-4),
            source_path=coupled_flow_path,
        )
        for name in ("arterial_inlet_flow_mean_ml_s", "arterial_inlet_flow_min_ml_s", "arterial_inlet_flow_max_ml_s"):
            _metric(
                metrics,
                category="coupled_pulsatile_flow",
                metric=name,
                value=f"{_safe_float(coupled_summary.get(name)):.6f}",
                threshold="record",
                status="pass",
                source_path=coupled_flow_path,
            )

    rt_qa = _load_yaml(rt_qa_path)
    synthetic_target = _as_mapping(rt_qa.get("synthetic_target"))
    if synthetic_target:
        _metric(
            metrics,
            category="radiotherapy_geometry",
            metric="gtv_volume_cm3",
            value=f"{_safe_float(synthetic_target.get('gtv_volume_cm3')):.6f}",
            threshold="record",
            status="pass",
            source_path=rt_qa_path,
        )
        _metric(
            metrics,
            category="radiotherapy_geometry",
            metric="ptv_volume_cm3",
            value=f"{_safe_float(synthetic_target.get('ptv_volume_cm3')):.6f}",
            threshold="record",
            status="pass",
            source_path=rt_qa_path,
        )

    rt_plan = _load_yaml(rt_plan_path)
    rt_outputs = _as_mapping(rt_plan.get("outputs"))
    dose_metrics = _read_csv_rows(rt_outputs.get("dose_metrics_csv"))
    ptv_static = next(
        (
            row
            for row in dose_metrics
            if row.get("mask_id") == "target_ptv_synthetic_vertebral" and row.get("state") == "static"
        ),
        {},
    )
    if ptv_static:
        d95 = _safe_float(ptv_static.get("d95_gy"))
        _metric(
            metrics,
            category="radiotherapy_planning",
            metric="ptv_static_d95_gy",
            value=f"{d95:.6f}",
            threshold=">= 19.0 Gy for 20 Gy placeholder prescription",
            status=_status_at_least(d95, 19.0),
            source_path=rt_outputs.get("dose_metrics_csv"),
        )

    comparison_rows = _read_csv_rows(rt_outputs.get("dose_comparison_csv"))
    ptv_peak = next(
        (
            row
            for row in comparison_rows
            if row.get("mask_id") == "target_ptv_synthetic_vertebral"
            and row.get("comparison_state") == "pulsatile_peak"
        ),
        {},
    )
    if ptv_peak:
        delta_d95 = abs(_safe_float(ptv_peak.get("delta_d95_gy")))
        _metric(
            metrics,
            category="radiotherapy_planning",
            metric="ptv_peak_abs_delta_d95_gy",
            value=f"{delta_d95:.6f}",
            threshold="<= 0.05 Gy engineering delta",
            status=_status_at_most(delta_d95, 0.05),
            source_path=rt_outputs.get("dose_comparison_csv"),
        )

    gamma = _load_yaml(gamma_path)
    state_results = [item for item in _as_list(gamma.get("state_results")) if isinstance(item, dict)]
    if state_results:
        min_pass = min(_safe_float(item.get("pass_rate_percent")) for item in state_results)
        max_p95 = max(_safe_float(item.get("p95_gamma")) for item in state_results)
        _metric(
            metrics,
            category="dose_gamma_qa",
            metric="minimum_gamma_pass_rate_percent",
            value=f"{min_pass:.6f}",
            threshold=">= 95%",
            status=_status_at_least(min_pass, 95.0),
            source_path=gamma_path,
        )
        _metric(
            metrics,
            category="dose_gamma_qa",
            metric="maximum_p95_gamma",
            value=f"{max_p95:.6f}",
            threshold="<= 1.0",
            status=_status_at_most(max_p95, 1.0),
            source_path=gamma_path,
        )

    return tuple(metrics)


def _summary_from_metrics(
    artifacts: tuple[ReleaseArtifact, ...], metrics: tuple[ReleaseMetric, ...]
) -> dict[str, Any]:
    status_counts = _count_statuses(metrics)
    copied_count = sum(1 for item in artifacts if item.packaged_path)
    large_count = sum(item.copy_policy == "indexed_only_large_or_volume" for item in artifacts)
    group_counts: dict[str, int] = {}
    for item in artifacts:
        group_counts[item.group] = group_counts.get(item.group, 0) + 1
    return {
        "artifact_count": len(artifacts),
        "copied_artifact_count": copied_count,
        "indexed_large_or_volume_artifact_count": large_count,
        "artifact_group_counts": group_counts,
        "qa_metric_count": len(metrics),
        "qa_status_counts": status_counts,
    }


def _readiness_status(metrics: tuple[ReleaseMetric, ...]) -> str:
    counts = _count_statuses(metrics)
    if counts["fail"]:
        return "blocked_by_release_qa_failure"
    if counts["review"]:
        return "research_release_candidate_review_required"
    return "research_release_candidate"


def _write_artifact_index(path: Path, artifacts: tuple[ReleaseArtifact, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "role",
        "file_type",
        "source_path",
        "exists",
        "size_bytes",
        "sha256",
        "copy_policy",
        "packaged_path",
        "notes",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(
                {
                    "group": artifact.group,
                    "role": artifact.role,
                    "file_type": artifact.file_type,
                    "source_path": artifact.source_path,
                    "exists": artifact.exists,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                    "copy_policy": artifact.copy_policy,
                    "packaged_path": artifact.packaged_path,
                    "notes": ";".join(artifact.notes),
                }
            )


def _write_qa_summary(path: Path, metrics: tuple[ReleaseMetric, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "metric", "value", "threshold", "status", "source_path", "notes"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "category": metric.category,
                    "metric": metric.metric,
                    "value": metric.value,
                    "threshold": metric.threshold,
                    "status": metric.status,
                    "source_path": metric.source_path,
                    "notes": ";".join(metric.notes),
                }
            )


def _write_manifest(path: Path, result: ResearchReleasePackageResult, stage_root: Path, reports_dir: Path) -> None:
    payload = {
        "case_id": result.case_id,
        "release_id": result.release_id,
        "package_type": "digital_phantom_research_release_candidate",
        "readiness_status": result.readiness_status,
        "stage_root": str(stage_root),
        "reports_dir": str(reports_dir),
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "artifact_index_csv": result.artifact_index_csv_path,
            "qa_summary_csv": result.qa_summary_csv_path,
            "command_log": result.command_log_path,
            "limitations_markdown": result.limitations_markdown_path,
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
        },
        "summary": result.summary,
        "key_artifacts": {
            artifact.role: artifact.source_path
            for artifact in result.artifacts
            if artifact.file_type in {"yaml", "csv", "png", "markdown"}
            and artifact.group
            in {
                "vascular_voxelization",
                "vessel_organ_validation",
                "vessel_radius_validation",
                "coupled_pulsatile_flow",
                "rt_planning",
                "dose_gamma_qa",
                "reports",
            }
        },
        "large_or_volume_artifacts": [
            {
                "path": artifact.source_path,
                "size_bytes": artifact.size_bytes,
                "file_type": artifact.file_type,
                "notes": list(artifact.notes),
            }
            for artifact in result.artifacts
            if artifact.copy_policy == "indexed_only_large_or_volume"
        ],
        "qa_metrics": [
            {
                "category": metric.category,
                "metric": metric.metric,
                "value": metric.value,
                "threshold": metric.threshold,
                "status": metric.status,
                "source_path": metric.source_path,
                "notes": list(metric.notes),
            }
            for metric in result.metrics
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_command_log(
    path: Path,
    case_id: str,
    stage_root: Path,
    release_output_dir: Path,
    report_path: Path,
) -> None:
    voxel_spec = stage_root / "vascular_network_voxelized" / f"{case_id}_vascular_network_voxelized_spec_v001.yaml"
    graph = "outputs/digital/patient_builds/mode03_neg_patient_build_stage006_radius_tuned/vascular_graph_radius_tuned/mode03_neg_patient_build_stage006_radius_tuned_radius_tuned_vascular_graph_v001.yaml"
    anatomy = "outputs/digital/patient_builds/mode03_neg_patient_build_stage003_bonefix/torso/mode03_neg_patient_build_stage003_bonefix_torso_torso_material_labels_v001.nii.gz"
    flow_boundary = stage_root / "flow_boundary_conditions" / f"{case_id}_flow_boundary_conditions_v001.yaml"
    flow_1d = stage_root / "flow_1d" / f"{case_id}_flow_1d_model_v001.yaml"
    rt_package = stage_root / "radiotherapy_qa_package" / f"{case_id}_radiotherapy_qa_package_spec_v001.yaml"
    rt_planning = stage_root / "rt_planning_bundle" / f"{case_id}_rt_planning_bundle_spec_v001.yaml"
    gamma_config = stage_root / "rt_planning_bundle" / f"{case_id}_pymedphys_dose_eval_config_v001.yaml"
    prefix = "TMPDIR=.tmp PYTHONPATH=.deps/python:src python -m phantom_twin.cli"
    lines = [
        "# Research Release Candidate Reproducibility Commands",
        "",
        "These commands document the stage007 release path. They are intended for local reproduction of this research-engineering bundle, not clinical execution.",
        "",
        "```bash",
        f"{prefix} validate-vessel-organ-anatomy --voxelized-spec {voxel_spec} --graph {graph} --anatomy-labels {anatomy} --case-id {case_id} --output-dir {stage_root / 'validation' / 'vessel_organ_anatomy'} --report outputs/reports/{case_id}_vessel_organ.md",
        f"{prefix} validate-vessel-radius-anatomy --voxelized-spec {voxel_spec} --graph {graph} --anatomy-labels {anatomy} --case-id {case_id} --output-dir {stage_root / 'validation' / 'vessel_radius_anatomy'} --report outputs/reports/{case_id}_vessel_radius.md",
        f"{prefix} build-flow-boundary-package --voxelized-spec {voxel_spec} --graph {graph} --case-id {case_id} --output-dir {stage_root / 'flow_boundary_conditions'} --report outputs/reports/{case_id}_flow_boundary.md",
        f"{prefix} build-flow-1d-model --graph {graph} --boundary-config {flow_boundary} --case-id {case_id} --output-dir {stage_root / 'flow_1d'} --report outputs/reports/{case_id}_flow_1d.md",
        f"{prefix} build-coupled-pulsatile-flow-model --flow-1d-model {flow_1d} --boundary-config {flow_boundary} --case-id {case_id} --output-dir {stage_root / 'flow_coupled_pulsatile'} --report outputs/reports/{case_id}_flow_coupled.md",
        f"{prefix} build-radiotherapy-qa-package --combined-spec {voxel_spec} --case-id {case_id} --output-dir {stage_root / 'radiotherapy_qa_package'} --scenario blood --report outputs/reports/{case_id}_rt_qa.md",
        f"{prefix} build-rt-planning-bundle --rt-package-spec {rt_package} --coupled-flow-model {stage_root / 'flow_coupled_pulsatile' / f'{case_id}_coupled_pulsatile_flow_model_v001.yaml'} --case-id {case_id} --output-dir {stage_root / 'rt_planning_bundle'} --skip-dicom --report outputs/reports/{case_id}_rt_planning.md",
        f"{prefix} build-dose-gamma-qa --pymedphys-eval-config {gamma_config} --case-id {case_id} --output-dir {stage_root / 'dose_gamma_qa'} --skip-volume-outputs --report outputs/reports/{case_id}_dose_gamma_qa.md",
        f"{prefix} build-research-release-package --case-id {case_id} --stage-root {stage_root} --output-dir {release_output_dir} --report {report_path}",
        "```",
        "",
        "Exact upstream graph, anatomy, and patient-adapter inputs are indexed in the release manifest and artifact index.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_limitations(path: Path) -> None:
    lines = [
        "# Release Candidate Limitations",
        "",
        "- This is a research-engineering digital phantom release candidate, not a clinical device, treatment-planning result, or patient-specific twin.",
        "- The torso anatomy is based on staged/anonymized public or adapted segmentation inputs and simplified material labels, not a fully validated whole-human anatomical equivalence model.",
        "- Several vascular branches remain template-derived or registered from labelled teaching/template data; they are more realistic than placeholders, but not guaranteed patient-specific.",
        "- Blood flow is a graph-coupled 1D pulsatile model with rigid circular tubes, Poiseuille-style resistance, and terminal RCR outlets; it does not yet include 3D CFD, wall motion, turbulence, autoregulation, or lymphatics.",
        "- RT dose volumes are synthetic engineering test patterns with PyMedPhys QA, not TPS, Monte Carlo, or measured dose calculations.",
        "- Material density, RED, and HU mappings are simplified and should be calibrated before any device, imaging, or dosimetry claims.",
        "- QA thresholds are project engineering gates for reproducibility and plausibility, not regulatory or clinical acceptance criteria.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _figure_path(artifacts: tuple[ReleaseArtifact, ...], group: str, contains: str = "") -> str:
    for artifact in artifacts:
        if artifact.group != group or artifact.file_type != "png":
            continue
        if contains and contains not in Path(artifact.source_path).name:
            continue
        return artifact.source_path
    return ""


def _write_atlas(path: Path, result: ResearchReleasePackageResult) -> None:
    plt = _import_plotting()
    figure_specs = [
        ("Vascular Domain", _figure_path(result.artifacts, "vascular_voxelization")),
        ("Organ-Aware Vessel QA", _figure_path(result.artifacts, "vessel_organ_validation")),
        ("Radius-Aware Vessel QA", _figure_path(result.artifacts, "vessel_radius_validation")),
        ("Pulsatile Flow", _figure_path(result.artifacts, "coupled_pulsatile_flow", "pressure_flow")),
        ("RT QA Geometry", _figure_path(result.artifacts, "radiotherapy_qa")),
        ("RT Planning", _figure_path(result.artifacts, "rt_planning")),
        ("Dose Gamma QA", _figure_path(result.artifacts, "dose_gamma_qa")),
    ]
    fig = plt.figure(figsize=(18, 13))
    grid = fig.add_gridspec(3, 3)
    axes = [fig.add_subplot(grid[row, col]) for row in range(3) for col in range(3)]
    fig.suptitle(f"Digital Phantom Release Candidate\n{result.release_id}", fontsize=18, fontweight="bold")
    for ax, (title, image_path) in zip(axes[:7], figure_specs, strict=False):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        if image_path and Path(image_path).exists():
            ax.imshow(plt.imread(image_path))
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", color="#7f8c8d")
    summary_ax = axes[7]
    summary_ax.axis("off")
    counts = result.summary.get("qa_status_counts", {})
    text = "\n".join(
        [
            "Release Snapshot",
            "",
            f"Status: {result.readiness_status}",
            f"QA metrics: {result.summary.get('qa_metric_count', 0)}",
            f"Pass/review/fail: {counts.get('pass', 0)} / {counts.get('review', 0)} / {counts.get('fail', 0)}",
            f"Artifacts indexed: {result.summary.get('artifact_count', 0)}",
            f"Small artifacts copied: {result.summary.get('copied_artifact_count', 0)}",
            f"Large/volume artifacts indexed only: {result.summary.get('indexed_large_or_volume_artifact_count', 0)}",
            "",
            "Scope",
            "CT/material torso",
            "arterial + venous vascular domains",
            "coupled pulsatile 1D flow",
            "RT-style dose metrics + gamma QA",
        ]
    )
    summary_ax.text(
        0.02,
        0.98,
        text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        bbox={"facecolor": "#f7f9fb", "edgecolor": "#ccd6dd", "boxstyle": "round,pad=0.6"},
    )
    axes[8].axis("off")
    axes[8].text(
        0.02,
        0.98,
        "Not clinical use\n\nThis bundle captures a reproducible engineering state. It is intentionally disk-light: volumes are indexed, while reports, tables, YAML, and figures are copied.",
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "#fff8e7", "edgecolor": "#dfb95b", "boxstyle": "round,pad=0.6"},
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_report(result: ResearchReleasePackageResult) -> str:
    counts = result.summary.get("qa_status_counts", {})
    atlas_rel = os.path.relpath(result.atlas_png_path, start=Path(result.report_path).parent)
    lines = [
        "# Digital Phantom Research Release Candidate",
        "",
        f"Case ID: `{result.case_id}`",
        f"Release ID: `{result.release_id}`",
        f"Readiness status: `{result.readiness_status}`",
        "",
        f"![Release atlas]({atlas_rel})",
        "",
        "## Package Summary",
        "",
        f"- QA pass / review / fail: {counts.get('pass', 0)} / {counts.get('review', 0)} / {counts.get('fail', 0)}",
        f"- Artifacts indexed: {result.summary.get('artifact_count', 0)}",
        f"- Small artifacts copied into package: {result.summary.get('copied_artifact_count', 0)}",
        f"- Large or volume artifacts indexed only: {result.summary.get('indexed_large_or_volume_artifact_count', 0)}",
        "",
        "## Outputs",
        "",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Artifact index: `{result.artifact_index_csv_path}`",
        f"- QA summary: `{result.qa_summary_csv_path}`",
        f"- Reproducibility commands: `{result.command_log_path}`",
        f"- Limitations: `{result.limitations_markdown_path}`",
        f"- Atlas PNG: `{result.atlas_png_path}`",
        "",
        "## Interpretation",
        "",
        "- This bundle preserves the current best digital phantom state for reproducible research review.",
        "- The package is deliberately disk-light: NIfTI volumes remain in place and are indexed with paths/sizes instead of duplicated.",
        "- The current scope includes anatomy/material labels, repaired arterial/venous domains, organ/radius vascular QA, coupled pulsatile 1D flow, RT-style planning metrics, and PyMedPhys gamma QA.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_research_release_package_result(result: ResearchReleasePackageResult) -> str:
    return _format_report(result)


def build_research_release_package(
    stage_root: str | Path = "outputs/digital/patient_builds/mode03_neg_patient_build_stage007_domain_repaired",
    reports_dir: str | Path = "outputs/reports",
    output_dir: str | Path = "outputs/releases/mode03_neg_stage007_rc1",
    case_id: str = "mode03_neg_patient_build_stage007_domain_repaired",
    release_id: str | None = None,
    report_path: str | Path | None = "outputs/reports/mode03_neg_stage007_research_release_candidate.md",
    copy_small_artifacts: bool = True,
    large_file_threshold_mb: float = 25.0,
) -> ResearchReleasePackageResult:
    stage = Path(stage_root)
    reports = Path(reports_dir)
    output = Path(output_dir)
    release = release_id or f"{case_id}_rc1"
    manifest = output / f"{release}_release_manifest_v001.yaml"
    artifact_index = output / f"{release}_artifact_index_v001.csv"
    qa_summary = output / f"{release}_qa_summary_v001.csv"
    command_log = output / f"{release}_reproducibility_commands_v001.md"
    limitations = output / f"{release}_limitations_v001.md"
    atlas_png = output / f"{release}_release_atlas_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{release}_release_report_v001.md"
    notes = (
        "research_release_candidate_not_for_clinical_use",
        "disk_light_package_copies_small_artifacts_and_indexes_large_volumes",
        "qa_thresholds_are_engineering_plausibility_gates",
    )

    if not stage.exists():
        raise FileNotFoundError(f"Stage root does not exist: {stage}")

    output.mkdir(parents=True, exist_ok=True)
    discovered = _discover_artifact_paths(stage, reports, case_id)
    artifacts = _stage_artifacts(
        discovered,
        stage_root=stage,
        reports_dir=reports,
        output_dir=output,
        copy_small_artifacts=copy_small_artifacts,
        large_threshold_bytes=int(large_file_threshold_mb * 1024 * 1024),
    )
    metrics = _build_metrics(stage)
    summary = _summary_from_metrics(artifacts, metrics)
    readiness = _readiness_status(metrics)

    result = ResearchReleasePackageResult(
        case_id=case_id,
        release_id=release,
        output_dir=str(output),
        manifest_yaml_path=str(manifest),
        artifact_index_csv_path=str(artifact_index),
        qa_summary_csv_path=str(qa_summary),
        command_log_path=str(command_log),
        limitations_markdown_path=str(limitations),
        atlas_png_path=str(atlas_png),
        report_path=str(report),
        readiness_status=readiness,
        summary=summary,
        artifacts=artifacts,
        metrics=metrics,
        notes=notes,
    )

    _write_artifact_index(artifact_index, artifacts)
    _write_qa_summary(qa_summary, metrics)
    _write_command_log(command_log, case_id, stage, output, report)
    _write_limitations(limitations)
    _write_manifest(manifest, result, stage, reports)
    _write_atlas(atlas_png, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
