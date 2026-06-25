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
class CorrectedBranchReleaseArtifact:
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
class CorrectedBranchReleaseResult:
    case_id: str
    release_id: str
    output_dir: str
    manifest_yaml_path: str
    artifact_index_csv_path: str
    readme_markdown_path: str
    limitations_markdown_path: str
    command_log_path: str
    report_path: str
    summary: dict[str, Any]
    artifacts: tuple[CorrectedBranchReleaseArtifact, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text())
    return data if isinstance(data, dict) else {}


def _file_type(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".nii.gz"):
        return "nifti_volume"
    if lower.endswith((".yaml", ".yml")):
        return "yaml"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".png"):
        return "png"
    if lower.endswith(".gif"):
        return "gif"
    if lower.endswith(".md"):
        return "markdown"
    if lower.endswith((".stl", ".ply", ".obj")):
        return "mesh"
    return path.suffix.lower().lstrip(".") or "unknown"


def _artifact_group(path: Path) -> str:
    parts = set(path.parts)
    if "label_vessel_flow_domain" in parts and "digital" in parts:
        return "corrected_vascular_domain"
    if "flow_boundary_conditions" in parts:
        return "flow_boundary_conditions"
    if "flow_1d" in parts:
        return "steady_1d_flow"
    if "flow_coupled_pulsatile" in parts:
        return "coupled_pulsatile_flow"
    if "flow_4d_visualization" in parts:
        return "flow_4d_visualization"
    if "qa_package" in parts and "radiotherapy" in parts:
        return "radiotherapy_qa"
    if "planning_bundle" in parts:
        return "rt_planning"
    if "spatial_rt_flow_coupling" in parts:
        return "spatial_rt_flow_coupling"
    if "dose_gamma_qa" in parts:
        return "dose_gamma_qa"
    if "spatial_rt_flow_dose" in parts:
        return "spatial_rt_flow_dose"
    if "corrected_branch_status" in parts:
        return "status_package"
    if "reports" in parts:
        return "reports"
    return "dependency"


def _is_large_or_volume(path: Path, large_threshold_bytes: int) -> bool:
    lower = path.name.lower()
    return lower.endswith(VOLUME_SUFFIXES) or path.stat().st_size > large_threshold_bytes


def _sha256(path: Path, max_size_bytes: int) -> str:
    if not path.exists() or path.stat().st_size > max_size_bytes:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_relative_path(path: Path) -> Path:
    cwd = Path.cwd()
    try:
        return path.resolve().relative_to(cwd.resolve())
    except ValueError:
        return Path(path.name)


def _packaged_destination(path: Path, output_dir: Path) -> Path:
    return output_dir / "artifacts" / _source_relative_path(path)


def _looks_like_path(value: str) -> bool:
    if not value or "\n" in value:
        return False
    if value.startswith(("http://", "https://", "s3://", "doi:")):
        return False
    lower = value.lower()
    return any(
        lower.endswith(suffix)
        for suffix in (
            ".yaml",
            ".yml",
            ".csv",
            ".png",
            ".gif",
            ".md",
            ".nii.gz",
            ".nii",
            ".stl",
            ".ply",
            ".obj",
        )
    )


def _resolve_candidate(raw_value: str, base_path: Path | None = None) -> Path | None:
    value = raw_value.strip()
    if not _looks_like_path(value):
        return None
    candidate = Path(value)
    if candidate.exists():
        return candidate
    if base_path is not None:
        relative = base_path.parent / candidate
        if relative.exists():
            return relative
    return None


def _walk_path_values(value: Any, base_path: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, str):
        candidate = _resolve_candidate(value, base_path)
        if candidate is not None:
            paths.append(candidate)
    elif isinstance(value, dict):
        for nested in value.values():
            paths.extend(_walk_path_values(nested, base_path))
    elif isinstance(value, list):
        for nested in value:
            paths.extend(_walk_path_values(nested, base_path))
    return paths


def _initial_artifacts_from_status(status_manifest: dict[str, Any], status_manifest_path: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {"status_manifest": status_manifest_path}
    outputs = status_manifest.get("outputs", {})
    if isinstance(outputs, dict):
        for role, raw_path in outputs.items():
            if isinstance(raw_path, str):
                candidate = _resolve_candidate(raw_path, status_manifest_path)
                if candidate is not None:
                    artifacts[str(role)] = candidate
    status_artifacts = status_manifest.get("artifacts", {})
    if isinstance(status_artifacts, dict):
        for role, payload in status_artifacts.items():
            if not isinstance(payload, dict):
                continue
            raw_path = payload.get("path")
            if isinstance(raw_path, str):
                candidate = _resolve_candidate(raw_path, status_manifest_path)
                if candidate is not None:
                    artifacts[str(role)] = candidate
    return artifacts


def _discover_referenced_artifacts(seed_artifacts: dict[str, Path]) -> dict[str, Path]:
    artifacts = dict(seed_artifacts)
    visited_yaml: set[str] = set()
    changed = True
    while changed:
        changed = False
        for path in list(artifacts.values()):
            if _file_type(path) != "yaml" or str(path) in visited_yaml or not path.exists():
                continue
            visited_yaml.add(str(path))
            spec = _load_yaml(path)
            for candidate in _walk_path_values(spec, path):
                role = candidate.stem
                key = str(candidate)
                if key not in {str(existing) for existing in artifacts.values()}:
                    artifacts[role] = candidate
                    changed = True
    return artifacts


def _stage_artifacts(
    artifacts_by_role: dict[str, Path],
    output_dir: Path,
    copy_small_artifacts: bool,
    large_threshold_bytes: int,
) -> tuple[CorrectedBranchReleaseArtifact, ...]:
    staged: list[CorrectedBranchReleaseArtifact] = []
    seen: set[str] = set()
    for role, path in sorted(artifacts_by_role.items(), key=lambda item: str(item[1])):
        if str(path) in seen:
            continue
        seen.add(str(path))
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        file_type = _file_type(path)
        is_large = exists and _is_large_or_volume(path, large_threshold_bytes)
        copy_policy = "indexed_only_large_or_volume" if is_large else "copied"
        packaged_path = ""
        notes: list[str] = []
        if not exists:
            copy_policy = "missing"
            notes.append("source_path_missing")
        elif is_large:
            notes.append("not_copied_to_keep_release_disk_light")
        elif copy_small_artifacts:
            destination = _packaged_destination(path, output_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            packaged_path = str(destination)
        else:
            copy_policy = "indexed_only_copy_disabled"
        staged.append(
            CorrectedBranchReleaseArtifact(
                group=_artifact_group(path),
                role=role,
                file_type=file_type,
                source_path=str(path),
                exists=exists,
                size_bytes=size_bytes,
                sha256=_sha256(path, large_threshold_bytes) if exists and not is_large else "",
                copy_policy=copy_policy,
                packaged_path=packaged_path,
                notes=tuple(notes),
            )
        )
    return tuple(staged)


def _write_artifact_index(path: Path, artifacts: tuple[CorrectedBranchReleaseArtifact, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
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
    )
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


def _summary(status_manifest: dict[str, Any], artifacts: tuple[CorrectedBranchReleaseArtifact, ...]) -> dict[str, Any]:
    copied = [artifact for artifact in artifacts if artifact.copy_policy == "copied"]
    indexed_large = [artifact for artifact in artifacts if artifact.copy_policy == "indexed_only_large_or_volume"]
    missing = [artifact for artifact in artifacts if not artifact.exists]
    return {
        "artifact_count": len(artifacts),
        "copied_artifact_count": len(copied),
        "indexed_large_or_volume_artifact_count": len(indexed_large),
        "missing_artifact_count": len(missing),
        "source_total_size_bytes": sum(artifact.size_bytes for artifact in artifacts),
        "copied_total_size_bytes": sum(artifact.size_bytes for artifact in copied),
        "status_summary": status_manifest.get("summary", {}),
    }


def _write_manifest(path: Path, result: CorrectedBranchReleaseResult, status_manifest_path: Path) -> None:
    payload = {
        "case_id": result.case_id,
        "release_id": result.release_id,
        "package_type": "corrected_branch_labelled_phantom_disk_light_release",
        "source_status_manifest": str(status_manifest_path),
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "artifact_index_csv": result.artifact_index_csv_path,
            "readme_markdown": result.readme_markdown_path,
            "limitations_markdown": result.limitations_markdown_path,
            "command_log": result.command_log_path,
            "report": result.report_path,
        },
        "summary": result.summary,
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
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_command_log(path: Path, case_id: str, status_manifest_path: Path, release_output_dir: Path, report_path: Path) -> None:
    prefix = "TMPDIR=.tmp PYTHONPATH=.deps/python:src python -m phantom_twin.cli"
    lines = [
        "# Corrected Branch-Labelled Release Commands",
        "",
        "These commands document the disk-light release packaging step. Upstream pipeline commands are indexed in the status report artifact links.",
        "",
        "```bash",
        f"{prefix} build-corrected-branch-status-report --case-id {case_id} --output-dir outputs/reports/corrected_branch_status --report outputs/reports/{case_id}_status_report.md",
        f"{prefix} build-corrected-branch-release-package --status-manifest {status_manifest_path} --case-id {case_id} --output-dir {release_output_dir} --report {report_path}",
        "```",
        "",
        "Large NIfTI and dose volumes are intentionally indexed only. Rebuild them from the upstream specs before clinical-style review.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_limitations(path: Path) -> None:
    lines = [
        "# Corrected Branch-Labelled Release Limitations",
        "",
        "- This is a research-engineering release bundle, not a clinical medical device package.",
        "- Corrected vessel masks are branch-labelled template vessels staged onto the CT grid; they are not yet patient-specific CTA/CTV deformable registrations.",
        "- Flow outputs use a 1D graph/RCR surrogate with placeholder venous return dynamics, not calibrated physiology or 3D CFD.",
        "- RT outputs are synthetic surrogate dose states for pipeline validation, not TPS-commissioned or Monte Carlo dose calculations.",
        "- Large NIfTI volumes are not copied into the release bundle; they are indexed by source path to keep disk use controlled.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GiB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    return f"{size_bytes} B"


def _format_report(result: CorrectedBranchReleaseResult) -> str:
    status = result.summary.get("status_summary", {}) if isinstance(result.summary.get("status_summary"), dict) else {}
    flow = status.get("flow", {}) if isinstance(status.get("flow"), dict) else {}
    rt = status.get("rt_flow", {}) if isinstance(status.get("rt_flow"), dict) else {}
    gamma = status.get("gamma", {}) if isinstance(status.get("gamma"), dict) else {}
    lines = [
        "# Corrected Branch-Labelled Disk-Light Release Bundle",
        "",
        f"Case ID: `{result.case_id}`",
        f"Release ID: `{result.release_id}`",
        "",
        "## Bundle Summary",
        "",
        f"- Artifacts indexed: {result.summary.get('artifact_count', 0)}",
        f"- Small artifacts copied: {result.summary.get('copied_artifact_count', 0)}",
        f"- Large/volume artifacts indexed only: {result.summary.get('indexed_large_or_volume_artifact_count', 0)}",
        f"- Missing artifacts: {result.summary.get('missing_artifact_count', 0)}",
        f"- Source footprint indexed: {_format_size(int(result.summary.get('source_total_size_bytes', 0)))}",
        f"- Bundle copied footprint: {_format_size(int(result.summary.get('copied_total_size_bytes', 0)))}",
        "",
        "## Headline QA",
        "",
        f"- Aorta flow mean/min/max: {flow.get('aorta_flow_mean_ml_s', 0.0):.3f} / {flow.get('aorta_flow_min_ml_s', 0.0):.3f} / {flow.get('aorta_flow_max_ml_s', 0.0):.3f} mL/s",
        f"- Selected spatial RT-flow edges: {rt.get('selected_edge_count', 0)}",
        f"- Max spatial peak/trough delta: {rt.get('max_peak_delta_mgy', 0.0):.3f} / {rt.get('max_trough_delta_mgy', 0.0):.3f} mGy",
        f"- Gamma min pass rate: {gamma.get('min_pass_rate_percent', 0.0):.3f}%",
        "",
        "## Outputs",
        "",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Artifact index: `{result.artifact_index_csv_path}`",
        f"- README: `{result.readme_markdown_path}`",
        f"- Limitations: `{result.limitations_markdown_path}`",
        f"- Command log: `{result.command_log_path}`",
        "",
        "## Notes",
    ]
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _write_readme(path: Path, result: CorrectedBranchReleaseResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_report(result))


def build_corrected_branch_release_package(
    status_manifest_path: str | Path = "outputs/reports/corrected_branch_status/mode03_neg_branch_ctgrid_corrected_flow_corrected_branch_status_manifest_v001.yaml",
    output_dir: str | Path = "outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1",
    case_id: str = "mode03_neg_branch_ctgrid_corrected_flow",
    release_id: str | None = None,
    copy_small_artifacts: bool = True,
    large_file_threshold_mb: float = 5.0,
    report_path: str | Path | None = "outputs/reports/mode03_neg_branch_ctgrid_corrected_release_bundle.md",
) -> CorrectedBranchReleaseResult:
    status_path = Path(status_manifest_path)
    if not status_path.exists():
        raise FileNotFoundError(f"Status manifest does not exist: {status_path}")
    status_manifest = _load_yaml(status_path)
    output = Path(output_dir)
    release = release_id or f"{case_id}_rc1"
    manifest = output / f"{release}_release_manifest_v001.yaml"
    artifact_index = output / f"{release}_artifact_index_v001.csv"
    readme = output / "README.md"
    limitations = output / f"{release}_limitations_v001.md"
    command_log = output / f"{release}_reproducibility_commands_v001.md"
    report = Path(report_path) if report_path is not None else output / f"{release}_release_report_v001.md"
    output.mkdir(parents=True, exist_ok=True)

    seed_artifacts = _initial_artifacts_from_status(status_manifest, status_path)
    discovered = _discover_referenced_artifacts(seed_artifacts)
    artifacts = _stage_artifacts(
        discovered,
        output_dir=output,
        copy_small_artifacts=copy_small_artifacts,
        large_threshold_bytes=int(large_file_threshold_mb * 1024 * 1024),
    )
    summary = _summary(status_manifest, artifacts)
    notes = (
        "disk_light_release_copies_small_specs_reports_figures_and_indexes_large_volumes",
        "source_of_truth_is_corrected_branch_status_manifest",
        "not_for_clinical_use",
    )
    result = CorrectedBranchReleaseResult(
        case_id=case_id,
        release_id=release,
        output_dir=str(output),
        manifest_yaml_path=str(manifest),
        artifact_index_csv_path=str(artifact_index),
        readme_markdown_path=str(readme),
        limitations_markdown_path=str(limitations),
        command_log_path=str(command_log),
        report_path=str(report),
        summary=summary,
        artifacts=artifacts,
        notes=notes,
    )
    _write_artifact_index(artifact_index, artifacts)
    _write_limitations(limitations)
    _write_command_log(command_log, case_id, status_path, output, report)
    _write_manifest(manifest, result, status_path)
    _write_readme(readme, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_corrected_branch_release_result(result: CorrectedBranchReleaseResult) -> str:
    return "\n".join(
        [
            "Corrected branch-labelled disk-light release package created",
            f"Case ID: {result.case_id}",
            f"Release ID: {result.release_id}",
            f"Artifacts indexed: {result.summary.get('artifact_count', 0)}",
            f"Small artifacts copied: {result.summary.get('copied_artifact_count', 0)}",
            f"Large/volume indexed only: {result.summary.get('indexed_large_or_volume_artifact_count', 0)}",
            f"Manifest YAML: {result.manifest_yaml_path}",
            f"Artifact index CSV: {result.artifact_index_csv_path}",
            f"Report: {result.report_path}",
        ]
    )
