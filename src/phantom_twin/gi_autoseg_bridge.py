from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

import yaml

from .gi_segmentation_staging import (
    GISegmentationStagingResult,
    _directory_files_by_target,
    _load_yaml,
    _nifti_candidates,
    stage_gi_segmentation,
)


SUPPORTED_SEGMENTERS: dict[str, tuple[str, ...]] = {
    "totalsegmentator": ("TotalSegmentator", "totalsegmentator"),
}


@dataclass(frozen=True)
class GIAutoSegBridgeResult:
    case_id: str
    ct_path: str
    output_dir: str
    segmenter: str
    segmenter_mode: str
    readiness_status: str
    segmenter_output_dir: str
    command: tuple[str, ...]
    command_returncode: int | None
    command_stdout_path: str | None
    command_stderr_path: str | None
    staging_manifest_path: str | None
    normalized_gi_segmentation_path: str | None
    metrics_csv_path: str | None
    preview_png_path: str | None
    manifest_yaml_path: str
    report_path: str
    detected_targets: tuple[str, ...]
    missing_targets: tuple[str, ...]
    notes: tuple[str, ...]


def _which_segmenter(segmenter: str, executable_override: str | Path | None = None) -> str | None:
    if executable_override is not None and str(executable_override) != "":
        executable = Path(executable_override)
        if executable.exists():
            return str(executable)
        resolved = shutil.which(str(executable_override))
        return resolved
    for candidate in SUPPORTED_SEGMENTERS.get(segmenter, ()):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _detect_supported_targets(output_dir: Path, labelmap_path: str | Path) -> tuple[str, ...]:
    if not output_dir.exists() or not output_dir.is_dir():
        return ()
    labelmap = _load_yaml(labelmap_path)
    targets: list[str] = []
    for target, names in _directory_files_by_target(labelmap).items():
        if _nifti_candidates(output_dir, names):
            targets.append(target)
    return tuple(targets)


def _write_manifest(path: Path, result: GIAutoSegBridgeResult) -> None:
    payload: dict[str, Any] = {
        "case_id": result.case_id,
        "package_type": "gi_auto_segmentation_bridge",
        "readiness_status": result.readiness_status,
        "segmenter": result.segmenter,
        "segmenter_mode": result.segmenter_mode,
        "inputs": {
            "ct": result.ct_path,
        },
        "segmenter_execution": {
            "output_dir": result.segmenter_output_dir,
            "command": list(result.command),
            "returncode": result.command_returncode,
            "stdout_path": result.command_stdout_path,
            "stderr_path": result.command_stderr_path,
        },
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "report": result.report_path,
            "staging_manifest": result.staging_manifest_path,
            "normalized_gi_segmentation": result.normalized_gi_segmentation_path,
            "metrics_csv": result.metrics_csv_path,
            "preview_png": result.preview_png_path,
        },
        "detected_targets": list(result.detected_targets),
        "missing_targets": list(result.missing_targets),
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: GIAutoSegBridgeResult) -> None:
    lines = [
        "# GI Auto-Segmentation Bridge",
        "",
        f"Case ID: `{result.case_id}`",
        f"Readiness status: `{result.readiness_status}`",
        f"Segmenter: `{result.segmenter}`",
        f"Mode: `{result.segmenter_mode}`",
        "",
        "## Target Detection",
        "",
        f"- Detected targets: `{', '.join(result.detected_targets) or 'none'}`",
        f"- Missing targets: `{', '.join(result.missing_targets) or 'none'}`",
        "",
        "## Segmenter Execution",
        "",
        f"- Output directory: `{result.segmenter_output_dir}`",
        f"- Return code: `{result.command_returncode if result.command_returncode is not None else 'not_run'}`",
        f"- Stdout: `{result.command_stdout_path or 'not_written'}`",
        f"- Stderr: `{result.command_stderr_path or 'not_written'}`",
        "",
        "## Product Handoff",
        "",
    ]
    if result.normalized_gi_segmentation_path:
        lines.extend(
            [
                "Use this argument in `build-product-case`:",
                "",
                "```bash",
                f"--gi-seg {result.normalized_gi_segmentation_path} --gi-labelmap configs/labelmaps/gi_tract.yaml",
                "```",
            ]
        )
    else:
        lines.append("- No normalized GI segmentation was produced yet.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Manifest YAML: `{result.manifest_yaml_path}`",
            f"- Report: `{result.report_path}`",
            f"- Staging manifest: `{result.staging_manifest_path or 'not_written'}`",
            f"- Normalized GI segmentation: `{result.normalized_gi_segmentation_path or 'not_written'}`",
            f"- Preview PNG: `{result.preview_png_path or 'not_written'}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _build_segmenter_command(
    *,
    executable: str,
    ct_path: str | Path,
    output_dir: Path,
    segmenter_args: str,
) -> tuple[str, ...]:
    return (
        executable,
        "-i",
        str(ct_path),
        "-o",
        str(output_dir),
        *tuple(shlex.split(segmenter_args or "")),
    )


def _run_command(
    *,
    command: tuple[str, ...],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int | None,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        completed = subprocess.run(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    return int(completed.returncode)


def auto_stage_gi_segmentation(
    *,
    ct_path: str | Path,
    segmenter: str = "totalsegmentator",
    segmenter_output_dir: str | Path | None = None,
    segmenter_executable: str | Path | None = None,
    segmenter_args: str = "",
    gi_labelmap_path: str | Path = "configs/labelmaps/gi_tract.yaml",
    output_dir: str | Path = "outputs/digital/gi_autoseg_bridge",
    case_id: str = "gi_autoseg_stage001",
    report_path: str | Path | None = "outputs/reports/gi_autoseg_bridge_stage001.md",
    require_targets: tuple[str, ...] = ("small_bowel", "colon", "rectum"),
    force_rerun: bool = False,
    dry_run: bool = False,
    timeout_s: int | None = None,
) -> GIAutoSegBridgeResult:
    if segmenter not in SUPPORTED_SEGMENTERS:
        supported = ", ".join(sorted(SUPPORTED_SEGMENTERS))
        raise ValueError(f"Unsupported GI auto-segmenter: {segmenter}. Supported: {supported}")
    output = Path(output_dir) / case_id
    raw_output = Path(segmenter_output_dir) if segmenter_output_dir is not None else output / f"{segmenter}_output"
    manifest = output / f"{case_id}_gi_autoseg_bridge_manifest_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_gi_autoseg_bridge_report_v001.md"
    stdout_path = output / "logs" / f"{case_id}_{segmenter}_stdout.txt"
    stderr_path = output / "logs" / f"{case_id}_{segmenter}_stderr.txt"
    notes: list[str] = ["bridge_accepts_existing_segmenter_output_or_runs_supported_local_segmenter"]

    initial_targets = _detect_supported_targets(raw_output, gi_labelmap_path)
    command: tuple[str, ...] = ()
    command_returncode: int | None = None
    mode = "existing_output_folder" if initial_targets and not force_rerun else "segmenter_execution"

    if initial_targets and not force_rerun:
        notes.append("existing_supported_gi_output_folder_detected")
    elif dry_run:
        executable = _which_segmenter(segmenter, segmenter_executable)
        if executable is None:
            notes.append("dry_run_segmenter_executable_not_found")
            command = tuple()
        else:
            command = _build_segmenter_command(
                executable=executable,
                ct_path=ct_path,
                output_dir=raw_output,
                segmenter_args=segmenter_args,
            )
            notes.append("dry_run_no_segmenter_execution")
        result = GIAutoSegBridgeResult(
            case_id=case_id,
            ct_path=str(ct_path),
            output_dir=str(output),
            segmenter=segmenter,
            segmenter_mode="dry_run",
            readiness_status="planned_only",
            segmenter_output_dir=str(raw_output),
            command=command,
            command_returncode=None,
            command_stdout_path=None,
            command_stderr_path=None,
            staging_manifest_path=None,
            normalized_gi_segmentation_path=None,
            metrics_csv_path=None,
            preview_png_path=None,
            manifest_yaml_path=str(manifest),
            report_path=str(report),
            detected_targets=initial_targets,
            missing_targets=tuple(target for target in require_targets if target not in initial_targets),
            notes=tuple(notes),
        )
        _write_manifest(manifest, result)
        _write_report(report, result)
        return result
    else:
        executable = _which_segmenter(segmenter, segmenter_executable)
        if executable is None:
            notes.append("supported_segmenter_executable_not_found")
            detected = initial_targets
            result = GIAutoSegBridgeResult(
                case_id=case_id,
                ct_path=str(ct_path),
                output_dir=str(output),
                segmenter=segmenter,
                segmenter_mode="blocked_no_segmenter",
                readiness_status="blocked_segmenter_not_available",
                segmenter_output_dir=str(raw_output),
                command=(),
                command_returncode=None,
                command_stdout_path=None,
                command_stderr_path=None,
                staging_manifest_path=None,
                normalized_gi_segmentation_path=None,
                metrics_csv_path=None,
                preview_png_path=None,
                manifest_yaml_path=str(manifest),
                report_path=str(report),
                detected_targets=detected,
                missing_targets=tuple(target for target in require_targets if target not in detected),
                notes=tuple(notes),
            )
            _write_manifest(manifest, result)
            _write_report(report, result)
            return result
        raw_output.mkdir(parents=True, exist_ok=True)
        command = _build_segmenter_command(
            executable=executable,
            ct_path=ct_path,
            output_dir=raw_output,
            segmenter_args=segmenter_args,
        )
        try:
            command_returncode = _run_command(
                command=command,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_s=timeout_s,
            )
        except subprocess.TimeoutExpired:
            notes.append("segmenter_execution_timeout")
            command_returncode = -1
        if command_returncode != 0:
            notes.append("segmenter_execution_failed")
            detected = _detect_supported_targets(raw_output, gi_labelmap_path)
            result = GIAutoSegBridgeResult(
                case_id=case_id,
                ct_path=str(ct_path),
                output_dir=str(output),
                segmenter=segmenter,
                segmenter_mode=mode,
                readiness_status="blocked_segmenter_execution_failed",
                segmenter_output_dir=str(raw_output),
                command=command,
                command_returncode=command_returncode,
                command_stdout_path=str(stdout_path),
                command_stderr_path=str(stderr_path),
                staging_manifest_path=None,
                normalized_gi_segmentation_path=None,
                metrics_csv_path=None,
                preview_png_path=None,
                manifest_yaml_path=str(manifest),
                report_path=str(report),
                detected_targets=detected,
                missing_targets=tuple(target for target in require_targets if target not in detected),
                notes=tuple(notes),
            )
            _write_manifest(manifest, result)
            _write_report(report, result)
            return result

    detected_targets = _detect_supported_targets(raw_output, gi_labelmap_path)
    if not detected_targets:
        notes.append("segmenter_output_contains_no_supported_gi_masks")
        result = GIAutoSegBridgeResult(
            case_id=case_id,
            ct_path=str(ct_path),
            output_dir=str(output),
            segmenter=segmenter,
            segmenter_mode=mode,
            readiness_status="blocked_no_supported_gi_targets",
            segmenter_output_dir=str(raw_output),
            command=command,
            command_returncode=command_returncode,
            command_stdout_path=str(stdout_path) if command_returncode is not None else None,
            command_stderr_path=str(stderr_path) if command_returncode is not None else None,
            staging_manifest_path=None,
            normalized_gi_segmentation_path=None,
            metrics_csv_path=None,
            preview_png_path=None,
            manifest_yaml_path=str(manifest),
            report_path=str(report),
            detected_targets=detected_targets,
            missing_targets=tuple(target for target in require_targets if target not in detected_targets),
            notes=tuple(notes),
        )
        _write_manifest(manifest, result)
        _write_report(report, result)
        return result

    staging = stage_gi_segmentation(
        ct_path=ct_path,
        gi_segmentation_path=raw_output,
        gi_labelmap_path=gi_labelmap_path,
        output_dir=output / "staging",
        case_id=f"{case_id}_gi_staged",
        report_path=output / f"{case_id}_gi_segmentation_staging_report_v001.md",
        require_targets=require_targets,
    )
    readiness = staging.readiness_status
    notes.append("gi_segmentation_staging_completed")
    result = GIAutoSegBridgeResult(
        case_id=case_id,
        ct_path=str(ct_path),
        output_dir=str(output),
        segmenter=segmenter,
        segmenter_mode=mode,
        readiness_status=readiness,
        segmenter_output_dir=str(raw_output),
        command=command,
        command_returncode=command_returncode,
        command_stdout_path=str(stdout_path) if command_returncode is not None else None,
        command_stderr_path=str(stderr_path) if command_returncode is not None else None,
        staging_manifest_path=staging.manifest_yaml_path,
        normalized_gi_segmentation_path=staging.normalized_gi_segmentation_path,
        metrics_csv_path=staging.metrics_csv_path,
        preview_png_path=staging.preview_png_path,
        manifest_yaml_path=str(manifest),
        report_path=str(report),
        detected_targets=staging.present_targets,
        missing_targets=staging.missing_targets,
        notes=tuple(notes),
    )
    _write_manifest(manifest, result)
    _write_report(report, result)
    return result


def format_gi_autoseg_bridge_result(result: GIAutoSegBridgeResult) -> str:
    return "\n".join(
        [
            "GI Auto-Segmentation Bridge",
            f"Case ID: {result.case_id}",
            f"Readiness: {result.readiness_status}",
            f"Segmenter: {result.segmenter}",
            f"Mode: {result.segmenter_mode}",
            f"Detected targets: {', '.join(result.detected_targets) or 'none'}",
            f"Missing targets: {', '.join(result.missing_targets) or 'none'}",
            f"Normalized GI segmentation: {result.normalized_gi_segmentation_path or 'not_written'}",
            f"Report: {result.report_path}",
        ]
    )
