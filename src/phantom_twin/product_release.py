from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import hashlib
import shutil

import yaml


VOLUME_SUFFIXES = (".nii.gz", ".nii", ".dcm", ".dicom", ".mha", ".mhd", ".nrrd")
MESH_SUFFIXES = (".stl", ".ply", ".obj")


@dataclass(frozen=True)
class ProductReleaseArtifact:
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
class ProductReleaseResult:
    case_id: str
    patient_id: str
    release_id: str
    readiness_status: str
    source_product_manifest_path: str
    output_dir: str
    release_manifest_path: str
    readme_path: str
    artifact_index_csv_path: str
    validation_summary_csv_path: str
    command_log_path: str
    limitations_path: str
    overview_png_path: str
    report_path: str
    artifact_count: int
    copied_artifact_count: int
    indexed_large_artifact_count: int
    qa_pass_count: int
    qa_review_count: int
    qa_fail_count: int
    artifacts: tuple[ProductReleaseArtifact, ...]
    notes: tuple[str, ...]


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
    if name.endswith(MESH_SUFFIXES):
        return "mesh"
    return path.suffix.lower().lstrip(".") or "unknown"


def _artifact_group(path: Path) -> str:
    parts = set(path.parts)
    if "aorta_registration_benchmark_stage001" in parts or "aorta_registration_benchmark" in parts:
        return "aorta_registration_benchmark"
    if "vascular_network_learned_aorta_stage004" in parts or "vascular_network_learned_aorta" in parts:
        return "learned_aorta_graph"
    if "vessel_edge_reroutes_stage004_learned_aorta_strong" in parts or "vessel_edge_reroutes_stage004_learned_aorta" in parts:
        return "learned_aorta_reroute"
    if "vessel_radius_tuned_stage004_learned_aorta_safe" in parts or "vessel_radius_tuned_stage004_learned_aorta" in parts:
        return "learned_aorta_radius_tuning"
    if "product_cases" in parts:
        return "product_case"
    if "torso" in parts:
        return "torso_materials"
    if "vascular_graph" in parts or "btcv_coarse_vessel_graph_stage003" in parts:
        return "vascular_graph"
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
    if "render3d" in parts:
        return "render3d"
    if "reports" in parts:
        return "reports"
    return "dependency"


def _resolve_supplemental_paths(paths: tuple[str | Path, ...]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            resolved.append(path)
        elif path.is_dir():
            resolved.extend(item for item in path.rglob("*") if item.is_file())
    return resolved


def _sha256(path: Path, max_size_bytes: int) -> str:
    if not path.exists() or path.stat().st_size > max_size_bytes:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_large_or_external_binary(path: Path, large_threshold_bytes: int) -> bool:
    name = path.name.lower()
    return name.endswith(VOLUME_SUFFIXES) or name.endswith(MESH_SUFFIXES) or path.stat().st_size > large_threshold_bytes


def _relative_copy_destination(path: Path, output_dir: Path) -> Path:
    safe_parts = [part for part in path.parts if part not in {"", "/"}]
    if path.is_absolute():
        safe_parts = safe_parts[-min(len(safe_parts), 8) :]
    return output_dir / "artifacts" / Path(*safe_parts)


def _maybe_path(value: Any) -> Path | None:
    if not isinstance(value, str) or value == "":
        return None
    path = Path(value)
    return path if path.exists() and path.is_file() else None


def _collect_paths_from_mapping(mapping: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for value in mapping.values():
        if isinstance(value, str):
            path = _maybe_path(value)
            if path is not None:
                paths.append(path)
        elif isinstance(value, list):
            for item in value:
                path = _maybe_path(item)
                if path is not None:
                    paths.append(path)
        elif isinstance(value, dict):
            paths.extend(_collect_paths_from_mapping(value))
    return paths


def _discover_artifacts(product_manifest_path: Path, product_manifest: dict[str, Any]) -> list[Path]:
    product_root = product_manifest_path.parent
    paths: list[Path] = []
    paths.append(product_manifest_path)
    paths.extend(path for path in product_root.rglob("*") if path.is_file())

    outputs = _as_mapping(product_manifest.get("outputs"))
    paths.extend(_collect_paths_from_mapping(outputs))

    for stage in _as_list(product_manifest.get("stages")):
        if not isinstance(stage, dict):
            continue
        for key in ("primary_output_path", "report_path"):
            path = _maybe_path(stage.get(key))
            if path is not None:
                paths.append(path)

    build_manifest_path = _maybe_path(outputs.get("build_manifest"))
    if build_manifest_path is not None:
        build_manifest = _load_yaml(build_manifest_path)
        paths.append(build_manifest_path)
        paths.extend(_collect_paths_from_mapping(_as_mapping(build_manifest.get("outputs"))))
        build_root = build_manifest_path.parent
        if build_root.exists():
            paths.extend(path for path in build_root.rglob("*") if path.is_file())

    adapter_manifest_path = _maybe_path(outputs.get("adapter_manifest"))
    if adapter_manifest_path is not None:
        paths.append(adapter_manifest_path)
        adapter_manifest = _load_yaml(adapter_manifest_path)
        paths.extend(_collect_paths_from_mapping(_as_mapping(adapter_manifest.get("outputs"))))

    render_scene_path = _maybe_path(outputs.get("render_scene_spec"))
    if render_scene_path is not None:
        render_scene = _load_yaml(render_scene_path)
        paths.append(render_scene_path)
        paths.extend(_collect_paths_from_mapping(render_scene))

    unique: dict[str, Path] = {}
    for path in paths:
        if path.exists() and path.is_file():
            unique[str(path)] = path
    return sorted(unique.values(), key=lambda item: str(item))


def _stage_artifacts(
    paths: list[Path],
    output_dir: Path,
    *,
    copy_small_artifacts: bool,
    large_threshold_bytes: int,
) -> tuple[ProductReleaseArtifact, ...]:
    artifacts: list[ProductReleaseArtifact] = []
    for path in paths:
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        large = exists and _is_large_or_external_binary(path, large_threshold_bytes)
        copy_policy = "indexed_only_large_volume_or_mesh" if large else "copied"
        packaged_path = ""
        notes: list[str] = []
        if large:
            notes.append("not_copied_to_keep_product_release_disk_light")
        elif exists and copy_small_artifacts:
            destination = _relative_copy_destination(path, output_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            packaged_path = str(destination)
        elif not copy_small_artifacts:
            copy_policy = "indexed_only_copy_disabled"
        artifacts.append(
            ProductReleaseArtifact(
                group=_artifact_group(path),
                role=path.stem,
                file_type=_file_type(path),
                source_path=str(path),
                exists=exists,
                size_bytes=size_bytes,
                sha256=_sha256(path, large_threshold_bytes) if exists and not large else "",
                copy_policy=copy_policy,
                packaged_path=packaged_path,
                notes=tuple(notes),
            )
        )
    return tuple(artifacts)


def _qa_counts(product_manifest: dict[str, Any]) -> tuple[int, int, int]:
    qa_yaml = _maybe_path(_as_mapping(product_manifest.get("outputs")).get("qa_yaml"))
    qa = _load_yaml(qa_yaml)
    summary = _as_mapping(qa.get("summary"))
    return (
        int(summary.get("pass_count", 0) or 0),
        int(summary.get("review_count", 0) or 0),
        int(summary.get("fail_count", 0) or 0),
    )


def _write_artifact_index(path: Path, artifacts: tuple[ProductReleaseArtifact, ...]) -> None:
    fields = [
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
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


def _write_validation_summary(path: Path, product_manifest: dict[str, Any]) -> None:
    qa_yaml = _maybe_path(_as_mapping(product_manifest.get("outputs")).get("qa_yaml"))
    qa = _load_yaml(qa_yaml)
    checks = [item for item in _as_list(qa.get("checks")) if isinstance(item, dict)]
    fields = ["check_id", "category", "status", "metric", "value", "threshold", "source_path", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for check in checks:
            writer.writerow(
                {
                    "check_id": check.get("check_id", ""),
                    "category": check.get("category", ""),
                    "status": check.get("status", ""),
                    "metric": check.get("metric", ""),
                    "value": check.get("value", ""),
                    "threshold": check.get("threshold", ""),
                    "source_path": check.get("source_path", ""),
                    "notes": ";".join(str(note) for note in _as_list(check.get("notes"))),
                }
            )


def _write_command_log(path: Path, result: ProductReleaseResult, command_lines: tuple[str, ...]) -> None:
    lines = [
        "# Product Release Command Log",
        "",
        "## Rebuild This Release Wrapper",
        "",
        "```bash",
        (
            "python -m phantom_twin.cli build-product-release-package "
            f"--product-manifest {result.source_product_manifest_path} "
            f"--release-id {result.release_id}"
        ),
        "```",
        "",
        "## Upstream Commands",
        "",
    ]
    if command_lines:
        for command in command_lines:
            lines.extend(["```bash", command, "```", ""])
    else:
        lines.append("- No upstream command lines were supplied. Use the product manifest and build manifest paths in the README to reproduce manually.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def _write_limitations(path: Path, product_manifest: dict[str, Any]) -> None:
    lines = [
        "# Product Release Limitations",
        "",
        "- This package is a research/engineering demonstrator, not a cleared clinical device.",
        "- The current BTCV-ready release is scoped to coarse major-vessel domains only: aorta segment plus IVC return segment.",
        "- Branch-rich renal, hepatic, splenic, iliac, and venous vascular claims are intentionally excluded from this release.",
        "- Anatomical validity depends on CT/segmentation quality, registration, label harmonization, and QA thresholds.",
        "- Flow outputs are digital model outputs and require calibration against measured flow/pressure before physiological claims.",
        "- RT outputs are workflow and QA demonstrators unless connected to a validated TPS or dose engine.",
        "",
        f"Source product status: `{product_manifest.get('final_status', 'unknown')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_overview_png(path: Path, product_manifest: dict[str, Any], qa_counts: tuple[int, int, int]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Product release overview figure generation requires matplotlib.") from exc

    render_path = _maybe_path(_as_mapping(product_manifest.get("outputs")).get("render_preview_png"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=170)
    fig.patch.set_facecolor("#f7f1e3")
    for ax in axes:
        ax.set_facecolor("#f7f1e3")
    if render_path is not None:
        try:
            axes[0].imshow(plt.imread(render_path))
            axes[0].set_title("3D CAD Preview")
        except Exception:
            axes[0].text(0.5, 0.5, "3D preview linked\nbut could not be loaded", ha="center", va="center")
    else:
        axes[0].text(0.5, 0.5, "No 3D preview", ha="center", va="center")
    axes[0].axis("off")

    pass_count, review_count, fail_count = qa_counts
    axes[1].bar(["pass", "review", "fail"], [pass_count, review_count, fail_count], color=["#2a9d8f", "#f4a261", "#e76f51"])
    axes[1].set_title("Product QA Gate")
    axes[1].set_ylabel("checks")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Product Release: {product_manifest.get('case_id', 'case')}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_readme(path: Path, result: ProductReleaseResult, product_manifest: dict[str, Any]) -> None:
    groups = sorted({artifact.group for artifact in result.artifacts})
    lines = [
        f"# {result.case_id} Product Release",
        "",
        f"Release ID: `{result.release_id}`",
        f"Patient/profile ID: `{result.patient_id}`",
        f"Readiness status: `{result.readiness_status}`",
        "",
        "## Summary",
        "",
        f"- Source product status: `{product_manifest.get('final_status', 'unknown')}`",
        f"- QA pass / review / fail: {result.qa_pass_count} / {result.qa_review_count} / {result.qa_fail_count}",
        f"- Indexed artifacts: {result.artifact_count}",
        f"- Copied small artifacts: {result.copied_artifact_count}",
        f"- Indexed large/volume/mesh artifacts: {result.indexed_large_artifact_count}",
        f"- Artifact groups: {', '.join(groups)}",
        "",
        "## Important Files",
        "",
        f"- Product manifest: `{_as_mapping(product_manifest.get('outputs')).get('product_manifest', '')}`",
        f"- Product report: `{_as_mapping(product_manifest.get('outputs')).get('report', '')}`",
        f"- QA gate: `{_as_mapping(product_manifest.get('outputs')).get('qa_yaml', '')}`",
        f"- 3D render: `{_as_mapping(product_manifest.get('outputs')).get('render_preview_png', '')}`",
        f"- Artifact index: `{result.artifact_index_csv_path}`",
        f"- Validation summary: `{result.validation_summary_csv_path}`",
        f"- Limitations: `{result.limitations_path}`",
        "",
        "## Scope",
        "",
        "This release is ready as a coarse major-vessel research demonstrator. It does not claim branch-rich patient-specific vascular anatomy.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_manifest(path: Path, result: ProductReleaseResult) -> None:
    payload = {
        "case_id": result.case_id,
        "patient_id": result.patient_id,
        "release_id": result.release_id,
        "package_type": "phantom_twin_product_release_package",
        "readiness_status": result.readiness_status,
        "source_product_manifest": result.source_product_manifest_path,
        "summary": {
            "artifact_count": result.artifact_count,
            "copied_artifact_count": result.copied_artifact_count,
            "indexed_large_artifact_count": result.indexed_large_artifact_count,
            "qa_pass_count": result.qa_pass_count,
            "qa_review_count": result.qa_review_count,
            "qa_fail_count": result.qa_fail_count,
        },
        "outputs": {
            "release_manifest": result.release_manifest_path,
            "readme": result.readme_path,
            "artifact_index_csv": result.artifact_index_csv_path,
            "validation_summary_csv": result.validation_summary_csv_path,
            "command_log": result.command_log_path,
            "limitations": result.limitations_path,
            "overview_png": result.overview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: ProductReleaseResult) -> None:
    lines = [
        "# Product Release Package Report",
        "",
        f"Case ID: `{result.case_id}`",
        f"Release ID: `{result.release_id}`",
        f"Readiness status: `{result.readiness_status}`",
        "",
        "## QA",
        "",
        f"- Pass / review / fail: {result.qa_pass_count} / {result.qa_review_count} / {result.qa_fail_count}",
        "",
        "## Artifacts",
        "",
        f"- Indexed artifacts: {result.artifact_count}",
        f"- Copied small artifacts: {result.copied_artifact_count}",
        f"- Indexed large/volume/mesh artifacts: {result.indexed_large_artifact_count}",
        "",
        "## Outputs",
        "",
        f"- README: `{result.readme_path}`",
        f"- Artifact index: `{result.artifact_index_csv_path}`",
        f"- Validation summary: `{result.validation_summary_csv_path}`",
        f"- Overview PNG: `{result.overview_png_path}`",
        f"- Limitations: `{result.limitations_path}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_product_release_package(
    *,
    product_manifest_path: str | Path,
    output_dir: str | Path = "outputs/releases/product_cases",
    release_id: str = "product_mvp_rc1",
    copy_small_artifacts: bool = True,
    large_threshold_bytes: int = 25_000_000,
    command_lines: tuple[str, ...] = (),
    supplemental_artifact_paths: tuple[str | Path, ...] = (),
    report_path: str | Path | None = None,
) -> ProductReleaseResult:
    product_manifest_file = Path(product_manifest_path)
    product_manifest = _load_yaml(product_manifest_file)
    if not product_manifest:
        raise ValueError(f"Could not load product manifest: {product_manifest_path}")
    case_id = str(product_manifest.get("case_id", product_manifest_file.stem))
    patient_id = str(product_manifest.get("patient_id", "patient"))
    output = Path(output_dir) / f"{case_id}_{release_id}"
    output.mkdir(parents=True, exist_ok=True)

    paths = _discover_artifacts(product_manifest_file, product_manifest)
    paths.extend(_resolve_supplemental_paths(supplemental_artifact_paths))
    paths = sorted({str(path): path for path in paths if path.exists() and path.is_file()}.values(), key=lambda item: str(item))
    artifacts = _stage_artifacts(
        paths,
        output,
        copy_small_artifacts=copy_small_artifacts,
        large_threshold_bytes=large_threshold_bytes,
    )
    qa_pass, qa_review, qa_fail = _qa_counts(product_manifest)
    readiness = "product_release_ready" if product_manifest.get("final_status") == "research_demo_ready" and qa_fail == 0 and qa_review == 0 else "product_release_review_required"

    release_manifest = output / f"{case_id}_{release_id}_release_manifest_v001.yaml"
    readme = output / "README.md"
    artifact_index = output / f"{case_id}_{release_id}_artifact_index_v001.csv"
    validation_summary = output / f"{case_id}_{release_id}_validation_summary_v001.csv"
    command_log = output / "COMMANDS.md"
    limitations = output / "LIMITATIONS.md"
    overview = output / f"{case_id}_{release_id}_overview_v001.png"
    report = Path(report_path) if report_path else output / f"{case_id}_{release_id}_release_report_v001.md"

    result = ProductReleaseResult(
        case_id=case_id,
        patient_id=patient_id,
        release_id=release_id,
        readiness_status=readiness,
        output_dir=str(output),
        source_product_manifest_path=str(product_manifest_file),
        release_manifest_path=str(release_manifest),
        readme_path=str(readme),
        artifact_index_csv_path=str(artifact_index),
        validation_summary_csv_path=str(validation_summary),
        command_log_path=str(command_log),
        limitations_path=str(limitations),
        overview_png_path=str(overview),
        report_path=str(report),
        artifact_count=len(artifacts),
        copied_artifact_count=sum(1 for item in artifacts if item.packaged_path),
        indexed_large_artifact_count=sum(item.copy_policy == "indexed_only_large_volume_or_mesh" for item in artifacts),
        qa_pass_count=qa_pass,
        qa_review_count=qa_review,
        qa_fail_count=qa_fail,
        artifacts=artifacts,
        notes=(
            "product_release_package_is_disk_light_large_volumes_and_meshes_are_indexed_not_copied",
            "release_scope_is_coarse_major_vessel_research_demonstrator",
            "supplemental_artifacts_indexed_when_supplied",
        ),
    )
    _write_artifact_index(artifact_index, artifacts)
    _write_validation_summary(validation_summary, product_manifest)
    _write_command_log(command_log, result, command_lines)
    _write_limitations(limitations, product_manifest)
    _write_overview_png(overview, product_manifest, (qa_pass, qa_review, qa_fail))
    _write_readme(readme, result, product_manifest)
    _write_manifest(release_manifest, result)
    _write_report(report, result)
    return result


def format_product_release_result(result: ProductReleaseResult) -> str:
    return "\n".join(
        [
            "Product release package built",
            f"Case ID: {result.case_id}",
            f"Release ID: {result.release_id}",
            f"Readiness status: {result.readiness_status}",
            f"QA pass/review/fail: {result.qa_pass_count}/{result.qa_review_count}/{result.qa_fail_count}",
            f"Artifacts indexed/copied/large: {result.artifact_count}/{result.copied_artifact_count}/{result.indexed_large_artifact_count}",
            f"README: {result.readme_path}",
            f"Release manifest: {result.release_manifest_path}",
            f"Report: {result.report_path}",
        ]
    )
