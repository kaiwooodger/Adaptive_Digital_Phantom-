from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .product_case_runner import ProductCaseResult, build_product_case
from .product_release import ProductReleaseResult, build_product_release_package


@dataclass(frozen=True)
class ProductReleaseCaseResult:
    case_id: str
    patient_id: str
    workflow_status: str
    output_dir: str
    workflow_manifest_path: str
    workflow_report_path: str
    product_manifest_path: str
    product_report_path: str
    product_final_status: str
    release_manifest_path: str
    release_readme_path: str
    release_report_path: str
    release_readiness_status: str
    release_artifact_index_csv_path: str
    release_overview_png_path: str
    qa_pass_count: int
    qa_review_count: int
    qa_fail_count: int
    notes: tuple[str, ...]


def _status(product: ProductCaseResult, release: ProductReleaseResult) -> str:
    if release.readiness_status == "product_release_ready":
        return "release_ready"
    if product.final_status == "research_demo_ready":
        return "release_review_required"
    if product.final_status == "research_demo_review_required":
        return "product_review_required"
    if product.final_status == "research_demo_needs_corrections":
        return "product_corrections_required"
    return "workflow_incomplete"


def _write_manifest(path: Path, result: ProductReleaseCaseResult, product: ProductCaseResult, release: ProductReleaseResult) -> None:
    payload: dict[str, Any] = {
        "case_id": result.case_id,
        "patient_id": result.patient_id,
        "package_type": "phantom_twin_product_release_case_workflow",
        "workflow_status": result.workflow_status,
        "summary": {
            "product_final_status": result.product_final_status,
            "release_readiness_status": result.release_readiness_status,
            "qa_pass_count": result.qa_pass_count,
            "qa_review_count": result.qa_review_count,
            "qa_fail_count": result.qa_fail_count,
        },
        "outputs": {
            "workflow_manifest": result.workflow_manifest_path,
            "workflow_report": result.workflow_report_path,
            "product_manifest": result.product_manifest_path,
            "product_report": result.product_report_path,
            "release_manifest": result.release_manifest_path,
            "release_readme": result.release_readme_path,
            "release_report": result.release_report_path,
            "release_artifact_index_csv": result.release_artifact_index_csv_path,
            "release_overview_png": result.release_overview_png_path,
        },
        "product_case": {
            "final_status": product.final_status,
            "adapter_manifest": product.adapter_manifest_path,
            "build_manifest": product.build_manifest_path,
            "qa_yaml": product.qa_yaml_path,
            "render_preview_png": product.render_preview_png_path,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "status": stage.status,
                    "primary_output_path": stage.primary_output_path,
                    "report_path": stage.report_path,
                    "notes": list(stage.notes),
                }
                for stage in product.stages
            ],
        },
        "release": {
            "release_id": release.release_id,
            "readiness_status": release.readiness_status,
            "artifact_count": release.artifact_count,
            "copied_artifact_count": release.copied_artifact_count,
            "indexed_large_artifact_count": release.indexed_large_artifact_count,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: ProductReleaseCaseResult) -> None:
    lines = [
        "# Product Release Case Workflow",
        "",
        f"Case ID: `{result.case_id}`",
        f"Patient/profile ID: `{result.patient_id}`",
        f"Workflow status: `{result.workflow_status}`",
        "",
        "## Summary",
        "",
        f"- Product status: `{result.product_final_status}`",
        f"- Release status: `{result.release_readiness_status}`",
        f"- QA pass / review / fail: {result.qa_pass_count} / {result.qa_review_count} / {result.qa_fail_count}",
        "",
        "## Outputs",
        "",
        f"- Product manifest: `{result.product_manifest_path}`",
        f"- Product report: `{result.product_report_path}`",
        f"- Release README: `{result.release_readme_path}`",
        f"- Release manifest: `{result.release_manifest_path}`",
        f"- Release report: `{result.release_report_path}`",
        f"- Release artifact index: `{result.release_artifact_index_csv_path}`",
        f"- Release overview PNG: `{result.release_overview_png_path}`",
        "",
        "## Interpretation",
        "",
    ]
    if result.workflow_status == "release_ready":
        lines.append("- The case produced a release-ready research demonstrator package.")
    elif result.workflow_status == "product_review_required":
        lines.append("- The product case completed but has review items; inspect the QA gate before using it as a release candidate.")
    elif result.workflow_status == "product_corrections_required":
        lines.append("- The product case has QA failures and needs correction before release-readiness claims.")
    else:
        lines.append("- The workflow did not reach full release readiness; inspect the product and release reports.")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_product_release_case(
    *,
    input_ct_path: str | Path | None = None,
    input_cta_path: str | Path | None = None,
    input_ctv_path: str | Path | None = None,
    organ_seg_path: str | Path | None = None,
    vessel_seg_path: str | Path | None = None,
    existing_build_manifest_path: str | Path | None = None,
    patient_id: str = "patient_demo",
    case_id: str | None = None,
    output_dir: str | Path = "outputs/product_release_cases",
    product_output_dir: str | Path = "outputs/product_cases",
    release_output_dir: str | Path = "outputs/releases/product_cases",
    release_id: str = "product_case_rc1",
    organ_labelmap_path: str | Path = "configs/labelmaps/ct_org.yaml",
    materials_path: str | Path = "configs/materials.yaml",
    baseline_graph_path: str | Path | None = None,
    baseline_combined_spec_path: str | Path | None = None,
    approved_set_manifest_path: str | Path | None = None,
    target_height_cm: float | None = None,
    target_weight_kg: float | None = None,
    target_bmi: float | None = None,
    target_waist_cm: float | None = None,
    copy_inputs: bool = False,
    allow_template_vessels: bool = False,
    dry_run: bool = False,
    run_rt: bool = True,
    export_dicom: bool = False,
    sample_step_mm: float = 0.75,
    vessel_wall_thickness_mm: float = 2.0,
    arterial_inlet_flow_ml_s: float = 80.0,
    heart_rate_bpm: float = 60.0,
    organ_label_mode: str = "auto",
    correct_bone_conflicts: bool = False,
    bone_clearance_mm: float = 8.0,
    run_qa: bool = True,
    qa_expected_lumen_components: int = 1,
    render_3d: bool = True,
    existing_render_preview_path: str | Path | None = None,
    existing_render_scene_spec_path: str | Path | None = None,
    render_target_max_faces: int = 90_000,
    copy_small_release_artifacts: bool = True,
    release_large_threshold_bytes: int = 25_000_000,
    release_command_lines: tuple[str, ...] = (),
    supplemental_artifact_paths: tuple[str | Path, ...] = (),
    report_path: str | Path | None = None,
) -> ProductReleaseCaseResult:
    product = build_product_case(
        input_ct_path=input_ct_path,
        input_cta_path=input_cta_path,
        input_ctv_path=input_ctv_path,
        organ_seg_path=organ_seg_path,
        vessel_seg_path=vessel_seg_path,
        existing_build_manifest_path=existing_build_manifest_path,
        patient_id=patient_id,
        case_id=case_id,
        output_dir=product_output_dir,
        organ_labelmap_path=organ_labelmap_path,
        materials_path=materials_path,
        baseline_graph_path=baseline_graph_path,
        baseline_combined_spec_path=baseline_combined_spec_path,
        approved_set_manifest_path=approved_set_manifest_path,
        target_height_cm=target_height_cm,
        target_weight_kg=target_weight_kg,
        target_bmi=target_bmi,
        target_waist_cm=target_waist_cm,
        copy_inputs=copy_inputs,
        allow_template_vessels=allow_template_vessels,
        dry_run=dry_run,
        run_rt=run_rt,
        export_dicom=export_dicom,
        sample_step_mm=sample_step_mm,
        vessel_wall_thickness_mm=vessel_wall_thickness_mm,
        arterial_inlet_flow_ml_s=arterial_inlet_flow_ml_s,
        heart_rate_bpm=heart_rate_bpm,
        organ_label_mode=organ_label_mode,
        correct_bone_conflicts=correct_bone_conflicts,
        bone_clearance_mm=bone_clearance_mm,
        run_qa=run_qa,
        qa_expected_lumen_components=qa_expected_lumen_components,
        render_3d=render_3d,
        existing_render_preview_path=existing_render_preview_path,
        existing_render_scene_spec_path=existing_render_scene_spec_path,
        render_target_max_faces=render_target_max_faces,
    )
    command_lines = release_command_lines
    if not command_lines:
        command_lines = (
            "python -m phantom_twin.cli build-product-release-case "
            "--input-ct <ct.nii.gz-or-dicom-dir> --organ-seg <labels.nii.gz> "
            "--vessel-seg <vessels.nii.gz> --baseline-graph <graph.yaml>",
        )
    release = build_product_release_package(
        product_manifest_path=product.product_manifest_path,
        output_dir=release_output_dir,
        release_id=release_id,
        copy_small_artifacts=copy_small_release_artifacts,
        large_threshold_bytes=release_large_threshold_bytes,
        command_lines=command_lines,
        supplemental_artifact_paths=supplemental_artifact_paths,
    )

    workflow_root = Path(output_dir) / product.case_id
    workflow_manifest = workflow_root / f"{product.case_id}_{release_id}_workflow_manifest_v001.yaml"
    workflow_report = Path(report_path) if report_path is not None else workflow_root / f"{product.case_id}_{release_id}_workflow_report_v001.md"
    notes = (
        "single_command_product_case_and_release_workflow",
        "release_package_keeps_large_volumes_and_meshes_indexed_by_default",
    )
    result = ProductReleaseCaseResult(
        case_id=product.case_id,
        patient_id=product.patient_id,
        workflow_status=_status(product, release),
        output_dir=str(workflow_root),
        workflow_manifest_path=str(workflow_manifest),
        workflow_report_path=str(workflow_report),
        product_manifest_path=product.product_manifest_path,
        product_report_path=product.report_path,
        product_final_status=product.final_status,
        release_manifest_path=release.release_manifest_path,
        release_readme_path=release.readme_path,
        release_report_path=release.report_path,
        release_readiness_status=release.readiness_status,
        release_artifact_index_csv_path=release.artifact_index_csv_path,
        release_overview_png_path=release.overview_png_path,
        qa_pass_count=release.qa_pass_count,
        qa_review_count=release.qa_review_count,
        qa_fail_count=release.qa_fail_count,
        notes=notes,
    )
    _write_manifest(workflow_manifest, result, product, release)
    _write_report(workflow_report, result)
    return result


def format_product_release_case_result(result: ProductReleaseCaseResult) -> str:
    return "\n".join(
        [
            "Product release case workflow",
            f"Case ID: {result.case_id}",
            f"Patient/profile ID: {result.patient_id}",
            f"Workflow status: {result.workflow_status}",
            f"Product status: {result.product_final_status}",
            f"Release status: {result.release_readiness_status}",
            f"QA pass/review/fail: {result.qa_pass_count}/{result.qa_review_count}/{result.qa_fail_count}",
            f"Workflow manifest: {result.workflow_manifest_path}",
            f"Release README: {result.release_readme_path}",
            f"Report: {result.workflow_report_path}",
        ]
    )
