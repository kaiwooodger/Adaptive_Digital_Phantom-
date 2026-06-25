from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .patient_adapter import PatientPhantomAdapterResult, build_patient_phantom_adapter
from .patient_build import PatientPhantomBuildResult, run_patient_phantom_build
from .patient_build_qa import PatientBuildQAResult, qa_patient_phantom_build
from .render3d import Render3DResult, generate_vascular_network_3d_render
from .stage007_baseline import resolve_stage007_active_baseline


@dataclass(frozen=True)
class ProductCaseStage:
    stage_id: str
    status: str
    primary_output_path: str | None
    report_path: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ProductCaseQABlocker:
    check_id: str
    category: str
    status: str
    metric: str
    value: str
    threshold: str
    source_path: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ProductCaseResult:
    case_id: str
    patient_id: str
    output_dir: str
    product_manifest_path: str
    report_path: str
    final_status: str
    adapter_manifest_path: str | None
    build_manifest_path: str | None
    qa_yaml_path: str | None
    render_preview_png_path: str | None
    render_scene_spec_path: str | None
    vessel_visible_preview_png_path: str | None
    stages: tuple[ProductCaseStage, ...]
    qa_blockers: tuple[ProductCaseQABlocker, ...]
    recommended_next_actions: tuple[str, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "case"


def _output_path(manifest: dict[str, Any], key: str) -> str | None:
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    value = outputs.get(key)
    return None if value is None or str(value) == "" else str(value)


def _existing(path: str | Path | None) -> str | None:
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    return str(candidate) if candidate.exists() else None


def _render_inputs_from_build(build_manifest_path: str | Path) -> dict[str, str]:
    build_manifest = _load_yaml(build_manifest_path)
    torso_spec_path = _output_path(build_manifest, "torso_spec")
    voxelized_spec_path = _output_path(build_manifest, "voxelized_spec")
    if torso_spec_path is None or voxelized_spec_path is None:
        raise ValueError("Build manifest must include torso_spec and voxelized_spec outputs for 3D rendering.")

    torso_spec = _load_yaml(torso_spec_path)
    voxel_spec = _load_yaml(voxelized_spec_path)
    torso_outputs = torso_spec.get("outputs", {})
    voxel_outputs = voxel_spec.get("outputs", {})
    voxelization = voxel_spec.get("voxelization", {})
    if not isinstance(torso_outputs, dict) or not isinstance(voxel_outputs, dict):
        raise ValueError("Torso and voxelized specs must contain outputs mappings for 3D rendering.")

    context_labels = torso_outputs.get("material_label_map") or voxelization.get("source_combined_labels")
    required = {
        "context_labels": context_labels,
        "arterial_lumen_mask": voxel_outputs.get("arterial_lumen_mask"),
        "venous_lumen_mask": voxel_outputs.get("venous_lumen_mask"),
        "flow_domain_labels": voxel_outputs.get("flow_domain_labels"),
        "vessel_wall_mask": voxel_outputs.get("vessel_wall_mask"),
    }
    missing = [key for key, value in required.items() if _existing(value) is None]
    if missing:
        raise ValueError(f"Missing render input(s): {', '.join(missing)}")
    render_inputs = {key: str(value) for key, value in required.items()}
    vascular_graph = _output_path(build_manifest, "vascular_graph")
    if _existing(vascular_graph) is not None:
        render_inputs["vascular_graph"] = str(vascular_graph)
    return render_inputs


def _qa_summary(qa_result: PatientBuildQAResult | None, qa_yaml_path: str | None) -> tuple[str, str, str, str]:
    if qa_result is not None:
        return (
            qa_result.readiness_status,
            str(qa_result.pass_count),
            str(qa_result.review_count),
            str(qa_result.fail_count),
        )
    if qa_yaml_path is None or not Path(qa_yaml_path).exists():
        return ("not_run", "n/a", "n/a", "n/a")
    qa = _load_yaml(qa_yaml_path)
    summary = qa.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return (
        str(qa.get("readiness_status", "unknown")),
        str(summary.get("pass_count", "n/a")),
        str(summary.get("review_count", "n/a")),
        str(summary.get("fail_count", "n/a")),
    )


def _extract_qa_blockers(
    qa_result: PatientBuildQAResult | None,
    qa_yaml_path: str | None,
    *,
    max_items: int = 8,
) -> tuple[ProductCaseQABlocker, ...]:
    if qa_result is not None:
        checks = [
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
            for item in qa_result.checks
        ]
    elif qa_yaml_path is not None and Path(qa_yaml_path).exists():
        qa = _load_yaml(qa_yaml_path)
        raw_checks = qa.get("checks", [])
        checks = [item for item in raw_checks if isinstance(item, dict)]
    else:
        checks = []

    ranked = sorted(
        (item for item in checks if str(item.get("status", "")) in {"fail", "review"}),
        key=lambda item: 0 if str(item.get("status", "")) == "fail" else 1,
    )
    blockers: list[ProductCaseQABlocker] = []
    for item in ranked[:max_items]:
        notes_raw = item.get("notes", [])
        if isinstance(notes_raw, str):
            notes = (notes_raw,) if notes_raw else ()
        elif isinstance(notes_raw, list):
            notes = tuple(str(value) for value in notes_raw[:4])
        else:
            notes = ()
        blockers.append(
            ProductCaseQABlocker(
                check_id=str(item.get("check_id", "")),
                category=str(item.get("category", "")),
                status=str(item.get("status", "")),
                metric=str(item.get("metric", "")),
                value=str(item.get("value", "")),
                threshold=str(item.get("threshold", "")),
                source_path=None if item.get("source_path") is None else str(item.get("source_path")),
                notes=notes,
            )
        )
    return tuple(blockers)


def _final_status(
    *,
    build_result: PatientPhantomBuildResult | None,
    build_manifest_path: str | None,
    qa_result: PatientBuildQAResult | None,
    qa_yaml_path: str | None,
    render_preview_png_path: str | None,
) -> str:
    build_status = None
    if build_result is not None:
        build_status = build_result.overall_status
    elif build_manifest_path is not None and Path(build_manifest_path).exists():
        build_status = str(_load_yaml(build_manifest_path).get("overall_status", "unknown"))

    if build_status != "completed":
        if build_status == "planned_only":
            return "build_planned_only"
        return "blocked_before_complete_build"

    qa_status, _, _, fail_count = _qa_summary(qa_result, qa_yaml_path)
    if fail_count not in {"0", "n/a"}:
        return "research_demo_needs_corrections"
    if qa_status == "review_required":
        return "research_demo_review_required"
    if qa_status == "approved_research_use":
        return "research_demo_ready"
    if render_preview_png_path is None:
        return "completed_missing_3d_render"
    return "completed_qa_not_run"


def _recommended_actions(final_status: str, qa_result: PatientBuildQAResult | None) -> tuple[str, ...]:
    if qa_result is not None and qa_result.recommended_actions:
        return qa_result.recommended_actions
    if final_status == "research_demo_ready":
        return (
            "Freeze this package as a versioned research demonstrator and add the command line invocation to the release notes.",
            "Run the same product case command on additional paired CT/vessel cases to quantify reproducibility.",
        )
    if final_status == "research_demo_needs_corrections":
        return (
            "Inspect the QA report first, then reroute/prune vessels or fix registration before making product-readiness claims.",
        )
    if final_status == "build_planned_only":
        return (
            "Rerun without --dry-run once inputs, baseline graph, and label modes are confirmed.",
        )
    if final_status == "completed_missing_3d_render":
        return (
            "Run the same command with 3D rendering enabled to produce the user-facing CAD preview and STL exports.",
        )
    return (
        "Resolve the blocked stage in the product report, then rerun the product case command.",
    )


def _write_product_manifest(path: Path, result: ProductCaseResult) -> None:
    payload = {
        "case_id": result.case_id,
        "patient_id": result.patient_id,
        "package_type": "phantom_twin_product_case_package",
        "final_status": result.final_status,
        "outputs": {
            "product_manifest": result.product_manifest_path,
            "report": result.report_path,
            "adapter_manifest": result.adapter_manifest_path,
            "build_manifest": result.build_manifest_path,
            "qa_yaml": result.qa_yaml_path,
            "render_preview_png": result.render_preview_png_path,
            "render_scene_spec": result.render_scene_spec_path,
            "vessel_visible_preview_png": result.vessel_visible_preview_png_path,
        },
        "stages": [
            {
                "stage_id": stage.stage_id,
                "status": stage.status,
                "primary_output_path": stage.primary_output_path,
                "report_path": stage.report_path,
                "notes": list(stage.notes),
            }
            for stage in result.stages
        ],
        "recommended_next_actions": list(result.recommended_next_actions),
        "qa_blockers": [
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
            for item in result.qa_blockers
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_product_report(
    path: Path,
    result: ProductCaseResult,
    *,
    qa_result: PatientBuildQAResult | None,
    qa_yaml_path: str | None,
) -> None:
    qa_status, qa_pass, qa_review, qa_fail = _qa_summary(qa_result, qa_yaml_path)
    lines = [
        "# Phantom Twin Product Case Report",
        "",
        f"Case ID: `{result.case_id}`",
        f"Patient/profile ID: `{result.patient_id}`",
        f"Final status: `{result.final_status}`",
        "",
        "## What This Package Contains",
        "",
        "- A staged patient-input manifest for CT, organ labels, vessel labels, and optional CTA/CTV metadata.",
        "- A generated digital phantom build manifest covering torso material labels, vascular graph/voxel masks, flow, and RT outputs when available.",
        "- A QA gate report that separates pass, review, and fail conditions.",
        "- A CAD-style 3D render package when rendering is enabled and the build has voxelized vessel outputs.",
        "",
        "## Stage Status",
        "",
        "| stage | status | output | report |",
        "| --- | --- | --- | --- |",
    ]
    for stage in result.stages:
        output = "" if stage.primary_output_path is None else f"`{stage.primary_output_path}`"
        report = "" if stage.report_path is None else f"`{stage.report_path}`"
        lines.append(f"| {stage.stage_id} | `{stage.status}` | {output} | {report} |")

    lines.extend(
        [
            "",
            "## Readiness Snapshot",
            "",
            f"- QA status: `{qa_status}`",
            f"- QA pass / review / fail: {qa_pass} / {qa_review} / {qa_fail}",
            f"- 3D render preview: `{result.render_preview_png_path or 'not_written'}`",
            f"- Vessel-visible 3D preview: `{result.vessel_visible_preview_png_path or 'not_written'}`",
            f"- Product manifest: `{result.product_manifest_path}`",
            "",
            "## Top QA Blockers",
            "",
        ]
    )
    if result.qa_blockers:
        lines.extend(
            [
                "| status | check | category | value | threshold | notes |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for item in result.qa_blockers:
            notes = "<br>".join(item.notes) if item.notes else ""
            lines.append(
                f"| `{item.status}` | `{item.check_id}` | {item.category} | {item.value} | {item.threshold} | {notes} |"
            )
    else:
        lines.append("- No fail/review QA blockers were recorded.")
    lines.extend(
        [
            "",
            "## User-Facing Outputs",
            "",
            f"- Build manifest: `{result.build_manifest_path or 'not_written'}`",
            f"- QA YAML: `{result.qa_yaml_path or 'not_written'}`",
            f"- 3D scene spec: `{result.render_scene_spec_path or 'not_written'}`",
            f"- Vessel-visible preview PNG: `{result.vessel_visible_preview_png_path or 'not_written'}`",
            f"- Product report: `{result.report_path}`",
            "",
            "## Recommended Next Actions",
            "",
        ]
    )
    lines.extend(f"- {action}" for action in result.recommended_next_actions)
    lines.extend(
        [
            "",
            "## Product Limitations",
            "",
            "- This is a research/engineering demonstrator package, not a cleared clinical device or patient-care output.",
            "- Anatomical claims depend on the quality and registration of input CT, organ labels, and vessel labels.",
            "- Flow outputs are digital model outputs and still need calibration/validation against measured pressure or flow data.",
            "- RT outputs are workflow/QA demonstrators unless connected to a validated dose engine or TPS workflow.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_product_case(
    *,
    input_ct_path: str | Path | None = None,
    input_cta_path: str | Path | None = None,
    input_ctv_path: str | Path | None = None,
    organ_seg_path: str | Path | None = None,
    gi_seg_path: str | Path | None = None,
    vessel_seg_path: str | Path | None = None,
    existing_build_manifest_path: str | Path | None = None,
    patient_id: str = "patient_demo",
    case_id: str | None = None,
    output_dir: str | Path = "outputs/product_cases",
    organ_labelmap_path: str | Path = "configs/labelmaps/ct_org.yaml",
    gi_labelmap_path: str | Path = "configs/labelmaps/gi_tract.yaml",
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
    render_vessel_visible: bool = True,
    report_path: str | Path | None = None,
) -> ProductCaseResult:
    if existing_build_manifest_path is None and input_ct_path is None:
        raise ValueError("Provide either --existing-build-manifest or --input-ct.")

    if existing_build_manifest_path is not None:
        build_manifest = _load_yaml(existing_build_manifest_path)
        resolved_case_id = case_id or str(build_manifest.get("case_id", "product_case"))
        resolved_patient_id = patient_id if patient_id != "patient_demo" else str(build_manifest.get("patient_id", patient_id))
    else:
        resolved_case_id = case_id or f"{_slug(patient_id)}_product_case"
        resolved_patient_id = patient_id

    output_root = Path(output_dir) / resolved_case_id
    product_manifest = output_root / f"{resolved_case_id}_product_case_manifest_v001.yaml"
    report = Path(report_path) if report_path is not None else output_root / f"{resolved_case_id}_product_case_report_v001.md"
    stages: list[ProductCaseStage] = []
    notes = [
        "product_case_runner_orchestrates_adapter_build_qa_and_render_outputs",
        "single_command_interface_for_research_demonstrator_packaging",
    ]
    active_baseline = resolve_stage007_active_baseline()
    resolved_baseline_graph = _existing(baseline_graph_path) or active_baseline.graph_path
    resolved_baseline_combined_spec = _existing(baseline_combined_spec_path) or active_baseline.voxelized_spec_path
    if existing_build_manifest_path is None and baseline_graph_path is None and active_baseline.graph_path is not None:
        notes.append("baseline_graph_auto_resolved_from_stage007_active_baseline")
    if existing_build_manifest_path is None and baseline_combined_spec_path is None and active_baseline.voxelized_spec_path is not None:
        notes.append("baseline_reference_spec_auto_resolved_from_stage007_active_voxelized_spec")

    adapter_result: PatientPhantomAdapterResult | None = None
    build_result: PatientPhantomBuildResult | None = None
    qa_result: PatientBuildQAResult | None = None
    render_result: Render3DResult | None = None

    adapter_manifest_path: str | None = None
    build_manifest_path: str | None = None
    qa_yaml_path: str | None = None
    render_preview_png_path: str | None = None
    render_scene_spec_path: str | None = None
    vessel_visible_preview_png_path: str | None = None

    if existing_build_manifest_path is None:
        adapter_dir = output_root / "adapter"
        adapter_report = output_root / "reports" / f"{resolved_case_id}_input_adapter.md"
        adapter_result = build_patient_phantom_adapter(
            input_ct_path=input_ct_path,
            input_cta_path=input_cta_path,
            input_ctv_path=input_ctv_path,
            organ_seg_path=organ_seg_path,
            gi_seg_path=gi_seg_path,
            vessel_seg_path=vessel_seg_path,
            patient_id=resolved_patient_id,
            case_id=f"{resolved_case_id}_adapter",
            adaptation_mode="hybrid",
            output_dir=adapter_dir,
            organ_labelmap_path=organ_labelmap_path,
            gi_labelmap_path=gi_labelmap_path,
            materials_path=materials_path,
            approved_set_manifest_path=approved_set_manifest_path,
            baseline_graph_path=resolved_baseline_graph,
            baseline_combined_spec_path=resolved_baseline_combined_spec,
            target_height_cm=target_height_cm,
            target_weight_kg=target_weight_kg,
            target_bmi=target_bmi,
            target_waist_cm=target_waist_cm,
            copy_inputs=copy_inputs,
            report_path=adapter_report,
        )
        adapter_manifest_path = adapter_result.manifest_yaml_path
        stages.append(
            ProductCaseStage(
                "input_adapter",
                adapter_result.overall_status,
                adapter_result.manifest_yaml_path,
                adapter_result.report_path,
                (f"inputs={adapter_result.input_count}",),
            )
        )

        build_report = output_root / "reports" / f"{resolved_case_id}_build_executor.md"
        build_result = run_patient_phantom_build(
            patient_manifest_path=adapter_result.manifest_yaml_path,
            output_dir=output_root / "build",
            case_id=resolved_case_id,
            report_path=build_report,
            organ_labelmap_path=organ_labelmap_path,
            materials_path=materials_path,
            baseline_graph_path=resolved_baseline_graph,
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
        )
        build_manifest_path = build_result.build_manifest_yaml_path
        stages.append(
            ProductCaseStage(
                "phantom_build",
                build_result.overall_status,
                build_result.build_manifest_yaml_path,
                build_result.report_path,
                tuple(f"{step.step_id}:{step.status}" for step in build_result.steps),
            )
        )
    else:
        build_manifest_path = str(existing_build_manifest_path)
        build_manifest = _load_yaml(build_manifest_path)
        adapter_manifest_path = build_manifest.get("source_patient_manifest")
        stages.append(
            ProductCaseStage(
                "existing_build",
                str(build_manifest.get("overall_status", "unknown")),
                build_manifest_path,
                _output_path(build_manifest, "report"),
                ("existing_build_manifest_supplied_no_rebuild_performed",),
            )
        )
        notes.append("existing_build_manifest_mode_skips_input_adapter_and_build_execution")

    if run_qa and build_manifest_path is not None and Path(build_manifest_path).exists():
        qa_report = output_root / "reports" / f"{resolved_case_id}_qa_gate.md"
        qa_result = qa_patient_phantom_build(
            build_manifest_path=build_manifest_path,
            output_dir=output_root / "qa",
            case_id=resolved_case_id,
            report_path=qa_report,
            expected_lumen_components=qa_expected_lumen_components,
        )
        qa_yaml_path = qa_result.qa_yaml_path
        stages.append(
            ProductCaseStage(
                "qa_gate",
                qa_result.readiness_status,
                qa_result.qa_yaml_path,
                qa_result.report_path,
                (f"pass_review_fail={qa_result.pass_count}/{qa_result.review_count}/{qa_result.fail_count}",),
            )
        )
    elif not run_qa:
        stages.append(ProductCaseStage("qa_gate", "skipped", None, None, ("qa_skipped_by_user_option",)))

    if render_3d and build_manifest_path is not None and Path(build_manifest_path).exists():
        try:
            render_inputs = _render_inputs_from_build(build_manifest_path)
            render_result = generate_vascular_network_3d_render(
                context_labels_path=render_inputs["context_labels"],
                arterial_lumen_mask_path=render_inputs["arterial_lumen_mask"],
                venous_lumen_mask_path=render_inputs["venous_lumen_mask"],
                flow_domain_labels_path=render_inputs["flow_domain_labels"],
                vessel_wall_mask_path=render_inputs["vessel_wall_mask"],
                output_dir=output_root / "render3d",
                case_id=resolved_case_id,
                formats=("stl",),
                target_max_faces=render_target_max_faces,
                vascular_graph_path=render_inputs.get("vascular_graph"),
                render_vessel_visible_preview=render_vessel_visible,
                report_path=output_root / "reports" / f"{resolved_case_id}_3d_render.md",
            )
            render_preview_png_path = render_result.preview_png_path
            render_scene_spec_path = render_result.spec_yaml_path
            vessel_visible_preview_png_path = render_result.vessel_visible_preview_png_path
            stages.append(
                ProductCaseStage(
                    "render_3d",
                    "completed",
                    render_result.preview_png_path,
                    render_result.report_path,
                    (f"mesh_groups={len(render_result.meshes)}",),
                )
            )
            if render_result.vessel_visible_preview_png_path:
                stages.append(
                    ProductCaseStage(
                        "render_vessel_visible",
                        "completed",
                        render_result.vessel_visible_preview_png_path,
                        render_result.vessel_visible_report_path,
                        ("display_enlarged_vessel_context_preview",),
                    )
                )
        except Exception as exc:
            notes.append(f"render_3d_failed={type(exc).__name__}: {exc}")
            stages.append(ProductCaseStage("render_3d", "failed", None, None, (f"{type(exc).__name__}: {exc}",)))
    elif _existing(existing_render_preview_path) is not None:
        render_preview_png_path = str(existing_render_preview_path)
        render_scene_spec_path = _existing(existing_render_scene_spec_path)
        stages.append(
            ProductCaseStage(
                "render_3d",
                "linked_existing",
                render_preview_png_path,
                render_scene_spec_path,
                ("existing_3d_render_linked_into_product_case",),
            )
        )
    elif not render_3d:
        stages.append(ProductCaseStage("render_3d", "skipped", None, None, ("render_skipped_by_user_option",)))

    qa_blockers = _extract_qa_blockers(qa_result, qa_yaml_path)
    final_status = _final_status(
        build_result=build_result,
        build_manifest_path=build_manifest_path,
        qa_result=qa_result,
        qa_yaml_path=qa_yaml_path,
        render_preview_png_path=render_preview_png_path,
    )
    actions = _recommended_actions(final_status, qa_result)
    result = ProductCaseResult(
        case_id=resolved_case_id,
        patient_id=resolved_patient_id,
        output_dir=str(output_root),
        product_manifest_path=str(product_manifest),
        report_path=str(report),
        final_status=final_status,
        adapter_manifest_path=adapter_manifest_path,
        build_manifest_path=build_manifest_path,
        qa_yaml_path=qa_yaml_path,
        render_preview_png_path=render_preview_png_path,
        render_scene_spec_path=render_scene_spec_path,
        vessel_visible_preview_png_path=vessel_visible_preview_png_path,
        stages=tuple(stages),
        qa_blockers=qa_blockers,
        recommended_next_actions=actions,
        notes=tuple(notes),
    )
    _write_product_manifest(product_manifest, result)
    _write_product_report(report, result, qa_result=qa_result, qa_yaml_path=qa_yaml_path)
    return result


def format_product_case_result(result: ProductCaseResult) -> str:
    lines = [
        "Phantom Twin Product Case",
        f"Case ID: {result.case_id}",
        f"Patient/profile ID: {result.patient_id}",
        f"Final status: {result.final_status}",
        f"Product manifest: {result.product_manifest_path}",
        f"Report: {result.report_path}",
    ]
    if result.render_preview_png_path:
        lines.append(f"3D preview PNG: {result.render_preview_png_path}")
    if result.vessel_visible_preview_png_path:
        lines.append(f"Vessel-visible 3D preview PNG: {result.vessel_visible_preview_png_path}")
    for stage in result.stages:
        lines.append(f"- {stage.stage_id}: {stage.status}")
    return "\n".join(lines)
