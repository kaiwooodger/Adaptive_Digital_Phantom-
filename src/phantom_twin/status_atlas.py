from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import yaml


@dataclass(frozen=True)
class PhantomStatusAtlasResult:
    case_id: str
    output_dir: str
    atlas_png_path: str
    report_path: str
    spec_yaml_path: str
    summary: dict[str, Any]
    figure_paths: dict[str, str]
    notes: tuple[str, ...]


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Status atlas generation requires matplotlib.") from exc
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


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    return int(round(_as_float(value, float(default))))


def _existing_path(path: str | Path | None) -> str:
    if path is None or str(path) == "":
        return ""
    resolved = Path(path)
    return str(resolved) if resolved.exists() else ""


def _rows_by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows}


def _spatial_metric(metrics_csv: str | Path | None, mask_id: str, state: str, field: str) -> float:
    for row in _read_csv_rows(metrics_csv):
        if row.get("mask_id") == mask_id and row.get("state") == state:
            return _as_float(row.get(field))
    return 0.0


def _build_summary(
    combined_spec_path: str | Path | None,
    pca_metrics_csv_path: str | Path | None,
    approved_manifest_path: str | Path | None,
    vascular_voxel_spec_path: str | Path | None,
    flow_model_spec_path: str | Path | None,
    spatial_dose_spec_path: str | Path | None,
    gamma_spec_path: str | Path | None,
    scalar_spatial_compare_spec_path: str | Path | None,
) -> dict[str, Any]:
    combined = _load_yaml(combined_spec_path)
    vascular = _load_yaml(vascular_voxel_spec_path)
    flow = _load_yaml(flow_model_spec_path)
    spatial = _load_yaml(spatial_dose_spec_path)
    gamma = _load_yaml(gamma_spec_path)
    scalar_compare = _load_yaml(scalar_spatial_compare_spec_path)
    approved = _load_yaml(approved_manifest_path)
    pca_rows = _read_csv_rows(pca_metrics_csv_path)
    spatial_outputs = spatial.get("outputs", {}) if isinstance(spatial.get("outputs", {}), dict) else {}
    spatial_metrics_csv = spatial_outputs.get("dose_metrics_csv")
    gamma_states = gamma.get("state_results", []) if isinstance(gamma.get("state_results", []), list) else []
    pass_rates = [_as_float(state.get("pass_rate_percent")) for state in gamma_states if isinstance(state, dict)]
    p95_values = [_as_float(state.get("p95_gamma")) for state in gamma_states if isinstance(state, dict)]

    flow_summary = flow.get("summary", {}) if isinstance(flow.get("summary", {}), dict) else {}
    vascular_summary = vascular.get("voxelization", {}) if isinstance(vascular.get("voxelization", {}), dict) else {}
    spatial_model = spatial.get("dose_model", {}) if isinstance(spatial.get("dose_model", {}), dict) else {}
    compare_summary = scalar_compare.get("summary", {}) if isinstance(scalar_compare.get("summary", {}), dict) else {}

    approved_variants = approved.get("approved_variants")
    if not isinstance(approved_variants, list) or not approved_variants:
        approved_variants = [
            item
            for item in approved.get("variants", [])
            if isinstance(item, dict) and str(item.get("release_role", "")).startswith("approved")
        ] if isinstance(approved.get("variants", []), list) else []

    return {
        "anatomy": {
            "combined_case_id": combined.get("case_id", "unknown"),
            "material_regions": len(combined.get("regions", [])) if isinstance(combined.get("regions", []), list) else 0,
            "flow_boundary_label_count": len(combined.get("flow_boundary_labels", []))
            if isinstance(combined.get("flow_boundary_labels", []), list)
            else 0,
        },
        "population_adaptation": {
            "pca_variant_rows": len(pca_rows),
            "approved_variant_count": len(approved_variants),
        },
        "vascular": {
            "connected_components": _as_int(vascular_summary.get("connected_components")),
            "arterial_components": _as_int(vascular_summary.get("arterial_components")),
            "venous_components": _as_int(vascular_summary.get("venous_components")),
            "arterial_venous_overlap_after_cleanup": _as_int(
                vascular_summary.get("arterial_venous_overlap_voxels_after_cleanup")
            ),
        },
        "flow": {
            "edge_count": _as_int(flow_summary.get("edge_count")),
            "node_count": _as_int(flow_summary.get("node_count")),
            "arterial_inlet_flow_mean_ml_s": _as_float(flow_summary.get("arterial_inlet_flow_mean_ml_s")),
            "arterial_inlet_flow_min_ml_s": _as_float(flow_summary.get("arterial_inlet_flow_min_ml_s")),
            "arterial_inlet_flow_max_ml_s": _as_float(flow_summary.get("arterial_inlet_flow_max_ml_s")),
            "aorta_pressure_mean_mmhg": _as_float(flow_summary.get("aorta_pressure_mean_pa")) / 133.32236842105263,
            "max_outlet_split_range_percentage_points": _as_float(
                flow_summary.get("max_outlet_split_range_percentage_points")
            ),
        },
        "radiotherapy": {
            "selected_spatial_edges": _as_int(spatial_model.get("selected_edge_count")),
            "spatial_peak_phase": _as_float(spatial_model.get("peak_phase")),
            "spatial_trough_phase": _as_float(spatial_model.get("trough_phase")),
            "max_spatial_peak_delta_mgy": 1000.0 * _as_float(spatial_model.get("max_abs_peak_delta_gy")),
            "max_spatial_trough_delta_mgy": 1000.0 * _as_float(spatial_model.get("max_abs_trough_delta_gy")),
            "ptv_static_mean_gy": _spatial_metric(spatial_metrics_csv, "target_ptv_synthetic_vertebral", "static", "mean_dose_gy"),
            "ptv_peak_mean_gy": _spatial_metric(
                spatial_metrics_csv,
                "target_ptv_synthetic_vertebral",
                "spatial_pulsatile_peak",
                "mean_dose_gy",
            ),
            "ptv_peak_v95_percent": _spatial_metric(
                spatial_metrics_csv,
                "target_ptv_synthetic_vertebral",
                "spatial_pulsatile_peak",
                "v95_percent",
            ),
        },
        "qa": {
            "gamma_min_pass_rate_percent": min(pass_rates) if pass_rates else 0.0,
            "gamma_max_p95": max(p95_values) if p95_values else 0.0,
            "gamma_state_count": len(gamma_states),
            "scalar_vs_spatial_max_abs_mgy": 1000.0 * _as_float(compare_summary.get("max_abs_volume_difference_gy")),
            "scalar_vs_spatial_ptv_peak_mean_delta_mgy": 1000.0
            * _as_float(compare_summary.get("ptv_peak_mean_delta_gy")),
            "scalar_vs_spatial_ptv_peak_v95_delta_pp": _as_float(
                compare_summary.get("ptv_peak_v95_delta_percentage_points")
            ),
        },
    }


def _write_atlas_png(path: Path, case_id: str, figure_paths: dict[str, str], summary: dict[str, Any]) -> None:
    plt = _import_plotting()
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Anatomy Volume", figure_paths.get("combined_preview", "")),
        ("PCA/Population Variants", figure_paths.get("pca_variant_atlas", "") or figure_paths.get("approved_pca_preview", "")),
        ("Voxelized Arterial/Venous Network", figure_paths.get("vascular_voxel_preview", "")),
        ("Spatial RT-Flow Coupling", figure_paths.get("spatial_coupling_preview", "")),
        ("Spatial RT-Flow Dose", figure_paths.get("spatial_dose_preview", "")),
        ("Gamma QA", figure_paths.get("gamma_qa_preview", "")),
        ("Scalar vs Spatial Validation", figure_paths.get("scalar_spatial_preview", "")),
    ]
    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.92])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[2, 0:2]),
    ]
    summary_ax = fig.add_subplot(grid[2, 2])
    fig.suptitle(f"Current Digital Phantom Status Atlas\n{case_id}", fontsize=18, fontweight="bold")
    for ax, (title, image_path) in zip(axes, panels, strict=False):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        if image_path and Path(image_path).exists():
            image = plt.imread(str(image_path))
            ax.imshow(image)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12, color="#7f8c8d")

    flow = summary.get("flow", {})
    rt = summary.get("radiotherapy", {})
    qa = summary.get("qa", {})
    vascular = summary.get("vascular", {})
    text_lines = [
        "Status Snapshot",
        "",
        f"Flow graph: {flow.get('node_count', 0)} nodes / {flow.get('edge_count', 0)} edges",
        f"Aortic flow: {flow.get('arterial_inlet_flow_mean_ml_s', 0.0):.1f} mL/s mean",
        f"Aortic range: {flow.get('arterial_inlet_flow_min_ml_s', 0.0):.1f}-{flow.get('arterial_inlet_flow_max_ml_s', 0.0):.1f} mL/s",
        f"Vascular components: A{vascular.get('arterial_components', 0)} / V{vascular.get('venous_components', 0)}",
        f"Overlap after cleanup: {vascular.get('arterial_venous_overlap_after_cleanup', 0)} voxels",
        "",
        f"Spatial dose edges: {rt.get('selected_spatial_edges', 0)}",
        f"Peak/trough phase: {rt.get('spatial_peak_phase', 0.0):.2f} / {rt.get('spatial_trough_phase', 0.0):.2f}",
        f"Max peak delta: {rt.get('max_spatial_peak_delta_mgy', 0.0):.1f} mGy",
        f"PTV peak V95: {rt.get('ptv_peak_v95_percent', 0.0):.1f}%",
        "",
        f"Gamma min pass: {qa.get('gamma_min_pass_rate_percent', 0.0):.1f}%",
        f"Max p95 gamma: {qa.get('gamma_max_p95', 0.0):.4f}",
        f"Scalar-spatial max delta: {qa.get('scalar_vs_spatial_max_abs_mgy', 0.0):.1f} mGy",
    ]
    summary_ax.axis("off")
    summary_ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        bbox={"facecolor": "#f7f9fb", "edgecolor": "#ccd6dd", "boxstyle": "round,pad=0.6"},
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _write_spec(path: Path, result: PhantomStatusAtlasResult) -> None:
    payload = {
        "case_id": result.case_id,
        "atlas_type": "current_digital_phantom_status",
        "outputs": {
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
        },
        "figure_paths": result.figure_paths,
        "summary": result.summary,
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: PhantomStatusAtlasResult) -> str:
    summary = result.summary
    anatomy = summary.get("anatomy", {})
    population = summary.get("population_adaptation", {})
    vascular = summary.get("vascular", {})
    flow = summary.get("flow", {})
    rt = summary.get("radiotherapy", {})
    qa = summary.get("qa", {})
    atlas_link = os.path.relpath(result.atlas_png_path, start=Path(result.report_path).parent)
    lines = [
        "# Current Digital Phantom Status Atlas",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        f"![Current phantom status atlas]({atlas_link})",
        "",
        "## Executive Snapshot",
        "",
        f"- Anatomy/material model: {anatomy.get('material_regions', 0)} material regions, {anatomy.get('flow_boundary_label_count', 0)} baseline flow boundary labels.",
        f"- Population adaptation: {population.get('pca_variant_rows', 0)} PCA variant rows and {population.get('approved_variant_count', 0)} approved variants in the current release set.",
        f"- Vascular domain: {vascular.get('connected_components', 0)} connected lumen component, arterial/venous overlap after cleanup = {vascular.get('arterial_venous_overlap_after_cleanup', 0)} voxels.",
        f"- Pulsatile flow: {flow.get('node_count', 0)} nodes, {flow.get('edge_count', 0)} edges, aortic flow {flow.get('arterial_inlet_flow_mean_ml_s', 0.0):.1f} mL/s mean.",
        f"- Spatial RT-flow dose: {rt.get('selected_spatial_edges', 0)} selected vessel edges; max peak perturbation {rt.get('max_spatial_peak_delta_mgy', 0.0):.1f} mGy.",
        f"- Dose QA: minimum sampled gamma pass rate {qa.get('gamma_min_pass_rate_percent', 0.0):.1f}%; max p95 gamma {qa.get('gamma_max_p95', 0.0):.4f}.",
        "",
        "## Key RT/Flow Numbers",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| Aortic flow mean | {flow.get('arterial_inlet_flow_mean_ml_s', 0.0):.3f} mL/s |",
        f"| Aortic flow min/max | {flow.get('arterial_inlet_flow_min_ml_s', 0.0):.3f} / {flow.get('arterial_inlet_flow_max_ml_s', 0.0):.3f} mL/s |",
        f"| Aortic pressure mean | {flow.get('aorta_pressure_mean_mmhg', 0.0):.3f} mmHg |",
        f"| Max outlet split range | {flow.get('max_outlet_split_range_percentage_points', 0.0):.3f} pp |",
        f"| Spatial peak phase | {rt.get('spatial_peak_phase', 0.0):.3f} |",
        f"| Spatial trough phase | {rt.get('spatial_trough_phase', 0.0):.3f} |",
        f"| PTV static mean | {rt.get('ptv_static_mean_gy', 0.0):.4f} Gy |",
        f"| PTV spatial peak mean | {rt.get('ptv_peak_mean_gy', 0.0):.4f} Gy |",
        f"| PTV spatial peak V95 | {rt.get('ptv_peak_v95_percent', 0.0):.3f}% |",
        f"| Scalar-vs-spatial max voxel delta | {qa.get('scalar_vs_spatial_max_abs_mgy', 0.0):.3f} mGy |",
        f"| Scalar-vs-spatial PTV peak mean delta | {qa.get('scalar_vs_spatial_ptv_peak_mean_delta_mgy', 0.0):.3f} mGy |",
        "",
        "## Figure Sources",
        "",
    ]
    for label, path in result.figure_paths.items():
        if path:
            lines.append(f"- `{label}`: `{path}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current phantom is now a digital torso testbed with material labels, PCA anatomy variants, a synthetic arterial/venous graph, pulsatile 1D flow, spatial RT-flow dose perturbations, and PyMedPhys QA.",
            "- It is adaptable through BMI/waist/PCA-mode inputs and variant-specific vascular deformation, but it is still a research surrogate rather than a subject-specific anatomical twin.",
            "- The newest milestone is spatial coupling: RT perturbations now follow selected vessel edges and pulsatile waveforms instead of one scalar vascular amplitude.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_current_phantom_status_atlas(
    output_dir: str | Path = "outputs/reports/status_atlas",
    case_id: str = "ct_org_label_population8_pca_modes_stage001_mode01_pos",
    combined_spec_path: str | Path = "outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml",
    combined_preview_path: str | Path = "outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_preview_v001.png",
    pca_variant_atlas_path: str | Path = "outputs/digital/pca_mode_variants/ct_org_label_population8_pca_modes_stage001_pca_mode_variant_atlas_v001.png",
    pca_metrics_csv_path: str | Path = "outputs/digital/pca_mode_variants/ct_org_label_population8_pca_modes_stage001_pca_mode_variant_metrics_v001.csv",
    approved_manifest_path: str | Path = "outputs/digital/approved_pca_phantom_set/ct_org_label_population8_pca_modes_stage001_approved_pca_phantom_set_manifest_v001.yaml",
    approved_pca_preview_path: str | Path = "outputs/digital/approved_pca_phantom_set/ct_org_label_population8_pca_modes_stage001_approved_pca_phantom_set_preview_v001.png",
    vascular_voxel_spec_path: str | Path = "outputs/digital/variant_flow/mode01_pos/vascular_network_voxelized/ct_org_label_population8_pca_modes_stage001_mode01_pos_vascular_network_voxelized_spec_v001.yaml",
    vascular_voxel_preview_path: str | Path = "outputs/digital/variant_flow/mode01_pos/vascular_network_voxelized/ct_org_label_population8_pca_modes_stage001_mode01_pos_vascular_network_voxelized_preview_v001.png",
    flow_model_spec_path: str | Path = "outputs/sim/variant_flow/mode01_pos/flow_coupled_pulsatile/ct_org_label_population8_pca_modes_stage001_mode01_pos_coupled_pulsatile_flow_model_v001.yaml",
    spatial_coupling_preview_path: str | Path = "outputs/experiments/spatial_rt_flow_coupling/mode01_pos/ct_org_label_population8_pca_modes_stage001_mode01_pos_spatial_rt_flow_coupling_preview_v001.png",
    spatial_dose_spec_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/mode01_pos/ct_org_label_population8_pca_modes_stage001_mode01_pos_rt_spatial_flow_dose_model_spec_v001.yaml",
    spatial_dose_preview_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/mode01_pos/ct_org_label_population8_pca_modes_stage001_mode01_pos_rt_spatial_flow_dose_model_preview_v001.png",
    gamma_spec_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/mode01_pos/dose_gamma_qa/ct_org_label_population8_pca_modes_stage001_mode01_pos_spatial_flow_dose_gamma_qa_spec_v001.yaml",
    gamma_qa_preview_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/mode01_pos/dose_gamma_qa/ct_org_label_population8_pca_modes_stage001_mode01_pos_spatial_flow_dose_gamma_qa_preview_v001.png",
    scalar_spatial_compare_spec_path: str | Path = "outputs/experiments/scalar_vs_spatial_rt_flow/mode01_pos/ct_org_label_population8_pca_modes_stage001_mode01_pos_scalar_vs_spatial_rt_flow_comparison_spec_v001.yaml",
    scalar_spatial_preview_path: str | Path = "outputs/experiments/scalar_vs_spatial_rt_flow/mode01_pos/ct_org_label_population8_pca_modes_stage001_mode01_pos_scalar_vs_spatial_rt_flow_comparison_preview_v001.png",
    report_path: str | Path | None = "outputs/reports/current_phantom_status_atlas_stage001.md",
) -> PhantomStatusAtlasResult:
    output = Path(output_dir)
    atlas_png = output / f"{case_id}_current_phantom_status_atlas_v001.png"
    spec_yaml = output / f"{case_id}_current_phantom_status_atlas_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_current_phantom_status_atlas_report_v001.md"
    figure_paths = {
        "combined_preview": _existing_path(combined_preview_path),
        "pca_variant_atlas": _existing_path(pca_variant_atlas_path),
        "approved_pca_preview": _existing_path(approved_pca_preview_path),
        "vascular_voxel_preview": _existing_path(vascular_voxel_preview_path),
        "spatial_coupling_preview": _existing_path(spatial_coupling_preview_path),
        "spatial_dose_preview": _existing_path(spatial_dose_preview_path),
        "gamma_qa_preview": _existing_path(gamma_qa_preview_path),
        "scalar_spatial_preview": _existing_path(scalar_spatial_preview_path),
    }
    summary = _build_summary(
        combined_spec_path=combined_spec_path,
        pca_metrics_csv_path=pca_metrics_csv_path,
        approved_manifest_path=approved_manifest_path,
        vascular_voxel_spec_path=vascular_voxel_spec_path,
        flow_model_spec_path=flow_model_spec_path,
        spatial_dose_spec_path=spatial_dose_spec_path,
        gamma_spec_path=gamma_spec_path,
        scalar_spatial_compare_spec_path=scalar_spatial_compare_spec_path,
    )
    notes = (
        "status_atlas_references_existing_outputs_without_recomputing_large_volumes",
        "current_build_focuses_on_mode01_pos_variant_specific_flow_and_spatial_rt_coupling",
        "phantom_is_research_surrogate_not_subject_specific_anatomical_twin",
    )
    result = PhantomStatusAtlasResult(
        case_id=case_id,
        output_dir=str(output),
        atlas_png_path=str(atlas_png),
        report_path=str(report),
        spec_yaml_path=str(spec_yaml),
        summary=summary,
        figure_paths=figure_paths,
        notes=notes,
    )
    _write_atlas_png(atlas_png, case_id, figure_paths, summary)
    _write_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_current_phantom_status_atlas_result(result: PhantomStatusAtlasResult) -> str:
    qa = result.summary.get("qa", {})
    flow = result.summary.get("flow", {})
    rt = result.summary.get("radiotherapy", {})
    return "\n".join(
        [
            "Current phantom status atlas created",
            f"Case ID: {result.case_id}",
            f"Aortic flow mean: {flow.get('arterial_inlet_flow_mean_ml_s', 0.0):.3f} mL/s",
            f"Spatial RT-flow selected edges: {rt.get('selected_spatial_edges', 0)}",
            f"Gamma min pass rate: {qa.get('gamma_min_pass_rate_percent', 0.0):.3f}%",
            f"Atlas PNG: {result.atlas_png_path}",
            f"Spec YAML: {result.spec_yaml_path}",
            f"Report: {result.report_path}",
        ]
    )
