from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import yaml


@dataclass(frozen=True)
class ProfileRerunComparisonResult:
    case_id: str
    profile_id: str
    output_dir: str
    atlas_png_path: str
    spec_yaml_path: str
    report_path: str
    summary: dict[str, Any]
    figure_paths: dict[str, str]
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


def _spatial_metrics_csv(spatial_spec: dict[str, Any]) -> str:
    outputs = spatial_spec.get("outputs", {})
    if not isinstance(outputs, dict):
        return ""
    return str(outputs.get("dose_metrics_csv", ""))


def _spatial_metric(metrics_csv: str | Path | None, mask_id: str, state: str, field: str) -> float:
    for row in _read_csv_rows(metrics_csv):
        if row.get("mask_id") == mask_id and row.get("state") == state:
            return _as_float(row.get(field))
    return 0.0


def _gamma_summary(gamma_spec: dict[str, Any]) -> dict[str, float]:
    states = gamma_spec.get("state_results", [])
    if not isinstance(states, list):
        states = []
    pass_rates = [_as_float(state.get("pass_rate_percent")) for state in states if isinstance(state, dict)]
    p95_values = [_as_float(state.get("p95_gamma")) for state in states if isinstance(state, dict)]
    max_values = [_as_float(state.get("max_gamma_value")) for state in states if isinstance(state, dict)]
    return {
        "state_count": float(len(states)),
        "min_pass_rate_percent": min(pass_rates) if pass_rates else 0.0,
        "max_p95_gamma": max(p95_values) if p95_values else 0.0,
        "max_gamma": max(max_values) if max_values else 0.0,
    }


def _vascular_summary(vascular_spec: dict[str, Any]) -> dict[str, float]:
    voxelization = vascular_spec.get("voxelization", {})
    if not isinstance(voxelization, dict):
        voxelization = {}
    return {
        "connected_components": float(_as_int(voxelization.get("connected_components"))),
        "arterial_components": float(_as_int(voxelization.get("arterial_components"))),
        "venous_components": float(_as_int(voxelization.get("venous_components"))),
        "overlap_before_cleanup": float(_as_int(voxelization.get("arterial_venous_overlap_voxels_before_cleanup"))),
        "overlap_after_cleanup": float(_as_int(voxelization.get("arterial_venous_overlap_voxels_after_cleanup"))),
        "outside_body_fraction_before_clip": _as_float(voxelization.get("outside_body_fraction_before_clip")),
    }


def _flow_summary(flow_spec: dict[str, Any]) -> dict[str, float]:
    summary = flow_spec.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return {
        "node_count": float(_as_int(summary.get("node_count"))),
        "edge_count": float(_as_int(summary.get("edge_count"))),
        "boundary_count": float(_as_int(summary.get("boundary_count"))),
        "aortic_flow_mean_ml_s": _as_float(summary.get("arterial_inlet_flow_mean_ml_s")),
        "aortic_flow_min_ml_s": _as_float(summary.get("arterial_inlet_flow_min_ml_s")),
        "aortic_flow_max_ml_s": _as_float(summary.get("arterial_inlet_flow_max_ml_s")),
        "aortic_pressure_mean_mmhg": _as_float(summary.get("aorta_pressure_mean_pa")) / 133.32236842105263,
        "max_outlet_split_range_pp": _as_float(summary.get("max_outlet_split_range_percentage_points")),
        "mass_balance_residual_ml_s": _as_float(summary.get("max_abs_mass_balance_residual_ml_s")),
        "pressure_residual_pa": _as_float(summary.get("max_abs_pressure_equation_residual_pa")),
    }


def _rt_summary(spatial_spec: dict[str, Any], gamma_spec: dict[str, Any]) -> dict[str, float]:
    model = spatial_spec.get("dose_model", {})
    if not isinstance(model, dict):
        model = {}
    metrics_csv = _spatial_metrics_csv(spatial_spec)
    gamma = _gamma_summary(gamma_spec)
    return {
        "selected_edge_count": float(_as_int(model.get("selected_edge_count"))),
        "peak_phase": _as_float(model.get("peak_phase")),
        "trough_phase": _as_float(model.get("trough_phase")),
        "max_peak_delta_mgy": 1000.0 * _as_float(model.get("max_abs_peak_delta_gy")),
        "max_trough_delta_mgy": 1000.0 * _as_float(model.get("max_abs_trough_delta_gy")),
        "ptv_static_mean_gy": _spatial_metric(metrics_csv, "target_ptv_synthetic_vertebral", "static", "mean_dose_gy"),
        "ptv_peak_mean_gy": _spatial_metric(
            metrics_csv,
            "target_ptv_synthetic_vertebral",
            "spatial_pulsatile_peak",
            "mean_dose_gy",
        ),
        "ptv_peak_v95_percent": _spatial_metric(
            metrics_csv,
            "target_ptv_synthetic_vertebral",
            "spatial_pulsatile_peak",
            "v95_percent",
        ),
        "gamma_min_pass_rate_percent": gamma["min_pass_rate_percent"],
        "gamma_max_p95": gamma["max_p95_gamma"],
        "gamma_max": gamma["max_gamma"],
    }


def _delta(profile: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    keys = set(profile) | set(reference)
    return {key: profile.get(key, 0.0) - reference.get(key, 0.0) for key in sorted(keys)}


def _build_summary(
    profile_adapter_spec_path: str | Path | None,
    anthropometric_spec_path: str | Path | None,
    reference_vascular_spec_path: str | Path | None,
    profile_vascular_spec_path: str | Path | None,
    reference_flow_spec_path: str | Path | None,
    profile_flow_spec_path: str | Path | None,
    reference_spatial_dose_spec_path: str | Path | None,
    profile_spatial_dose_spec_path: str | Path | None,
    reference_gamma_spec_path: str | Path | None,
    profile_gamma_spec_path: str | Path | None,
) -> dict[str, Any]:
    profile_adapter = _load_yaml(profile_adapter_spec_path)
    anthropometric = _load_yaml(anthropometric_spec_path)
    reference_vascular = _vascular_summary(_load_yaml(reference_vascular_spec_path))
    profile_vascular = _vascular_summary(_load_yaml(profile_vascular_spec_path))
    reference_flow = _flow_summary(_load_yaml(reference_flow_spec_path))
    profile_flow = _flow_summary(_load_yaml(profile_flow_spec_path))
    reference_rt = _rt_summary(_load_yaml(reference_spatial_dose_spec_path), _load_yaml(reference_gamma_spec_path))
    profile_rt = _rt_summary(_load_yaml(profile_spatial_dose_spec_path), _load_yaml(profile_gamma_spec_path))

    anthropometry = anthropometric.get("anthropometry", {}) if isinstance(anthropometric.get("anthropometry", {}), dict) else {}
    quality = anthropometric.get("quality_summary", {}) if isinstance(anthropometric.get("quality_summary", {}), dict) else {}
    selection = profile_adapter.get("selection", {}) if isinstance(profile_adapter.get("selection", {}), dict) else {}
    target = profile_adapter.get("target", {}) if isinstance(profile_adapter.get("target", {}), dict) else {}
    return {
        "profile": {
            "target_bmi": _as_float(target.get("bmi", anthropometry.get("target_bmi"))),
            "target_waist_cm": _as_float(target.get("waist_cm", anthropometry.get("target_waist_cm"))),
            "target_height_cm": _as_float(target.get("height_cm", anthropometry.get("target_height_cm"))),
            "selected_pca_variant": str(selection.get("selected_variant_id", "")),
            "profile_fit_status": str(selection.get("fit_status", "")),
        },
        "anthropometry": {
            "baseline_waist_cm": _as_float(anthropometry.get("baseline_waist_cm")),
            "target_waist_cm": _as_float(anthropometry.get("target_waist_cm")),
            "achieved_waist_cm": _as_float(anthropometry.get("achieved_waist_cm")),
            "body_radial_scale": _as_float(anthropometry.get("body_radial_scale")),
            "height_scale": _as_float(anthropometry.get("height_scale")),
            "baseline_body_volume_l": _as_float(quality.get("baseline_body_volume_cm3")) / 1000.0,
            "morphed_body_volume_l": _as_float(quality.get("morphed_body_volume_cm3")) / 1000.0,
            "body_volume_change_percent": _as_float(quality.get("body_volume_change_percent")),
            "vascular_components_after_morph": float(_as_int(quality.get("vascular_components"))),
        },
        "reference": {
            "vascular": reference_vascular,
            "flow": reference_flow,
            "rt": reference_rt,
        },
        "profile_rerun": {
            "vascular": profile_vascular,
            "flow": profile_flow,
            "rt": profile_rt,
        },
        "delta_profile_minus_reference": {
            "vascular": _delta(profile_vascular, reference_vascular),
            "flow": _delta(profile_flow, reference_flow),
            "rt": _delta(profile_rt, reference_rt),
        },
    }


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Profile rerun comparison atlas generation requires matplotlib.") from exc
    return plt


def _write_atlas_png(path: Path, case_id: str, profile_id: str, figure_paths: dict[str, str], summary: dict[str, Any]) -> None:
    plt = _import_plotting()
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Profile Fit", figure_paths.get("profile_adapter_preview", "")),
        ("Target Morph", figure_paths.get("anthropometric_preview", "")),
        ("Vascular Graph", figure_paths.get("vascular_graph_preview", "")),
        ("Voxelized Flow Domains", figure_paths.get("vascular_voxel_preview", "")),
        ("Pulsatile Flow", figure_paths.get("flow_preview", "")),
        ("Spatial RT-Flow Coupling", figure_paths.get("spatial_coupling_preview", "")),
        ("Spatial RT-Flow Dose", figure_paths.get("spatial_dose_preview", "")),
        ("Gamma QA", figure_paths.get("gamma_qa_preview", "")),
    ]
    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(3, 3)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
    ]
    summary_ax = fig.add_subplot(grid[2, 2])
    fig.suptitle(f"Profile-Specific Phantom Rerun Comparison\n{profile_id} | {case_id}", fontsize=18, fontweight="bold")
    for ax, (title, image_path) in zip(axes, panels, strict=False):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        if image_path and Path(image_path).exists():
            ax.imshow(plt.imread(image_path))
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12, color="#7f8c8d")

    profile = summary["profile"]
    anthro = summary["anthropometry"]
    flow = summary["profile_rerun"]["flow"]
    rt = summary["profile_rerun"]["rt"]
    delta_flow = summary["delta_profile_minus_reference"]["flow"]
    delta_rt = summary["delta_profile_minus_reference"]["rt"]
    vascular = summary["profile_rerun"]["vascular"]
    text_lines = [
        "Profile Snapshot",
        "",
        f"BMI / waist: {profile['target_bmi']:.1f} / {profile['target_waist_cm']:.1f} cm",
        f"Achieved waist: {anthro['achieved_waist_cm']:.1f} cm",
        f"Body volume: {anthro['baseline_body_volume_l']:.1f} -> {anthro['morphed_body_volume_l']:.1f} L",
        f"Body change: {anthro['body_volume_change_percent']:.1f}%",
        f"Fit status: {profile['profile_fit_status'] or 'direct morph'}",
        "",
        f"Vascular A/V overlap: {vascular['overlap_after_cleanup']:.0f} voxels",
        f"Aortic flow mean: {flow['aortic_flow_mean_ml_s']:.1f} mL/s",
        f"Outlet split range: {flow['max_outlet_split_range_pp']:.3f} pp",
        f"Split delta vs ref: {delta_flow['max_outlet_split_range_pp']:+.3f} pp",
        "",
        f"Max peak dose delta: {rt['max_peak_delta_mgy']:.1f} mGy",
        f"Peak delta vs ref: {delta_rt['max_peak_delta_mgy']:+.1f} mGy",
        f"PTV peak V95: {rt['ptv_peak_v95_percent']:.1f}%",
        f"Gamma min pass: {rt['gamma_min_pass_rate_percent']:.1f}%",
    ]
    summary_ax.axis("off")
    summary_ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
        bbox={"facecolor": "#f7f9fb", "edgecolor": "#ccd6dd", "boxstyle": "round,pad=0.6"},
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _write_spec(path: Path, result: ProfileRerunComparisonResult) -> None:
    payload = {
        "case_id": result.case_id,
        "profile_id": result.profile_id,
        "package_type": "profile_rerun_comparison_atlas",
        "outputs": {
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
            "spec_yaml": result.spec_yaml_path,
        },
        "figure_paths": result.figure_paths,
        "summary": result.summary,
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _metric_table(reference: dict[str, float], profile: dict[str, float], fields: list[tuple[str, str, str]]) -> list[str]:
    lines = ["| metric | reference | profile | delta |", "| --- | ---: | ---: | ---: |"]
    delta = _delta(profile, reference)
    for key, label, fmt in fields:
        ref_value = reference.get(key, 0.0)
        prof_value = profile.get(key, 0.0)
        delta_value = delta.get(key, 0.0)
        lines.append(f"| {label} | {format(ref_value, fmt)} | {format(prof_value, fmt)} | {format(delta_value, '+' + fmt)} |")
    return lines


def _format_report(result: ProfileRerunComparisonResult) -> str:
    summary = result.summary
    profile = summary["profile"]
    anthro = summary["anthropometry"]
    reference = summary["reference"]
    profile_rerun = summary["profile_rerun"]
    atlas_link = os.path.relpath(result.atlas_png_path, start=Path(result.report_path).parent)
    lines = [
        "# Profile-Specific Phantom Rerun Comparison",
        "",
        f"Case ID: `{result.case_id}`",
        f"Profile ID: `{result.profile_id}`",
        "",
        f"![Profile rerun comparison atlas]({atlas_link})",
        "",
        "## Profile Result",
        "",
        f"- Target BMI/waist/height: {profile['target_bmi']:.2f} / {profile['target_waist_cm']:.2f} cm / {profile['target_height_cm']:.1f} cm",
        f"- PCA nearest-neighbor selection: `{profile['selected_pca_variant'] or 'not used directly'}`",
        f"- Profile adapter fit status: `{profile['profile_fit_status'] or 'direct_morph_rerun'}`",
        f"- Achieved waist proxy: {anthro['achieved_waist_cm']:.2f} cm",
        f"- Body volume: {anthro['baseline_body_volume_l']:.3f} L -> {anthro['morphed_body_volume_l']:.3f} L ({anthro['body_volume_change_percent']:.2f}%)",
        f"- Vascular components after morph: {anthro['vascular_components_after_morph']:.0f}",
        "",
        "## Vascular Comparison",
        "",
    ]
    lines.extend(
        _metric_table(
            reference["vascular"],
            profile_rerun["vascular"],
            [
                ("connected_components", "Connected lumen components", ".0f"),
                ("arterial_components", "Arterial components", ".0f"),
                ("venous_components", "Venous components", ".0f"),
                ("overlap_before_cleanup", "A/V overlap before cleanup voxels", ".0f"),
                ("overlap_after_cleanup", "A/V overlap after cleanup voxels", ".0f"),
            ],
        )
    )
    lines.extend(["", "## Flow Comparison", ""])
    lines.extend(
        _metric_table(
            reference["flow"],
            profile_rerun["flow"],
            [
                ("aortic_flow_mean_ml_s", "Aortic flow mean mL/s", ".3f"),
                ("aortic_flow_min_ml_s", "Aortic flow min mL/s", ".3f"),
                ("aortic_flow_max_ml_s", "Aortic flow max mL/s", ".3f"),
                ("aortic_pressure_mean_mmhg", "Aortic pressure mean mmHg", ".3f"),
                ("max_outlet_split_range_pp", "Max outlet split range pp", ".3f"),
                ("mass_balance_residual_ml_s", "Max mass-balance residual mL/s", ".3e"),
            ],
        )
    )
    lines.extend(["", "## RT-Flow QA Comparison", ""])
    lines.extend(
        _metric_table(
            reference["rt"],
            profile_rerun["rt"],
            [
                ("selected_edge_count", "Selected spatial edges", ".0f"),
                ("max_peak_delta_mgy", "Max peak perturbation mGy", ".3f"),
                ("max_trough_delta_mgy", "Max trough perturbation mGy", ".3f"),
                ("ptv_static_mean_gy", "PTV static mean Gy", ".4f"),
                ("ptv_peak_mean_gy", "PTV peak mean Gy", ".4f"),
                ("ptv_peak_v95_percent", "PTV peak V95 %", ".3f"),
                ("gamma_min_pass_rate_percent", "Gamma min pass rate %", ".3f"),
                ("gamma_max_p95", "Gamma max p95", ".6f"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The BMI/waist-specific morph has now been propagated through vascular deformation, voxelization, pulsatile flow, spatial RT-flow dose, and gamma QA.",
            "- The adapted case is still a digital engineering phantom, not a subject-specific clinical anatomy; the profile adapter showed this anthropometry was outside the current PCA release envelope, so the direct morph rerun is the more relevant adapted geometry.",
            "- The main observed profile-specific RT-flow change is a larger peak spatial perturbation while maintaining 100% sampled gamma pass rate under the current synthetic criteria.",
            "",
            "## Figure Sources",
            "",
        ]
    )
    for label, path in result.figure_paths.items():
        if path:
            lines.append(f"- `{label}`: `{path}`")
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def build_profile_rerun_comparison_atlas(
    output_dir: str | Path = "outputs/reports/profile_rerun_comparison",
    case_id: str = "ct_org_label_population8_bmi32_waist110_height175",
    profile_id: str = "bmi32_waist110_height175",
    profile_adapter_spec_path: str | Path = "outputs/digital/user_profile_adapter/ct_org_label_population8_bmi32_waist110_height175_bmi32_waist110_height175_profile_adapter_v001.yaml",
    anthropometric_spec_path: str | Path = "outputs/digital/anthropometric_morph/bmi32_waist110_height175/ct_org_label_population8_bmi32_waist110_height175_anthro_morph_spec_v001.yaml",
    reference_vascular_spec_path: str | Path = "outputs/digital/variant_flow/mode01_pos/vascular_network_voxelized/ct_org_label_population8_pca_modes_stage001_mode01_pos_vascular_network_voxelized_spec_v001.yaml",
    profile_vascular_spec_path: str | Path = "outputs/digital/profile_flow/bmi32_waist110_height175/vascular_network_voxelized/ct_org_label_population8_bmi32_waist110_height175_vascular_network_voxelized_spec_v001.yaml",
    reference_flow_spec_path: str | Path = "outputs/sim/variant_flow/mode01_pos/flow_coupled_pulsatile/ct_org_label_population8_pca_modes_stage001_mode01_pos_coupled_pulsatile_flow_model_v001.yaml",
    profile_flow_spec_path: str | Path = "outputs/sim/profile_flow/bmi32_waist110_height175/flow_coupled_pulsatile/ct_org_label_population8_bmi32_waist110_height175_coupled_pulsatile_flow_model_v001.yaml",
    reference_spatial_dose_spec_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/mode01_pos/ct_org_label_population8_pca_modes_stage001_mode01_pos_rt_spatial_flow_dose_model_spec_v001.yaml",
    profile_spatial_dose_spec_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/bmi32_waist110_height175/ct_org_label_population8_bmi32_waist110_height175_rt_spatial_flow_dose_model_spec_v001.yaml",
    reference_gamma_spec_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/mode01_pos/dose_gamma_qa/ct_org_label_population8_pca_modes_stage001_mode01_pos_spatial_flow_dose_gamma_qa_spec_v001.yaml",
    profile_gamma_spec_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/bmi32_waist110_height175/dose_gamma_qa/ct_org_label_population8_bmi32_waist110_height175_spatial_flow_dose_dose_gamma_qa_spec_v001.yaml",
    profile_adapter_preview_path: str | Path = "outputs/digital/user_profile_adapter/ct_org_label_population8_bmi32_waist110_height175_bmi32_waist110_height175_profile_adapter_preview_v001.png",
    anthropometric_preview_path: str | Path = "outputs/digital/anthropometric_morph/bmi32_waist110_height175/ct_org_label_population8_bmi32_waist110_height175_anthro_morph_preview_v001.png",
    vascular_graph_preview_path: str | Path = "outputs/digital/profile_flow/bmi32_waist110_height175/vascular_graph/ct_org_label_population8_bmi32_waist110_height175_variant_deformed_vascular_graph_preview_v001.png",
    vascular_voxel_preview_path: str | Path = "outputs/digital/profile_flow/bmi32_waist110_height175/vascular_network_voxelized/ct_org_label_population8_bmi32_waist110_height175_vascular_network_voxelized_preview_v001.png",
    flow_preview_path: str | Path = "outputs/sim/profile_flow/bmi32_waist110_height175/flow_coupled_pulsatile/plots/ct_org_label_population8_bmi32_waist110_height175_coupled_pulsatile_pressure_flow_preview_v001.png",
    spatial_coupling_preview_path: str | Path = "outputs/experiments/spatial_rt_flow_coupling/bmi32_waist110_height175/ct_org_label_population8_bmi32_waist110_height175_spatial_rt_flow_coupling_preview_v001.png",
    spatial_dose_preview_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/bmi32_waist110_height175/ct_org_label_population8_bmi32_waist110_height175_rt_spatial_flow_dose_model_preview_v001.png",
    gamma_qa_preview_path: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose/bmi32_waist110_height175/dose_gamma_qa/ct_org_label_population8_bmi32_waist110_height175_spatial_flow_dose_dose_gamma_qa_preview_v001.png",
    report_path: str | Path | None = "outputs/reports/profile_rerun_comparison_stage001.md",
) -> ProfileRerunComparisonResult:
    output = Path(output_dir)
    atlas_png = output / f"{case_id}_profile_rerun_comparison_atlas_v001.png"
    spec_yaml = output / f"{case_id}_profile_rerun_comparison_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_profile_rerun_comparison_report_v001.md"
    figure_paths = {
        "profile_adapter_preview": _existing_path(profile_adapter_preview_path),
        "anthropometric_preview": _existing_path(anthropometric_preview_path),
        "vascular_graph_preview": _existing_path(vascular_graph_preview_path),
        "vascular_voxel_preview": _existing_path(vascular_voxel_preview_path),
        "flow_preview": _existing_path(flow_preview_path),
        "spatial_coupling_preview": _existing_path(spatial_coupling_preview_path),
        "spatial_dose_preview": _existing_path(spatial_dose_preview_path),
        "gamma_qa_preview": _existing_path(gamma_qa_preview_path),
    }
    summary = _build_summary(
        profile_adapter_spec_path=profile_adapter_spec_path,
        anthropometric_spec_path=anthropometric_spec_path,
        reference_vascular_spec_path=reference_vascular_spec_path,
        profile_vascular_spec_path=profile_vascular_spec_path,
        reference_flow_spec_path=reference_flow_spec_path,
        profile_flow_spec_path=profile_flow_spec_path,
        reference_spatial_dose_spec_path=reference_spatial_dose_spec_path,
        profile_spatial_dose_spec_path=profile_spatial_dose_spec_path,
        reference_gamma_spec_path=reference_gamma_spec_path,
        profile_gamma_spec_path=profile_gamma_spec_path,
    )
    notes = (
        "comparison_uses_mode01_pos_variant_flow_as_current_reference_for_flow_and_rt_metrics",
        "anthropometric_deltas_are_from_the_source_combined_phantom_to_the_target_profile_morph",
        "profile_rerun_outputs_reference_existing_artifacts_without_copying_large_nifti_volumes",
        "phantom_is_research_surrogate_not_subject_specific_clinical_twin",
    )
    result = ProfileRerunComparisonResult(
        case_id=case_id,
        profile_id=profile_id,
        output_dir=str(output),
        atlas_png_path=str(atlas_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        summary=summary,
        figure_paths=figure_paths,
        notes=notes,
    )
    _write_atlas_png(atlas_png, case_id, profile_id, figure_paths, summary)
    _write_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_profile_rerun_comparison_result(result: ProfileRerunComparisonResult) -> str:
    anthro = result.summary["anthropometry"]
    rt = result.summary["profile_rerun"]["rt"]
    flow = result.summary["profile_rerun"]["flow"]
    return "\n".join(
        [
            "Profile rerun comparison atlas created",
            f"Case ID: {result.case_id}",
            f"Profile ID: {result.profile_id}",
            f"Achieved waist: {anthro['achieved_waist_cm']:.2f} cm",
            f"Body volume change: {anthro['body_volume_change_percent']:.2f}%",
            f"Aortic flow mean: {flow['aortic_flow_mean_ml_s']:.3f} mL/s",
            f"Max peak RT-flow perturbation: {rt['max_peak_delta_mgy']:.3f} mGy",
            f"Gamma min pass rate: {rt['gamma_min_pass_rate_percent']:.3f}%",
            f"Atlas PNG: {result.atlas_png_path}",
            f"Spec YAML: {result.spec_yaml_path}",
            f"Report: {result.report_path}",
        ]
    )
