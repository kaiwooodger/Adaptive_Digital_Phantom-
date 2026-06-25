from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import os
from typing import Any

import yaml

from .anthropometry import build_anthropometric_torso_morph
from .dose_gamma_qa import build_dose_gamma_qa
from .flow_1d import build_flow_1d_model
from .flow_boundary import build_flow_boundary_package
from .flow_coupled import build_coupled_pulsatile_flow_model
from .radiotherapy import build_radiotherapy_qa_package
from .rt_planning import analyze_spatial_rt_flow_coupling, build_rt_planning_bundle, build_spatial_rt_flow_dose_model
from .vascular_network import deform_vascular_graph_for_variant
from .vascular_voxelize import voxelize_vascular_network


@dataclass(frozen=True)
class ProfileSweepTarget:
    profile_id: str
    target_bmi: float
    target_waist_cm: float
    target_height_cm: float


@dataclass(frozen=True)
class ProfileSweepRow:
    profile_id: str
    case_id: str
    target_bmi: float
    target_waist_cm: float
    target_height_cm: float
    morph_mode: str
    xy_padding_voxels: int
    achieved_waist_cm: float
    waist_error_cm: float
    body_volume_l: float
    body_volume_change_percent: float
    morph_vascular_components: int
    mean_node_displacement_mm: float
    max_node_displacement_mm: float
    arterial_venous_overlap_before_cleanup: int
    arterial_venous_overlap_after_cleanup: int
    connected_components: int
    arterial_components: int
    venous_components: int
    boundary_count: int
    mapped_boundary_count: int
    aortic_flow_mean_ml_s: float
    aortic_flow_min_ml_s: float
    aortic_flow_max_ml_s: float
    aortic_pressure_mean_mmhg: float
    max_outlet_split_range_pp: float
    max_mass_balance_residual_ml_s: float
    selected_spatial_edges: int
    max_peak_delta_mgy: float
    max_trough_delta_mgy: float
    ptv_static_mean_gy: float
    ptv_peak_mean_gy: float
    ptv_peak_v95_percent: float
    gamma_min_pass_rate_percent: float
    gamma_max_p95: float
    anatomy_status: str
    vascular_status: str
    flow_status: str
    rt_status: str
    overall_status: str
    morph_preview_png: str
    vascular_preview_png: str
    flow_preview_png: str
    spatial_dose_preview_png: str
    gamma_preview_png: str
    profile_root: str


@dataclass(frozen=True)
class ProfileSweepResult:
    sweep_id: str
    output_dir: str
    manifest_yaml_path: str
    metrics_csv_path: str
    atlas_png_path: str
    report_path: str
    profile_count: int
    pass_count: int
    warn_count: int
    fail_count: int
    rows: tuple[ProfileSweepRow, ...]
    notes: tuple[str, ...]


DEFAULT_PROFILE_SPECS = (
    "bmi22_waist85_height175:22:85:175",
    "bmi27_waist95_height175:27:95:175",
    "bmi32_waist110_height175:32:110:175",
    "bmi38_waist125_height175:38:125:175",
)


def _slug(raw: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "profile"


def parse_profile_sweep_target(raw: str) -> ProfileSweepTarget:
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError("Profile specs must use 'profile_id:bmi:waist_cm:height_cm'")
    profile_id, bmi, waist, height = parts
    return ProfileSweepTarget(
        profile_id=_slug(profile_id),
        target_bmi=float(bmi),
        target_waist_cm=float(waist),
        target_height_cm=float(height),
    )


def default_profile_sweep_targets() -> tuple[ProfileSweepTarget, ...]:
    return tuple(parse_profile_sweep_target(item) for item in DEFAULT_PROFILE_SPECS)


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


def _metric_from_csv(metrics_csv_path: str | Path, mask_id: str, state: str, field: str) -> float:
    for row in _read_csv_rows(metrics_csv_path):
        if row.get("mask_id") == mask_id and row.get("state") == state:
            return _as_float(row.get(field))
    return 0.0


def _gamma_summary(gamma_result) -> tuple[float, float]:
    pass_rates = [item.pass_rate_percent for item in gamma_result.state_results]
    p95_values = [item.p95_gamma for item in gamma_result.state_results]
    return (min(pass_rates) if pass_rates else 0.0, max(p95_values) if p95_values else 0.0)


def _status_rank(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2}.get(status, 2)


def _overall_status(*statuses: str) -> str:
    return max(statuses, key=_status_rank)


def _anatomy_status(waist_error_cm: float, vascular_components: int) -> str:
    if vascular_components != 1:
        return "fail"
    if abs(waist_error_cm) <= 3.0:
        return "pass"
    if abs(waist_error_cm) <= 8.0:
        return "warn"
    return "fail"


def _vascular_status(connected_components: int, arterial_components: int, venous_components: int, overlap_after_cleanup: int) -> str:
    if connected_components != 1 or arterial_components != 1 or venous_components != 1:
        return "fail"
    if overlap_after_cleanup != 0:
        return "fail"
    return "pass"


def _flow_status(boundary_count: int, mapped_boundary_count: int, mass_balance_residual: float) -> str:
    if boundary_count == 0 or mapped_boundary_count < boundary_count:
        return "fail"
    if mass_balance_residual > 1e-6:
        return "fail"
    return "pass"


def _rt_status(selected_edges: int, gamma_min_pass: float, ptv_peak_v95: float) -> str:
    if selected_edges <= 0 or gamma_min_pass < 90.0:
        return "fail"
    if gamma_min_pass < 95.0 or ptv_peak_v95 < 80.0:
        return "warn"
    return "pass"


def _write_metrics_csv(path: Path, rows: tuple[ProfileSweepRow, ...]) -> None:
    fieldnames = list(ProfileSweepRow.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def _write_manifest(path: Path, result: ProfileSweepResult) -> None:
    payload = {
        "sweep_id": result.sweep_id,
        "package_type": "profile_anthropometric_sweep",
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "metrics_csv": result.metrics_csv_path,
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
        },
        "summary": {
            "profile_count": result.profile_count,
            "pass_count": result.pass_count,
            "warn_count": result.warn_count,
            "fail_count": result.fail_count,
        },
        "profiles": [
            {
                "profile_id": row.profile_id,
                "case_id": row.case_id,
                "target_bmi": row.target_bmi,
                "target_waist_cm": row.target_waist_cm,
                "target_height_cm": row.target_height_cm,
                "morph_mode": row.morph_mode,
                "xy_padding_voxels": row.xy_padding_voxels,
                "achieved_waist_cm": row.achieved_waist_cm,
                "body_volume_l": row.body_volume_l,
                "overall_status": row.overall_status,
                "profile_root": row.profile_root,
                "previews": {
                    "morph": row.morph_preview_png,
                    "vascular": row.vascular_preview_png,
                    "flow": row.flow_preview_png,
                    "spatial_dose": row.spatial_dose_preview_png,
                    "gamma": row.gamma_preview_png,
                },
            }
            for row in result.rows
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Profile sweep atlas generation requires matplotlib.") from exc
    return plt


def _write_atlas(path: Path, rows: tuple[ProfileSweepRow, ...], sweep_id: str) -> None:
    plt = _import_plotting()
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("Morph", "morph_preview_png"),
        ("Vascular Domains", "vascular_preview_png"),
        ("Pulsatile Flow", "flow_preview_png"),
        ("Spatial RT-Flow Dose", "spatial_dose_preview_png"),
        ("Gamma QA", "gamma_preview_png"),
    ]
    fig, axes = plt.subplots(max(len(rows), 1), len(columns), figsize=(20, max(4.2 * len(rows), 4.2)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"Anthropometric Profile Sweep\n{sweep_id}", fontsize=18, fontweight="bold")
    for row_index, row in enumerate(rows):
        for col_index, (title, field) in enumerate(columns):
            ax = axes[row_index, col_index]
            ax.axis("off")
            if row_index == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            path_value = str(getattr(row, field))
            if path_value and Path(path_value).exists():
                ax.imshow(plt.imread(path_value))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=11, color="#7f8c8d")
            if col_index == 0:
                label = (
                    f"{row.profile_id}\n"
                    f"BMI {row.target_bmi:.0f}, waist {row.target_waist_cm:.0f} cm\n"
                    f"{row.morph_mode}, pad {row.xy_padding_voxels} vox/side\n"
                    f"achieved {row.achieved_waist_cm:.1f} cm | {row.overall_status.upper()}"
                )
                ax.text(
                    0.02,
                    0.98,
                    label,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8.5,
                    bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.35"},
                )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, result: ProfileSweepResult) -> None:
    atlas_link = os.path.relpath(result.atlas_png_path, start=Path(result.report_path).parent)
    lines = [
        "# Anthropometric Profile Sweep Stage 001",
        "",
        f"Sweep ID: `{result.sweep_id}`",
        "",
        f"![Profile sweep atlas]({atlas_link})",
        "",
        "## Summary",
        "",
        f"- Profiles evaluated: {result.profile_count}",
        f"- Pass / warn / fail: {result.pass_count} / {result.warn_count} / {result.fail_count}",
        "- Each profile was propagated through torso morphing, vascular graph deformation, vascular voxelization, flow boundary setup, steady + coupled pulsatile flow, RT package generation, spatial RT-flow dose modeling, and sampled PyMedPhys gamma QA.",
        "",
        "## Operating Envelope Table",
        "",
        "| profile | morph mode | target BMI | target waist cm | achieved waist cm | body L | anatomy | vascular | flow | RT | overall | PTV peak V95 % | gamma min pass % |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.profile_id} | {row.morph_mode} | {row.target_bmi:.1f} | {row.target_waist_cm:.1f} | {row.achieved_waist_cm:.1f} | "
            f"{row.body_volume_l:.2f} | {row.anatomy_status} | {row.vascular_status} | {row.flow_status} | "
            f"{row.rt_status} | {row.overall_status} | {row.ptv_peak_v95_percent:.2f} | {row.gamma_min_pass_rate_percent:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Key Trends",
            "",
        ]
    )
    passing = [row for row in result.rows if row.overall_status == "pass"]
    if passing:
        low = min(passing, key=lambda row: row.target_waist_cm)
        high = max(passing, key=lambda row: row.target_waist_cm)
        lines.append(f"- Current pass envelope spans approximately waist {low.target_waist_cm:.1f}-{high.target_waist_cm:.1f} cm across the tested profiles.")
    else:
        lines.append("- No profiles fully passed the current guardrails.")
    warnings = [row for row in result.rows if row.overall_status == "warn"]
    failures = [row for row in result.rows if row.overall_status == "fail"]
    if warnings:
        lines.append("- Warning profiles should be reviewed before being used for claims: " + ", ".join(row.profile_id for row in warnings) + ".")
    if failures:
        lines.append("- Failed profiles require additional morph/vascular tuning before use: " + ", ".join(row.profile_id for row in failures) + ".")
    lines.extend(
        [
            "- This is a digital engineering validation sweep, not a clinical population validation.",
            "",
            "## Output Files",
            "",
            f"- Manifest YAML: `{result.manifest_yaml_path}`",
            f"- Metrics CSV: `{result.metrics_csv_path}`",
            f"- Atlas PNG: `{result.atlas_png_path}`",
            "",
            "## Notes",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _run_profile(
    target: ProfileSweepTarget,
    sweep_id: str,
    output_dir: Path,
    combined_spec_path: str | Path,
    baseline_graph_path: str | Path,
    baseline_labels_path: str | Path,
    materials_path: str | Path,
    baseline_height_cm: float,
    baseline_bmi: float,
    arterial_inlet_flow_ml_s: float,
    export_dicom: bool,
    gamma_random_subset: int | None,
    high_bmi_waist_threshold_cm: float,
    high_bmi_xy_padding_voxels: int,
    padding_transition_margin_cm: float,
) -> ProfileSweepRow:
    case_id = f"{sweep_id}_{target.profile_id}"
    profile_root = output_dir / "profiles" / target.profile_id
    morph_mode = "high-bmi" if target.target_waist_cm >= high_bmi_waist_threshold_cm else "standard"
    padding_threshold_cm = high_bmi_waist_threshold_cm - max(0.0, float(padding_transition_margin_cm))
    xy_padding_voxels = (
        max(0, int(high_bmi_xy_padding_voxels)) if target.target_waist_cm >= padding_threshold_cm else 0
    )

    morph = build_anthropometric_torso_morph(
        combined_spec_path=combined_spec_path,
        output_dir=profile_root / "anthropometry",
        case_id=case_id,
        target_height_cm=target.target_height_cm,
        target_bmi=target.target_bmi,
        target_waist_cm=target.target_waist_cm,
        baseline_height_cm=baseline_height_cm,
        baseline_bmi=baseline_bmi,
        morph_mode=morph_mode,
        xy_padding_voxels=xy_padding_voxels,
        report_path=profile_root / "reports" / f"{case_id}_anthropometric_morph.md",
    )
    graph = deform_vascular_graph_for_variant(
        baseline_graph_path=baseline_graph_path,
        baseline_labels_path=baseline_labels_path,
        variant_labels_path=morph.morphed_blood_material_labels_path,
        output_dir=profile_root / "vascular_graph",
        case_id=case_id,
        variant_id=target.profile_id,
        report_path=profile_root / "reports" / f"{case_id}_vascular_graph_deformation.md",
    )
    voxel = voxelize_vascular_network(
        graph_yaml_path=graph.graph_yaml_path,
        combined_labels_path=morph.morphed_blood_material_labels_path,
        materials_path=materials_path,
        output_dir=profile_root / "vascular_network_voxelized",
        case_id=case_id,
        body_mask_path=morph.morphed_body_mask_path,
        sample_step_mm=0.75,
        vessel_wall_thickness_mm=2.0,
        contrast_mode="arterial",
        collision_cleanup="nearest-centerline",
        clip_to_body=True,
        write_material_volumes=True,
        report_path=profile_root / "reports" / f"{case_id}_vascular_network_voxelized.md",
    )
    boundary = build_flow_boundary_package(
        voxelized_spec_path=voxel.spec_yaml_path,
        graph_yaml_path=graph.graph_yaml_path,
        output_dir=profile_root / "flow_boundary_conditions",
        case_id=case_id,
        arterial_inlet_flow_ml_s=arterial_inlet_flow_ml_s,
        nominal_outlet_pressure_drop_pa=8000.0,
        venous_outlet_pressure_pa=667.0,
        boundary_slab_thickness_mm=5.0,
        report_path=profile_root / "reports" / f"{case_id}_flow_boundary_conditions.md",
    )
    flow_1d = build_flow_1d_model(
        graph_yaml_path=graph.graph_yaml_path,
        boundary_config_path=boundary.config_yaml_path,
        output_dir=profile_root / "flow_1d",
        case_id=case_id,
        blood_viscosity_cp=3.5,
        arterial_inlet_pressure_pa=13332.0,
        report_path=profile_root / "reports" / f"{case_id}_flow_1d_model.md",
    )
    flow = build_coupled_pulsatile_flow_model(
        flow_1d_model_path=flow_1d.model_yaml_path,
        boundary_config_path=boundary.config_yaml_path,
        output_dir=profile_root / "flow_coupled_pulsatile",
        case_id=case_id,
        heart_rate_bpm=60.0,
        samples_per_cycle=160,
        settling_cycles=3,
        rcr_proximal_resistance_fraction=0.1,
        rcr_time_constant_s=1.2,
        venous_pulsatility_fraction=0.35,
        venous_phase_lag_fraction=0.15,
        report_path=profile_root / "reports" / f"{case_id}_flow_coupled_pulsatile.md",
    )
    rt_package = build_radiotherapy_qa_package(
        combined_spec_path=voxel.spec_yaml_path,
        output_dir=profile_root / "rt_qa_package",
        case_id=case_id,
        scenario="blood",
        target_radius_mm=12.0,
        ptv_margin_mm=5.0,
        report_path=profile_root / "reports" / f"{case_id}_radiotherapy_qa_package.md",
    )
    rt_plan = build_rt_planning_bundle(
        rt_package_spec_path=rt_package.package_spec_yaml_path,
        coupled_flow_model_path=flow.model_yaml_path,
        output_dir=profile_root / "rt_planning_bundle",
        case_id=case_id,
        prescription_dose_gy=20.0,
        vascular_dose_sensitivity=0.015,
        export_dicom=export_dicom,
        report_path=profile_root / "reports" / f"{case_id}_rt_planning_bundle.md",
    )
    coupling = analyze_spatial_rt_flow_coupling(
        rt_package_spec_path=rt_package.package_spec_yaml_path,
        rt_planning_spec_path=rt_plan.bundle_spec_yaml_path,
        vascular_graph_path=graph.graph_yaml_path,
        edge_timeseries_csv_path=flow.edge_timeseries_csv_path,
        output_dir=profile_root / "spatial_rt_flow_coupling",
        case_id=case_id,
        sample_step_mm=2.0,
        influence_radius_mm=25.0,
        coordinate_mode="voxel-mm",
        report_path=profile_root / "reports" / f"{case_id}_spatial_rt_flow_coupling.md",
    )
    spatial_dose = build_spatial_rt_flow_dose_model(
        rt_package_spec_path=rt_package.package_spec_yaml_path,
        rt_planning_spec_path=rt_plan.bundle_spec_yaml_path,
        vascular_graph_path=graph.graph_yaml_path,
        edge_timeseries_csv_path=flow.edge_timeseries_csv_path,
        edge_coupling_csv_path=coupling.edge_coupling_csv_path,
        output_dir=profile_root / "spatial_rt_flow_dose",
        case_id=case_id,
        sample_step_mm=2.0,
        influence_falloff_mm=7.5,
        max_fractional_perturbation=0.05,
        max_edges=12,
        min_coupling_score=0.0,
        coordinate_mode="voxel-mm",
        report_path=profile_root / "reports" / f"{case_id}_spatial_rt_flow_dose.md",
    )
    gamma = build_dose_gamma_qa(
        pymedphys_eval_config_path=spatial_dose.pymedphys_eval_config_yaml_path,
        output_dir=profile_root / "dose_gamma_qa",
        case_id=f"{case_id}_spatial_flow_dose",
        random_subset=gamma_random_subset,
        random_seed=20260526,
        write_volume_outputs=False,
        report_path=profile_root / "reports" / f"{case_id}_spatial_flow_dose_gamma_qa.md",
    )

    gamma_min_pass, gamma_max_p95 = _gamma_summary(gamma)
    ptv_static_mean = _metric_from_csv(
        spatial_dose.dose_metrics_csv_path,
        "target_ptv_synthetic_vertebral",
        "static",
        "mean_dose_gy",
    )
    ptv_peak_mean = _metric_from_csv(
        spatial_dose.dose_metrics_csv_path,
        "target_ptv_synthetic_vertebral",
        "spatial_pulsatile_peak",
        "mean_dose_gy",
    )
    ptv_peak_v95 = _metric_from_csv(
        spatial_dose.dose_metrics_csv_path,
        "target_ptv_synthetic_vertebral",
        "spatial_pulsatile_peak",
        "v95_percent",
    )
    waist_error = morph.achieved_waist_cm - target.target_waist_cm
    anatomy_status = _anatomy_status(waist_error, morph.vascular_components)
    vascular_status = _vascular_status(
        voxel.connected_components,
        voxel.arterial_components,
        voxel.venous_components,
        voxel.overlap_voxels_after_cleanup,
    )
    flow_status = _flow_status(boundary.boundary_count, boundary.mapped_boundary_count, flow.max_abs_mass_balance_residual_ml_s)
    rt_status = _rt_status(spatial_dose.selected_edge_count, gamma_min_pass, ptv_peak_v95)
    overall = _overall_status(anatomy_status, vascular_status, flow_status, rt_status)
    return ProfileSweepRow(
        profile_id=target.profile_id,
        case_id=case_id,
        target_bmi=target.target_bmi,
        target_waist_cm=target.target_waist_cm,
        target_height_cm=target.target_height_cm,
        morph_mode=morph.morph_mode,
        xy_padding_voxels=morph.xy_padding_voxels,
        achieved_waist_cm=morph.achieved_waist_cm,
        waist_error_cm=waist_error,
        body_volume_l=morph.morphed_body_volume_cm3 / 1000.0,
        body_volume_change_percent=morph.body_volume_change_percent,
        morph_vascular_components=morph.vascular_components,
        mean_node_displacement_mm=graph.mean_node_displacement_mm,
        max_node_displacement_mm=graph.max_node_displacement_mm,
        arterial_venous_overlap_before_cleanup=voxel.overlap_voxels_before_cleanup,
        arterial_venous_overlap_after_cleanup=voxel.overlap_voxels_after_cleanup,
        connected_components=voxel.connected_components,
        arterial_components=voxel.arterial_components,
        venous_components=voxel.venous_components,
        boundary_count=boundary.boundary_count,
        mapped_boundary_count=boundary.mapped_boundary_count,
        aortic_flow_mean_ml_s=flow.arterial_inlet_flow_mean_ml_s,
        aortic_flow_min_ml_s=flow.arterial_inlet_flow_min_ml_s,
        aortic_flow_max_ml_s=flow.arterial_inlet_flow_max_ml_s,
        aortic_pressure_mean_mmhg=flow.aorta_pressure_mean_pa / 133.32236842105263,
        max_outlet_split_range_pp=flow.max_outlet_split_range_percentage_points,
        max_mass_balance_residual_ml_s=flow.max_abs_mass_balance_residual_ml_s,
        selected_spatial_edges=spatial_dose.selected_edge_count,
        max_peak_delta_mgy=1000.0 * spatial_dose.max_abs_peak_delta_gy,
        max_trough_delta_mgy=1000.0 * spatial_dose.max_abs_trough_delta_gy,
        ptv_static_mean_gy=ptv_static_mean,
        ptv_peak_mean_gy=ptv_peak_mean,
        ptv_peak_v95_percent=ptv_peak_v95,
        gamma_min_pass_rate_percent=gamma_min_pass,
        gamma_max_p95=gamma_max_p95,
        anatomy_status=anatomy_status,
        vascular_status=vascular_status,
        flow_status=flow_status,
        rt_status=rt_status,
        overall_status=overall,
        morph_preview_png=morph.preview_png_path,
        vascular_preview_png=voxel.preview_png_path,
        flow_preview_png=flow.plot_paths[0] if flow.plot_paths else "",
        spatial_dose_preview_png=spatial_dose.preview_png_path,
        gamma_preview_png=gamma.preview_png_path,
        profile_root=str(profile_root),
    )


def build_profile_sweep(
    output_dir: str | Path = "outputs/experiments/profile_sweep",
    sweep_id: str = "ct_org_profile_sweep_stage001",
    profile_specs: tuple[str, ...] | None = None,
    combined_spec_path: str | Path = "outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml",
    baseline_graph_path: str | Path = "outputs/digital/vascular_network/ct_org_case0_imagetbad_case125_vascular_network_graph_v001.yaml",
    baseline_labels_path: str | Path = "outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_material_labels_blood_v001.nii.gz",
    materials_path: str | Path = "configs/materials.yaml",
    baseline_height_cm: float = 170.0,
    baseline_bmi: float = 24.0,
    arterial_inlet_flow_ml_s: float = 80.0,
    export_dicom: bool = False,
    gamma_random_subset: int | None = 25000,
    high_bmi_waist_threshold_cm: float = 115.0,
    high_bmi_xy_padding_voxels: int = 96,
    padding_transition_margin_cm: float = 5.0,
    report_path: str | Path | None = "outputs/reports/profile_sweep_stage001.md",
) -> ProfileSweepResult:
    output = Path(output_dir)
    targets = tuple(parse_profile_sweep_target(item) for item in profile_specs) if profile_specs else default_profile_sweep_targets()
    if not targets:
        raise ValueError("At least one profile target is required")

    rows = tuple(
        _run_profile(
            target=target,
            sweep_id=sweep_id,
            output_dir=output,
            combined_spec_path=combined_spec_path,
            baseline_graph_path=baseline_graph_path,
            baseline_labels_path=baseline_labels_path,
            materials_path=materials_path,
            baseline_height_cm=baseline_height_cm,
            baseline_bmi=baseline_bmi,
            arterial_inlet_flow_ml_s=arterial_inlet_flow_ml_s,
            export_dicom=export_dicom,
            gamma_random_subset=gamma_random_subset,
            high_bmi_waist_threshold_cm=high_bmi_waist_threshold_cm,
            high_bmi_xy_padding_voxels=high_bmi_xy_padding_voxels,
            padding_transition_margin_cm=padding_transition_margin_cm,
        )
        for target in targets
    )

    manifest = output / f"{sweep_id}_profile_sweep_manifest_v001.yaml"
    metrics = output / f"{sweep_id}_profile_sweep_metrics_v001.csv"
    atlas = output / f"{sweep_id}_profile_sweep_atlas_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{sweep_id}_profile_sweep_report_v001.md"
    notes = (
        "sweep_runs_target_specific_morph_flow_and_rt_pipeline_for_each_profile",
        "rt_planning_dicom_export_is_disabled_by_default_to_control_disk_usage",
        "gamma_volume_outputs_are_disabled_by_default_for_sweep_summary_runs",
        f"profiles_with_target_waist_at_or_above_{high_bmi_waist_threshold_cm:.1f}_cm_use_high_bmi_calibration_mode",
        f"profiles_with_target_waist_at_or_above_{high_bmi_waist_threshold_cm - max(0.0, float(padding_transition_margin_cm)):.1f}_cm_use_xy_padding_{int(high_bmi_xy_padding_voxels)}_voxels_per_side_to_smooth_the_transition",
        "profile_sweep_is_an_engineering_operating_envelope_not_population_validation",
    )
    result = ProfileSweepResult(
        sweep_id=sweep_id,
        output_dir=str(output),
        manifest_yaml_path=str(manifest),
        metrics_csv_path=str(metrics),
        atlas_png_path=str(atlas),
        report_path=str(report),
        profile_count=len(rows),
        pass_count=sum(1 for row in rows if row.overall_status == "pass"),
        warn_count=sum(1 for row in rows if row.overall_status == "warn"),
        fail_count=sum(1 for row in rows if row.overall_status == "fail"),
        rows=rows,
        notes=notes,
    )
    _write_metrics_csv(metrics, rows)
    _write_atlas(atlas, rows, sweep_id)
    _write_manifest(manifest, result)
    _write_report(report, result)
    return result


def format_profile_sweep_result(result: ProfileSweepResult) -> str:
    return "\n".join(
        [
            "Profile sweep completed",
            f"Sweep ID: {result.sweep_id}",
            f"Profiles: {result.profile_count}",
            f"Pass/warn/fail: {result.pass_count}/{result.warn_count}/{result.fail_count}",
            f"Manifest YAML: {result.manifest_yaml_path}",
            f"Metrics CSV: {result.metrics_csv_path}",
            f"Atlas PNG: {result.atlas_png_path}",
            f"Report: {result.report_path}",
        ]
    )
