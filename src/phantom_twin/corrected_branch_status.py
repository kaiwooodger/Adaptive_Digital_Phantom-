from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import yaml


@dataclass(frozen=True)
class CorrectedBranchStatusResult:
    case_id: str
    output_dir: str
    report_path: str
    manifest_yaml_path: str
    metrics_csv_path: str
    atlas_png_path: str
    summary: dict[str, Any]
    artifact_paths: dict[str, str]
    notes: tuple[str, ...]


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Corrected branch status report generation requires matplotlib.") from exc
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
    return str(path) if Path(path).exists() else ""


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _mask_by_id(rt_spec: dict[str, Any], mask_id: str) -> dict[str, Any]:
    masks = rt_spec.get("masks", [])
    if not isinstance(masks, list):
        return {}
    for mask in masks:
        if isinstance(mask, dict) and str(mask.get("mask_id", "")) == mask_id:
            return mask
    return {}


def _metric_row(rows: list[dict[str, str]], mask_id: str, state: str) -> dict[str, str]:
    for row in rows:
        if row.get("mask_id") == mask_id and row.get("state") == state:
            return row
    return {}


def _first_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def _artifact_size(path: str | Path | None) -> int:
    if path is None or str(path) == "":
        return 0
    resolved = Path(path)
    return resolved.stat().st_size if resolved.exists() and resolved.is_file() else 0


def _artifact_status(path: str | Path | None) -> str:
    return "exists" if path and Path(path).exists() else "missing"


def _build_summary(
    *,
    vessel_flow_manifest_path: str | Path,
    vessel_flow_spec_path: str | Path,
    rt_package_spec_path: str | Path,
    coupled_flow_model_path: str | Path,
    flow4d_spec_path: str | Path,
    spatial_coupling_csv_path: str | Path,
    spatial_dose_spec_path: str | Path,
    gamma_spec_path: str | Path,
) -> dict[str, Any]:
    vessel_manifest = _load_yaml(vessel_flow_manifest_path)
    vessel_spec = _load_yaml(vessel_flow_spec_path)
    rt_spec = _load_yaml(rt_package_spec_path)
    flow_model = _load_yaml(coupled_flow_model_path)
    flow4d = _load_yaml(flow4d_spec_path)
    spatial_dose = _load_yaml(spatial_dose_spec_path)
    gamma = _load_yaml(gamma_spec_path)
    coupling_rows = _read_csv_rows(spatial_coupling_csv_path)
    spatial_metrics_csv = _nested(spatial_dose, "outputs", "dose_metrics_csv")
    spatial_metrics = _read_csv_rows(spatial_metrics_csv)
    gamma_states = gamma.get("state_results", []) if isinstance(gamma.get("state_results", []), list) else []
    pass_rates = [_as_float(state.get("pass_rate_percent")) for state in gamma_states if isinstance(state, dict)]
    p95_values = [_as_float(state.get("p95_gamma")) for state in gamma_states if isinstance(state, dict)]

    vessel_metrics = vessel_manifest.get("metrics", {}) if isinstance(vessel_manifest.get("metrics"), dict) else {}
    voxelization = vessel_spec.get("voxelization", {}) if isinstance(vessel_spec.get("voxelization"), dict) else {}
    vascular_fluid = _mask_by_id(rt_spec, "vascular_fluid")
    vessel_wall = _mask_by_id(rt_spec, "vessel_wall")
    ptv = _mask_by_id(rt_spec, "target_ptv_synthetic_vertebral")
    flow_summary = flow_model.get("summary", {}) if isinstance(flow_model.get("summary"), dict) else {}
    dose_model = spatial_dose.get("dose_model", {}) if isinstance(spatial_dose.get("dose_model"), dict) else {}
    top_coupling = _first_row(coupling_rows)
    ptv_static = _metric_row(spatial_metrics, "target_ptv_synthetic_vertebral", "static")
    ptv_peak = _metric_row(spatial_metrics, "target_ptv_synthetic_vertebral", "spatial_pulsatile_peak")
    ptv_trough = _metric_row(spatial_metrics, "target_ptv_synthetic_vertebral", "spatial_pulsatile_trough")
    vascular_peak = _metric_row(spatial_metrics, "vascular_fluid", "spatial_pulsatile_peak")

    return {
        "case_id": vessel_spec.get("case_id") or vessel_manifest.get("case_id") or flow_model.get("case_id"),
        "vascular_domain": {
            "arterial_voxels": _as_int(vessel_metrics.get("arterial_voxels")),
            "venous_voxels": _as_int(vessel_metrics.get("venous_voxels")),
            "vessel_wall_voxels": _as_int(vessel_metrics.get("vessel_wall_voxels")),
            "arterial_components": _as_int(voxelization.get("arterial_components")),
            "venous_components": _as_int(voxelization.get("venous_components")),
            "connected_components": _as_int(voxelization.get("connected_components")),
            "snapped_boundary_nodes": _as_int(vessel_metrics.get("snapped_boundary_node_count")),
            "unclassified_labels": vessel_metrics.get("unclassified_labels", []),
        },
        "rt_material_package": {
            "vascular_fluid_volume_cm3": _as_float(vascular_fluid.get("volume_cm3")),
            "vascular_fluid_voxels": _as_int(vascular_fluid.get("voxel_count")),
            "vessel_wall_volume_cm3": _as_float(vessel_wall.get("volume_cm3")),
            "vessel_wall_voxels": _as_int(vessel_wall.get("voxel_count")),
            "ptv_volume_cm3": _as_float(ptv.get("volume_cm3")),
            "ptv_mean_hu": _as_float(ptv.get("mean_hu")),
        },
        "flow": {
            "edge_count": _as_int(flow_summary.get("edge_count")),
            "node_count": _as_int(flow_summary.get("node_count")),
            "boundary_count": _as_int(flow_summary.get("boundary_count")),
            "aorta_flow_mean_ml_s": _as_float(flow_summary.get("arterial_inlet_flow_mean_ml_s")),
            "aorta_flow_min_ml_s": _as_float(flow_summary.get("arterial_inlet_flow_min_ml_s")),
            "aorta_flow_max_ml_s": _as_float(flow_summary.get("arterial_inlet_flow_max_ml_s")),
            "aorta_pressure_mean_mmhg": _as_float(flow_summary.get("aorta_pressure_mean_pa")) / 133.32236842105263,
            "max_mass_balance_residual_ml_s": _as_float(flow_summary.get("max_abs_mass_balance_residual_ml_s")),
            "max_outlet_split_range_pp": _as_float(flow_summary.get("max_outlet_split_range_percentage_points")),
        },
        "flow4d": {
            "frame_count": len(flow4d.get("frames", [])) if isinstance(flow4d.get("frames"), list) else 0,
            "color_by": flow4d.get("color_by", ""),
            "color_min": _as_float(_nested(flow4d, "color_range", "min")),
            "color_max": _as_float(_nested(flow4d, "color_range", "max")),
            "animation_gif": _nested(flow4d, "outputs", "animation_gif"),
            "contact_sheet_png": _nested(flow4d, "outputs", "contact_sheet_png"),
        },
        "rt_flow": {
            "top_coupled_edge": top_coupling.get("edge_id", ""),
            "top_coupled_edge_score": _as_float(top_coupling.get("coupling_score")),
            "top_coupled_edge_ptv_distance_mm": _as_float(top_coupling.get("effective_distance_to_ptv_mm")),
            "selected_edge_count": _as_int(dose_model.get("selected_edge_count")),
            "peak_phase": _as_float(dose_model.get("peak_phase")),
            "trough_phase": _as_float(dose_model.get("trough_phase")),
            "max_peak_delta_mgy": 1000.0 * _as_float(dose_model.get("max_abs_peak_delta_gy")),
            "max_trough_delta_mgy": 1000.0 * _as_float(dose_model.get("max_abs_trough_delta_gy")),
            "ptv_static_d95_gy": _as_float(ptv_static.get("d95_gy")),
            "ptv_peak_d95_gy": _as_float(ptv_peak.get("d95_gy")),
            "ptv_peak_v95_percent": _as_float(ptv_peak.get("v95_percent")),
            "ptv_trough_d95_gy": _as_float(ptv_trough.get("d95_gy")),
            "vascular_peak_mean_gy": _as_float(vascular_peak.get("mean_dose_gy")),
        },
        "gamma": {
            "state_count": len(gamma_states),
            "min_pass_rate_percent": min(pass_rates) if pass_rates else 0.0,
            "max_p95_gamma": max(p95_values) if p95_values else 0.0,
            "max_gamma": max((_as_float(state.get("max_gamma_value")) for state in gamma_states if isinstance(state, dict)), default=0.0),
            "criteria": {
                "dose_percent": _as_float(_nested(gamma, "gamma_settings", "dose_percent_threshold")),
                "distance_mm": _as_float(_nested(gamma, "gamma_settings", "distance_mm_threshold")),
                "sampled_points": _as_int(_nested(gamma, "gamma_settings", "random_subset")),
            },
        },
    }


def _write_metrics_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[tuple[str, str, Any, str]] = [
        ("vascular", "arterial_voxels", _nested(summary, "vascular_domain", "arterial_voxels"), "voxels"),
        ("vascular", "venous_voxels", _nested(summary, "vascular_domain", "venous_voxels"), "voxels"),
        ("vascular", "rt_vascular_fluid_volume", _nested(summary, "rt_material_package", "vascular_fluid_volume_cm3"), "cm3"),
        ("vascular", "rt_vessel_wall_volume", _nested(summary, "rt_material_package", "vessel_wall_volume_cm3"), "cm3"),
        ("flow", "aorta_flow_mean", _nested(summary, "flow", "aorta_flow_mean_ml_s"), "mL/s"),
        ("flow", "aorta_flow_min", _nested(summary, "flow", "aorta_flow_min_ml_s"), "mL/s"),
        ("flow", "aorta_flow_max", _nested(summary, "flow", "aorta_flow_max_ml_s"), "mL/s"),
        ("flow", "mass_balance_residual", _nested(summary, "flow", "max_mass_balance_residual_ml_s"), "mL/s"),
        ("rt_flow", "selected_edges", _nested(summary, "rt_flow", "selected_edge_count"), "edges"),
        ("rt_flow", "max_peak_delta", _nested(summary, "rt_flow", "max_peak_delta_mgy"), "mGy"),
        ("rt_flow", "max_trough_delta", _nested(summary, "rt_flow", "max_trough_delta_mgy"), "mGy"),
        ("rt_flow", "ptv_peak_d95", _nested(summary, "rt_flow", "ptv_peak_d95_gy"), "Gy"),
        ("rt_flow", "ptv_peak_v95", _nested(summary, "rt_flow", "ptv_peak_v95_percent"), "%"),
        ("gamma", "min_pass_rate", _nested(summary, "gamma", "min_pass_rate_percent"), "%"),
        ("gamma", "max_p95_gamma", _nested(summary, "gamma", "max_p95_gamma"), "gamma"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("category", "metric", "value", "unit"))
        for category, metric, value, unit in rows:
            writer.writerow((category, metric, value, unit))


def _write_manifest(path: Path, result: CorrectedBranchStatusResult) -> None:
    payload = {
        "case_id": result.case_id,
        "report_type": "corrected_branch_labelled_phantom_status",
        "outputs": {
            "report": result.report_path,
            "manifest_yaml": result.manifest_yaml_path,
            "metrics_csv": result.metrics_csv_path,
            "atlas_png": result.atlas_png_path,
        },
        "artifacts": {
            name: {
                "path": artifact_path,
                "status": _artifact_status(artifact_path),
                "size_bytes": _artifact_size(artifact_path),
            }
            for name, artifact_path in result.artifact_paths.items()
        },
        "summary": result.summary,
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_atlas_png(path: Path, case_id: str, artifact_paths: dict[str, str], summary: dict[str, Any]) -> None:
    plt = _import_plotting()
    panels = [
        ("Corrected Vessel Flow Domain", artifact_paths.get("flow_domain_preview", "")),
        ("Corrected RT QA Package", artifact_paths.get("rt_qa_preview", "")),
        ("4D Flow Contact Sheet", artifact_paths.get("flow4d_contact_sheet", "")),
        ("Coupled Flow Waveforms", artifact_paths.get("coupled_flow_preview", "")),
        ("Spatial RT-Flow Dose", artifact_paths.get("spatial_dose_preview", "")),
        ("Gamma QA", artifact_paths.get("gamma_qa_preview", "")),
    ]
    fig = plt.figure(figsize=(18, 13), facecolor="#f7f4ee")
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.82])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    summary_ax = fig.add_subplot(grid[2, :])
    fig.suptitle(f"Corrected Branch-Labelled Phantom Status\n{case_id}", fontsize=18, fontweight="bold", color="#17212b")
    for ax, (title, image_path) in zip(axes, panels, strict=False):
        ax.set_title(title, fontsize=12, fontweight="bold", color="#17212b")
        ax.axis("off")
        if image_path and Path(image_path).exists() and Path(image_path).suffix.lower() == ".png":
            ax.imshow(plt.imread(str(image_path)))
        else:
            ax.text(0.5, 0.5, "missing preview", ha="center", va="center", color="#8a6f47", fontsize=12)

    vascular = summary.get("rt_material_package", {})
    flow = summary.get("flow", {})
    rt = summary.get("rt_flow", {})
    gamma = summary.get("gamma", {})
    flow4d = summary.get("flow4d", {})
    text = (
        f"Corrected vascular fluid: {vascular.get('vascular_fluid_volume_cm3', 0.0):.1f} cm3; "
        f"vessel wall: {vascular.get('vessel_wall_volume_cm3', 0.0):.1f} cm3\n"
        f"Flow graph: {flow.get('node_count', 0)} nodes / {flow.get('edge_count', 0)} edges; "
        f"aorta flow {flow.get('aorta_flow_mean_ml_s', 0.0):.1f} "
        f"({flow.get('aorta_flow_min_ml_s', 0.0):.1f}-{flow.get('aorta_flow_max_ml_s', 0.0):.1f}) mL/s\n"
        f"4D render: {flow4d.get('frame_count', 0)} frames colored by {flow4d.get('color_by', 'n/a')}; "
        f"RT-flow selected edges: {rt.get('selected_edge_count', 0)}; "
        f"top edge: {rt.get('top_coupled_edge', 'n/a')}\n"
        f"Spatial dose max peak/trough delta: {rt.get('max_peak_delta_mgy', 0.0):.1f} / "
        f"{rt.get('max_trough_delta_mgy', 0.0):.1f} mGy; "
        f"PTV peak D95/V95: {rt.get('ptv_peak_d95_gy', 0.0):.3f} Gy / {rt.get('ptv_peak_v95_percent', 0.0):.1f}%\n"
        f"Gamma QA: min pass {gamma.get('min_pass_rate_percent', 0.0):.1f}% at "
        f"{gamma.get('criteria', {}).get('dose_percent', 0.0):.0f}%/{gamma.get('criteria', {}).get('distance_mm', 0.0):.0f} mm; "
        f"max p95 gamma {gamma.get('max_p95_gamma', 0.0):.4f}"
    )
    summary_ax.axis("off")
    summary_ax.text(
        0.02,
        0.92,
        text,
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        color="#17212b",
        bbox={"facecolor": "#fffdf8", "edgecolor": "#d0c4ad", "boxstyle": "round,pad=0.7"},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_link(path: str, report_path: Path) -> str:
    if not path:
        return "`missing`"
    rel = os.path.relpath(path, start=report_path.parent)
    return f"[`{Path(path).name}`]({rel})"


def _format_report(result: CorrectedBranchStatusResult) -> str:
    summary = result.summary
    vascular = summary.get("vascular_domain", {})
    rt_pkg = summary.get("rt_material_package", {})
    flow = summary.get("flow", {})
    flow4d = summary.get("flow4d", {})
    rt = summary.get("rt_flow", {})
    gamma = summary.get("gamma", {})
    report_path = Path(result.report_path)
    atlas_rel = os.path.relpath(result.atlas_png_path, start=report_path.parent)
    lines = [
        "# Corrected Branch-Labelled Phantom Status Report",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        f"![Corrected branch-labelled phantom status]({atlas_rel})",
        "",
        "## Executive Snapshot",
        "",
        f"- Corrected branch-labelled vessel mask is now the source for arterial/venous flow domains and RT vascular-fluid masks.",
        f"- Vascular labels: arterial {vascular.get('arterial_voxels', 0)} voxels, venous {vascular.get('venous_voxels', 0)} voxels, unclassified labels `{vascular.get('unclassified_labels', [])}`.",
        f"- RT vascular-fluid mask: {rt_pkg.get('vascular_fluid_volume_cm3', 0.0):.3f} cm3; vessel wall: {rt_pkg.get('vessel_wall_volume_cm3', 0.0):.3f} cm3.",
        f"- Coupled pulsatile flow: {flow.get('node_count', 0)} nodes / {flow.get('edge_count', 0)} edges; aorta mean flow {flow.get('aorta_flow_mean_ml_s', 0.0):.3f} mL/s.",
        f"- 4D flow visualization: {flow4d.get('frame_count', 0)} frames; animation `{Path(str(flow4d.get('animation_gif', ''))).name}`.",
        f"- Spatial RT-flow: {rt.get('selected_edge_count', 0)} selected edges; top coupled edge `{rt.get('top_coupled_edge', '')}`.",
        f"- Gamma QA: minimum pass rate {gamma.get('min_pass_rate_percent', 0.0):.3f}% at {gamma.get('criteria', {}).get('dose_percent', 0.0):.1f}%/{gamma.get('criteria', {}).get('distance_mm', 0.0):.1f} mm.",
        "",
        "## Key Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| Boundary nodes snapped to corrected lumen | {vascular.get('snapped_boundary_nodes', 0)} |",
        f"| Arterial / venous components | {vascular.get('arterial_components', 0)} / {vascular.get('venous_components', 0)} |",
        f"| Aorta flow mean/min/max | {flow.get('aorta_flow_mean_ml_s', 0.0):.3f} / {flow.get('aorta_flow_min_ml_s', 0.0):.3f} / {flow.get('aorta_flow_max_ml_s', 0.0):.3f} mL/s |",
        f"| Flow mass-balance residual | {flow.get('max_mass_balance_residual_ml_s', 0.0):.3e} mL/s |",
        f"| Top RT-flow edge score | {rt.get('top_coupled_edge_score', 0.0):.5f} |",
        f"| Spatial peak/trough phase | {rt.get('peak_phase', 0.0):.3f} / {rt.get('trough_phase', 0.0):.3f} |",
        f"| Max spatial peak/trough delta | {rt.get('max_peak_delta_mgy', 0.0):.3f} / {rt.get('max_trough_delta_mgy', 0.0):.3f} mGy |",
        f"| PTV peak D95 / V95 | {rt.get('ptv_peak_d95_gy', 0.0):.4f} Gy / {rt.get('ptv_peak_v95_percent', 0.0):.3f}% |",
        f"| Gamma max p95 / max gamma | {gamma.get('max_p95_gamma', 0.0):.4f} / {gamma.get('max_gamma', 0.0):.4f} |",
        "",
        "## Artifact Index",
        "",
    ]
    for name, path in result.artifact_paths.items():
        lines.append(f"- `{name}`: {_markdown_link(path, report_path)}")
    lines.extend(
        [
            "",
            "## Current Limitations",
            "",
            "- This remains a research digital phantom, not a clinically commissioned patient-specific twin.",
            "- The vascular branch labels are real labelled-vessel template labels staged onto this CT grid, but not yet deformably registered from a patient-specific CTA/CTV into this anatomy.",
            "- Flow is a 1D graph/RCR surrogate with placeholder venous pulsatility; it is not 3D CFD or measured physiology.",
            "- RT dose states are synthetic surrogate dose grids for pipeline QA, not TPS or Monte Carlo dose calculations.",
            "- The 4D visualization maps time-varying graph quantities onto centerlines; it is not particle/pathline blood transport.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_corrected_branch_status_report(
    output_dir: str | Path = "outputs/reports/corrected_branch_status",
    case_id: str = "mode03_neg_branch_ctgrid_corrected_flow",
    vessel_flow_manifest_path: str | Path = "outputs/digital/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/mode03_neg_branch_ctgrid_corrected_flow_label_vessel_flow_domain_manifest_v001.yaml",
    vessel_flow_spec_path: str | Path = "outputs/digital/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/mode03_neg_branch_ctgrid_corrected_flow_label_vessel_flow_domain_spec_v001.yaml",
    vessel_flow_preview_path: str | Path = "outputs/digital/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/mode03_neg_branch_ctgrid_corrected_flow_label_vessel_flow_domain_preview_v001.png",
    rt_package_spec_path: str | Path = "outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/qa_package/mode03_neg_branch_ctgrid_corrected_flow_radiotherapy_qa_package_spec_v001.yaml",
    rt_qa_preview_path: str | Path = "outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/qa_package/mode03_neg_branch_ctgrid_corrected_flow_radiotherapy_qa_preview_v001.png",
    coupled_flow_model_path: str | Path = "outputs/sim/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/flow_coupled_pulsatile/mode03_neg_branch_ctgrid_corrected_flow_coupled_pulsatile_flow_model_v001.yaml",
    coupled_flow_preview_path: str | Path = "outputs/sim/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/flow_coupled_pulsatile/plots/mode03_neg_branch_ctgrid_corrected_flow_coupled_pulsatile_pressure_flow_preview_v001.png",
    flow4d_spec_path: str | Path = "outputs/sim/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/flow_4d_visualization/mode03_neg_branch_ctgrid_corrected_flow_flow4d_visualization_spec_v001.yaml",
    spatial_coupling_spec_path: str | Path = "outputs/experiments/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_coupling/mode03_neg_branch_ctgrid_corrected_flow_spatial_rt_flow_coupling_spec_v001.yaml",
    spatial_coupling_csv_path: str | Path = "outputs/experiments/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_coupling/mode03_neg_branch_ctgrid_corrected_flow_spatial_rt_flow_edge_coupling_v001.csv",
    spatial_coupling_preview_path: str | Path = "outputs/experiments/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_coupling/mode03_neg_branch_ctgrid_corrected_flow_spatial_rt_flow_coupling_preview_v001.png",
    spatial_dose_spec_path: str | Path = "outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_dose/mode03_neg_branch_ctgrid_corrected_flow_rt_spatial_flow_dose_model_spec_v001.yaml",
    spatial_dose_preview_path: str | Path = "outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_dose/mode03_neg_branch_ctgrid_corrected_flow_rt_spatial_flow_dose_model_preview_v001.png",
    gamma_spec_path: str | Path = "outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_dose/dose_gamma_qa/mode03_neg_branch_ctgrid_corrected_flow_spatial_dose_gamma_qa_spec_v001.yaml",
    gamma_qa_preview_path: str | Path = "outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_dose/dose_gamma_qa/mode03_neg_branch_ctgrid_corrected_flow_spatial_dose_gamma_qa_preview_v001.png",
    report_path: str | Path | None = "outputs/reports/mode03_neg_branch_ctgrid_corrected_status_report.md",
) -> CorrectedBranchStatusResult:
    output = Path(output_dir)
    atlas_png = output / f"{case_id}_corrected_branch_status_atlas_v001.png"
    manifest_yaml = output / f"{case_id}_corrected_branch_status_manifest_v001.yaml"
    metrics_csv = output / f"{case_id}_corrected_branch_status_metrics_v001.csv"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_corrected_branch_status_report_v001.md"
    flow4d = _load_yaml(flow4d_spec_path)
    artifact_paths = {
        "vessel_flow_manifest": _existing_path(vessel_flow_manifest_path),
        "vessel_flow_spec": _existing_path(vessel_flow_spec_path),
        "flow_domain_preview": _existing_path(vessel_flow_preview_path),
        "rt_package_spec": _existing_path(rt_package_spec_path),
        "rt_qa_preview": _existing_path(rt_qa_preview_path),
        "coupled_flow_model": _existing_path(coupled_flow_model_path),
        "coupled_flow_preview": _existing_path(coupled_flow_preview_path),
        "flow4d_spec": _existing_path(flow4d_spec_path),
        "flow4d_animation_gif": _existing_path(_nested(flow4d, "outputs", "animation_gif")),
        "flow4d_contact_sheet": _existing_path(_nested(flow4d, "outputs", "contact_sheet_png")),
        "spatial_coupling_spec": _existing_path(spatial_coupling_spec_path),
        "spatial_coupling_csv": _existing_path(spatial_coupling_csv_path),
        "spatial_coupling_preview": _existing_path(spatial_coupling_preview_path),
        "spatial_dose_spec": _existing_path(spatial_dose_spec_path),
        "spatial_dose_preview": _existing_path(spatial_dose_preview_path),
        "gamma_spec": _existing_path(gamma_spec_path),
        "gamma_qa_preview": _existing_path(gamma_qa_preview_path),
    }
    summary = _build_summary(
        vessel_flow_manifest_path=vessel_flow_manifest_path,
        vessel_flow_spec_path=vessel_flow_spec_path,
        rt_package_spec_path=rt_package_spec_path,
        coupled_flow_model_path=coupled_flow_model_path,
        flow4d_spec_path=flow4d_spec_path,
        spatial_coupling_csv_path=spatial_coupling_csv_path,
        spatial_dose_spec_path=spatial_dose_spec_path,
        gamma_spec_path=gamma_spec_path,
    )
    notes = (
        "report_indexes_existing_corrected_branch_outputs_without_copying_large_volumes",
        "rt_package_uses_corrected_branch_labelled_vascular_fluid_mask",
        "flow_and_rt_models_are_research_surrogates_not_clinical_calculations",
    )
    result = CorrectedBranchStatusResult(
        case_id=case_id,
        output_dir=str(output),
        report_path=str(report),
        manifest_yaml_path=str(manifest_yaml),
        metrics_csv_path=str(metrics_csv),
        atlas_png_path=str(atlas_png),
        summary=summary,
        artifact_paths=artifact_paths,
        notes=notes,
    )
    _write_atlas_png(atlas_png, case_id, artifact_paths, summary)
    _write_metrics_csv(metrics_csv, summary)
    _write_manifest(manifest_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_corrected_branch_status_result(result: CorrectedBranchStatusResult) -> str:
    summary = result.summary
    flow = summary.get("flow", {})
    rt = summary.get("rt_flow", {})
    gamma = summary.get("gamma", {})
    return "\n".join(
        [
            "Corrected branch-labelled phantom status report created",
            f"Case ID: {result.case_id}",
            f"Aorta flow mean: {flow.get('aorta_flow_mean_ml_s', 0.0):.3f} mL/s",
            f"Spatial selected edges: {rt.get('selected_edge_count', 0)}",
            f"Gamma min pass rate: {gamma.get('min_pass_rate_percent', 0.0):.3f}%",
            f"Atlas PNG: {result.atlas_png_path}",
            f"Metrics CSV: {result.metrics_csv_path}",
            f"Manifest YAML: {result.manifest_yaml_path}",
            f"Report: {result.report_path}",
        ]
    )
