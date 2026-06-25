from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np

from .flow_pulsatile import (
    PA_PER_MMHG,
    TerminalRCRModel,
    _boundary_flows_for_domain,
    _boundary_payload_by_node,
    _build_terminal_rcr_models,
    _domain_nodes,
    _domain_segments,
    _import_dependencies,
    _lagged_venous_multiplier,
    _load_yaml,
    _safe_float,
    _series,
    _solve_edge_flows,
    _solve_pressures,
    _vessel_types,
    _waveform_samples,
    _write_boundary_summary,
    _write_boundary_timeseries,
    _write_edge_timeseries,
    _write_node_timeseries,
)


@dataclass(frozen=True)
class CoupledPulsatileFlowResult:
    case_id: str
    output_dir: str
    model_yaml_path: str
    edge_timeseries_csv_path: str
    node_timeseries_csv_path: str
    boundary_timeseries_csv_path: str
    boundary_summary_csv_path: str
    outlet_split_csv_path: str
    qa_timeseries_csv_path: str
    report_path: str
    plot_paths: tuple[str, ...]
    source_flow_1d_model_path: str
    source_boundary_config_path: str
    heart_rate_bpm: float
    cycle_duration_s: float
    samples_per_cycle: int
    settling_cycles: int
    rcr_proximal_resistance_fraction: float
    rcr_time_constant_s: float
    venous_pulsatility_fraction: float
    venous_phase_lag_fraction: float
    edge_count: int
    node_count: int
    boundary_count: int
    terminal_rcr_count: int
    arterial_inlet_flow_mean_ml_s: float
    arterial_inlet_flow_min_ml_s: float
    arterial_inlet_flow_max_ml_s: float
    aorta_pressure_mean_pa: float
    aorta_pressure_min_pa: float
    aorta_pressure_max_pa: float
    max_outlet_split_range_percentage_points: float
    max_abs_mass_balance_residual_ml_s: float
    max_abs_pressure_equation_residual_pa: float
    notes: tuple[str, ...]


def _arterial_inlet_flow_by_node(nodes: list[dict[str, Any]], multiplier: float) -> dict[str, float]:
    return {
        str(node["node_id"]): _safe_float(node.get("boundary_flow_ml_s")) * multiplier
        for node in nodes
        if str(node.get("boundary_role")) == "arterial_inlet"
        and _safe_float(node.get("boundary_flow_ml_s")) > 0.0
    }


def _solve_coupled_arterial_step(
    nodes: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    inlet_flow_by_node_ml_s: dict[str, float],
    rcr_models: dict[str, TerminalRCRModel],
    capacitor_pressure_by_node: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    node_ids = tuple(str(node["node_id"]) for node in nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    matrix = np.zeros((len(node_ids), len(node_ids)), dtype=float)
    rhs = np.zeros(len(node_ids), dtype=float)

    for segment in segments:
        source = str(segment["source"])
        target = str(segment["target"])
        resistance = max(float(segment["hydraulic_resistance_pa_s_per_m3"]), 1e-12)
        conductance = 1.0 / resistance
        source_index = node_index[source]
        target_index = node_index[target]
        matrix[source_index, source_index] += conductance
        matrix[target_index, target_index] += conductance
        matrix[source_index, target_index] -= conductance
        matrix[target_index, source_index] -= conductance

    for node_id, flow_ml_s in inlet_flow_by_node_ml_s.items():
        if node_id in node_index:
            rhs[node_index[node_id]] += flow_ml_s * 1e-6

    for node_id, rcr in rcr_models.items():
        if node_id not in node_index:
            continue
        proximal_resistance = max(rcr.proximal_resistance_pa_s_per_m3, 1e-12)
        conductance = 1.0 / proximal_resistance
        index = node_index[node_id]
        matrix[index, index] += conductance
        rhs[index] += conductance * capacitor_pressure_by_node[node_id]

    if not np.any(matrix):
        return {}, {}, {}, {}, {}, {}

    pressures_array = np.linalg.solve(matrix, rhs)
    pressure_by_node = {node_id: float(pressures_array[index]) for node_id, index in node_index.items()}

    edge_flow_by_id: dict[str, float] = {}
    pressure_residual_by_edge: dict[str, float] = {}
    residual_by_node = {node_id: 0.0 for node_id in node_ids}
    for segment in segments:
        edge_id = str(segment["edge_id"])
        source = str(segment["source"])
        target = str(segment["target"])
        resistance = max(float(segment["hydraulic_resistance_pa_s_per_m3"]), 1e-12)
        flow_m3_s = (pressure_by_node[source] - pressure_by_node[target]) / resistance
        flow_ml_s = flow_m3_s * 1e6
        edge_flow_by_id[edge_id] = float(flow_ml_s)
        residual_by_node[source] -= flow_ml_s
        residual_by_node[target] += flow_ml_s
        pressure_residual_by_edge[edge_id] = float((pressure_by_node[source] - pressure_by_node[target]) - resistance * flow_m3_s)

    outlet_flow_by_node: dict[str, float] = {}
    terminal_pressure_by_node: dict[str, float] = {}
    for node_id, rcr in rcr_models.items():
        if node_id not in pressure_by_node:
            continue
        proximal_resistance = max(rcr.proximal_resistance_pa_s_per_m3, 1e-12)
        outlet_flow_m3_s = (pressure_by_node[node_id] - capacitor_pressure_by_node[node_id]) / proximal_resistance
        outlet_flow_ml_s = outlet_flow_m3_s * 1e6
        outlet_flow_by_node[node_id] = float(outlet_flow_ml_s)
        terminal_pressure_by_node[node_id] = pressure_by_node[node_id]
        residual_by_node[node_id] -= outlet_flow_ml_s

    for node_id, flow_ml_s in inlet_flow_by_node_ml_s.items():
        if node_id in residual_by_node:
            residual_by_node[node_id] += flow_ml_s

    return (
        pressure_by_node,
        edge_flow_by_id,
        outlet_flow_by_node,
        residual_by_node,
        pressure_residual_by_edge,
        terminal_pressure_by_node,
    )


def _update_rcr_states(
    capacitor_pressure_by_node: dict[str, float],
    rcr_models: dict[str, TerminalRCRModel],
    outlet_flow_by_node_ml_s: dict[str, float],
    dt_s: float,
) -> None:
    for node_id, rcr in rcr_models.items():
        flow_m3_s = outlet_flow_by_node_ml_s.get(node_id, 0.0) * 1e-6
        alpha = math.exp(-dt_s / max(rcr.time_constant_s, 1e-9))
        previous = capacitor_pressure_by_node[node_id]
        capacitor_pressure_by_node[node_id] = (
            rcr.distal_pressure_pa
            + (previous - rcr.distal_pressure_pa) * alpha
            + rcr.distal_resistance_pa_s_per_m3 * flow_m3_s * (1.0 - alpha)
        )


def _write_outlet_split_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["time_s", "phase", "node_id", "label", "flow_ml_s", "split_fraction", "split_percent"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_qa_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s",
        "phase",
        "arterial_inlet_flow_ml_s",
        "arterial_outlet_flow_ml_s",
        "arterial_flow_balance_error_ml_s",
        "venous_inlet_flow_ml_s",
        "venous_outlet_flow_ml_s",
        "venous_flow_balance_error_ml_s",
        "max_abs_node_mass_residual_ml_s",
        "max_abs_pressure_equation_residual_pa",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _max_outlet_split_range(split_rows: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = {}
    for row in split_rows:
        grouped.setdefault(str(row["node_id"]), []).append(float(row["split_percent"]))
    if not grouped:
        return 0.0
    return max(max(values) - min(values) for values in grouped.values())


def _write_coupled_plots(
    output_dir: Path,
    case_id: str,
    phases: np.ndarray,
    arterial_multiplier: np.ndarray,
    node_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    plt, _ = _import_dependencies()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    preview_path = plot_dir / f"{case_id}_coupled_pulsatile_pressure_flow_preview_v001.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    time, aorta_flow = _series(boundary_rows, "aorta_inlet", "flow_ml_s")
    if len(time):
        axes[0].plot(time, aorta_flow, color="#b21f2d", linewidth=2.2, label="aorta inlet")
    for node_id, color, label in [
        ("left_common_iliac_outlet", "#df8f1f", "left iliac outlet"),
        ("right_common_iliac_outlet", "#a65f00", "right iliac outlet"),
        ("left_renal_outlet", "#4e9f50", "left renal outlet"),
        ("right_renal_outlet", "#1d6f42", "right renal outlet"),
    ]:
        time, flow = _series(boundary_rows, node_id, "flow_ml_s")
        if len(time):
            axes[0].plot(time, np.abs(flow), color=color, linewidth=1.8, label=label)
    axes[0].set_ylabel("Flow (mL/s)")
    axes[0].set_title("Coupled Boundary Flow")
    axes[0].grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, ncol=2, fontsize=8)

    for node_id, color, label in [
        ("aorta_inlet", "#b21f2d", "aorta inlet"),
        ("visceral_branch_origin", "#c77700", "visceral origin"),
        ("renal_branch_origin", "#4e9f50", "renal origin"),
        ("aortic_bifurcation", "#7f4fa3", "bifurcation"),
    ]:
        time, pressure = _series(node_rows, node_id, "pressure_mmhg")
        if len(time):
            axes[1].plot(time, pressure, linewidth=1.9, color=color, label=label)
    axes[1].set_ylabel("Pressure (mmHg)")
    axes[1].set_title("Coupled Arterial Node Pressures")
    axes[1].grid(True, alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, ncol=2, fontsize=8)

    qa_time = np.array([float(row["time_s"]) for row in qa_rows], dtype=float)
    balance = np.array([float(row["arterial_flow_balance_error_ml_s"]) for row in qa_rows], dtype=float)
    pressure_residual = np.array([float(row["max_abs_pressure_equation_residual_pa"]) for row in qa_rows], dtype=float)
    if len(qa_time):
        axes[2].plot(qa_time, balance, color="#444444", linewidth=1.8, label="arterial balance error")
        axes[2].plot(qa_time, pressure_residual, color="#b21f2d", linewidth=1.8, label="max pressure residual Pa")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("QA value")
    axes[2].set_title("Coupled Solver QA")
    axes[2].grid(True, alpha=0.25)
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(preview_path, dpi=180)
    plt.close(fig)
    paths.append(str(preview_path))

    split_path = plot_dir / f"{case_id}_coupled_pulsatile_outlet_splits_v001.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in split_rows:
        grouped.setdefault(str(row["node_id"]), []).append(row)
    colors = ["#df8f1f", "#a65f00", "#4e9f50", "#1d6f42", "#2d7fb8", "#7f4fa3", "#555555"]
    for index, (node_id, rows) in enumerate(sorted(grouped.items())):
        row_time = np.array([float(row["time_s"]) for row in rows], dtype=float)
        row_flow = np.array([float(row["flow_ml_s"]) for row in rows], dtype=float)
        row_split = np.array([float(row["split_percent"]) for row in rows], dtype=float)
        label = str(rows[0]["label"])
        color = colors[index % len(colors)]
        axes[0].plot(row_time, row_flow, linewidth=1.8, color=color, label=label)
        axes[1].plot(row_time, row_split, linewidth=1.8, color=color, label=label)
    axes[0].set_ylabel("Outlet flow (mL/s)")
    axes[0].set_title("Dynamic Arterial Outlet Flow")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Outlet split (%)")
    axes[1].set_title("Dynamic Flow Split")
    axes[1].grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(split_path, dpi=180)
    plt.close(fig)
    paths.append(str(split_path))

    waveform_path = plot_dir / f"{case_id}_coupled_pulsatile_inlet_waveform_v001.png"
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(phases, arterial_multiplier, color="#b21f2d", linewidth=2.5)
    ax.set_xlabel("Cardiac phase")
    ax.set_ylabel("Normalized inlet multiplier")
    ax.set_title("Coupled Model Inlet Waveform")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(waveform_path, dpi=180)
    plt.close(fig)
    paths.append(str(waveform_path))
    return tuple(paths)


def _write_model_yaml(path: Path, result: CoupledPulsatileFlowResult, rcr_models: dict[str, TerminalRCRModel]) -> None:
    _, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "model_type": "coupled_pulsatile_1d_graph_flow_network",
        "source_flow_1d_model": result.source_flow_1d_model_path,
        "source_boundary_config": result.source_boundary_config_path,
        "units": {
            "time": "s",
            "flow": "mL/s",
            "pressure": "Pa",
            "pressure_display": "mmHg",
            "resistance": "Pa*s/m^3",
            "compliance": "m^3/Pa",
        },
        "inputs": {
            "heart_rate_bpm": result.heart_rate_bpm,
            "cycle_duration_s": result.cycle_duration_s,
            "samples_per_cycle": result.samples_per_cycle,
            "settling_cycles": result.settling_cycles,
            "rcr_proximal_resistance_fraction": result.rcr_proximal_resistance_fraction,
            "rcr_time_constant_s": result.rcr_time_constant_s,
            "venous_pulsatility_fraction": result.venous_pulsatility_fraction,
            "venous_phase_lag_fraction": result.venous_phase_lag_fraction,
        },
        "assumptions": [
            "arterial_branch_flows_are_solved_from_graph_conductance_and_RCR_terminal_states_each_timestep",
            "aortic_inlet_flow_is_prescribed_from_the_boundary_waveform",
            "arterial_terminal_RCR_states_are_advanced_with_an_analytic_constant_flow_step",
            "venous_return_remains_a_damped_phase_lagged_placeholder_domain",
            "no_wave_speed_inertance_or_distributed_wall_compliance_yet",
        ],
        "outputs": {
            "edge_timeseries_csv": result.edge_timeseries_csv_path,
            "node_timeseries_csv": result.node_timeseries_csv_path,
            "boundary_timeseries_csv": result.boundary_timeseries_csv_path,
            "boundary_summary_csv": result.boundary_summary_csv_path,
            "outlet_split_csv": result.outlet_split_csv_path,
            "qa_timeseries_csv": result.qa_timeseries_csv_path,
            "plots": list(result.plot_paths),
            "report": result.report_path,
        },
        "summary": {
            "edge_count": result.edge_count,
            "node_count": result.node_count,
            "boundary_count": result.boundary_count,
            "terminal_rcr_count": result.terminal_rcr_count,
            "arterial_inlet_flow_mean_ml_s": result.arterial_inlet_flow_mean_ml_s,
            "arterial_inlet_flow_min_ml_s": result.arterial_inlet_flow_min_ml_s,
            "arterial_inlet_flow_max_ml_s": result.arterial_inlet_flow_max_ml_s,
            "aorta_pressure_mean_pa": result.aorta_pressure_mean_pa,
            "aorta_pressure_min_pa": result.aorta_pressure_min_pa,
            "aorta_pressure_max_pa": result.aorta_pressure_max_pa,
            "max_outlet_split_range_percentage_points": result.max_outlet_split_range_percentage_points,
            "max_abs_mass_balance_residual_ml_s": result.max_abs_mass_balance_residual_ml_s,
            "max_abs_pressure_equation_residual_pa": result.max_abs_pressure_equation_residual_pa,
        },
        "terminal_rcr_models": [
            {
                "node_id": model.node_id,
                "label": model.label,
                "mean_flow_ml_s": model.mean_flow_ml_s,
                "total_resistance_pa_s_per_m3": model.total_resistance_pa_s_per_m3,
                "proximal_resistance_pa_s_per_m3": model.proximal_resistance_pa_s_per_m3,
                "distal_resistance_pa_s_per_m3": model.distal_resistance_pa_s_per_m3,
                "compliance_m3_per_pa": model.compliance_m3_per_pa,
                "time_constant_s": model.time_constant_s,
                "distal_pressure_pa": model.distal_pressure_pa,
                "initial_capacitor_pressure_pa": model.initial_capacitor_pressure_pa,
            }
            for model in rcr_models.values()
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: CoupledPulsatileFlowResult, rcr_models: dict[str, TerminalRCRModel]) -> str:
    lines = [
        "# Coupled Pulsatile 1D Flow Model Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Heart rate: {result.heart_rate_bpm:.1f} bpm",
        f"- Cycle duration: {result.cycle_duration_s:.4f} s",
        f"- Samples per cycle: {result.samples_per_cycle}",
        f"- Settling cycles before export: {result.settling_cycles}",
        f"- Edges/nodes/boundaries: {result.edge_count} / {result.node_count} / {result.boundary_count}",
        f"- Arterial terminal RCR models: {result.terminal_rcr_count}",
        f"- Aorta inlet flow mean/min/max: {result.arterial_inlet_flow_mean_ml_s:.3f} / {result.arterial_inlet_flow_min_ml_s:.3f} / {result.arterial_inlet_flow_max_ml_s:.3f} mL/s",
        f"- Aorta pressure mean/min/max: {result.aorta_pressure_mean_pa / PA_PER_MMHG:.2f} / {result.aorta_pressure_min_pa / PA_PER_MMHG:.2f} / {result.aorta_pressure_max_pa / PA_PER_MMHG:.2f} mmHg",
        f"- Max dynamic outlet split range: {result.max_outlet_split_range_percentage_points:.3f} percentage points",
        f"- Max mass-balance residual: {result.max_abs_mass_balance_residual_ml_s:.9f} mL/s",
        f"- Max pressure-equation residual: {result.max_abs_pressure_equation_residual_pa:.9f} Pa",
        "",
        "## Outputs",
        "",
        f"- Model YAML: `{Path(result.model_yaml_path).name}`",
        f"- Edge time-series CSV: `{Path(result.edge_timeseries_csv_path).name}`",
        f"- Node time-series CSV: `{Path(result.node_timeseries_csv_path).name}`",
        f"- Boundary time-series CSV: `{Path(result.boundary_timeseries_csv_path).name}`",
        f"- Boundary summary CSV: `{Path(result.boundary_summary_csv_path).name}`",
        f"- Outlet split CSV: `{Path(result.outlet_split_csv_path).name}`",
        f"- QA time-series CSV: `{Path(result.qa_timeseries_csv_path).name}`",
    ]
    for plot_path in result.plot_paths:
        lines.append(f"- Plot: `{Path(plot_path).name}`")

    lines.extend(
        [
            "",
            "## Coupled RCR Outlets",
            "",
            "| outlet node | mean baseline flow mL/s | R total Pa*s/m3 | C m3/Pa | distal pressure mmHg |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in rcr_models.values():
        lines.append(
            f"| `{model.node_id}` | {model.mean_flow_ml_s:.3f} | "
            f"{model.total_resistance_pa_s_per_m3:.3e} | {model.compliance_m3_per_pa:.3e} | "
            f"{model.distal_pressure_pa / PA_PER_MMHG:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The arterial network is now solved as a pressure-flow graph at each timestep.",
            "- Inlet flow is prescribed, but terminal branch flows are determined by segment resistances and RCR capacitor states.",
            "- This produces dynamic branch split changes instead of scaling every arterial outlet by the same waveform factor.",
            "- Venous return is still a placeholder domain and should be coupled to tissue/organ return compartments later.",
            "- This is still a research scaffold, not a calibrated clinical hemodynamic model.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_coupled_pulsatile_flow_model(
    flow_1d_model_path: str | Path,
    boundary_config_path: str | Path,
    output_dir: str | Path = "outputs/sim/flow_coupled_pulsatile",
    case_id: str = "ct_org_case0_imagetbad_case125",
    heart_rate_bpm: float = 60.0,
    samples_per_cycle: int = 160,
    settling_cycles: int = 3,
    rcr_proximal_resistance_fraction: float = 0.1,
    rcr_time_constant_s: float = 1.2,
    venous_pulsatility_fraction: float = 0.35,
    venous_phase_lag_fraction: float = 0.15,
    report_path: str | Path | None = "outputs/reports/flow_coupled_pulsatile_model_stage001.md",
) -> CoupledPulsatileFlowResult:
    flow_model_path = Path(flow_1d_model_path)
    boundary_path = Path(boundary_config_path)
    model = _load_yaml(flow_model_path)
    boundary_config = _load_yaml(boundary_path)
    boundary_by_node = _boundary_payload_by_node(boundary_config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if samples_per_cycle < 8:
        raise ValueError("samples_per_cycle must be at least 8")
    if heart_rate_bpm <= 0.0:
        raise ValueError("heart_rate_bpm must be positive")

    cycle_duration_s = 60.0 / heart_rate_bpm
    dt_s = cycle_duration_s / samples_per_cycle
    phases, arterial_multiplier = _waveform_samples(boundary_config, samples_per_cycle)
    venous_multiplier = _lagged_venous_multiplier(
        phases,
        arterial_multiplier,
        venous_pulsatility_fraction=venous_pulsatility_fraction,
        venous_phase_lag_fraction=venous_phase_lag_fraction,
    )

    arterial_nodes = _domain_nodes(model, "arterial")
    arterial_segments = _domain_segments(model, "arterial")
    venous_nodes = _domain_nodes(model, "venous")
    venous_segments = _domain_segments(model, "venous")
    rcr_models = _build_terminal_rcr_models(
        arterial_nodes,
        boundary_by_node=boundary_by_node,
        proximal_resistance_fraction=rcr_proximal_resistance_fraction,
        time_constant_s=rcr_time_constant_s,
    )
    capacitor_pressure_by_node = {
        node_id: rcr.initial_capacitor_pressure_pa for node_id, rcr in rcr_models.items()
    }

    edge_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    max_mass_residual = 0.0
    max_pressure_residual = 0.0
    venous_outlet_pressure = float(model.get("inputs", {}).get("venous_outlet_pressure_pa", 667.0))
    total_steps = (max(settling_cycles, 0) + 1) * samples_per_cycle
    first_recorded_step = max(settling_cycles, 0) * samples_per_cycle

    for global_step in range(total_steps):
        sample_index = global_step % samples_per_cycle
        record_step = global_step >= first_recorded_step
        output_sample_index = global_step - first_recorded_step
        phase = float(phases[sample_index])
        time_s = float(output_sample_index * dt_s)

        inlet_flow_by_node = _arterial_inlet_flow_by_node(arterial_nodes, float(arterial_multiplier[sample_index]))
        (
            arterial_pressure_by_node,
            arterial_flow_by_edge,
            arterial_outlet_flow_by_node,
            arterial_residual_by_node,
            arterial_pressure_residual_by_edge,
            terminal_pressure_by_node,
        ) = _solve_coupled_arterial_step(
            arterial_nodes,
            arterial_segments,
            inlet_flow_by_node_ml_s=inlet_flow_by_node,
            rcr_models=rcr_models,
            capacitor_pressure_by_node=capacitor_pressure_by_node,
        )
        max_mass_residual = max(max_mass_residual, max((abs(value) for value in arterial_residual_by_node.values()), default=0.0))
        max_pressure_residual = max(
            max_pressure_residual,
            max((abs(value) for value in arterial_pressure_residual_by_edge.values()), default=0.0),
        )

        venous_boundary_flows = _boundary_flows_for_domain(venous_nodes, float(venous_multiplier[sample_index]))
        venous_node_ids = tuple(str(node["node_id"]) for node in venous_nodes)
        venous_flow_by_edge, venous_residual_by_node = _solve_edge_flows(venous_node_ids, venous_segments, venous_boundary_flows)
        venous_references = {
            str(node["node_id"]): venous_outlet_pressure
            for node in venous_nodes
            if str(node.get("boundary_role")) == "venous_outlet"
        }
        if not venous_references and venous_nodes:
            venous_references[str(venous_nodes[0]["node_id"])] = venous_outlet_pressure
        venous_pressure_by_node, venous_pressure_residual_by_edge = _solve_pressures(
            venous_node_ids,
            venous_segments,
            venous_flow_by_edge,
            venous_references,
            reference_weight=50.0,
        )
        max_mass_residual = max(max_mass_residual, max((abs(value) for value in venous_residual_by_node.values()), default=0.0))
        max_pressure_residual = max(
            max_pressure_residual,
            max((abs(value) for value in venous_pressure_residual_by_edge.values()), default=0.0),
        )

        pressure_by_node = {**arterial_pressure_by_node, **venous_pressure_by_node}
        flow_by_edge = {**arterial_flow_by_edge, **venous_flow_by_edge}
        residual_by_node = {**arterial_residual_by_node, **venous_residual_by_node}
        pressure_residual_by_edge = {**arterial_pressure_residual_by_edge, **venous_pressure_residual_by_edge}
        boundary_flow_by_node: dict[str, float] = {}
        boundary_flow_by_node.update(inlet_flow_by_node)
        boundary_flow_by_node.update({node_id: -flow for node_id, flow in arterial_outlet_flow_by_node.items()})
        boundary_flow_by_node.update(venous_boundary_flows)

        if record_step:
            incoming_by_node = {str(node["node_id"]): 0.0 for node in model.get("nodes", [])}
            outgoing_by_node = {str(node["node_id"]): 0.0 for node in model.get("nodes", [])}
            for segment in model.get("segments", []):
                edge_id = str(segment["edge_id"])
                flow_ml_s = flow_by_edge.get(edge_id, 0.0)
                outgoing_by_node[str(segment["source"])] += flow_ml_s
                incoming_by_node[str(segment["target"])] += flow_ml_s
                pressure_source = pressure_by_node.get(str(segment["source"]), 0.0)
                pressure_target = pressure_by_node.get(str(segment["target"]), 0.0)
                area_mm2 = float(segment.get("area_mm2", 0.0))
                velocity_cm_s = (flow_ml_s * 1000.0 / area_mm2) / 10.0 if area_mm2 > 0.0 else 0.0
                edge_rows.append(
                    {
                        "time_s": f"{time_s:.6f}",
                        "phase": f"{phase:.6f}",
                        "edge_id": edge_id,
                        "vessel_type": segment.get("vessel_type", ""),
                        "flow_role": segment.get("flow_role", ""),
                        "flow_ml_s": f"{flow_ml_s:.6f}",
                        "mean_velocity_cm_s": f"{velocity_cm_s:.6f}",
                        "pressure_source_pa": f"{pressure_source:.6f}",
                        "pressure_target_pa": f"{pressure_target:.6f}",
                        "pressure_drop_pa": f"{(pressure_source - pressure_target):.6f}",
                        "pressure_source_mmhg": f"{pressure_source / PA_PER_MMHG:.6f}",
                        "pressure_target_mmhg": f"{pressure_target / PA_PER_MMHG:.6f}",
                        "pressure_equation_residual_pa": f"{pressure_residual_by_edge.get(edge_id, 0.0):.9f}",
                    }
                )

            for node in model.get("nodes", []):
                node_id = str(node["node_id"])
                pressure = pressure_by_node.get(node_id, 0.0)
                boundary_flow = boundary_flow_by_node.get(node_id, 0.0)
                node_rows.append(
                    {
                        "time_s": f"{time_s:.6f}",
                        "phase": f"{phase:.6f}",
                        "node_id": node_id,
                        "vessel_type": node.get("vessel_type", ""),
                        "role": node.get("role", ""),
                        "boundary_role": node.get("boundary_role", ""),
                        "pressure_pa": f"{pressure:.6f}",
                        "pressure_mmhg": f"{pressure / PA_PER_MMHG:.6f}",
                        "boundary_flow_ml_s": f"{boundary_flow:.6f}",
                        "incoming_edge_flow_ml_s": f"{incoming_by_node.get(node_id, 0.0):.6f}",
                        "outgoing_edge_flow_ml_s": f"{outgoing_by_node.get(node_id, 0.0):.6f}",
                        "mass_balance_residual_ml_s": f"{residual_by_node.get(node_id, 0.0):.9f}",
                    }
                )
                if node.get("boundary_role"):
                    boundary = boundary_by_node.get(node_id, {})
                    rcr = rcr_models.get(node_id)
                    capacitor_pressure = capacitor_pressure_by_node.get(node_id)
                    terminal_pressure = terminal_pressure_by_node.get(node_id)
                    boundary_rows.append(
                        {
                            "time_s": f"{time_s:.6f}",
                            "phase": f"{phase:.6f}",
                            "boundary_id": boundary.get("boundary_id", ""),
                            "node_id": node_id,
                            "label": boundary.get("label", node.get("label", node_id)),
                            "vessel_type": node.get("vessel_type", ""),
                            "role": node.get("boundary_role", ""),
                            "flow_ml_s": f"{boundary_flow:.6f}",
                            "pressure_pa": f"{pressure:.6f}",
                            "pressure_mmhg": f"{pressure / PA_PER_MMHG:.6f}",
                            "rcr_terminal_pressure_pa": "" if terminal_pressure is None else f"{terminal_pressure:.6f}",
                            "rcr_capacitor_pressure_pa": "" if capacitor_pressure is None else f"{capacitor_pressure:.6f}",
                            "rcr_distal_pressure_pa": "" if rcr is None else f"{rcr.distal_pressure_pa:.6f}",
                        }
                    )

            arterial_inlet_total = sum(inlet_flow_by_node.values())
            arterial_outlet_total = sum(max(flow, 0.0) for flow in arterial_outlet_flow_by_node.values())
            venous_inlet_total = sum(flow for flow in venous_boundary_flows.values() if flow > 0.0)
            venous_outlet_total = abs(sum(flow for flow in venous_boundary_flows.values() if flow < 0.0))
            total_outlet = max(arterial_outlet_total, 1e-12)
            for node_id, outlet_flow in sorted(arterial_outlet_flow_by_node.items()):
                rcr = rcr_models[node_id]
                split_fraction = max(outlet_flow, 0.0) / total_outlet
                split_rows.append(
                    {
                        "time_s": f"{time_s:.6f}",
                        "phase": f"{phase:.6f}",
                        "node_id": node_id,
                        "label": rcr.label,
                        "flow_ml_s": f"{max(outlet_flow, 0.0):.6f}",
                        "split_fraction": f"{split_fraction:.9f}",
                        "split_percent": f"{split_fraction * 100.0:.6f}",
                    }
                )
            qa_rows.append(
                {
                    "time_s": f"{time_s:.6f}",
                    "phase": f"{phase:.6f}",
                    "arterial_inlet_flow_ml_s": f"{arterial_inlet_total:.6f}",
                    "arterial_outlet_flow_ml_s": f"{arterial_outlet_total:.6f}",
                    "arterial_flow_balance_error_ml_s": f"{(arterial_inlet_total - arterial_outlet_total):.9f}",
                    "venous_inlet_flow_ml_s": f"{venous_inlet_total:.6f}",
                    "venous_outlet_flow_ml_s": f"{venous_outlet_total:.6f}",
                    "venous_flow_balance_error_ml_s": f"{(venous_inlet_total - venous_outlet_total):.9f}",
                    "max_abs_node_mass_residual_ml_s": f"{max(abs(value) for value in residual_by_node.values()):.9f}",
                    "max_abs_pressure_equation_residual_pa": f"{max(abs(value) for value in pressure_residual_by_edge.values()):.9f}",
                }
            )

        _update_rcr_states(capacitor_pressure_by_node, rcr_models, arterial_outlet_flow_by_node, dt_s)

    base = output / case_id
    model_yaml = base.with_name(f"{case_id}_coupled_pulsatile_flow_model_v001.yaml")
    edge_csv = base.with_name(f"{case_id}_coupled_pulsatile_edge_timeseries_v001.csv")
    node_csv = base.with_name(f"{case_id}_coupled_pulsatile_node_timeseries_v001.csv")
    boundary_csv = base.with_name(f"{case_id}_coupled_pulsatile_boundary_timeseries_v001.csv")
    boundary_summary_csv = base.with_name(f"{case_id}_coupled_pulsatile_boundary_summary_v001.csv")
    outlet_split_csv = base.with_name(f"{case_id}_coupled_pulsatile_outlet_splits_v001.csv")
    qa_csv = base.with_name(f"{case_id}_coupled_pulsatile_qa_timeseries_v001.csv")
    report = Path(report_path) if report_path else base.with_name(f"{case_id}_coupled_pulsatile_flow_model_report_v001.md")

    _write_edge_timeseries(edge_csv, edge_rows)
    _write_node_timeseries(node_csv, node_rows)
    _write_boundary_timeseries(boundary_csv, boundary_rows)
    _write_boundary_summary(boundary_summary_csv, boundary_rows, rcr_models)
    _write_outlet_split_csv(outlet_split_csv, split_rows)
    _write_qa_csv(qa_csv, qa_rows)
    plot_paths = _write_coupled_plots(output, case_id, phases, arterial_multiplier, node_rows, boundary_rows, split_rows, qa_rows)

    _, aorta_flow = _series(boundary_rows, "aorta_inlet", "flow_ml_s")
    _, aorta_pressure = _series(node_rows, "aorta_inlet", "pressure_pa")
    if aorta_flow.size == 0:
        aorta_flow = np.array([0.0])
    if aorta_pressure.size == 0:
        aorta_pressure = np.array([0.0])

    result = CoupledPulsatileFlowResult(
        case_id=case_id,
        output_dir=str(output),
        model_yaml_path=str(model_yaml),
        edge_timeseries_csv_path=str(edge_csv),
        node_timeseries_csv_path=str(node_csv),
        boundary_timeseries_csv_path=str(boundary_csv),
        boundary_summary_csv_path=str(boundary_summary_csv),
        outlet_split_csv_path=str(outlet_split_csv),
        qa_timeseries_csv_path=str(qa_csv),
        report_path=str(report),
        plot_paths=plot_paths,
        source_flow_1d_model_path=str(flow_model_path),
        source_boundary_config_path=str(boundary_path),
        heart_rate_bpm=heart_rate_bpm,
        cycle_duration_s=cycle_duration_s,
        samples_per_cycle=samples_per_cycle,
        settling_cycles=max(settling_cycles, 0),
        rcr_proximal_resistance_fraction=rcr_proximal_resistance_fraction,
        rcr_time_constant_s=rcr_time_constant_s,
        venous_pulsatility_fraction=venous_pulsatility_fraction,
        venous_phase_lag_fraction=venous_phase_lag_fraction,
        edge_count=len(model.get("segments", [])),
        node_count=len(model.get("nodes", [])),
        boundary_count=len({row["node_id"] for row in boundary_rows}),
        terminal_rcr_count=len(rcr_models),
        arterial_inlet_flow_mean_ml_s=float(aorta_flow.mean()),
        arterial_inlet_flow_min_ml_s=float(aorta_flow.min()),
        arterial_inlet_flow_max_ml_s=float(aorta_flow.max()),
        aorta_pressure_mean_pa=float(aorta_pressure.mean()),
        aorta_pressure_min_pa=float(aorta_pressure.min()),
        aorta_pressure_max_pa=float(aorta_pressure.max()),
        max_outlet_split_range_percentage_points=_max_outlet_split_range(split_rows),
        max_abs_mass_balance_residual_ml_s=max_mass_residual,
        max_abs_pressure_equation_residual_pa=max_pressure_residual,
        notes=(
            "arterial_branch_flows_solved_from_graph_conductance_each_timestep",
            "terminal_RCR_outlets_drive_dynamic_flow_split_changes",
            "venous_domain_remains_phase_lagged_placeholder",
            "not_calibrated_for_patient_specific_or_clinical_use",
        ),
    )
    _write_model_yaml(model_yaml, result, rcr_models)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, rcr_models))
    return result


def format_coupled_pulsatile_flow_result(result: CoupledPulsatileFlowResult) -> str:
    lines = [
        "Coupled pulsatile 1D flow model created",
        f"Case ID: {result.case_id}",
        f"Heart rate: {result.heart_rate_bpm:.1f} bpm",
        f"Samples per cycle: {result.samples_per_cycle}",
        f"Edges/nodes/boundaries: {result.edge_count}/{result.node_count}/{result.boundary_count}",
        f"Terminal RCR outlets: {result.terminal_rcr_count}",
        f"Aorta flow mean/min/max: {result.arterial_inlet_flow_mean_ml_s:.3f}/{result.arterial_inlet_flow_min_ml_s:.3f}/{result.arterial_inlet_flow_max_ml_s:.3f} mL/s",
        f"Aorta pressure mean/min/max: {result.aorta_pressure_mean_pa / PA_PER_MMHG:.2f}/{result.aorta_pressure_min_pa / PA_PER_MMHG:.2f}/{result.aorta_pressure_max_pa / PA_PER_MMHG:.2f} mmHg",
        f"Max outlet split range: {result.max_outlet_split_range_percentage_points:.3f} percentage points",
        f"Max mass-balance residual: {result.max_abs_mass_balance_residual_ml_s:.9f} mL/s",
        f"Max pressure residual: {result.max_abs_pressure_equation_residual_pa:.9f} Pa",
        f"Model YAML: {result.model_yaml_path}",
        f"Edge time series: {result.edge_timeseries_csv_path}",
        f"Node time series: {result.node_timeseries_csv_path}",
        f"Boundary time series: {result.boundary_timeseries_csv_path}",
        f"Boundary summary: {result.boundary_summary_csv_path}",
        f"Outlet splits: {result.outlet_split_csv_path}",
        f"QA time series: {result.qa_timeseries_csv_path}",
        f"Report: {result.report_path}",
    ]
    for plot_path in result.plot_paths:
        lines.append(f"Plot: {plot_path}")
    return "\n".join(lines)
