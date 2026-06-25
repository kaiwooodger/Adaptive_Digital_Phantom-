from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np


PA_PER_MMHG = 133.322


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Pulsatile flow modeling requires matplotlib and PyYAML.") from exc
    return plt, yaml


@dataclass(frozen=True)
class TerminalRCRModel:
    node_id: str
    label: str
    mean_flow_ml_s: float
    total_resistance_pa_s_per_m3: float
    proximal_resistance_pa_s_per_m3: float
    distal_resistance_pa_s_per_m3: float
    compliance_m3_per_pa: float
    time_constant_s: float
    distal_pressure_pa: float
    initial_capacitor_pressure_pa: float


@dataclass(frozen=True)
class PulsatileFlowResult:
    case_id: str
    output_dir: str
    model_yaml_path: str
    edge_timeseries_csv_path: str
    node_timeseries_csv_path: str
    boundary_timeseries_csv_path: str
    boundary_summary_csv_path: str
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
    venous_outlet_pressure_mean_pa: float
    max_abs_mass_balance_residual_ml_s: float
    max_abs_pressure_equation_residual_pa: float
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    _, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _boundary_payload_by_node(boundary_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(boundary["node_id"]): boundary
        for boundary in boundary_config.get("boundaries", [])
        if boundary.get("node_id") is not None
    }


def _waveform_samples(boundary_config: dict[str, Any], samples_per_cycle: int) -> tuple[np.ndarray, np.ndarray]:
    waveform = boundary_config.get("waveforms", {}).get("arterial_inlet_unit_cycle", [])
    if not waveform:
        phases = np.arange(samples_per_cycle, dtype=float) / samples_per_cycle
        multipliers = 1.0 + 0.35 * np.sin(2.0 * math.pi * phases)
        return phases, multipliers / float(multipliers.mean())

    raw_phase = np.array([float(sample["phase"]) for sample in waveform], dtype=float)
    raw_multiplier = np.array([float(sample["normalized_flow_multiplier"]) for sample in waveform], dtype=float)
    order = np.argsort(raw_phase)
    raw_phase = raw_phase[order]
    raw_multiplier = raw_multiplier[order]
    raw_phase = np.mod(raw_phase, 1.0)
    order = np.argsort(raw_phase)
    raw_phase = raw_phase[order]
    raw_multiplier = raw_multiplier[order]

    phases = np.arange(samples_per_cycle, dtype=float) / samples_per_cycle
    extended_phase = np.concatenate([raw_phase, [raw_phase[0] + 1.0]])
    extended_multiplier = np.concatenate([raw_multiplier, [raw_multiplier[0]]])
    multipliers = np.interp(phases, extended_phase, extended_multiplier)
    multipliers = np.maximum(multipliers, 0.01)
    multipliers = multipliers / float(multipliers.mean())
    return phases, multipliers


def _lagged_venous_multiplier(
    phases: np.ndarray,
    arterial_multiplier: np.ndarray,
    venous_pulsatility_fraction: float,
    venous_phase_lag_fraction: float,
) -> np.ndarray:
    extended_phase = np.concatenate([phases, [1.0]])
    extended_multiplier = np.concatenate([arterial_multiplier, [arterial_multiplier[0]]])
    sample_phase = np.mod(phases - venous_phase_lag_fraction, 1.0)
    lagged = np.interp(sample_phase, extended_phase, extended_multiplier)
    multiplier = 1.0 + venous_pulsatility_fraction * (lagged - 1.0)
    multiplier = np.maximum(multiplier, 0.05)
    return multiplier / float(multiplier.mean())


def _domain_segments(model: dict[str, Any], vessel_type: str) -> list[dict[str, Any]]:
    return [segment for segment in model.get("segments", []) if str(segment.get("vessel_type")) == vessel_type]


def _domain_nodes(model: dict[str, Any], vessel_type: str) -> list[dict[str, Any]]:
    return [node for node in model.get("nodes", []) if str(node.get("vessel_type")) == vessel_type]


def _vessel_types(model: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({str(segment.get("vessel_type")) for segment in model.get("segments", [])}))


def _incidence_matrix(
    node_ids: tuple[str, ...],
    segments: list[dict[str, Any]],
) -> np.ndarray:
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    incidence = np.zeros((len(node_ids), len(segments)), dtype=float)
    for edge_index, segment in enumerate(segments):
        incidence[node_index[str(segment["source"])], edge_index] = -1.0
        incidence[node_index[str(segment["target"])], edge_index] = 1.0
    return incidence


def _solve_edge_flows(
    node_ids: tuple[str, ...],
    segments: list[dict[str, Any]],
    boundary_flow_by_node: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    if not segments:
        return {}, {}
    incidence = _incidence_matrix(node_ids, segments)
    boundary = np.array([boundary_flow_by_node.get(node_id, 0.0) for node_id in node_ids], dtype=float)
    flows, *_ = np.linalg.lstsq(incidence, -boundary, rcond=None)
    residuals = incidence @ flows + boundary
    return (
        {str(segment["edge_id"]): float(flow) for segment, flow in zip(segments, flows)},
        {node_id: float(residual) for node_id, residual in zip(node_ids, residuals)},
    )


def _solve_pressures(
    node_ids: tuple[str, ...],
    segments: list[dict[str, Any]],
    edge_flow_by_id: dict[str, float],
    reference_pressure_by_node: dict[str, float],
    reference_weight: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if not node_ids:
        return {}, {}
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for segment in segments:
        row = np.zeros(len(node_ids), dtype=float)
        row[node_index[str(segment["source"])]] = 1.0
        row[node_index[str(segment["target"])]] = -1.0
        edge_id = str(segment["edge_id"])
        resistance = float(segment["hydraulic_resistance_pa_s_per_m3"])
        rows.append(row)
        rhs.append(resistance * edge_flow_by_id.get(edge_id, 0.0) * 1e-6)

    for node_id, pressure_pa in reference_pressure_by_node.items():
        if node_id not in node_index:
            continue
        row = np.zeros(len(node_ids), dtype=float)
        row[node_index[node_id]] = reference_weight
        rows.append(row)
        rhs.append(float(pressure_pa) * reference_weight)

    matrix = np.vstack(rows)
    vector = np.array(rhs, dtype=float)
    solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    pressures = {node_id: float(solution[index]) for node_id, index in node_index.items()}
    residuals: dict[str, float] = {}
    for segment in segments:
        edge_id = str(segment["edge_id"])
        expected = float(segment["hydraulic_resistance_pa_s_per_m3"]) * edge_flow_by_id.get(edge_id, 0.0) * 1e-6
        actual = pressures[str(segment["source"])] - pressures[str(segment["target"])]
        residuals[edge_id] = float(actual - expected)
    return pressures, residuals


def _build_terminal_rcr_models(
    nodes: list[dict[str, Any]],
    boundary_by_node: dict[str, dict[str, Any]],
    proximal_resistance_fraction: float,
    time_constant_s: float,
) -> dict[str, TerminalRCRModel]:
    models: dict[str, TerminalRCRModel] = {}
    for node in nodes:
        if str(node.get("boundary_role")) != "arterial_outlet":
            continue
        node_id = str(node["node_id"])
        mean_flow_ml_s = abs(_safe_float(node.get("boundary_flow_ml_s")))
        if mean_flow_ml_s <= 0.0:
            continue
        mean_flow_m3_s = mean_flow_ml_s * 1e-6
        boundary = boundary_by_node.get(node_id, {})
        total_resistance = _optional_float(boundary.get("resistance_pa_s_per_m3"))
        if total_resistance is None:
            total_resistance = _optional_float(node.get("boundary_resistance_pa_s_per_m3"))
        if total_resistance is None or total_resistance <= 0.0:
            total_resistance = 8000.0 / mean_flow_m3_s

        proximal = max(min(proximal_resistance_fraction, 0.95), 0.0) * total_resistance
        distal = max(total_resistance - proximal, total_resistance * 0.05)
        compliance = max(time_constant_s, 1e-6) / distal
        mean_node_pressure = _safe_float(node.get("pressure_pa"))
        distal_pressure = mean_node_pressure - total_resistance * mean_flow_m3_s
        initial_capacitor_pressure = distal_pressure + distal * mean_flow_m3_s
        models[node_id] = TerminalRCRModel(
            node_id=node_id,
            label=str(node.get("label", node_id)),
            mean_flow_ml_s=mean_flow_ml_s,
            total_resistance_pa_s_per_m3=total_resistance,
            proximal_resistance_pa_s_per_m3=proximal,
            distal_resistance_pa_s_per_m3=distal,
            compliance_m3_per_pa=compliance,
            time_constant_s=time_constant_s,
            distal_pressure_pa=distal_pressure,
            initial_capacitor_pressure_pa=initial_capacitor_pressure,
        )
    return models


def _boundary_flows_for_domain(
    nodes: list[dict[str, Any]],
    multiplier: float,
) -> dict[str, float]:
    flows = {
        str(node["node_id"]): _safe_float(node.get("boundary_flow_ml_s")) * multiplier
        for node in nodes
        if abs(_safe_float(node.get("boundary_flow_ml_s"))) > 0.0
    }
    net_flow = sum(flows.values())
    negative_nodes = [node_id for node_id, flow in flows.items() if flow < 0.0]
    if abs(net_flow) > 1e-9 and negative_nodes:
        negative_total = abs(sum(flows[node_id] for node_id in negative_nodes))
        if negative_total > 0.0:
            for node_id in negative_nodes:
                flows[node_id] -= net_flow * (abs(flows[node_id]) / negative_total)
    return flows


def _node_reference_pressures(
    vessel_type: str,
    nodes: list[dict[str, Any]],
    terminal_pressure_by_node: dict[str, float],
    venous_outlet_pressure_pa: float,
    fallback_pressure_pa: float,
) -> dict[str, float]:
    references: dict[str, float] = {}
    if vessel_type == "arterial":
        references.update(terminal_pressure_by_node)
    elif vessel_type == "venous":
        for node in nodes:
            if str(node.get("boundary_role")) == "venous_outlet":
                references[str(node["node_id"])] = venous_outlet_pressure_pa

    if not references and nodes:
        references[str(nodes[0]["node_id"])] = fallback_pressure_pa
    return references


def _write_edge_timeseries(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s",
        "phase",
        "edge_id",
        "vessel_type",
        "flow_role",
        "flow_ml_s",
        "mean_velocity_cm_s",
        "pressure_source_pa",
        "pressure_target_pa",
        "pressure_drop_pa",
        "pressure_source_mmhg",
        "pressure_target_mmhg",
        "pressure_equation_residual_pa",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_node_timeseries(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s",
        "phase",
        "node_id",
        "vessel_type",
        "role",
        "boundary_role",
        "pressure_pa",
        "pressure_mmhg",
        "boundary_flow_ml_s",
        "incoming_edge_flow_ml_s",
        "outgoing_edge_flow_ml_s",
        "mass_balance_residual_ml_s",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_boundary_timeseries(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s",
        "phase",
        "boundary_id",
        "node_id",
        "label",
        "vessel_type",
        "role",
        "flow_ml_s",
        "pressure_pa",
        "pressure_mmhg",
        "rcr_terminal_pressure_pa",
        "rcr_capacitor_pressure_pa",
        "rcr_distal_pressure_pa",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_boundary_summary(path: Path, rows: list[dict[str, Any]], rcr_models: dict[str, TerminalRCRModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["node_id"]), []).append(row)

    fields = [
        "node_id",
        "label",
        "vessel_type",
        "role",
        "mean_flow_ml_s",
        "min_flow_ml_s",
        "max_flow_ml_s",
        "mean_pressure_pa",
        "min_pressure_pa",
        "max_pressure_pa",
        "mean_pressure_mmhg",
        "min_pressure_mmhg",
        "max_pressure_mmhg",
        "rcr_total_resistance_pa_s_per_m3",
        "rcr_compliance_m3_per_pa",
        "rcr_time_constant_s",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for node_id, node_rows in sorted(grouped.items()):
            flow = np.array([float(row["flow_ml_s"]) for row in node_rows], dtype=float)
            pressure = np.array([float(row["pressure_pa"]) for row in node_rows], dtype=float)
            first = node_rows[0]
            rcr = rcr_models.get(node_id)
            writer.writerow(
                {
                    "node_id": node_id,
                    "label": first["label"],
                    "vessel_type": first["vessel_type"],
                    "role": first["role"],
                    "mean_flow_ml_s": f"{float(flow.mean()):.6f}",
                    "min_flow_ml_s": f"{float(flow.min()):.6f}",
                    "max_flow_ml_s": f"{float(flow.max()):.6f}",
                    "mean_pressure_pa": f"{float(pressure.mean()):.6f}",
                    "min_pressure_pa": f"{float(pressure.min()):.6f}",
                    "max_pressure_pa": f"{float(pressure.max()):.6f}",
                    "mean_pressure_mmhg": f"{float(pressure.mean() / PA_PER_MMHG):.6f}",
                    "min_pressure_mmhg": f"{float(pressure.min() / PA_PER_MMHG):.6f}",
                    "max_pressure_mmhg": f"{float(pressure.max() / PA_PER_MMHG):.6f}",
                    "rcr_total_resistance_pa_s_per_m3": "" if rcr is None else f"{rcr.total_resistance_pa_s_per_m3:.6f}",
                    "rcr_compliance_m3_per_pa": "" if rcr is None else f"{rcr.compliance_m3_per_pa:.12e}",
                    "rcr_time_constant_s": "" if rcr is None else f"{rcr.time_constant_s:.6f}",
                }
            )


def _series(rows: list[dict[str, Any]], key: str, value: str, match_field: str = "node_id") -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if str(row.get(match_field)) == key]
    return (
        np.array([float(row["time_s"]) for row in selected], dtype=float),
        np.array([float(row[value]) for row in selected], dtype=float),
    )


def _write_plots(
    output_dir: Path,
    case_id: str,
    phases: np.ndarray,
    arterial_multiplier: np.ndarray,
    venous_multiplier: np.ndarray,
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    plt, _ = _import_dependencies()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    waveform_path = plot_dir / f"{case_id}_pulsatile_waveforms_v001.png"
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(phases, arterial_multiplier, color="#b21f2d", linewidth=2.5, label="arterial inlet multiplier")
    ax.plot(phases, venous_multiplier, color="#235a97", linewidth=2.5, label="venous return multiplier")
    ax.set_xlabel("Cardiac phase")
    ax.set_ylabel("Normalized flow multiplier")
    ax.set_title("Pulsatile Boundary Waveforms")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(waveform_path, dpi=180)
    plt.close(fig)
    paths.append(str(waveform_path))

    preview_path = plot_dir / f"{case_id}_pulsatile_pressure_flow_preview_v001.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    time, aorta_flow = _series(boundary_rows, "aorta_inlet", "flow_ml_s")
    if len(time):
        axes[0].plot(time, aorta_flow, color="#b21f2d", linewidth=2.2, label="aorta inlet")
    time, ivc_flow = _series(boundary_rows, "ivc_outlet", "flow_ml_s")
    if len(time):
        axes[0].plot(time, np.abs(ivc_flow), color="#235a97", linewidth=2.2, label="IVC return outlet")
    axes[0].set_ylabel("Flow (mL/s)")
    axes[0].set_title("Boundary Flow")
    axes[0].grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels)

    for node_id, color, label in [
        ("aorta_inlet", "#b21f2d", "aorta inlet"),
        ("visceral_branch_origin", "#c77700", "visceral origin"),
        ("ivc_outlet", "#235a97", "IVC outlet"),
    ]:
        time, pressure = _series(node_rows, node_id, "pressure_mmhg")
        if len(time):
            axes[1].plot(time, pressure, linewidth=2.0, color=color, label=label)
    axes[1].set_ylabel("Pressure (mmHg)")
    axes[1].set_title("Selected Node Pressures")
    axes[1].grid(True, alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels)

    for edge_id, color, label in [
        ("aorta_inlet_to_descending", "#b21f2d", "aorta trunk"),
        ("bifurcation_to_left_common_iliac", "#d27a22", "left iliac"),
        ("ivc_hepatic_to_outlet", "#235a97", "IVC outlet segment"),
    ]:
        time, flow = _series(edge_rows, edge_id, "flow_ml_s", match_field="edge_id")
        if len(time):
            axes[2].plot(time, np.abs(flow), linewidth=2.0, color=color, label=label)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Flow magnitude (mL/s)")
    axes[2].set_title("Selected Segment Flows")
    axes[2].grid(True, alpha=0.25)
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels)

    fig.tight_layout()
    fig.savefig(preview_path, dpi=180)
    plt.close(fig)
    paths.append(str(preview_path))
    return tuple(paths)


def _rcr_to_payload(model: TerminalRCRModel) -> dict[str, Any]:
    return {
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


def _write_model_yaml(path: Path, result: PulsatileFlowResult, rcr_models: dict[str, TerminalRCRModel]) -> None:
    _, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "model_type": "first_pass_pulsatile_1d_flow_network",
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
            "one_way_pulsatile_flow_scaling_from_boundary_waveform",
            "fixed_mean_flow_splits_from_steady_1d_solution",
            "arterial_terminal_outlets_use_first_pass_RCR_windkessel_placeholders",
            "venous_return_uses_damped_phase_lagged_waveform",
            "vessel_walls_remain_rigid_no_wave_speed_or_compliance_along_segments",
        ],
        "outputs": {
            "edge_timeseries_csv": result.edge_timeseries_csv_path,
            "node_timeseries_csv": result.node_timeseries_csv_path,
            "boundary_timeseries_csv": result.boundary_timeseries_csv_path,
            "boundary_summary_csv": result.boundary_summary_csv_path,
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
            "venous_outlet_pressure_mean_pa": result.venous_outlet_pressure_mean_pa,
            "max_abs_mass_balance_residual_ml_s": result.max_abs_mass_balance_residual_ml_s,
            "max_abs_pressure_equation_residual_pa": result.max_abs_pressure_equation_residual_pa,
        },
        "terminal_rcr_models": [_rcr_to_payload(model) for model in rcr_models.values()],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: PulsatileFlowResult, rcr_models: dict[str, TerminalRCRModel]) -> str:
    lines = [
        "# Pulsatile 1D Flow Model Stage 001",
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
        f"- Venous outlet pressure mean: {result.venous_outlet_pressure_mean_pa / PA_PER_MMHG:.2f} mmHg",
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
    ]
    for plot_path in result.plot_paths:
        lines.append(f"- Plot: `{Path(plot_path).name}`")

    lines.extend(
        [
            "",
            "## Terminal RCR Placeholders",
            "",
            "| outlet node | mean flow mL/s | R total Pa*s/m3 | C m3/Pa | distal pressure mmHg |",
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
            "- This is a digital-only pulsatile vascular simulation scaffold, not a calibrated hemodynamic model.",
            "- Edge flows preserve the steady 1D flow split while applying the arterial waveform over one cardiac cycle.",
            "- Arterial outlets use first-pass RCR/Windkessel placeholders so outlet pressures can evolve over time.",
            "- Venous return is represented by a damped, phase-lagged copy of the arterial waveform.",
            "- The next upgrade is a coupled pressure-flow solve where outlet RCR states determine branch flow splits dynamically.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_pulsatile_flow_model(
    flow_1d_model_path: str | Path,
    boundary_config_path: str | Path,
    output_dir: str | Path = "outputs/sim/flow_pulsatile",
    case_id: str = "ct_org_case0_imagetbad_case125",
    heart_rate_bpm: float = 60.0,
    samples_per_cycle: int = 160,
    settling_cycles: int = 3,
    rcr_proximal_resistance_fraction: float = 0.1,
    rcr_time_constant_s: float = 1.2,
    venous_pulsatility_fraction: float = 0.35,
    venous_phase_lag_fraction: float = 0.15,
    pressure_reference_weight: float = 50.0,
    report_path: str | Path | None = "outputs/reports/flow_pulsatile_model_stage001.md",
) -> PulsatileFlowResult:
    flow_model_path = Path(flow_1d_model_path)
    boundary_path = Path(boundary_config_path)
    model = _load_yaml(flow_model_path)
    boundary_config = _load_yaml(boundary_path)
    boundary_by_node = _boundary_payload_by_node(boundary_config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if samples_per_cycle < 8:
        raise ValueError("samples_per_cycle must be at least 8")
    if heart_rate_bpm <= 0:
        raise ValueError("heart_rate_bpm must be positive")

    cycle_duration_s = 60.0 / heart_rate_bpm
    dt = cycle_duration_s / samples_per_cycle
    phases, arterial_multiplier = _waveform_samples(boundary_config, samples_per_cycle)
    venous_multiplier = _lagged_venous_multiplier(
        phases,
        arterial_multiplier,
        venous_pulsatility_fraction=venous_pulsatility_fraction,
        venous_phase_lag_fraction=venous_phase_lag_fraction,
    )

    arterial_nodes = _domain_nodes(model, "arterial")
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
    max_mass_residual = 0.0
    max_pressure_residual = 0.0
    vessel_types = _vessel_types(model)
    venous_outlet_pressure = float(model.get("inputs", {}).get("venous_outlet_pressure_pa", 667.0))
    arterial_reference = float(model.get("inputs", {}).get("arterial_inlet_pressure_pa", 13332.0))
    total_steps = (max(settling_cycles, 0) + 1) * samples_per_cycle
    first_recorded_step = max(settling_cycles, 0) * samples_per_cycle

    for global_step in range(total_steps):
        sample_index = global_step % samples_per_cycle
        record_step = global_step >= first_recorded_step
        output_sample_index = global_step - first_recorded_step
        phase = float(phases[sample_index])
        time_s = float(output_sample_index * dt)
        multiplier_by_domain = {
            "arterial": float(arterial_multiplier[sample_index]),
            "venous": float(venous_multiplier[sample_index]),
        }

        flow_by_edge: dict[str, float] = {}
        residual_by_node: dict[str, float] = {}
        pressure_by_node: dict[str, float] = {}
        pressure_residual_by_edge: dict[str, float] = {}
        boundary_flow_by_node_all: dict[str, float] = {}
        terminal_pressure_by_node: dict[str, float] = {}
        capacitor_pressure_snapshot: dict[str, float] = {}

        for vessel_type in vessel_types:
            segments = _domain_segments(model, vessel_type)
            nodes = _domain_nodes(model, vessel_type)
            node_ids = tuple(str(node["node_id"]) for node in nodes)
            multiplier = multiplier_by_domain.get(vessel_type, float(arterial_multiplier[sample_index]))
            boundary_flow_by_node = _boundary_flows_for_domain(nodes, multiplier)
            boundary_flow_by_node_all.update(boundary_flow_by_node)

            if vessel_type == "arterial":
                for node_id, rcr in rcr_models.items():
                    q_m3_s = abs(boundary_flow_by_node.get(node_id, 0.0)) * 1e-6
                    capacitor_pressure = capacitor_pressure_by_node[node_id]
                    terminal_pressure = capacitor_pressure + rcr.proximal_resistance_pa_s_per_m3 * q_m3_s
                    terminal_pressure_by_node[node_id] = terminal_pressure
                    capacitor_pressure_snapshot[node_id] = capacitor_pressure

            domain_flow_by_edge, domain_residual_by_node = _solve_edge_flows(node_ids, segments, boundary_flow_by_node)
            flow_by_edge.update(domain_flow_by_edge)
            residual_by_node.update(domain_residual_by_node)
            if domain_residual_by_node:
                max_mass_residual = max(max_mass_residual, max(abs(value) for value in domain_residual_by_node.values()))

            references = _node_reference_pressures(
                vessel_type,
                nodes,
                terminal_pressure_by_node=terminal_pressure_by_node,
                venous_outlet_pressure_pa=venous_outlet_pressure,
                fallback_pressure_pa=arterial_reference,
            )
            domain_pressure_by_node, domain_pressure_residual_by_edge = _solve_pressures(
                node_ids,
                segments,
                domain_flow_by_edge,
                references,
                reference_weight=pressure_reference_weight,
            )
            pressure_by_node.update(domain_pressure_by_node)
            pressure_residual_by_edge.update(domain_pressure_residual_by_edge)
            if domain_pressure_residual_by_edge:
                max_pressure_residual = max(
                    max_pressure_residual,
                    max(abs(value) for value in domain_pressure_residual_by_edge.values()),
                )

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
                boundary_flow = boundary_flow_by_node_all.get(node_id, 0.0)
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
                    boundary_id = boundary.get("boundary_id", "")
                    rcr = rcr_models.get(node_id)
                    rcr_terminal = terminal_pressure_by_node.get(node_id)
                    capacitor = capacitor_pressure_snapshot.get(node_id)
                    boundary_rows.append(
                        {
                            "time_s": f"{time_s:.6f}",
                            "phase": f"{phase:.6f}",
                            "boundary_id": boundary_id,
                            "node_id": node_id,
                            "label": boundary.get("label", node.get("label", node_id)),
                            "vessel_type": node.get("vessel_type", ""),
                            "role": node.get("boundary_role", ""),
                            "flow_ml_s": f"{boundary_flow:.6f}",
                            "pressure_pa": f"{pressure:.6f}",
                            "pressure_mmhg": f"{pressure / PA_PER_MMHG:.6f}",
                            "rcr_terminal_pressure_pa": "" if rcr_terminal is None else f"{rcr_terminal:.6f}",
                            "rcr_capacitor_pressure_pa": "" if capacitor is None else f"{capacitor:.6f}",
                            "rcr_distal_pressure_pa": "" if rcr is None else f"{rcr.distal_pressure_pa:.6f}",
                        }
                    )

        for node_id, rcr in rcr_models.items():
            q_m3_s = abs(boundary_flow_by_node_all.get(node_id, 0.0)) * 1e-6
            alpha = math.exp(-dt / max(rcr.time_constant_s, 1e-9))
            previous = capacitor_pressure_by_node[node_id]
            capacitor_pressure_by_node[node_id] = (
                rcr.distal_pressure_pa
                + (previous - rcr.distal_pressure_pa) * alpha
                + rcr.distal_resistance_pa_s_per_m3 * q_m3_s * (1.0 - alpha)
            )

    base = output / case_id
    model_yaml = base.with_name(f"{case_id}_pulsatile_flow_model_v001.yaml")
    edge_csv = base.with_name(f"{case_id}_pulsatile_edge_timeseries_v001.csv")
    node_csv = base.with_name(f"{case_id}_pulsatile_node_timeseries_v001.csv")
    boundary_csv = base.with_name(f"{case_id}_pulsatile_boundary_timeseries_v001.csv")
    boundary_summary_csv = base.with_name(f"{case_id}_pulsatile_boundary_summary_v001.csv")
    report = Path(report_path) if report_path else base.with_name(f"{case_id}_pulsatile_flow_model_report_v001.md")

    _write_edge_timeseries(edge_csv, edge_rows)
    _write_node_timeseries(node_csv, node_rows)
    _write_boundary_timeseries(boundary_csv, boundary_rows)
    _write_boundary_summary(boundary_summary_csv, boundary_rows, rcr_models)
    plot_paths = _write_plots(output, case_id, phases, arterial_multiplier, venous_multiplier, node_rows, edge_rows, boundary_rows)

    _, aorta_flow = _series(boundary_rows, "aorta_inlet", "flow_ml_s")
    _, aorta_pressure = _series(node_rows, "aorta_inlet", "pressure_pa")
    _, venous_outlet_pressure_values = _series(node_rows, "ivc_outlet", "pressure_pa")
    if aorta_flow.size == 0:
        aorta_flow = np.array([0.0])
    if aorta_pressure.size == 0:
        aorta_pressure = np.array([0.0])
    if venous_outlet_pressure_values.size == 0:
        venous_outlet_pressure_values = np.array([venous_outlet_pressure])

    result = PulsatileFlowResult(
        case_id=case_id,
        output_dir=str(output),
        model_yaml_path=str(model_yaml),
        edge_timeseries_csv_path=str(edge_csv),
        node_timeseries_csv_path=str(node_csv),
        boundary_timeseries_csv_path=str(boundary_csv),
        boundary_summary_csv_path=str(boundary_summary_csv),
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
        venous_outlet_pressure_mean_pa=float(venous_outlet_pressure_values.mean()),
        max_abs_mass_balance_residual_ml_s=max_mass_residual,
        max_abs_pressure_equation_residual_pa=max_pressure_residual,
        notes=(
            "pulsatile_model_uses_existing_steady_1d_flow_splits",
            "arterial_outlets_have_RCR_placeholder_dynamics",
            "venous_return_is_damped_and_phase_lagged",
            "not_calibrated_for_patient_specific_or_clinical_use",
        ),
    )
    _write_model_yaml(model_yaml, result, rcr_models)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, rcr_models))
    return result


def format_pulsatile_flow_result(result: PulsatileFlowResult) -> str:
    lines = [
        "Pulsatile 1D flow model created",
        f"Case ID: {result.case_id}",
        f"Heart rate: {result.heart_rate_bpm:.1f} bpm",
        f"Samples per cycle: {result.samples_per_cycle}",
        f"Edges/nodes/boundaries: {result.edge_count}/{result.node_count}/{result.boundary_count}",
        f"Terminal RCR outlets: {result.terminal_rcr_count}",
        f"Aorta flow mean/min/max: {result.arterial_inlet_flow_mean_ml_s:.3f}/{result.arterial_inlet_flow_min_ml_s:.3f}/{result.arterial_inlet_flow_max_ml_s:.3f} mL/s",
        f"Aorta pressure mean/min/max: {result.aorta_pressure_mean_pa / PA_PER_MMHG:.2f}/{result.aorta_pressure_min_pa / PA_PER_MMHG:.2f}/{result.aorta_pressure_max_pa / PA_PER_MMHG:.2f} mmHg",
        f"Max mass-balance residual: {result.max_abs_mass_balance_residual_ml_s:.9f} mL/s",
        f"Max pressure residual: {result.max_abs_pressure_equation_residual_pa:.9f} Pa",
        f"Model YAML: {result.model_yaml_path}",
        f"Edge time series: {result.edge_timeseries_csv_path}",
        f"Node time series: {result.node_timeseries_csv_path}",
        f"Boundary time series: {result.boundary_timeseries_csv_path}",
        f"Boundary summary: {result.boundary_summary_csv_path}",
        f"Report: {result.report_path}",
    ]
    for plot_path in result.plot_paths:
        lines.append(f"Plot: {plot_path}")
    return "\n".join(lines)
