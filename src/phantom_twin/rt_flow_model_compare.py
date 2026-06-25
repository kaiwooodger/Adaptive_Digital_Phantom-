from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import numpy as np
import yaml


SELECTED_MASKS = (
    "target_ptv_synthetic_vertebral",
    "target_gtv_synthetic_vertebral",
    "vascular_fluid",
    "vessel_wall",
    "oar_lungs",
    "oar_liver",
    "oar_kidneys",
    "oar_bone",
    "body",
)


@dataclass(frozen=True)
class RTFlowMetricComparison:
    mask_id: str
    label: str
    role: str
    state_pair: str
    scalar_state: str
    spatial_state: str
    scalar_mean_dose_gy: float
    spatial_mean_dose_gy: float
    delta_mean_dose_gy: float
    scalar_d95_gy: float
    spatial_d95_gy: float
    delta_d95_gy: float
    scalar_v95_percent: float
    spatial_v95_percent: float
    delta_v95_percentage_points: float


@dataclass(frozen=True)
class RTFlowVolumeComparison:
    state_pair: str
    scalar_state: str
    spatial_state: str
    scalar_dose_path: str
    spatial_dose_path: str
    compared_voxels: int
    mean_signed_diff_gy: float
    mean_abs_diff_gy: float
    p95_abs_diff_gy: float
    max_abs_diff_gy: float
    mean_signed_diff_percent_of_prescription: float
    max_abs_diff_percent_of_prescription: float
    correlation: float


@dataclass(frozen=True)
class RTFlowModelComparisonResult:
    case_id: str
    output_dir: str
    scalar_planning_spec_path: str
    spatial_dose_spec_path: str
    metric_comparison_csv_path: str
    volume_comparison_csv_path: str
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    prescription_dose_gy: float
    metric_comparisons: tuple[RTFlowMetricComparison, ...]
    volume_comparisons: tuple[RTFlowVolumeComparison, ...]
    notes: tuple[str, ...]


def _import_medical_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RT flow model comparison requires matplotlib and nibabel.") from exc
    return plt, nib


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(raw_path: str | Path | None, reference_yaml_path: Path | None = None) -> Path | None:
    if raw_path is None or str(raw_path) == "":
        return None
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    if reference_yaml_path is not None:
        candidate = reference_yaml_path.parent / path
        if candidate.exists():
            return candidate
    return path


def _read_csv_rows(path: str | Path | None, reference_yaml_path: Path | None = None) -> list[dict[str, str]]:
    resolved = _resolve_path(path, reference_yaml_path)
    if resolved is None or not resolved.exists():
        return []
    with resolved.open(newline="") as csvfile:
        return [dict(row) for row in csv.DictReader(csvfile)]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _rows_by_key(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row.get(key, "") for key in keys): row for row in rows}


def _state_pairs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("static", "static", "static"),
        ("pulsatile_mean", "spatial_pulsatile_mean", "mean"),
        ("pulsatile_peak", "spatial_pulsatile_peak", "peak"),
        ("pulsatile_trough", "spatial_pulsatile_trough", "trough"),
    )


def _build_metric_comparisons(
    scalar_rows: list[dict[str, str]],
    spatial_rows: list[dict[str, str]],
) -> tuple[RTFlowMetricComparison, ...]:
    scalar_by_key = _rows_by_key(scalar_rows, ("mask_id", "state"))
    spatial_by_key = _rows_by_key(spatial_rows, ("mask_id", "state"))
    comparisons: list[RTFlowMetricComparison] = []
    for scalar_state, spatial_state, state_pair in _state_pairs():
        for mask_id in SELECTED_MASKS:
            scalar = scalar_by_key.get((mask_id, scalar_state))
            spatial = spatial_by_key.get((mask_id, spatial_state))
            if scalar is None or spatial is None:
                continue
            scalar_mean = _as_float(scalar.get("mean_dose_gy"))
            spatial_mean = _as_float(spatial.get("mean_dose_gy"))
            scalar_d95 = _as_float(scalar.get("d95_gy"))
            spatial_d95 = _as_float(spatial.get("d95_gy"))
            scalar_v95 = _as_float(scalar.get("v95_percent"))
            spatial_v95 = _as_float(spatial.get("v95_percent"))
            comparisons.append(
                RTFlowMetricComparison(
                    mask_id=mask_id,
                    label=str(spatial.get("label") or scalar.get("label") or mask_id),
                    role=str(spatial.get("role") or scalar.get("role") or ""),
                    state_pair=state_pair,
                    scalar_state=scalar_state,
                    spatial_state=spatial_state,
                    scalar_mean_dose_gy=scalar_mean,
                    spatial_mean_dose_gy=spatial_mean,
                    delta_mean_dose_gy=spatial_mean - scalar_mean,
                    scalar_d95_gy=scalar_d95,
                    spatial_d95_gy=spatial_d95,
                    delta_d95_gy=spatial_d95 - scalar_d95,
                    scalar_v95_percent=scalar_v95,
                    spatial_v95_percent=spatial_v95,
                    delta_v95_percentage_points=spatial_v95 - scalar_v95,
                )
            )
    return tuple(comparisons)


def _dose_paths(scalar_spec: dict[str, Any], spatial_spec: dict[str, Any]) -> dict[str, tuple[str, str, str, str]]:
    scalar_outputs = scalar_spec.get("outputs", {})
    spatial_outputs = spatial_spec.get("outputs", {})
    if not isinstance(scalar_outputs, dict) or not isinstance(spatial_outputs, dict):
        raise ValueError("Both specs must contain output mappings")
    return {
        "peak": (
            "pulsatile_peak",
            "spatial_pulsatile_peak",
            str(scalar_outputs.get("pulsatile_peak_dose_nifti", "")),
            str(spatial_outputs.get("spatial_peak_dose_nifti", "")),
        ),
        "trough": (
            "pulsatile_trough",
            "spatial_pulsatile_trough",
            str(scalar_outputs.get("pulsatile_trough_dose_nifti", "")),
            str(spatial_outputs.get("spatial_trough_dose_nifti", "")),
        ),
    }


def _load_dose(path: Path):
    _, nib = _import_medical_dependencies()
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(np.float32), image


def _compare_volume_pair(
    state_pair: str,
    scalar_state: str,
    spatial_state: str,
    scalar_path: Path,
    spatial_path: Path,
    prescription_dose_gy: float,
) -> RTFlowVolumeComparison:
    scalar, _ = _load_dose(scalar_path)
    spatial, _ = _load_dose(spatial_path)
    if scalar.shape != spatial.shape:
        raise ValueError(f"Dose shapes differ for {state_pair}: {scalar.shape} vs {spatial.shape}")
    mask = np.isfinite(scalar) & np.isfinite(spatial) & ((scalar > 0.0) | (spatial > 0.0))
    scalar_values = scalar[mask].astype(np.float64)
    spatial_values = spatial[mask].astype(np.float64)
    diff = spatial_values - scalar_values
    abs_diff = np.abs(diff)
    if scalar_values.size > 1 and np.std(scalar_values) > 0.0 and np.std(spatial_values) > 0.0:
        correlation = float(np.corrcoef(scalar_values, spatial_values)[0, 1])
    else:
        correlation = 1.0
    norm = max(float(prescription_dose_gy), 1e-9)
    return RTFlowVolumeComparison(
        state_pair=state_pair,
        scalar_state=scalar_state,
        spatial_state=spatial_state,
        scalar_dose_path=str(scalar_path),
        spatial_dose_path=str(spatial_path),
        compared_voxels=int(mask.sum()),
        mean_signed_diff_gy=float(np.mean(diff)) if diff.size else 0.0,
        mean_abs_diff_gy=float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        p95_abs_diff_gy=float(np.percentile(abs_diff, 95.0)) if abs_diff.size else 0.0,
        max_abs_diff_gy=float(np.max(abs_diff)) if abs_diff.size else 0.0,
        mean_signed_diff_percent_of_prescription=float(np.mean(diff) / norm * 100.0) if diff.size else 0.0,
        max_abs_diff_percent_of_prescription=float(np.max(abs_diff) / norm * 100.0) if abs_diff.size else 0.0,
        correlation=correlation,
    )


def _write_metric_csv(path: Path, rows: tuple[RTFlowMetricComparison, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RTFlowMetricComparison.__dataclass_fields__.keys())
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: f"{getattr(row, field):.9g}" if isinstance(getattr(row, field), float) else getattr(row, field)
                    for field in fields
                }
            )


def _write_volume_csv(path: Path, rows: tuple[RTFlowVolumeComparison, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RTFlowVolumeComparison.__dataclass_fields__.keys())
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: f"{getattr(row, field):.9g}" if isinstance(getattr(row, field), float) else getattr(row, field)
                    for field in fields
                }
            )


def _metric_delta(
    rows: tuple[RTFlowMetricComparison, ...],
    mask_id: str,
    state_pair: str,
    field: str,
) -> float:
    for row in rows:
        if row.mask_id == mask_id and row.state_pair == state_pair:
            return float(getattr(row, field))
    return 0.0


def _write_preview(
    path: Path,
    scalar_spec_path: Path,
    spatial_spec_path: Path,
    scalar_spec: dict[str, Any],
    spatial_spec: dict[str, Any],
    metric_rows: tuple[RTFlowMetricComparison, ...],
) -> None:
    plt, _ = _import_medical_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = _dose_paths(scalar_spec, spatial_spec)
    scalar_peak_path = _resolve_path(paths["peak"][2], scalar_spec_path)
    spatial_peak_path = _resolve_path(paths["peak"][3], spatial_spec_path)
    scalar_trough_path = _resolve_path(paths["trough"][2], scalar_spec_path)
    spatial_trough_path = _resolve_path(paths["trough"][3], spatial_spec_path)
    if scalar_peak_path is None or spatial_peak_path is None or scalar_trough_path is None or spatial_trough_path is None:
        raise ValueError("Missing peak/trough dose paths for preview")
    scalar_peak, _ = _load_dose(scalar_peak_path)
    spatial_peak, _ = _load_dose(spatial_peak_path)
    scalar_trough, _ = _load_dose(scalar_trough_path)
    spatial_trough, _ = _load_dose(spatial_trough_path)
    peak_diff_mgy = (spatial_peak - scalar_peak) * 1000.0
    trough_diff_mgy = (spatial_trough - scalar_trough) * 1000.0
    z_index = int(np.argmax(np.max(np.abs(peak_diff_mgy), axis=(0, 1))))
    vmax = float(max(np.percentile(np.abs(peak_diff_mgy), 99.8), np.percentile(np.abs(trough_diff_mgy), 99.8), 1.0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Scalar vs Spatial RT-Flow Dose Model", fontsize=16, fontweight="bold")
    panels = [
        (axes[0, 0], peak_diff_mgy[:, :, z_index], "Spatial - scalar peak (mGy)"),
        (axes[0, 1], trough_diff_mgy[:, :, z_index], "Spatial - scalar trough (mGy)"),
    ]
    for ax, image, title in panels:
        im = ax.imshow(image.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    peak_mask = np.isfinite(peak_diff_mgy) & ((scalar_peak > 0.0) | (spatial_peak > 0.0))
    trough_mask = np.isfinite(trough_diff_mgy) & ((scalar_trough > 0.0) | (spatial_trough > 0.0))
    axes[0, 2].hist(peak_diff_mgy[peak_mask].ravel(), bins=80, alpha=0.65, label="peak", color="#c0392b")
    axes[0, 2].hist(trough_diff_mgy[trough_mask].ravel(), bins=80, alpha=0.65, label="trough", color="#2471a3")
    axes[0, 2].set_title("Spatial - scalar voxel differences")
    axes[0, 2].set_xlabel("Dose difference (mGy)")
    axes[0, 2].set_ylabel("voxels")
    axes[0, 2].set_yscale("log")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)

    masks = ["target_ptv_synthetic_vertebral", "vascular_fluid", "vessel_wall", "oar_bone", "body"]
    labels = ["PTV", "vascular fluid", "vessel wall", "bone", "body"]
    x = np.arange(len(labels))
    width = 0.38
    axes[1, 0].bar(x - width / 2, [_metric_delta(metric_rows, mask, "peak", "delta_mean_dose_gy") * 1000.0 for mask in masks], width, label="peak")
    axes[1, 0].bar(x + width / 2, [_metric_delta(metric_rows, mask, "trough", "delta_mean_dose_gy") * 1000.0 for mask in masks], width, label="trough")
    axes[1, 0].axhline(0.0, color="#1f2933", linewidth=0.8)
    axes[1, 0].set_title("Mean dose: spatial - scalar")
    axes[1, 0].set_ylabel("mGy")
    axes[1, 0].set_xticks(x, labels, rotation=25, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(x - width / 2, [_metric_delta(metric_rows, mask, "peak", "delta_d95_gy") * 1000.0 for mask in masks], width, label="peak")
    axes[1, 1].bar(x + width / 2, [_metric_delta(metric_rows, mask, "trough", "delta_d95_gy") * 1000.0 for mask in masks], width, label="trough")
    axes[1, 1].axhline(0.0, color="#1f2933", linewidth=0.8)
    axes[1, 1].set_title("D95: spatial - scalar")
    axes[1, 1].set_ylabel("mGy")
    axes[1, 1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)

    axes[1, 2].bar(x - width / 2, [_metric_delta(metric_rows, mask, "peak", "delta_v95_percentage_points") for mask in masks], width, label="peak")
    axes[1, 2].bar(x + width / 2, [_metric_delta(metric_rows, mask, "trough", "delta_v95_percentage_points") for mask in masks], width, label="trough")
    axes[1, 2].axhline(0.0, color="#1f2933", linewidth=0.8)
    axes[1, 2].set_title("V95: spatial - scalar")
    axes[1, 2].set_ylabel("percentage points")
    axes[1, 2].set_xticks(x, labels, rotation=25, ha="right")
    axes[1, 2].legend()
    axes[1, 2].grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _write_spec(path: Path, result: RTFlowModelComparisonResult) -> None:
    payload = {
        "case_id": result.case_id,
        "comparison_type": "scalar_vs_spatial_rt_flow_dose_model",
        "inputs": {
            "scalar_planning_spec": result.scalar_planning_spec_path,
            "spatial_dose_spec": result.spatial_dose_spec_path,
        },
        "outputs": {
            "metric_comparison_csv": result.metric_comparison_csv_path,
            "volume_comparison_csv": result.volume_comparison_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "summary": {
            "prescription_dose_gy": result.prescription_dose_gy,
            "metric_comparison_count": len(result.metric_comparisons),
            "volume_comparison_count": len(result.volume_comparisons),
            "max_abs_volume_difference_gy": max((row.max_abs_diff_gy for row in result.volume_comparisons), default=0.0),
            "ptv_peak_mean_delta_gy": _metric_delta(
                result.metric_comparisons,
                "target_ptv_synthetic_vertebral",
                "peak",
                "delta_mean_dose_gy",
            ),
            "ptv_peak_v95_delta_percentage_points": _metric_delta(
                result.metric_comparisons,
                "target_ptv_synthetic_vertebral",
                "peak",
                "delta_v95_percentage_points",
            ),
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: RTFlowModelComparisonResult) -> str:
    selected_metric_rows = [
        row
        for row in result.metric_comparisons
        if row.mask_id in {"target_ptv_synthetic_vertebral", "vascular_fluid", "vessel_wall"} and row.state_pair in {"peak", "trough"}
    ]
    lines = [
        "# Scalar vs Spatial RT-Flow Dose Comparison",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Prescription dose: {result.prescription_dose_gy:.2f} Gy",
        f"- Metric comparisons: {len(result.metric_comparisons)}",
        f"- Volume comparisons: {len(result.volume_comparisons)}",
        f"- Max absolute voxel difference: {max((row.max_abs_diff_gy for row in result.volume_comparisons), default=0.0) * 1000.0:.3f} mGy",
        f"- PTV peak mean dose delta: {_metric_delta(result.metric_comparisons, 'target_ptv_synthetic_vertebral', 'peak', 'delta_mean_dose_gy') * 1000.0:.3f} mGy",
        f"- PTV peak V95 delta: {_metric_delta(result.metric_comparisons, 'target_ptv_synthetic_vertebral', 'peak', 'delta_v95_percentage_points'):.3f} percentage points",
        "",
        "## Outputs",
        "",
        f"- Metric comparison CSV: `{Path(result.metric_comparison_csv_path).name}`",
        f"- Volume comparison CSV: `{Path(result.volume_comparison_csv_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Spec YAML: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Key Metric Deltas",
        "",
        "| mask | state | scalar mean Gy | spatial mean Gy | delta mean mGy | delta D95 mGy | delta V95 pp |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected_metric_rows:
        lines.append(
            f"| `{row.mask_id}` | `{row.state_pair}` | {row.scalar_mean_dose_gy:.4f} | {row.spatial_mean_dose_gy:.4f} | "
            f"{1000.0 * row.delta_mean_dose_gy:.3f} | {1000.0 * row.delta_d95_gy:.3f} | {row.delta_v95_percentage_points:.3f} |"
        )
    lines.extend(["", "## Volume Deltas", "", "| state | mean abs mGy | p95 abs mGy | max abs mGy | correlation |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in result.volume_comparisons:
        lines.append(
            f"| `{row.state_pair}` | {1000.0 * row.mean_abs_diff_gy:.4f} | {1000.0 * row.p95_abs_diff_gy:.4f} | "
            f"{1000.0 * row.max_abs_diff_gy:.3f} | {row.correlation:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The scalar model applies one global vascular proximity response, while the spatial model localizes perturbations by ranked vessel edges and edge-specific waveforms.",
            "- This comparison quantifies the model-change effect, not a patient-specific clinical dose difference.",
            "- The largest changes are expected near highly ranked vascular edges and around the synthetic vertebral target where vessel influence overlaps the high-dose region.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def compare_scalar_vs_spatial_rt_flow_dose(
    scalar_rt_planning_spec_path: str | Path,
    spatial_rt_flow_dose_spec_path: str | Path,
    output_dir: str | Path = "outputs/experiments/scalar_vs_spatial_rt_flow",
    case_id: str = "ct_org_case0_imagetbad_case125",
    report_path: str | Path | None = "outputs/reports/scalar_vs_spatial_rt_flow_dose_stage001.md",
) -> RTFlowModelComparisonResult:
    scalar_spec_path = Path(scalar_rt_planning_spec_path)
    spatial_spec_path = Path(spatial_rt_flow_dose_spec_path)
    scalar_spec = _load_yaml(scalar_spec_path)
    spatial_spec = _load_yaml(spatial_spec_path)
    scalar_outputs = scalar_spec.get("outputs", {})
    spatial_outputs = spatial_spec.get("outputs", {})
    scalar_dose_model = scalar_spec.get("dose_model", {})
    if not isinstance(scalar_outputs, dict) or not isinstance(spatial_outputs, dict):
        raise ValueError("Both specs must include outputs")
    if not isinstance(scalar_dose_model, dict):
        scalar_dose_model = {}
    prescription = _as_float(scalar_dose_model.get("prescription_dose_gy"), 20.0)
    scalar_metric_rows = _read_csv_rows(scalar_outputs.get("dose_metrics_csv"), scalar_spec_path)
    spatial_metric_rows = _read_csv_rows(spatial_outputs.get("dose_metrics_csv"), spatial_spec_path)
    metric_comparisons = _build_metric_comparisons(scalar_metric_rows, spatial_metric_rows)

    volume_comparisons: list[RTFlowVolumeComparison] = []
    for state_pair, (scalar_state, spatial_state, scalar_raw, spatial_raw) in _dose_paths(scalar_spec, spatial_spec).items():
        scalar_path = _resolve_path(scalar_raw, scalar_spec_path)
        spatial_path = _resolve_path(spatial_raw, spatial_spec_path)
        if scalar_path is None or spatial_path is None or not scalar_path.exists() or not spatial_path.exists():
            continue
        volume_comparisons.append(
            _compare_volume_pair(
                state_pair=state_pair,
                scalar_state=scalar_state,
                spatial_state=spatial_state,
                scalar_path=scalar_path,
                spatial_path=spatial_path,
                prescription_dose_gy=prescription,
            )
        )

    output = Path(output_dir)
    metric_csv = output / f"{case_id}_scalar_vs_spatial_rt_flow_metric_comparison_v001.csv"
    volume_csv = output / f"{case_id}_scalar_vs_spatial_rt_flow_volume_comparison_v001.csv"
    preview_png = output / f"{case_id}_scalar_vs_spatial_rt_flow_comparison_preview_v001.png"
    spec_yaml = output / f"{case_id}_scalar_vs_spatial_rt_flow_comparison_spec_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_scalar_vs_spatial_rt_flow_comparison_report_v001.md"
    notes = (
        "scalar_model_uses_global_vascular_proximity_surrogate",
        "spatial_model_uses_ranked_edge_influence_fields_and_edge_waveforms",
        "comparison_is_engineering_model_delta_not_clinical_dose_validation",
    )
    result = RTFlowModelComparisonResult(
        case_id=case_id,
        output_dir=str(output),
        scalar_planning_spec_path=str(scalar_spec_path),
        spatial_dose_spec_path=str(spatial_spec_path),
        metric_comparison_csv_path=str(metric_csv),
        volume_comparison_csv_path=str(volume_csv),
        preview_png_path=str(preview_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        prescription_dose_gy=prescription,
        metric_comparisons=metric_comparisons,
        volume_comparisons=tuple(volume_comparisons),
        notes=notes,
    )
    _write_metric_csv(metric_csv, result.metric_comparisons)
    _write_volume_csv(volume_csv, result.volume_comparisons)
    _write_preview(preview_png, scalar_spec_path, spatial_spec_path, scalar_spec, spatial_spec, result.metric_comparisons)
    _write_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_rt_flow_model_comparison_result(result: RTFlowModelComparisonResult) -> str:
    lines = [
        "Scalar vs spatial RT-flow comparison created",
        f"Case ID: {result.case_id}",
        f"Metric comparisons: {len(result.metric_comparisons)}",
        f"Volume comparisons: {len(result.volume_comparisons)}",
        f"Max absolute voxel delta: {max((row.max_abs_diff_gy for row in result.volume_comparisons), default=0.0) * 1000.0:.3f} mGy",
        f"Metric CSV: {result.metric_comparison_csv_path}",
        f"Volume CSV: {result.volume_comparison_csv_path}",
        f"Preview PNG: {result.preview_png_path}",
        f"Spec YAML: {result.spec_yaml_path}",
        f"Report: {result.report_path}",
    ]
    return "\n".join(lines)
