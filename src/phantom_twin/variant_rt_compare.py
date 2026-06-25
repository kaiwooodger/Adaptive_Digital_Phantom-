from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import yaml


@dataclass(frozen=True)
class DoseMetricComparison:
    mask_id: str
    label: str
    role: str
    state: str
    baseline_volume_cm3: float
    variant_volume_cm3: float
    delta_volume_cm3: float
    delta_volume_percent: float
    baseline_mean_dose_gy: float
    variant_mean_dose_gy: float
    delta_mean_dose_gy: float
    baseline_d95_gy: float
    variant_d95_gy: float
    delta_d95_gy: float
    baseline_v95_percent: float
    variant_v95_percent: float
    delta_v95_percentage_points: float


@dataclass(frozen=True)
class PulsatileDeltaComparison:
    mask_id: str
    label: str
    role: str
    comparison_state: str
    baseline_delta_mean_gy: float
    variant_delta_mean_gy: float
    delta_of_delta_mean_gy: float
    baseline_delta_v95_percentage_points: float
    variant_delta_v95_percentage_points: float
    delta_of_delta_v95_percentage_points: float


@dataclass(frozen=True)
class GammaComparison:
    state: str
    baseline_pass_rate_percent: float
    variant_pass_rate_percent: float
    delta_pass_rate_percentage_points: float
    baseline_p95_gamma: float
    variant_p95_gamma: float
    delta_p95_gamma: float
    baseline_max_gamma: float
    variant_max_gamma: float
    delta_max_gamma: float
    baseline_sampled_points: int
    variant_sampled_points: int


@dataclass(frozen=True)
class VariantRTComparisonResult:
    case_id: str
    variant_id: str
    output_dir: str
    dose_metric_comparison_csv_path: str
    pulsatile_comparison_csv_path: str
    gamma_comparison_csv_path: str | None
    spec_yaml_path: str
    preview_png_path: str
    report_path: str
    baseline_rt_planning_spec_path: str
    variant_rt_planning_spec_path: str
    baseline_gamma_spec_path: str | None
    variant_gamma_spec_path: str | None
    dose_metric_comparisons: tuple[DoseMetricComparison, ...]
    pulsatile_comparisons: tuple[PulsatileDeltaComparison, ...]
    gamma_comparisons: tuple[GammaComparison, ...]
    summary: dict[str, Any]
    notes: tuple[str, ...]


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


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(raw_path: str | Path | None, reference_yaml_path: Path | None = None) -> Path | None:
    if raw_path is None or str(raw_path) == "":
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    if reference_yaml_path is not None:
        return reference_yaml_path.parent / path
    return cwd_candidate


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


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def _safe_percent_delta(value: float, baseline: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0 if abs(value) < 1e-9 else 100.0
    return (value - baseline) / baseline * 100.0


def _rows_by_key(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row.get(key, "") for key in keys): row for row in rows}


def _planning_outputs(spec_path: Path) -> dict[str, Any]:
    spec = _load_yaml(spec_path)
    outputs = spec.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError(f"RT planning spec has malformed outputs: {spec_path}")
    return outputs


def _gamma_summary_rows(gamma_spec_path: str | Path | None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if gamma_spec_path is None:
        return [], {}
    path = Path(gamma_spec_path)
    spec = _load_yaml(path)
    outputs = spec.get("outputs", {})
    if not isinstance(outputs, dict):
        return [], spec
    return _read_csv_rows(outputs.get("summary_csv"), path), spec


def _build_dose_metric_comparisons(
    baseline_rows: list[dict[str, str]],
    variant_rows: list[dict[str, str]],
) -> tuple[DoseMetricComparison, ...]:
    baseline_by_key = _rows_by_key(baseline_rows, ("mask_id", "state"))
    variant_by_key = _rows_by_key(variant_rows, ("mask_id", "state"))
    comparisons: list[DoseMetricComparison] = []
    for key in sorted(set(baseline_by_key) & set(variant_by_key), key=lambda item: (item[1], item[0])):
        baseline = baseline_by_key[key]
        variant = variant_by_key[key]
        baseline_volume = _as_float(baseline.get("volume_cm3"))
        variant_volume = _as_float(variant.get("volume_cm3"))
        baseline_mean = _as_float(baseline.get("mean_dose_gy"))
        variant_mean = _as_float(variant.get("mean_dose_gy"))
        baseline_d95 = _as_float(baseline.get("d95_gy"))
        variant_d95 = _as_float(variant.get("d95_gy"))
        baseline_v95 = _as_float(baseline.get("v95_percent"))
        variant_v95 = _as_float(variant.get("v95_percent"))
        comparisons.append(
            DoseMetricComparison(
                mask_id=key[0],
                label=str(variant.get("label") or baseline.get("label") or key[0]),
                role=str(variant.get("role") or baseline.get("role") or ""),
                state=key[1],
                baseline_volume_cm3=baseline_volume,
                variant_volume_cm3=variant_volume,
                delta_volume_cm3=variant_volume - baseline_volume,
                delta_volume_percent=_safe_percent_delta(variant_volume, baseline_volume),
                baseline_mean_dose_gy=baseline_mean,
                variant_mean_dose_gy=variant_mean,
                delta_mean_dose_gy=variant_mean - baseline_mean,
                baseline_d95_gy=baseline_d95,
                variant_d95_gy=variant_d95,
                delta_d95_gy=variant_d95 - baseline_d95,
                baseline_v95_percent=baseline_v95,
                variant_v95_percent=variant_v95,
                delta_v95_percentage_points=variant_v95 - baseline_v95,
            )
        )
    return tuple(comparisons)


def _build_pulsatile_comparisons(
    baseline_rows: list[dict[str, str]],
    variant_rows: list[dict[str, str]],
) -> tuple[PulsatileDeltaComparison, ...]:
    baseline_by_key = _rows_by_key(baseline_rows, ("mask_id", "comparison_state"))
    variant_by_key = _rows_by_key(variant_rows, ("mask_id", "comparison_state"))
    comparisons: list[PulsatileDeltaComparison] = []
    for key in sorted(set(baseline_by_key) & set(variant_by_key), key=lambda item: (item[1], item[0])):
        baseline = baseline_by_key[key]
        variant = variant_by_key[key]
        baseline_delta_mean = _as_float(baseline.get("delta_mean_gy"))
        variant_delta_mean = _as_float(variant.get("delta_mean_gy"))
        baseline_delta_v95 = _as_float(baseline.get("delta_v95_percentage_points"))
        variant_delta_v95 = _as_float(variant.get("delta_v95_percentage_points"))
        comparisons.append(
            PulsatileDeltaComparison(
                mask_id=key[0],
                label=str(variant.get("label") or baseline.get("label") or key[0]),
                role=str(variant.get("role") or baseline.get("role") or ""),
                comparison_state=key[1],
                baseline_delta_mean_gy=baseline_delta_mean,
                variant_delta_mean_gy=variant_delta_mean,
                delta_of_delta_mean_gy=variant_delta_mean - baseline_delta_mean,
                baseline_delta_v95_percentage_points=baseline_delta_v95,
                variant_delta_v95_percentage_points=variant_delta_v95,
                delta_of_delta_v95_percentage_points=variant_delta_v95 - baseline_delta_v95,
            )
        )
    return tuple(comparisons)


def _build_gamma_comparisons(
    baseline_rows: list[dict[str, str]],
    variant_rows: list[dict[str, str]],
) -> tuple[GammaComparison, ...]:
    baseline_by_key = _rows_by_key(baseline_rows, ("state",))
    variant_by_key = _rows_by_key(variant_rows, ("state",))
    comparisons: list[GammaComparison] = []
    for key in sorted(set(baseline_by_key) & set(variant_by_key)):
        baseline = baseline_by_key[key]
        variant = variant_by_key[key]
        baseline_pass = _as_float(baseline.get("pass_rate_percent"))
        variant_pass = _as_float(variant.get("pass_rate_percent"))
        baseline_p95 = _as_float(baseline.get("p95_gamma"))
        variant_p95 = _as_float(variant.get("p95_gamma"))
        baseline_max = _as_float(baseline.get("max_gamma_value"))
        variant_max = _as_float(variant.get("max_gamma_value"))
        comparisons.append(
            GammaComparison(
                state=key[0],
                baseline_pass_rate_percent=baseline_pass,
                variant_pass_rate_percent=variant_pass,
                delta_pass_rate_percentage_points=variant_pass - baseline_pass,
                baseline_p95_gamma=baseline_p95,
                variant_p95_gamma=variant_p95,
                delta_p95_gamma=variant_p95 - baseline_p95,
                baseline_max_gamma=baseline_max,
                variant_max_gamma=variant_max,
                delta_max_gamma=variant_max - baseline_max,
                baseline_sampled_points=_as_int(baseline.get("finite_gamma_points")),
                variant_sampled_points=_as_int(variant.get("finite_gamma_points")),
            )
        )
    return tuple(comparisons)


def _find_dose(
    comparisons: tuple[DoseMetricComparison, ...],
    mask_id: str,
    state: str = "static",
) -> DoseMetricComparison | None:
    return next((item for item in comparisons if item.mask_id == mask_id and item.state == state), None)


def _summary(
    dose_comparisons: tuple[DoseMetricComparison, ...],
    pulsatile_comparisons: tuple[PulsatileDeltaComparison, ...],
    gamma_comparisons: tuple[GammaComparison, ...],
) -> dict[str, Any]:
    static_rows = [item for item in dose_comparisons if item.state == "static"]
    ptv = _find_dose(dose_comparisons, "target_ptv_synthetic_vertebral")
    vascular = _find_dose(dose_comparisons, "vascular_fluid")
    max_static_mean = max(static_rows, key=lambda item: abs(item.delta_mean_dose_gy), default=None)
    max_static_volume = max(static_rows, key=lambda item: abs(item.delta_volume_percent), default=None)
    max_pulsatile_mean = max(pulsatile_comparisons, key=lambda item: abs(item.delta_of_delta_mean_gy), default=None)
    max_pulsatile_v95 = max(
        pulsatile_comparisons,
        key=lambda item: abs(item.delta_of_delta_v95_percentage_points),
        default=None,
    )
    return {
        "matched_dose_metric_rows": len(dose_comparisons),
        "matched_pulsatile_comparison_rows": len(pulsatile_comparisons),
        "matched_gamma_states": len(gamma_comparisons),
        "ptv_static_mean_delta_gy": None if ptv is None else ptv.delta_mean_dose_gy,
        "ptv_static_d95_delta_gy": None if ptv is None else ptv.delta_d95_gy,
        "ptv_static_v95_delta_percentage_points": None if ptv is None else ptv.delta_v95_percentage_points,
        "ptv_volume_delta_percent": None if ptv is None else ptv.delta_volume_percent,
        "vascular_fluid_static_mean_delta_gy": None if vascular is None else vascular.delta_mean_dose_gy,
        "vascular_fluid_volume_delta_percent": None if vascular is None else vascular.delta_volume_percent,
        "max_abs_static_mean_delta_mask": None if max_static_mean is None else max_static_mean.mask_id,
        "max_abs_static_mean_delta_gy": None if max_static_mean is None else max_static_mean.delta_mean_dose_gy,
        "max_abs_static_volume_delta_mask": None if max_static_volume is None else max_static_volume.mask_id,
        "max_abs_static_volume_delta_percent": None if max_static_volume is None else max_static_volume.delta_volume_percent,
        "max_delta_of_pulsatile_mean_mask": None if max_pulsatile_mean is None else max_pulsatile_mean.mask_id,
        "max_delta_of_pulsatile_mean_state": None if max_pulsatile_mean is None else max_pulsatile_mean.comparison_state,
        "max_delta_of_pulsatile_mean_gy": None if max_pulsatile_mean is None else max_pulsatile_mean.delta_of_delta_mean_gy,
        "max_delta_of_pulsatile_v95_mask": None if max_pulsatile_v95 is None else max_pulsatile_v95.mask_id,
        "max_delta_of_pulsatile_v95_state": None if max_pulsatile_v95 is None else max_pulsatile_v95.comparison_state,
        "max_delta_of_pulsatile_v95_percentage_points": None
        if max_pulsatile_v95 is None
        else max_pulsatile_v95.delta_of_delta_v95_percentage_points,
        "minimum_variant_gamma_pass_rate_percent": min(
            (item.variant_pass_rate_percent for item in gamma_comparisons),
            default=None,
        ),
        "maximum_variant_gamma_p95": max((item.variant_p95_gamma for item in gamma_comparisons), default=None),
        "maximum_variant_gamma_value": max((item.variant_max_gamma for item in gamma_comparisons), default=None),
    }


def _write_dose_csv(path: Path, rows: tuple[DoseMetricComparison, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mask_id",
        "label",
        "role",
        "state",
        "baseline_volume_cm3",
        "variant_volume_cm3",
        "delta_volume_cm3",
        "delta_volume_percent",
        "baseline_mean_dose_gy",
        "variant_mean_dose_gy",
        "delta_mean_dose_gy",
        "baseline_d95_gy",
        "variant_d95_gy",
        "delta_d95_gy",
        "baseline_v95_percent",
        "variant_v95_percent",
        "delta_v95_percentage_points",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def _write_pulsatile_csv(path: Path, rows: tuple[PulsatileDeltaComparison, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mask_id",
        "label",
        "role",
        "comparison_state",
        "baseline_delta_mean_gy",
        "variant_delta_mean_gy",
        "delta_of_delta_mean_gy",
        "baseline_delta_v95_percentage_points",
        "variant_delta_v95_percentage_points",
        "delta_of_delta_v95_percentage_points",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def _write_gamma_csv(path: Path, rows: tuple[GammaComparison, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "state",
        "baseline_pass_rate_percent",
        "variant_pass_rate_percent",
        "delta_pass_rate_percentage_points",
        "baseline_p95_gamma",
        "variant_p95_gamma",
        "delta_p95_gamma",
        "baseline_max_gamma",
        "variant_max_gamma",
        "delta_max_gamma",
        "baseline_sampled_points",
        "variant_sampled_points",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def _write_preview(path: Path, result: VariantRTComparisonResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    selected_static = [
        row
        for row in result.dose_metric_comparisons
        if row.state == "static" and row.mask_id in SELECTED_MASKS
    ]
    selected_static = sorted(selected_static, key=lambda row: SELECTED_MASKS.index(row.mask_id))
    labels = [row.label.replace("Synthetic vertebral ", "") for row in selected_static]
    mean_deltas = [row.delta_mean_dose_gy for row in selected_static]
    volume_deltas = [row.delta_volume_percent for row in selected_static]
    gamma_states = [row.state.replace("pulsatile_", "") for row in result.gamma_comparisons]
    gamma_p95 = [row.variant_p95_gamma for row in result.gamma_comparisons]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    colors = ["#b91c1c" if value < 0 else "#1d4ed8" for value in mean_deltas]
    axes[0].barh(labels, mean_deltas, color=colors)
    axes[0].axvline(0.0, color="#111827", linewidth=1.0)
    axes[0].set_title("Static Mean Dose Delta")
    axes[0].set_xlabel("Variant - baseline Gy")

    colors = ["#b91c1c" if value < 0 else "#047857" for value in volume_deltas]
    axes[1].barh(labels, volume_deltas, color=colors)
    axes[1].axvline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_title("Anatomy Volume Delta")
    axes[1].set_xlabel("Variant - baseline %")

    if gamma_states:
        axes[2].bar(gamma_states, gamma_p95, color="#7c2d12")
        axes[2].set_ylim(0.0, max(0.01, max(gamma_p95) * 1.25))
    axes[2].set_title("Variant Gamma p95")
    axes[2].set_ylabel("gamma")
    axes[2].tick_params(axis="x", rotation=25)

    fig.suptitle(f"RT Impact Comparison: {result.variant_id}", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _format_value(value: Any, precision: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}{suffix}"


def _format_report(result: VariantRTComparisonResult) -> str:
    selected = [
        row
        for row in result.dose_metric_comparisons
        if row.state == "static" and row.mask_id in SELECTED_MASKS
    ]
    selected = sorted(selected, key=lambda row: SELECTED_MASKS.index(row.mask_id))
    gamma = result.gamma_comparisons
    lines = [
        "# Variant RT Impact Comparison Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        f"Variant ID: `{result.variant_id}`",
        "",
        "## Summary",
        "",
        f"- Matched dose metric rows: {result.summary['matched_dose_metric_rows']}",
        f"- Matched pulsatile comparison rows: {result.summary['matched_pulsatile_comparison_rows']}",
        f"- PTV static mean dose delta: {_format_value(result.summary['ptv_static_mean_delta_gy'], 4)} Gy",
        f"- PTV static D95 delta: {_format_value(result.summary['ptv_static_d95_delta_gy'], 4)} Gy",
        f"- PTV static V95 delta: {_format_value(result.summary['ptv_static_v95_delta_percentage_points'], 3)} percentage points",
        f"- Vascular-fluid static mean dose delta: {_format_value(result.summary['vascular_fluid_static_mean_delta_gy'], 4)} Gy",
        f"- Variant gamma min pass rate: {_format_value(result.summary['minimum_variant_gamma_pass_rate_percent'], 3)}%",
        "",
        "## Static Dose And Anatomy Deltas",
        "",
        "| mask | variant volume delta % | static mean dose delta Gy | static D95 delta Gy | static V95 delta pp |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| {row.label} | {row.delta_volume_percent:+.2f} | {row.delta_mean_dose_gy:+.4f} | "
            f"{row.delta_d95_gy:+.4f} | {row.delta_v95_percentage_points:+.3f} |"
        )
    lines.extend(["", "## Pulsatile Delta Change", ""])
    lines.append(
        "| mask | state | baseline delta mean Gy | variant delta mean Gy | delta-of-delta Gy | delta-of-delta V95 pp |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    focused_pulsatile = [
        row
        for row in result.pulsatile_comparisons
        if row.mask_id in ("target_ptv_synthetic_vertebral", "vascular_fluid", "vessel_wall", "oar_bone")
    ]
    for row in focused_pulsatile:
        lines.append(
            f"| {row.label} | {row.comparison_state} | {row.baseline_delta_mean_gy:+.5f} | "
            f"{row.variant_delta_mean_gy:+.5f} | {row.delta_of_delta_mean_gy:+.5f} | "
            f"{row.delta_of_delta_v95_percentage_points:+.3f} |"
        )
    lines.extend(["", "## Gamma QA Comparison", ""])
    if gamma:
        lines.append("| state | baseline pass % | variant pass % | variant p95 gamma | variant max gamma | sampled points |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in gamma:
            lines.append(
                f"| {row.state} | {row.baseline_pass_rate_percent:.3f} | {row.variant_pass_rate_percent:.3f} | "
                f"{row.variant_p95_gamma:.4f} | {row.variant_max_gamma:.4f} | {row.variant_sampled_points} |"
            )
    else:
        lines.append("- Gamma specs were not supplied, so only dose-metric comparisons were generated.")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- Dose metric comparison CSV: `{Path(result.dose_metric_comparison_csv_path).name}`")
    lines.append(f"- Pulsatile comparison CSV: `{Path(result.pulsatile_comparison_csv_path).name}`")
    if result.gamma_comparison_csv_path:
        lines.append(f"- Gamma comparison CSV: `{Path(result.gamma_comparison_csv_path).name}`")
    lines.append(f"- Spec YAML: `{Path(result.spec_yaml_path).name}`")
    lines.append(f"- Preview PNG: `{Path(result.preview_png_path).name}`")
    lines.extend(["", "## Notes"])
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _write_spec(path: Path, result: VariantRTComparisonResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "variant_rt_impact_comparison",
        "variant_id": result.variant_id,
        "inputs": {
            "baseline_rt_planning_spec": result.baseline_rt_planning_spec_path,
            "variant_rt_planning_spec": result.variant_rt_planning_spec_path,
            "baseline_gamma_spec": result.baseline_gamma_spec_path,
            "variant_gamma_spec": result.variant_gamma_spec_path,
        },
        "outputs": {
            "dose_metric_comparison_csv": result.dose_metric_comparison_csv_path,
            "pulsatile_comparison_csv": result.pulsatile_comparison_csv_path,
            "gamma_comparison_csv": result.gamma_comparison_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "summary": result.summary,
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def compare_variant_rt_impact(
    baseline_rt_planning_spec_path: str | Path,
    variant_rt_planning_spec_path: str | Path,
    baseline_gamma_spec_path: str | Path | None = None,
    variant_gamma_spec_path: str | Path | None = None,
    output_dir: str | Path = "outputs/experiments/variant_rt_comparison",
    case_id: str | None = None,
    variant_id: str = "mode01_neg",
    report_path: str | Path | None = "outputs/reports/variant_rt_impact_comparison_stage001.md",
) -> VariantRTComparisonResult:
    baseline_spec_path = Path(baseline_rt_planning_spec_path)
    variant_spec_path = Path(variant_rt_planning_spec_path)
    baseline_outputs = _planning_outputs(baseline_spec_path)
    variant_outputs = _planning_outputs(variant_spec_path)

    baseline_metrics = _read_csv_rows(baseline_outputs.get("dose_metrics_csv"), baseline_spec_path)
    variant_metrics = _read_csv_rows(variant_outputs.get("dose_metrics_csv"), variant_spec_path)
    baseline_pulsatile = _read_csv_rows(baseline_outputs.get("dose_comparison_csv"), baseline_spec_path)
    variant_pulsatile = _read_csv_rows(variant_outputs.get("dose_comparison_csv"), variant_spec_path)
    baseline_gamma, baseline_gamma_spec = _gamma_summary_rows(baseline_gamma_spec_path)
    variant_gamma, variant_gamma_spec = _gamma_summary_rows(variant_gamma_spec_path)

    dose_comparisons = _build_dose_metric_comparisons(baseline_metrics, variant_metrics)
    pulsatile_comparisons = _build_pulsatile_comparisons(baseline_pulsatile, variant_pulsatile)
    gamma_comparisons = _build_gamma_comparisons(baseline_gamma, variant_gamma)
    summary = _summary(dose_comparisons, pulsatile_comparisons, gamma_comparisons)

    resolved_case_id = case_id or f"{variant_id}_rt_impact_comparison"
    output = Path(output_dir)
    dose_csv = output / f"{resolved_case_id}_dose_metric_comparison_v001.csv"
    pulsatile_csv = output / f"{resolved_case_id}_pulsatile_delta_comparison_v001.csv"
    gamma_csv = output / f"{resolved_case_id}_gamma_comparison_v001.csv" if gamma_comparisons else None
    spec_yaml = output / f"{resolved_case_id}_variant_rt_impact_comparison_spec_v001.yaml"
    preview_png = output / f"{resolved_case_id}_variant_rt_impact_comparison_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{resolved_case_id}_variant_rt_impact_comparison_report_v001.md"

    notes = [
        "compares_baseline_static_and_pulsatile_surrogate_dose_metrics_against_variant_specific_rerun",
        "clinical_interpretation_requires_real_tps_or_monte_carlo_dose_calculation",
    ]
    if baseline_gamma_spec and variant_gamma_spec:
        baseline_subset = baseline_gamma_spec.get("gamma_settings", {}).get("random_subset")
        variant_subset = variant_gamma_spec.get("gamma_settings", {}).get("random_subset")
        if baseline_subset != variant_subset:
            notes.append("baseline_and_variant_gamma_random_subset_sizes_differ")
    if not gamma_comparisons:
        notes.append("gamma_comparison_not_generated_because_one_or_both_gamma_specs_were_missing")

    result = VariantRTComparisonResult(
        case_id=resolved_case_id,
        variant_id=variant_id,
        output_dir=str(output),
        dose_metric_comparison_csv_path=str(dose_csv),
        pulsatile_comparison_csv_path=str(pulsatile_csv),
        gamma_comparison_csv_path=None if gamma_csv is None else str(gamma_csv),
        spec_yaml_path=str(spec_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        baseline_rt_planning_spec_path=str(baseline_rt_planning_spec_path),
        variant_rt_planning_spec_path=str(variant_rt_planning_spec_path),
        baseline_gamma_spec_path=None if baseline_gamma_spec_path is None else str(baseline_gamma_spec_path),
        variant_gamma_spec_path=None if variant_gamma_spec_path is None else str(variant_gamma_spec_path),
        dose_metric_comparisons=dose_comparisons,
        pulsatile_comparisons=pulsatile_comparisons,
        gamma_comparisons=gamma_comparisons,
        summary=summary,
        notes=tuple(notes),
    )

    _write_dose_csv(dose_csv, dose_comparisons)
    _write_pulsatile_csv(pulsatile_csv, pulsatile_comparisons)
    if gamma_csv is not None:
        _write_gamma_csv(gamma_csv, gamma_comparisons)
    _write_preview(preview_png, result)
    _write_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_variant_rt_comparison_result(result: VariantRTComparisonResult) -> str:
    lines = [
        "# Variant RT Impact Comparison",
        "",
        f"Case ID: `{result.case_id}`",
        f"Variant ID: `{result.variant_id}`",
        f"Matched dose metric rows: {result.summary['matched_dose_metric_rows']}",
        f"Matched gamma states: {result.summary['matched_gamma_states']}",
        f"PTV static mean delta: {_format_value(result.summary['ptv_static_mean_delta_gy'], 4)} Gy",
        f"Variant gamma min pass rate: {_format_value(result.summary['minimum_variant_gamma_pass_rate_percent'], 3)}%",
        "",
        "## Outputs",
        "",
        f"- Dose comparison CSV: `{result.dose_metric_comparison_csv_path}`",
        f"- Pulsatile comparison CSV: `{result.pulsatile_comparison_csv_path}`",
    ]
    if result.gamma_comparison_csv_path:
        lines.append(f"- Gamma comparison CSV: `{result.gamma_comparison_csv_path}`")
    lines.extend(
        [
            f"- Preview PNG: `{result.preview_png_path}`",
            f"- Spec YAML: `{result.spec_yaml_path}`",
            f"- Report: `{result.report_path}`",
        ]
    )
    return "\n".join(lines)
