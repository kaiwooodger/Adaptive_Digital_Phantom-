from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import yaml


PROFILE_FIELD_ORDER = [
    "profile_id",
    "case_id",
    "target_bmi",
    "target_waist_cm",
    "target_height_cm",
    "morph_mode",
    "xy_padding_voxels",
    "achieved_waist_cm",
    "waist_error_cm",
    "body_volume_l",
    "body_volume_change_percent",
    "morph_vascular_components",
    "mean_node_displacement_mm",
    "max_node_displacement_mm",
    "arterial_venous_overlap_before_cleanup",
    "arterial_venous_overlap_after_cleanup",
    "connected_components",
    "arterial_components",
    "venous_components",
    "boundary_count",
    "mapped_boundary_count",
    "aortic_flow_mean_ml_s",
    "aortic_flow_min_ml_s",
    "aortic_flow_max_ml_s",
    "aortic_pressure_mean_mmhg",
    "max_outlet_split_range_pp",
    "max_mass_balance_residual_ml_s",
    "selected_spatial_edges",
    "max_peak_delta_mgy",
    "max_trough_delta_mgy",
    "ptv_static_mean_gy",
    "ptv_peak_mean_gy",
    "ptv_peak_v95_percent",
    "gamma_min_pass_rate_percent",
    "gamma_max_p95",
    "anatomy_status",
    "vascular_status",
    "flow_status",
    "rt_status",
    "overall_status",
    "morph_preview_png",
    "vascular_preview_png",
    "flow_preview_png",
    "spatial_dose_preview_png",
    "gamma_preview_png",
    "profile_root",
    "source_metrics_csv",
]


@dataclass(frozen=True)
class PredictionDelta:
    profile_id: str
    expected_body_volume_l: float
    actual_body_volume_l: float
    body_volume_error_l: float
    expected_ptv_peak_v95_percent: float
    actual_ptv_peak_v95_percent: float
    ptv_peak_v95_error_percent: float
    expected_gamma_min_pass_rate_percent: float
    actual_gamma_min_pass_rate_percent: float
    gamma_min_pass_error_percent: float
    target_waist_cm: float
    achieved_waist_cm: float
    achieved_waist_error_cm: float


@dataclass(frozen=True)
class ProfileEnvelopeResult:
    envelope_id: str
    output_dir: str
    metrics_csv_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    source_metrics_csv_paths: tuple[str, ...]
    prescription_yaml_path: str | None
    profile_count: int
    pass_count: int
    warn_count: int
    fail_count: int
    waist_range_cm: tuple[float, float]
    bmi_range: tuple[float, float]
    high_bmi_profile_count: int
    rows: tuple[dict[str, str], ...]
    prediction_delta: PredictionDelta | None
    notes: tuple[str, ...]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _range(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (float(min(values)), float(max(values)))


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    with resolved.open(newline="") as csvfile:
        rows = [dict(row) for row in csv.DictReader(csvfile)]
    for row in rows:
        row["source_metrics_csv"] = str(path)
    return rows


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text())
    return data if isinstance(data, dict) else {}


def _merge_rows(metrics_csv_paths: tuple[str | Path, ...]) -> tuple[dict[str, str], ...]:
    if not metrics_csv_paths:
        raise ValueError("At least one metrics CSV path is required")
    by_profile: dict[str, dict[str, str]] = {}
    for path in metrics_csv_paths:
        for row in _load_csv_rows(path):
            profile_id = str(row.get("profile_id", "")).strip()
            if not profile_id:
                continue
            by_profile[profile_id] = row
    return tuple(
        sorted(
            by_profile.values(),
            key=lambda row: (
                _as_float(row.get("target_waist_cm")),
                _as_float(row.get("target_bmi")),
                str(row.get("profile_id", "")),
            ),
        )
    )


def _fieldnames(rows: tuple[dict[str, str], ...]) -> list[str]:
    fields = list(PROFILE_FIELD_ORDER)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _write_metrics_csv(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    fields = _fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _prediction_delta(prescription_yaml_path: str | Path | None, rows: tuple[dict[str, str], ...]) -> PredictionDelta | None:
    prescription = _load_yaml(prescription_yaml_path)
    if not prescription:
        return None
    profile_id = str(prescription.get("profile_id", ""))
    target = prescription.get("target", {}) if isinstance(prescription.get("target", {}), dict) else {}
    expected = prescription.get("interpolated_expectations", {})
    expected = expected if isinstance(expected, dict) else {}
    actual = next((row for row in rows if str(row.get("profile_id", "")) == profile_id), None)
    if actual is None:
        return None
    expected_body = _as_float(expected.get("body_volume_l"))
    actual_body = _as_float(actual.get("body_volume_l"))
    expected_ptv = _as_float(expected.get("ptv_peak_v95_percent"))
    actual_ptv = _as_float(actual.get("ptv_peak_v95_percent"))
    expected_gamma = _as_float(expected.get("gamma_min_pass_rate_percent"))
    actual_gamma = _as_float(actual.get("gamma_min_pass_rate_percent"))
    target_waist = _as_float(target.get("waist_cm"), _as_float(actual.get("target_waist_cm")))
    achieved_waist = _as_float(actual.get("achieved_waist_cm"))
    return PredictionDelta(
        profile_id=profile_id,
        expected_body_volume_l=expected_body,
        actual_body_volume_l=actual_body,
        body_volume_error_l=actual_body - expected_body,
        expected_ptv_peak_v95_percent=expected_ptv,
        actual_ptv_peak_v95_percent=actual_ptv,
        ptv_peak_v95_error_percent=actual_ptv - expected_ptv,
        expected_gamma_min_pass_rate_percent=expected_gamma,
        actual_gamma_min_pass_rate_percent=actual_gamma,
        gamma_min_pass_error_percent=actual_gamma - expected_gamma,
        target_waist_cm=target_waist,
        achieved_waist_cm=achieved_waist,
        achieved_waist_error_cm=achieved_waist - target_waist,
    )


def _status_counts(rows: tuple[dict[str, str], ...]) -> tuple[int, int, int]:
    pass_count = sum(1 for row in rows if str(row.get("overall_status", "")).lower() == "pass")
    warn_count = sum(1 for row in rows if str(row.get("overall_status", "")).lower() == "warn")
    fail_count = sum(1 for row in rows if str(row.get("overall_status", "")).lower() == "fail")
    return pass_count, warn_count, fail_count


def _write_preview(path: Path, result: ProfileEnvelopeResult) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Profile envelope preview generation requires matplotlib.") from exc

    rows = list(result.rows)
    waist = [_as_float(row.get("target_waist_cm")) for row in rows]
    body = [_as_float(row.get("body_volume_l")) for row in rows]
    ptv = [_as_float(row.get("ptv_peak_v95_percent")) for row in rows]
    gamma = [_as_float(row.get("gamma_min_pass_rate_percent")) for row in rows]
    colors = ["#e76f51" if str(row.get("morph_mode")) == "high-bmi" else "#2a9d8f" for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), dpi=170)
    fig.suptitle(f"Consolidated Profile Operating Envelope\n{result.envelope_id}", fontsize=15, fontweight="bold")

    axes[0].scatter(waist, body, s=90, color=colors, edgecolor="#1f2933")
    axes[0].plot(waist, body, color="#264653", linewidth=1.3, alpha=0.7)
    axes[0].set_xlabel("target waist (cm)")
    axes[0].set_ylabel("body volume (L)")
    axes[0].set_title("Anatomy envelope")
    axes[0].grid(True, color="#d8dee9", linewidth=0.7)

    axes[1].plot(waist, ptv, marker="o", color="#457b9d", linewidth=2.0, label="PTV peak V95")
    axes[1].plot(waist, gamma, marker="s", color="#2a9d8f", linewidth=2.0, label="gamma min pass")
    axes[1].set_xlabel("target waist (cm)")
    axes[1].set_ylabel("percent")
    axes[1].set_ylim(75, 103)
    axes[1].set_title("RT-flow QA envelope")
    axes[1].grid(True, color="#d8dee9", linewidth=0.7)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].axis("off")
    summary = [
        f"Profiles: {result.profile_count}",
        f"Pass / warn / fail: {result.pass_count} / {result.warn_count} / {result.fail_count}",
        f"Waist range: {result.waist_range_cm[0]:.1f}-{result.waist_range_cm[1]:.1f} cm",
        f"BMI range: {result.bmi_range[0]:.1f}-{result.bmi_range[1]:.1f}",
        f"High-BMI profiles: {result.high_bmi_profile_count}",
    ]
    if result.prediction_delta is not None:
        delta = result.prediction_delta
        summary.extend(
            [
                "",
                f"Prescription validation: {delta.profile_id}",
                f"Body error: {delta.body_volume_error_l:+.2f} L",
                f"PTV V95 error: {delta.ptv_peak_v95_error_percent:+.2f} pp",
                f"Gamma pass error: {delta.gamma_min_pass_error_percent:+.2f} pp",
                f"Waist error: {delta.achieved_waist_error_cm:+.2f} cm",
            ]
        )
    axes[2].text(
        0.02,
        0.98,
        "\n".join(summary),
        va="top",
        ha="left",
        fontsize=11,
        bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.55"},
    )
    for axis in axes[:2]:
        for row in rows:
            axis.annotate(
                str(row.get("profile_id", "")).replace("_height175", ""),
                (_as_float(row.get("target_waist_cm")), _as_float(row.get("body_volume_l")) if axis is axes[0] else _as_float(row.get("ptv_peak_v95_percent"))),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color="#1f2933",
            )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _prediction_delta_payload(delta: PredictionDelta | None) -> dict[str, Any] | None:
    if delta is None:
        return None
    return {
        "profile_id": delta.profile_id,
        "body_volume_l": {
            "expected": delta.expected_body_volume_l,
            "actual": delta.actual_body_volume_l,
            "error": delta.body_volume_error_l,
        },
        "ptv_peak_v95_percent": {
            "expected": delta.expected_ptv_peak_v95_percent,
            "actual": delta.actual_ptv_peak_v95_percent,
            "error": delta.ptv_peak_v95_error_percent,
        },
        "gamma_min_pass_rate_percent": {
            "expected": delta.expected_gamma_min_pass_rate_percent,
            "actual": delta.actual_gamma_min_pass_rate_percent,
            "error": delta.gamma_min_pass_error_percent,
        },
        "waist_cm": {
            "target": delta.target_waist_cm,
            "achieved": delta.achieved_waist_cm,
            "error": delta.achieved_waist_error_cm,
        },
    }


def _write_manifest(path: Path, result: ProfileEnvelopeResult) -> None:
    payload = {
        "envelope_id": result.envelope_id,
        "package_type": "consolidated_profile_operating_envelope",
        "source_metrics_csv_paths": list(result.source_metrics_csv_paths),
        "source_prescription_yaml": result.prescription_yaml_path,
        "outputs": {
            "metrics_csv": result.metrics_csv_path,
            "manifest_yaml": result.manifest_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "summary": {
            "profile_count": result.profile_count,
            "pass_count": result.pass_count,
            "warn_count": result.warn_count,
            "fail_count": result.fail_count,
            "waist_range_cm": list(result.waist_range_cm),
            "bmi_range": list(result.bmi_range),
            "high_bmi_profile_count": result.high_bmi_profile_count,
        },
        "prediction_delta": _prediction_delta_payload(result.prediction_delta),
        "profiles": list(result.rows),
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: ProfileEnvelopeResult) -> None:
    preview_link = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    lines = [
        "# Consolidated Profile Operating Envelope",
        "",
        f"Envelope ID: `{result.envelope_id}`",
        "",
        f"![Profile envelope preview]({preview_link})",
        "",
        "## Summary",
        "",
        f"- Profiles: {result.profile_count}",
        f"- Pass / warn / fail: {result.pass_count} / {result.warn_count} / {result.fail_count}",
        f"- Waist range: {result.waist_range_cm[0]:.1f}-{result.waist_range_cm[1]:.1f} cm",
        f"- BMI range: {result.bmi_range[0]:.1f}-{result.bmi_range[1]:.1f}",
        f"- High-BMI padded profiles: {result.high_bmi_profile_count}",
        "",
        "## Profile Table",
        "",
        "| profile | BMI | target waist cm | achieved waist cm | body L | morph | anatomy | vascular | flow | RT | overall | PTV V95 % | gamma pass % |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.get('profile_id', '')} | {_as_float(row.get('target_bmi')):.1f} | {_as_float(row.get('target_waist_cm')):.1f} | "
            f"{_as_float(row.get('achieved_waist_cm')):.1f} | {_as_float(row.get('body_volume_l')):.2f} | "
            f"{row.get('morph_mode', '')} | {row.get('anatomy_status', '')} | {row.get('vascular_status', '')} | "
            f"{row.get('flow_status', '')} | {row.get('rt_status', '')} | {row.get('overall_status', '')} | "
            f"{_as_float(row.get('ptv_peak_v95_percent')):.2f} | {_as_float(row.get('gamma_min_pass_rate_percent')):.2f} |"
        )
    if result.prediction_delta is not None:
        delta = result.prediction_delta
        lines.extend(
            [
                "",
                "## Prescription Validation",
                "",
                f"- Profile: `{delta.profile_id}`",
                f"- Body-volume prediction error: {delta.body_volume_error_l:+.2f} L",
                f"- PTV peak V95 prediction error: {delta.ptv_peak_v95_error_percent:+.2f} percentage points",
                f"- Gamma min pass prediction error: {delta.gamma_min_pass_error_percent:+.2f} percentage points",
                f"- Achieved waist error: {delta.achieved_waist_error_cm:+.2f} cm",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Consolidated metrics CSV: `{result.metrics_csv_path}`",
            f"- Manifest YAML: `{result.manifest_yaml_path}`",
            f"- Preview PNG: `{result.preview_png_path}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_profile_operating_envelope(
    metrics_csv_paths: tuple[str | Path, ...],
    output_dir: str | Path = "outputs/experiments/profile_envelope",
    envelope_id: str = "ct_org_profile_envelope_stage001",
    prescription_yaml_path: str | Path | None = None,
    report_path: str | Path | None = "outputs/reports/profile_operating_envelope_stage001.md",
) -> ProfileEnvelopeResult:
    rows = _merge_rows(metrics_csv_paths)
    if not rows:
        raise ValueError("No profile rows were found in the supplied metrics CSV files")
    output = Path(output_dir)
    metrics = output / f"{envelope_id}_consolidated_profile_metrics_v001.csv"
    manifest = output / f"{envelope_id}_profile_envelope_manifest_v001.yaml"
    preview = output / f"{envelope_id}_profile_envelope_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{envelope_id}_profile_envelope_report_v001.md"

    pass_count, warn_count, fail_count = _status_counts(rows)
    waist_range = _range([_as_float(row.get("target_waist_cm")) for row in rows if str(row.get("overall_status", "")).lower() == "pass"])
    bmi_range = _range([_as_float(row.get("target_bmi")) for row in rows if str(row.get("overall_status", "")).lower() == "pass"])
    high_bmi_count = sum(1 for row in rows if str(row.get("morph_mode", "")) == "high-bmi")
    delta = _prediction_delta(prescription_yaml_path, rows)
    notes = [
        "consolidated_envelope_merges_profile_sweep_metrics_without_rerunning_simulations",
        "duplicate_profile_ids_are_resolved_by_using_the_last_supplied_metrics_csv",
        "profile_envelope_is_an_engineering_validation_artifact_not_population_clinical_validation",
    ]
    if delta is not None:
        notes.append("prescription_prediction_delta_compares_interpolated_triage_estimate_to_actual_rerun")

    result = ProfileEnvelopeResult(
        envelope_id=envelope_id,
        output_dir=str(output),
        metrics_csv_path=str(metrics),
        manifest_yaml_path=str(manifest),
        preview_png_path=str(preview),
        report_path=str(report),
        source_metrics_csv_paths=tuple(str(path) for path in metrics_csv_paths),
        prescription_yaml_path=None if prescription_yaml_path is None else str(prescription_yaml_path),
        profile_count=len(rows),
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        waist_range_cm=waist_range,
        bmi_range=bmi_range,
        high_bmi_profile_count=high_bmi_count,
        rows=rows,
        prediction_delta=delta,
        notes=tuple(notes),
    )
    _write_metrics_csv(metrics, rows)
    _write_preview(preview, result)
    _write_manifest(manifest, result)
    _write_report(report, result)
    return result


def format_profile_envelope_result(result: ProfileEnvelopeResult) -> str:
    return "\n".join(
        [
            "Profile operating envelope consolidated",
            f"Envelope ID: {result.envelope_id}",
            f"Profiles: {result.profile_count}",
            f"Pass/warn/fail: {result.pass_count}/{result.warn_count}/{result.fail_count}",
            f"Waist range: {result.waist_range_cm[0]:.1f}-{result.waist_range_cm[1]:.1f} cm",
            f"High-BMI profiles: {result.high_bmi_profile_count}",
            f"Metrics CSV: {result.metrics_csv_path}",
            f"Manifest YAML: {result.manifest_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
