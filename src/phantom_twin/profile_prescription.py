from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import os
from typing import Any

import yaml


@dataclass(frozen=True)
class ProfilePrescriptionResult:
    case_id: str
    profile_id: str
    output_dir: str
    prescription_yaml_path: str
    preview_png_path: str
    report_path: str
    source_metrics_csv_path: str
    target_height_cm: float
    target_weight_kg: float | None
    target_bmi: float
    target_waist_cm: float
    fit_status: str
    morph_mode: str
    xy_padding_voxels: int
    nearest_profile_id: str
    lower_profile_id: str
    upper_profile_id: str
    pass_waist_range_cm: tuple[float, float]
    pass_bmi_range: tuple[float, float]
    pass_height_range_cm: tuple[float, float]
    interpolated_body_volume_l: float
    interpolated_ptv_peak_v95_percent: float
    interpolated_gamma_min_pass_rate_percent: float
    validation_recommendation: str
    recommended_commands: tuple[str, ...]
    notes: tuple[str, ...]


def _slug(raw: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "profile"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_rows(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Profile sweep metrics CSV not found: {path}")
    with resolved.open(newline="") as csvfile:
        return [dict(row) for row in csv.DictReader(csvfile)]


def _derive_target_bmi(target_height_cm: float, target_weight_kg: float | None, target_bmi: float | None) -> float:
    if target_bmi is not None:
        return float(target_bmi)
    if target_weight_kg is None:
        raise ValueError("Provide either target_bmi or target_weight_kg")
    height_m = target_height_cm / 100.0
    if height_m <= 0.0:
        raise ValueError("target_height_cm must be positive")
    return float(target_weight_kg / (height_m**2))


def _derive_target_waist(
    rows: list[dict[str, str]],
    target_waist_cm: float | None,
    target_bmi: float,
    baseline_bmi: float,
    baseline_waist_cm: float | None,
) -> float:
    if target_waist_cm is not None:
        return float(target_waist_cm)
    if baseline_bmi <= 0.0:
        raise ValueError("baseline_bmi must be positive")
    baseline_waist = baseline_waist_cm
    if baseline_waist is None:
        if not rows:
            raise ValueError("Cannot infer target waist without sweep rows or baseline_waist_cm")
        nearest = min(rows, key=lambda row: abs(_as_float(row.get("target_bmi")) - baseline_bmi))
        baseline_waist = _as_float(nearest.get("target_waist_cm"), 95.0)
    return float(baseline_waist * math.sqrt(max(target_bmi, 1e-6) / baseline_bmi))


def _range(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (float(min(values)), float(max(values)))


def _interpolate(rows: list[dict[str, str]], target_waist_cm: float, field: str) -> float:
    sorted_rows = sorted(rows, key=lambda row: _as_float(row.get("target_waist_cm")))
    if not sorted_rows:
        return 0.0
    if len(sorted_rows) == 1 or target_waist_cm <= _as_float(sorted_rows[0].get("target_waist_cm")):
        return _as_float(sorted_rows[0].get(field))
    if target_waist_cm >= _as_float(sorted_rows[-1].get("target_waist_cm")):
        return _as_float(sorted_rows[-1].get(field))
    for lower, upper in zip(sorted_rows, sorted_rows[1:]):
        lower_waist = _as_float(lower.get("target_waist_cm"))
        upper_waist = _as_float(upper.get("target_waist_cm"))
        if lower_waist <= target_waist_cm <= upper_waist:
            span = max(upper_waist - lower_waist, 1e-6)
            fraction = (target_waist_cm - lower_waist) / span
            return _as_float(lower.get(field)) + fraction * (_as_float(upper.get(field)) - _as_float(lower.get(field)))
    return _as_float(sorted_rows[-1].get(field))


def _bracketing_rows(rows: list[dict[str, str]], target_waist_cm: float) -> tuple[dict[str, str], dict[str, str]]:
    sorted_rows = sorted(rows, key=lambda row: _as_float(row.get("target_waist_cm")))
    if not sorted_rows:
        raise ValueError("No passing profile rows are available for prescription")
    lower = sorted_rows[0]
    upper = sorted_rows[-1]
    for row in sorted_rows:
        if _as_float(row.get("target_waist_cm")) <= target_waist_cm:
            lower = row
        if _as_float(row.get("target_waist_cm")) >= target_waist_cm:
            upper = row
            break
    return lower, upper


def _fit_status(
    target_bmi: float,
    target_waist_cm: float,
    target_height_cm: float,
    nearest: dict[str, str],
    pass_waist_range: tuple[float, float],
    pass_bmi_range: tuple[float, float],
    pass_height_range: tuple[float, float],
    waist_tolerance_cm: float,
    bmi_tolerance: float,
    height_tolerance_cm: float,
) -> str:
    if target_waist_cm < pass_waist_range[0] or target_waist_cm > pass_waist_range[1]:
        return "outside_validated_sweep_envelope"
    bmi_inside = pass_bmi_range[0] - bmi_tolerance <= target_bmi <= pass_bmi_range[1] + bmi_tolerance
    height_inside = pass_height_range[0] - height_tolerance_cm <= target_height_cm <= pass_height_range[1] + height_tolerance_cm
    if not bmi_inside or not height_inside:
        return "inside_waist_envelope_with_bmi_or_height_extrapolation"
    nearest_waist_delta = abs(_as_float(nearest.get("target_waist_cm")) - target_waist_cm)
    nearest_bmi_delta = abs(_as_float(nearest.get("target_bmi")) - target_bmi)
    if nearest_waist_delta <= waist_tolerance_cm and nearest_bmi_delta <= bmi_tolerance:
        return "matched_validated_sweep_profile"
    return "interpolated_inside_validated_sweep_envelope"


def _recommended_commands(
    case_id: str,
    profile_id: str,
    target_bmi: float,
    target_waist_cm: float,
    target_height_cm: float,
    combined_spec_path: str | Path,
    high_bmi_waist_threshold_cm: float,
    high_bmi_xy_padding_voxels: int,
    padding_transition_margin_cm: float,
) -> tuple[str, ...]:
    profile_spec = f"{profile_id}:{target_bmi:.3f}:{target_waist_cm:.3f}:{target_height_cm:.3f}"
    morph_mode = "high-bmi" if target_waist_cm >= high_bmi_waist_threshold_cm else "standard"
    padding_threshold_cm = high_bmi_waist_threshold_cm - max(0.0, float(padding_transition_margin_cm))
    xy_padding = high_bmi_xy_padding_voxels if target_waist_cm >= padding_threshold_cm else 0
    return (
        (
            "python -m phantom_twin.cli build-profile-sweep "
            f"--sweep-id {case_id} "
            f"--output-dir outputs/experiments/profile_sweep/{profile_id} "
            f"--profile {profile_spec} "
            "--skip-dicom "
            "--gamma-random-subset 25000 "
            f"--high-bmi-waist-threshold-cm {high_bmi_waist_threshold_cm:.3f} "
            f"--high-bmi-xy-padding-voxels {int(high_bmi_xy_padding_voxels)} "
            f"--padding-transition-margin-cm {float(padding_transition_margin_cm):.3f}"
        ),
        (
            "python -m phantom_twin.cli build-anthropometric-torso-morph "
            f"--combined-spec {combined_spec_path} "
            f"--case-id {case_id} "
            f"--target-height-cm {target_height_cm:.3f} "
            f"--target-bmi {target_bmi:.3f} "
            f"--target-waist-cm {target_waist_cm:.3f} "
            f"--morph-mode {morph_mode} "
            f"--xy-padding-voxels {int(xy_padding)}"
        ),
    )


def _write_preview(path: Path, rows: list[dict[str, str]], result: ProfilePrescriptionResult) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Profile prescription preview generation requires matplotlib.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: _as_float(row.get("target_waist_cm")))
    waist = [_as_float(row.get("target_waist_cm")) for row in ordered]
    body_l = [_as_float(row.get("body_volume_l")) for row in ordered]
    ptv_v95 = [_as_float(row.get("ptv_peak_v95_percent")) for row in ordered]
    gamma = [_as_float(row.get("gamma_min_pass_rate_percent")) for row in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=170)
    fig.suptitle(f"Profile Prescription Envelope\n{result.profile_id}", fontsize=15, fontweight="bold")

    axes[0].axvspan(result.pass_waist_range_cm[0], result.pass_waist_range_cm[1], color="#d8f3dc", alpha=0.55, label="validated pass envelope")
    axes[0].plot(waist, body_l, marker="o", color="#264653", linewidth=2.0, label="sweep profiles")
    axes[0].scatter(result.target_waist_cm, result.interpolated_body_volume_l, marker="*", s=220, color="#e63946", edgecolor="#1f2933", label="target")
    for row in ordered:
        axes[0].annotate(str(row.get("profile_id", "")), (_as_float(row.get("target_waist_cm")), _as_float(row.get("body_volume_l"))), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("target waist (cm)")
    axes[0].set_ylabel("body volume (L)")
    axes[0].set_title(f"{result.fit_status.replace('_', ' ')}\nmode: {result.morph_mode}, pad {result.xy_padding_voxels} vox/side")
    axes[0].grid(True, color="#d8dee9", linewidth=0.7)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(waist, ptv_v95, marker="o", color="#457b9d", linewidth=2.0, label="PTV peak V95 %")
    axes[1].plot(waist, gamma, marker="s", color="#2a9d8f", linewidth=2.0, label="gamma min pass %")
    axes[1].axvline(result.target_waist_cm, color="#e63946", linestyle="--", linewidth=1.6, label="target waist")
    axes[1].scatter(result.target_waist_cm, result.interpolated_ptv_peak_v95_percent, marker="*", s=160, color="#e63946", edgecolor="#1f2933")
    axes[1].set_xlabel("target waist (cm)")
    axes[1].set_ylabel("QA percent")
    axes[1].set_ylim(75, 103)
    axes[1].set_title("Interpolated RT-flow QA expectation")
    axes[1].grid(True, color="#d8dee9", linewidth=0.7)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_yaml(path: Path, result: ProfilePrescriptionResult, rows: list[dict[str, str]]) -> None:
    payload = {
        "case_id": result.case_id,
        "profile_id": result.profile_id,
        "package_type": "profile_operating_envelope_prescription",
        "source_metrics_csv": result.source_metrics_csv_path,
        "target": {
            "height_cm": result.target_height_cm,
            "weight_kg": result.target_weight_kg,
            "bmi": result.target_bmi,
            "waist_cm": result.target_waist_cm,
        },
        "prescription": {
            "fit_status": result.fit_status,
            "morph_mode": result.morph_mode,
            "xy_padding_voxels": result.xy_padding_voxels,
            "nearest_profile_id": result.nearest_profile_id,
            "lower_profile_id": result.lower_profile_id,
            "upper_profile_id": result.upper_profile_id,
            "validation_recommendation": result.validation_recommendation,
        },
        "validated_envelope": {
            "pass_waist_range_cm": list(result.pass_waist_range_cm),
            "pass_bmi_range": list(result.pass_bmi_range),
            "pass_height_range_cm": list(result.pass_height_range_cm),
        },
        "interpolated_expectations": {
            "body_volume_l": result.interpolated_body_volume_l,
            "ptv_peak_v95_percent": result.interpolated_ptv_peak_v95_percent,
            "gamma_min_pass_rate_percent": result.interpolated_gamma_min_pass_rate_percent,
        },
        "outputs": {
            "prescription_yaml": result.prescription_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "recommended_commands": list(result.recommended_commands),
        "source_rows": rows,
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: ProfilePrescriptionResult) -> None:
    preview_link = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    lines = [
        "# Profile Operating Envelope Prescription",
        "",
        f"Case ID: `{result.case_id}`",
        f"Profile ID: `{result.profile_id}`",
        "",
        f"![Profile prescription preview]({preview_link})",
        "",
        "## Target",
        "",
        f"- Height: {result.target_height_cm:.1f} cm",
        f"- Weight: {'not supplied' if result.target_weight_kg is None else f'{result.target_weight_kg:.1f} kg'}",
        f"- BMI: {result.target_bmi:.2f}",
        f"- Waist: {result.target_waist_cm:.2f} cm",
        "",
        "## Prescription",
        "",
        f"- Fit status: `{result.fit_status}`",
        f"- Morph mode: `{result.morph_mode}`",
        f"- XY padding: {result.xy_padding_voxels} voxels per side",
        f"- Nearest validated profile: `{result.nearest_profile_id}`",
        f"- Bracketing profiles: `{result.lower_profile_id}` to `{result.upper_profile_id}`",
        f"- Validation recommendation: `{result.validation_recommendation}`",
        "",
        "## Validated Envelope",
        "",
        f"- Passing waist range: {result.pass_waist_range_cm[0]:.1f}-{result.pass_waist_range_cm[1]:.1f} cm",
        f"- Passing BMI range: {result.pass_bmi_range[0]:.1f}-{result.pass_bmi_range[1]:.1f}",
        f"- Passing height range: {result.pass_height_range_cm[0]:.1f}-{result.pass_height_range_cm[1]:.1f} cm",
        "",
        "## Expected QA If Rerun",
        "",
        f"- Interpolated body volume: {result.interpolated_body_volume_l:.2f} L",
        f"- Interpolated PTV peak V95: {result.interpolated_ptv_peak_v95_percent:.2f}%",
        f"- Interpolated gamma min pass: {result.interpolated_gamma_min_pass_rate_percent:.2f}%",
        "",
        "## Recommended Commands",
        "",
    ]
    for command in result.recommended_commands:
        lines.append(f"```bash\n{command}\n```")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Prescription YAML: `{result.prescription_yaml_path}`",
            f"- Preview PNG: `{result.preview_png_path}`",
            f"- Source metrics CSV: `{result.source_metrics_csv_path}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_profile_operating_prescription(
    metrics_csv_path: str | Path = "outputs/experiments/profile_sweep/ct_org_profile_sweep_stage001_profile_sweep_metrics_v001.csv",
    output_dir: str | Path = "outputs/digital/profile_prescriptions",
    profile_id: str = "bmi35_waist118_height175",
    case_id: str | None = None,
    target_height_cm: float = 175.0,
    target_weight_kg: float | None = None,
    target_bmi: float | None = 35.0,
    target_waist_cm: float | None = 118.0,
    baseline_bmi: float = 24.0,
    baseline_waist_cm: float | None = None,
    waist_tolerance_cm: float = 3.0,
    bmi_tolerance: float = 2.0,
    height_tolerance_cm: float = 10.0,
    high_bmi_waist_threshold_cm: float = 115.0,
    high_bmi_xy_padding_voxels: int = 96,
    padding_transition_margin_cm: float = 5.0,
    combined_spec_path: str | Path = "outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml",
    report_path: str | Path | None = "outputs/reports/profile_operating_prescription_stage001.md",
) -> ProfilePrescriptionResult:
    rows = _load_rows(metrics_csv_path)
    pass_rows = [row for row in rows if str(row.get("overall_status", "")).lower() == "pass"]
    if not pass_rows:
        raise ValueError("Profile sweep metrics contain no passing profiles")

    bmi = _derive_target_bmi(target_height_cm=target_height_cm, target_weight_kg=target_weight_kg, target_bmi=target_bmi)
    waist = _derive_target_waist(
        rows=pass_rows,
        target_waist_cm=target_waist_cm,
        target_bmi=bmi,
        baseline_bmi=baseline_bmi,
        baseline_waist_cm=baseline_waist_cm,
    )
    nearest = min(
        pass_rows,
        key=lambda row: (
            abs(_as_float(row.get("target_waist_cm")) - waist) / max(waist_tolerance_cm, 1e-6)
            + abs(_as_float(row.get("target_bmi")) - bmi) / max(bmi_tolerance, 1e-6)
        ),
    )
    lower, upper = _bracketing_rows(pass_rows, waist)
    pass_waist_range = _range([_as_float(row.get("target_waist_cm")) for row in pass_rows])
    pass_bmi_range = _range([_as_float(row.get("target_bmi")) for row in pass_rows])
    pass_height_range = _range([_as_float(row.get("target_height_cm")) for row in pass_rows])
    status = _fit_status(
        target_bmi=bmi,
        target_waist_cm=waist,
        target_height_cm=target_height_cm,
        nearest=nearest,
        pass_waist_range=pass_waist_range,
        pass_bmi_range=pass_bmi_range,
        pass_height_range=pass_height_range,
        waist_tolerance_cm=waist_tolerance_cm,
        bmi_tolerance=bmi_tolerance,
        height_tolerance_cm=height_tolerance_cm,
    )
    morph_mode = "high-bmi" if waist >= high_bmi_waist_threshold_cm else "standard"
    padding_threshold_cm = high_bmi_waist_threshold_cm - max(0.0, float(padding_transition_margin_cm))
    xy_padding = int(high_bmi_xy_padding_voxels) if waist >= padding_threshold_cm else 0
    validation_recommendation = (
        "run_single_profile_full_pipeline_before_claims"
        if status in {"outside_validated_sweep_envelope", "inside_waist_envelope_with_bmi_or_height_extrapolation"}
        else "single_profile_rerun_recommended_for_final_outputs"
    )

    resolved_profile_id = _slug(profile_id)
    resolved_case_id = case_id or f"profile_prescription_{resolved_profile_id}"
    output = Path(output_dir)
    yaml_path = output / f"{resolved_case_id}_{resolved_profile_id}_profile_prescription_v001.yaml"
    preview = output / f"{resolved_case_id}_{resolved_profile_id}_profile_prescription_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{resolved_case_id}_{resolved_profile_id}_profile_prescription_report_v001.md"
    commands = _recommended_commands(
        case_id=resolved_case_id,
        profile_id=resolved_profile_id,
        target_bmi=bmi,
        target_waist_cm=waist,
        target_height_cm=target_height_cm,
        combined_spec_path=combined_spec_path,
        high_bmi_waist_threshold_cm=high_bmi_waist_threshold_cm,
        high_bmi_xy_padding_voxels=high_bmi_xy_padding_voxels,
        padding_transition_margin_cm=padding_transition_margin_cm,
    )
    notes = [
        "prescription_is_derived_from_profile_sweep_metrics_not_a_new_simulation",
        "interpolated_expectations_are_for_triage_only_full_pipeline_rerun_is_required_for_final_outputs",
        "profile_adaptation_remains_a_digital_engineering_phantom_not_subject_specific_anatomical_equivalence",
    ]
    if status == "outside_validated_sweep_envelope":
        notes.append("target_profile_is_outside_current_pass_envelope_add_a_new_sweep_point")
    if morph_mode == "high-bmi":
        notes.append("high_bmi_mode_uses_expanded_canvas_and_radial_calibration_limits")
    if xy_padding > 0:
        notes.append("xy_padding_transition_band_helps_avoid_waist_fit_discontinuities_near_high_bmi_threshold")

    result = ProfilePrescriptionResult(
        case_id=resolved_case_id,
        profile_id=resolved_profile_id,
        output_dir=str(output),
        prescription_yaml_path=str(yaml_path),
        preview_png_path=str(preview),
        report_path=str(report),
        source_metrics_csv_path=str(metrics_csv_path),
        target_height_cm=float(target_height_cm),
        target_weight_kg=target_weight_kg,
        target_bmi=bmi,
        target_waist_cm=waist,
        fit_status=status,
        morph_mode=morph_mode,
        xy_padding_voxels=xy_padding,
        nearest_profile_id=str(nearest.get("profile_id", "")),
        lower_profile_id=str(lower.get("profile_id", "")),
        upper_profile_id=str(upper.get("profile_id", "")),
        pass_waist_range_cm=pass_waist_range,
        pass_bmi_range=pass_bmi_range,
        pass_height_range_cm=pass_height_range,
        interpolated_body_volume_l=_interpolate(pass_rows, waist, "body_volume_l"),
        interpolated_ptv_peak_v95_percent=_interpolate(pass_rows, waist, "ptv_peak_v95_percent"),
        interpolated_gamma_min_pass_rate_percent=_interpolate(pass_rows, waist, "gamma_min_pass_rate_percent"),
        validation_recommendation=validation_recommendation,
        recommended_commands=commands,
        notes=tuple(notes),
    )
    _write_preview(preview, pass_rows, result)
    _write_yaml(yaml_path, result, pass_rows)
    _write_report(report, result)
    return result


def format_profile_prescription_result(result: ProfilePrescriptionResult) -> str:
    return "\n".join(
        [
            "Profile operating-envelope prescription created",
            f"Case ID: {result.case_id}",
            f"Profile ID: {result.profile_id}",
            f"Target BMI/waist/height: {result.target_bmi:.2f} / {result.target_waist_cm:.2f} cm / {result.target_height_cm:.1f} cm",
            f"Fit status: {result.fit_status}",
            f"Morph mode: {result.morph_mode}",
            f"XY padding voxels/side: {result.xy_padding_voxels}",
            f"Nearest validated profile: {result.nearest_profile_id}",
            f"Prescription YAML: {result.prescription_yaml_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
