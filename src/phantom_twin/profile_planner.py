from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import yaml


@dataclass(frozen=True)
class PlannedProfileCandidate:
    rank: int
    profile_id: str
    target_bmi: float
    target_waist_cm: float
    target_height_cm: float
    morph_mode: str
    xy_padding_voxels: int
    priority_score: float
    reason: str
    lower_profile_id: str
    upper_profile_id: str
    recommended_command: str


@dataclass(frozen=True)
class ProfilePlanningResult:
    plan_id: str
    output_dir: str
    source_metrics_csv_path: str
    plan_yaml_path: str
    candidates_csv_path: str
    preview_png_path: str
    report_path: str
    candidate_count: int
    existing_profile_count: int
    high_bmi_threshold_cm: float
    top_candidates: tuple[PlannedProfileCandidate, ...]
    notes: tuple[str, ...]


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
        raise FileNotFoundError(f"Profile metrics CSV not found: {path}")
    with resolved.open(newline="") as csvfile:
        return [dict(row) for row in csv.DictReader(csvfile)]


def _slug_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}".replace(".", "p")


def _profile_id(bmi: float, waist_cm: float, height_cm: float) -> str:
    return f"bmi{_slug_number(bmi)}_waist{_slug_number(waist_cm)}_height{_slug_number(height_cm)}"


def _interp(lower: dict[str, str], upper: dict[str, str], waist_cm: float, field: str) -> float:
    lower_waist = _as_float(lower.get("target_waist_cm"))
    upper_waist = _as_float(upper.get("target_waist_cm"))
    if abs(upper_waist - lower_waist) < 1e-6:
        return _as_float(lower.get(field))
    fraction = (waist_cm - lower_waist) / (upper_waist - lower_waist)
    return _as_float(lower.get(field)) + fraction * (_as_float(upper.get(field)) - _as_float(lower.get(field)))


def _nearest_existing_distance(rows: list[dict[str, str]], waist_cm: float) -> float:
    return min((abs(_as_float(row.get("target_waist_cm")) - waist_cm) for row in rows), default=999.0)


def _command(
    plan_id: str,
    candidate: PlannedProfileCandidate,
    high_bmi_threshold_cm: float,
    high_bmi_xy_padding_voxels: int,
    padding_transition_margin_cm: float,
    gamma_random_subset: int,
) -> str:
    return (
        "python -m phantom_twin.cli build-profile-sweep "
        f"--sweep-id {plan_id}_{candidate.profile_id} "
        f"--output-dir outputs/experiments/profile_sweep/{candidate.profile_id} "
        f"--profile {candidate.profile_id}:{candidate.target_bmi:.3f}:{candidate.target_waist_cm:.3f}:{candidate.target_height_cm:.3f} "
        "--skip-dicom "
        f"--gamma-random-subset {int(gamma_random_subset)} "
        f"--high-bmi-waist-threshold-cm {high_bmi_threshold_cm:.3f} "
        f"--high-bmi-xy-padding-voxels {int(high_bmi_xy_padding_voxels)} "
        f"--padding-transition-margin-cm {float(padding_transition_margin_cm):.3f}"
    )


def _candidate_from_interval(
    plan_id: str,
    lower: dict[str, str],
    upper: dict[str, str],
    waist_cm: float,
    reason: str,
    score: float,
    high_bmi_threshold_cm: float,
    high_bmi_xy_padding_voxels: int,
    padding_transition_margin_cm: float,
    gamma_random_subset: int,
) -> PlannedProfileCandidate:
    bmi = _interp(lower, upper, waist_cm, "target_bmi")
    height = _interp(lower, upper, waist_cm, "target_height_cm")
    morph_mode = "high-bmi" if waist_cm >= high_bmi_threshold_cm else "standard"
    padding_threshold_cm = high_bmi_threshold_cm - max(0.0, float(padding_transition_margin_cm))
    xy_padding = high_bmi_xy_padding_voxels if waist_cm >= padding_threshold_cm else 0
    stub = PlannedProfileCandidate(
        rank=0,
        profile_id=_profile_id(bmi, waist_cm, height),
        target_bmi=float(bmi),
        target_waist_cm=float(waist_cm),
        target_height_cm=float(height),
        morph_mode=morph_mode,
        xy_padding_voxels=int(xy_padding),
        priority_score=float(score),
        reason=reason,
        lower_profile_id=str(lower.get("profile_id", "")),
        upper_profile_id=str(upper.get("profile_id", "")),
        recommended_command="",
    )
    return PlannedProfileCandidate(
        **{
            **stub.__dict__,
            "recommended_command": _command(
                plan_id=plan_id,
                candidate=stub,
                high_bmi_threshold_cm=high_bmi_threshold_cm,
                high_bmi_xy_padding_voxels=high_bmi_xy_padding_voxels,
                padding_transition_margin_cm=padding_transition_margin_cm,
                gamma_random_subset=gamma_random_subset,
            ),
        }
    )


def _dedupe_candidates(candidates: list[PlannedProfileCandidate]) -> list[PlannedProfileCandidate]:
    by_waist: dict[float, PlannedProfileCandidate] = {}
    for item in candidates:
        key = round(item.target_waist_cm, 3)
        previous = by_waist.get(key)
        if previous is None:
            by_waist[key] = item
            continue
        reason_parts = sorted(set(previous.reason.split(";") + item.reason.split(";")))
        if item.priority_score >= previous.priority_score:
            chosen = item
        else:
            chosen = previous
        by_waist[key] = PlannedProfileCandidate(
            **{
                **chosen.__dict__,
                "priority_score": max(previous.priority_score, item.priority_score),
                "reason": ";".join(reason_parts),
            }
        )
    return list(by_waist.values())


def _rank_candidates(candidates: list[PlannedProfileCandidate], max_candidates: int) -> tuple[PlannedProfileCandidate, ...]:
    ranked = sorted(candidates, key=lambda item: (-item.priority_score, item.target_waist_cm, item.profile_id))[:max_candidates]
    return tuple(
        PlannedProfileCandidate(
            **{
                **item.__dict__,
                "rank": rank,
            }
        )
        for rank, item in enumerate(ranked, start=1)
    )


def _write_candidates_csv(path: Path, candidates: tuple[PlannedProfileCandidate, ...]) -> None:
    fieldnames = list(PlannedProfileCandidate.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in candidates:
            writer.writerow({field: getattr(item, field) for field in fieldnames})


def _write_preview(path: Path, rows: list[dict[str, str]], result: ProfilePlanningResult) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Profile planning preview generation requires matplotlib.") from exc

    ordered = sorted(rows, key=lambda row: _as_float(row.get("target_waist_cm")))
    waist = [_as_float(row.get("target_waist_cm")) for row in ordered]
    body = [_as_float(row.get("body_volume_l")) for row in ordered]
    ptv = [_as_float(row.get("ptv_peak_v95_percent")) for row in ordered]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=170)
    fig.suptitle(f"Next Profile Planning\n{result.plan_id}", fontsize=15, fontweight="bold")

    axes[0].plot(waist, body, marker="o", color="#264653", linewidth=2.0, label="validated profiles")
    axes[0].axvline(result.high_bmi_threshold_cm, color="#e76f51", linestyle="--", linewidth=1.5, label="high-BMI threshold")
    for item in result.top_candidates:
        axes[0].scatter(item.target_waist_cm, _interp_for_plot(ordered, item.target_waist_cm, "body_volume_l"), marker="*", s=160, color="#e63946", edgecolor="#1f2933")
        axes[0].annotate(f"#{item.rank} {item.profile_id}", (item.target_waist_cm, _interp_for_plot(ordered, item.target_waist_cm, "body_volume_l")), xytext=(5, 5), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("target waist (cm)")
    axes[0].set_ylabel("body volume (L)")
    axes[0].set_title("Anatomy gaps to sample")
    axes[0].grid(True, color="#d8dee9", linewidth=0.7)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(waist, ptv, marker="o", color="#457b9d", linewidth=2.0, label="validated PTV V95")
    axes[1].axvline(result.high_bmi_threshold_cm, color="#e76f51", linestyle="--", linewidth=1.5, label="high-BMI threshold")
    axes[1].set_xlabel("target waist (cm)")
    axes[1].set_ylabel("PTV peak V95 (%)")
    axes[1].set_ylim(80, 90)
    axes[1].set_title("RT-flow QA gaps")
    axes[1].grid(True, color="#d8dee9", linewidth=0.7)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _interp_for_plot(rows: list[dict[str, str]], waist_cm: float, field: str) -> float:
    if not rows:
        return 0.0
    if waist_cm <= _as_float(rows[0].get("target_waist_cm")):
        return _as_float(rows[0].get(field))
    if waist_cm >= _as_float(rows[-1].get("target_waist_cm")):
        return _as_float(rows[-1].get(field))
    for lower, upper in zip(rows, rows[1:]):
        if _as_float(lower.get("target_waist_cm")) <= waist_cm <= _as_float(upper.get("target_waist_cm")):
            return _interp(lower, upper, waist_cm, field)
    return _as_float(rows[-1].get(field))


def _write_plan_yaml(path: Path, result: ProfilePlanningResult) -> None:
    payload = {
        "plan_id": result.plan_id,
        "package_type": "active_profile_validation_plan",
        "source_metrics_csv": result.source_metrics_csv_path,
        "summary": {
            "existing_profile_count": result.existing_profile_count,
            "candidate_count": result.candidate_count,
            "high_bmi_threshold_cm": result.high_bmi_threshold_cm,
        },
        "outputs": {
            "plan_yaml": result.plan_yaml_path,
            "candidates_csv": result.candidates_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "candidates": [
            {
                "rank": item.rank,
                "profile_id": item.profile_id,
                "target_bmi": item.target_bmi,
                "target_waist_cm": item.target_waist_cm,
                "target_height_cm": item.target_height_cm,
                "morph_mode": item.morph_mode,
                "xy_padding_voxels": item.xy_padding_voxels,
                "priority_score": item.priority_score,
                "reason": item.reason,
                "lower_profile_id": item.lower_profile_id,
                "upper_profile_id": item.upper_profile_id,
                "recommended_command": item.recommended_command,
            }
            for item in result.top_candidates
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: ProfilePlanningResult) -> None:
    preview_link = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    lines = [
        "# Active Profile Validation Plan",
        "",
        f"Plan ID: `{result.plan_id}`",
        "",
        f"![Profile planning preview]({preview_link})",
        "",
        "## Summary",
        "",
        f"- Existing validated profiles: {result.existing_profile_count}",
        f"- Candidate profiles: {result.candidate_count}",
        f"- High-BMI threshold: {result.high_bmi_threshold_cm:.1f} cm waist",
        "",
        "## Ranked Candidates",
        "",
        "| rank | profile | BMI | waist cm | morph | score | reason | bracket |",
        "| ---: | --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for item in result.top_candidates:
        lines.append(
            f"| {item.rank} | {item.profile_id} | {item.target_bmi:.2f} | {item.target_waist_cm:.2f} | "
            f"{item.morph_mode} | {item.priority_score:.2f} | {item.reason} | {item.lower_profile_id} to {item.upper_profile_id} |"
        )
    lines.extend(["", "## Recommended Commands", ""])
    for item in result.top_candidates:
        lines.append(f"### {item.rank}. {item.profile_id}")
        lines.append(f"```bash\n{item.recommended_command}\n```")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Plan YAML: `{result.plan_yaml_path}`",
            f"- Candidates CSV: `{result.candidates_csv_path}`",
            f"- Preview PNG: `{result.preview_png_path}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def plan_next_profile_validations(
    metrics_csv_path: str | Path = "outputs/experiments/profile_envelope/ct_org_profile_envelope_with_bmi35_stage001_consolidated_profile_metrics_v001.csv",
    output_dir: str | Path = "outputs/experiments/profile_planning",
    plan_id: str = "ct_org_next_profile_plan_stage001",
    high_bmi_waist_threshold_cm: float = 115.0,
    high_bmi_xy_padding_voxels: int = 96,
    padding_transition_margin_cm: float = 5.0,
    transition_margin_cm: float = 1.0,
    min_distance_from_existing_cm: float = 1.5,
    max_candidates: int = 5,
    gamma_random_subset: int = 25000,
    report_path: str | Path | None = "outputs/reports/next_profile_validation_plan_stage001.md",
) -> ProfilePlanningResult:
    rows = [row for row in _load_rows(metrics_csv_path) if str(row.get("overall_status", "")).lower() == "pass"]
    if len(rows) < 2:
        raise ValueError("At least two passing profiles are required for active profile planning")
    rows = sorted(rows, key=lambda row: _as_float(row.get("target_waist_cm")))

    candidates: list[PlannedProfileCandidate] = []
    for lower, upper in zip(rows, rows[1:]):
        lower_waist = _as_float(lower.get("target_waist_cm"))
        upper_waist = _as_float(upper.get("target_waist_cm"))
        gap = upper_waist - lower_waist
        if gap <= 0.0:
            continue
        body_delta = _as_float(upper.get("body_volume_l")) - _as_float(lower.get("body_volume_l"))
        ptv_delta = abs(_as_float(upper.get("ptv_peak_v95_percent")) - _as_float(lower.get("ptv_peak_v95_percent")))
        interval_score = gap * 2.0 + ptv_delta * 4.0
        reason = "largest_gap_midpoint"
        if body_delta < 0.0:
            interval_score += 30.0
            reason += ";nonmonotonic_body_volume_interval"
        midpoint = lower_waist + gap / 2.0
        if _nearest_existing_distance(rows, midpoint) >= min_distance_from_existing_cm:
            candidates.append(
                _candidate_from_interval(
                    plan_id,
                    lower,
                    upper,
                    midpoint,
                    reason,
                    interval_score,
                    high_bmi_waist_threshold_cm,
                    high_bmi_xy_padding_voxels,
                    padding_transition_margin_cm,
                    gamma_random_subset,
                )
            )
        if lower_waist < high_bmi_waist_threshold_cm < upper_waist:
            for waist_cm, threshold_reason, bonus in (
                (high_bmi_waist_threshold_cm - transition_margin_cm, "sample_below_high_bmi_transition", 75.0),
                (high_bmi_waist_threshold_cm + transition_margin_cm, "sample_above_high_bmi_transition", 78.0),
            ):
                if lower_waist < waist_cm < upper_waist and _nearest_existing_distance(rows, waist_cm) >= min_distance_from_existing_cm:
                    candidates.append(
                        _candidate_from_interval(
                            plan_id,
                            lower,
                            upper,
                            waist_cm,
                            threshold_reason,
                            interval_score + bonus,
                            high_bmi_waist_threshold_cm,
                            high_bmi_xy_padding_voxels,
                            padding_transition_margin_cm,
                            gamma_random_subset,
                        )
                    )
    ranked = _rank_candidates(_dedupe_candidates(candidates), max_candidates=max_candidates)

    output = Path(output_dir)
    plan_yaml = output / f"{plan_id}_active_profile_plan_v001.yaml"
    candidates_csv = output / f"{plan_id}_active_profile_candidates_v001.csv"
    preview = output / f"{plan_id}_active_profile_plan_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{plan_id}_active_profile_plan_report_v001.md"
    notes = (
        "planner_prioritizes_large_waist_gaps_high_bmi_transition_and_nonmonotonic_body_volume_intervals",
        "recommended_profiles_are_validation_targets_not_newly_simulated_outputs",
        "run_candidates_one_at_a_time_then_rebuild_the_consolidated_envelope",
    )
    result = ProfilePlanningResult(
        plan_id=plan_id,
        output_dir=str(output),
        source_metrics_csv_path=str(metrics_csv_path),
        plan_yaml_path=str(plan_yaml),
        candidates_csv_path=str(candidates_csv),
        preview_png_path=str(preview),
        report_path=str(report),
        candidate_count=len(ranked),
        existing_profile_count=len(rows),
        high_bmi_threshold_cm=float(high_bmi_waist_threshold_cm),
        top_candidates=ranked,
        notes=notes,
    )
    _write_candidates_csv(candidates_csv, result.top_candidates)
    _write_preview(preview, rows, result)
    _write_plan_yaml(plan_yaml, result)
    _write_report(report, result)
    return result


def format_profile_planning_result(result: ProfilePlanningResult) -> str:
    top = result.top_candidates[0] if result.top_candidates else None
    lines = [
        "Active profile validation plan created",
        f"Plan ID: {result.plan_id}",
        f"Existing profiles: {result.existing_profile_count}",
        f"Candidate profiles: {result.candidate_count}",
    ]
    if top is not None:
        lines.extend(
            [
                f"Top candidate: {top.profile_id}",
                f"Top candidate BMI/waist: {top.target_bmi:.2f} / {top.target_waist_cm:.2f} cm",
                f"Top candidate reason: {top.reason}",
            ]
        )
    lines.extend(
        [
            f"Plan YAML: {result.plan_yaml_path}",
            f"Candidates CSV: {result.candidates_csv_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
    return "\n".join(lines)
