from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import math

import yaml


@dataclass(frozen=True)
class RegistrationAnchorThresholds:
    approve_min_mean_dice: float = 0.70
    approve_min_target_mean_dice: float = 0.70
    approve_max_mean_volume_cv: float = 0.50
    approve_max_target_centroid_dispersion_mm: float = 125.0
    review_min_mean_dice: float = 0.35
    review_min_target_mean_dice: float = 0.25
    review_max_mean_volume_cv: float = 0.85
    review_max_target_centroid_dispersion_mm: float = 185.0


@dataclass(frozen=True)
class RegistrationAnchorDecision:
    rank: int
    label_id: int
    label_name: str
    decision: str
    score: float
    use_role: str
    target_count: int
    mean_dice: float
    min_target_mean_dice: float
    median_dice: float
    mean_volume_cv: float
    max_volume_cv: float
    mean_centroid_dispersion_mm: float
    max_centroid_dispersion_mm: float
    mean_consensus_volume_ml: float
    issues: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RegistrationAnchorQaResult:
    case_id: str
    benchmark_spec_path: str
    output_dir: str
    ranking_csv_path: str
    decisions_yaml_path: str
    report_path: str
    approved_anchor_labels: tuple[int, ...]
    review_anchor_labels: tuple[int, ...]
    rejected_anchor_labels: tuple[int, ...]
    decisions: tuple[RegistrationAnchorDecision, ...]
    thresholds: RegistrationAnchorThresholds
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    return data if isinstance(data, dict) else {}


def _as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _safe_min(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(min(finite)) if finite else float("nan")


def _safe_max(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(max(finite)) if finite else float("nan")


def _read_label_metrics(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            rows.append(
                {
                    "target_case_id": str(row.get("target_case_id", "")).strip(),
                    "label_id": _as_int(row.get("label_id")),
                    "label_name": str(row.get("label_name", "")).strip(),
                    "consensus_volume_ml": _as_float(row.get("consensus_volume_ml")),
                    "mean_propagated_volume_ml": _as_float(row.get("mean_propagated_volume_ml")),
                    "volume_cv": _as_float(row.get("volume_cv")),
                    "mean_dice_to_consensus": _as_float(row.get("mean_dice_to_consensus")),
                    "median_dice_to_consensus": _as_float(row.get("median_dice_to_consensus")),
                    "min_dice_to_consensus": _as_float(row.get("min_dice_to_consensus")),
                    "mean_centroid_dispersion_mm": _as_float(row.get("mean_centroid_dispersion_mm")),
                    "max_centroid_dispersion_mm": _as_float(row.get("max_centroid_dispersion_mm")),
                }
            )
    if not rows:
        raise ValueError(f"No registration label metrics found in {path}")
    return rows


def _score_label(
    *,
    mean_dice: float,
    min_target_mean_dice: float,
    mean_volume_cv: float,
    max_centroid_dispersion_mm: float,
    mean_consensus_volume_ml: float,
) -> float:
    score = 100.0
    score -= max(0.0, 0.80 - mean_dice) * 55.0
    score -= max(0.0, 0.60 - min_target_mean_dice) * 40.0
    score -= max(0.0, mean_volume_cv - 0.30) * 22.0
    score -= max(0.0, max_centroid_dispersion_mm - 75.0) / 75.0 * 18.0
    if mean_consensus_volume_ml < 5.0:
        score -= 30.0
    elif mean_consensus_volume_ml < 25.0:
        score -= 12.0
    return float(max(0.0, min(100.0, score)))


def _decision(
    *,
    mean_dice: float,
    min_target_mean_dice: float,
    mean_volume_cv: float,
    max_centroid_dispersion_mm: float,
    mean_consensus_volume_ml: float,
    thresholds: RegistrationAnchorThresholds,
) -> tuple[str, tuple[str, ...]]:
    issues: list[str] = []
    if mean_consensus_volume_ml < 1.0:
        issues.append("consensus_volume_near_zero")
    if mean_dice < thresholds.review_min_mean_dice:
        issues.append(f"mean_dice_below_review:{mean_dice:.3f}<{thresholds.review_min_mean_dice:.3f}")
    if min_target_mean_dice < thresholds.review_min_target_mean_dice:
        issues.append(
            f"target_mean_dice_below_review:{min_target_mean_dice:.3f}<{thresholds.review_min_target_mean_dice:.3f}"
        )
    if mean_volume_cv > thresholds.review_max_mean_volume_cv:
        issues.append(f"mean_volume_cv_above_review:{mean_volume_cv:.3f}>{thresholds.review_max_mean_volume_cv:.3f}")
    if max_centroid_dispersion_mm > thresholds.review_max_target_centroid_dispersion_mm:
        issues.append(
            "target_centroid_dispersion_above_review:"
            f"{max_centroid_dispersion_mm:.1f}>{thresholds.review_max_target_centroid_dispersion_mm:.1f}"
        )
    if issues:
        return "reject", tuple(issues)

    approve_issues: list[str] = []
    if mean_dice < thresholds.approve_min_mean_dice:
        approve_issues.append(f"mean_dice_below_approve:{mean_dice:.3f}<{thresholds.approve_min_mean_dice:.3f}")
    if min_target_mean_dice < thresholds.approve_min_target_mean_dice:
        approve_issues.append(
            f"target_mean_dice_below_approve:{min_target_mean_dice:.3f}<{thresholds.approve_min_target_mean_dice:.3f}"
        )
    if mean_volume_cv > thresholds.approve_max_mean_volume_cv:
        approve_issues.append(f"mean_volume_cv_above_approve:{mean_volume_cv:.3f}>{thresholds.approve_max_mean_volume_cv:.3f}")
    if max_centroid_dispersion_mm > thresholds.approve_max_target_centroid_dispersion_mm:
        approve_issues.append(
            "target_centroid_dispersion_above_approve:"
            f"{max_centroid_dispersion_mm:.1f}>{thresholds.approve_max_target_centroid_dispersion_mm:.1f}"
        )
    if approve_issues:
        return "review", tuple(approve_issues)
    return "approve", ()


def _use_role(decision: str) -> str:
    if decision == "approve":
        return "primary_deformation_anchor"
    if decision == "review":
        return "secondary_anchor_manual_review_required"
    return "excluded_from_deformation_anchor_set"


def _rank_decisions(decisions: list[RegistrationAnchorDecision]) -> tuple[RegistrationAnchorDecision, ...]:
    order = {"approve": 0, "review": 1, "reject": 2}
    ranked = sorted(decisions, key=lambda item: (order.get(item.decision, 9), -item.score, -item.mean_dice, item.label_id))
    return tuple(
        RegistrationAnchorDecision(
            rank=index,
            label_id=item.label_id,
            label_name=item.label_name,
            decision=item.decision,
            score=item.score,
            use_role=item.use_role,
            target_count=item.target_count,
            mean_dice=item.mean_dice,
            min_target_mean_dice=item.min_target_mean_dice,
            median_dice=item.median_dice,
            mean_volume_cv=item.mean_volume_cv,
            max_volume_cv=item.max_volume_cv,
            mean_centroid_dispersion_mm=item.mean_centroid_dispersion_mm,
            max_centroid_dispersion_mm=item.max_centroid_dispersion_mm,
            mean_consensus_volume_ml=item.mean_consensus_volume_ml,
            issues=item.issues,
            notes=item.notes,
        )
        for index, item in enumerate(ranked, start=1)
    )


def _build_decisions(
    rows: list[dict[str, Any]],
    thresholds: RegistrationAnchorThresholds,
) -> tuple[RegistrationAnchorDecision, ...]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["label_id"]), []).append(row)
    decisions: list[RegistrationAnchorDecision] = []
    for label_id, label_rows in sorted(grouped.items()):
        label_name = str(label_rows[0].get("label_name", f"label_{label_id}"))
        mean_dice = _safe_mean([float(row["mean_dice_to_consensus"]) for row in label_rows])
        min_target_mean_dice = _safe_min([float(row["mean_dice_to_consensus"]) for row in label_rows])
        median_dice = _safe_mean([float(row["median_dice_to_consensus"]) for row in label_rows])
        mean_volume_cv = _safe_mean([float(row["volume_cv"]) for row in label_rows])
        max_volume_cv = _safe_max([float(row["volume_cv"]) for row in label_rows])
        mean_centroid = _safe_mean([float(row["mean_centroid_dispersion_mm"]) for row in label_rows])
        max_centroid = _safe_max([float(row["max_centroid_dispersion_mm"]) for row in label_rows])
        mean_consensus_volume_ml = _safe_mean([float(row["consensus_volume_ml"]) for row in label_rows])
        decision, issues = _decision(
            mean_dice=mean_dice,
            min_target_mean_dice=min_target_mean_dice,
            mean_volume_cv=mean_volume_cv,
            max_centroid_dispersion_mm=max_centroid,
            mean_consensus_volume_ml=mean_consensus_volume_ml,
            thresholds=thresholds,
        )
        score = _score_label(
            mean_dice=mean_dice,
            min_target_mean_dice=min_target_mean_dice,
            mean_volume_cv=mean_volume_cv,
            max_centroid_dispersion_mm=max_centroid,
            mean_consensus_volume_ml=mean_consensus_volume_ml,
        )
        notes = ["stable_enough_for_anchor_use"] if decision == "approve" else []
        if decision == "review":
            notes.append("usable_only_with_manual_or_case_specific_review")
        if decision == "reject":
            notes.append("do_not_use_as_registration_deformation_anchor")
        decisions.append(
            RegistrationAnchorDecision(
                rank=0,
                label_id=label_id,
                label_name=label_name,
                decision=decision,
                score=score,
                use_role=_use_role(decision),
                target_count=len(label_rows),
                mean_dice=mean_dice,
                min_target_mean_dice=min_target_mean_dice,
                median_dice=median_dice,
                mean_volume_cv=mean_volume_cv,
                max_volume_cv=max_volume_cv,
                mean_centroid_dispersion_mm=mean_centroid,
                max_centroid_dispersion_mm=max_centroid,
                mean_consensus_volume_ml=mean_consensus_volume_ml,
                issues=issues,
                notes=tuple(notes),
            )
        )
    return _rank_decisions(decisions)


def _write_ranking_csv(path: Path, decisions: tuple[RegistrationAnchorDecision, ...]) -> None:
    fields = [
        "rank",
        "label_id",
        "label_name",
        "decision",
        "score",
        "use_role",
        "target_count",
        "mean_dice",
        "min_target_mean_dice",
        "median_dice",
        "mean_volume_cv",
        "max_volume_cv",
        "mean_centroid_dispersion_mm",
        "max_centroid_dispersion_mm",
        "mean_consensus_volume_ml",
        "issues",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for item in decisions:
            writer.writerow(
                {
                    "rank": item.rank,
                    "label_id": item.label_id,
                    "label_name": item.label_name,
                    "decision": item.decision,
                    "score": f"{item.score:.2f}",
                    "use_role": item.use_role,
                    "target_count": item.target_count,
                    "mean_dice": f"{item.mean_dice:.5f}",
                    "min_target_mean_dice": f"{item.min_target_mean_dice:.5f}",
                    "median_dice": f"{item.median_dice:.5f}",
                    "mean_volume_cv": f"{item.mean_volume_cv:.5f}",
                    "max_volume_cv": f"{item.max_volume_cv:.5f}",
                    "mean_centroid_dispersion_mm": f"{item.mean_centroid_dispersion_mm:.3f}",
                    "max_centroid_dispersion_mm": f"{item.max_centroid_dispersion_mm:.3f}",
                    "mean_consensus_volume_ml": f"{item.mean_consensus_volume_ml:.3f}",
                    "issues": ";".join(item.issues),
                }
            )


def _write_decisions_yaml(path: Path, result: RegistrationAnchorQaResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "registration_anchor_qa",
        "benchmark_spec": result.benchmark_spec_path,
        "summary": {
            "approved_anchor_labels": list(result.approved_anchor_labels),
            "review_anchor_labels": list(result.review_anchor_labels),
            "rejected_anchor_labels": list(result.rejected_anchor_labels),
        },
        "thresholds": result.thresholds.__dict__,
        "outputs": {
            "ranking_csv": result.ranking_csv_path,
            "decisions_yaml": result.decisions_yaml_path,
            "report": result.report_path,
        },
        "decisions": [
            {
                "rank": item.rank,
                "label_id": item.label_id,
                "label_name": item.label_name,
                "decision": item.decision,
                "score": item.score,
                "use_role": item.use_role,
                "target_count": item.target_count,
                "mean_dice": item.mean_dice,
                "min_target_mean_dice": item.min_target_mean_dice,
                "median_dice": item.median_dice,
                "mean_volume_cv": item.mean_volume_cv,
                "max_volume_cv": item.max_volume_cv,
                "mean_centroid_dispersion_mm": item.mean_centroid_dispersion_mm,
                "max_centroid_dispersion_mm": item.max_centroid_dispersion_mm,
                "mean_consensus_volume_ml": item.mean_consensus_volume_ml,
                "issues": list(item.issues),
                "notes": list(item.notes),
            }
            for item in result.decisions
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: RegistrationAnchorQaResult) -> None:
    approved = [item for item in result.decisions if item.decision == "approve"]
    review = [item for item in result.decisions if item.decision == "review"]
    rejected = [item for item in result.decisions if item.decision == "reject"]
    lines = [
        "# Registration Anchor QA",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Approved anchors: {len(approved)}",
        f"- Review anchors: {len(review)}",
        f"- Rejected anchors: {len(rejected)}",
        f"- Approved label IDs: {', '.join(str(value) for value in result.approved_anchor_labels) or 'none'}",
        "",
        "## Approved Anchors",
        "",
    ]
    if approved:
        for item in approved:
            lines.append(
                f"- `{item.label_id}` {item.label_name}: score={item.score:.1f}, "
                f"mean Dice={item.mean_dice:.3f}, max centroid dispersion={item.max_centroid_dispersion_mm:.1f} mm"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Review Anchors", ""])
    if review:
        for item in review:
            lines.append(
                f"- `{item.label_id}` {item.label_name}: score={item.score:.1f}, "
                f"mean Dice={item.mean_dice:.3f}; issues={'; '.join(item.issues)}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Rejected Anchors", ""])
    if rejected:
        for item in rejected:
            lines.append(
                f"- `{item.label_id}` {item.label_name}: score={item.score:.1f}, "
                f"mean Dice={item.mean_dice:.3f}; issues={'; '.join(item.issues)}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Ranking CSV: `{result.ranking_csv_path}`",
            f"- Decisions YAML: `{result.decisions_yaml_path}`",
            "",
            "## Interpretation",
            "",
            "- Approved anchors can drive statistical/registration-based phantom deformation.",
            "- Review anchors can be displayed and measured, but should not drive deformation without manual or case-specific QA.",
            "- Rejected anchors are too unstable in this benchmark subset for anatomy morphing decisions.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def rank_registration_anchors(
    *,
    benchmark_spec_path: str | Path = "outputs/digital/reg_training_testing_benchmark/reg_training_testing_benchmark_stage001/reg_training_testing_benchmark_stage001_benchmark_spec_v001.yaml",
    output_dir: str | Path = "outputs/digital/registration_anchor_qa",
    case_id: str = "reg_training_testing_anchor_qa_stage001",
    report_path: str | Path | None = "outputs/reports/registration_anchor_qa_stage001.md",
    thresholds: RegistrationAnchorThresholds | None = None,
) -> RegistrationAnchorQaResult:
    thresholds = thresholds or RegistrationAnchorThresholds()
    benchmark_spec = _load_yaml(benchmark_spec_path)
    label_metrics_csv = benchmark_spec.get("outputs", {}).get("label_metrics_csv")
    if not label_metrics_csv:
        raise ValueError(f"Benchmark spec does not list outputs.label_metrics_csv: {benchmark_spec_path}")
    rows = _read_label_metrics(label_metrics_csv)
    decisions = _build_decisions(rows, thresholds)
    approved = tuple(item.label_id for item in decisions if item.decision == "approve")
    review = tuple(item.label_id for item in decisions if item.decision == "review")
    rejected = tuple(item.label_id for item in decisions if item.decision == "reject")
    output = Path(output_dir) / case_id
    ranking_csv = output / f"{case_id}_registration_anchor_ranking_v001.csv"
    decisions_yaml = output / f"{case_id}_registration_anchor_decisions_v001.yaml"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_registration_anchor_qa_report_v001.md"
    result = RegistrationAnchorQaResult(
        case_id=case_id,
        benchmark_spec_path=str(benchmark_spec_path),
        output_dir=str(output),
        ranking_csv_path=str(ranking_csv),
        decisions_yaml_path=str(decisions_yaml),
        report_path=str(report),
        approved_anchor_labels=approved,
        review_anchor_labels=review,
        rejected_anchor_labels=rejected,
        decisions=decisions,
        thresholds=thresholds,
        notes=(
            "anchor_decisions_are_derived_from_registration_consistency_not_native_target_ground_truth",
            "approved_anchor_set_should_be_recomputed_when_new_registration_cases_are_added",
            "small_mobile_structures_are_expected_to_require_manual_or_case_specific_registration_QA",
        ),
    )
    _write_ranking_csv(ranking_csv, decisions)
    _write_decisions_yaml(decisions_yaml, result)
    _write_report(report, result)
    return result


def format_registration_anchor_qa_result(result: RegistrationAnchorQaResult) -> str:
    return "\n".join(
        [
            "Registration anchor QA built",
            f"Case ID: {result.case_id}",
            f"Approved/review/rejected: {len(result.approved_anchor_labels)}/{len(result.review_anchor_labels)}/{len(result.rejected_anchor_labels)}",
            f"Approved labels: {', '.join(str(value) for value in result.approved_anchor_labels) or 'none'}",
            f"Ranking CSV: {result.ranking_csv_path}",
            f"Decisions YAML: {result.decisions_yaml_path}",
            f"Report: {result.report_path}",
        ]
    )
