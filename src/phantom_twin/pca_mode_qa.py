from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import yaml


QA_GROUP_ORDER = (
    "lungs",
    "liver",
    "kidneys",
    "bladder",
    "bone",
    "vessel_wall",
    "vascular_fluid",
)


@dataclass(frozen=True)
class PcaModeQaThresholds:
    max_waist_delta_cm: float = 2.0
    max_body_delta_l: float = 1.25
    max_group_delta_percent: float = 35.0
    max_vascular_volume_delta_percent: float = 10.0
    expected_vascular_components: int = 1
    min_score_for_approval: float = 70.0


@dataclass(frozen=True)
class PcaVariantQaMetric:
    variant_id: str
    label: str
    mode_index: int | None
    mode_weight: float
    body_volume_cm3: float
    waist_cm: float
    bbox_mm: tuple[float, float, float]
    vascular_components: int
    group_volumes_cm3: dict[str, float]
    material_labels_path: str
    preview_png_path: str


@dataclass(frozen=True)
class PcaModeQaDecision:
    rank: int
    mode_index: int
    decision: str
    score: float
    interpretation: str
    variant_ids: tuple[str, ...]
    waist_max_delta_cm: float
    body_max_delta_l: float
    largest_group_delta_id: str
    largest_group_delta_percent: float
    vascular_components_ok: bool
    issues: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PcaModeQaResult:
    case_id: str
    output_dir: str
    metrics_csv_path: str
    atlas_spec_path: str
    ranking_csv_path: str
    decisions_yaml_path: str
    report_path: str
    approved_modes: tuple[int, ...]
    rejected_modes: tuple[int, ...]
    decisions: tuple[PcaModeQaDecision, ...]
    thresholds: PcaModeQaThresholds
    notes: tuple[str, ...]


def _as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def _group_from_column(column: str) -> str | None:
    if column.startswith("group_") and column.endswith("_volume_cm3"):
        return column.removeprefix("group_").removesuffix("_volume_cm3")
    return None


def _read_metrics_csv(path: str | Path) -> tuple[PcaVariantQaMetric, ...]:
    metrics_path = Path(path)
    with metrics_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = []
        for row in reader:
            group_volumes = {
                group_id: _as_float(value)
                for column, value in row.items()
                if (group_id := _group_from_column(column)) is not None
            }
            rows.append(
                PcaVariantQaMetric(
                    variant_id=str(row.get("variant_id", "")).strip(),
                    label=str(row.get("label", "")).strip(),
                    mode_index=None if not row.get("mode_index") else _as_int(row.get("mode_index")),
                    mode_weight=_as_float(row.get("mode_weight")),
                    body_volume_cm3=_as_float(row.get("body_volume_cm3")),
                    waist_cm=_as_float(row.get("waist_cm")),
                    bbox_mm=(
                        _as_float(row.get("bbox_x_mm")),
                        _as_float(row.get("bbox_y_mm")),
                        _as_float(row.get("bbox_z_mm")),
                    ),
                    vascular_components=_as_int(row.get("vascular_components")),
                    group_volumes_cm3=group_volumes,
                    material_labels_path=str(row.get("material_labels_path", "")).strip(),
                    preview_png_path=str(row.get("preview_png_path", "")).strip(),
                )
            )
    if not rows:
        raise ValueError(f"No PCA variant metrics found in {metrics_path}")
    return tuple(rows)


def _load_atlas_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    data = yaml.safe_load(spec_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Atlas spec is not a mapping: {spec_path}")
    return data


def _safe_percent_delta(value: float, reference: float) -> float:
    if math.isclose(reference, 0.0, abs_tol=1e-9):
        return 0.0 if math.isclose(value, 0.0, abs_tol=1e-9) else 100.0
    return abs(value - reference) / abs(reference) * 100.0


def _signed_delta(value: float, reference: float) -> float:
    return value - reference


def _penalty_over(value: float, limit: float, scale: float, cap: float) -> float:
    if limit <= 0.0 or value <= limit:
        return 0.0
    return min(cap, (value - limit) / limit * scale)


def _variant_pairs(metrics: tuple[PcaVariantQaMetric, ...]) -> dict[int, dict[str, PcaVariantQaMetric]]:
    pairs: dict[int, dict[str, PcaVariantQaMetric]] = {}
    for item in metrics:
        if item.mode_index is None:
            continue
        sign = "positive" if item.mode_weight > 0.0 else "negative" if item.mode_weight < 0.0 else "zero"
        pairs.setdefault(item.mode_index, {})[sign] = item
    return pairs


def _largest_group_delta(
    mean: PcaVariantQaMetric,
    variants: tuple[PcaVariantQaMetric, ...],
) -> tuple[str, float]:
    largest_group = "none"
    largest_percent = 0.0
    group_ids = [
        group_id
        for group_id in QA_GROUP_ORDER
        if group_id in mean.group_volumes_cm3 and group_id != "vascular_fluid"
    ]
    for group_id in group_ids:
        reference = mean.group_volumes_cm3.get(group_id, 0.0)
        for variant in variants:
            percent = _safe_percent_delta(variant.group_volumes_cm3.get(group_id, 0.0), reference)
            if percent > largest_percent:
                largest_percent = percent
                largest_group = group_id
    return largest_group, largest_percent


def _interpret_mode(
    mean: PcaVariantQaMetric,
    negative: PcaVariantQaMetric | None,
    positive: PcaVariantQaMetric | None,
    largest_group: str,
    body_max_delta_l: float,
) -> str:
    if negative is None or positive is None:
        return "incomplete mode pair"
    lung_delta = _signed_delta(positive.group_volumes_cm3.get("lungs", 0.0), mean.group_volumes_cm3.get("lungs", 0.0))
    liver_delta = _signed_delta(positive.group_volumes_cm3.get("liver", 0.0), mean.group_volumes_cm3.get("liver", 0.0))
    if abs(lung_delta) > 50.0 and abs(liver_delta) > 50.0 and lung_delta * liver_delta < 0.0:
        return "lung-liver tradeoff mode"
    if body_max_delta_l > 0.5:
        return "body-volume/depth mode"
    if largest_group != "none":
        return f"{largest_group.replace('_', ' ')} dominant mode"
    return "low-amplitude anatomical mode"


def _score_mode(
    mean: PcaVariantQaMetric,
    mode_index: int,
    paired: dict[str, PcaVariantQaMetric],
    thresholds: PcaModeQaThresholds,
) -> tuple[float, bool, tuple[str, ...], tuple[str, ...], dict[str, float | str | bool | tuple[str, ...]]]:
    negative = paired.get("negative")
    positive = paired.get("positive")
    variants = tuple(item for item in (negative, positive) if item is not None)
    issues: list[str] = []
    notes: list[str] = []
    hard_fail = False
    score = 100.0

    if negative is None or positive is None:
        hard_fail = True
        score -= 45.0
        issues.append("missing_plus_minus_variant_pair")

    vascular_components_ok = all(
        item.vascular_components == thresholds.expected_vascular_components for item in variants
    )
    if not vascular_components_ok:
        hard_fail = True
        score -= 35.0
        issues.append(
            "vascular_component_count_mismatch:"
            + ",".join(f"{item.variant_id}={item.vascular_components}" for item in variants)
        )

    waist_max_delta_cm = max((abs(item.waist_cm - mean.waist_cm) for item in variants), default=0.0)
    body_max_delta_l = max((abs(item.body_volume_cm3 - mean.body_volume_cm3) / 1000.0 for item in variants), default=0.0)
    largest_group, largest_group_delta_percent = _largest_group_delta(mean, variants)
    vascular_volume_delta_percent = max(
        (
            _safe_percent_delta(
                item.group_volumes_cm3.get("vascular_fluid", 0.0),
                mean.group_volumes_cm3.get("vascular_fluid", 0.0),
            )
            for item in variants
        ),
        default=0.0,
    )

    waist_penalty = _penalty_over(waist_max_delta_cm, thresholds.max_waist_delta_cm, scale=10.0, cap=15.0)
    body_penalty = _penalty_over(body_max_delta_l, thresholds.max_body_delta_l, scale=14.0, cap=20.0)
    group_penalty = _penalty_over(
        largest_group_delta_percent,
        thresholds.max_group_delta_percent,
        scale=14.0,
        cap=25.0,
    )
    vascular_volume_penalty = _penalty_over(
        vascular_volume_delta_percent,
        thresholds.max_vascular_volume_delta_percent,
        scale=12.0,
        cap=15.0,
    )
    score -= waist_penalty + body_penalty + group_penalty + vascular_volume_penalty

    if waist_penalty > 0.0:
        notes.append(f"waist_delta_exceeds_soft_limit:{waist_max_delta_cm:.2f}cm")
    if body_penalty > 0.0:
        notes.append(f"body_volume_delta_exceeds_soft_limit:{body_max_delta_l:.2f}L")
    if group_penalty > 0.0:
        notes.append(f"{largest_group}_delta_exceeds_soft_limit:{largest_group_delta_percent:.1f}%")
    if vascular_volume_penalty > 0.0:
        notes.append(f"vascular_fluid_delta_exceeds_soft_limit:{vascular_volume_delta_percent:.1f}%")
    if not notes and not issues:
        notes.append("mode_within_current_stage001_guardrails")

    score = max(0.0, min(100.0, score))
    decision = "approved" if not hard_fail and score >= thresholds.min_score_for_approval else "rejected"
    if decision == "rejected" and score < thresholds.min_score_for_approval:
        issues.append(f"score_below_approval_threshold:{score:.1f}<{thresholds.min_score_for_approval:.1f}")

    payload: dict[str, float | str | bool | tuple[str, ...]] = {
        "decision": decision,
        "waist_max_delta_cm": waist_max_delta_cm,
        "body_max_delta_l": body_max_delta_l,
        "largest_group_delta_id": largest_group,
        "largest_group_delta_percent": largest_group_delta_percent,
        "vascular_components_ok": vascular_components_ok,
        "variant_ids": tuple(item.variant_id for item in variants),
        "interpretation": _interpret_mode(mean, negative, positive, largest_group, body_max_delta_l),
    }
    return score, hard_fail, tuple(issues), tuple(notes), payload


def _write_ranking_csv(path: Path, decisions: tuple[PcaModeQaDecision, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "rank",
                "mode_index",
                "decision",
                "score",
                "interpretation",
                "waist_max_delta_cm",
                "body_max_delta_l",
                "largest_group_delta_id",
                "largest_group_delta_percent",
                "vascular_components_ok",
                "variant_ids",
                "issues",
                "notes",
            ]
        )
        for item in decisions:
            writer.writerow(
                [
                    item.rank,
                    item.mode_index,
                    item.decision,
                    f"{item.score:.3f}",
                    item.interpretation,
                    f"{item.waist_max_delta_cm:.3f}",
                    f"{item.body_max_delta_l:.6f}",
                    item.largest_group_delta_id,
                    f"{item.largest_group_delta_percent:.3f}",
                    "true" if item.vascular_components_ok else "false",
                    " ".join(item.variant_ids),
                    ";".join(item.issues),
                    ";".join(item.notes),
                ]
            )


def _write_decisions_yaml(path: Path, result: PcaModeQaResult, atlas_spec: dict[str, Any]) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "pca_mode_qa_ranking",
        "source_metrics_csv": result.metrics_csv_path,
        "source_atlas_spec": result.atlas_spec_path,
        "source_atlas_case_id": atlas_spec.get("case_id"),
        "thresholds": result.thresholds.__dict__,
        "outputs": {
            "ranking_csv": result.ranking_csv_path,
            "decisions_yaml": result.decisions_yaml_path,
            "report": result.report_path,
        },
        "approved_mode_indices": list(result.approved_modes),
        "rejected_mode_indices": list(result.rejected_modes),
        "decisions": [
            {
                "rank": item.rank,
                "mode_index": item.mode_index,
                "decision": item.decision,
                "score": item.score,
                "interpretation": item.interpretation,
                "variant_ids": list(item.variant_ids),
                "waist_max_delta_cm": item.waist_max_delta_cm,
                "body_max_delta_l": item.body_max_delta_l,
                "largest_group_delta_id": item.largest_group_delta_id,
                "largest_group_delta_percent": item.largest_group_delta_percent,
                "vascular_components_ok": item.vascular_components_ok,
                "issues": list(item.issues),
                "notes": list(item.notes),
            }
            for item in result.decisions
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: PcaModeQaResult) -> str:
    thresholds = result.thresholds
    lines = [
        "# PCA Mode QA / Ranking Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Approved modes: {len(result.approved_modes)} ({', '.join(str(item) for item in result.approved_modes) or 'none'})",
        f"- Rejected modes: {len(result.rejected_modes)} ({', '.join(str(item) for item in result.rejected_modes) or 'none'})",
        f"- Ranking CSV: `{Path(result.ranking_csv_path).name}`",
        f"- Decisions YAML: `{Path(result.decisions_yaml_path).name}`",
        "",
        "## Guardrails",
        "",
        f"- Max waist delta: {thresholds.max_waist_delta_cm:.2f} cm",
        f"- Max body volume delta: {thresholds.max_body_delta_l:.2f} L",
        f"- Max organ/group delta: {thresholds.max_group_delta_percent:.1f}%",
        f"- Max vascular fluid delta: {thresholds.max_vascular_volume_delta_percent:.1f}%",
        f"- Expected vascular components: {thresholds.expected_vascular_components}",
        f"- Minimum approval score: {thresholds.min_score_for_approval:.1f}",
        "",
        "## Ranked Modes",
        "",
        "| rank | mode | decision | score | interpretation | waist delta cm | body delta L | largest group delta | vascular OK |",
        "| ---: | ---: | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for item in result.decisions:
        vascular_ok = "yes" if item.vascular_components_ok else "no"
        lines.append(
            f"| {item.rank} | {item.mode_index} | {item.decision} | {item.score:.1f} | "
            f"{item.interpretation} | {item.waist_max_delta_cm:.2f} | {item.body_max_delta_l:.2f} | "
            f"{item.largest_group_delta_id} {item.largest_group_delta_percent:.1f}% | {vascular_ok} |"
        )
    lines.extend(["", "## Mode Notes", ""])
    for item in result.decisions:
        note_text = "; ".join((*item.notes, *item.issues)) or "no notes"
        lines.append(f"- Mode {item.mode_index}: {note_text}")
    lines.extend(["", "## Caveats"])
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def rank_pca_modes(
    metrics_csv_path: str | Path,
    atlas_spec_path: str | Path,
    output_dir: str | Path = "outputs/digital/pca_mode_qa",
    case_id: str | None = None,
    max_waist_delta_cm: float = 2.0,
    max_body_delta_l: float = 1.25,
    max_group_delta_percent: float = 35.0,
    max_vascular_volume_delta_percent: float = 10.0,
    expected_vascular_components: int = 1,
    min_score_for_approval: float = 70.0,
    report_path: str | Path | None = "outputs/reports/pca_mode_qa_stage001.md",
) -> PcaModeQaResult:
    metrics = _read_metrics_csv(metrics_csv_path)
    atlas_spec = _load_atlas_spec(atlas_spec_path)
    mean = next((item for item in metrics if item.mode_index is None or item.variant_id == "mean"), None)
    if mean is None:
        raise ValueError("PCA metrics CSV must include a mean row")

    resolved_case_id = case_id or str(atlas_spec.get("case_id") or "pca_modes")
    thresholds = PcaModeQaThresholds(
        max_waist_delta_cm=max_waist_delta_cm,
        max_body_delta_l=max_body_delta_l,
        max_group_delta_percent=max_group_delta_percent,
        max_vascular_volume_delta_percent=max_vascular_volume_delta_percent,
        expected_vascular_components=expected_vascular_components,
        min_score_for_approval=min_score_for_approval,
    )
    raw_decisions: list[PcaModeQaDecision] = []
    for mode_index, paired in sorted(_variant_pairs(metrics).items()):
        score, _, issues, notes, payload = _score_mode(mean, mode_index, paired, thresholds)
        raw_decisions.append(
            PcaModeQaDecision(
                rank=0,
                mode_index=mode_index,
                decision=str(payload["decision"]),
                score=score,
                interpretation=str(payload["interpretation"]),
                variant_ids=tuple(payload["variant_ids"]),  # type: ignore[arg-type]
                waist_max_delta_cm=float(payload["waist_max_delta_cm"]),
                body_max_delta_l=float(payload["body_max_delta_l"]),
                largest_group_delta_id=str(payload["largest_group_delta_id"]),
                largest_group_delta_percent=float(payload["largest_group_delta_percent"]),
                vascular_components_ok=bool(payload["vascular_components_ok"]),
                issues=issues,
                notes=notes,
            )
        )
    ranked = tuple(
        PcaModeQaDecision(
            rank=index,
            mode_index=item.mode_index,
            decision=item.decision,
            score=item.score,
            interpretation=item.interpretation,
            variant_ids=item.variant_ids,
            waist_max_delta_cm=item.waist_max_delta_cm,
            body_max_delta_l=item.body_max_delta_l,
            largest_group_delta_id=item.largest_group_delta_id,
            largest_group_delta_percent=item.largest_group_delta_percent,
            vascular_components_ok=item.vascular_components_ok,
            issues=item.issues,
            notes=item.notes,
        )
        for index, item in enumerate(
            sorted(raw_decisions, key=lambda entry: (entry.decision != "approved", -entry.score, entry.mode_index)),
            start=1,
        )
    )

    output = Path(output_dir)
    ranking_csv = output / f"{resolved_case_id}_pca_mode_qa_ranking_v001.csv"
    decisions_yaml = output / f"{resolved_case_id}_pca_mode_qa_decisions_v001.yaml"
    report_out = Path(report_path) if report_path is not None else output / f"{resolved_case_id}_pca_mode_qa_report_v001.md"
    approved = tuple(item.mode_index for item in ranked if item.decision == "approved")
    rejected = tuple(item.mode_index for item in ranked if item.decision == "rejected")
    notes = (
        "approval_means_suitable_for_current_stage001_digital_phantom_experiments_not_clinical_anatomical_validation",
        "hard_failures_are_missing_mode_pairs_or_disconnected_unexpected_vascular_component_counts",
        "soft_scores_rank_morphological_stability_relative_to_the_mean_pca_variant",
    )
    result = PcaModeQaResult(
        case_id=resolved_case_id,
        output_dir=str(output),
        metrics_csv_path=str(metrics_csv_path),
        atlas_spec_path=str(atlas_spec_path),
        ranking_csv_path=str(ranking_csv),
        decisions_yaml_path=str(decisions_yaml),
        report_path=str(report_out),
        approved_modes=approved,
        rejected_modes=rejected,
        decisions=ranked,
        thresholds=thresholds,
        notes=notes,
    )
    _write_ranking_csv(ranking_csv, ranked)
    _write_decisions_yaml(decisions_yaml, result, atlas_spec)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(_format_report(result))
    return result


def format_pca_mode_qa_result(result: PcaModeQaResult) -> str:
    approved = ", ".join(str(item) for item in result.approved_modes) or "none"
    rejected = ", ".join(str(item) for item in result.rejected_modes) or "none"
    lines = [
        "# PCA Mode QA / Ranking",
        "",
        f"Case ID: `{result.case_id}`",
        f"Approved modes: {approved}",
        f"Rejected modes: {rejected}",
        "",
        "## Outputs",
        "",
        f"- Ranking CSV: `{result.ranking_csv_path}`",
        f"- Decisions YAML: `{result.decisions_yaml_path}`",
        f"- Report: `{result.report_path}`",
    ]
    return "\n".join(lines)
