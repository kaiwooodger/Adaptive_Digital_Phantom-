from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import yaml


ANATOMY_GROUPS = (
    "lungs",
    "liver",
    "kidneys",
    "bone",
    "vessel_wall",
    "vascular_fluid",
)


@dataclass(frozen=True)
class VariantExperimentMetric:
    variant_id: str
    release_role: str
    warning_status: str
    mode_index: int | None
    mode_weight: float
    qa_score: float | None
    body_volume_l: float
    body_delta_l: float
    body_delta_percent: float
    waist_cm: float
    waist_delta_cm: float
    vascular_components: int
    vascular_fluid_delta_percent: float
    largest_group_delta_id: str
    largest_group_delta_percent: float
    anatomy_impact_score: float
    anatomy_impact_status: str
    rt_status: str
    flow_status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PhantomExperimentSetResult:
    case_id: str
    output_dir: str
    variant_metrics_csv_path: str
    manifest_yaml_path: str
    report_path: str
    source_approved_set_manifest: str
    rt_planning_spec_path: str | None
    dose_gamma_spec_path: str | None
    flow_model_spec_path: str | None
    variant_count: int
    high_impact_variant_count: int
    warning_variant_count: int
    rt_summary: dict[str, Any]
    flow_summary: dict[str, Any]
    variants: tuple[VariantExperimentMetric, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(path: str | Path | None, base_path: Path | None = None) -> Path | None:
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    if base_path is not None:
        return base_path.parent / candidate
    return cwd_candidate


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def _safe_percent_delta(value: float, baseline: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0 if abs(value) < 1e-9 else 100.0
    return (value - baseline) / baseline * 100.0


def _read_csv_rows(path: str | Path | None) -> list[dict[str, str]]:
    resolved = _resolve_path(path)
    if resolved is None or not resolved.exists():
        return []
    with resolved.open(newline="") as csvfile:
        return [dict(row) for row in csv.DictReader(csvfile)]


def _approved_metrics_rows(manifest: dict[str, Any], manifest_path: Path) -> dict[str, dict[str, str]]:
    outputs = manifest.get("outputs", {})
    metrics_path = None
    if isinstance(outputs, dict):
        metrics_path = outputs.get("metrics_csv")
    rows = _read_csv_rows(_resolve_path(metrics_path, manifest_path))
    return {str(row.get("variant_id", "")): row for row in rows}


def _group_volume(row: dict[str, str], group_id: str) -> float:
    return _as_float(row.get(f"group_{group_id}_volume_cm3"))


def _largest_group_delta(row: dict[str, str], baseline: dict[str, str]) -> tuple[str, float]:
    largest_group = "none"
    largest_abs_delta = 0.0
    for group_id in ANATOMY_GROUPS:
        delta = _safe_percent_delta(_group_volume(row, group_id), _group_volume(baseline, group_id))
        if abs(delta) > abs(largest_abs_delta):
            largest_abs_delta = delta
            largest_group = group_id
    return largest_group, largest_abs_delta


def _impact_status(score: float) -> str:
    if score >= 55.0:
        return "high"
    if score >= 25.0:
        return "moderate"
    if score > 0.0:
        return "low"
    return "baseline"


def _anatomy_impact_score(
    body_delta_l: float,
    waist_delta_cm: float,
    largest_group_delta_percent: float,
    warning_status: str,
    vascular_components: int,
) -> float:
    score = abs(body_delta_l) * 10.0 + abs(waist_delta_cm) * 4.0 + abs(largest_group_delta_percent) * 0.85
    if warning_status == "warning":
        score += 8.0
    if vascular_components != 1:
        score += 35.0
    return min(100.0, score)


def _rt_summary(rt_planning_spec_path: str | Path | None, dose_gamma_spec_path: str | Path | None) -> dict[str, Any]:
    if rt_planning_spec_path is None:
        return {"status": "not_supplied"}
    spec_path = _resolve_path(rt_planning_spec_path)
    if spec_path is None or not spec_path.exists():
        return {"status": "missing", "spec_path": str(rt_planning_spec_path)}
    spec = _load_yaml(spec_path)
    outputs = spec.get("outputs", {})
    dose_rows = _read_csv_rows(_resolve_path(outputs.get("dose_metrics_csv") if isinstance(outputs, dict) else None, spec_path))
    comparison_rows = _read_csv_rows(_resolve_path(outputs.get("dose_comparison_csv") if isinstance(outputs, dict) else None, spec_path))
    static_ptv = next(
        (row for row in dose_rows if row.get("mask_id") == "target_ptv_synthetic_vertebral" and row.get("state") == "static"),
        {},
    )
    vascular_static = next(
        (row for row in dose_rows if row.get("mask_id") == "vascular_fluid" and row.get("state") == "static"),
        {},
    )
    max_abs_delta_mean_gy = max((_as_float(row.get("delta_mean_gy")) for row in comparison_rows), key=abs, default=0.0)
    max_abs_delta_v95_pp = max((_as_float(row.get("delta_v95_percentage_points")) for row in comparison_rows), key=abs, default=0.0)

    gamma_status: dict[str, Any] = {"status": "not_supplied"}
    if dose_gamma_spec_path is not None:
        gamma_path = _resolve_path(dose_gamma_spec_path)
        if gamma_path is not None and gamma_path.exists():
            gamma_spec = _load_yaml(gamma_path)
            gamma_outputs = gamma_spec.get("outputs", {})
            gamma_rows = _read_csv_rows(
                _resolve_path(gamma_outputs.get("summary_csv") if isinstance(gamma_outputs, dict) else None, gamma_path)
            )
            gamma_status = {
                "status": "attached",
                "min_pass_rate_percent": min((_as_float(row.get("pass_rate_percent")) for row in gamma_rows), default=None),
                "max_gamma_value": max((_as_float(row.get("max_gamma_value")) for row in gamma_rows), default=None),
                "state_count": len(gamma_rows),
            }
        else:
            gamma_status = {"status": "missing", "spec_path": str(dose_gamma_spec_path)}

    return {
        "status": "attached",
        "spec_path": str(rt_planning_spec_path),
        "prescription_dose_gy": spec.get("dose_model", {}).get("prescription_dose_gy") if isinstance(spec.get("dose_model"), dict) else None,
        "static_ptv_mean_gy": _as_optional_float(static_ptv.get("mean_dose_gy")),
        "static_ptv_d95_gy": _as_optional_float(static_ptv.get("d95_gy")),
        "static_ptv_v95_percent": _as_optional_float(static_ptv.get("v95_percent")),
        "static_vascular_fluid_mean_gy": _as_optional_float(vascular_static.get("mean_dose_gy")),
        "max_static_vs_pulsatile_delta_mean_gy": max_abs_delta_mean_gy,
        "max_static_vs_pulsatile_delta_v95_percentage_points": max_abs_delta_v95_pp,
        "dose_metric_rows": len(dose_rows),
        "gamma": gamma_status,
        "interpretation": "baseline_reference_only_not_recomputed_for_each_pca_variant",
    }


def _flow_summary(flow_model_spec_path: str | Path | None) -> dict[str, Any]:
    if flow_model_spec_path is None:
        return {"status": "not_supplied"}
    spec_path = _resolve_path(flow_model_spec_path)
    if spec_path is None or not spec_path.exists():
        return {"status": "missing", "spec_path": str(flow_model_spec_path)}
    spec = _load_yaml(spec_path)
    summary = spec.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return {
        "status": "attached",
        "spec_path": str(flow_model_spec_path),
        "edge_count": summary.get("edge_count"),
        "node_count": summary.get("node_count"),
        "boundary_count": summary.get("boundary_count"),
        "arterial_inlet_flow_mean_ml_s": summary.get("arterial_inlet_flow_mean_ml_s"),
        "arterial_inlet_flow_min_ml_s": summary.get("arterial_inlet_flow_min_ml_s"),
        "arterial_inlet_flow_max_ml_s": summary.get("arterial_inlet_flow_max_ml_s"),
        "aorta_pressure_mean_pa": summary.get("aorta_pressure_mean_pa"),
        "max_outlet_split_range_percentage_points": summary.get("max_outlet_split_range_percentage_points"),
        "max_abs_mass_balance_residual_ml_s": summary.get("max_abs_mass_balance_residual_ml_s"),
        "max_abs_pressure_equation_residual_pa": summary.get("max_abs_pressure_equation_residual_pa"),
        "interpretation": "baseline_flow_reference_attached_not_recomputed_for_each_pca_variant",
    }


def _variant_status(impact_status: str, warning_status: str, reference_attached: bool) -> str:
    if not reference_attached:
        return "reference_missing"
    if impact_status == "high" or warning_status == "warning":
        return "needs_variant_specific_rerun"
    return "baseline_reference_compatible"


def _build_variant_metrics(
    manifest: dict[str, Any],
    metrics_rows: dict[str, dict[str, str]],
    rt_attached: bool,
    flow_attached: bool,
) -> tuple[VariantExperimentMetric, ...]:
    variants = manifest.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("Approved PCA phantom-set manifest must contain variants")
    baseline_item = next((item for item in variants if isinstance(item, dict) and item.get("release_role") == "baseline"), variants[0])
    if not isinstance(baseline_item, dict):
        raise ValueError("Baseline variant is malformed")
    baseline_row = metrics_rows.get(str(baseline_item.get("variant_id")), {})
    baseline_body = _as_float(baseline_item.get("body_volume_cm3") or baseline_row.get("body_volume_cm3"))
    baseline_waist = _as_float(baseline_item.get("waist_cm") or baseline_row.get("waist_cm"))

    results: list[VariantExperimentMetric] = []
    for item in variants:
        if not isinstance(item, dict):
            continue
        variant_id = str(item.get("variant_id", ""))
        row = metrics_rows.get(variant_id, {})
        body_cm3 = _as_float(item.get("body_volume_cm3") or row.get("body_volume_cm3"))
        waist_cm = _as_float(item.get("waist_cm") or row.get("waist_cm"))
        body_delta_l = (body_cm3 - baseline_body) / 1000.0
        body_delta_percent = _safe_percent_delta(body_cm3, baseline_body)
        waist_delta_cm = waist_cm - baseline_waist
        largest_group, largest_group_delta = _largest_group_delta(row, baseline_row) if row and baseline_row else ("none", 0.0)
        vascular_delta = _safe_percent_delta(_group_volume(row, "vascular_fluid"), _group_volume(baseline_row, "vascular_fluid")) if row and baseline_row else 0.0
        warning_status = str(item.get("warning_status", "clean"))
        vascular_components = _as_int(item.get("vascular_components") or row.get("vascular_components"), default=0)
        impact_score = _anatomy_impact_score(
            body_delta_l=body_delta_l,
            waist_delta_cm=waist_delta_cm,
            largest_group_delta_percent=largest_group_delta,
            warning_status=warning_status,
            vascular_components=vascular_components,
        )
        impact_status = _impact_status(impact_score)
        notes: list[str] = []
        notes.extend(str(note) for note in item.get("qa_notes", []))
        notes.extend(str(issue) for issue in item.get("qa_issues", []))
        if variant_id == str(baseline_item.get("variant_id")):
            notes.append("baseline_variant")
        if vascular_components != 1:
            notes.append("vascular_component_count_not_equal_to_one")
        results.append(
            VariantExperimentMetric(
                variant_id=variant_id,
                release_role=str(item.get("release_role", "")),
                warning_status=warning_status,
                mode_index=None if item.get("mode_index") is None else _as_int(item.get("mode_index")),
                mode_weight=_as_float(item.get("mode_weight")),
                qa_score=_as_optional_float(item.get("qa_score")),
                body_volume_l=body_cm3 / 1000.0,
                body_delta_l=body_delta_l,
                body_delta_percent=body_delta_percent,
                waist_cm=waist_cm,
                waist_delta_cm=waist_delta_cm,
                vascular_components=vascular_components,
                vascular_fluid_delta_percent=vascular_delta,
                largest_group_delta_id=largest_group,
                largest_group_delta_percent=largest_group_delta,
                anatomy_impact_score=impact_score,
                anatomy_impact_status=impact_status,
                rt_status=_variant_status(impact_status, warning_status, rt_attached),
                flow_status=_variant_status(impact_status, warning_status, flow_attached),
                notes=tuple(notes),
            )
        )
    return tuple(sorted(results, key=lambda entry: (entry.release_role != "baseline", -entry.anatomy_impact_score, entry.variant_id)))


def _write_variant_csv(path: Path, variants: tuple[VariantExperimentMetric, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "variant_id",
                "release_role",
                "warning_status",
                "mode_index",
                "mode_weight",
                "qa_score",
                "body_volume_l",
                "body_delta_l",
                "body_delta_percent",
                "waist_cm",
                "waist_delta_cm",
                "vascular_components",
                "vascular_fluid_delta_percent",
                "largest_group_delta_id",
                "largest_group_delta_percent",
                "anatomy_impact_score",
                "anatomy_impact_status",
                "rt_status",
                "flow_status",
                "notes",
            ]
        )
        for item in variants:
            writer.writerow(
                [
                    item.variant_id,
                    item.release_role,
                    item.warning_status,
                    "" if item.mode_index is None else item.mode_index,
                    f"{item.mode_weight:.6f}",
                    "" if item.qa_score is None else f"{item.qa_score:.3f}",
                    f"{item.body_volume_l:.6f}",
                    f"{item.body_delta_l:.6f}",
                    f"{item.body_delta_percent:.6f}",
                    f"{item.waist_cm:.6f}",
                    f"{item.waist_delta_cm:.6f}",
                    item.vascular_components,
                    f"{item.vascular_fluid_delta_percent:.6f}",
                    item.largest_group_delta_id,
                    f"{item.largest_group_delta_percent:.6f}",
                    f"{item.anatomy_impact_score:.6f}",
                    item.anatomy_impact_status,
                    item.rt_status,
                    item.flow_status,
                    ";".join(item.notes),
                ]
            )


def _write_manifest(path: Path, result: PhantomExperimentSetResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "approved_pca_phantom_experiment_set",
        "source_approved_set_manifest": result.source_approved_set_manifest,
        "rt_planning_spec": result.rt_planning_spec_path,
        "dose_gamma_spec": result.dose_gamma_spec_path,
        "flow_model_spec": result.flow_model_spec_path,
        "outputs": {
            "variant_metrics_csv": result.variant_metrics_csv_path,
            "manifest_yaml": result.manifest_yaml_path,
            "report": result.report_path,
        },
        "summary": {
            "variant_count": result.variant_count,
            "high_impact_variant_count": result.high_impact_variant_count,
            "warning_variant_count": result.warning_variant_count,
        },
        "rt_summary": result.rt_summary,
        "flow_summary": result.flow_summary,
        "variants": [
            {
                "variant_id": item.variant_id,
                "release_role": item.release_role,
                "warning_status": item.warning_status,
                "mode_index": item.mode_index,
                "mode_weight": item.mode_weight,
                "qa_score": item.qa_score,
                "body_delta_l": item.body_delta_l,
                "waist_delta_cm": item.waist_delta_cm,
                "vascular_fluid_delta_percent": item.vascular_fluid_delta_percent,
                "largest_group_delta_id": item.largest_group_delta_id,
                "largest_group_delta_percent": item.largest_group_delta_percent,
                "anatomy_impact_score": item.anatomy_impact_score,
                "anatomy_impact_status": item.anatomy_impact_status,
                "rt_status": item.rt_status,
                "flow_status": item.flow_status,
                "notes": list(item.notes),
            }
            for item in result.variants
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_float(value: Any, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _format_report(result: PhantomExperimentSetResult) -> str:
    high_variants = [item for item in result.variants if item.anatomy_impact_status == "high"]
    rerun_variants = [item for item in result.variants if item.rt_status == "needs_variant_specific_rerun" or item.flow_status == "needs_variant_specific_rerun"]
    lines = [
        "# Approved PCA Phantom Experiment Set Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Variants evaluated: {result.variant_count}",
        f"- High-impact anatomy variants: {result.high_impact_variant_count}",
        f"- QA-warning variants: {result.warning_variant_count}",
        f"- Variants recommended for RT/flow rerun: {len(rerun_variants)}",
        "- Execution policy: disk-light metadata comparison only; no new dose/NIfTI volumes generated",
        "",
        "## Variant Ranking",
        "",
        "| variant | role | impact | score | body delta L | waist delta cm | largest anatomy delta | RT status | flow status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in result.variants:
        lines.append(
            f"| {item.variant_id} | {item.release_role} | {item.anatomy_impact_status} | "
            f"{item.anatomy_impact_score:.1f} | {item.body_delta_l:+.2f} | {item.waist_delta_cm:+.2f} | "
            f"{item.largest_group_delta_id} {item.largest_group_delta_percent:+.1f}% | "
            f"{item.rt_status} | {item.flow_status} |"
        )
    lines.extend(["", "## RT Reference", ""])
    if result.rt_summary.get("status") == "attached":
        gamma = result.rt_summary.get("gamma", {})
        lines.append(f"- Static PTV mean dose: {_format_float(result.rt_summary.get('static_ptv_mean_gy'))} Gy")
        lines.append(f"- Static PTV D95: {_format_float(result.rt_summary.get('static_ptv_d95_gy'))} Gy")
        lines.append(f"- Max static-vs-pulsatile mean dose delta: {_format_float(result.rt_summary.get('max_static_vs_pulsatile_delta_mean_gy'), 4)} Gy")
        lines.append(f"- Gamma min pass rate: {_format_float(gamma.get('min_pass_rate_percent') if isinstance(gamma, dict) else None, 2)}%")
    else:
        lines.append(f"- RT reference status: {result.rt_summary.get('status')}")
    lines.extend(["", "## Flow Reference", ""])
    if result.flow_summary.get("status") == "attached":
        lines.append(f"- Mean arterial inlet flow: {_format_float(result.flow_summary.get('arterial_inlet_flow_mean_ml_s'))} mL/s")
        lines.append(f"- Aorta mean pressure: {_format_float(result.flow_summary.get('aorta_pressure_mean_pa'))} Pa")
        lines.append(f"- Max mass-balance residual: {_format_float(result.flow_summary.get('max_abs_mass_balance_residual_ml_s'), 9)} mL/s")
        lines.append(f"- Boundary count: {result.flow_summary.get('boundary_count')}")
    else:
        lines.append(f"- Flow reference status: {result.flow_summary.get('status')}")
    lines.extend(["", "## Recommended Next Reruns", ""])
    if rerun_variants:
        for item in rerun_variants:
            lines.append(f"- {item.variant_id}: {item.anatomy_impact_status} anatomy impact or QA warning; rerun RT dose and flow boundary checks on this anatomy.")
    else:
        lines.append("- No variant-specific RT/flow reruns are recommended by the current surrogate thresholds.")
    lines.extend(["", "## Caveats"])
    for note in result.notes:
        lines.append(f"- {note}")
    if high_variants:
        lines.append("- High-impact ranking is a triage signal, not a clinical endpoint.")
    return "\n".join(lines)


def run_phantom_experiment_set(
    approved_set_manifest_path: str | Path,
    rt_planning_spec_path: str | Path | None = None,
    dose_gamma_spec_path: str | Path | None = None,
    flow_model_spec_path: str | Path | None = None,
    output_dir: str | Path = "outputs/experiments/approved_pca_phantom_set",
    case_id: str | None = None,
    report_path: str | Path | None = "outputs/reports/approved_pca_phantom_experiment_set_stage001.md",
) -> PhantomExperimentSetResult:
    manifest_path = Path(approved_set_manifest_path)
    manifest = _load_yaml(manifest_path)
    resolved_case_id = case_id or str(manifest.get("case_id") or "approved_pca_phantom_experiment_set")
    metrics_rows = _approved_metrics_rows(manifest, manifest_path)
    rt = _rt_summary(rt_planning_spec_path, dose_gamma_spec_path)
    flow = _flow_summary(flow_model_spec_path)
    variants = _build_variant_metrics(
        manifest=manifest,
        metrics_rows=metrics_rows,
        rt_attached=rt.get("status") == "attached",
        flow_attached=flow.get("status") == "attached",
    )
    output = Path(output_dir)
    variant_csv = output / f"{resolved_case_id}_phantom_experiment_variant_metrics_v001.csv"
    manifest_yaml = output / f"{resolved_case_id}_phantom_experiment_manifest_v001.yaml"
    report_out = Path(report_path) if report_path is not None else output / f"{resolved_case_id}_phantom_experiment_report_v001.md"
    notes = (
        "variant_rows_compare_approved_pca_anatomies_against_the_mean_baseline",
        "rt_and_flow_values_are_attached_baseline_references_not_recomputed_per_variant",
        "needs_variant_specific_rerun_marks_variants_with_high_anatomy_impact_or_carried_QA_warnings",
        "this_stage_is_designed_to_select_next_heavy_simulations_without_creating_large_files",
    )
    result = PhantomExperimentSetResult(
        case_id=resolved_case_id,
        output_dir=str(output),
        variant_metrics_csv_path=str(variant_csv),
        manifest_yaml_path=str(manifest_yaml),
        report_path=str(report_out),
        source_approved_set_manifest=str(approved_set_manifest_path),
        rt_planning_spec_path=None if rt_planning_spec_path is None else str(rt_planning_spec_path),
        dose_gamma_spec_path=None if dose_gamma_spec_path is None else str(dose_gamma_spec_path),
        flow_model_spec_path=None if flow_model_spec_path is None else str(flow_model_spec_path),
        variant_count=len(variants),
        high_impact_variant_count=sum(1 for item in variants if item.anatomy_impact_status == "high"),
        warning_variant_count=sum(1 for item in variants if item.warning_status == "warning"),
        rt_summary=rt,
        flow_summary=flow,
        variants=variants,
        notes=notes,
    )
    _write_variant_csv(variant_csv, variants)
    _write_manifest(manifest_yaml, result)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(_format_report(result))
    return result


def format_phantom_experiment_set_result(result: PhantomExperimentSetResult) -> str:
    lines = [
        "# Approved PCA Phantom Experiment Set",
        "",
        f"Case ID: `{result.case_id}`",
        f"Variants evaluated: {result.variant_count}",
        f"High-impact variants: {result.high_impact_variant_count}",
        f"QA-warning variants: {result.warning_variant_count}",
        "",
        "## Outputs",
        "",
        f"- Variant metrics CSV: `{result.variant_metrics_csv_path}`",
        f"- Manifest YAML: `{result.manifest_yaml_path}`",
        f"- Report: `{result.report_path}`",
    ]
    return "\n".join(lines)
