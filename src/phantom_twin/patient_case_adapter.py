from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import math

import numpy as np
import yaml

from .deformation_match_experiment import (
    ProfileMetric,
    ScanMetric,
    _bbox_mm,
    _estimate_waist_cm,
    _match,
    _read_profile_specs,
    _score_components,
    _spacing_from_affine,
    _status as _match_status,
    _synthetic_height_profiles,
)
from .product_case_runner import ProductCaseResult, build_product_case
from .stage007_baseline import resolve_stage007_active_baseline


DEFAULT_BASELINE_GRAPH_CANDIDATES = (
    "outputs/digital/vessel_radius_tuned_stage004_learned_aorta_safe/"
    "btcv_abdomen_case0001_stage004_learned_aorta_radius_tuned_safe_radius_tuned_vascular_graph_v001.yaml",
    "outputs/digital/vessel_radius_tuned_stage004_learned_aorta/"
    "btcv_abdomen_case0001_stage004_learned_aorta_radius_tuned_radius_tuned_vascular_graph_v001.yaml",
    "outputs/digital/vascular_network_learned_aorta_stage004/"
    "btcv_abdomen_case0001_population_learned_aorta_stage004_learned_aorta_vascular_graph_v001.yaml",
    "outputs/digital/vascular_network/ct_org_case0_imagetbad_case125_vascular_network_graph_v001.yaml",
)
DEFAULT_BASELINE_COMBINED_SPEC_CANDIDATES = (
    "outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml",
)


@dataclass(frozen=True)
class PatientCaseMetric:
    scan_id: str
    source_path: str
    scope: str
    body_volume_l: float | None
    waist_cm: float | None
    z_extent_cm: float | None
    liver_volume_ml: float | None
    kidney_volume_ml: float | None
    aorta_volume_ml: float | None
    ivc_volume_ml: float | None
    organ_label_mode: str
    vessel_volume_ml: float | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PatientProfileMatch:
    profile_id: str
    variant_id: str
    profile_source: str
    source_path: str
    match_score: float
    match_status: str
    scan_geometry_score: float
    profile_input_score: float | None
    anthropometric_score: float | None
    anchor_score: float | None
    waist_delta_cm: float | None
    body_delta_l: float | None
    z_extent_delta_cm: float | None
    liver_delta_percent: float | None
    kidney_delta_percent: float | None
    vascular_proxy_delta_percent: float | None
    compared_features: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PatientCaseAdapterResult:
    case_id: str
    patient_id: str
    output_dir: str
    final_status: str
    selected_profile_id: str
    selected_variant_id: str
    selected_match_score: float
    baseline_graph_path: str | None
    baseline_combined_spec_path: str | None
    patient_metric_yaml_path: str
    match_scores_csv_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    product_manifest_path: str | None
    adapter_manifest_path: str | None
    build_manifest_path: str | None
    qa_yaml_path: str | None
    render_preview_png_path: str | None
    profile_count: int
    product_status: str | None
    stage_statuses: tuple[str, ...]
    notes: tuple[str, ...]


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Patient case adapter requires nibabel for CT/segmentation metrics.") from exc
    return nib


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Patient case adapter preview generation requires matplotlib.") from exc
    return plt


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "patient_case"


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _volume_ml(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    return float(np.count_nonzero(mask) * np.prod(np.asarray(spacing_mm, dtype=float)) / 1000.0)


def _label_volume_ml(labels: np.ndarray, label_ids: tuple[int, ...], spacing_mm: tuple[float, float, float]) -> float:
    if not label_ids:
        return 0.0
    return _volume_ml(np.isin(labels, label_ids), spacing_mm)


def _detect_label_mode(labels: np.ndarray) -> str:
    unique = {int(value) for value in np.unique(labels)[:256].tolist()}
    nonzero = unique - {0}
    if {2, 3, 6}.issubset(nonzero) and ({8, 9, 10} & nonzero):
        return "btcv"
    if nonzero & {7, 8, 10, 11, 13, 14, 15}:
        return "material"
    return "ct-org"


def _label_map_for_mode(mode: str) -> dict[str, tuple[int, ...]]:
    if mode == "btcv":
        return {"liver": (6,), "kidneys": (2, 3), "aorta": (8,), "ivc": (9, 10)}
    if mode == "material":
        return {"liver": (6,), "kidneys": (7,), "aorta": (13, 14), "ivc": (13, 14)}
    return {"liver": (1,), "kidneys": (4,), "aorta": (), "ivc": ()}


def _ct_body_mask(ct_data: np.ndarray) -> np.ndarray:
    finite = np.asarray(ct_data)[np.isfinite(ct_data)]
    if finite.size == 0:
        return np.zeros(ct_data.shape, dtype=bool)
    # CT convention: outside air is near -1000 HU. A -700 HU threshold keeps lung/body envelope in torso CTs.
    return np.asarray(ct_data > -700.0, dtype=bool)


def _resolve_existing(path: str | Path | None) -> str | None:
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    return str(candidate) if candidate.exists() else None


def _first_existing(paths: tuple[str, ...]) -> str | None:
    for item in paths:
        resolved = _resolve_existing(item)
        if resolved is not None:
            return resolved
    return None


def _read_profile_metrics_csv(path: str | Path) -> list[ProfileMetric]:
    if not Path(path).exists():
        return []
    profiles: list[ProfileMetric] = []
    with Path(path).open(newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            notes_raw = str(row.get("notes", ""))
            profiles.append(
                ProfileMetric(
                    profile_id=str(row.get("profile_id", "")),
                    variant_id=str(row.get("variant_id", "")),
                    source=str(row.get("source", "profile_metrics_csv")),
                    source_path=str(row.get("source_path", "")),
                    target_bmi=float(_as_float(row.get("target_bmi")) or 0.0),
                    target_height_cm=float(_as_float(row.get("target_height_cm")) or 0.0),
                    target_waist_cm=float(_as_float(row.get("target_waist_cm")) or 0.0),
                    achieved_waist_cm=float(_as_float(row.get("achieved_waist_cm")) or 0.0),
                    body_volume_l=float(_as_float(row.get("body_volume_l")) or 0.0),
                    z_extent_cm=float(_as_float(row.get("z_extent_cm")) or 0.0),
                    liver_volume_ml=float(_as_float(row.get("liver_volume_ml")) or 0.0),
                    kidney_volume_ml=float(_as_float(row.get("kidney_volume_ml")) or 0.0),
                    vascular_fluid_volume_ml=float(_as_float(row.get("vascular_fluid_volume_ml")) or 0.0),
                    notes=tuple(part for part in notes_raw.split(";") if part),
                )
            )
    return profiles


def _patient_metric_from_inputs(
    *,
    case_id: str,
    ct_path: str | Path,
    organ_seg_path: str | Path | None,
    vessel_seg_path: str | Path | None,
) -> PatientCaseMetric:
    nib = _import_nibabel()
    notes: list[str] = ["patient_metric_extracted_from_primary_ct_grid"]
    ct_image = nib.load(str(ct_path))
    ct_data = np.asarray(ct_image.dataobj, dtype=np.float32)
    spacing = _spacing_from_affine(np.asarray(ct_image.affine, dtype=float))
    body = _ct_body_mask(ct_data)
    organ_label_mode = "none"
    liver_volume = kidney_volume = aorta_volume = ivc_volume = None
    vessel_volume: float | None = None

    if organ_seg_path is not None and Path(organ_seg_path).exists():
        label_image = nib.load(str(organ_seg_path))
        labels = np.rint(np.asarray(label_image.dataobj)).astype(np.int16)
        if labels.shape == ct_data.shape:
            organ_label_mode = _detect_label_mode(labels)
            label_map = _label_map_for_mode(organ_label_mode)
            label_spacing = _spacing_from_affine(np.asarray(label_image.affine, dtype=float))
            label_body = labels > 0
            if np.count_nonzero(label_body) > 0:
                body = body | label_body
            liver_volume = _label_volume_ml(labels, label_map["liver"], label_spacing)
            kidney_volume = _label_volume_ml(labels, label_map["kidneys"], label_spacing)
            aorta_volume = _label_volume_ml(labels, label_map["aorta"], label_spacing)
            ivc_volume = _label_volume_ml(labels, label_map["ivc"], label_spacing)
            notes.append(f"organ_segmentation_metrics_mode={organ_label_mode}")
        else:
            notes.append("organ_segmentation_shape_mismatch_metrics_skipped")

    if vessel_seg_path is not None and Path(vessel_seg_path).exists():
        vessel_image = nib.load(str(vessel_seg_path))
        vessel = np.asarray(vessel_image.dataobj)
        if vessel.shape == ct_data.shape:
            vessel_spacing = _spacing_from_affine(np.asarray(vessel_image.affine, dtype=float))
            vessel_volume = _volume_ml(vessel > 0, vessel_spacing)
            if aorta_volume is None or aorta_volume <= 0.0:
                aorta_volume = vessel_volume
                notes.append("vessel_segmentation_volume_used_as_aorta_proxy")
        else:
            notes.append("vessel_segmentation_shape_mismatch_metrics_skipped")

    bbox = _bbox_mm(body, spacing)
    return PatientCaseMetric(
        scan_id=case_id,
        source_path=str(ct_path),
        scope="patient_ct_with_optional_segmentations",
        body_volume_l=float(_volume_ml(body, spacing) / 1000.0) if np.count_nonzero(body) else None,
        waist_cm=_estimate_waist_cm(body, spacing) if np.count_nonzero(body) else None,
        z_extent_cm=float(bbox[2] / 10.0) if bbox[2] else None,
        liver_volume_ml=liver_volume,
        kidney_volume_ml=kidney_volume,
        aorta_volume_ml=aorta_volume,
        ivc_volume_ml=ivc_volume,
        organ_label_mode=organ_label_mode,
        vessel_volume_ml=vessel_volume,
        notes=tuple(notes),
    )


def _to_scan_metric(metric: PatientCaseMetric) -> ScanMetric:
    return ScanMetric(
        scan_id=metric.scan_id,
        source="patient_case",
        scope=metric.scope,
        source_path=metric.source_path,
        body_volume_l=metric.body_volume_l,
        waist_cm=metric.waist_cm,
        z_extent_cm=metric.z_extent_cm,
        liver_volume_ml=metric.liver_volume_ml,
        kidney_volume_ml=metric.kidney_volume_ml,
        aorta_volume_ml=metric.aorta_volume_ml,
        ivc_volume_ml=metric.ivc_volume_ml,
        volume_stability_cv=None,
        notes=metric.notes,
    )


def _profile_input_score(
    profile: ProfileMetric,
    *,
    target_height_cm: float | None,
    target_bmi: float | None,
    target_waist_cm: float | None,
) -> tuple[float | None, tuple[str, ...]]:
    errors: list[float] = []
    features: list[str] = []
    if target_height_cm is not None and profile.target_height_cm > 0.0:
        errors.append(abs(profile.target_height_cm - target_height_cm) / 15.0)
        features.append("target_height")
    if target_bmi is not None and profile.target_bmi > 0.0:
        errors.append(abs(profile.target_bmi - target_bmi) / 6.0)
        features.append("target_bmi")
    if target_waist_cm is not None and profile.achieved_waist_cm > 0.0:
        errors.append(abs(profile.achieved_waist_cm - target_waist_cm) / 10.0)
        features.append("target_waist")
    return _score_components(errors), tuple(features)


def _score_profiles(
    metric: PatientCaseMetric,
    profiles: list[ProfileMetric],
    *,
    target_height_cm: float | None,
    target_bmi: float | None,
    target_waist_cm: float | None,
) -> tuple[PatientProfileMatch, ...]:
    scan = _to_scan_metric(metric)
    rows: list[PatientProfileMatch] = []
    for profile in profiles:
        base = _match(profile, scan)
        demographic_score, demographic_features = _profile_input_score(
            profile,
            target_height_cm=target_height_cm,
            target_bmi=target_bmi,
            target_waist_cm=target_waist_cm,
        )
        if demographic_score is None:
            score = base.match_score
            notes = list(base.notes)
        elif base.compared_features:
            score = 0.65 * base.match_score + 0.35 * demographic_score
            notes = [*base.notes, "match_score_blends_scan_geometry_and_user_profile_inputs"]
        else:
            score = demographic_score
            notes = [*base.notes, "match_score_from_user_profile_inputs_only"]
        features = tuple([*base.compared_features, *demographic_features])
        rows.append(
            PatientProfileMatch(
                profile_id=profile.profile_id,
                variant_id=profile.variant_id,
                profile_source=profile.source,
                source_path=profile.source_path,
                match_score=float(score),
                match_status=_match_status(float(score)),
                scan_geometry_score=base.match_score,
                profile_input_score=demographic_score,
                anthropometric_score=base.anthropometric_score,
                anchor_score=base.anchor_score,
                waist_delta_cm=base.waist_delta_cm,
                body_delta_l=base.body_delta_l,
                z_extent_delta_cm=base.z_extent_delta_cm,
                liver_delta_percent=base.liver_delta_percent,
                kidney_delta_percent=base.kidney_delta_percent,
                vascular_proxy_delta_percent=base.aorta_proxy_delta_percent,
                compared_features=features,
                notes=tuple(notes),
            )
        )
    return tuple(sorted(rows, key=lambda item: (-item.match_score, item.profile_id, item.variant_id)))


def _write_metric_yaml(path: Path, metric: PatientCaseMetric) -> None:
    payload = {
        "scan_id": metric.scan_id,
        "package_type": "patient_case_metric",
        "source_path": metric.source_path,
        "scope": metric.scope,
        "metrics": {
            "body_volume_l": metric.body_volume_l,
            "waist_cm": metric.waist_cm,
            "z_extent_cm": metric.z_extent_cm,
            "liver_volume_ml": metric.liver_volume_ml,
            "kidney_volume_ml": metric.kidney_volume_ml,
            "aorta_volume_ml": metric.aorta_volume_ml,
            "ivc_volume_ml": metric.ivc_volume_ml,
            "organ_label_mode": metric.organ_label_mode,
            "vessel_volume_ml": metric.vessel_volume_ml,
        },
        "notes": list(metric.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_scores_csv(path: Path, rows: tuple[PatientProfileMatch, ...]) -> None:
    fields = list(PatientProfileMatch.__dataclass_fields__)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload: dict[str, Any] = {}
            for field in fields:
                value = getattr(row, field)
                if isinstance(value, tuple):
                    payload[field] = ";".join(str(item) for item in value)
                elif isinstance(value, float):
                    payload[field] = f"{value:.6f}"
                elif value is None:
                    payload[field] = ""
                else:
                    payload[field] = value
            writer.writerow(payload)


def _write_preview(path: Path, rows: tuple[PatientProfileMatch, ...], metric: PatientCaseMetric, case_id: str) -> None:
    plt = _import_plotting()
    top = list(rows[:10])
    labels = [item.profile_id for item in top]
    scores = [item.match_score for item in top]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=170)
    fig.patch.set_facecolor("#f4f0e8")
    axes[0].barh(range(len(labels)), scores, color="#2f6f73")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("match score")
    axes[0].set_title("Closest Morph Profiles")
    axes[1].axis("off")
    metric_lines = [
        f"Case: {case_id}",
        f"Waist estimate: {_fmt(metric.waist_cm, 'cm')}",
        f"Body volume: {_fmt(metric.body_volume_l, 'L')}",
        f"Z extent: {_fmt(metric.z_extent_cm, 'cm')}",
        f"Liver: {_fmt(metric.liver_volume_ml, 'mL')}",
        f"Kidneys: {_fmt(metric.kidney_volume_ml, 'mL')}",
        f"Vascular proxy: {_fmt(metric.aorta_volume_ml, 'mL')}",
        f"Label mode: {metric.organ_label_mode}",
    ]
    axes[1].text(0.02, 0.95, "\n".join(metric_lines), va="top", ha="left", fontsize=10, family="monospace")
    fig.suptitle("Patient Case Adapter: Profile Match", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{value:.2f}{suffix}"


def _final_status(product_result: ProductCaseResult | None) -> str:
    if product_result is None:
        return "patient_case_profile_matched_score_only"
    if product_result.final_status == "research_demo_ready":
        return "patient_case_research_demo_ready"
    if product_result.final_status == "build_planned_only":
        return "patient_case_profile_matched_build_planned"
    if product_result.final_status == "research_demo_needs_corrections":
        return "patient_case_needs_corrections"
    if product_result.final_status == "research_demo_review_required":
        return "patient_case_review_required"
    if "blocked" in product_result.final_status:
        return "patient_case_profile_matched_build_blocked"
    return f"patient_case_{product_result.final_status}"


def _stage_statuses(product_result: ProductCaseResult | None) -> tuple[str, ...]:
    if product_result is None:
        return ()
    return tuple(f"{stage.stage_id}:{stage.status}" for stage in product_result.stages)


def _write_manifest(path: Path, result: PatientCaseAdapterResult, metric: PatientCaseMetric, matches: tuple[PatientProfileMatch, ...]) -> None:
    selected = matches[0]
    payload = {
        "case_id": result.case_id,
        "patient_id": result.patient_id,
        "package_type": "patient_case_adapter",
        "final_status": result.final_status,
        "selected_profile": {
            "profile_id": selected.profile_id,
            "variant_id": selected.variant_id,
            "profile_source": selected.profile_source,
            "source_path": selected.source_path,
            "match_score": selected.match_score,
            "match_status": selected.match_status,
        },
        "patient_metric": {
            "metric_yaml": result.patient_metric_yaml_path,
            "scope": metric.scope,
            "body_volume_l": metric.body_volume_l,
            "waist_cm": metric.waist_cm,
            "z_extent_cm": metric.z_extent_cm,
            "liver_volume_ml": metric.liver_volume_ml,
            "kidney_volume_ml": metric.kidney_volume_ml,
            "aorta_volume_ml": metric.aorta_volume_ml,
            "ivc_volume_ml": metric.ivc_volume_ml,
            "organ_label_mode": metric.organ_label_mode,
            "vessel_volume_ml": metric.vessel_volume_ml,
        },
        "configuration": {
            "baseline_graph": result.baseline_graph_path,
            "baseline_combined_spec": result.baseline_combined_spec_path,
        },
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "patient_metric_yaml": result.patient_metric_yaml_path,
            "match_scores_csv": result.match_scores_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
            "product_manifest": result.product_manifest_path,
            "adapter_manifest": result.adapter_manifest_path,
            "build_manifest": result.build_manifest_path,
            "qa_yaml": result.qa_yaml_path,
            "render_preview_png": result.render_preview_png_path,
        },
        "stage_statuses": list(result.stage_statuses),
        "top_matches": [
            {
                "profile_id": item.profile_id,
                "variant_id": item.variant_id,
                "match_score": item.match_score,
                "status": item.match_status,
                "features": list(item.compared_features),
            }
            for item in matches[:8]
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: PatientCaseAdapterResult, metric: PatientCaseMetric, matches: tuple[PatientProfileMatch, ...]) -> None:
    selected = matches[0]
    lines = [
        "# Patient Case Adapter",
        "",
        f"Case ID: `{result.case_id}`",
        f"Patient/profile ID: `{result.patient_id}`",
        f"Final status: `{result.final_status}`",
        "",
        "## Selected Morph Profile",
        "",
        f"- Selected profile: `{selected.profile_id}`",
        f"- Selected variant: `{selected.variant_id}`",
        f"- Match score: {selected.match_score:.1f} (`{selected.match_status}`)",
        f"- Scan geometry score: {selected.scan_geometry_score:.1f}",
        f"- User-input profile score: `{_fmt(selected.profile_input_score, '')}`",
        "",
        "## Patient Metrics Used For Matching",
        "",
        f"- Waist estimate: `{_fmt(metric.waist_cm, 'cm')}`",
        f"- Body volume: `{_fmt(metric.body_volume_l, 'L')}`",
        f"- Z extent: `{_fmt(metric.z_extent_cm, 'cm')}`",
        f"- Liver volume: `{_fmt(metric.liver_volume_ml, 'mL')}`",
        f"- Kidney volume: `{_fmt(metric.kidney_volume_ml, 'mL')}`",
        f"- Vascular/aorta proxy volume: `{_fmt(metric.aorta_volume_ml, 'mL')}`",
        f"- Organ label mode: `{metric.organ_label_mode}`",
        "",
        "## Automatic Product Rerun",
        "",
        f"- Product status: `{result.product_status or 'not_run'}`",
        f"- Product manifest: `{result.product_manifest_path or 'not_written'}`",
        f"- Build manifest: `{result.build_manifest_path or 'not_written'}`",
        f"- QA YAML: `{result.qa_yaml_path or 'not_written'}`",
        f"- Render preview: `{result.render_preview_png_path or 'not_written'}`",
        f"- Baseline graph: `{result.baseline_graph_path or 'not_resolved'}`",
        "",
        "## Top Matches",
        "",
        "| rank | profile | variant | score | status | features |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    for index, item in enumerate(matches[:10], start=1):
        features = ", ".join(item.compared_features)
        lines.append(f"| {index} | {item.profile_id} | {item.variant_id} | {item.match_score:.1f} | `{item.match_status}` | {features} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Patient metric YAML: `{result.patient_metric_yaml_path}`",
            f"- Match scores CSV: `{result.match_scores_csv_path}`",
            f"- Preview PNG: `{result.preview_png_path}`",
            f"- Adapter manifest: `{result.manifest_yaml_path}`",
            "",
            "## Limitations",
            "",
            "- CT-only cases can be matched to the morph library, but organ-specific material maps require organ/material segmentation.",
            "- Missing vessel segmentation uses the template vascular graph only when template vessels are explicitly allowed.",
            "- Profile matching is a research-demonstrator scoring step, not a validated clinical registration result.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_patient_case_adapter(
    *,
    input_ct_path: str | Path,
    input_cta_path: str | Path | None = None,
    input_ctv_path: str | Path | None = None,
    organ_seg_path: str | Path | None = None,
    vessel_seg_path: str | Path | None = None,
    patient_id: str = "patient_demo",
    case_id: str | None = None,
    output_dir: str | Path = "outputs/patient_case_adapter",
    profile_spec_glob: str = "outputs/experiments/profile_sweep/**/anthropometry/*_anthro_morph_spec_v001.yaml",
    profile_metrics_csv_path: str | Path | None = None,
    include_metric_scaled_height_grid: bool = True,
    organ_labelmap_path: str | Path = "configs/labelmaps/ct_org.yaml",
    materials_path: str | Path = "configs/materials.yaml",
    baseline_graph_path: str | Path | None = None,
    baseline_combined_spec_path: str | Path | None = None,
    approved_set_manifest_path: str | Path | None = None,
    target_height_cm: float | None = None,
    target_weight_kg: float | None = None,
    target_bmi: float | None = None,
    target_waist_cm: float | None = None,
    copy_inputs: bool = False,
    allow_template_vessels: bool = True,
    dry_run: bool = False,
    run_rt: bool = True,
    export_dicom: bool = False,
    sample_step_mm: float = 0.75,
    vessel_wall_thickness_mm: float = 2.0,
    arterial_inlet_flow_ml_s: float = 80.0,
    heart_rate_bpm: float = 60.0,
    organ_label_mode: str = "auto",
    correct_bone_conflicts: bool = False,
    bone_clearance_mm: float = 8.0,
    run_qa: bool = True,
    qa_expected_lumen_components: int = 1,
    render_3d: bool = True,
    render_target_max_faces: int = 90_000,
    score_only: bool = False,
    report_path: str | Path | None = None,
) -> PatientCaseAdapterResult:
    if input_ct_path is None or str(input_ct_path) == "":
        raise ValueError("input_ct_path is required for patient case adaptation.")
    if not Path(input_ct_path).exists():
        raise FileNotFoundError(f"Input CT does not exist: {input_ct_path}")

    resolved_case_id = case_id or f"{_slug(patient_id)}_patient_case_adapter"
    output_root = Path(output_dir) / resolved_case_id
    metric_yaml = output_root / f"{resolved_case_id}_patient_metric_v001.yaml"
    scores_csv = output_root / f"{resolved_case_id}_profile_match_scores_v001.csv"
    manifest_yaml = output_root / f"{resolved_case_id}_patient_case_adapter_manifest_v001.yaml"
    preview_png = output_root / f"{resolved_case_id}_profile_match_preview_v001.png"
    report = Path(report_path) if report_path is not None else output_root / f"{resolved_case_id}_patient_case_adapter_report_v001.md"

    metric = _patient_metric_from_inputs(
        case_id=resolved_case_id,
        ct_path=input_ct_path,
        organ_seg_path=organ_seg_path,
        vessel_seg_path=vessel_seg_path,
    )
    profiles = _read_profile_metrics_csv(profile_metrics_csv_path) if profile_metrics_csv_path is not None else []
    if not profiles:
        profiles = _read_profile_specs(profile_spec_glob)
        if include_metric_scaled_height_grid:
            profiles.extend(_synthetic_height_profiles(profiles))
    if not profiles:
        raise ValueError("No morph/profile library entries were found for patient case adaptation.")

    effective_target_waist = target_waist_cm if target_waist_cm is not None else metric.waist_cm
    matches = _score_profiles(
        metric,
        profiles,
        target_height_cm=target_height_cm,
        target_bmi=target_bmi,
        target_waist_cm=effective_target_waist,
    )
    selected_profile = next(profile for profile in profiles if profile.profile_id == matches[0].profile_id and profile.variant_id == matches[0].variant_id)

    active_baseline = resolve_stage007_active_baseline()
    resolved_baseline_graph = (
        _resolve_existing(baseline_graph_path)
        or active_baseline.graph_path
        or _first_existing(DEFAULT_BASELINE_GRAPH_CANDIDATES)
    )
    resolved_combined_spec = (
        _resolve_existing(baseline_combined_spec_path)
        or active_baseline.voxelized_spec_path
        or _first_existing(DEFAULT_BASELINE_COMBINED_SPEC_CANDIDATES)
    )
    notes = [
        "patient_case_adapter_scores_patient_ct_and_optional_segmentations_against_morph_library",
        "selected_profile_is_recorded_as_population_prior_for_patient_build",
    ]
    if baseline_graph_path is None and active_baseline.graph_path is not None:
        notes.append("baseline_graph_auto_resolved_from_stage007_active_baseline")
    elif baseline_graph_path is None and resolved_baseline_graph is not None:
        notes.append("baseline_graph_auto_discovered_from_legacy_project_outputs")
    if baseline_combined_spec_path is None and active_baseline.voxelized_spec_path is not None:
        notes.append("baseline_reference_spec_auto_resolved_from_stage007_active_voxelized_spec")
    if organ_seg_path is None:
        notes.append("organ_segmentation_missing_product_build_will_be_planned_or_blocked")
    if vessel_seg_path is None and allow_template_vessels:
        notes.append("vessel_segmentation_missing_template_vascular_graph_allowed")

    product_result: ProductCaseResult | None = None
    if not score_only:
        product_result = build_product_case(
            input_ct_path=input_ct_path,
            input_cta_path=input_cta_path,
            input_ctv_path=input_ctv_path,
            organ_seg_path=organ_seg_path,
            vessel_seg_path=vessel_seg_path,
            existing_build_manifest_path=None,
            patient_id=patient_id,
            case_id=resolved_case_id,
            output_dir=output_root / "product_case",
            organ_labelmap_path=organ_labelmap_path,
            materials_path=materials_path,
            baseline_graph_path=resolved_baseline_graph,
            baseline_combined_spec_path=resolved_combined_spec,
            approved_set_manifest_path=approved_set_manifest_path,
            target_height_cm=target_height_cm or (selected_profile.target_height_cm if selected_profile.target_height_cm > 0.0 else None),
            target_weight_kg=target_weight_kg,
            target_bmi=target_bmi or (selected_profile.target_bmi if selected_profile.target_bmi > 0.0 else None),
            target_waist_cm=target_waist_cm or metric.waist_cm or (selected_profile.achieved_waist_cm if selected_profile.achieved_waist_cm > 0.0 else None),
            copy_inputs=copy_inputs,
            allow_template_vessels=allow_template_vessels,
            dry_run=dry_run,
            run_rt=run_rt,
            export_dicom=export_dicom,
            sample_step_mm=sample_step_mm,
            vessel_wall_thickness_mm=vessel_wall_thickness_mm,
            arterial_inlet_flow_ml_s=arterial_inlet_flow_ml_s,
            heart_rate_bpm=heart_rate_bpm,
            organ_label_mode=organ_label_mode,
            correct_bone_conflicts=correct_bone_conflicts,
            bone_clearance_mm=bone_clearance_mm,
            run_qa=run_qa,
            qa_expected_lumen_components=qa_expected_lumen_components,
            render_3d=render_3d,
            render_target_max_faces=render_target_max_faces,
        )

    final_status = _final_status(product_result)
    result = PatientCaseAdapterResult(
        case_id=resolved_case_id,
        patient_id=patient_id,
        output_dir=str(output_root),
        final_status=final_status,
        selected_profile_id=matches[0].profile_id,
        selected_variant_id=matches[0].variant_id,
        selected_match_score=matches[0].match_score,
        baseline_graph_path=resolved_baseline_graph,
        baseline_combined_spec_path=resolved_combined_spec,
        patient_metric_yaml_path=str(metric_yaml),
        match_scores_csv_path=str(scores_csv),
        manifest_yaml_path=str(manifest_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        product_manifest_path=None if product_result is None else product_result.product_manifest_path,
        adapter_manifest_path=None if product_result is None else product_result.adapter_manifest_path,
        build_manifest_path=None if product_result is None else product_result.build_manifest_path,
        qa_yaml_path=None if product_result is None else product_result.qa_yaml_path,
        render_preview_png_path=None if product_result is None else product_result.render_preview_png_path,
        profile_count=len(profiles),
        product_status=None if product_result is None else product_result.final_status,
        stage_statuses=_stage_statuses(product_result),
        notes=tuple(notes),
    )
    _write_metric_yaml(metric_yaml, metric)
    _write_scores_csv(scores_csv, matches)
    _write_preview(preview_png, matches, metric, resolved_case_id)
    _write_manifest(manifest_yaml, result, metric, matches)
    _write_report(report, result, metric, matches)
    return result


def format_patient_case_adapter_result(result: PatientCaseAdapterResult) -> str:
    lines = [
        "Patient Case Adapter",
        f"Case ID: {result.case_id}",
        f"Patient/profile ID: {result.patient_id}",
        f"Final status: {result.final_status}",
        f"Selected profile: {result.selected_profile_id}",
        f"Selected variant: {result.selected_variant_id}",
        f"Selected match score: {result.selected_match_score:.1f}",
        f"Profiles scored: {result.profile_count}",
        f"Manifest: {result.manifest_yaml_path}",
        f"Report: {result.report_path}",
    ]
    if result.product_manifest_path:
        lines.append(f"Product manifest: {result.product_manifest_path}")
    if result.render_preview_png_path:
        lines.append(f"3D preview PNG: {result.render_preview_png_path}")
    for status in result.stage_statuses:
        lines.append(f"- {status}")
    return "\n".join(lines)
