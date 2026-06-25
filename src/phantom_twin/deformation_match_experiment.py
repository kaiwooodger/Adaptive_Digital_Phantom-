from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import glob
import math

import numpy as np
import yaml


@dataclass(frozen=True)
class ScanMetric:
    scan_id: str
    source: str
    scope: str
    source_path: str
    body_volume_l: float | None
    waist_cm: float | None
    z_extent_cm: float | None
    liver_volume_ml: float | None
    kidney_volume_ml: float | None
    aorta_volume_ml: float | None
    ivc_volume_ml: float | None
    volume_stability_cv: float | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ProfileMetric:
    profile_id: str
    variant_id: str
    source: str
    source_path: str
    target_bmi: float
    target_height_cm: float
    target_waist_cm: float
    achieved_waist_cm: float
    body_volume_l: float
    z_extent_cm: float
    liver_volume_ml: float
    kidney_volume_ml: float
    vascular_fluid_volume_ml: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DeformationMatchRow:
    scan_id: str
    scan_source: str
    scan_scope: str
    profile_id: str
    variant_id: str
    profile_source: str
    match_score: float
    match_status: str
    anthropometric_score: float | None
    anchor_score: float | None
    waist_delta_cm: float | None
    body_delta_l: float | None
    z_extent_delta_cm: float | None
    liver_delta_percent: float | None
    kidney_delta_percent: float | None
    aorta_proxy_delta_percent: float | None
    compared_features: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DeformationMatchExperimentResult:
    experiment_id: str
    output_dir: str
    scan_metrics_csv_path: str
    profile_metrics_csv_path: str
    match_matrix_csv_path: str
    top_matches_csv_path: str
    manifest_yaml_path: str
    atlas_png_path: str
    report_path: str
    scan_count: int
    profile_count: int
    match_count: int
    best_match_score: float
    median_best_scan_score: float
    notes: tuple[str, ...]


PROFILE_BMI_GRID = (22.0, 27.0, 32.0, 35.0, 38.0)
PROFILE_HEIGHT_GRID = (165.0, 175.0, 185.0)


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Deformation match experiment requires nibabel.") from exc
    return nib


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Deformation match experiment preview requires matplotlib.") from exc
    return plt


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    return data if isinstance(data, dict) else {}


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))  # type: ignore[return-value]


def _bbox_mm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0, 0.0, 0.0
    extent = coords.max(axis=0) - coords.min(axis=0) + 1
    return tuple(float(value) for value in extent * np.asarray(spacing_mm, dtype=float))  # type: ignore[return-value]


def _ellipse_circumference_cm(width_mm: float, depth_mm: float) -> float:
    a = max(width_mm / 2.0, 1e-6)
    b = max(depth_mm / 2.0, 1e-6)
    circumference_mm = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
    return float(circumference_mm / 10.0)


def _estimate_waist_cm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    z_min, z_max = int(coords[:, 2].min()), int(coords[:, 2].max())
    z_index = int(round(z_min + 0.46 * (z_max - z_min)))
    slice_mask = mask[:, :, z_index]
    slice_coords = np.argwhere(slice_mask)
    if slice_coords.size == 0:
        bbox = _bbox_mm(mask, spacing_mm)
        return _ellipse_circumference_cm(bbox[0], bbox[1])
    width_mm = float((slice_coords[:, 0].max() - slice_coords[:, 0].min() + 1) * spacing_mm[0])
    depth_mm = float((slice_coords[:, 1].max() - slice_coords[:, 1].min() + 1) * spacing_mm[1])
    return _ellipse_circumference_cm(width_mm, depth_mm)


def _volume_ml(labels: np.ndarray, label_ids: tuple[int, ...], voxel_volume_ml: float) -> float:
    if not label_ids:
        return 0.0
    return float(np.count_nonzero(np.isin(labels, label_ids)) * voxel_volume_ml)


def _metric_from_label_path(
    *,
    scan_id: str,
    source: str,
    scope: str,
    path: str | Path,
    label_map: dict[str, tuple[int, ...]],
    notes: tuple[str, ...] = (),
) -> ScanMetric:
    nib = _import_nibabel()
    image = nib.load(str(path))
    labels = np.asarray(image.dataobj, dtype=np.int16)
    spacing = _spacing_from_affine(np.asarray(image.affine, dtype=float))
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    body = labels > 0
    bbox = _bbox_mm(body, spacing)
    body_volume_l = float(np.count_nonzero(body) * voxel_volume_ml / 1000.0) if scope == "body_envelope" else None
    waist = _estimate_waist_cm(body, spacing) if scope == "body_envelope" else None
    return ScanMetric(
        scan_id=scan_id,
        source=source,
        scope=scope,
        source_path=str(path),
        body_volume_l=body_volume_l,
        waist_cm=waist,
        z_extent_cm=float(bbox[2] / 10.0) if bbox[2] else None,
        liver_volume_ml=_volume_ml(labels, label_map.get("liver", ()), voxel_volume_ml),
        kidney_volume_ml=_volume_ml(labels, label_map.get("kidneys", ()), voxel_volume_ml),
        aorta_volume_ml=_volume_ml(labels, label_map.get("aorta", ()), voxel_volume_ml),
        ivc_volume_ml=_volume_ml(labels, label_map.get("ivc", ()), voxel_volume_ml),
        volume_stability_cv=None,
        notes=notes,
    )


def _read_ct_org_scans(manifest_csv: str | Path) -> list[ScanMetric]:
    scans: list[ScanMetric] = []
    if not Path(manifest_csv).exists():
        return scans
    with Path(manifest_csv).open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            path = row.get("material_label_path", "")
            if not path or not Path(path).exists():
                continue
            scans.append(
                _metric_from_label_path(
                    scan_id=str(row.get("case_id", Path(path).stem)),
                    source="ct_org_material_population",
                    scope="body_envelope",
                    path=path,
                    label_map={"liver": (6,), "kidneys": (7,), "aorta": (), "ivc": ()},
                    notes=("material_label_population_case",),
                )
            )
    return scans


def _read_btcv_scan(label_path: str | Path) -> list[ScanMetric]:
    if not Path(label_path).exists():
        return []
    return [
        _metric_from_label_path(
            scan_id="btcv_abdomen_case0001",
            source="btcv_abdomen",
            scope="partial_abdomen_organs",
            path=label_path,
            label_map={"liver": (6,), "kidneys": (2, 3), "aorta": (8,), "ivc": (9,)},
            notes=("partial_abdomen_organ_union_not_full_body",),
        )
    ]


def _read_reg_training_scans(staged_manifest_path: str | Path) -> list[ScanMetric]:
    manifest = _load_yaml(staged_manifest_path)
    csv_path = manifest.get("outputs", {}).get("manifest_csv")
    if not csv_path or not Path(csv_path).exists():
        return []
    grouped: dict[str, list[str]] = {}
    with Path(csv_path).open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row.get("label_path"):
                grouped.setdefault(str(row["target_case_id"]), []).append(str(row["label_path"]))
    nib = _import_nibabel()
    scans: list[ScanMetric] = []
    for target_id, paths in sorted(grouped.items()):
        if not paths:
            continue
        first = nib.load(paths[0])
        spacing = _spacing_from_affine(np.asarray(first.affine, dtype=float))
        voxel_volume_ml = float(np.prod(spacing) / 1000.0)
        volumes_by_label: dict[int, list[float]] = {label_id: [] for label_id in range(1, 14)}
        for path in paths:
            labels = np.asarray(nib.load(path).dataobj, dtype=np.uint8)
            counts = np.bincount(labels.ravel(), minlength=14)
            for label_id in range(1, 14):
                volumes_by_label[label_id].append(float(counts[label_id] * voxel_volume_ml))
        mean_volume = {label_id: float(np.mean(values)) for label_id, values in volumes_by_label.items()}
        cvs = [
            float(np.std(values) / np.mean(values))
            for values in volumes_by_label.values()
            if values and np.mean(values) > 1e-6
        ]
        scans.append(
            ScanMetric(
                scan_id=f"reg_target_{target_id}",
                source="reg_training_testing_all_labels",
                scope="deformation_label_target",
                source_path=str(paths[0]),
                body_volume_l=None,
                waist_cm=None,
                z_extent_cm=float(first.shape[2] * spacing[2] / 10.0),
                liver_volume_ml=mean_volume.get(6, 0.0),
                kidney_volume_ml=mean_volume.get(2, 0.0) + mean_volume.get(3, 0.0),
                aorta_volume_ml=mean_volume.get(8, 0.0),
                ivc_volume_ml=mean_volume.get(9, 0.0),
                volume_stability_cv=float(np.mean(cvs)) if cvs else None,
                notes=("mean_volume_across_30_propagated_registration_labels", "uses_all_deformation_label_maps_for_target"),
            )
        )
    return scans


def _read_avt_aorta_scans(manifest_csv: str | Path) -> list[ScanMetric]:
    scans: list[ScanMetric] = []
    if not Path(manifest_csv).exists():
        return scans
    with Path(manifest_csv).open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            scans.append(
                ScanMetric(
                    scan_id=str(row.get("case_id", "")),
                    source="avt_kits_aorta",
                    scope="aorta_only",
                    source_path=str(row.get("aorta_mask_nifti_path", "")),
                    body_volume_l=None,
                    waist_cm=None,
                    z_extent_cm=_as_float(row.get("aorta_z_span_mm"), 0.0) / 10.0 if _as_float(row.get("aorta_z_span_mm")) is not None else None,
                    liver_volume_ml=None,
                    kidney_volume_ml=None,
                    aorta_volume_ml=_as_float(row.get("aorta_volume_ml")),
                    ivc_volume_ml=None,
                    volume_stability_cv=None,
                    notes=("aorta_only_registration_benchmark_case",),
                )
            )
    return scans


def _region_volume(spec: dict[str, Any], label_name: str) -> float:
    for item in spec.get("region_stats", []):
        if isinstance(item, dict) and str(item.get("name", "")).lower() == label_name.lower():
            value = _as_float(item.get("morphed_volume_cm3"), 0.0)
            return float(value or 0.0)
    return 0.0


def _profile_from_spec(path: str | Path) -> ProfileMetric:
    spec = _load_yaml(path)
    anthropometry = spec.get("anthropometry", {})
    quality = spec.get("quality_summary", {})
    profile_id = Path(path).parents[1].name
    return ProfileMetric(
        profile_id=profile_id,
        variant_id=str(spec.get("case_id", profile_id)),
        source="actual_deformation_profile",
        source_path=str(path),
        target_bmi=float(anthropometry.get("target_bmi") or 0.0),
        target_height_cm=float(anthropometry.get("target_height_cm") or 0.0),
        target_waist_cm=float(anthropometry.get("target_waist_cm") or 0.0),
        achieved_waist_cm=float(anthropometry.get("achieved_waist_cm") or 0.0),
        body_volume_l=float(quality.get("morphed_body_volume_cm3") or 0.0) / 1000.0,
        z_extent_cm=float((quality.get("morphed_bbox_mm") or [0.0, 0.0, 0.0])[2]) / 10.0,
        liver_volume_ml=_region_volume(spec, "liver"),
        kidney_volume_ml=_region_volume(spec, "kidneys"),
        vascular_fluid_volume_ml=_region_volume(spec, "blood_equivalent_fluid"),
        notes=("actual_morphed_nifti_outputs_exist",),
    )


def _read_profile_specs(profile_spec_glob: str) -> list[ProfileMetric]:
    if Path(profile_spec_glob).is_absolute():
        paths = [Path(path) for path in sorted(glob.glob(profile_spec_glob, recursive=True))]
    else:
        paths = sorted(Path().glob(profile_spec_glob))
    profiles = [_profile_from_spec(path) for path in paths]
    unique: dict[tuple[str, str], ProfileMetric] = {}
    for profile in profiles:
        unique[(profile.profile_id, profile.variant_id)] = profile
    return list(unique.values())


def _waist_from_bmi(bmi: float) -> float:
    return float(85.0 + (bmi - 22.0) / max(38.0 - 22.0, 1e-6) * (125.0 - 85.0))


def _synthetic_height_profiles(actual_profiles: list[ProfileMetric]) -> list[ProfileMetric]:
    if not actual_profiles:
        return []
    profiles: list[ProfileMetric] = []
    for bmi in PROFILE_BMI_GRID:
        nearest = min(actual_profiles, key=lambda item: abs(item.target_bmi - bmi))
        target_waist = _waist_from_bmi(bmi)
        waist_scale = target_waist / max(nearest.achieved_waist_cm, 1e-6)
        for height in PROFILE_HEIGHT_GRID:
            height_scale = height / max(nearest.target_height_cm, 1e-6)
            volume_scale = waist_scale * waist_scale * height_scale
            profiles.append(
                ProfileMetric(
                    profile_id=f"bmi{bmi:g}_waist{target_waist:g}_height{height:g}",
                    variant_id=f"metric_scaled_bmi{bmi:g}_height{height:g}",
                    source="metric_scaled_height_bmi_variant",
                    source_path=nearest.source_path,
                    target_bmi=bmi,
                    target_height_cm=height,
                    target_waist_cm=target_waist,
                    achieved_waist_cm=target_waist,
                    body_volume_l=nearest.body_volume_l * volume_scale,
                    z_extent_cm=nearest.z_extent_cm * height_scale,
                    liver_volume_ml=nearest.liver_volume_ml * height_scale,
                    kidney_volume_ml=nearest.kidney_volume_ml * height_scale,
                    vascular_fluid_volume_ml=nearest.vascular_fluid_volume_ml * height_scale,
                    notes=("metric_scaled_variant_no_new_nifti_volume_written", f"nearest_actual_profile={nearest.profile_id}"),
                )
            )
    return profiles


def _percent_delta(profile_value: float | None, scan_value: float | None) -> float | None:
    if profile_value is None or scan_value is None or scan_value <= 1e-6:
        return None
    return float((profile_value - scan_value) / scan_value * 100.0)


def _score_components(errors: list[float]) -> float | None:
    if not errors:
        return None
    rms = math.sqrt(sum(value * value for value in errors) / len(errors))
    return float(100.0 / (1.0 + rms))


def _status(score: float) -> str:
    if score >= 70.0:
        return "close_match"
    if score >= 45.0:
        return "moderate_match"
    return "weak_match"


def _match(profile: ProfileMetric, scan: ScanMetric) -> DeformationMatchRow:
    anthropometric_errors: list[float] = []
    anchor_errors: list[float] = []
    features: list[str] = []
    notes: list[str] = []

    waist_delta = profile.achieved_waist_cm - scan.waist_cm if scan.waist_cm is not None else None
    if waist_delta is not None:
        anthropometric_errors.append(abs(waist_delta) / 10.0)
        features.append("waist")
    body_delta = profile.body_volume_l - scan.body_volume_l if scan.body_volume_l is not None else None
    if body_delta is not None:
        anthropometric_errors.append(abs(body_delta) / max(5.0, 0.25 * scan.body_volume_l))
        features.append("body_volume")
    z_delta = profile.z_extent_cm - scan.z_extent_cm if scan.z_extent_cm is not None else None
    if z_delta is not None:
        anthropometric_errors.append(abs(z_delta) / 15.0)
        features.append("z_extent")

    liver_delta = _percent_delta(profile.liver_volume_ml, scan.liver_volume_ml)
    if liver_delta is not None:
        anchor_errors.append(abs(liver_delta) / 50.0)
        features.append("approved_liver_anchor")
    kidney_delta = _percent_delta(profile.kidney_volume_ml, scan.kidney_volume_ml)
    if kidney_delta is not None:
        anchor_errors.append(0.35 * abs(kidney_delta) / 70.0)
        features.append("review_kidney_anchor")
    aorta_delta = _percent_delta(profile.vascular_fluid_volume_ml, scan.aorta_volume_ml)
    if aorta_delta is not None:
        anchor_errors.append(0.25 * abs(aorta_delta) / 80.0)
        features.append("review_aorta_proxy")
        notes.append("profile_vascular_fluid_compared_to_scan_aorta_as_proxy")

    anthropometric_score = _score_components(anthropometric_errors)
    anchor_score = _score_components(anchor_errors)
    all_errors = anthropometric_errors + anchor_errors
    match_score = _score_components(all_errors) or 0.0
    if scan.volume_stability_cv is not None and scan.volume_stability_cv > 0.85:
        match_score *= 0.90
        notes.append("registration_target_has_high_label_volume_variability")
    return DeformationMatchRow(
        scan_id=scan.scan_id,
        scan_source=scan.source,
        scan_scope=scan.scope,
        profile_id=profile.profile_id,
        variant_id=profile.variant_id,
        profile_source=profile.source,
        match_score=match_score,
        match_status=_status(match_score),
        anthropometric_score=anthropometric_score,
        anchor_score=anchor_score,
        waist_delta_cm=waist_delta,
        body_delta_l=body_delta,
        z_extent_delta_cm=z_delta,
        liver_delta_percent=liver_delta,
        kidney_delta_percent=kidney_delta,
        aorta_proxy_delta_percent=aorta_delta,
        compared_features=tuple(features),
        notes=tuple(notes),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _optional(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.6f}"


def _write_scan_csv(path: Path, scans: list[ScanMetric]) -> None:
    fields = list(ScanMetric.__dataclass_fields__)
    _write_csv(path, [{field: ";".join(getattr(row, field)) if field == "notes" else getattr(row, field) for field in fields} for row in scans], fields)


def _write_profile_csv(path: Path, profiles: list[ProfileMetric]) -> None:
    fields = list(ProfileMetric.__dataclass_fields__)
    _write_csv(path, [{field: ";".join(getattr(row, field)) if field == "notes" else getattr(row, field) for field in fields} for row in profiles], fields)


def _write_match_csv(path: Path, rows: list[DeformationMatchRow]) -> None:
    fields = list(DeformationMatchRow.__dataclass_fields__)
    payload = []
    for row in rows:
        item: dict[str, Any] = {}
        for field in fields:
            value = getattr(row, field)
            if isinstance(value, tuple):
                item[field] = ";".join(str(part) for part in value)
            elif isinstance(value, float):
                item[field] = f"{value:.6f}"
            elif value is None:
                item[field] = ""
            else:
                item[field] = value
        payload.append(item)
    _write_csv(path, payload, fields)


def _top_matches(rows: list[DeformationMatchRow]) -> list[DeformationMatchRow]:
    best: dict[str, DeformationMatchRow] = {}
    for row in rows:
        current = best.get(row.scan_id)
        if current is None or row.match_score > current.match_score:
            best[row.scan_id] = row
    return sorted(best.values(), key=lambda row: (-row.match_score, row.scan_source, row.scan_id))


def _write_top_csv(path: Path, rows: list[DeformationMatchRow]) -> None:
    _write_match_csv(path, _top_matches(rows))


def _write_manifest(path: Path, result: DeformationMatchExperimentResult) -> None:
    payload = {
        "experiment_id": result.experiment_id,
        "package_type": "deformation_match_experiment",
        "summary": {
            "scan_count": result.scan_count,
            "profile_count": result.profile_count,
            "match_count": result.match_count,
            "best_match_score": result.best_match_score,
            "median_best_scan_score": result.median_best_scan_score,
        },
        "outputs": {
            "scan_metrics_csv": result.scan_metrics_csv_path,
            "profile_metrics_csv": result.profile_metrics_csv_path,
            "match_matrix_csv": result.match_matrix_csv_path,
            "top_matches_csv": result.top_matches_csv_path,
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_atlas(path: Path, rows: list[DeformationMatchRow], experiment_id: str) -> None:
    plt = _import_plotting()
    top = _top_matches(rows)
    source_order = sorted({row.scan_source for row in top})
    profile_order = sorted({row.profile_id for row in top[:40]})
    if not source_order or not profile_order:
        return
    matrix = np.full((len(source_order), len(profile_order)), np.nan, dtype=float)
    for row in rows:
        if row.scan_source in source_order and row.profile_id in profile_order:
            i = source_order.index(row.scan_source)
            j = profile_order.index(row.profile_id)
            matrix[i, j] = np.nanmax([matrix[i, j], row.match_score])
    fig, ax = plt.subplots(figsize=(max(8, len(profile_order) * 0.55), 4.8), dpi=170)
    fig.patch.set_facecolor("#f7f1e3")
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_title(f"Best Deformation Match Scores by Scan Source\n{experiment_id}")
    ax.set_yticks(range(len(source_order)))
    ax.set_yticklabels(source_order)
    ax.set_xticks(range(len(profile_order)))
    ax.set_xticklabels(profile_order, rotation=55, ha="right", fontsize=7)
    fig.colorbar(image, ax=ax, label="match score")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, result: DeformationMatchExperimentResult, rows: list[DeformationMatchRow]) -> None:
    top = _top_matches(rows)
    by_source: dict[str, list[DeformationMatchRow]] = {}
    for row in top:
        by_source.setdefault(row.scan_source, []).append(row)
    lines = [
        "# Deformation Match Experiment",
        "",
        f"Experiment ID: `{result.experiment_id}`",
        "",
        "## Summary",
        "",
        f"- Scan-derived targets: {result.scan_count}",
        f"- BMI/height profile variants: {result.profile_count}",
        f"- Pairwise matches scored: {result.match_count}",
        f"- Best match score: {result.best_match_score:.1f}",
        f"- Median best-per-scan score: {result.median_best_scan_score:.1f}",
        "",
        "## Best Matches By Source",
        "",
        "| source | target count | median best score | best profile examples |",
        "| --- | ---: | ---: | --- |",
    ]
    for source, source_rows in sorted(by_source.items()):
        scores = [row.match_score for row in source_rows]
        examples = ", ".join(f"{row.scan_id}->{row.profile_id} ({row.match_score:.1f})" for row in source_rows[:3])
        lines.append(f"| {source} | {len(source_rows)} | {float(np.median(scores)):.1f} | {examples} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- CT-ORG body-envelope cases are the strongest check for waist/body-volume/profile matching.",
            "- Reg-Training-Testing labels contribute all available deformation-label targets, but labels-only scoring omits CT intensity NCC to avoid extracting the 15 GB image archive.",
            "- AVT/KiTS contributes all 20 aorta cases as a vessel-only match check.",
            "- Metric-scaled height variants test BMI/height factors without writing new large NIfTI volumes.",
            "",
            "## Outputs",
            "",
            f"- Scan metrics CSV: `{result.scan_metrics_csv_path}`",
            f"- Profile metrics CSV: `{result.profile_metrics_csv_path}`",
            f"- Match matrix CSV: `{result.match_matrix_csv_path}`",
            f"- Top matches CSV: `{result.top_matches_csv_path}`",
            f"- Atlas PNG: `{result.atlas_png_path}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_deformation_match_experiment(
    *,
    experiment_id: str = "all_available_deformation_match_stage001",
    output_dir: str | Path = "outputs/experiments/deformation_match",
    ct_org_manifest_csv: str | Path = "data/processed/ct_org_label_population/ct_org_label_population_manifest_v001.csv",
    btcv_label_path: str | Path = "data/raw/btcv_abdomen/case0001/label0001.nii.gz",
    reg_training_manifest_path: str | Path = "data/processed/reg_training_testing_all_labels/reg_training_testing_all_labels_stage001_manifest_v001.yaml",
    avt_aorta_manifest_csv: str | Path = "data/processed/avt_kits_aorta/avt_kits_aorta_stage001_manifest_v001.csv",
    profile_spec_glob: str = "outputs/experiments/profile_sweep/**/anthropometry/*_anthro_morph_spec_v001.yaml",
    include_metric_scaled_height_grid: bool = True,
    report_path: str | Path | None = "outputs/reports/deformation_match_experiment_stage001.md",
) -> DeformationMatchExperimentResult:
    output = Path(output_dir) / experiment_id
    scans: list[ScanMetric] = []
    scans.extend(_read_ct_org_scans(ct_org_manifest_csv))
    scans.extend(_read_btcv_scan(btcv_label_path))
    scans.extend(_read_reg_training_scans(reg_training_manifest_path))
    scans.extend(_read_avt_aorta_scans(avt_aorta_manifest_csv))
    profiles = _read_profile_specs(profile_spec_glob)
    if include_metric_scaled_height_grid:
        profiles.extend(_synthetic_height_profiles(profiles))
    if not scans:
        raise ValueError("No scan-derived targets were found for deformation match experiment.")
    if not profiles:
        raise ValueError("No profile variants were found for deformation match experiment.")
    rows = [_match(profile, scan) for scan in scans for profile in profiles]
    best_rows = _top_matches(rows)
    best_scores = [row.match_score for row in best_rows]
    scan_metrics_csv = output / f"{experiment_id}_scan_metrics_v001.csv"
    profile_metrics_csv = output / f"{experiment_id}_profile_metrics_v001.csv"
    match_matrix_csv = output / f"{experiment_id}_match_matrix_v001.csv"
    top_matches_csv = output / f"{experiment_id}_top_matches_v001.csv"
    manifest_yaml = output / f"{experiment_id}_manifest_v001.yaml"
    atlas_png = output / f"{experiment_id}_match_atlas_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{experiment_id}_report_v001.md"
    result = DeformationMatchExperimentResult(
        experiment_id=experiment_id,
        output_dir=str(output),
        scan_metrics_csv_path=str(scan_metrics_csv),
        profile_metrics_csv_path=str(profile_metrics_csv),
        match_matrix_csv_path=str(match_matrix_csv),
        top_matches_csv_path=str(top_matches_csv),
        manifest_yaml_path=str(manifest_yaml),
        atlas_png_path=str(atlas_png),
        report_path=str(report),
        scan_count=len(scans),
        profile_count=len(profiles),
        match_count=len(rows),
        best_match_score=max(best_scores) if best_scores else 0.0,
        median_best_scan_score=float(np.median(best_scores)) if best_scores else 0.0,
        notes=(
            "uses_all_currently_staged_scan_targets_and_all_label_only_reg_training_targets",
            "full_reg_training_testing_images_are_not_extracted_due_to_disk_constraints",
            "metric_scaled_height_variants_are_scores_only_and_do_not_write_new_deformed_nifti_volumes",
            "approved_liver_anchor_is_weighted_more_strongly_than_review_only_kidney_aorta_ivc_features",
        ),
    )
    _write_scan_csv(scan_metrics_csv, scans)
    _write_profile_csv(profile_metrics_csv, profiles)
    _write_match_csv(match_matrix_csv, rows)
    _write_top_csv(top_matches_csv, rows)
    _write_atlas(atlas_png, rows, experiment_id)
    _write_manifest(manifest_yaml, result)
    _write_report(report, result, rows)
    return result


def format_deformation_match_experiment_result(result: DeformationMatchExperimentResult) -> str:
    return "\n".join(
        [
            "Deformation match experiment completed",
            f"Experiment ID: {result.experiment_id}",
            f"Scans/profiles/matches: {result.scan_count}/{result.profile_count}/{result.match_count}",
            f"Best match score: {result.best_match_score:.1f}",
            f"Median best-per-scan score: {result.median_best_scan_score:.1f}",
            f"Manifest: {result.manifest_yaml_path}",
            f"Match atlas: {result.atlas_png_path}",
            f"Report: {result.report_path}",
        ]
    )
