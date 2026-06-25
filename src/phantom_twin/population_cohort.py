from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re
from typing import Any

import numpy as np

from .statistical_anatomy import (
    GROUP_DEFINITIONS,
    ShapeGroupStats,
    _all_group_stats,
    _bbox_mm,
    _case_feature_vector,
    _fit_feature_pca,
    _import_dependencies,
    _load_label_like_reference,
    _load_yaml,
    _population_case_ids,
    _register_labels_to_reference,
    _resolve_path,
    _slice_indices,
    _voxel_volume_cm3,
    _write_nifti,
)


@dataclass(frozen=True)
class CohortCaseResult:
    case_id: str
    source_path: str
    registered_label_path: str
    preview_png_path: str
    body_volume_cm3: float
    waist_cm: float
    bbox_mm: tuple[float, float, float]
    body_dice_to_reference: float
    body_overlap_fraction: float
    registration_scale: tuple[float, float, float]
    registration_translation_voxels: tuple[float, float, float]
    missing_groups: tuple[str, ...]
    qc_status: str
    qc_notes: tuple[str, ...]


@dataclass(frozen=True)
class PopulationCohortResult:
    cohort_id: str
    output_dir: str
    reference_labels_path: str
    case_count: int
    shape_mode_count: int
    manifest_csv_path: str
    registration_qc_csv_path: str
    group_metrics_csv_path: str
    feature_matrix_csv_path: str
    pca_loadings_csv_path: str
    shape_model_npz_path: str
    atlas_png_path: str
    spec_yaml_path: str
    report_path: str
    registered_label_paths: tuple[str, ...]
    case_results: tuple[CohortCaseResult, ...]
    notes: tuple[str, ...]


def _safe_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("._-") or "case"


def _body_dice(a: np.ndarray, b: np.ndarray) -> float:
    a_count = int(a.sum())
    b_count = int(b.sum())
    denominator = a_count + b_count
    if denominator == 0:
        return 1.0
    return float(2.0 * int((a & b).sum()) / denominator)


def _body_overlap_fraction(registered_body: np.ndarray, reference_body: np.ndarray) -> float:
    registered_count = int(registered_body.sum())
    if registered_count == 0:
        return 0.0
    return float(int((registered_body & reference_body).sum()) / registered_count)


def _group_presence_notes(
    reference_stats: dict[str, ShapeGroupStats],
    case_stats: dict[str, ShapeGroupStats],
    essential_group_ids: tuple[str, ...],
) -> tuple[str, ...]:
    missing = []
    for group_id in essential_group_ids:
        if reference_stats[group_id].present and not case_stats[group_id].present:
            missing.append(group_id)
    return tuple(missing)


def _case_qc_status(
    body_dice: float,
    overlap_fraction: float,
    missing_groups: tuple[str, ...],
    min_body_dice: float,
    min_body_overlap: float,
) -> tuple[str, tuple[str, ...]]:
    notes: list[str] = []
    status = "pass"
    if body_dice < min_body_dice:
        notes.append(f"body_dice_below_threshold:{body_dice:.3f}<{min_body_dice:.3f}")
        status = "warn"
    if overlap_fraction < min_body_overlap:
        notes.append(f"body_overlap_below_threshold:{overlap_fraction:.3f}<{min_body_overlap:.3f}")
        status = "warn"
    if missing_groups:
        notes.append("missing_groups:" + ",".join(missing_groups))
        status = "warn"
    if overlap_fraction <= 0.05:
        status = "fail"
    if not notes:
        notes.append("registration_qc_passed")
    return status, tuple(notes)


def _write_case_preview(
    path: Path,
    reference_labels: np.ndarray,
    registered_labels: np.ndarray,
    reference_body: np.ndarray,
    registered_body: np.ndarray,
    regions: list[dict[str, Any]],
    case_result: CohortCaseResult,
) -> None:
    plt, ListedColormap, Patch, *_ = _import_dependencies()
    colors = [str(region.get("color", "#000000")) for region in sorted(regions, key=lambda item: int(item["index"]))]
    vmax = max(int(region["index"]) for region in regions)
    cmap = ListedColormap(colors)
    _, _, z_index = _slice_indices(reference_body | registered_body)
    x_index, y_index, _ = _slice_indices(registered_body)
    views = [
        ("Axial", reference_labels[:, :, z_index], registered_labels[:, :, z_index]),
        ("Coronal", reference_labels[:, y_index, :], registered_labels[:, y_index, :]),
        ("Sagittal", reference_labels[x_index, :, :], registered_labels[x_index, :, :]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), dpi=160)
    fig.patch.set_facecolor("#f6f1e8")
    for ax in axes.ravel():
        ax.axis("off")
        ax.set_facecolor("#f6f1e8")
    for col, (title, reference_view, registered_view) in enumerate(views):
        axes[0, col].imshow(np.rot90(reference_view), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[0, col].set_title(f"Reference {title}", fontsize=9, color="#13202a")
        axes[1, col].imshow(np.rot90(registered_view), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[1, col].set_title(f"Registered {title}", fontsize=9, color="#13202a")
    handles = [
        Patch(facecolor="#ffd166", label="adipose"),
        Patch(facecolor="#d95d39", label="muscle"),
        Patch(facecolor="#48cae4", label="lungs"),
        Patch(facecolor="#9d4edd", label="liver"),
        Patch(facecolor="#f72585", label="kidneys"),
        Patch(facecolor="#e9ecef", label="bone"),
        Patch(facecolor="#0077b6", label="vascular fluid"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.50), fontsize=7)
    fig.suptitle(f"Population Cohort Registration QC: {case_result.case_id}", fontsize=13, color="#13202a")
    fig.text(
        0.06,
        0.02,
        f"QC: {case_result.qc_status}; body Dice {case_result.body_dice_to_reference:.3f}; "
        f"overlap {case_result.body_overlap_fraction:.3f}; waist {case_result.waist_cm:.1f} cm; "
        f"volume {case_result.body_volume_cm3 / 1000.0:.2f} L",
        fontsize=8.5,
        color="#1e2a32",
    )
    fig.tight_layout(rect=(0, 0.04, 0.90, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_atlas(
    path: Path,
    reference_labels: np.ndarray,
    case_labels: list[np.ndarray],
    case_results: tuple[CohortCaseResult, ...],
    regions: list[dict[str, Any]],
    max_cases: int,
) -> None:
    plt, ListedColormap, *_ = _import_dependencies()
    colors = [str(region.get("color", "#000000")) for region in sorted(regions, key=lambda item: int(item["index"]))]
    vmax = max(int(region["index"]) for region in regions)
    cmap = ListedColormap(colors)
    display_count = min(len(case_results), max_cases)
    rows = display_count + 1
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, max(3.2, 2.25 * rows)), dpi=160)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.patch.set_facecolor("#f6f1e8")
    reference_body = reference_labels > 0
    rx, ry, rz = _slice_indices(reference_body)
    reference_views = [
        ("Reference axial", reference_labels[:, :, rz]),
        ("Reference coronal", reference_labels[:, ry, :]),
        ("Reference sagittal", reference_labels[rx, :, :]),
    ]
    for col, (title, view) in enumerate(reference_views):
        axes[0, col].imshow(np.rot90(view), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[0, col].set_title(title, fontsize=9, color="#13202a")
        axes[0, col].axis("off")
    for row, (labels, result) in enumerate(zip(case_labels[:display_count], case_results[:display_count], strict=True), start=1):
        body = labels > 0
        x, y, z = _slice_indices(body)
        views = [
            labels[:, :, z],
            labels[:, y, :],
            labels[x, :, :],
        ]
        for col, view in enumerate(views):
            axes[row, col].imshow(np.rot90(view), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
            axes[row, col].axis("off")
        axes[row, 0].set_title(
            f"{result.case_id}: {result.qc_status}, waist {result.waist_cm:.1f} cm, Dice {result.body_dice_to_reference:.3f}",
            fontsize=8,
            color="#13202a",
        )
    fig.suptitle("Population Cohort Registration Atlas", fontsize=14, color="#13202a")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_manifest(path: Path, result: PopulationCohortResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "case_id",
                "source_path",
                "registered_label_path",
                "preview_png_path",
                "body_volume_cm3",
                "waist_cm",
                "bbox_x_mm",
                "bbox_y_mm",
                "bbox_z_mm",
                "qc_status",
            ]
        )
        for item in result.case_results:
            writer.writerow(
                [
                    item.case_id,
                    item.source_path,
                    item.registered_label_path,
                    item.preview_png_path,
                    f"{item.body_volume_cm3:.6f}",
                    f"{item.waist_cm:.6f}",
                    *[f"{value:.6f}" for value in item.bbox_mm],
                    item.qc_status,
                ]
            )


def _write_registration_qc(path: Path, result: PopulationCohortResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "case_id",
                "qc_status",
                "body_dice_to_reference",
                "body_overlap_fraction",
                "registration_scale_x",
                "registration_scale_y",
                "registration_scale_z",
                "registration_translation_x_vox",
                "registration_translation_y_vox",
                "registration_translation_z_vox",
                "missing_groups",
                "qc_notes",
            ]
        )
        for item in result.case_results:
            writer.writerow(
                [
                    item.case_id,
                    item.qc_status,
                    f"{item.body_dice_to_reference:.6f}",
                    f"{item.body_overlap_fraction:.6f}",
                    *[f"{value:.6f}" for value in item.registration_scale],
                    *[f"{value:.6f}" for value in item.registration_translation_voxels],
                    " ".join(item.missing_groups),
                    " | ".join(item.qc_notes),
                ]
            )


def _write_group_metrics(
    path: Path,
    case_ids: tuple[str, ...],
    group_stats_by_case: tuple[dict[str, ShapeGroupStats], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "case_id",
                "group_id",
                "group_name",
                "present",
                "voxel_count",
                "volume_cm3",
                "centroid_x_mm",
                "centroid_y_mm",
                "centroid_z_mm",
                "bbox_x_mm",
                "bbox_y_mm",
                "bbox_z_mm",
                "waist_cm",
            ]
        )
        for case_id, stats_map in zip(case_ids, group_stats_by_case, strict=True):
            for group_id, name, *_ in GROUP_DEFINITIONS:
                stats = stats_map[group_id]
                writer.writerow(
                    [
                        case_id,
                        group_id,
                        name,
                        int(stats.present),
                        stats.voxel_count,
                        f"{stats.volume_cm3:.6f}",
                        *[f"{value:.6f}" for value in stats.centroid_mm],
                        *[f"{value:.6f}" for value in stats.bbox_mm],
                        "" if stats.waist_cm is None else f"{stats.waist_cm:.6f}",
                    ]
                )


def _write_feature_matrix(path: Path, case_ids: tuple[str, ...], feature_names: list[str], feature_matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["case_id", *feature_names])
        for case_id, row in zip(case_ids, feature_matrix, strict=True):
            writer.writerow([case_id, *[f"{float(value):.9g}" for value in row]])


def _write_pca_loadings(path: Path, feature_names: list[str], components: np.ndarray, singular_values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mode_index", "singular_value", "feature_name", "loading"])
        for mode_index, component in enumerate(components, start=1):
            singular = float(singular_values[mode_index - 1]) if mode_index - 1 < len(singular_values) else 0.0
            for feature_name, loading in zip(feature_names, component, strict=True):
                writer.writerow([mode_index, f"{singular:.9g}", feature_name, f"{float(loading):.9g}"])


def _write_spec(path: Path, result: PopulationCohortResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "cohort_id": result.cohort_id,
        "package_type": "population_registration_cohort",
        "reference_labels": result.reference_labels_path,
        "case_count": result.case_count,
        "shape_mode_count": result.shape_mode_count,
        "outputs": {
            "manifest_csv": result.manifest_csv_path,
            "registration_qc_csv": result.registration_qc_csv_path,
            "group_metrics_csv": result.group_metrics_csv_path,
            "feature_matrix_csv": result.feature_matrix_csv_path,
            "pca_loadings_csv": result.pca_loadings_csv_path,
            "shape_model_npz": result.shape_model_npz_path,
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
            "registered_label_paths": list(result.registered_label_paths),
        },
        "cases": [
            {
                "case_id": item.case_id,
                "source_path": item.source_path,
                "registered_label_path": item.registered_label_path,
                "preview_png": item.preview_png_path,
                "qc_status": item.qc_status,
                "body_dice_to_reference": item.body_dice_to_reference,
                "body_overlap_fraction": item.body_overlap_fraction,
                "missing_groups": list(item.missing_groups),
            }
            for item in result.case_results
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: PopulationCohortResult) -> str:
    pass_count = sum(1 for item in result.case_results if item.qc_status == "pass")
    warn_count = sum(1 for item in result.case_results if item.qc_status == "warn")
    fail_count = sum(1 for item in result.case_results if item.qc_status == "fail")
    lines = [
        "# Population Registration Cohort QA Stage 001",
        "",
        f"Cohort ID: `{result.cohort_id}`",
        "",
        "## Cohort Summary",
        "",
        f"- Cases staged: {result.case_count}",
        f"- QC pass/warn/fail: {pass_count}/{warn_count}/{fail_count}",
        f"- PCA shape modes available: {result.shape_mode_count}",
        f"- Reference labels: `{result.reference_labels_path}`",
        "",
        "## Outputs",
        "",
        f"- Manifest CSV: `{Path(result.manifest_csv_path).name}`",
        f"- Registration QC CSV: `{Path(result.registration_qc_csv_path).name}`",
        f"- Group metrics CSV: `{Path(result.group_metrics_csv_path).name}`",
        f"- PCA-ready feature matrix CSV: `{Path(result.feature_matrix_csv_path).name}`",
        f"- PCA loadings CSV: `{Path(result.pca_loadings_csv_path).name}`",
        f"- Shape model NPZ: `{Path(result.shape_model_npz_path).name}`",
        f"- Cohort atlas PNG: `{Path(result.atlas_png_path).name}`",
        f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Case QC",
        "",
        "| case | status | body Dice | overlap | waist cm | body L | missing groups |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result.case_results:
        lines.append(
            f"| {item.case_id} | {item.qc_status} | {item.body_dice_to_reference:.3f} | "
            f"{item.body_overlap_fraction:.3f} | {item.waist_cm:.1f} | "
            f"{item.body_volume_cm3 / 1000.0:.2f} | {', '.join(item.missing_groups) if item.missing_groups else 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This cohort builder prepares staged segmented CT cases for statistical anatomy modeling.",
            "- Registered NIfTI files preserve the project material-label convention and reference grid.",
            "- PCA-ready tables are feature descriptors; dense diffeomorphic registration is still a later upgrade.",
            "- Stable population modes require several independent, label-harmonized CT cases.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_population_cohort(
    combined_spec_path: str | Path,
    population_label_paths: tuple[str | Path, ...],
    output_dir: str | Path = "outputs/digital/population_cohort",
    cohort_id: str = "ct_org_population_cohort",
    population_case_ids: tuple[str, ...] | None = None,
    max_modes: int = 6,
    min_body_dice: float = 0.55,
    min_body_overlap: float = 0.60,
    max_atlas_cases: int = 24,
    essential_group_ids: tuple[str, ...] = ("lungs", "liver", "kidneys", "bone"),
    report_path: str | Path | None = "outputs/reports/population_cohort_stage001.md",
) -> PopulationCohortResult:
    if not population_label_paths:
        raise ValueError("At least one population label map is required")
    _, _, _, nib, ndimage, _ = _import_dependencies()
    spec_path = Path(combined_spec_path)
    spec = _load_yaml(spec_path)
    outputs = spec.get("outputs", {})
    regions = list(spec.get("regions", []))
    if not isinstance(outputs, dict) or not isinstance(regions, list):
        raise ValueError("Combined spec must contain outputs and regions")

    reference_labels_path = _resolve_path(str(outputs["blood_material_labels"]), spec_path)
    reference_image = nib.load(str(reference_labels_path))
    reference_labels = np.rint(np.asanyarray(reference_image.dataobj)).astype(np.int16)
    reference_body = reference_labels > 0
    if not np.any(reference_body):
        raise ValueError("Reference combined phantom has an empty body")

    spacing_mm = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
    voxel_volume = _voxel_volume_cm3(spacing_mm)
    reference_stats = _all_group_stats(reference_labels, spacing_mm, voxel_volume)
    case_ids = _population_case_ids(tuple(population_label_paths), population_case_ids)

    output = Path(output_dir)
    registered_dir = output / "registered_labels"
    previews_dir = output / "case_previews"
    registered_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[CohortCaseResult] = []
    registered_label_paths: list[str] = []
    registered_label_arrays: list[np.ndarray] = []
    group_stats_by_case: list[dict[str, ShapeGroupStats]] = []
    feature_names: list[str] | None = None
    feature_rows: list[np.ndarray] = []

    for raw_case_id, label_path in zip(case_ids, population_label_paths, strict=True):
        case_id = _safe_id(raw_case_id)
        loaded_labels = _load_label_like_reference(label_path, reference_image, nib, ndimage)
        registered_labels, scale, translation = _register_labels_to_reference(loaded_labels, reference_body, ndimage)
        registered_path = registered_dir / f"{cohort_id}_{case_id}_registered_material_labels_v001.nii.gz"
        _write_nifti(registered_path, registered_labels.astype(np.int16), reference_image, nib)

        stats = _all_group_stats(registered_labels, spacing_mm, voxel_volume)
        names, feature_row = _case_feature_vector(stats)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Population feature schema changed unexpectedly")
        feature_rows.append(feature_row)
        group_stats_by_case.append(stats)

        body = registered_labels > 0
        body_stats = stats["body"]
        dice = _body_dice(body, reference_body)
        overlap = _body_overlap_fraction(body, reference_body)
        missing_groups = _group_presence_notes(reference_stats, stats, essential_group_ids)
        qc_status, qc_notes = _case_qc_status(dice, overlap, missing_groups, min_body_dice, min_body_overlap)
        preview_path = previews_dir / f"{cohort_id}_{case_id}_registration_qc_v001.png"

        case_result = CohortCaseResult(
            case_id=case_id,
            source_path=str(label_path),
            registered_label_path=str(registered_path),
            preview_png_path=str(preview_path),
            body_volume_cm3=body_stats.volume_cm3,
            waist_cm=float(body_stats.waist_cm or 0.0),
            bbox_mm=body_stats.bbox_mm,
            body_dice_to_reference=dice,
            body_overlap_fraction=overlap,
            registration_scale=scale,
            registration_translation_voxels=translation,
            missing_groups=missing_groups,
            qc_status=qc_status,
            qc_notes=qc_notes,
        )
        _write_case_preview(preview_path, reference_labels, registered_labels, reference_body, body, regions, case_result)
        case_results.append(case_result)
        registered_label_paths.append(str(registered_path))
        registered_label_arrays.append(registered_labels)

    assert feature_names is not None
    feature_matrix = np.vstack(feature_rows)
    mean, std, components, singular_values = _fit_feature_pca(feature_matrix, max_modes=max_modes)

    manifest_out = output / f"{cohort_id}_population_manifest_v001.csv"
    qc_out = output / f"{cohort_id}_registration_qc_v001.csv"
    group_metrics_out = output / f"{cohort_id}_group_metrics_v001.csv"
    feature_matrix_out = output / f"{cohort_id}_pca_feature_matrix_v001.csv"
    pca_loadings_out = output / f"{cohort_id}_pca_loadings_v001.csv"
    shape_model_out = output / f"{cohort_id}_pca_shape_model_v001.npz"
    atlas_out = output / f"{cohort_id}_registration_qc_atlas_v001.png"
    spec_out = output / f"{cohort_id}_population_cohort_spec_v001.yaml"

    notes = [
        "population_cohort_stage001_affine_body_registration",
        "feature_matrix_is_ready_for_statistical_anatomy_morphing",
        "dense_diffeomorphic_registration_not_yet_enabled",
    ]
    if len(population_label_paths) < 3:
        notes.append("limited_population_size_use_three_or_more_cases_for_stable_pca_modes")
    if any(item.qc_status != "pass" for item in case_results):
        notes.append("one_or_more_cases_have_registration_qc_warnings")

    result = PopulationCohortResult(
        cohort_id=cohort_id,
        output_dir=str(output),
        reference_labels_path=str(reference_labels_path),
        case_count=len(case_results),
        shape_mode_count=int(components.shape[0]),
        manifest_csv_path=str(manifest_out),
        registration_qc_csv_path=str(qc_out),
        group_metrics_csv_path=str(group_metrics_out),
        feature_matrix_csv_path=str(feature_matrix_out),
        pca_loadings_csv_path=str(pca_loadings_out),
        shape_model_npz_path=str(shape_model_out),
        atlas_png_path=str(atlas_out),
        spec_yaml_path=str(spec_out),
        report_path=str(report_path) if report_path is not None else str(output / f"{cohort_id}_population_cohort_report_v001.md"),
        registered_label_paths=tuple(registered_label_paths),
        case_results=tuple(case_results),
        notes=tuple(notes),
    )

    _write_manifest(manifest_out, result)
    _write_registration_qc(qc_out, result)
    _write_group_metrics(group_metrics_out, tuple(item.case_id for item in case_results), tuple(group_stats_by_case))
    _write_feature_matrix(feature_matrix_out, tuple(item.case_id for item in case_results), feature_names, feature_matrix)
    _write_pca_loadings(pca_loadings_out, feature_names, components, singular_values)
    np.savez_compressed(
        shape_model_out,
        feature_names=np.array(feature_names, dtype=object),
        feature_matrix=feature_matrix,
        feature_mean=mean,
        feature_std=std,
        components=components,
        singular_values=singular_values,
        case_ids=np.array([item.case_id for item in case_results], dtype=object),
        registered_label_paths=np.array(registered_label_paths, dtype=object),
    )
    _write_atlas(atlas_out, reference_labels, registered_label_arrays, tuple(case_results), regions, max_cases=max_atlas_cases)
    _write_spec(spec_out, result)
    report = _format_report(result)
    Path(result.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result.report_path).write_text(report)
    return result


def format_population_cohort_result(result: PopulationCohortResult) -> str:
    pass_count = sum(1 for item in result.case_results if item.qc_status == "pass")
    warn_count = sum(1 for item in result.case_results if item.qc_status == "warn")
    fail_count = sum(1 for item in result.case_results if item.qc_status == "fail")
    lines = [
        "# Population Cohort Builder",
        "",
        f"Cohort ID: `{result.cohort_id}`",
        f"Cases staged: {result.case_count}",
        f"QC pass/warn/fail: {pass_count}/{warn_count}/{fail_count}",
        f"PCA shape modes: {result.shape_mode_count}",
        "",
        "## Outputs",
        "",
        f"- Manifest: `{result.manifest_csv_path}`",
        f"- Registration QC: `{result.registration_qc_csv_path}`",
        f"- Group metrics: `{result.group_metrics_csv_path}`",
        f"- PCA feature matrix: `{result.feature_matrix_csv_path}`",
        f"- PCA loadings: `{result.pca_loadings_csv_path}`",
        f"- Shape model: `{result.shape_model_npz_path}`",
        f"- QC atlas: `{result.atlas_png_path}`",
        f"- Spec YAML: `{result.spec_yaml_path}`",
    ]
    return "\n".join(lines)
