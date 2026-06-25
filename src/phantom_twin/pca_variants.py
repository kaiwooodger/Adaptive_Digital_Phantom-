from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import numpy as np

from .statistical_anatomy import (
    GROUP_DEFINITIONS,
    StatisticalAnatomyResult,
    _all_group_stats,
    _estimate_waist_cm,
    _import_dependencies,
    _load_yaml,
    _resolve_path,
    _slice_indices,
    _voxel_volume_cm3,
    build_statistical_anatomy_morph,
)


@dataclass(frozen=True)
class PcaVariantMetric:
    variant_id: str
    label: str
    mode_index: int | None
    mode_weight: float
    weights: tuple[float, ...]
    material_labels_path: str
    preview_png_path: str
    body_volume_cm3: float
    waist_cm: float
    bbox_mm: tuple[float, float, float]
    vascular_components: int
    group_volumes_cm3: dict[str, float]


@dataclass(frozen=True)
class PcaModeVariantAtlasResult:
    case_id: str
    output_dir: str
    variant_count: int
    mode_count: int
    amplitude: float
    metrics_csv_path: str
    atlas_png_path: str
    spec_yaml_path: str
    report_path: str
    variants: tuple[PcaVariantMetric, ...]
    notes: tuple[str, ...]


def _cohort_population_paths(cohort_spec_path: str | Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    spec_path = Path(cohort_spec_path)
    spec = _load_yaml(spec_path)
    outputs = spec.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Cohort spec must contain outputs")
    paths = outputs.get("registered_label_paths", [])
    if not isinstance(paths, list) or not paths:
        raise ValueError("Cohort spec outputs.registered_label_paths must be a non-empty list")
    cases = spec.get("cases", [])
    if isinstance(cases, list) and len(cases) == len(paths):
        case_ids = tuple(str(item.get("case_id", f"case_{index}")) for index, item in enumerate(cases))
    else:
        case_ids = tuple(Path(str(path)).name.replace(".nii.gz", "").replace(".nii", "") for path in paths)
    resolved = tuple(str(_resolve_path(str(path), spec_path)) for path in paths)
    return resolved, case_ids


def _variant_definitions(mode_count: int, amplitude: float) -> list[tuple[str, str, int | None, float, tuple[float, ...]]]:
    definitions: list[tuple[str, str, int | None, float, tuple[float, ...]]] = [
        ("mean", "Mean anatomy", None, 0.0, tuple(0.0 for _ in range(mode_count))),
    ]
    for mode_index in range(1, mode_count + 1):
        for sign, label_prefix in [(-1.0, "-"), (1.0, "+")]:
            weights = [0.0 for _ in range(mode_count)]
            weights[mode_index - 1] = sign * amplitude
            suffix = "neg" if sign < 0.0 else "pos"
            definitions.append(
                (
                    f"mode{mode_index:02d}_{suffix}",
                    f"{label_prefix}{amplitude:g} Mode {mode_index}",
                    mode_index,
                    sign * amplitude,
                    tuple(weights),
                )
            )
    return definitions


def _metric_from_result(result: StatisticalAnatomyResult) -> tuple[float, float, tuple[float, float, float], int, dict[str, float]]:
    *_, nib, ndimage, _ = _import_dependencies()
    image = nib.load(result.morphed_blood_material_labels_path)
    labels = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    spacing_mm = tuple(float(value) for value in image.header.get_zooms()[:3])
    voxel_volume = _voxel_volume_cm3(spacing_mm)
    body = labels > 0
    stats = _all_group_stats(labels, spacing_mm, voxel_volume)
    vascular_fluid = np.isin(labels, (14, 15))
    _, components = ndimage.label(vascular_fluid, structure=np.ones((3, 3, 3), dtype=bool))
    return (
        float(body.sum() * voxel_volume),
        _estimate_waist_cm(body, spacing_mm),
        stats["body"].bbox_mm,
        int(components),
        {group_id: stats[group_id].volume_cm3 for group_id, *_ in GROUP_DEFINITIONS},
    )


def _write_metrics_csv(path: Path, variants: tuple[PcaVariantMetric, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group_ids = [group_id for group_id, *_ in GROUP_DEFINITIONS]
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "variant_id",
                "label",
                "mode_index",
                "mode_weight",
                "weights",
                "body_volume_cm3",
                "waist_cm",
                "bbox_x_mm",
                "bbox_y_mm",
                "bbox_z_mm",
                "vascular_components",
                *[f"group_{group_id}_volume_cm3" for group_id in group_ids],
                "material_labels_path",
                "preview_png_path",
            ]
        )
        for item in variants:
            writer.writerow(
                [
                    item.variant_id,
                    item.label,
                    "" if item.mode_index is None else item.mode_index,
                    f"{item.mode_weight:.6f}",
                    " ".join(f"{value:.6f}" for value in item.weights),
                    f"{item.body_volume_cm3:.6f}",
                    f"{item.waist_cm:.6f}",
                    *[f"{value:.6f}" for value in item.bbox_mm],
                    item.vascular_components,
                    *[f"{item.group_volumes_cm3.get(group_id, 0.0):.6f}" for group_id in group_ids],
                    item.material_labels_path,
                    item.preview_png_path,
                ]
            )


def _write_atlas(
    path: Path,
    variants: tuple[PcaVariantMetric, ...],
    regions: list[dict[str, Any]],
) -> None:
    plt, ListedColormap, Patch, nib, *_ = _import_dependencies()
    colors = [str(region.get("color", "#000000")) for region in sorted(regions, key=lambda item: int(item["index"]))]
    cmap = ListedColormap(colors)
    vmax = max(int(region["index"]) for region in regions)
    rows = len(variants)
    fig, axes = plt.subplots(rows, 3, figsize=(13.0, max(3.5, 2.35 * rows)), dpi=150)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.patch.set_facecolor("#f6f1e8")
    for row, item in enumerate(variants):
        labels = np.rint(np.asanyarray(nib.load(item.material_labels_path).dataobj)).astype(np.int16)
        body = labels > 0
        x_index, y_index, z_index = _slice_indices(body)
        views = [
            ("Axial", labels[:, :, z_index]),
            ("Coronal", labels[:, y_index, :]),
            ("Sagittal", labels[x_index, :, :]),
        ]
        for col, (title, view) in enumerate(views):
            axes[row, col].imshow(np.rot90(view), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=9, color="#13202a")
        axes[row, 0].text(
            -0.02,
            1.05,
            f"{item.label}: waist {item.waist_cm:.1f} cm, body {item.body_volume_cm3 / 1000.0:.2f} L",
            transform=axes[row, 0].transAxes,
            fontsize=8,
            color="#13202a",
            ha="left",
            va="bottom",
        )
    handles = [
        Patch(facecolor="#ffd166", label="adipose"),
        Patch(facecolor="#d95d39", label="muscle / soft tissue"),
        Patch(facecolor="#48cae4", label="lungs"),
        Patch(facecolor="#9d4edd", label="liver"),
        Patch(facecolor="#f72585", label="kidneys"),
        Patch(facecolor="#e9ecef", label="bone"),
        Patch(facecolor="#ff9f1c", label="vessel wall"),
        Patch(facecolor="#0077b6", label="vascular fluid"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("PCA Anatomy Mode Variant Atlas", fontsize=15, color="#13202a")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_spec(path: Path, result: PcaModeVariantAtlasResult, cohort_spec_path: str | Path, combined_spec_path: str | Path) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "package_type": "pca_mode_variant_atlas",
        "source_combined_spec": str(combined_spec_path),
        "source_cohort_spec": str(cohort_spec_path),
        "mode_count": result.mode_count,
        "amplitude": result.amplitude,
        "outputs": {
            "metrics_csv": result.metrics_csv_path,
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
        },
        "variants": [
            {
                "variant_id": item.variant_id,
                "label": item.label,
                "mode_index": item.mode_index,
                "mode_weight": item.mode_weight,
                "weights": list(item.weights),
                "material_labels": item.material_labels_path,
                "preview_png": item.preview_png_path,
                "body_volume_cm3": item.body_volume_cm3,
                "waist_cm": item.waist_cm,
                "vascular_components": item.vascular_components,
            }
            for item in result.variants
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: PcaModeVariantAtlasResult) -> str:
    mean = next((item for item in result.variants if item.variant_id == "mean"), None)
    lines = [
        "# PCA Anatomy Mode Variant Atlas Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Variants generated: {result.variant_count}",
        f"- PCA modes compared: {result.mode_count}",
        f"- Mode amplitude: {result.amplitude:.3f}",
        f"- Metrics CSV: `{Path(result.metrics_csv_path).name}`",
        f"- Atlas PNG: `{Path(result.atlas_png_path).name}`",
        f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Variant Metrics",
        "",
        "| variant | mode | waist cm | body L | liver cm3 | lungs cm3 | bone cm3 | vascular cm3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result.variants:
        mode = "mean" if item.mode_index is None else f"{item.mode_weight:+.2f} mode {item.mode_index}"
        vascular_cm3 = item.group_volumes_cm3.get("vascular_fluid", 0.0)
        lines.append(
            f"| {item.variant_id} | {mode} | {item.waist_cm:.1f} | {item.body_volume_cm3 / 1000.0:.2f} | "
            f"{item.group_volumes_cm3.get('liver', 0.0):.1f} | "
            f"{item.group_volumes_cm3.get('lungs', 0.0):.1f} | "
            f"{item.group_volumes_cm3.get('bone', 0.0):.1f} | {vascular_cm3:.1f} |"
        )
    if mean is not None:
        lines.extend(["", "## Deltas From Mean", "", "| variant | waist delta cm | body delta L | liver delta cm3 | lungs delta cm3 |", "| --- | ---: | ---: | ---: | ---: |"])
        for item in result.variants:
            if item.variant_id == "mean":
                continue
            lines.append(
                f"| {item.variant_id} | {item.waist_cm - mean.waist_cm:+.1f} | "
                f"{(item.body_volume_cm3 - mean.body_volume_cm3) / 1000.0:+.2f} | "
                f"{item.group_volumes_cm3.get('liver', 0.0) - mean.group_volumes_cm3.get('liver', 0.0):+.1f} | "
                f"{item.group_volumes_cm3.get('lungs', 0.0) - mean.group_volumes_cm3.get('lungs', 0.0):+.1f} |"
            )
    lines.extend(["", "## Notes"])
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def generate_pca_mode_variants(
    combined_spec_path: str | Path,
    cohort_spec_path: str | Path,
    output_dir: str | Path = "outputs/digital/pca_mode_variants",
    case_id: str = "ct_org_label_population8_pca_modes",
    mode_count: int = 3,
    amplitude: float = 1.0,
    target_height_cm: float | None = None,
    target_weight_kg: float | None = None,
    target_bmi: float | None = None,
    target_waist_cm: float | None = None,
    baseline_height_cm: float = 170.0,
    baseline_bmi: float = 24.0,
    max_modes: int = 6,
    adipose_layer_mm: float = 18.0,
    report_path: str | Path | None = "outputs/reports/pca_mode_variant_atlas_stage001.md",
) -> PcaModeVariantAtlasResult:
    if mode_count < 1:
        raise ValueError("mode_count must be at least 1")
    population_paths, population_case_ids = _cohort_population_paths(cohort_spec_path)
    combined_spec = _load_yaml(combined_spec_path)
    regions = list(combined_spec.get("regions", []))
    if not regions:
        raise ValueError("Combined spec must contain regions for rendering")
    output = Path(output_dir)
    variants_dir = output / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[PcaVariantMetric] = []
    for variant_id, label, mode_index, mode_weight, weights in _variant_definitions(mode_count, amplitude):
        variant_case_id = f"{case_id}_{variant_id}"
        variant_output = variants_dir / variant_id
        result = build_statistical_anatomy_morph(
            combined_spec_path=combined_spec_path,
            population_label_paths=population_paths,
            output_dir=variant_output,
            case_id=variant_case_id,
            population_case_ids=population_case_ids,
            target_height_cm=target_height_cm,
            target_weight_kg=target_weight_kg,
            target_bmi=target_bmi,
            target_waist_cm=target_waist_cm,
            baseline_height_cm=baseline_height_cm,
            baseline_bmi=baseline_bmi,
            mode_weights=weights,
            max_modes=max_modes,
            adipose_layer_mm=adipose_layer_mm,
            report_path=variant_output / f"{variant_case_id}_report_v001.md",
        )
        body_volume, waist, bbox, vascular_components, group_volumes = _metric_from_result(result)
        metrics.append(
            PcaVariantMetric(
                variant_id=variant_id,
                label=label,
                mode_index=mode_index,
                mode_weight=mode_weight,
                weights=weights,
                material_labels_path=result.morphed_blood_material_labels_path,
                preview_png_path=result.preview_png_path,
                body_volume_cm3=body_volume,
                waist_cm=waist,
                bbox_mm=bbox,
                vascular_components=vascular_components,
                group_volumes_cm3=group_volumes,
            )
        )

    metrics_out = output / f"{case_id}_pca_mode_variant_metrics_v001.csv"
    atlas_out = output / f"{case_id}_pca_mode_variant_atlas_v001.png"
    spec_out = output / f"{case_id}_pca_mode_variant_atlas_spec_v001.yaml"
    notes = (
        "mean_and_plus_minus_pca_mode_variants_generated",
        "variants_use_stage001_group_affine_statistical_shape_model",
        "label_only_ct_org_population_supports_shape_pca_not_full_subject_specific_hu_validation",
    )
    result = PcaModeVariantAtlasResult(
        case_id=case_id,
        output_dir=str(output),
        variant_count=len(metrics),
        mode_count=mode_count,
        amplitude=amplitude,
        metrics_csv_path=str(metrics_out),
        atlas_png_path=str(atlas_out),
        spec_yaml_path=str(spec_out),
        report_path=str(report_path) if report_path is not None else str(output / f"{case_id}_pca_mode_variant_atlas_report_v001.md"),
        variants=tuple(metrics),
        notes=notes,
    )
    _write_metrics_csv(metrics_out, result.variants)
    _write_atlas(atlas_out, result.variants, regions)
    _write_spec(spec_out, result, cohort_spec_path, combined_spec_path)
    report = _format_report(result)
    Path(result.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result.report_path).write_text(report)
    return result


def format_pca_mode_variant_atlas_result(result: PcaModeVariantAtlasResult) -> str:
    lines = [
        "# PCA Mode Variant Atlas",
        "",
        f"Case ID: `{result.case_id}`",
        f"Variants generated: {result.variant_count}",
        f"Modes compared: {result.mode_count}",
        f"Amplitude: {result.amplitude:.3f}",
        "",
        "## Outputs",
        "",
        f"- Metrics CSV: `{result.metrics_csv_path}`",
        f"- Atlas PNG: `{result.atlas_png_path}`",
        f"- Spec YAML: `{result.spec_yaml_path}`",
        f"- Report: `{result.report_path}`",
    ]
    return "\n".join(lines)
