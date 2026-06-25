from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import numpy as np
import yaml

from .abdominal_organs import GI_DIRECTORY_FILES, GI_TARGET_ALIASES, GI_TARGETS


TARGET_LABEL_IDS: dict[str, int] = {
    "stomach": 1,
    "duodenum": 2,
    "small_bowel": 3,
    "colon": 4,
    "rectum": 5,
}


@dataclass(frozen=True)
class GISegmentationTargetMetric:
    target: str
    present: bool
    source_labels: tuple[int, ...]
    source_files: tuple[str, ...]
    voxel_count: int
    volume_cm3: float
    centroid_ijk: tuple[float, float, float] | None
    centroid_mm: tuple[float, float, float] | None
    bbox_ijk: tuple[int, int, int, int, int, int] | None
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GISegmentationStagingResult:
    case_id: str
    ct_path: str
    gi_segmentation_path: str
    gi_labelmap_path: str
    output_dir: str
    readiness_status: str
    geometry_status: str
    normalized_gi_segmentation_path: str
    metrics_csv_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    present_targets: tuple[str, ...]
    missing_targets: tuple[str, ...]
    target_metrics: tuple[GISegmentationTargetMetric, ...]
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from matplotlib.colors import ListedColormap  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("GI segmentation staging requires matplotlib and nibabel.") from exc
    return plt, ListedColormap, Patch, nib


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    return data if isinstance(data, dict) else {}


def _normalise_name(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def _target_from_name(value: str) -> str | None:
    normalised = _normalise_name(value)
    for target, aliases in GI_TARGET_ALIASES.items():
        if normalised == target:
            return target
        for alias in aliases:
            alias_norm = _normalise_name(alias)
            if normalised == alias_norm or alias_norm in normalised:
                return target
    return None


def _label_ids_by_target(labelmap: dict[str, Any]) -> dict[str, set[int]]:
    by_target: dict[str, set[int]] = {target: set() for target in GI_TARGETS}
    labels = labelmap.get("labels", {})
    if not isinstance(labels, dict):
        return by_target
    for raw_label, payload in labels.items():
        try:
            label_id = int(raw_label)
        except (TypeError, ValueError):
            continue
        target = None
        name = ""
        if isinstance(payload, dict):
            name = str(payload.get("name", ""))
            raw_target = payload.get("target")
            target = _target_from_name(str(raw_target)) if raw_target is not None else None
        else:
            name = str(payload)
        target = target or _target_from_name(name)
        if target is not None:
            by_target[target].add(label_id)
    return by_target


def _directory_files_by_target(labelmap: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    files = {target: list(GI_DIRECTORY_FILES[target]) for target in GI_TARGETS}
    targets = labelmap.get("targets", {})
    if isinstance(targets, dict):
        for raw_target, payload in targets.items():
            target = _target_from_name(str(raw_target))
            if target is None or not isinstance(payload, dict):
                continue
            for raw_file in payload.get("files", []) or []:
                if raw_file is not None:
                    files[target].append(str(raw_file))
    return {target: tuple(dict.fromkeys(values)) for target, values in files.items()}


def _nifti_candidates(root: Path, names: tuple[str, ...]) -> tuple[Path, ...]:
    matches: list[Path] = []
    for name in names:
        direct = root / name
        if direct.exists() and direct.is_file():
            matches.append(direct)
    if matches:
        return tuple(dict.fromkeys(matches))
    wanted = {Path(name).name for name in names}
    return tuple(sorted(path for path in root.rglob("*") if path.is_file() and path.name in wanted))


def _spacing(image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _geometry_status(image, ct_image) -> str:
    image_shape = tuple(int(value) for value in image.shape[:3])
    ct_shape = tuple(int(value) for value in ct_image.shape[:3])
    shape_match = image_shape == ct_shape
    spacing_match = np.allclose(np.asarray(_spacing(image)), np.asarray(_spacing(ct_image)), atol=1e-3)
    affine_match = np.allclose(np.asarray(image.affine), np.asarray(ct_image.affine), atol=1e-3)
    if shape_match and spacing_match and affine_match:
        return "co_registered_to_ct_grid"
    if shape_match and spacing_match:
        return "same_shape_spacing_but_affine_differs"
    return "registration_required_to_ct_grid"


def _write_nifti(path: Path, data: np.ndarray, reference_image, nib) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _centroid_and_bbox(mask: np.ndarray, affine: np.ndarray) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None, tuple[int, int, int, int, int, int] | None]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None, None, None
    centroid_ijk_array = coords.mean(axis=0)
    centroid_mm_array = np.r_[centroid_ijk_array, 1.0] @ affine.T
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return (
        tuple(float(value) for value in centroid_ijk_array),
        tuple(float(value) for value in centroid_mm_array[:3]),
        (
            int(mins[0]),
            int(mins[1]),
            int(mins[2]),
            int(maxs[0]),
            int(maxs[1]),
            int(maxs[2]),
        ),
    )


def _load_masks_from_directory(source: Path, labelmap: dict[str, Any], ct_image, nib) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], tuple[str, ...]]:
    masks: dict[str, np.ndarray] = {}
    source_files: dict[str, tuple[str, ...]] = {target: () for target in GI_TARGETS}
    notes_by_target: dict[str, tuple[str, ...]] = {target: () for target in GI_TARGETS}
    notes: list[str] = ["gi_source_format=directory_binary_masks"]
    for target, names in _directory_files_by_target(labelmap).items():
        target_notes: list[str] = []
        matched = _nifti_candidates(source, names)
        source_files[target] = tuple(str(path) for path in matched)
        if not matched:
            target_notes.append("target_mask_file_not_found")
            notes_by_target[target] = tuple(target_notes)
            continue
        mask = np.zeros(tuple(int(value) for value in ct_image.shape[:3]), dtype=bool)
        usable_files = 0
        for path in matched:
            image = nib.load(str(path))
            status = _geometry_status(image, ct_image)
            if status != "co_registered_to_ct_grid":
                target_notes.append(f"{path.name}:{status}")
                continue
            mask |= np.asanyarray(image.dataobj) > 0
            usable_files += 1
        if mask.any():
            masks[target] = mask
        target_notes.append(f"usable_mask_files={usable_files}")
        notes_by_target[target] = tuple(target_notes)
    return masks, source_files, notes_by_target, tuple(notes)


def _load_masks_from_multilabel(source: Path, labelmap: dict[str, Any], ct_image, nib) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]], str, tuple[str, ...]]:
    image = nib.load(str(source))
    status = _geometry_status(image, ct_image)
    if status != "co_registered_to_ct_grid":
        return {}, {target: () for target in GI_TARGETS}, status, ("gi_multilabel_geometry_not_ct_registered",)
    data = np.rint(np.asanyarray(image.dataobj)).astype(np.int32)
    masks: dict[str, np.ndarray] = {}
    source_labels: dict[str, tuple[int, ...]] = {}
    for target, label_ids in _label_ids_by_target(labelmap).items():
        sorted_ids = tuple(sorted(label_ids))
        source_labels[target] = sorted_ids
        if not sorted_ids:
            continue
        mask = np.isin(data, np.asarray(sorted_ids, dtype=np.int32))
        if mask.any():
            masks[target] = mask
    return masks, source_labels, status, ("gi_source_format=single_multilabel_nifti",)


def _target_metric(
    *,
    target: str,
    mask: np.ndarray | None,
    source_labels: tuple[int, ...] = (),
    source_files: tuple[str, ...] = (),
    voxel_volume_cm3: float,
    affine: np.ndarray,
    notes: tuple[str, ...] = (),
) -> GISegmentationTargetMetric:
    if mask is None:
        mask = np.zeros((0, 0, 0), dtype=bool)
    voxel_count = int(mask.sum())
    centroid_ijk, centroid_mm, bbox = _centroid_and_bbox(mask, affine) if voxel_count else (None, None, None)
    if voxel_count > 0:
        status = "pass"
        metric_notes = notes
    elif source_files or source_labels:
        status = "review"
        metric_notes = (*notes, "target_configured_but_no_voxels_detected")
    else:
        status = "missing"
        metric_notes = (*notes, "target_not_supplied")
    return GISegmentationTargetMetric(
        target=target,
        present=voxel_count > 0,
        source_labels=source_labels,
        source_files=source_files,
        voxel_count=voxel_count,
        volume_cm3=voxel_count * voxel_volume_cm3,
        centroid_ijk=centroid_ijk,
        centroid_mm=centroid_mm,
        bbox_ijk=bbox,
        status=status,
        notes=metric_notes,
    )


def _write_metrics_csv(path: Path, metrics: tuple[GISegmentationTargetMetric, ...]) -> None:
    fields = [
        "target",
        "present",
        "source_labels",
        "source_files",
        "voxel_count",
        "volume_cm3",
        "centroid_ijk",
        "centroid_mm",
        "bbox_ijk",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "target": metric.target,
                    "present": metric.present,
                    "source_labels": " ".join(str(value) for value in metric.source_labels),
                    "source_files": " ".join(metric.source_files),
                    "voxel_count": metric.voxel_count,
                    "volume_cm3": f"{metric.volume_cm3:.6f}",
                    "centroid_ijk": "" if metric.centroid_ijk is None else " ".join(f"{value:.3f}" for value in metric.centroid_ijk),
                    "centroid_mm": "" if metric.centroid_mm is None else " ".join(f"{value:.3f}" for value in metric.centroid_mm),
                    "bbox_ijk": "" if metric.bbox_ijk is None else " ".join(str(value) for value in metric.bbox_ijk),
                    "status": metric.status,
                    "notes": ";".join(metric.notes),
                }
            )


def _write_manifest(path: Path, result: GISegmentationStagingResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "gi_segmentation_staging",
        "readiness_status": result.readiness_status,
        "geometry_status": result.geometry_status,
        "inputs": {
            "ct": result.ct_path,
            "gi_segmentation": result.gi_segmentation_path,
            "gi_labelmap": result.gi_labelmap_path,
        },
        "outputs": {
            "normalized_gi_segmentation": result.normalized_gi_segmentation_path,
            "metrics_csv": result.metrics_csv_path,
            "manifest_yaml": result.manifest_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "present_targets": list(result.present_targets),
        "missing_targets": list(result.missing_targets),
        "targets": [
            {
                "target": metric.target,
                "present": metric.present,
                "source_labels": list(metric.source_labels),
                "source_files": list(metric.source_files),
                "voxel_count": metric.voxel_count,
                "volume_cm3": metric.volume_cm3,
                "centroid_ijk": None if metric.centroid_ijk is None else list(metric.centroid_ijk),
                "centroid_mm": None if metric.centroid_mm is None else list(metric.centroid_mm),
                "bbox_ijk": None if metric.bbox_ijk is None else list(metric.bbox_ijk),
                "status": metric.status,
                "notes": list(metric.notes),
            }
            for metric in result.target_metrics
        ],
        "recommended_product_case_argument": f"--gi-seg {result.normalized_gi_segmentation_path}",
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(value // 2 for value in mask.shape)
    return tuple(int(round(float(np.median(coords[:, axis])))) for axis in range(3))


def _write_preview(path: Path, ct_hu: np.ndarray, normalized_labels: np.ndarray) -> None:
    plt, ListedColormap, Patch, _ = _import_dependencies()
    mask = normalized_labels > 0
    x_index, y_index, z_index = _slice_indices(mask if mask.any() else np.ones(normalized_labels.shape, dtype=bool))
    colors = ["#000000", "#ffb703", "#f4a261", "#e9c46a", "#2a9d8f", "#8d6a9f"]
    cmap = ListedColormap(colors)
    views = [
        ("Axial", ct_hu[:, :, z_index], normalized_labels[:, :, z_index]),
        ("Coronal", ct_hu[:, y_index, :], normalized_labels[:, y_index, :]),
        ("Sagittal", ct_hu[x_index, :, :], normalized_labels[x_index, :, :]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=160)
    fig.patch.set_facecolor("#f7f1e3")
    for ax, (title, ct_view, label_view) in zip(axes, views, strict=True):
        overlay = np.ma.masked_where(label_view == 0, label_view)
        ax.imshow(np.rot90(np.clip(ct_view, -1000, 1000)), cmap="gray", vmin=-1000, vmax=1000)
        ax.imshow(np.rot90(overlay), cmap=cmap, vmin=0, vmax=5, alpha=0.72, interpolation="nearest")
        ax.set_title(title, fontsize=9, color="#13202a")
        ax.axis("off")
    handles = [
        Patch(facecolor="#ffb703", label="1 stomach"),
        Patch(facecolor="#f4a261", label="2 duodenum"),
        Patch(facecolor="#e9c46a", label="3 small bowel"),
        Patch(facecolor="#2a9d8f", label="4 colon"),
        Patch(facecolor="#8d6a9f", label="5 rectum"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7)
    fig.suptitle("Real GI Segmentation Staging QA", fontsize=13, fontweight="bold", color="#13202a")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, result: GISegmentationStagingResult) -> None:
    lines = [
        "# GI Segmentation Staging QA",
        "",
        f"Case ID: `{result.case_id}`",
        f"Readiness status: `{result.readiness_status}`",
        f"Geometry status: `{result.geometry_status}`",
        "",
        "## Target Coverage",
        "",
        "| target | status | voxels | volume cm3 | centroid mm |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for metric in result.target_metrics:
        centroid = "n/a" if metric.centroid_mm is None else ", ".join(f"{value:.1f}" for value in metric.centroid_mm)
        lines.append(f"| {metric.target} | `{metric.status}` | {metric.voxel_count} | {metric.volume_cm3:.2f} | {centroid} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Normalized GI segmentation: `{result.normalized_gi_segmentation_path}`",
            f"- Metrics CSV: `{result.metrics_csv_path}`",
            f"- Manifest YAML: `{result.manifest_yaml_path}`",
            f"- Preview PNG: `{result.preview_png_path}`",
            "",
            "## Product Handoff",
            "",
            "Use the normalized output as the product-case GI input once readiness is `ready_for_real_gi_replacement`:",
            "",
            "```bash",
            f"--gi-seg {result.normalized_gi_segmentation_path} --gi-labelmap {result.gi_labelmap_path}",
            "```",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def stage_gi_segmentation(
    *,
    ct_path: str | Path,
    gi_segmentation_path: str | Path,
    gi_labelmap_path: str | Path = "configs/labelmaps/gi_tract.yaml",
    output_dir: str | Path = "outputs/digital/gi_segmentation_staging",
    case_id: str = "gi_segmentation_stage001",
    report_path: str | Path | None = "outputs/reports/gi_segmentation_staging_stage001.md",
    require_targets: tuple[str, ...] = ("small_bowel", "colon", "rectum"),
) -> GISegmentationStagingResult:
    _, _, _, nib = _import_dependencies()
    output = Path(output_dir) / case_id
    ct = nib.load(str(ct_path))
    ct_hu = np.asanyarray(ct.dataobj).astype(np.float32)
    labelmap = _load_yaml(gi_labelmap_path)
    source = Path(gi_segmentation_path)
    if not source.exists():
        raise FileNotFoundError(f"GI segmentation input not found: {source}")

    notes: list[str] = []
    if source.is_dir():
        masks, source_files, notes_by_target, load_notes = _load_masks_from_directory(source, labelmap, ct, nib)
        source_labels = {target: () for target in GI_TARGETS}
        if masks:
            geometry_status = "co_registered_to_ct_grid"
        elif any(source_files.get(target) for target in GI_TARGETS):
            geometry_status = "registration_required_to_ct_grid"
        else:
            geometry_status = "not_evaluated_no_supported_gi_files"
        notes.extend(load_notes)
    else:
        masks, source_labels, geometry_status, load_notes = _load_masks_from_multilabel(source, labelmap, ct, nib)
        source_files = {target: (str(source),) if source_labels.get(target) else () for target in GI_TARGETS}
        notes_by_target = {target: () for target in GI_TARGETS}
        notes.extend(load_notes)

    normalized = np.zeros(tuple(int(value) for value in ct.shape[:3]), dtype=np.int16)
    for target in GI_TARGETS:
        mask = masks.get(target)
        if mask is not None and mask.any():
            normalized[mask] = TARGET_LABEL_IDS[target]

    spacing = _spacing(ct)
    voxel_volume_cm3 = float(np.prod(np.asarray(spacing, dtype=float)) / 1000.0)
    metrics = tuple(
        _target_metric(
            target=target,
            mask=masks.get(target),
            source_labels=tuple(source_labels.get(target, ())),
            source_files=tuple(source_files.get(target, ())),
            voxel_volume_cm3=voxel_volume_cm3,
            affine=np.asarray(ct.affine, dtype=float),
            notes=tuple(notes_by_target.get(target, ())),
        )
        for target in GI_TARGETS
    )
    present = tuple(metric.target for metric in metrics if metric.present)
    missing = tuple(target for target in GI_TARGETS if target not in present)
    required_missing = tuple(target for target in require_targets if target not in present)
    if geometry_status == "not_evaluated_no_supported_gi_files":
        readiness = "blocked_no_supported_gi_targets"
        notes.append("no_supported_gi_mask_files_detected")
    elif geometry_status != "co_registered_to_ct_grid":
        readiness = "blocked_registration_required"
        notes.append("gi_segmentation_must_be_resampled_to_primary_ct_grid_before_replacement")
    elif required_missing:
        readiness = "partial_real_gi_replacement_ready"
        notes.append(f"required_targets_missing={','.join(required_missing)}")
    elif present:
        readiness = "ready_for_real_gi_replacement"
    else:
        readiness = "blocked_no_supported_gi_targets"
        notes.append("no_supported_gi_targets_detected")

    normalized_path = output / f"{case_id}_normalized_gi_segmentation_v001.nii.gz"
    metrics_csv = output / f"{case_id}_gi_target_metrics_v001.csv"
    manifest_yaml = output / f"{case_id}_gi_segmentation_staging_manifest_v001.yaml"
    preview_png = output / f"{case_id}_gi_segmentation_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_gi_segmentation_staging_report_v001.md"

    _write_nifti(normalized_path, normalized, ct, nib)
    _write_metrics_csv(metrics_csv, metrics)
    _write_preview(preview_png, ct_hu, normalized)
    result = GISegmentationStagingResult(
        case_id=case_id,
        ct_path=str(ct_path),
        gi_segmentation_path=str(gi_segmentation_path),
        gi_labelmap_path=str(gi_labelmap_path),
        output_dir=str(output),
        readiness_status=readiness,
        geometry_status=geometry_status,
        normalized_gi_segmentation_path=str(normalized_path),
        metrics_csv_path=str(metrics_csv),
        manifest_yaml_path=str(manifest_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        present_targets=present,
        missing_targets=missing,
        target_metrics=metrics,
        notes=tuple(notes),
    )
    _write_manifest(manifest_yaml, result)
    _write_report(report, result)
    return result


def format_gi_segmentation_staging_result(result: GISegmentationStagingResult) -> str:
    return "\n".join(
        [
            "GI Segmentation Staging QA",
            f"Case ID: {result.case_id}",
            f"Readiness: {result.readiness_status}",
            f"Geometry: {result.geometry_status}",
            f"Present targets: {', '.join(result.present_targets) or 'none'}",
            f"Missing targets: {', '.join(result.missing_targets) or 'none'}",
            f"Normalized GI segmentation: {result.normalized_gi_segmentation_path}",
            f"Report: {result.report_path}",
        ]
    )
