from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import numpy as np
import yaml

from .validation_intake import DEFAULT_REQUIRED_VESSEL_LABELS


ORGAN_GROUPS = ("body", "bone", "lungs", "liver", "kidneys", "left_kidney", "right_kidney")


@dataclass(frozen=True)
class VesselLabelAnatomyQAResult:
    case_id: str
    anatomy_labels_path: str
    vessel_labels_path: str
    output_dir: str
    label_metrics_csv_path: str
    organ_stats_csv_path: str
    overlap_csv_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    geometry_status: str
    label_count: int
    required_label_count: int
    present_required_label_count: int
    missing_required_labels: tuple[int, ...]
    pass_count: int
    review_count: int
    fail_count: int
    outside_body_label_count: int
    bone_overlap_label_count: int
    organ_distance_review_count: int
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from scipy import ndimage  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Vessel-label anatomy QA requires matplotlib, nibabel, and scipy.") from exc
    return plt, nib, ndimage


def _load_label_names(path: str | Path | None) -> dict[int, str]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        return {}
    names: dict[int, str] = {}
    for key, value in data.get("labels", {}).items():
        try:
            names[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return names


def _spacing(image: Any) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _geometry_status(reference_image: Any, candidate_image: Any) -> str:
    ref_shape = tuple(int(value) for value in reference_image.shape)
    cand_shape = tuple(int(value) for value in candidate_image.shape)
    ref_spacing = tuple(float(value) for value in reference_image.header.get_zooms()[: len(ref_shape)])
    cand_spacing = tuple(float(value) for value in candidate_image.header.get_zooms()[: len(cand_shape)])
    shape_match = ref_shape == cand_shape
    spacing_match = np.allclose(np.asarray(ref_spacing), np.asarray(cand_spacing), atol=1e-3)
    affine_match = np.allclose(np.asarray(reference_image.affine), np.asarray(candidate_image.affine), atol=1e-3)
    if shape_match and spacing_match and affine_match:
        return "co_registered_to_ct_grid"
    if shape_match and spacing_match:
        return "same_shape_spacing_but_affine_differs"
    return "registration_required_to_ct_grid"


def _mask_for_group(labels: np.ndarray, group: str) -> np.ndarray:
    if group == "body":
        return labels != 0
    if group == "bone":
        return np.isin(labels, (10, 11))
    if group == "lungs":
        return labels == 8
    if group == "liver":
        return labels == 6
    if group == "kidneys":
        return labels == 7
    if group in {"left_kidney", "right_kidney"}:
        kidneys = labels == 7
        coords = np.argwhere(kidneys)
        if len(coords) == 0:
            return kidneys
        split_x = float(np.median(coords[:, 0]))
        x_grid = np.indices(labels.shape, sparse=True)[0]
        if group == "left_kidney":
            return kidneys & (x_grid <= split_x)
        return kidneys & (x_grid > split_x)
    raise ValueError(f"Unknown organ group: {group}")


def _organ_stats(group: str, mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> dict[str, Any]:
    coords = np.argwhere(mask)
    voxel_volume_cm3 = float(np.prod(np.asarray(spacing_mm, dtype=float)) / 1000.0)
    if len(coords) == 0:
        zeros = (0.0, 0.0, 0.0)
        return {
            "organ": group,
            "voxel_count": 0,
            "volume_cm3": 0.0,
            "centroid_mm": zeros,
            "bbox_min_mm": zeros,
            "bbox_max_mm": zeros,
        }
    coords_mm = coords.astype(float) * np.asarray(spacing_mm, dtype=float)
    return {
        "organ": group,
        "voxel_count": int(len(coords)),
        "volume_cm3": float(len(coords) * voxel_volume_cm3),
        "centroid_mm": tuple(float(value) for value in coords_mm.mean(axis=0)),
        "bbox_min_mm": tuple(float(value) for value in coords_mm.min(axis=0)),
        "bbox_max_mm": tuple(float(value) for value in coords_mm.max(axis=0)),
    }


def _expected_target(label_id: int, label_name: str) -> tuple[str | None, float | None, str]:
    lowered = f"{label_id} {label_name}".lower()
    if label_id == 28 or "left renal artery" in lowered:
        return "left_kidney", 45.0, "left renal artery should approach the left kidney region"
    if label_id == 27 or "right renal artery" in lowered:
        return "right_kidney", 45.0, "right renal artery should approach the right kidney region"
    if label_id == 24 or "left renal vein" in lowered:
        return "left_kidney", 50.0, "left renal vein should approach the left kidney region"
    if label_id == 25 or "right renal vein" in lowered:
        return "right_kidney", 50.0, "right renal vein should approach the right kidney region"
    if label_id in {5, 6, 8, 9, 10, 14, 15, 16, 21, 33, 34, 35} or "hepatic" in lowered or "portal" in lowered:
        return "liver", 55.0, "hepatic/portal vessels should approach the liver label"
    if label_id in {13, 22} or "splenic" in lowered or "gastric" in lowered:
        return None, None, "spleen/stomach are not represented in the current material-label anatomy"
    return "body", 0.0, "major vessel should remain inside the body envelope"


def _status_for_row(row: dict[str, Any]) -> tuple[str, str]:
    notes: list[str] = []
    outside = 1.0 - float(row["inside_body_fraction"])
    bone = float(row["inside_bone_fraction"])
    if outside > 0.10:
        return "fail", "more_than_10_percent_of_label_outside_body"
    if outside > 0.02:
        notes.append("more_than_2_percent_of_label_outside_body")
    if bone > 0.10:
        return "fail", "more_than_10_percent_of_label_overlaps_bone"
    if bone > 0.01:
        notes.append("more_than_1_percent_of_label_overlaps_bone")
    expected = str(row["expected_target"])
    threshold = row["expected_threshold_mm"]
    if expected and expected != "body" and threshold != "":
        distance = row.get(f"min_distance_to_{expected}_mm", "")
        if distance == "" or not np.isfinite(float(distance)):
            return "fail", f"expected_target_missing={expected}"
        distance_value = float(distance)
        threshold_value = float(threshold)
        if distance_value > threshold_value * 2.0:
            return "fail", f"{expected}_distance_exceeds_fail_threshold"
        if distance_value > threshold_value:
            notes.append(f"{expected}_distance_exceeds_review_threshold")
    if expected == "":
        notes.append("no_direct_organ_target_available")
    return ("review", ";".join(notes)) if notes else ("pass", "within_engineering_rules")


def _write_organ_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "organ",
        "voxel_count",
        "volume_cm3",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "bbox_min_x_mm",
        "bbox_min_y_mm",
        "bbox_min_z_mm",
        "bbox_max_x_mm",
        "bbox_max_y_mm",
        "bbox_max_z_mm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "organ": row["organ"],
                    "voxel_count": row["voxel_count"],
                    "volume_cm3": f"{row['volume_cm3']:.6f}",
                    "centroid_x_mm": f"{row['centroid_mm'][0]:.6f}",
                    "centroid_y_mm": f"{row['centroid_mm'][1]:.6f}",
                    "centroid_z_mm": f"{row['centroid_mm'][2]:.6f}",
                    "bbox_min_x_mm": f"{row['bbox_min_mm'][0]:.6f}",
                    "bbox_min_y_mm": f"{row['bbox_min_mm'][1]:.6f}",
                    "bbox_min_z_mm": f"{row['bbox_min_mm'][2]:.6f}",
                    "bbox_max_x_mm": f"{row['bbox_max_mm'][0]:.6f}",
                    "bbox_max_y_mm": f"{row['bbox_max_mm'][1]:.6f}",
                    "bbox_max_z_mm": f"{row['bbox_max_mm'][2]:.6f}",
                }
            )


def _write_label_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "label_id",
        "label_name",
        "status",
        "status_note",
        "voxel_count",
        "volume_cm3",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "inside_body_fraction",
        "inside_bone_fraction",
        "inside_liver_fraction",
        "inside_kidneys_fraction",
        "inside_lungs_fraction",
        "expected_target",
        "expected_threshold_mm",
        "min_distance_to_liver_mm",
        "mean_distance_to_liver_mm",
        "min_distance_to_kidneys_mm",
        "mean_distance_to_kidneys_mm",
        "min_distance_to_left_kidney_mm",
        "mean_distance_to_left_kidney_mm",
        "min_distance_to_right_kidney_mm",
        "mean_distance_to_right_kidney_mm",
        "min_distance_to_bone_mm",
        "mean_distance_to_bone_mm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_overlap(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["label_id", "label_name", "organ", "overlap_voxels", "overlap_cm3", "label_fraction", "organ_fraction"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _slice_index(mask: np.ndarray, axis: int) -> int:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return mask.shape[axis] // 2
    return int(round(float(np.median(coords[:, axis]))))


def _write_preview(path: Path, anatomy: np.ndarray, vessel: np.ndarray, masks: dict[str, np.ndarray], label_rows: list[dict[str, Any]]) -> None:
    plt, _, _ = _import_dependencies()
    vessel_mask = vessel != 0
    z_kidney = _slice_index(masks["kidneys"] if masks["kidneys"].any() else vessel_mask, 2)
    z_liver = _slice_index(masks["liver"] if masks["liver"].any() else vessel_mask, 2)
    y_vessel = _slice_index(vessel_mask, 1)
    status_counts = {"pass": 0, "review": 0, "fail": 0}
    for row in label_rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=170)
    fig.patch.set_facecolor("#f8f3e8")
    for ax in axes.ravel():
        ax.set_facecolor("#f8f3e8")
        ax.axis("off")

    def draw_axial(ax, z_index: int, title: str) -> None:
        ax.imshow(np.rot90(anatomy[:, :, z_index]), cmap="bone", interpolation="nearest")
        overlays = (
            (masks["bone"][:, :, z_index], "#ffffff", 0.8),
            (masks["liver"][:, :, z_index], "#8f3ffc", 1.1),
            (masks["kidneys"][:, :, z_index], "#f72585", 1.1),
            (vessel_mask[:, :, z_index], "#00b4d8", 1.1),
        )
        for mask, color, width in overlays:
            if np.any(mask):
                ax.contour(np.rot90(mask.astype(float)), levels=[0.5], colors=[color], linewidths=width)
        ax.set_title(title, fontsize=10)

    draw_axial(axes[0, 0], z_kidney, f"Renal axial z={z_kidney}")
    draw_axial(axes[0, 1], z_liver, f"Hepatic axial z={z_liver}")
    axes[1, 0].imshow(np.rot90(anatomy[:, y_vessel, :]), cmap="bone", interpolation="nearest")
    for mask, color, width in (
        (masks["bone"][:, y_vessel, :], "#ffffff", 0.8),
        (masks["liver"][:, y_vessel, :], "#8f3ffc", 1.1),
        (masks["kidneys"][:, y_vessel, :], "#f72585", 1.1),
        (vessel_mask[:, y_vessel, :], "#00b4d8", 1.1),
    ):
        if np.any(mask):
            axes[1, 0].contour(np.rot90(mask.astype(float)), levels=[0.5], colors=[color], linewidths=width)
    axes[1, 0].set_title(f"Coronal vessel y={y_vessel}", fontsize=10)

    axes[1, 1].axis("on")
    axes[1, 1].bar(list(status_counts), [status_counts["pass"], status_counts["review"], status_counts["fail"]], color=["#16a34a", "#f59e0b", "#dc2626"])
    axes[1, 1].set_title("Vessel Label QA Status", fontsize=10)
    axes[1, 1].set_ylabel("label count")

    fig.suptitle("CT-Grid Vessel Label Anatomy QA", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_manifest(path: Path, result: VesselLabelAnatomyQAResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "p1_vessel_label_anatomy_qa",
        "geometry_status": result.geometry_status,
        "anatomy_labels_path": result.anatomy_labels_path,
        "vessel_labels_path": result.vessel_labels_path,
        "summary": {
            "label_count": result.label_count,
            "required_label_count": result.required_label_count,
            "present_required_label_count": result.present_required_label_count,
            "missing_required_labels": list(result.missing_required_labels),
            "pass_count": result.pass_count,
            "review_count": result.review_count,
            "fail_count": result.fail_count,
            "outside_body_label_count": result.outside_body_label_count,
            "bone_overlap_label_count": result.bone_overlap_label_count,
            "organ_distance_review_count": result.organ_distance_review_count,
        },
        "outputs": {
            "label_metrics_csv": result.label_metrics_csv_path,
            "organ_stats_csv": result.organ_stats_csv_path,
            "overlap_csv": result.overlap_csv_path,
            "manifest_yaml": result.manifest_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VesselLabelAnatomyQAResult, label_rows: list[dict[str, Any]], organ_rows: list[dict[str, Any]]) -> str:
    image_rel = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    review_rows = [row for row in label_rows if row["status"] != "pass"]
    lines = [
        "# P1 Vessel Label Anatomy QA",
        "",
        f"Case ID: `{result.case_id}`",
        f"Geometry status: `{result.geometry_status}`",
        "",
        f"![Vessel label anatomy QA]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Vessel labels evaluated: {result.label_count}",
        f"- Required labels present: {result.present_required_label_count}/{result.required_label_count}",
        f"- Missing required labels: `{', '.join(str(label) for label in result.missing_required_labels) or 'none'}`",
        f"- Pass / review / fail: {result.pass_count} / {result.review_count} / {result.fail_count}",
        f"- Labels with body-envelope concern: {result.outside_body_label_count}",
        f"- Labels with bone-overlap concern: {result.bone_overlap_label_count}",
        f"- Labels with organ-distance concern: {result.organ_distance_review_count}",
        "",
        "## Outputs",
        "",
        f"- Label metrics CSV: `{result.label_metrics_csv_path}`",
        f"- Organ stats CSV: `{result.organ_stats_csv_path}`",
        f"- Overlap CSV: `{result.overlap_csv_path}`",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        "",
        "## Organ Volumes",
        "",
        "| organ | volume cm3 | centroid mm |",
        "| --- | ---: | --- |",
    ]
    for row in organ_rows:
        centroid = ", ".join(f"{value:.1f}" for value in row["centroid_mm"])
        lines.append(f"| `{row['organ']}` | {row['volume_cm3']:.2f} | {centroid} |")
    lines.extend(["", "## Review / Fail Labels", "", "| label | name | status | expected target | body inside % | bone overlap % | note |", "| ---: | --- | --- | --- | ---: | ---: | --- |"])
    if review_rows:
        for row in review_rows[:20]:
            lines.append(
                f"| {row['label_id']} | {row['label_name']} | `{row['status']}` | `{row['expected_target'] or 'none'}` | "
                f"{float(row['inside_body_fraction']) * 100.0:.2f} | {float(row['inside_bone_fraction']) * 100.0:.2f} | {row['status_note']} |"
            )
    else:
        lines.append("| none | none | `pass` | none | 100.00 | 0.00 | no review items |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This QA checks spatial plausibility of labelled vessels on the CT/material-label grid.",
            "- It is an engineering gate for template or registered vessel placement, not proof of patient-specific CTA/CTV anatomical equivalence.",
            "- Left/right kidney checks split the current kidney label by image x-index because the material label map does not encode separate kidney IDs.",
            "- Splenic and gastric vessel checks are limited because the current material labels do not include spleen or stomach labels.",
            "",
            "## Notes",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_vessel_label_anatomy_qa_result(result: VesselLabelAnatomyQAResult) -> str:
    return Path(result.report_path).read_text()


def qa_vessel_label_anatomy(
    *,
    anatomy_labels_path: str | Path,
    vessel_labels_path: str | Path,
    case_id: str = "vessel_label_anatomy_qa",
    output_dir: str | Path = "outputs/validation/vessel_label_anatomy_qa",
    vessel_label_config: str | Path = "configs/labelmaps/medseg_abdominal_vasculature.yaml",
    required_vessel_labels: tuple[int, ...] | None = DEFAULT_REQUIRED_VESSEL_LABELS,
    report_path: str | Path | None = None,
) -> VesselLabelAnatomyQAResult:
    _, nib, ndimage = _import_dependencies()
    anatomy_image = nib.load(str(anatomy_labels_path))
    vessel_image = nib.load(str(vessel_labels_path))
    anatomy = np.rint(np.asanyarray(anatomy_image.dataobj)).astype(np.int16)
    vessel = np.rint(np.asanyarray(vessel_image.dataobj)).astype(np.int16)
    if anatomy.shape != vessel.shape:
        raise ValueError("Anatomy and vessel labels must be on the same grid before anatomy QA")
    spacing_mm = _spacing(anatomy_image)
    geometry = _geometry_status(anatomy_image, vessel_image)
    label_names = _load_label_names(vessel_label_config)
    output = Path(output_dir)
    label_metrics = output / f"{case_id}_vessel_label_anatomy_metrics_v001.csv"
    organ_stats = output / f"{case_id}_organ_stats_v001.csv"
    overlap_csv = output / f"{case_id}_vessel_organ_overlap_v001.csv"
    manifest_yaml = output / f"{case_id}_vessel_label_anatomy_qa_manifest_v001.yaml"
    preview_png = output / f"{case_id}_vessel_label_anatomy_qa_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_vessel_label_anatomy_qa_report_v001.md"

    masks = {group: _mask_for_group(anatomy, group) for group in ORGAN_GROUPS}
    organ_rows = [_organ_stats(group, masks[group], spacing_mm) for group in ORGAN_GROUPS]
    distance_maps: dict[str, np.ndarray] = {}
    for group in ("liver", "kidneys", "left_kidney", "right_kidney", "bone"):
        mask = masks[group]
        if np.any(mask):
            distance_maps[group] = ndimage.distance_transform_edt(~mask, sampling=spacing_mm)
        else:
            distance_maps[group] = np.full(anatomy.shape, np.nan, dtype=np.float32)

    vessel_labels = sorted(int(value) for value in np.unique(vessel).tolist() if int(value) != 0)
    label_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    voxel_volume_cm3 = float(np.prod(np.asarray(spacing_mm)) / 1000.0)
    for label_id in vessel_labels:
        label_mask = vessel == label_id
        coords = np.argwhere(label_mask)
        label_name = label_names.get(label_id, f"label_{label_id}")
        coords_mm = coords.astype(float) * np.asarray(spacing_mm, dtype=float)
        expected, threshold, expected_note = _expected_target(label_id, label_name)
        row: dict[str, Any] = {
            "label_id": label_id,
            "label_name": label_name,
            "voxel_count": int(len(coords)),
            "volume_cm3": f"{len(coords) * voxel_volume_cm3:.6f}",
            "centroid_x_mm": f"{coords_mm[:, 0].mean():.6f}",
            "centroid_y_mm": f"{coords_mm[:, 1].mean():.6f}",
            "centroid_z_mm": f"{coords_mm[:, 2].mean():.6f}",
            "inside_body_fraction": float(np.count_nonzero(masks["body"][label_mask]) / max(len(coords), 1)),
            "inside_bone_fraction": float(np.count_nonzero(masks["bone"][label_mask]) / max(len(coords), 1)),
            "inside_liver_fraction": float(np.count_nonzero(masks["liver"][label_mask]) / max(len(coords), 1)),
            "inside_kidneys_fraction": float(np.count_nonzero(masks["kidneys"][label_mask]) / max(len(coords), 1)),
            "inside_lungs_fraction": float(np.count_nonzero(masks["lungs"][label_mask]) / max(len(coords), 1)),
            "expected_target": expected or "",
            "expected_threshold_mm": "" if threshold is None else f"{threshold:.6f}",
            "expected_note": expected_note,
        }
        for group in ("liver", "kidneys", "left_kidney", "right_kidney", "bone"):
            distances = distance_maps[group][label_mask]
            finite = distances[np.isfinite(distances)]
            row[f"min_distance_to_{group}_mm"] = "" if finite.size == 0 else f"{float(np.min(finite)):.6f}"
            row[f"mean_distance_to_{group}_mm"] = "" if finite.size == 0 else f"{float(np.mean(finite)):.6f}"
        status, note = _status_for_row(row)
        row["status"] = status
        row["status_note"] = note
        label_rows.append(row)
        for group in ("bone", "lungs", "liver", "kidneys"):
            overlap = int(np.count_nonzero(masks[group][label_mask]))
            organ_voxels = int(np.count_nonzero(masks[group]))
            overlap_rows.append(
                {
                    "label_id": label_id,
                    "label_name": label_name,
                    "organ": group,
                    "overlap_voxels": overlap,
                    "overlap_cm3": f"{overlap * voxel_volume_cm3:.6f}",
                    "label_fraction": f"{overlap / max(len(coords), 1):.8f}",
                    "organ_fraction": f"{overlap / max(organ_voxels, 1):.8f}",
                }
            )

    required = tuple(required_vessel_labels or DEFAULT_REQUIRED_VESSEL_LABELS)
    present_required = tuple(label for label in required if label in set(vessel_labels))
    missing_required = tuple(label for label in required if label not in set(vessel_labels))
    pass_count = sum(row["status"] == "pass" for row in label_rows)
    review_count = sum(row["status"] == "review" for row in label_rows)
    fail_count = sum(row["status"] == "fail" for row in label_rows)
    outside_count = sum(1.0 - float(row["inside_body_fraction"]) > 0.02 for row in label_rows)
    bone_count = sum(float(row["inside_bone_fraction"]) > 0.01 for row in label_rows)
    distance_count = sum("distance_exceeds" in str(row["status_note"]) for row in label_rows)
    notes = (
        "qa_requires_anatomy_and_vessel_labels_on_the_same_grid",
        "thresholds_are_engineering_plausibility_gates_not_clinical_validation",
        "spleen_and_stomach_specific_targets_are_not_available_in_current_material_labels",
    )
    result = VesselLabelAnatomyQAResult(
        case_id=case_id,
        anatomy_labels_path=str(anatomy_labels_path),
        vessel_labels_path=str(vessel_labels_path),
        output_dir=str(output),
        label_metrics_csv_path=str(label_metrics),
        organ_stats_csv_path=str(organ_stats),
        overlap_csv_path=str(overlap_csv),
        manifest_yaml_path=str(manifest_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        geometry_status=geometry,
        label_count=len(vessel_labels),
        required_label_count=len(required),
        present_required_label_count=len(present_required),
        missing_required_labels=missing_required,
        pass_count=pass_count,
        review_count=review_count,
        fail_count=fail_count,
        outside_body_label_count=int(outside_count),
        bone_overlap_label_count=int(bone_count),
        organ_distance_review_count=int(distance_count),
        notes=notes,
    )
    _write_label_metrics(label_metrics, label_rows)
    _write_organ_stats(organ_stats, organ_rows)
    _write_overlap(overlap_csv, overlap_rows)
    _write_preview(preview_png, anatomy, vessel, masks, label_rows)
    _write_manifest(manifest_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, label_rows, organ_rows))
    return result
