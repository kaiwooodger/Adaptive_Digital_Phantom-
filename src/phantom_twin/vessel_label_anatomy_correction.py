from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class VesselLabelCorrectionRow:
    label_id: int
    label_name: str
    original_voxels: int
    original_invalid_voxels: int
    kept_voxels: int
    regrown_voxels: int
    final_voxels: int
    final_invalid_voxels: int
    final_bone_voxels: int
    final_outside_body_voxels: int
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class VesselLabelAnatomyCorrectionResult:
    case_id: str
    anatomy_labels_path: str
    vessel_labels_path: str
    output_dir: str
    corrected_vessel_path: str
    label_correction_csv_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    label_count: int
    original_nonzero_voxels: int
    corrected_nonzero_voxels: int
    original_invalid_voxels: int
    corrected_invalid_voxels: int
    original_bone_voxels: int
    corrected_bone_voxels: int
    original_outside_body_voxels: int
    corrected_outside_body_voxels: int
    corrected_label_count: int
    fully_corrected_label_count: int
    partial_label_count: int
    lost_label_count: int
    max_regrow_iterations: int
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from scipy import ndimage  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Vessel-label anatomy correction requires matplotlib, nibabel, and scipy.") from exc
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


def _geometry_match(reference_image: Any, candidate_image: Any) -> bool:
    if tuple(reference_image.shape) != tuple(candidate_image.shape):
        return False
    ref_spacing = reference_image.header.get_zooms()[: len(reference_image.shape)]
    cand_spacing = candidate_image.header.get_zooms()[: len(candidate_image.shape)]
    return bool(
        np.allclose(np.asarray(ref_spacing, dtype=float), np.asarray(cand_spacing, dtype=float), atol=1e-3)
        and np.allclose(np.asarray(reference_image.affine, dtype=float), np.asarray(candidate_image.affine, dtype=float), atol=1e-3)
    )


def _write_nifti(path: Path, data: np.ndarray, reference_image: Any, nib: Any) -> None:
    header = reference_image.header.copy()
    header.set_data_dtype(np.int16)
    image = nib.Nifti1Image(data.astype(np.int16), reference_image.affine, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _write_rows(path: Path, rows: tuple[VesselLabelCorrectionRow, ...]) -> None:
    fields = [
        "label_id",
        "label_name",
        "original_voxels",
        "original_invalid_voxels",
        "kept_voxels",
        "regrown_voxels",
        "final_voxels",
        "final_invalid_voxels",
        "final_bone_voxels",
        "final_outside_body_voxels",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label_id": row.label_id,
                    "label_name": row.label_name,
                    "original_voxels": row.original_voxels,
                    "original_invalid_voxels": row.original_invalid_voxels,
                    "kept_voxels": row.kept_voxels,
                    "regrown_voxels": row.regrown_voxels,
                    "final_voxels": row.final_voxels,
                    "final_invalid_voxels": row.final_invalid_voxels,
                    "final_bone_voxels": row.final_bone_voxels,
                    "final_outside_body_voxels": row.final_outside_body_voxels,
                    "status": row.status,
                    "notes": ";".join(row.notes),
                }
            )


def _slice_index(mask: np.ndarray, axis: int) -> int:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return mask.shape[axis] // 2
    return int(round(float(np.median(coords[:, axis]))))


def _write_preview(path: Path, anatomy: np.ndarray, original: np.ndarray, corrected: np.ndarray, body: np.ndarray, bone: np.ndarray, rows: tuple[VesselLabelCorrectionRow, ...]) -> None:
    plt, _, _ = _import_dependencies()
    original_mask = original != 0
    corrected_mask = corrected != 0
    z_index = _slice_index(original_mask | corrected_mask | bone, 2)
    y_index = _slice_index(original_mask | corrected_mask | bone, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=170)
    fig.patch.set_facecolor("#f8f3e8")
    for ax in axes.ravel():
        ax.set_facecolor("#f8f3e8")
        ax.axis("off")

    def draw_axial(ax, vessel_mask: np.ndarray, title: str) -> None:
        ax.imshow(np.rot90(anatomy[:, :, z_index]), cmap="bone", interpolation="nearest")
        for mask, color, width in (
            (body[:, :, z_index], "#94a3b8", 0.5),
            (bone[:, :, z_index], "#ffffff", 0.9),
            (vessel_mask[:, :, z_index], "#00b4d8", 1.1),
        ):
            if np.any(mask):
                ax.contour(np.rot90(mask.astype(float)), levels=[0.5], colors=[color], linewidths=width)
        ax.set_title(title, fontsize=10)

    def draw_coronal(ax, vessel_mask: np.ndarray, title: str) -> None:
        ax.imshow(np.rot90(anatomy[:, y_index, :]), cmap="bone", interpolation="nearest")
        for mask, color, width in (
            (body[:, y_index, :], "#94a3b8", 0.5),
            (bone[:, y_index, :], "#ffffff", 0.9),
            (vessel_mask[:, y_index, :], "#00b4d8", 1.1),
        ):
            if np.any(mask):
                ax.contour(np.rot90(mask.astype(float)), levels=[0.5], colors=[color], linewidths=width)
        ax.set_title(title, fontsize=10)

    draw_axial(axes[0, 0], original_mask, f"Original axial z={z_index}")
    draw_axial(axes[0, 1], corrected_mask, f"Corrected axial z={z_index}")
    conflict = (original_mask & bone) | (original_mask & ~body)
    draw_axial(axes[0, 2], conflict, "Original conflicts")
    draw_coronal(axes[1, 0], original_mask, f"Original coronal y={y_index}")
    draw_coronal(axes[1, 1], corrected_mask, f"Corrected coronal y={y_index}")

    status_counts = {"corrected": 0, "partial": 0, "lost": 0, "unchanged_valid": 0}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    axes[1, 2].axis("on")
    axes[1, 2].bar(
        list(status_counts),
        [status_counts[key] for key in status_counts],
        color=["#16a34a", "#f59e0b", "#dc2626", "#0ea5e9"],
    )
    axes[1, 2].set_title("Correction Status By Label", fontsize=10)
    axes[1, 2].tick_params(axis="x", rotation=25, labelsize=7)
    axes[1, 2].tick_params(axis="y", labelsize=8)

    fig.suptitle("Organ-Aware Vessel Label Correction", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_manifest(path: Path, result: VesselLabelAnatomyCorrectionResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "p1_vessel_label_anatomy_correction",
        "anatomy_labels_path": result.anatomy_labels_path,
        "vessel_labels_path": result.vessel_labels_path,
        "summary": {
            "label_count": result.label_count,
            "corrected_label_count": result.corrected_label_count,
            "original_nonzero_voxels": result.original_nonzero_voxels,
            "corrected_nonzero_voxels": result.corrected_nonzero_voxels,
            "original_invalid_voxels": result.original_invalid_voxels,
            "corrected_invalid_voxels": result.corrected_invalid_voxels,
            "original_bone_voxels": result.original_bone_voxels,
            "corrected_bone_voxels": result.corrected_bone_voxels,
            "original_outside_body_voxels": result.original_outside_body_voxels,
            "corrected_outside_body_voxels": result.corrected_outside_body_voxels,
            "fully_corrected_label_count": result.fully_corrected_label_count,
            "partial_label_count": result.partial_label_count,
            "lost_label_count": result.lost_label_count,
            "max_regrow_iterations": result.max_regrow_iterations,
        },
        "outputs": {
            "corrected_vessel": result.corrected_vessel_path,
            "label_correction_csv": result.label_correction_csv_path,
            "manifest_yaml": result.manifest_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VesselLabelAnatomyCorrectionResult, rows: tuple[VesselLabelCorrectionRow, ...]) -> str:
    image_rel = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    problem_rows = [row for row in rows if row.status in {"partial", "lost"}]
    lines = [
        "# P1 Vessel Label Anatomy Correction",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        f"![Vessel label correction]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Labels corrected: {result.fully_corrected_label_count}/{result.label_count}",
        f"- Partial labels: {result.partial_label_count}",
        f"- Lost labels: {result.lost_label_count}",
        f"- Original invalid voxels: {result.original_invalid_voxels}",
        f"- Corrected invalid voxels: {result.corrected_invalid_voxels}",
        f"- Bone voxels before / after: {result.original_bone_voxels} / {result.corrected_bone_voxels}",
        f"- Outside-body voxels before / after: {result.original_outside_body_voxels} / {result.corrected_outside_body_voxels}",
        f"- Nonzero voxels before / after: {result.original_nonzero_voxels} / {result.corrected_nonzero_voxels}",
        "",
        "## Outputs",
        "",
        f"- Corrected vessel NIfTI: `{result.corrected_vessel_path}`",
        f"- Label correction CSV: `{result.label_correction_csv_path}`",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        "",
        "## Partial / Lost Labels",
        "",
        "| label | name | status | invalid before | regrown | invalid after | note |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if problem_rows:
        for row in problem_rows:
            lines.append(
                f"| {row.label_id} | {row.label_name} | `{row.status}` | {row.original_invalid_voxels} | {row.regrown_voxels} | "
                f"{row.final_invalid_voxels} | {'; '.join(row.notes)} |"
            )
    else:
        lines.append("| none | none | `corrected` | 0 | 0 | 0 | all labels cleared from forbidden anatomy |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This correction clears vessel voxels from bone and outside-body regions, then regrows labels into nearby allowed body voxels.",
            "- It preserves integer branch labels and CT-grid geometry, but it is not a deformable CTA/CTV registration.",
            "- Downstream flow and RT QA should use the corrected mask only with this limitation attached.",
            "",
            "## Notes",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_vessel_label_anatomy_correction_result(result: VesselLabelAnatomyCorrectionResult) -> str:
    return Path(result.report_path).read_text()


def correct_vessel_label_anatomy(
    *,
    anatomy_labels_path: str | Path,
    vessel_labels_path: str | Path,
    case_id: str = "vessel_label_anatomy_correction",
    output_dir: str | Path = "outputs/digital/vessel_label_anatomy_correction",
    vessel_label_config: str | Path = "configs/labelmaps/medseg_abdominal_vasculature.yaml",
    max_regrow_iterations: int = 8,
    report_path: str | Path | None = None,
) -> VesselLabelAnatomyCorrectionResult:
    _, nib, ndimage = _import_dependencies()
    anatomy_image = nib.load(str(anatomy_labels_path))
    vessel_image = nib.load(str(vessel_labels_path))
    if not _geometry_match(anatomy_image, vessel_image):
        raise ValueError("Anatomy and vessel labels must be co-registered before correction")
    anatomy = np.rint(np.asanyarray(anatomy_image.dataobj)).astype(np.int16)
    vessel = np.rint(np.asanyarray(vessel_image.dataobj)).astype(np.int16)
    body = anatomy != 0
    bone = np.isin(anatomy, (10, 11))
    allowed = body & ~bone
    forbidden = ~allowed
    label_names = _load_label_names(vessel_label_config)
    structure = ndimage.generate_binary_structure(3, 1)

    corrected = vessel.copy()
    original_conflict = (vessel != 0) & forbidden
    original_bone = (vessel != 0) & bone
    original_outside = (vessel != 0) & ~body
    corrected[original_conflict] = 0
    occupied = corrected != 0
    rows: list[VesselLabelCorrectionRow] = []

    for label_id in sorted(int(value) for value in np.unique(vessel).tolist() if int(value) != 0):
        original_mask = vessel == label_id
        original_voxels = int(np.count_nonzero(original_mask))
        original_invalid = int(np.count_nonzero(original_mask & forbidden))
        kept = int(np.count_nonzero(corrected == label_id))
        regrown = 0
        label_mask = corrected == label_id
        notes: list[str] = []
        needed = max(original_invalid - regrown, 0)
        if needed > 0 and not np.any(label_mask):
            notes.append("label_had_no_valid_seed_after_forbidden_voxels_removed")
            candidates = allowed & ~occupied
            if np.any(candidates):
                distance_to_original = ndimage.distance_transform_edt(~original_mask)
                candidate_coords = np.argwhere(candidates)
                candidate_distances = distance_to_original[candidates]
                take = min(needed, len(candidate_coords))
                chosen_order = np.argpartition(candidate_distances, take - 1)[:take] if take < len(candidate_coords) else np.arange(len(candidate_coords))
                chosen = candidate_coords[chosen_order]
                corrected[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = label_id
                occupied[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
                label_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
                regrown += take
                needed = max(original_invalid - regrown, 0)
                notes.append("label_reseeded_to_nearest_allowed_body_voxels")
        iterations = 0
        while needed > 0 and np.any(label_mask) and iterations < max_regrow_iterations:
            iterations += 1
            frontier = ndimage.binary_dilation(label_mask, structure=structure) & allowed & ~occupied
            count = int(np.count_nonzero(frontier))
            if count == 0:
                break
            coords = np.argwhere(frontier)
            take = min(needed, len(coords))
            chosen = coords[:take]
            corrected[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = label_id
            occupied[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
            label_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
            regrown += take
            needed = max(original_invalid - regrown, 0)
        final_mask = corrected == label_id
        final_voxels = int(np.count_nonzero(final_mask))
        final_bone = int(np.count_nonzero(final_mask & bone))
        final_outside = int(np.count_nonzero(final_mask & ~body))
        final_invalid = int(np.count_nonzero(final_mask & forbidden))
        if final_voxels == 0:
            status = "lost"
            notes.append("label_removed_because_no_allowed_body_voxels_were_available")
        elif final_invalid == 0 and regrown >= original_invalid:
            status = "corrected" if original_invalid else "unchanged_valid"
        elif final_invalid == 0:
            status = "partial"
            notes.append("forbidden_voxels_cleared_but_original_label_volume_not_fully_regrown")
        else:
            status = "partial"
            notes.append("some_forbidden_voxels_remain_after_correction")
        if original_invalid and regrown < original_invalid:
            notes.append("regrowth_capacity_limited_by_available_nearby_body_voxels_or_iteration_limit")
        rows.append(
            VesselLabelCorrectionRow(
                label_id=label_id,
                label_name=label_names.get(label_id, f"label_{label_id}"),
                original_voxels=original_voxels,
                original_invalid_voxels=original_invalid,
                kept_voxels=kept,
                regrown_voxels=regrown,
                final_voxels=final_voxels,
                final_invalid_voxels=final_invalid,
                final_bone_voxels=final_bone,
                final_outside_body_voxels=final_outside,
                status=status,
                notes=tuple(notes) or ("within_correction_rules",),
            )
        )

    output = Path(output_dir)
    corrected_path = output / f"{case_id}_corrected_vessel_labels_v001.nii.gz"
    rows_csv = output / f"{case_id}_vessel_label_correction_metrics_v001.csv"
    manifest_yaml = output / f"{case_id}_vessel_label_anatomy_correction_manifest_v001.yaml"
    preview_png = output / f"{case_id}_vessel_label_anatomy_correction_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_vessel_label_anatomy_correction_report_v001.md"
    _write_nifti(corrected_path, corrected, anatomy_image, nib)
    corrected_bone = (corrected != 0) & bone
    corrected_outside = (corrected != 0) & ~body
    corrected_invalid = (corrected != 0) & forbidden
    corrected_labels = set(int(value) for value in np.unique(corrected).tolist())
    corrected_labels.discard(0)
    fully_corrected = sum(row.status in {"corrected", "unchanged_valid"} for row in rows)
    partial = sum(row.status == "partial" for row in rows)
    lost = sum(row.status == "lost" for row in rows)
    result = VesselLabelAnatomyCorrectionResult(
        case_id=case_id,
        anatomy_labels_path=str(anatomy_labels_path),
        vessel_labels_path=str(vessel_labels_path),
        output_dir=str(output),
        corrected_vessel_path=str(corrected_path),
        label_correction_csv_path=str(rows_csv),
        manifest_yaml_path=str(manifest_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        label_count=len(rows),
        original_nonzero_voxels=int(np.count_nonzero(vessel)),
        corrected_nonzero_voxels=int(np.count_nonzero(corrected)),
        original_invalid_voxels=int(np.count_nonzero(original_conflict)),
        corrected_invalid_voxels=int(np.count_nonzero(corrected_invalid)),
        original_bone_voxels=int(np.count_nonzero(original_bone)),
        corrected_bone_voxels=int(np.count_nonzero(corrected_bone)),
        original_outside_body_voxels=int(np.count_nonzero(original_outside)),
        corrected_outside_body_voxels=int(np.count_nonzero(corrected_outside)),
        corrected_label_count=len(corrected_labels),
        fully_corrected_label_count=int(fully_corrected),
        partial_label_count=int(partial),
        lost_label_count=int(lost),
        max_regrow_iterations=int(max_regrow_iterations),
        notes=(
            "correction_preserves_integer_vessel_labels_and_CT_grid_geometry",
            "forbidden_regions_are_bone_labels_10_11_and_outside_body_label_0",
            "regrowth_is_local_morphological_cleanup_not_deformable_CTA_CTV_registration",
        ),
    )
    _write_rows(rows_csv, tuple(rows))
    _write_preview(preview_png, anatomy, vessel, corrected, body, bone, tuple(rows))
    _write_manifest(manifest_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result, tuple(rows)))
    return result
