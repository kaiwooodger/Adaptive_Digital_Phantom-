from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import re
import shutil
import zipfile

import numpy as np
import yaml


@dataclass(frozen=True)
class RegTrainingTestingTargetResult:
    target_case_id: str
    staged_pair_count: int
    staged_image_bytes: int
    staged_label_bytes: int
    first_image_path: str
    first_label_path: str
    shape: tuple[int, int, int] | None
    spacing_mm: tuple[float, float, float] | None
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RegTrainingTestingStagingResult:
    dataset_id: str
    zip_path: str
    output_dir: str
    manifest_csv_path: str
    manifest_yaml_path: str
    atlas_png_path: str
    report_path: str
    discovered_target_count: int
    staged_target_count: int
    staged_pair_count: int
    staged_bytes: int
    readiness_status: str
    target_results: tuple[RegTrainingTestingTargetResult, ...]
    notes: tuple[str, ...]


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reg-Training-Testing staging requires nibabel.") from exc
    return nib


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reg-Training-Testing preview generation requires matplotlib.") from exc
    return plt


def _natural_key(value: str) -> tuple[str, int, str]:
    match = re.search(r"(\d+)$", value)
    if match:
        return value[: match.start()], int(match.group(1)), value
    return value, -1, value


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))  # type: ignore[return-value]


def _extract_member(zip_file: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zip_file.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _safe_member_name(member: str, prefix: str) -> str:
    if not member.startswith(prefix):
        raise ValueError(f"Unexpected archive member outside {prefix}: {member}")
    return Path(member).name


def _discover_pairs(zip_file: zipfile.ZipFile) -> dict[str, dict[str, dict[str, Any]]]:
    pattern = re.compile(r"^Training-Testing/(img|label)/(\d{4})/(img|label)(\d{4})-(\d{4})\.nii\.gz$")
    targets: dict[str, dict[str, dict[str, Any]]] = {}
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        match = pattern.match(info.filename)
        if not match:
            continue
        folder_kind, target_case, file_kind, moving_case, suffix_target = match.groups()
        if target_case != suffix_target:
            continue
        if folder_kind != file_kind:
            continue
        pair = targets.setdefault(target_case, {}).setdefault(moving_case, {})
        pair[folder_kind] = info.filename
        pair[f"{folder_kind}_bytes"] = int(info.file_size)
    return targets


def _complete_pairs(pairs: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (moving_case, pair)
        for moving_case, pair in sorted(pairs.items(), key=lambda item: _natural_key(item[0]))
        if pair.get("img") and pair.get("label")
    ]


def _target_size(pairs: dict[str, dict[str, Any]]) -> int:
    return sum(int(pair.get("img_bytes", 0)) + int(pair.get("label_bytes", 0)) for _, pair in _complete_pairs(pairs))


def _select_targets(
    targets: dict[str, dict[str, dict[str, Any]]],
    requested_target_ids: tuple[str, ...],
    max_targets: int | None,
) -> list[str]:
    all_targets = sorted(targets, key=lambda target: (_target_size(targets[target]), _natural_key(target)))
    if requested_target_ids:
        requested = {case_id.strip() for case_id in requested_target_ids if case_id.strip()}
        all_targets = [case_id for case_id in all_targets if case_id in requested]
    if max_targets is not None:
        all_targets = all_targets[: max(0, int(max_targets))]
    return all_targets


def _target_header(image_path: Path) -> tuple[tuple[int, int, int] | None, tuple[float, float, float] | None, tuple[str, ...]]:
    nib = _import_nibabel()
    notes: list[str] = []
    try:
        image = nib.load(str(image_path))
        shape = tuple(int(value) for value in image.shape[:3])
        spacing = _spacing_from_affine(image.affine)
        return shape, spacing, tuple(notes)
    except Exception as exc:
        notes.append(f"header_read_failed={type(exc).__name__}: {exc}")
        return None, None, tuple(notes)


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "target_case_id",
        "moving_case_id",
        "image_path",
        "label_path",
        "image_size_mb",
        "label_size_mb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest_yaml(path: Path, result: RegTrainingTestingStagingResult) -> None:
    payload = {
        "dataset_id": result.dataset_id,
        "package_type": "reg_training_testing_subset",
        "zip_path": result.zip_path,
        "readiness_status": result.readiness_status,
        "summary": {
            "discovered_target_count": result.discovered_target_count,
            "staged_target_count": result.staged_target_count,
            "staged_pair_count": result.staged_pair_count,
            "staged_size_mb": round(result.staged_bytes / 1024**2, 2),
        },
        "outputs": {
            "manifest_csv": result.manifest_csv_path,
            "manifest_yaml": result.manifest_yaml_path,
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
        },
        "targets": [
            {
                "target_case_id": target.target_case_id,
                "staged_pair_count": target.staged_pair_count,
                "shape": list(target.shape) if target.shape else None,
                "spacing_mm": list(target.spacing_mm) if target.spacing_mm else None,
                "first_image_path": target.first_image_path,
                "first_label_path": target.first_label_path,
                "status": target.status,
                "notes": list(target.notes),
            }
            for target in result.target_results
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: RegTrainingTestingStagingResult) -> None:
    lines = [
        "# Reg-Training-Testing Staging Report",
        "",
        f"Dataset ID: `{result.dataset_id}`",
        f"Readiness status: `{result.readiness_status}`",
        "",
        "## Summary",
        "",
        f"- Discovered target cases: {result.discovered_target_count}",
        f"- Staged target cases: {result.staged_target_count}",
        f"- Staged image/label pairs: {result.staged_pair_count}",
        f"- Staged size: {result.staged_bytes / 1024**2:.1f} MB",
        "",
        "## Targets",
        "",
    ]
    for target in result.target_results:
        spacing = ", ".join(f"{value:.3g}" for value in target.spacing_mm) if target.spacing_mm else "unknown"
        lines.append(
            f"- `{target.target_case_id}`: {target.staged_pair_count} pairs, "
            f"shape={target.shape or 'unknown'}, spacing_mm={spacing}, status=`{target.status}`"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Manifest CSV: `{result.manifest_csv_path}`",
            f"- Manifest YAML: `{result.manifest_yaml_path}`",
            f"- Preview atlas: `{result.atlas_png_path}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_preview_atlas(path: Path, targets: tuple[RegTrainingTestingTargetResult, ...]) -> None:
    if not targets:
        return
    nib = _import_nibabel()
    plt = _import_plotting()
    fig, axes = plt.subplots(len(targets), 2, figsize=(8, 3.2 * len(targets)), dpi=150)
    if len(targets) == 1:
        axes = np.asarray([axes])
    fig.patch.set_facecolor("#f7f1e3")
    for row, target in zip(axes, targets):
        image_ax, label_ax = row
        for ax in row:
            ax.set_facecolor("#f7f1e3")
            ax.axis("off")
        image_ax.set_title(f"Target {target.target_case_id}: CT")
        label_ax.set_title("Propagated label")
        try:
            image = nib.load(target.first_image_path) if target.first_image_path else nib.load(target.first_label_path)
            label = nib.load(target.first_label_path)
            z_index = int(image.shape[2] // 2)
            image_slice = np.asarray(image.dataobj[:, :, z_index], dtype=float)
            label_slice = np.asarray(label.dataobj[:, :, min(z_index, label.shape[2] - 1)], dtype=float)
            low, high = np.nanpercentile(image_slice, [1, 99])
            image_ax.imshow(np.rot90(image_slice), cmap="gray", vmin=low, vmax=high)
            label_ax.imshow(np.rot90(label_slice), cmap="viridis")
        except Exception as exc:
            image_ax.text(0.5, 0.5, f"preview failed\n{type(exc).__name__}", ha="center", va="center")
            label_ax.text(0.5, 0.5, "preview failed", ha="center", va="center")
    fig.suptitle("Reg-Training-Testing Staged Subset", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def stage_reg_training_testing_zip(
    *,
    zip_path: str | Path,
    output_dir: str | Path = "data/processed/reg_training_testing",
    dataset_id: str = "reg_training_testing_subset",
    target_case_ids: tuple[str, ...] = (),
    max_targets: int | None = 3,
    max_pairs_per_target: int | None = None,
    extract_images: bool = True,
    report_path: str | Path = "outputs/reports/reg_training_testing_stage001.md",
) -> RegTrainingTestingStagingResult:
    archive_path = Path(zip_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Reg-Training-Testing zip not found: {zip_path}")

    output_root = Path(output_dir)
    manifest_csv = output_root / f"{dataset_id}_manifest_v001.csv"
    manifest_yaml = output_root / f"{dataset_id}_manifest_v001.yaml"
    atlas_png = output_root / "previews" / f"{dataset_id}_preview_atlas_v001.png"
    report = Path(report_path)
    rows: list[dict[str, Any]] = []
    target_results: list[RegTrainingTestingTargetResult] = []

    with zipfile.ZipFile(archive_path) as zip_file:
        targets = _discover_pairs(zip_file)
        selected_targets = _select_targets(targets, target_case_ids, max_targets)
        for target_case in selected_targets:
            complete_pairs = _complete_pairs(targets[target_case])
            if max_pairs_per_target is not None:
                complete_pairs = complete_pairs[: max(0, int(max_pairs_per_target))]
            image_bytes = 0
            label_bytes = 0
            first_image_path = ""
            first_label_path = ""
            notes: list[str] = []
            for moving_case, pair in complete_pairs:
                image_member = str(pair["img"])
                label_member = str(pair["label"])
                label_name = _safe_member_name(label_member, f"Training-Testing/label/{target_case}/")
                image_path = Path("")
                label_path = output_root / "labels" / target_case / label_name
                if extract_images:
                    image_name = _safe_member_name(image_member, f"Training-Testing/img/{target_case}/")
                    image_path = output_root / "images" / target_case / image_name
                    _extract_member(zip_file, image_member, image_path)
                _extract_member(zip_file, label_member, label_path)
                if not first_image_path:
                    first_image_path = str(image_path) if extract_images else ""
                    first_label_path = str(label_path)
                image_bytes += int(pair.get("img_bytes", 0)) if extract_images else 0
                label_bytes += int(pair.get("label_bytes", 0))
                rows.append(
                    {
                        "target_case_id": target_case,
                        "moving_case_id": moving_case,
                        "image_path": str(image_path) if extract_images else "",
                        "label_path": str(label_path),
                        "image_size_mb": f"{int(pair.get('img_bytes', 0)) / 1024**2:.3f}",
                        "label_size_mb": f"{int(pair.get('label_bytes', 0)) / 1024**2:.3f}",
                    }
                )
            header_path = Path(first_image_path) if first_image_path else Path(first_label_path)
            shape, spacing, header_notes = _target_header(header_path) if str(header_path) else (None, None, ())
            notes.extend(header_notes)
            status = "staged" if complete_pairs else "no_complete_pairs"
            target_results.append(
                RegTrainingTestingTargetResult(
                    target_case_id=target_case,
                    staged_pair_count=len(complete_pairs),
                    staged_image_bytes=image_bytes,
                    staged_label_bytes=label_bytes,
                    first_image_path=first_image_path,
                    first_label_path=first_label_path,
                    shape=shape,
                    spacing_mm=spacing,
                    status=status,
                    notes=tuple(notes),
                )
            )

    staged_pair_count = sum(target.staged_pair_count for target in target_results)
    staged_bytes = sum(target.staged_image_bytes + target.staged_label_bytes for target in target_results)
    readiness = "registration_subset_ready" if staged_pair_count > 0 else "registration_subset_empty"
    result = RegTrainingTestingStagingResult(
        dataset_id=dataset_id,
        zip_path=str(archive_path),
        output_dir=str(output_root),
        manifest_csv_path=str(manifest_csv),
        manifest_yaml_path=str(manifest_yaml),
        atlas_png_path=str(atlas_png),
        report_path=str(report),
        discovered_target_count=0,
        staged_target_count=len(target_results),
        staged_pair_count=staged_pair_count,
        staged_bytes=staged_bytes,
        readiness_status=readiness,
        target_results=tuple(target_results),
        notes=(
            "archive_is_staged_as_a_disk_safe_subset_not_full_extraction",
            "image_label_pairs_are_grouped_by_target_case_and_moving_case",
            "labels_are_registration_training_testing_propagated_labels_not_native_vessel_labels",
            "image_files_extracted" if extract_images else "label_only_staging_skips_large_image_files",
        ),
    )
    with zipfile.ZipFile(archive_path) as zip_file:
        discovered_target_count = len(_discover_pairs(zip_file))
    result = RegTrainingTestingStagingResult(
        **{**result.__dict__, "discovered_target_count": discovered_target_count}
    )
    _write_manifest_csv(manifest_csv, rows)
    _write_preview_atlas(atlas_png, result.target_results)
    _write_manifest_yaml(manifest_yaml, result)
    _write_report(report, result)
    return result


def format_reg_training_testing_staging_result(result: RegTrainingTestingStagingResult) -> str:
    return "\n".join(
        [
            "Reg-Training-Testing subset staged",
            f"Dataset ID: {result.dataset_id}",
            f"Readiness status: {result.readiness_status}",
            f"Discovered/staged targets: {result.discovered_target_count}/{result.staged_target_count}",
            f"Staged pairs: {result.staged_pair_count}",
            f"Staged size: {result.staged_bytes / 1024**2:.1f} MB",
            f"Manifest: {result.manifest_yaml_path}",
            f"Preview atlas: {result.atlas_png_path}",
            f"Report: {result.report_path}",
        ]
    )
