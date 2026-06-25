from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
from typing import Any

import numpy as np
import yaml

from .validation_intake import DEFAULT_REQUIRED_VESSEL_LABELS


@dataclass(frozen=True)
class VesselLabelMapping:
    source_label: int
    source_name: str
    target_label: int | None
    target_name: str
    source_voxels: int
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class VesselLabelHarmonizationResult:
    case_id: str
    vessel_seg_path: str
    output_dir: str
    mapping_template_csv_path: str
    mapping_summary_csv_path: str
    harmonized_nifti_path: str
    manifest_yaml_path: str
    preview_png_path: str
    report_path: str
    status: str
    source_label_count: int
    mapped_source_label_count: int
    required_vessel_label_count: int
    present_required_vessel_label_count: int
    vessel_label_coverage_percent: float
    missing_required_vessel_labels: tuple[int, ...]
    mappings: tuple[VesselLabelMapping, ...]
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Vessel label harmonization requires matplotlib and nibabel.") from exc
    return plt, nib


def _load_target_labels(path: str | Path | None) -> dict[int, str]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        return {}
    labels: dict[int, str] = {}
    for key, value in data.get("labels", {}).items():
        try:
            labels[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return labels


def _read_mapping_csv(path: str | Path | None) -> dict[int, int]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    mapping: dict[int, int] = {}
    with resolved.open(newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            source_raw = str(row.get("source_label", "")).strip()
            target_raw = str(row.get("target_label", "")).strip()
            if not source_raw or not target_raw:
                continue
            try:
                mapping[int(source_raw)] = int(target_raw)
            except ValueError:
                continue
    return mapping


def _source_label_counts(data: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(data.astype(np.int64), return_counts=True)
    return {int(label): int(count) for label, count in zip(labels.tolist(), counts.tolist()) if int(label) != 0}


def _mapping_rows(
    *,
    source_counts: dict[int, int],
    target_labels: dict[int, str],
    mapping: dict[int, int],
    auto_identity: bool,
) -> tuple[VesselLabelMapping, ...]:
    rows: list[VesselLabelMapping] = []
    for source_label, voxels in sorted(source_counts.items()):
        target_label: int | None = mapping.get(source_label)
        notes: list[str] = []
        if target_label is None and auto_identity and source_label in target_labels:
            target_label = source_label
            notes.append("auto_identity_mapping")
        if target_label is None:
            status = "unmapped"
            target_name = ""
        elif target_label == 0:
            status = "mapped_to_background"
            target_name = "background"
        else:
            status = "mapped"
            target_name = target_labels.get(target_label, "unknown_target_label")
            if target_label not in target_labels:
                notes.append("target_label_not_in_config")
        rows.append(
            VesselLabelMapping(
                source_label=source_label,
                source_name=target_labels.get(source_label, "") if source_label in target_labels else "",
                target_label=target_label,
                target_name=target_name,
                source_voxels=voxels,
                status=status,
                notes=tuple(notes),
            )
        )
    return tuple(rows)


def _write_mapping_template(path: Path, mappings: tuple[VesselLabelMapping, ...], target_labels: dict[int, str]) -> None:
    fields = [
        "source_label",
        "source_name",
        "source_voxels",
        "target_label",
        "target_name",
        "status",
        "notes",
        "available_target_labels_hint",
    ]
    target_hint = ";".join(f"{label}:{name}" for label, name in sorted(target_labels.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in mappings:
            writer.writerow(
                {
                    "source_label": row.source_label,
                    "source_name": row.source_name,
                    "source_voxels": row.source_voxels,
                    "target_label": "" if row.target_label is None else row.target_label,
                    "target_name": row.target_name,
                    "status": row.status,
                    "notes": ";".join(row.notes),
                    "available_target_labels_hint": target_hint,
                }
            )


def _write_mapping_summary(path: Path, mappings: tuple[VesselLabelMapping, ...]) -> None:
    fields = ["source_label", "source_voxels", "target_label", "target_name", "status", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in mappings:
            writer.writerow(
                {
                    "source_label": row.source_label,
                    "source_voxels": row.source_voxels,
                    "target_label": "" if row.target_label is None else row.target_label,
                    "target_name": row.target_name,
                    "status": row.status,
                    "notes": ";".join(row.notes),
                }
            )


def _remap_data(data: np.ndarray, mappings: tuple[VesselLabelMapping, ...], *, unmapped_policy: str) -> np.ndarray:
    source = data.astype(np.int64)
    remapped = np.zeros(source.shape, dtype=np.int16)
    mapped_sources = {row.source_label for row in mappings if row.target_label is not None}
    for row in mappings:
        if row.target_label is None:
            continue
        remapped[source == row.source_label] = int(row.target_label)
    if unmapped_policy == "preserve":
        for row in mappings:
            if row.source_label in mapped_sources:
                continue
            remapped[source == row.source_label] = int(row.source_label)
    return remapped


def _write_nifti(path: Path, data: np.ndarray, reference_image: Any, nib: Any) -> None:
    header = reference_image.header.copy()
    header.set_data_dtype(np.int16)
    image = nib.Nifti1Image(data.astype(np.int16), reference_image.affine, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _required_coverage(data: np.ndarray | None, required_labels: tuple[int, ...]) -> tuple[int, tuple[int, ...], float]:
    if data is None:
        return 0, required_labels, 0.0
    present = set(int(label) for label in np.unique(data.astype(np.int64)).tolist())
    missing = tuple(label for label in required_labels if label not in present)
    present_count = len(required_labels) - len(missing)
    coverage = 100.0 * present_count / max(len(required_labels), 1)
    return present_count, missing, coverage


def _status(
    *,
    mapping_provided: bool,
    auto_identity: bool,
    mapped_count: int,
    missing_required_labels: tuple[int, ...],
) -> str:
    if not mapping_provided and not auto_identity:
        return "template_only_mapping_required"
    if mapped_count == 0:
        return "harmonized_no_valid_mapping"
    if missing_required_labels:
        return "harmonized_partial_missing_required_labels"
    return "harmonized_ready_for_intake"


def _write_preview(path: Path, result: VesselLabelHarmonizationResult) -> None:
    plt, _ = _import_dependencies()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    mapped = sum(row.status == "mapped" for row in result.mappings)
    unmapped = sum(row.status == "unmapped" for row in result.mappings)
    removed = sum(row.status == "mapped_to_background" for row in result.mappings)
    ax.bar(["mapped", "unmapped", "background"], [mapped, unmapped, removed], color=["#15803d", "#b91c1c", "#64748b"])
    ax.set_ylabel("source label count")
    ax.set_title("Source Label Mapping")
    ax.grid(axis="y", alpha=0.25)

    axes[1].axis("off")
    text = "\n".join(
        [
            "Vessel Label Harmonization",
            "",
            f"Case: {result.case_id}",
            f"Status: {result.status}",
            f"Source labels: {result.source_label_count}",
            f"Mapped labels: {result.mapped_source_label_count}",
            f"Required coverage: {result.vessel_label_coverage_percent:.1f}%",
            f"Missing required: {len(result.missing_required_vessel_labels)}",
        ]
    )
    axes[1].text(
        0.02,
        0.98,
        text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        bbox={"facecolor": "#f7f9fb", "edgecolor": "#ccd6dd", "boxstyle": "round,pad=0.6"},
    )
    fig.suptitle("P1 Vessel Label Harmonization", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_manifest(path: Path, result: VesselLabelHarmonizationResult, *, target_label_config: str, mapping_csv_path: str | None) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "p1_vessel_label_harmonization",
        "status": result.status,
        "source_vessel_seg": result.vessel_seg_path,
        "target_label_config": target_label_config,
        "mapping_csv": "" if mapping_csv_path is None else str(mapping_csv_path),
        "summary": {
            "source_label_count": result.source_label_count,
            "mapped_source_label_count": result.mapped_source_label_count,
            "required_vessel_label_count": result.required_vessel_label_count,
            "present_required_vessel_label_count": result.present_required_vessel_label_count,
            "vessel_label_coverage_percent": result.vessel_label_coverage_percent,
            "missing_required_vessel_labels": list(result.missing_required_vessel_labels),
        },
        "outputs": {
            "mapping_template_csv": result.mapping_template_csv_path,
            "mapping_summary_csv": result.mapping_summary_csv_path,
            "harmonized_nifti": result.harmonized_nifti_path,
            "manifest_yaml": result.manifest_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "mappings": [
            {
                "source_label": row.source_label,
                "source_voxels": row.source_voxels,
                "target_label": row.target_label,
                "target_name": row.target_name,
                "status": row.status,
                "notes": list(row.notes),
            }
            for row in result.mappings
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: VesselLabelHarmonizationResult) -> str:
    image_rel = os.path.relpath(result.preview_png_path, start=Path(result.report_path).parent)
    lines = [
        "# P1 Vessel Label Harmonization",
        "",
        f"Case ID: `{result.case_id}`",
        f"Status: `{result.status}`",
        "",
        f"![Vessel label harmonization]({image_rel})",
        "",
        "## Summary",
        "",
        f"- Source labels: {result.source_label_count}",
        f"- Mapped source labels: {result.mapped_source_label_count}",
        f"- Required label coverage: {result.present_required_vessel_label_count}/{result.required_vessel_label_count} ({result.vessel_label_coverage_percent:.1f}%)",
        f"- Missing required labels: `{', '.join(str(label) for label in result.missing_required_vessel_labels) or 'none'}`",
        "",
        "## Outputs",
        "",
        f"- Mapping template CSV: `{result.mapping_template_csv_path}`",
        f"- Mapping summary CSV: `{result.mapping_summary_csv_path}`",
        f"- Harmonized NIfTI: `{result.harmonized_nifti_path or 'not_written'}`",
        f"- Manifest: `{result.manifest_yaml_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        "",
        "## Mapping Rows",
        "",
        "| source | voxels | target | target name | status |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for row in result.mappings:
        target = "" if row.target_label is None else str(row.target_label)
        lines.append(f"| {row.source_label} | {row.source_voxels} | {target} | {row.target_name} | `{row.status}` |")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def format_vessel_label_harmonization_result(result: VesselLabelHarmonizationResult) -> str:
    return _format_report(result)


def harmonize_vessel_labels(
    *,
    vessel_seg_path: str | Path,
    case_id: str,
    output_dir: str | Path = "outputs/digital/vessel_label_harmonization",
    target_label_config: str | Path = "configs/labelmaps/medseg_abdominal_vasculature.yaml",
    mapping_csv_path: str | Path | None = None,
    required_vessel_labels: tuple[int, ...] | None = DEFAULT_REQUIRED_VESSEL_LABELS,
    auto_identity: bool = False,
    unmapped_policy: str = "zero",
    report_path: str | Path | None = None,
) -> VesselLabelHarmonizationResult:
    if unmapped_policy not in {"zero", "preserve"}:
        raise ValueError("unmapped_policy must be 'zero' or 'preserve'")
    plt, nib = _import_dependencies()
    del plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vessel_path = Path(vessel_seg_path)
    image = nib.load(str(vessel_path))
    data = np.asanyarray(image.dataobj)
    source_counts = _source_label_counts(data)
    target_labels = _load_target_labels(target_label_config)
    mapping = _read_mapping_csv(mapping_csv_path)
    mappings = _mapping_rows(
        source_counts=source_counts,
        target_labels=target_labels,
        mapping=mapping,
        auto_identity=auto_identity,
    )
    mapped_count = sum(row.target_label is not None for row in mappings)
    resolved_required = tuple(required_vessel_labels or DEFAULT_REQUIRED_VESSEL_LABELS)
    base = output / case_id
    mapping_template = base.with_name(f"{case_id}_vessel_label_mapping_template_v001.csv")
    mapping_summary = base.with_name(f"{case_id}_vessel_label_mapping_summary_v001.csv")
    harmonized_nifti = base.with_name(f"{case_id}_harmonized_vessel_labels_v001.nii.gz")
    manifest_yaml = base.with_name(f"{case_id}_vessel_label_harmonization_manifest_v001.yaml")
    preview_png = base.with_name(f"{case_id}_vessel_label_harmonization_preview_v001.png")
    report = Path(report_path) if report_path is not None else output / f"{case_id}_vessel_label_harmonization_report_v001.md"

    remapped: np.ndarray | None = None
    harmonized_path = ""
    if mapping_csv_path is not None or auto_identity:
        remapped = _remap_data(data, mappings, unmapped_policy=unmapped_policy)
        if mapped_count > 0:
            _write_nifti(harmonized_nifti, remapped, image, nib)
            harmonized_path = str(harmonized_nifti)
    present_count, missing_labels, coverage = _required_coverage(remapped, resolved_required)
    status = _status(
        mapping_provided=mapping_csv_path is not None,
        auto_identity=auto_identity,
        mapped_count=mapped_count,
        missing_required_labels=missing_labels,
    )
    notes = [
        "harmonization_does_not_register_or_resample_geometry",
        "mapping_template_should_be_reviewed_against_dataset_documentation_before_clinical_claims",
        f"unmapped_policy={unmapped_policy}",
    ]
    if not mapping_csv_path and not auto_identity:
        notes.append("no_mapping_applied_template_only")
    if auto_identity:
        notes.append("auto_identity_only_maps_source_labels_already_present_in_target_label_config")
    result = VesselLabelHarmonizationResult(
        case_id=case_id,
        vessel_seg_path=str(vessel_path),
        output_dir=str(output),
        mapping_template_csv_path=str(mapping_template),
        mapping_summary_csv_path=str(mapping_summary),
        harmonized_nifti_path=harmonized_path,
        manifest_yaml_path=str(manifest_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        status=status,
        source_label_count=len(source_counts),
        mapped_source_label_count=mapped_count,
        required_vessel_label_count=len(resolved_required),
        present_required_vessel_label_count=present_count,
        vessel_label_coverage_percent=coverage,
        missing_required_vessel_labels=missing_labels,
        mappings=mappings,
        notes=tuple(notes),
    )
    _write_mapping_template(mapping_template, mappings, target_labels)
    _write_mapping_summary(mapping_summary, mappings)
    _write_preview(preview_png, result)
    _write_manifest(manifest_yaml, result, target_label_config=str(target_label_config), mapping_csv_path=str(mapping_csv_path) if mapping_csv_path is not None else None)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result
