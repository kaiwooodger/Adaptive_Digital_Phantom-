from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import yaml


@dataclass(frozen=True)
class ApprovedPcaVariant:
    variant_id: str
    label: str
    release_role: str
    warning_status: str
    mode_index: int | None
    mode_weight: float
    qa_rank: int | None
    qa_score: float | None
    qa_interpretation: str
    qa_notes: tuple[str, ...]
    qa_issues: tuple[str, ...]
    material_labels_path: str
    preview_png_path: str
    body_volume_cm3: float | None
    waist_cm: float | None
    vascular_components: int | None


@dataclass(frozen=True)
class ApprovedPcaPhantomSetResult:
    case_id: str
    output_dir: str
    manifest_yaml_path: str
    metrics_csv_path: str
    preview_png_path: str
    report_path: str
    source_atlas_spec_path: str
    source_qa_decisions_path: str
    source_metrics_csv_path: str | None
    variant_count: int
    approved_modes: tuple[int, ...]
    rejected_modes: tuple[int, ...]
    warning_variant_count: int
    variants: tuple[ApprovedPcaVariant, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _resolve_reference_path(path: str, base_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return base_path.parent / candidate


def _decision_has_warning(decision: dict[str, Any]) -> bool:
    issues = decision.get("issues", [])
    notes = decision.get("notes", [])
    if isinstance(issues, list) and issues:
        return True
    if not isinstance(notes, list):
        return False
    clean_notes = {"mode_within_current_stage001_guardrails"}
    return any(str(note) not in clean_notes for note in notes)


def _release_role(decision: dict[str, Any] | None) -> tuple[str, str]:
    if decision is None:
        return "baseline", "clean"
    has_warning = _decision_has_warning(decision)
    if has_warning:
        return "approved_with_warning", "warning"
    rank = int(decision.get("rank", 999))
    if rank == 1:
        return "approved_primary", "clean"
    if rank == 2:
        return "approved_secondary", "clean"
    return "approved_exploratory", "clean"


def _read_metrics_rows(path: str | Path | None) -> tuple[list[str], dict[str, dict[str, str]]]:
    if path is None:
        return [], {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return [], {}
    with metrics_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = {str(row.get("variant_id", "")): dict(row) for row in reader}
        return list(reader.fieldnames or []), rows


def _variant_lookup(atlas_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = atlas_spec.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Atlas spec must contain a variants list")
    return {str(item.get("variant_id")): item for item in variants if isinstance(item, dict)}


def _decision_lookup(qa_decisions: dict[str, Any]) -> dict[int, dict[str, Any]]:
    decisions = qa_decisions.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("QA decisions YAML must contain a decisions list")
    return {
        int(item["mode_index"]): item
        for item in decisions
        if isinstance(item, dict) and item.get("decision") == "approved" and item.get("mode_index") is not None
    }


def _source_metrics_path(qa_decisions: dict[str, Any], atlas_spec: dict[str, Any], metrics_csv_path: str | Path | None) -> str | None:
    if metrics_csv_path is not None:
        return str(metrics_csv_path)
    qa_metrics = qa_decisions.get("source_metrics_csv")
    if qa_metrics:
        return str(qa_metrics)
    outputs = atlas_spec.get("outputs", {})
    if isinstance(outputs, dict) and outputs.get("metrics_csv"):
        return str(outputs["metrics_csv"])
    return None


def _select_variants(
    atlas_spec: dict[str, Any],
    qa_decisions: dict[str, Any],
    metrics_rows: dict[str, dict[str, str]],
) -> tuple[ApprovedPcaVariant, ...]:
    variants_by_id = _variant_lookup(atlas_spec)
    decisions_by_mode = _decision_lookup(qa_decisions)
    selected_ids = ["mean"]
    for decision in sorted(decisions_by_mode.values(), key=lambda item: int(item.get("rank", 999))):
        selected_ids.extend(str(variant_id) for variant_id in decision.get("variant_ids", []))

    selected: list[ApprovedPcaVariant] = []
    seen: set[str] = set()
    for variant_id in selected_ids:
        if variant_id in seen:
            continue
        seen.add(variant_id)
        variant = variants_by_id.get(variant_id)
        if variant is None:
            raise ValueError(f"Approved variant {variant_id!r} was not found in the atlas spec")
        mode_index = _as_int(variant.get("mode_index"))
        decision = decisions_by_mode.get(mode_index) if mode_index is not None else None
        role, warning_status = _release_role(decision)
        notes = tuple(str(item) for item in (decision or {}).get("notes", []))
        issues = tuple(str(item) for item in (decision or {}).get("issues", []))
        metric_row = metrics_rows.get(variant_id, {})
        selected.append(
            ApprovedPcaVariant(
                variant_id=variant_id,
                label=str(variant.get("label", variant_id)),
                release_role=role,
                warning_status=warning_status,
                mode_index=mode_index,
                mode_weight=float(variant.get("mode_weight", 0.0)),
                qa_rank=None if decision is None else _as_int(decision.get("rank")),
                qa_score=None if decision is None else _as_float(decision.get("score")),
                qa_interpretation="" if decision is None else str(decision.get("interpretation", "")),
                qa_notes=notes,
                qa_issues=issues,
                material_labels_path=str(variant.get("material_labels", metric_row.get("material_labels_path", ""))),
                preview_png_path=str(variant.get("preview_png", metric_row.get("preview_png_path", ""))),
                body_volume_cm3=_as_float(metric_row.get("body_volume_cm3", variant.get("body_volume_cm3"))),
                waist_cm=_as_float(metric_row.get("waist_cm", variant.get("waist_cm"))),
                vascular_components=_as_int(metric_row.get("vascular_components", variant.get("vascular_components"))),
            )
        )
    return tuple(selected)


def _write_metrics_csv(
    path: Path,
    variants: tuple[ApprovedPcaVariant, ...],
    source_fieldnames: list[str],
    source_rows: dict[str, dict[str, str]],
) -> None:
    release_fields = [
        "release_role",
        "warning_status",
        "qa_rank",
        "qa_score",
        "qa_interpretation",
        "qa_notes",
        "qa_issues",
    ]
    fallback_fields = [
        "variant_id",
        "label",
        "mode_index",
        "mode_weight",
        "body_volume_cm3",
        "waist_cm",
        "vascular_components",
        "material_labels_path",
        "preview_png_path",
    ]
    fieldnames = release_fields + (source_fieldnames if source_fieldnames else fallback_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in variants:
            row = dict(source_rows.get(item.variant_id, {}))
            if not row:
                row = {
                    "variant_id": item.variant_id,
                    "label": item.label,
                    "mode_index": "" if item.mode_index is None else item.mode_index,
                    "mode_weight": item.mode_weight,
                    "body_volume_cm3": "" if item.body_volume_cm3 is None else f"{item.body_volume_cm3:.6f}",
                    "waist_cm": "" if item.waist_cm is None else f"{item.waist_cm:.6f}",
                    "vascular_components": "" if item.vascular_components is None else item.vascular_components,
                    "material_labels_path": item.material_labels_path,
                    "preview_png_path": item.preview_png_path,
                }
            row.update(
                {
                    "release_role": item.release_role,
                    "warning_status": item.warning_status,
                    "qa_rank": "" if item.qa_rank is None else item.qa_rank,
                    "qa_score": "" if item.qa_score is None else f"{item.qa_score:.3f}",
                    "qa_interpretation": item.qa_interpretation,
                    "qa_notes": ";".join(item.qa_notes),
                    "qa_issues": ";".join(item.qa_issues),
                }
            )
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_manifest(path: Path, result: ApprovedPcaPhantomSetResult) -> None:
    payload = {
        "case_id": result.case_id,
        "package_type": "approved_pca_phantom_set",
        "copy_policy": "reference_existing_nifti_and_preview_files_without_duplication",
        "source_atlas_spec": result.source_atlas_spec_path,
        "source_qa_decisions": result.source_qa_decisions_path,
        "source_metrics_csv": result.source_metrics_csv_path,
        "approved_mode_indices": list(result.approved_modes),
        "rejected_mode_indices": list(result.rejected_modes),
        "outputs": {
            "manifest_yaml": result.manifest_yaml_path,
            "metrics_csv": result.metrics_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "variants": [
            {
                "variant_id": item.variant_id,
                "label": item.label,
                "release_role": item.release_role,
                "warning_status": item.warning_status,
                "mode_index": item.mode_index,
                "mode_weight": item.mode_weight,
                "qa_rank": item.qa_rank,
                "qa_score": item.qa_score,
                "qa_interpretation": item.qa_interpretation,
                "qa_notes": list(item.qa_notes),
                "qa_issues": list(item.qa_issues),
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


def _write_preview(path: Path, variants: tuple[ApprovedPcaVariant, ...], atlas_spec_path: Path, max_columns: int = 3) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    columns = max(1, min(max_columns, len(variants)))
    rows = (len(variants) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, 3.65 * rows), dpi=120)
    if rows == 1 and columns == 1:
        axes_grid = [[axes]]
    elif rows == 1:
        axes_grid = [list(axes)]
    elif columns == 1:
        axes_grid = [[axis] for axis in axes]
    else:
        axes_grid = axes
    fig.patch.set_facecolor("#f6f1e8")
    for flat_index in range(rows * columns):
        row_index = flat_index // columns
        col_index = flat_index % columns
        ax = axes_grid[row_index][col_index]
        ax.axis("off")
        ax.set_facecolor("#f6f1e8")
        if flat_index >= len(variants):
            continue
        item = variants[flat_index]
        preview_path = _resolve_reference_path(item.preview_png_path, atlas_spec_path)
        if preview_path.exists():
            ax.imshow(mpimg.imread(preview_path))
        else:
            ax.text(0.5, 0.5, "preview missing", ha="center", va="center", fontsize=11, color="#7f1d1d")
        title = f"{item.release_role}\n{item.variant_id}"
        if item.warning_status == "warning":
            title += "\nwarning"
        ax.set_title(title, fontsize=9.5, color="#13202a")
    fig.suptitle("Approved PCA Digital Phantom Set", fontsize=16, color="#13202a")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _format_report(result: ApprovedPcaPhantomSetResult) -> str:
    lines = [
        "# Approved PCA Phantom Set Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Released variants: {result.variant_count}",
        f"- Approved modes: {', '.join(str(item) for item in result.approved_modes) or 'none'}",
        f"- Rejected modes excluded: {', '.join(str(item) for item in result.rejected_modes) or 'none'}",
        f"- Variants with QA warnings: {result.warning_variant_count}",
        "- Copy policy: references only; no NIfTI files were duplicated",
        "",
        "## Release Inventory",
        "",
        "| variant | role | mode | score | waist cm | body L | warning |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result.variants:
        mode = "mean" if item.mode_index is None else f"{item.mode_weight:+.1f} mode {item.mode_index}"
        score = "" if item.qa_score is None else f"{item.qa_score:.1f}"
        waist = "" if item.waist_cm is None else f"{item.waist_cm:.1f}"
        body = "" if item.body_volume_cm3 is None else f"{item.body_volume_cm3 / 1000.0:.2f}"
        warning = "yes" if item.warning_status == "warning" else "no"
        lines.append(
            f"| {item.variant_id} | {item.release_role} | {mode} | {score} | {waist} | {body} | {warning} |"
        )
    lines.extend(["", "## Warning Carry-Forward", ""])
    warning_items = [item for item in result.variants if item.warning_status == "warning"]
    if warning_items:
        for item in warning_items:
            detail = "; ".join((*item.qa_notes, *item.qa_issues)) or "warning flag present"
            lines.append(f"- {item.variant_id}: {detail}")
    else:
        lines.append("- No approved variants carry QA warnings.")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- Manifest YAML: `{Path(result.manifest_yaml_path).name}`")
    lines.append(f"- Release metrics CSV: `{Path(result.metrics_csv_path).name}`")
    lines.append(f"- Preview PNG: `{Path(result.preview_png_path).name}`")
    lines.extend(["", "## Notes"])
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_approved_pca_phantom_set(
    qa_decisions_path: str | Path,
    atlas_spec_path: str | Path,
    output_dir: str | Path = "outputs/digital/approved_pca_phantom_set",
    case_id: str | None = None,
    metrics_csv_path: str | Path | None = None,
    report_path: str | Path | None = "outputs/reports/approved_pca_phantom_set_stage001.md",
    max_preview_columns: int = 3,
) -> ApprovedPcaPhantomSetResult:
    qa_decisions = _load_yaml(qa_decisions_path)
    atlas_spec = _load_yaml(atlas_spec_path)
    resolved_case_id = case_id or str(qa_decisions.get("case_id") or atlas_spec.get("case_id") or "approved_pca_phantom_set")
    source_metrics = _source_metrics_path(qa_decisions, atlas_spec, metrics_csv_path)
    source_fieldnames, source_rows = _read_metrics_rows(source_metrics)
    variants = _select_variants(atlas_spec, qa_decisions, source_rows)
    approved_modes = tuple(int(item) for item in qa_decisions.get("approved_mode_indices", []))
    rejected_modes = tuple(int(item) for item in qa_decisions.get("rejected_mode_indices", []))
    output = Path(output_dir)
    manifest_path = output / f"{resolved_case_id}_approved_pca_phantom_set_manifest_v001.yaml"
    metrics_path = output / f"{resolved_case_id}_approved_pca_phantom_set_metrics_v001.csv"
    preview_path = output / f"{resolved_case_id}_approved_pca_phantom_set_preview_v001.png"
    report_out = Path(report_path) if report_path is not None else output / f"{resolved_case_id}_approved_pca_phantom_set_report_v001.md"
    notes = (
        "release_set_contains_mean_and_approved_plus_minus_pca_mode_variants_only",
        "nifti_volume_paths_are_referenced_not_copied_to_preserve_disk_space",
        "approved_with_warning_variants_should_be_used_for_sensitivity_analysis_not_primary_claims",
    )
    result = ApprovedPcaPhantomSetResult(
        case_id=resolved_case_id,
        output_dir=str(output),
        manifest_yaml_path=str(manifest_path),
        metrics_csv_path=str(metrics_path),
        preview_png_path=str(preview_path),
        report_path=str(report_out),
        source_atlas_spec_path=str(atlas_spec_path),
        source_qa_decisions_path=str(qa_decisions_path),
        source_metrics_csv_path=source_metrics,
        variant_count=len(variants),
        approved_modes=approved_modes,
        rejected_modes=rejected_modes,
        warning_variant_count=sum(1 for item in variants if item.warning_status == "warning"),
        variants=variants,
        notes=notes,
    )
    _write_metrics_csv(metrics_path, variants, source_fieldnames, source_rows)
    _write_manifest(manifest_path, result)
    _write_preview(preview_path, variants, Path(atlas_spec_path), max_columns=max_preview_columns)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(_format_report(result))
    return result


def format_approved_pca_phantom_set_result(result: ApprovedPcaPhantomSetResult) -> str:
    lines = [
        "# Approved PCA Phantom Set",
        "",
        f"Case ID: `{result.case_id}`",
        f"Released variants: {result.variant_count}",
        f"Approved modes: {', '.join(str(item) for item in result.approved_modes) or 'none'}",
        f"Warning variants: {result.warning_variant_count}",
        "",
        "## Outputs",
        "",
        f"- Manifest YAML: `{result.manifest_yaml_path}`",
        f"- Metrics CSV: `{result.metrics_csv_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        f"- Report: `{result.report_path}`",
    ]
    return "\n".join(lines)
