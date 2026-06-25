from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import yaml


@dataclass(frozen=True)
class ProfileCandidateScore:
    variant_id: str
    label: str
    release_role: str
    warning_status: str
    body_volume_cm3: float | None
    waist_cm: float | None
    body_delta_cm3: float | None
    waist_delta_cm: float | None
    normalized_distance: float
    selection_score: float
    material_labels_path: str
    preview_png_path: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class UserProfileAdapterResult:
    case_id: str
    profile_id: str
    output_dir: str
    profile_yaml_path: str
    score_csv_path: str
    preview_png_path: str
    report_path: str
    source_approved_set_manifest_path: str
    source_metrics_csv_path: str | None
    source_combined_spec_path: str | None
    target_height_cm: float
    target_weight_kg: float | None
    target_bmi: float
    target_waist_cm: float
    target_body_volume_cm3: float
    baseline_height_cm: float
    baseline_bmi: float
    baseline_waist_cm: float
    baseline_body_volume_cm3: float
    selected_variant_id: str
    selected_score: float
    fit_status: str
    release_waist_range_cm: tuple[float, float]
    release_body_volume_range_cm3: tuple[float, float]
    candidates: tuple[ProfileCandidateScore, ...]
    recommended_commands: tuple[str, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(raw_path: str | Path | None, reference_path: Path) -> Path | None:
    if raw_path is None or str(raw_path) == "":
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = reference_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _read_metric_rows(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None or str(path) == "":
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    with resolved.open(newline="") as csvfile:
        return {str(row.get("variant_id", "")): dict(row) for row in csv.DictReader(csvfile)}


def _source_metrics_path(
    approved_manifest: dict[str, Any],
    approved_manifest_path: Path,
    metrics_csv_path: str | Path | None,
) -> Path | None:
    if metrics_csv_path is not None:
        return _resolve_path(metrics_csv_path, approved_manifest_path)
    outputs = approved_manifest.get("outputs", {})
    if isinstance(outputs, dict) and outputs.get("metrics_csv"):
        return _resolve_path(str(outputs["metrics_csv"]), approved_manifest_path)
    if approved_manifest.get("source_metrics_csv"):
        return _resolve_path(str(approved_manifest["source_metrics_csv"]), approved_manifest_path)
    return None


def _source_combined_spec_path(approved_manifest: dict[str, Any], approved_manifest_path: Path) -> Path | None:
    atlas_path = _resolve_path(approved_manifest.get("source_atlas_spec"), approved_manifest_path)
    if atlas_path is None or not atlas_path.exists():
        return None
    atlas = _load_yaml(atlas_path)
    return _resolve_path(atlas.get("source_combined_spec"), atlas_path)


def _variant_value(
    variant: dict[str, Any],
    metric_row: dict[str, str],
    field: str,
) -> float | None:
    metric_value = _as_float(metric_row.get(field))
    if metric_value is not None:
        return metric_value
    return _as_float(variant.get(field))


def _baseline_variant(variants: list[dict[str, Any]], metrics: dict[str, dict[str, str]]) -> dict[str, Any]:
    for item in variants:
        if isinstance(item, dict) and item.get("release_role") == "baseline":
            return item
    for item in variants:
        if isinstance(item, dict) and str(item.get("variant_id")) == "mean":
            return item
    if not variants:
        raise ValueError("Approved PCA manifest must contain at least one variant")
    return variants[0]


def _derive_target_bmi(target_height_cm: float, target_weight_kg: float | None, target_bmi: float | None, baseline_bmi: float) -> float:
    if target_bmi is not None:
        return float(target_bmi)
    if target_weight_kg is not None:
        height_m = target_height_cm / 100.0
        if height_m <= 0.0:
            raise ValueError("target_height_cm must be positive")
        return float(target_weight_kg / (height_m**2))
    return float(baseline_bmi)


def _derive_target_waist(target_waist_cm: float | None, target_bmi: float, baseline_bmi: float, baseline_waist_cm: float) -> float:
    if target_waist_cm is not None:
        return float(target_waist_cm)
    if baseline_bmi <= 0.0:
        raise ValueError("baseline_bmi must be positive")
    return float(baseline_waist_cm * math.sqrt(max(target_bmi, 1e-6) / baseline_bmi))


def _derive_target_body_volume(
    baseline_body_volume_cm3: float,
    baseline_waist_cm: float,
    target_waist_cm: float,
    baseline_height_cm: float,
    target_height_cm: float,
) -> float:
    waist_scale = target_waist_cm / max(baseline_waist_cm, 1e-6)
    height_scale = target_height_cm / max(baseline_height_cm, 1e-6)
    return float(baseline_body_volume_cm3 * waist_scale * waist_scale * height_scale)


def _candidate_scores(
    variants: list[dict[str, Any]],
    metrics: dict[str, dict[str, str]],
    manifest_path: Path,
    target_body_volume_cm3: float,
    target_waist_cm: float,
    waist_tolerance_cm: float,
    body_volume_tolerance_cm3: float,
    warning_penalty: float,
) -> tuple[ProfileCandidateScore, ...]:
    candidates: list[ProfileCandidateScore] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        variant_id = str(variant.get("variant_id", ""))
        row = metrics.get(variant_id, {})
        waist_cm = _variant_value(variant, row, "waist_cm")
        body_volume_cm3 = _variant_value(variant, row, "body_volume_cm3")
        material_labels = str(variant.get("material_labels", row.get("material_labels_path", "")))
        preview_png = str(variant.get("preview_png", row.get("preview_png_path", "")))
        notes: list[str] = []
        if waist_cm is None:
            notes.append("missing_waist_metric")
        if body_volume_cm3 is None:
            notes.append("missing_body_volume_metric")
        if str(variant.get("warning_status", row.get("warning_status", ""))) == "warning":
            notes.append("approved_variant_has_qa_warning")

        waist_delta = None if waist_cm is None else waist_cm - target_waist_cm
        body_delta = None if body_volume_cm3 is None else body_volume_cm3 - target_body_volume_cm3
        waist_error = 3.0 if waist_delta is None else abs(waist_delta) / max(waist_tolerance_cm, 1e-6)
        body_error = 3.0 if body_delta is None else abs(body_delta) / max(body_volume_tolerance_cm3, 1e-6)
        distance = float(math.sqrt(waist_error**2 + body_error**2))
        penalty = warning_penalty if "approved_variant_has_qa_warning" in notes else 0.0
        if "missing_waist_metric" in notes or "missing_body_volume_metric" in notes:
            penalty += 25.0
        score = max(0.0, 100.0 / (1.0 + distance) - penalty)
        candidates.append(
            ProfileCandidateScore(
                variant_id=variant_id,
                label=str(variant.get("label", row.get("label", variant_id))),
                release_role=str(variant.get("release_role", row.get("release_role", ""))),
                warning_status=str(variant.get("warning_status", row.get("warning_status", ""))),
                body_volume_cm3=body_volume_cm3,
                waist_cm=waist_cm,
                body_delta_cm3=body_delta,
                waist_delta_cm=waist_delta,
                normalized_distance=distance,
                selection_score=score,
                material_labels_path=str(_resolve_path(material_labels, manifest_path) or ""),
                preview_png_path=str(_resolve_path(preview_png, manifest_path) or ""),
                notes=tuple(notes),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -item.selection_score,
                item.warning_status == "warning",
                item.normalized_distance,
                item.variant_id,
            ),
        )
    )


def _range(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (float(min(values)), float(max(values)))


def _fit_status(
    selected: ProfileCandidateScore,
    target_waist_cm: float,
    target_body_volume_cm3: float,
    waist_range: tuple[float, float],
    body_range: tuple[float, float],
    waist_tolerance_cm: float,
    body_volume_tolerance_cm3: float,
) -> str:
    outside_waist = target_waist_cm < waist_range[0] - waist_tolerance_cm or target_waist_cm > waist_range[1] + waist_tolerance_cm
    outside_body = (
        target_body_volume_cm3 < body_range[0] - body_volume_tolerance_cm3
        or target_body_volume_cm3 > body_range[1] + body_volume_tolerance_cm3
    )
    if outside_waist or outside_body:
        return "target_outside_current_release_envelope"
    waist_matched = selected.waist_delta_cm is not None and abs(selected.waist_delta_cm) <= waist_tolerance_cm
    body_matched = selected.body_delta_cm3 is not None and abs(selected.body_delta_cm3) <= body_volume_tolerance_cm3
    if waist_matched and body_matched:
        return "matched_to_approved_variant"
    return "nearest_approved_variant_only"


def _recommended_commands(
    result_stub: dict[str, Any],
    source_combined_spec: str | None,
    approved_manifest_path: str,
) -> tuple[str, ...]:
    case_id = str(result_stub["case_id"])
    selected = str(result_stub["selected_variant_id"])
    target_height = float(result_stub["target_height_cm"])
    target_bmi = float(result_stub["target_bmi"])
    target_waist = float(result_stub["target_waist_cm"])
    commands = [
        (
            "python -m phantom_twin.cli build-variant-rerun-harness "
            f"--approved-set-manifest {approved_manifest_path} "
            f"--variant-id {selected} "
            f"--case-id {case_id}_{selected}"
        )
    ]
    if source_combined_spec:
        commands.append(
            "python -m phantom_twin.cli build-anthropometric-torso-morph "
            f"--combined-spec {source_combined_spec} "
            f"--case-id {case_id}_bmi{target_bmi:.1f}_waist{target_waist:.1f} "
            f"--target-height-cm {target_height:.3f} "
            f"--target-bmi {target_bmi:.3f} "
            f"--target-waist-cm {target_waist:.3f}"
        )
    commands.append(
        "# After selecting or morphing anatomy, rerun vascular graph deformation, voxelization, flow, RT planning, and spatial RT-flow QA for that profile."
    )
    return tuple(commands)


def _write_score_csv(path: Path, candidates: tuple[ProfileCandidateScore, ...]) -> None:
    fieldnames = [
        "rank",
        "variant_id",
        "label",
        "release_role",
        "warning_status",
        "selection_score",
        "normalized_distance",
        "waist_cm",
        "waist_delta_cm",
        "body_volume_cm3",
        "body_delta_cm3",
        "material_labels_path",
        "preview_png_path",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "variant_id": item.variant_id,
                    "label": item.label,
                    "release_role": item.release_role,
                    "warning_status": item.warning_status,
                    "selection_score": f"{item.selection_score:.3f}",
                    "normalized_distance": f"{item.normalized_distance:.6f}",
                    "waist_cm": "" if item.waist_cm is None else f"{item.waist_cm:.6f}",
                    "waist_delta_cm": "" if item.waist_delta_cm is None else f"{item.waist_delta_cm:.6f}",
                    "body_volume_cm3": "" if item.body_volume_cm3 is None else f"{item.body_volume_cm3:.6f}",
                    "body_delta_cm3": "" if item.body_delta_cm3 is None else f"{item.body_delta_cm3:.6f}",
                    "material_labels_path": item.material_labels_path,
                    "preview_png_path": item.preview_png_path,
                    "notes": ";".join(item.notes),
                }
            )


def _write_preview(path: Path, result: UserProfileAdapterResult) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("User-profile adapter preview generation requires matplotlib.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (scatter_ax, score_ax) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 0.85]})
    fig.suptitle(f"User Profile Anatomy Adapter\n{result.profile_id}", fontsize=15, fontweight="bold")

    for item in result.candidates:
        if item.waist_cm is None or item.body_volume_cm3 is None:
            continue
        color = "#2a9d8f" if item.warning_status != "warning" else "#f4a261"
        marker = "o" if item.variant_id != result.selected_variant_id else "D"
        size = 72 if item.variant_id != result.selected_variant_id else 120
        scatter_ax.scatter(item.waist_cm, item.body_volume_cm3 / 1000.0, s=size, marker=marker, color=color, edgecolor="#1f2933")
        scatter_ax.annotate(item.variant_id, (item.waist_cm, item.body_volume_cm3 / 1000.0), xytext=(5, 4), textcoords="offset points", fontsize=8)
    scatter_ax.scatter(
        result.target_waist_cm,
        result.target_body_volume_cm3 / 1000.0,
        s=180,
        marker="*",
        color="#e63946",
        edgecolor="#1f2933",
        label="target profile",
    )
    scatter_ax.set_xlabel("waist estimate (cm)")
    scatter_ax.set_ylabel("body volume proxy (L)")
    scatter_ax.set_title("Target vs approved anatomy envelope")
    scatter_ax.grid(True, color="#d8dee9", linewidth=0.7)
    scatter_ax.legend(loc="best", fontsize=8)

    top = result.candidates[: min(8, len(result.candidates))]
    labels = [item.variant_id for item in reversed(top)]
    scores = [item.selection_score for item in reversed(top)]
    colors = ["#2a9d8f" if item.warning_status != "warning" else "#f4a261" for item in reversed(top)]
    score_ax.barh(labels, scores, color=colors, edgecolor="#1f2933")
    score_ax.set_xlim(0, 100)
    score_ax.set_xlabel("selection score")
    score_ax.set_title(f"Best fit: {result.selected_variant_id}\n{result.fit_status.replace('_', ' ')}")
    score_ax.grid(True, axis="x", color="#d8dee9", linewidth=0.7)
    for index, score in enumerate(scores):
        score_ax.text(min(score + 1.0, 99.0), index, f"{score:.1f}", va="center", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_profile_yaml(path: Path, result: UserProfileAdapterResult) -> None:
    payload = {
        "case_id": result.case_id,
        "profile_id": result.profile_id,
        "package_type": "user_profile_anatomy_adapter",
        "source_approved_set_manifest": result.source_approved_set_manifest_path,
        "source_metrics_csv": result.source_metrics_csv_path,
        "source_combined_spec": result.source_combined_spec_path,
        "target": {
            "height_cm": result.target_height_cm,
            "weight_kg": result.target_weight_kg,
            "bmi": result.target_bmi,
            "waist_cm": result.target_waist_cm,
            "body_volume_proxy_cm3": result.target_body_volume_cm3,
        },
        "baseline_reference": {
            "height_cm": result.baseline_height_cm,
            "bmi": result.baseline_bmi,
            "waist_cm": result.baseline_waist_cm,
            "body_volume_cm3": result.baseline_body_volume_cm3,
        },
        "selection": {
            "selected_variant_id": result.selected_variant_id,
            "selected_score": result.selected_score,
            "fit_status": result.fit_status,
            "release_waist_range_cm": list(result.release_waist_range_cm),
            "release_body_volume_range_cm3": list(result.release_body_volume_range_cm3),
        },
        "outputs": {
            "profile_yaml": result.profile_yaml_path,
            "score_csv": result.score_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "recommended_commands": list(result.recommended_commands),
        "candidates": [
            {
                "rank": index,
                "variant_id": item.variant_id,
                "label": item.label,
                "release_role": item.release_role,
                "warning_status": item.warning_status,
                "selection_score": item.selection_score,
                "normalized_distance": item.normalized_distance,
                "waist_cm": item.waist_cm,
                "waist_delta_cm": item.waist_delta_cm,
                "body_volume_cm3": item.body_volume_cm3,
                "body_delta_cm3": item.body_delta_cm3,
                "material_labels": item.material_labels_path,
                "preview_png": item.preview_png_path,
                "notes": list(item.notes),
            }
            for index, item in enumerate(result.candidates, start=1)
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: UserProfileAdapterResult) -> None:
    top_rows = result.candidates[: min(8, len(result.candidates))]
    lines = [
        "# User Profile Anatomy Adapter",
        "",
        f"Case ID: `{result.case_id}`",
        f"Profile ID: `{result.profile_id}`",
        "",
        "## Target Profile",
        "",
        f"- Height: {result.target_height_cm:.1f} cm",
        f"- Weight: {'not provided' if result.target_weight_kg is None else f'{result.target_weight_kg:.1f} kg'}",
        f"- BMI: {result.target_bmi:.2f}",
        f"- Waist: {result.target_waist_cm:.2f} cm",
        f"- Target torso/body-volume proxy: {result.target_body_volume_cm3 / 1000.0:.2f} L",
        "",
        "## Selection",
        "",
        f"- Selected approved variant: `{result.selected_variant_id}`",
        f"- Selection score: {result.selected_score:.1f} / 100",
        f"- Fit status: `{result.fit_status}`",
        f"- Approved waist envelope: {result.release_waist_range_cm[0]:.2f}-{result.release_waist_range_cm[1]:.2f} cm",
        f"- Approved body-volume envelope: {result.release_body_volume_range_cm3[0] / 1000.0:.2f}-{result.release_body_volume_range_cm3[1] / 1000.0:.2f} L",
        "",
        "## Candidate Ranking",
        "",
        "| rank | variant | role | warning | score | waist delta cm | body delta L |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(top_rows, start=1):
        waist_delta = "" if item.waist_delta_cm is None else f"{item.waist_delta_cm:.2f}"
        body_delta = "" if item.body_delta_cm3 is None else f"{item.body_delta_cm3 / 1000.0:.2f}"
        lines.append(
            f"| {rank} | {item.variant_id} | {item.release_role} | {item.warning_status} | "
            f"{item.selection_score:.1f} | {waist_delta} | {body_delta} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if result.fit_status == "target_outside_current_release_envelope":
        lines.append(
            "- The requested anthropometry is outside the current approved PCA release envelope, so this package gives a nearest-neighbor anatomy rather than a fully matched body geometry."
        )
        lines.append(
            "- Use the recommended morph/rerun commands to generate a new target-specific digital phantom and then repeat vascular, flow, and RT QA on that morphed volume."
        )
    elif result.fit_status == "matched_to_approved_variant":
        lines.append("- The selected PCA anatomy falls within the current waist/body-volume tolerances for this target profile.")
    else:
        lines.append("- The selected PCA anatomy is the closest approved release variant, but a target-specific morph would better match this profile.")
    lines.extend(
        [
            "- This is not a subject-specific anatomical equivalent; it is a population/PCA-guided digital phantom selection and morph-planning layer.",
            "",
            "## Recommended Commands",
            "",
        ]
    )
    for command in result.recommended_commands:
        lines.append(f"```bash\n{command}\n```")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Profile YAML: `{result.profile_yaml_path}`",
            f"- Candidate score CSV: `{result.score_csv_path}`",
            f"- Preview PNG: `{result.preview_png_path}`",
            f"- Source approved manifest: `{result.source_approved_set_manifest_path}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "profile"


def build_user_profile_adapter(
    approved_set_manifest_path: str | Path,
    output_dir: str | Path = "outputs/digital/user_profile_adapter",
    profile_id: str = "demo_bmi32_waist110",
    case_id: str | None = None,
    metrics_csv_path: str | Path | None = None,
    target_height_cm: float = 175.0,
    target_weight_kg: float | None = None,
    target_bmi: float | None = 32.0,
    target_waist_cm: float | None = 110.0,
    baseline_height_cm: float = 170.0,
    baseline_bmi: float = 24.0,
    waist_tolerance_cm: float = 3.0,
    body_volume_tolerance_l: float = 1.5,
    warning_penalty: float = 8.0,
    report_path: str | Path | None = "outputs/reports/user_profile_adapter_stage001.md",
) -> UserProfileAdapterResult:
    manifest_path = Path(approved_set_manifest_path)
    manifest = _load_yaml(manifest_path)
    variants_raw = manifest.get("variants", [])
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ValueError("Approved PCA manifest must contain a non-empty variants list")
    variants = [item for item in variants_raw if isinstance(item, dict)]
    metrics_path = _source_metrics_path(manifest, manifest_path, metrics_csv_path)
    metrics = _read_metric_rows(metrics_path)
    baseline = _baseline_variant(variants, metrics)
    baseline_id = str(baseline.get("variant_id", "mean"))
    baseline_row = metrics.get(baseline_id, {})
    baseline_waist = _variant_value(baseline, baseline_row, "waist_cm")
    baseline_body = _variant_value(baseline, baseline_row, "body_volume_cm3")
    if baseline_waist is None or baseline_body is None:
        raise ValueError("Baseline variant must contain waist_cm and body_volume_cm3 metrics")

    derived_bmi = _derive_target_bmi(
        target_height_cm=target_height_cm,
        target_weight_kg=target_weight_kg,
        target_bmi=target_bmi,
        baseline_bmi=baseline_bmi,
    )
    derived_waist = _derive_target_waist(
        target_waist_cm=target_waist_cm,
        target_bmi=derived_bmi,
        baseline_bmi=baseline_bmi,
        baseline_waist_cm=baseline_waist,
    )
    target_body = _derive_target_body_volume(
        baseline_body_volume_cm3=baseline_body,
        baseline_waist_cm=baseline_waist,
        target_waist_cm=derived_waist,
        baseline_height_cm=baseline_height_cm,
        target_height_cm=target_height_cm,
    )

    candidates = _candidate_scores(
        variants=variants,
        metrics=metrics,
        manifest_path=manifest_path,
        target_body_volume_cm3=target_body,
        target_waist_cm=derived_waist,
        waist_tolerance_cm=waist_tolerance_cm,
        body_volume_tolerance_cm3=body_volume_tolerance_l * 1000.0,
        warning_penalty=warning_penalty,
    )
    if not candidates:
        raise ValueError("No usable candidates were found in the approved PCA manifest")
    selected = candidates[0]
    waist_values = [item.waist_cm for item in candidates if item.waist_cm is not None]
    body_values = [item.body_volume_cm3 for item in candidates if item.body_volume_cm3 is not None]
    waist_range = _range([float(item) for item in waist_values])
    body_range = _range([float(item) for item in body_values])
    status = _fit_status(
        selected=selected,
        target_waist_cm=derived_waist,
        target_body_volume_cm3=target_body,
        waist_range=waist_range,
        body_range=body_range,
        waist_tolerance_cm=waist_tolerance_cm,
        body_volume_tolerance_cm3=body_volume_tolerance_l * 1000.0,
    )

    resolved_case_id = case_id or f"{manifest.get('case_id', 'approved_pca')}_{_slug(profile_id)}"
    output = Path(output_dir)
    slug = _slug(profile_id)
    profile_yaml = output / f"{resolved_case_id}_{slug}_profile_adapter_v001.yaml"
    score_csv = output / f"{resolved_case_id}_{slug}_profile_candidate_scores_v001.csv"
    preview_png = output / f"{resolved_case_id}_{slug}_profile_adapter_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{resolved_case_id}_{slug}_profile_adapter_report_v001.md"
    source_combined_spec = _source_combined_spec_path(manifest, manifest_path)

    notes = [
        "profile_adapter_selects_from_existing_approved_pca_variants_without_editing_nifti_volumes",
        "body_volume_target_is_a_torso_proxy_scaled_from_baseline_waist_and_height",
        "selection_is_population_guided_not_subject_specific_anatomical_equivalence",
    ]
    if status == "target_outside_current_release_envelope":
        notes.append("target_profile_exceeds_current_approved_pca_release_envelope")
    if selected.warning_status == "warning":
        notes.append("selected_variant_has_existing_pca_qa_warning_use_for_sensitivity_analysis")

    command_stub = {
        "case_id": resolved_case_id,
        "selected_variant_id": selected.variant_id,
        "target_height_cm": target_height_cm,
        "target_bmi": derived_bmi,
        "target_waist_cm": derived_waist,
    }
    commands = _recommended_commands(
        result_stub=command_stub,
        source_combined_spec=str(source_combined_spec) if source_combined_spec is not None else None,
        approved_manifest_path=str(approved_set_manifest_path),
    )

    result = UserProfileAdapterResult(
        case_id=resolved_case_id,
        profile_id=profile_id,
        output_dir=str(output),
        profile_yaml_path=str(profile_yaml),
        score_csv_path=str(score_csv),
        preview_png_path=str(preview_png),
        report_path=str(report),
        source_approved_set_manifest_path=str(approved_set_manifest_path),
        source_metrics_csv_path=None if metrics_path is None else str(metrics_path),
        source_combined_spec_path=None if source_combined_spec is None else str(source_combined_spec),
        target_height_cm=float(target_height_cm),
        target_weight_kg=target_weight_kg,
        target_bmi=derived_bmi,
        target_waist_cm=derived_waist,
        target_body_volume_cm3=target_body,
        baseline_height_cm=float(baseline_height_cm),
        baseline_bmi=float(baseline_bmi),
        baseline_waist_cm=baseline_waist,
        baseline_body_volume_cm3=baseline_body,
        selected_variant_id=selected.variant_id,
        selected_score=selected.selection_score,
        fit_status=status,
        release_waist_range_cm=waist_range,
        release_body_volume_range_cm3=body_range,
        candidates=candidates,
        recommended_commands=commands,
        notes=tuple(notes),
    )
    _write_score_csv(score_csv, result.candidates)
    _write_preview(preview_png, result)
    _write_profile_yaml(profile_yaml, result)
    _write_report(report, result)
    return result


def format_user_profile_adapter_result(result: UserProfileAdapterResult) -> str:
    return "\n".join(
        [
            "User Profile Anatomy Adapter",
            f"Case ID: {result.case_id}",
            f"Profile ID: {result.profile_id}",
            f"Target BMI/waist: {result.target_bmi:.2f} / {result.target_waist_cm:.2f} cm",
            f"Selected variant: {result.selected_variant_id}",
            f"Fit status: {result.fit_status}",
            f"Selected score: {result.selected_score:.1f}/100",
            f"Profile YAML: {result.profile_yaml_path}",
            f"Candidate scores: {result.score_csv_path}",
            f"Preview PNG: {result.preview_png_path}",
        ]
    )
