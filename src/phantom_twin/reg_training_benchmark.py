from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import math

import numpy as np
import yaml


@dataclass(frozen=True)
class RegBenchmarkTargetResult:
    target_case_id: str
    pair_count: int
    label_count: int
    mean_label_dice_to_consensus: float
    median_label_dice_to_consensus: float
    volume_weighted_label_dice_to_consensus: float
    min_label_dice_to_consensus: float
    mean_volume_cv: float
    max_volume_cv: float
    mean_centroid_dispersion_mm: float
    max_centroid_dispersion_mm: float
    mean_intensity_ncc_to_reference: float
    mean_intensity_mae_hu_to_reference: float
    consensus_label_path: str
    agreement_fraction_path: str
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RegTrainingBenchmarkResult:
    dataset_id: str
    staged_manifest_path: str
    output_dir: str
    target_summary_csv_path: str
    label_metrics_csv_path: str
    pair_metrics_csv_path: str
    spec_yaml_path: str
    atlas_png_path: str
    report_path: str
    target_count: int
    pair_count: int
    mean_label_dice_to_consensus: float
    volume_weighted_label_dice_to_consensus: float
    mean_intensity_ncc_to_reference: float
    readiness_status: str
    target_results: tuple[RegBenchmarkTargetResult, ...]
    notes: tuple[str, ...]


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reg-Training-Testing benchmark requires nibabel.") from exc
    return nib


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reg-Training-Testing benchmark preview requires matplotlib.") from exc
    return plt


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    return data if isinstance(data, dict) else {}


def _load_label_names(labelmap_path: str | Path | None) -> dict[int, str]:
    if labelmap_path is None or not Path(labelmap_path).exists():
        return {}
    payload = _load_yaml(labelmap_path)
    labels = payload.get("labels", {})
    names: dict[int, str] = {}
    if isinstance(labels, dict):
        for raw_label, value in labels.items():
            try:
                label_id = int(raw_label)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                names[label_id] = str(value.get("name", f"label_{label_id}"))
            else:
                names[label_id] = str(value)
    return names


def _rows_from_staged_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest = _load_yaml(path)
    csv_path = manifest.get("outputs", {}).get("manifest_csv")
    if not csv_path:
        raise ValueError(f"Staged manifest does not list outputs.manifest_csv: {path}")
    rows: list[dict[str, str]] = []
    with Path(csv_path).open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            rows.append({key: str(value) for key, value in row.items()})
    return rows


def _group_rows_by_target(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["target_case_id"]), []).append(row)
    for target_rows in grouped.values():
        target_rows.sort(key=lambda row: str(row["moving_case_id"]))
    return dict(sorted(grouped.items()))


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))  # type: ignore[return-value]


def _voxel_volume_ml(affine: np.ndarray) -> float:
    spacing = _spacing_from_affine(affine)
    return float(spacing[0] * spacing[1] * spacing[2] / 1000.0)


def _centroid_mm(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    homogeneous = np.c_[coords.astype(float), np.ones(len(coords), dtype=float)]
    world = homogeneous @ affine.T
    return np.asarray(world[:, :3].mean(axis=0), dtype=float)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a_count = int(np.count_nonzero(a))
    b_count = int(np.count_nonzero(b))
    denominator = a_count + b_count
    if denominator == 0:
        return 1.0
    return float(2.0 * int(np.count_nonzero(a & b)) / denominator)


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _safe_median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def _safe_max(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.max(finite)) if finite else float("nan")


def _ncc(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.size < 2 or candidate.size != reference.size:
        return float("nan")
    ref = reference.astype(float, copy=False)
    cand = candidate.astype(float, copy=False)
    ref = ref - float(np.mean(ref))
    cand = cand - float(np.mean(cand))
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    if denom <= 1e-8:
        return float("nan")
    return float(np.dot(ref.ravel(), cand.ravel()) / denom)


def _sample_image(path: str | Path, mask: np.ndarray, stride: int) -> np.ndarray:
    nib = _import_nibabel()
    image = nib.load(str(path))
    step = max(1, int(stride))
    z_step = max(1, int(round(step / 2)))
    slicer = (slice(None, None, step), slice(None, None, step), slice(None, None, z_step))
    data = np.asarray(image.dataobj[slicer], dtype=np.float32)
    sampled_mask = mask[slicer]
    finite = np.isfinite(data)
    if np.count_nonzero(sampled_mask & finite) > 100:
        data = data[sampled_mask & finite]
    else:
        data = data[finite]
    if data.size > 200_000:
        rng = np.random.default_rng(42)
        data = data[rng.choice(data.size, size=200_000, replace=False)]
    return np.asarray(data, dtype=np.float32)


def _target_status(
    *,
    volume_weighted_dice: float,
    mean_dice: float,
    max_centroid_dispersion_mm: float,
    mean_ncc: float,
    min_volume_weighted_dice: float,
    min_mean_dice: float,
    max_centroid_dispersion_threshold_mm: float,
    min_mean_ncc: float,
) -> str:
    if not math.isfinite(volume_weighted_dice) or not math.isfinite(mean_dice):
        return "fail"
    if volume_weighted_dice < 0.35 or mean_dice < 0.25:
        return "fail"
    review = False
    if volume_weighted_dice < min_volume_weighted_dice or mean_dice < min_mean_dice:
        review = True
    if math.isfinite(max_centroid_dispersion_mm) and max_centroid_dispersion_mm > max_centroid_dispersion_threshold_mm:
        review = True
    if math.isfinite(mean_ncc) and mean_ncc < min_mean_ncc:
        review = True
    return "review" if review else "pass"


def _build_consensus(
    labels: list[np.ndarray],
    label_ids: tuple[int, ...],
    min_consensus_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    if not labels:
        raise ValueError("Cannot build consensus from zero labels.")
    shape = labels[0].shape
    count_by_label: dict[int, np.ndarray] = {}
    winning_count = np.zeros(shape, dtype=np.uint8)
    consensus = np.zeros(shape, dtype=np.uint8)
    for label_id in label_ids:
        counts = np.zeros(shape, dtype=np.uint8)
        for label in labels:
            counts += label == label_id
        count_by_label[label_id] = counts
        update = counts > winning_count
        consensus[update] = np.uint8(label_id)
        winning_count[update] = counts[update]
    min_votes = max(1, int(math.ceil(float(min_consensus_fraction) * len(labels))))
    consensus[winning_count < min_votes] = 0
    agreement = np.asarray((winning_count.astype(np.float32) / max(1, len(labels))) * 100.0, dtype=np.uint8)
    agreement[consensus == 0] = 0
    return consensus, agreement, count_by_label


def _evaluate_target(
    *,
    target_case_id: str,
    rows: list[dict[str, str]],
    label_ids: tuple[int, ...],
    label_names: dict[int, str],
    output_dir: Path,
    min_consensus_fraction: float,
    intensity_sample_stride: int,
    min_volume_weighted_dice: float,
    min_mean_dice: float,
    max_centroid_dispersion_mm: float,
    min_mean_ncc: float,
) -> tuple[RegBenchmarkTargetResult, list[dict[str, Any]], list[dict[str, Any]]]:
    nib = _import_nibabel()
    labels: list[np.ndarray] = []
    affines: list[np.ndarray] = []
    for row in rows:
        image = nib.load(row["label_path"])
        labels.append(np.asarray(image.dataobj, dtype=np.uint8))
        affines.append(np.asarray(image.affine, dtype=float))
    affine = affines[0]
    voxel_volume_ml = _voxel_volume_ml(affine)
    consensus, agreement, _ = _build_consensus(labels, label_ids, min_consensus_fraction)
    consensus_dir = output_dir / "consensus"
    consensus_label_path = consensus_dir / f"{target_case_id}_consensus_labels_v001.nii.gz"
    agreement_path = consensus_dir / f"{target_case_id}_agreement_percent_v001.nii.gz"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(consensus.astype(np.uint8), affine), str(consensus_label_path))
    nib.save(nib.Nifti1Image(agreement.astype(np.uint8), affine), str(agreement_path))

    label_metric_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    label_mean_dice_values: list[float] = []
    label_median_dice_values: list[float] = []
    label_weights: list[float] = []
    label_volume_cvs: list[float] = []
    label_centroid_dispersions: list[float] = []
    pair_mean_dice: dict[str, list[float]] = {str(row["moving_case_id"]): [] for row in rows}

    for label_id in label_ids:
        consensus_mask = consensus == label_id
        consensus_volume_ml = float(np.count_nonzero(consensus_mask) * voxel_volume_ml)
        volumes: list[float] = []
        dices: list[float] = []
        centroids: list[np.ndarray] = []
        for row, label in zip(rows, labels, strict=True):
            moving_case_id = str(row["moving_case_id"])
            mask = label == label_id
            dice = _dice(mask, consensus_mask)
            dices.append(dice)
            pair_mean_dice[moving_case_id].append(dice)
            volumes.append(float(np.count_nonzero(mask) * voxel_volume_ml))
            centroid = _centroid_mm(mask, affine)
            if centroid is not None:
                centroids.append(centroid)
        volume_mean = _safe_mean(volumes)
        volume_std = float(np.std(volumes)) if volumes else float("nan")
        volume_cv = float(volume_std / volume_mean) if math.isfinite(volume_mean) and volume_mean > 0 else float("nan")
        if centroids:
            centroid_stack = np.vstack(centroids)
            centroid_mean = centroid_stack.mean(axis=0)
            centroid_dispersion = float(np.mean(np.linalg.norm(centroid_stack - centroid_mean, axis=1)))
            centroid_max_dispersion = float(np.max(np.linalg.norm(centroid_stack - centroid_mean, axis=1)))
        else:
            centroid_dispersion = float("nan")
            centroid_max_dispersion = float("nan")
        mean_dice = _safe_mean(dices)
        median_dice = _safe_median(dices)
        label_mean_dice_values.append(mean_dice)
        label_median_dice_values.append(median_dice)
        label_weights.append(max(consensus_volume_ml, 1e-6))
        label_volume_cvs.append(volume_cv)
        label_centroid_dispersions.append(centroid_max_dispersion)
        label_metric_rows.append(
            {
                "target_case_id": target_case_id,
                "label_id": label_id,
                "label_name": label_names.get(label_id, f"label_{label_id}"),
                "consensus_volume_ml": f"{consensus_volume_ml:.3f}",
                "mean_propagated_volume_ml": f"{volume_mean:.3f}",
                "volume_cv": f"{volume_cv:.5f}",
                "mean_dice_to_consensus": f"{mean_dice:.5f}",
                "median_dice_to_consensus": f"{median_dice:.5f}",
                "min_dice_to_consensus": f"{min(dices):.5f}" if dices else "",
                "mean_centroid_dispersion_mm": f"{centroid_dispersion:.3f}",
                "max_centroid_dispersion_mm": f"{centroid_max_dispersion:.3f}",
            }
        )

    consensus_body = consensus > 0
    image_rows = [row for row in rows if row.get("image_path") and Path(row["image_path"]).exists()]
    reference_vector = _sample_image(image_rows[0]["image_path"], consensus_body, intensity_sample_stride) if image_rows else np.asarray([], dtype=np.float32)
    ncc_values: list[float] = []
    mae_values: list[float] = []
    for row in rows:
        if reference_vector.size and row.get("image_path") and Path(row["image_path"]).exists():
            vector = _sample_image(row["image_path"], consensus_body, intensity_sample_stride)
            size = min(reference_vector.size, vector.size)
            ncc = _ncc(reference_vector[:size], vector[:size])
            mae = float(np.mean(np.abs(reference_vector[:size].astype(float) - vector[:size].astype(float)))) if size else float("nan")
        else:
            ncc = float("nan")
            mae = float("nan")
        ncc_values.append(ncc)
        mae_values.append(mae)
        dice_values = pair_mean_dice.get(str(row["moving_case_id"]), [])
        pair_metric_rows.append(
            {
                "target_case_id": target_case_id,
                "moving_case_id": row["moving_case_id"],
                "mean_dice_to_consensus": f"{_safe_mean(dice_values):.5f}",
                "min_dice_to_consensus": f"{min(dice_values):.5f}" if dice_values else "",
                "intensity_ncc_to_reference": f"{ncc:.5f}",
                "intensity_mae_hu_to_reference": f"{mae:.3f}",
            }
        )

    weights = np.asarray(label_weights, dtype=float)
    dice_arr = np.asarray(label_mean_dice_values, dtype=float)
    finite = np.isfinite(dice_arr) & np.isfinite(weights)
    volume_weighted_dice = float(np.average(dice_arr[finite], weights=weights[finite])) if np.any(finite) else float("nan")
    mean_dice = _safe_mean(label_mean_dice_values)
    median_dice = _safe_median(label_median_dice_values)
    min_dice = float(np.min(dice_arr[np.isfinite(dice_arr)])) if np.any(np.isfinite(dice_arr)) else float("nan")
    mean_ncc = _safe_mean(ncc_values[1:]) if len(ncc_values) > 1 else float("nan")
    mean_mae = _safe_mean(mae_values[1:]) if len(mae_values) > 1 else float("nan")
    max_centroid = _safe_max(label_centroid_dispersions)
    status = _target_status(
        volume_weighted_dice=volume_weighted_dice,
        mean_dice=mean_dice,
        max_centroid_dispersion_mm=max_centroid,
        mean_ncc=mean_ncc,
        min_volume_weighted_dice=min_volume_weighted_dice,
        min_mean_dice=min_mean_dice,
        max_centroid_dispersion_threshold_mm=max_centroid_dispersion_mm,
        min_mean_ncc=min_mean_ncc,
    )
    notes = []
    if status == "review":
        notes.append("registration_consistency_threshold_review_required")
    elif status == "fail":
        notes.append("registration_consistency_below_minimum_floor")
    else:
        notes.append("registration_consistency_thresholds_passed")
    return (
        RegBenchmarkTargetResult(
            target_case_id=target_case_id,
            pair_count=len(rows),
            label_count=len(label_ids),
            mean_label_dice_to_consensus=mean_dice,
            median_label_dice_to_consensus=median_dice,
            volume_weighted_label_dice_to_consensus=volume_weighted_dice,
            min_label_dice_to_consensus=min_dice,
            mean_volume_cv=_safe_mean(label_volume_cvs),
            max_volume_cv=_safe_max(label_volume_cvs),
            mean_centroid_dispersion_mm=_safe_mean(label_centroid_dispersions),
            max_centroid_dispersion_mm=max_centroid,
            mean_intensity_ncc_to_reference=mean_ncc,
            mean_intensity_mae_hu_to_reference=mean_mae,
            consensus_label_path=str(consensus_label_path),
            agreement_fraction_path=str(agreement_path),
            status=status,
            notes=tuple(notes),
        ),
        label_metric_rows,
        pair_metric_rows,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_atlas(path: Path, result: RegTrainingBenchmarkResult) -> None:
    if not result.target_results:
        return
    nib = _import_nibabel()
    plt = _import_plotting()
    fig, axes = plt.subplots(len(result.target_results), 2, figsize=(9, 3.2 * len(result.target_results)), dpi=160)
    if len(result.target_results) == 1:
        axes = np.asarray([axes])
    fig.patch.set_facecolor("#f7f1e3")
    for row, target in zip(axes, result.target_results):
        label_ax, agreement_ax = row
        for ax in row:
            ax.set_facecolor("#f7f1e3")
            ax.axis("off")
        try:
            labels = nib.load(target.consensus_label_path)
            agreement = nib.load(target.agreement_fraction_path)
            label_data = np.asarray(labels.dataobj, dtype=np.uint8)
            agreement_data = np.asarray(agreement.dataobj, dtype=np.uint8)
            occupied = np.argwhere(label_data > 0)
            z_index = int(np.median(occupied[:, 2])) if len(occupied) else label_data.shape[2] // 2
            label_ax.imshow(np.rot90(label_data[:, :, z_index]), cmap="tab20", interpolation="nearest")
            agreement_ax.imshow(np.rot90(agreement_data[:, :, z_index]), cmap="magma", vmin=0, vmax=100)
            label_ax.set_title(f"{target.target_case_id} consensus labels")
            agreement_ax.set_title(f"agreement %; status={target.status}")
        except Exception as exc:
            label_ax.text(0.5, 0.5, f"preview failed\n{type(exc).__name__}", ha="center", va="center")
            agreement_ax.text(0.5, 0.5, "preview failed", ha="center", va="center")
    fig.suptitle("Reg-Training-Testing Registration Consistency Benchmark", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _readiness(targets: tuple[RegBenchmarkTargetResult, ...]) -> str:
    statuses = {target.status for target in targets}
    if not targets:
        return "registration_benchmark_empty"
    if statuses == {"pass"}:
        return "registration_benchmark_pass"
    if "fail" in statuses:
        return "registration_benchmark_fail"
    return "registration_benchmark_review_required"


def _write_spec(path: Path, result: RegTrainingBenchmarkResult) -> None:
    payload = {
        "dataset_id": result.dataset_id,
        "package_type": "reg_training_testing_registration_benchmark",
        "staged_manifest": result.staged_manifest_path,
        "readiness_status": result.readiness_status,
        "summary": {
            "target_count": result.target_count,
            "pair_count": result.pair_count,
            "mean_label_dice_to_consensus": result.mean_label_dice_to_consensus,
            "volume_weighted_label_dice_to_consensus": result.volume_weighted_label_dice_to_consensus,
            "mean_intensity_ncc_to_reference": result.mean_intensity_ncc_to_reference,
        },
        "outputs": {
            "target_summary_csv": result.target_summary_csv_path,
            "label_metrics_csv": result.label_metrics_csv_path,
            "pair_metrics_csv": result.pair_metrics_csv_path,
            "atlas_png": result.atlas_png_path,
            "report": result.report_path,
        },
        "targets": [
            {
                "target_case_id": target.target_case_id,
                "status": target.status,
                "pair_count": target.pair_count,
                "mean_label_dice_to_consensus": target.mean_label_dice_to_consensus,
                "volume_weighted_label_dice_to_consensus": target.volume_weighted_label_dice_to_consensus,
                "mean_intensity_ncc_to_reference": target.mean_intensity_ncc_to_reference,
                "consensus_label_path": target.consensus_label_path,
                "agreement_fraction_path": target.agreement_fraction_path,
                "notes": list(target.notes),
            }
            for target in result.target_results
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_report(path: Path, result: RegTrainingBenchmarkResult) -> None:
    lines = [
        "# Reg-Training-Testing Registration Benchmark",
        "",
        f"Dataset ID: `{result.dataset_id}`",
        f"Readiness status: `{result.readiness_status}`",
        "",
        "## Summary",
        "",
        f"- Target cases: {result.target_count}",
        f"- Warped image/label pairs: {result.pair_count}",
        f"- Mean label Dice to consensus: {result.mean_label_dice_to_consensus:.3f}",
        f"- Volume-weighted label Dice to consensus: {result.volume_weighted_label_dice_to_consensus:.3f}",
        f"- Mean CT intensity NCC to first warped image: {result.mean_intensity_ncc_to_reference:.3f}",
        "",
        "## Target Results",
        "",
    ]
    for target in result.target_results:
        lines.append(
            f"- `{target.target_case_id}`: status=`{target.status}`, pairs={target.pair_count}, "
            f"mean Dice={target.mean_label_dice_to_consensus:.3f}, "
            f"volume-weighted Dice={target.volume_weighted_label_dice_to_consensus:.3f}, "
            f"max centroid dispersion={target.max_centroid_dispersion_mm:.1f} mm, "
            f"mean NCC={target.mean_intensity_ncc_to_reference:.3f}"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Target summary CSV: `{result.target_summary_csv_path}`",
            f"- Label metrics CSV: `{result.label_metrics_csv_path}`",
            f"- Pair metrics CSV: `{result.pair_metrics_csv_path}`",
            f"- Benchmark spec YAML: `{result.spec_yaml_path}`",
            f"- Preview atlas: `{result.atlas_png_path}`",
            "",
            "## Interpretation",
            "",
            "- This benchmark measures agreement among already warped atlas labels on each target grid.",
            "- It is not ground-truth clinical registration accuracy unless native target manual labels are added.",
            "- Use high-agreement organs and vessel labels as safer anchors for anatomy adaptation; inspect low-agreement labels before using them for phantom deformation.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_reg_training_testing_benchmark(
    *,
    staged_manifest_path: str | Path = "data/processed/reg_training_testing/reg_training_testing_stage001_manifest_v001.yaml",
    output_dir: str | Path = "outputs/digital/reg_training_testing_benchmark",
    dataset_id: str = "reg_training_testing_benchmark_stage001",
    labelmap_path: str | Path | None = "configs/labelmaps/btcv_abdomen.yaml",
    min_consensus_fraction: float = 0.35,
    min_volume_weighted_dice: float = 0.65,
    min_mean_dice: float = 0.50,
    max_centroid_dispersion_mm: float = 45.0,
    min_mean_ncc: float = 0.15,
    intensity_sample_stride: int = 8,
    report_path: str | Path | None = "outputs/reports/reg_training_testing_benchmark_stage001.md",
) -> RegTrainingBenchmarkResult:
    output = Path(output_dir) / dataset_id
    rows = _rows_from_staged_manifest(staged_manifest_path)
    grouped = _group_rows_by_target(rows)
    label_names = _load_label_names(labelmap_path)
    label_ids = tuple(label_id for label_id in sorted(label_names) if label_id > 0)
    if not label_ids:
        label_ids = tuple(range(1, 14))

    target_summary_rows: list[dict[str, Any]] = []
    label_metric_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    target_results: list[RegBenchmarkTargetResult] = []

    for target_case_id, target_rows in grouped.items():
        target_result, target_label_rows, target_pair_rows = _evaluate_target(
            target_case_id=target_case_id,
            rows=target_rows,
            label_ids=label_ids,
            label_names=label_names,
            output_dir=output,
            min_consensus_fraction=min_consensus_fraction,
            intensity_sample_stride=intensity_sample_stride,
            min_volume_weighted_dice=min_volume_weighted_dice,
            min_mean_dice=min_mean_dice,
            max_centroid_dispersion_mm=max_centroid_dispersion_mm,
            min_mean_ncc=min_mean_ncc,
        )
        target_results.append(target_result)
        label_metric_rows.extend(target_label_rows)
        pair_metric_rows.extend(target_pair_rows)
        target_summary_rows.append(
            {
                "target_case_id": target_result.target_case_id,
                "status": target_result.status,
                "pair_count": target_result.pair_count,
                "label_count": target_result.label_count,
                "mean_label_dice_to_consensus": f"{target_result.mean_label_dice_to_consensus:.5f}",
                "median_label_dice_to_consensus": f"{target_result.median_label_dice_to_consensus:.5f}",
                "volume_weighted_label_dice_to_consensus": f"{target_result.volume_weighted_label_dice_to_consensus:.5f}",
                "min_label_dice_to_consensus": f"{target_result.min_label_dice_to_consensus:.5f}",
                "mean_volume_cv": f"{target_result.mean_volume_cv:.5f}",
                "max_volume_cv": f"{target_result.max_volume_cv:.5f}",
                "mean_centroid_dispersion_mm": f"{target_result.mean_centroid_dispersion_mm:.3f}",
                "max_centroid_dispersion_mm": f"{target_result.max_centroid_dispersion_mm:.3f}",
                "mean_intensity_ncc_to_reference": f"{target_result.mean_intensity_ncc_to_reference:.5f}",
                "mean_intensity_mae_hu_to_reference": f"{target_result.mean_intensity_mae_hu_to_reference:.3f}",
                "consensus_label_path": target_result.consensus_label_path,
                "agreement_fraction_path": target_result.agreement_fraction_path,
            }
        )

    target_results_tuple = tuple(target_results)
    readiness = _readiness(target_results_tuple)
    target_summary_csv = output / f"{dataset_id}_target_summary_v001.csv"
    label_metrics_csv = output / f"{dataset_id}_label_metrics_v001.csv"
    pair_metrics_csv = output / f"{dataset_id}_pair_metrics_v001.csv"
    spec_yaml = output / f"{dataset_id}_benchmark_spec_v001.yaml"
    atlas_png = output / f"{dataset_id}_atlas_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{dataset_id}_benchmark_report_v001.md"

    result = RegTrainingBenchmarkResult(
        dataset_id=dataset_id,
        staged_manifest_path=str(staged_manifest_path),
        output_dir=str(output),
        target_summary_csv_path=str(target_summary_csv),
        label_metrics_csv_path=str(label_metrics_csv),
        pair_metrics_csv_path=str(pair_metrics_csv),
        spec_yaml_path=str(spec_yaml),
        atlas_png_path=str(atlas_png),
        report_path=str(report),
        target_count=len(target_results_tuple),
        pair_count=sum(target.pair_count for target in target_results_tuple),
        mean_label_dice_to_consensus=_safe_mean([target.mean_label_dice_to_consensus for target in target_results_tuple]),
        volume_weighted_label_dice_to_consensus=_safe_mean(
            [target.volume_weighted_label_dice_to_consensus for target in target_results_tuple]
        ),
        mean_intensity_ncc_to_reference=_safe_mean([target.mean_intensity_ncc_to_reference for target in target_results_tuple]),
        readiness_status=readiness,
        target_results=target_results_tuple,
        notes=(
            "benchmark_uses_propagated_atlas_label_consensus_not_native_target_ground_truth",
            "metrics_quantify_registration_consistency_for_adaptive_anatomy_development",
            "low_agreement_labels_should_not_drive_phantom_deformation_without_manual_review",
        ),
    )
    _write_csv(
        target_summary_csv,
        target_summary_rows,
        [
            "target_case_id",
            "status",
            "pair_count",
            "label_count",
            "mean_label_dice_to_consensus",
            "median_label_dice_to_consensus",
            "volume_weighted_label_dice_to_consensus",
            "min_label_dice_to_consensus",
            "mean_volume_cv",
            "max_volume_cv",
            "mean_centroid_dispersion_mm",
            "max_centroid_dispersion_mm",
            "mean_intensity_ncc_to_reference",
            "mean_intensity_mae_hu_to_reference",
            "consensus_label_path",
            "agreement_fraction_path",
        ],
    )
    _write_csv(
        label_metrics_csv,
        label_metric_rows,
        [
            "target_case_id",
            "label_id",
            "label_name",
            "consensus_volume_ml",
            "mean_propagated_volume_ml",
            "volume_cv",
            "mean_dice_to_consensus",
            "median_dice_to_consensus",
            "min_dice_to_consensus",
            "mean_centroid_dispersion_mm",
            "max_centroid_dispersion_mm",
        ],
    )
    _write_csv(
        pair_metrics_csv,
        pair_metric_rows,
        [
            "target_case_id",
            "moving_case_id",
            "mean_dice_to_consensus",
            "min_dice_to_consensus",
            "intensity_ncc_to_reference",
            "intensity_mae_hu_to_reference",
        ],
    )
    _write_atlas(atlas_png, result)
    _write_spec(spec_yaml, result)
    _write_report(report, result)
    return result


def format_reg_training_testing_benchmark_result(result: RegTrainingBenchmarkResult) -> str:
    return "\n".join(
        [
            "Reg-Training-Testing registration benchmark built",
            f"Dataset ID: {result.dataset_id}",
            f"Readiness status: {result.readiness_status}",
            f"Targets/pairs: {result.target_count}/{result.pair_count}",
            f"Mean label Dice: {result.mean_label_dice_to_consensus:.3f}",
            f"Volume-weighted label Dice: {result.volume_weighted_label_dice_to_consensus:.3f}",
            f"Mean intensity NCC: {result.mean_intensity_ncc_to_reference:.3f}",
            f"Spec: {result.spec_yaml_path}",
            f"Preview atlas: {result.atlas_png_path}",
            f"Report: {result.report_path}",
        ]
    )
