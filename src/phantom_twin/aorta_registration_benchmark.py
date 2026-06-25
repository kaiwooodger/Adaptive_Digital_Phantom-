from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import csv
import math
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class AortaBenchmarkCase:
    case_id: str
    mask_path: str
    centerline_csv_path: str
    loo_prediction_csv_path: str
    raw_centerline_points: int
    sample_count: int
    length_mm: float
    mean_radius_mm: float
    endpoint_span_mm: float
    loo_rmse_mm: float
    loo_max_error_mm: float
    loo_radius_mae_mm: float
    status: str


@dataclass(frozen=True)
class AortaRegistrationBenchmarkResult:
    dataset_id: str
    output_dir: str
    manifest_csv_path: str
    metrics_csv_path: str
    model_npz_path: str
    model_yaml_path: str
    atlas_png_path: str
    report_path: str
    case_count: int
    sample_count: int
    mean_loo_rmse_mm: float
    max_loo_rmse_mm: float
    mean_loo_radius_mae_mm: float
    readiness_status: str
    cases: tuple[AortaBenchmarkCase, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class LearnedAortaGraphResult:
    case_id: str
    output_dir: str
    graph_yaml_path: str
    nodes_csv_path: str
    edges_csv_path: str
    centerline_csv_path: str
    preview_png_path: str
    report_path: str
    source_graph_path: str
    model_yaml_path: str
    replaced_node_count: int
    replaced_edge_count: int
    learned_centerline_points: int
    learned_centerline_length_mm: float
    mean_model_loo_rmse_mm: float
    readiness_status: str
    notes: tuple[str, ...]


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from scipy.signal import savgol_filter  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Aorta registration benchmark requires matplotlib, nibabel, scipy, and PyYAML.") from exc
    return plt, nib, savgol_filter


def _line_length(points: np.ndarray | list[list[float]]) -> float:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or len(array) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array[:, :3], axis=0), axis=1).sum())


def _smooth(values: np.ndarray, savgol_filter) -> np.ndarray:
    if len(values) < 7:
        return values
    window = min(17, len(values) if len(values) % 2 else len(values) - 1)
    if window < 7:
        return values
    return savgol_filter(values, window_length=window, polyorder=3, mode="interp")


def _read_manifest_cases(path: str | Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            case_id = row.get("case_id") or row.get("source_case_id") or f"case_{index:03d}"
            mask_path = row.get("aorta_mask_nifti_path") or row.get("vessel_seg_path") or row.get("mask_path") or ""
            if mask_path:
                rows.append((str(case_id), mask_path))
    if not rows:
        raise ValueError(f"No aorta mask rows found in manifest: {path}")
    return rows


def _world_from_voxel(affine: np.ndarray, voxel_xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([voxel_xyz, np.ones(len(voxel_xyz), dtype=float)])
    return (affine @ homogeneous.T).T[:, :3]


def _centerline_from_mask(mask_path: str | Path, label_value: int | None, savgol_filter) -> np.ndarray:
    _, nib, _ = _import_dependencies()
    image = nib.load(str(mask_path))
    data = np.asanyarray(image.dataobj)
    mask = data > 0 if label_value is None else data == label_value
    if mask.ndim != 3:
        raise ValueError(f"Aorta mask must be 3D: {mask_path}")

    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    voxel_area_mm2 = float(zooms[0] * zooms[1])
    rows: list[tuple[float, float, float, float, int]] = []
    for k in range(mask.shape[2]):
        coords = np.argwhere(mask[:, :, k])
        if coords.size == 0:
            continue
        ijk = np.column_stack([coords[:, 0], coords[:, 1], np.full(len(coords), k, dtype=float)])
        centroid = np.asarray([[float(coords[:, 0].mean()), float(coords[:, 1].mean()), float(k)]])
        center_mm = _world_from_voxel(np.asarray(image.affine, dtype=float), centroid)[0]
        area_mm2 = float(len(coords) * voxel_area_mm2)
        radius_mm = math.sqrt(max(area_mm2, 1e-6) / math.pi)
        rows.append((float(center_mm[0]), float(center_mm[1]), float(center_mm[2]), radius_mm, int(len(ijk))))
    if len(rows) < 2:
        raise ValueError(f"Aorta mask needs at least two occupied slices: {mask_path}")

    centerline = np.asarray(rows, dtype=float)
    order = np.argsort(centerline[:, 2])
    centerline = centerline[order]
    centerline[:, 0] = _smooth(centerline[:, 0], savgol_filter)
    centerline[:, 1] = _smooth(centerline[:, 1], savgol_filter)
    centerline[:, 3] = np.maximum(_smooth(centerline[:, 3], savgol_filter), 0.1)
    return centerline


def _resample_centerline(centerline: np.ndarray, sample_count: int) -> np.ndarray:
    z = centerline[:, 2]
    if math.isclose(float(z[0]), float(z[-1])):
        distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centerline[:, :3], axis=0), axis=1))])
        axis = distances / max(float(distances[-1]), 1e-6)
    else:
        axis = (z - z[0]) / (z[-1] - z[0])
    u = np.linspace(0.0, 1.0, int(sample_count), dtype=float)
    out = np.zeros((len(u), 5), dtype=float)
    out[:, 0] = u
    for col in range(4):
        out[:, col + 1] = np.interp(u, axis, centerline[:, col])
    return out


def _offsets_from_sample(sampled: np.ndarray) -> np.ndarray:
    u = sampled[:, 0]
    start = sampled[0, 1:4]
    end = sampled[-1, 1:4]
    line = (1.0 - u[:, None]) * start[None, :] + u[:, None] * end[None, :]
    return sampled[:, 1:4] - line


def _reconstruct_from_offsets(sampled: np.ndarray, offsets: np.ndarray, radii: np.ndarray) -> np.ndarray:
    u = sampled[:, 0]
    start = sampled[0, 1:4]
    end = sampled[-1, 1:4]
    line = (1.0 - u[:, None]) * start[None, :] + u[:, None] * end[None, :]
    predicted = np.zeros_like(sampled)
    predicted[:, 0] = u
    predicted[:, 1:4] = line + offsets
    predicted[:, 4] = radii
    return predicted


def _write_centerline_csv(path: Path, sampled: np.ndarray, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "u", "x_mm", "y_mm", "z_mm", "radius_mm"]
    extra = extra or {}
    fields.extend(extra)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(sampled):
            payload = {
                "index": index,
                "u": f"{row[0]:.8f}",
                "x_mm": f"{row[1]:.8f}",
                "y_mm": f"{row[2]:.8f}",
                "z_mm": f"{row[3]:.8f}",
                "radius_mm": f"{row[4]:.8f}",
            }
            payload.update({key: value for key, value in extra.items()})
            writer.writerow(payload)


def _write_metrics_csv(path: Path, cases: tuple[AortaBenchmarkCase, ...]) -> None:
    fields = [
        "case_id",
        "mask_path",
        "raw_centerline_points",
        "sample_count",
        "length_mm",
        "mean_radius_mm",
        "endpoint_span_mm",
        "loo_rmse_mm",
        "loo_max_error_mm",
        "loo_radius_mae_mm",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in cases:
            writer.writerow(
                {
                    "case_id": item.case_id,
                    "mask_path": item.mask_path,
                    "raw_centerline_points": item.raw_centerline_points,
                    "sample_count": item.sample_count,
                    "length_mm": f"{item.length_mm:.6f}",
                    "mean_radius_mm": f"{item.mean_radius_mm:.6f}",
                    "endpoint_span_mm": f"{item.endpoint_span_mm:.6f}",
                    "loo_rmse_mm": f"{item.loo_rmse_mm:.6f}",
                    "loo_max_error_mm": f"{item.loo_max_error_mm:.6f}",
                    "loo_radius_mae_mm": f"{item.loo_radius_mae_mm:.6f}",
                    "status": item.status,
                }
            )


def _write_atlas(path: Path, sampled_by_case: dict[str, np.ndarray], predicted_by_case: dict[str, np.ndarray], cases: tuple[AortaBenchmarkCase, ...]) -> None:
    plt, _, _ = _import_dependencies()
    fig = plt.figure(figsize=(13, 8), dpi=170)
    fig.patch.set_facecolor("#f6f1e8")
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_facecolor("#f6f1e8")
    for case_id, sampled in sampled_by_case.items():
        ax3d.plot(sampled[:, 1], sampled[:, 2], sampled[:, 3], color="#94a3b8", linewidth=0.9, alpha=0.45)
        predicted = predicted_by_case[case_id]
        ax3d.plot(predicted[:, 1], predicted[:, 2], predicted[:, 3], color="#dc2626", linewidth=0.8, alpha=0.32)
    ax3d.set_title("20-case aorta registration benchmark\nblue = real masks, red = leave-one-out learned fit")
    ax3d.set_xlabel("x mm")
    ax3d.set_ylabel("y mm")
    ax3d.set_zlabel("z mm")
    ax3d.view_init(elev=17, azim=-58)

    ax = fig.add_subplot(1, 2, 2)
    labels = [item.case_id.replace("avt_kits_", "") for item in cases]
    errors = [item.loo_rmse_mm for item in cases]
    radii = [item.loo_radius_mae_mm for item in cases]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, errors, width=0.36, color="#dc2626", label="centerline RMSE mm")
    ax.bar(x + 0.18, radii, width=0.36, color="#2563eb", label="radius MAE mm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("mm")
    ax.set_title("Leave-one-out registration errors")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_benchmark_report(path: Path, result: AortaRegistrationBenchmarkResult) -> None:
    lines = [
        "# Aorta Registration Benchmark",
        "",
        f"Dataset ID: `{result.dataset_id}`",
        f"Readiness: `{result.readiness_status}`",
        f"Cases: {result.case_count}",
        f"Samples per centerline: {result.sample_count}",
        f"Mean/max leave-one-out centerline RMSE: {result.mean_loo_rmse_mm:.3f} / {result.max_loo_rmse_mm:.3f} mm",
        f"Mean leave-one-out radius MAE: {result.mean_loo_radius_mae_mm:.3f} mm",
        "",
        "## Outputs",
        "",
        f"- Metrics CSV: `{result.metrics_csv_path}`",
        f"- Model NPZ: `{result.model_npz_path}`",
        f"- Model YAML: `{result.model_yaml_path}`",
        f"- Atlas PNG: `{result.atlas_png_path}`",
        "",
        "## Interpretation",
        "",
        "- Each staged aorta mask was converted into an axial centerline and radius profile.",
        "- Centerlines were registered into a shared endpoint-normalized trunk coordinate system.",
        "- Leave-one-out errors estimate how well the learned cohort aorta model predicts a held-out case from endpoints alone.",
        "- This benchmark supports learned aorta-trunk generation, not branch-rich renal/iliac/hepatic/splenic/venous generation.",
        "",
        "## Case Metrics",
        "",
        "| Case | Points | Length mm | Mean radius mm | LOO RMSE mm | LOO max mm | Radius MAE mm | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result.cases:
        lines.append(
            f"| {item.case_id} | {item.raw_centerline_points} | {item.length_mm:.2f} | {item.mean_radius_mm:.2f} | "
            f"{item.loo_rmse_mm:.2f} | {item.loo_max_error_mm:.2f} | {item.loo_radius_mae_mm:.2f} | {item.status} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_aorta_registration_benchmark(
    *,
    manifest_csv_path: str | Path,
    output_dir: str | Path = "outputs/digital/aorta_registration_benchmark",
    dataset_id: str = "avt_kits_aorta_benchmark_stage001",
    sample_count: int = 64,
    label_value: int | None = 1,
    report_path: str | Path | None = "outputs/reports/aorta_registration_benchmark_stage001.md",
) -> AortaRegistrationBenchmarkResult:
    _, _, savgol_filter = _import_dependencies()
    manifest_path = Path(manifest_csv_path)
    output = Path(output_dir)
    centerline_dir = output / "centerlines"
    prediction_dir = output / "leave_one_out_predictions"
    metrics_csv = output / f"{dataset_id}_metrics_v001.csv"
    model_npz = output / f"{dataset_id}_model_v001.npz"
    model_yaml = output / f"{dataset_id}_model_v001.yaml"
    atlas_png = output / f"{dataset_id}_atlas_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{dataset_id}_report_v001.md"

    raw_cases = _read_manifest_cases(manifest_path)
    sampled_by_case: dict[str, np.ndarray] = {}
    offsets: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    raw_centerline_counts: dict[str, int] = {}
    lengths: dict[str, float] = {}
    spans: dict[str, float] = {}

    for case_id, mask_path in raw_cases:
        centerline = _centerline_from_mask(mask_path, label_value, savgol_filter)
        sampled = _resample_centerline(centerline, sample_count)
        sampled_by_case[case_id] = sampled
        offsets.append(_offsets_from_sample(sampled))
        radii.append(sampled[:, 4])
        raw_centerline_counts[case_id] = int(len(centerline))
        lengths[case_id] = _line_length(centerline[:, :3])
        spans[case_id] = float(np.linalg.norm(sampled[-1, 1:4] - sampled[0, 1:4]))

    offset_stack = np.stack(offsets, axis=0)
    radius_stack = np.stack(radii, axis=0)
    mean_offsets = offset_stack.mean(axis=0)
    sd_offsets = offset_stack.std(axis=0)
    mean_radii = radius_stack.mean(axis=0)
    sd_radii = radius_stack.std(axis=0)
    endpoint_spans = np.asarray([spans[case_id] for case_id, _ in raw_cases], dtype=float)

    cases: list[AortaBenchmarkCase] = []
    predicted_by_case: dict[str, np.ndarray] = {}
    for index, (case_id, mask_path) in enumerate(raw_cases):
        sampled = sampled_by_case[case_id]
        if len(raw_cases) > 1:
            loo_offsets = np.delete(offset_stack, index, axis=0).mean(axis=0)
            loo_radii = np.delete(radius_stack, index, axis=0).mean(axis=0)
        else:
            loo_offsets = mean_offsets
            loo_radii = mean_radii
        predicted = _reconstruct_from_offsets(sampled, loo_offsets, loo_radii)
        predicted_by_case[case_id] = predicted
        point_errors = np.linalg.norm(predicted[:, 1:4] - sampled[:, 1:4], axis=1)
        radius_errors = np.abs(predicted[:, 4] - sampled[:, 4])
        centerline_csv = centerline_dir / f"{case_id}_aorta_centerline_v001.csv"
        prediction_csv = prediction_dir / f"{case_id}_aorta_loo_prediction_v001.csv"
        _write_centerline_csv(centerline_csv, sampled)
        _write_centerline_csv(prediction_csv, predicted)
        rmse = float(np.sqrt(np.mean(point_errors**2)))
        max_error = float(np.max(point_errors))
        radius_mae = float(np.mean(radius_errors))
        status = "pass" if rmse <= 20.0 and radius_mae <= 6.0 else "review"
        cases.append(
            AortaBenchmarkCase(
                case_id=case_id,
                mask_path=mask_path,
                centerline_csv_path=str(centerline_csv),
                loo_prediction_csv_path=str(prediction_csv),
                raw_centerline_points=raw_centerline_counts[case_id],
                sample_count=int(sample_count),
                length_mm=lengths[case_id],
                mean_radius_mm=float(np.mean(sampled[:, 4])),
                endpoint_span_mm=spans[case_id],
                loo_rmse_mm=rmse,
                loo_max_error_mm=max_error,
                loo_radius_mae_mm=radius_mae,
                status=status,
            )
        )

    case_tuple = tuple(cases)
    mean_rmse = float(np.mean([item.loo_rmse_mm for item in case_tuple]))
    max_rmse = float(np.max([item.loo_rmse_mm for item in case_tuple]))
    mean_radius_mae = float(np.mean([item.loo_radius_mae_mm for item in case_tuple]))
    readiness = "learned_aorta_model_ready" if case_tuple and all(item.status == "pass" for item in case_tuple) else "learned_aorta_model_review_required"
    notes = (
        "model_is_endpoint_normalized_leave_one_out_aorta_trunk_fit",
        "cohort_contains_aorta_masks_only_no_branch_or_venous_labels",
        "use_for_aorta_trunk_generation_and_registration_QA_not_full_vascular_tree_validation",
    )

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_npz,
        u=np.linspace(0.0, 1.0, int(sample_count), dtype=float),
        mean_offsets_mm=mean_offsets,
        sd_offsets_mm=sd_offsets,
        mean_radius_mm=mean_radii,
        sd_radius_mm=sd_radii,
        mean_endpoint_span_mm=float(endpoint_spans.mean()),
        sd_endpoint_span_mm=float(endpoint_spans.std()),
        case_ids=np.asarray([case.case_id for case in case_tuple], dtype=object),
        loo_rmse_mm=np.asarray([case.loo_rmse_mm for case in case_tuple], dtype=float),
        loo_radius_mae_mm=np.asarray([case.loo_radius_mae_mm for case in case_tuple], dtype=float),
    )

    model_data = {
        "dataset_id": dataset_id,
        "package_type": "aorta_registration_benchmark_model",
        "manifest_csv_path": str(manifest_path),
        "model_npz_path": str(model_npz),
        "metrics_csv_path": str(metrics_csv),
        "sample_count": int(sample_count),
        "case_count": len(case_tuple),
        "mean_loo_rmse_mm": mean_rmse,
        "max_loo_rmse_mm": max_rmse,
        "mean_loo_radius_mae_mm": mean_radius_mae,
        "mean_endpoint_span_mm": float(endpoint_spans.mean()),
        "readiness_status": readiness,
        "notes": list(notes),
    }
    model_yaml.parent.mkdir(parents=True, exist_ok=True)
    model_yaml.write_text(yaml.safe_dump(model_data, sort_keys=False))
    _write_metrics_csv(metrics_csv, case_tuple)
    _write_atlas(atlas_png, sampled_by_case, predicted_by_case, case_tuple)

    result = AortaRegistrationBenchmarkResult(
        dataset_id=dataset_id,
        output_dir=str(output),
        manifest_csv_path=str(manifest_path),
        metrics_csv_path=str(metrics_csv),
        model_npz_path=str(model_npz),
        model_yaml_path=str(model_yaml),
        atlas_png_path=str(atlas_png),
        report_path=str(report),
        case_count=len(case_tuple),
        sample_count=int(sample_count),
        mean_loo_rmse_mm=mean_rmse,
        max_loo_rmse_mm=max_rmse,
        mean_loo_radius_mae_mm=mean_radius_mae,
        readiness_status=readiness,
        cases=case_tuple,
        notes=notes,
    )
    _write_benchmark_report(report, result)
    return result


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _load_model(model_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(model_path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        spec = _load_yaml(path)
        npz_path = Path(str(spec.get("model_npz_path", "")))
        if not npz_path.is_absolute():
            npz_path = Path.cwd() / npz_path
    else:
        spec = {"model_npz_path": str(path), "dataset_id": path.stem}
        npz_path = path
    loaded = np.load(npz_path, allow_pickle=True)
    arrays = {key: loaded[key] for key in loaded.files}
    return spec, arrays


def _node_lookup(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in graph.get("nodes", []) if isinstance(node, dict) and "id" in node}


def _append_note(item: dict[str, Any], note: str) -> None:
    notes = [str(value) for value in item.get("notes", [])]
    if note not in notes:
        notes.append(note)
    item["notes"] = notes


def _interp_model(arrays: dict[str, np.ndarray], u_values: np.ndarray, *, reverse: bool) -> tuple[np.ndarray, np.ndarray]:
    model_u = np.asarray(arrays["u"], dtype=float)
    offsets = np.asarray(arrays["mean_offsets_mm"], dtype=float)
    radii = np.asarray(arrays["mean_radius_mm"], dtype=float)
    if reverse:
        offsets = offsets[::-1]
        radii = radii[::-1]
    out_offsets = np.column_stack([np.interp(u_values, model_u, offsets[:, axis]) for axis in range(3)])
    out_radii = np.interp(u_values, model_u, radii)
    return out_offsets, out_radii


def _learned_segment(
    arrays: dict[str, np.ndarray],
    source: np.ndarray,
    target: np.ndarray,
    *,
    point_count: int,
    radius_scale: float,
    max_radius_mm: float | None,
    minimum_target_span_mm: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    point_count = max(2, int(point_count))
    u = np.linspace(0.0, 1.0, point_count, dtype=float)
    reverse = bool(source[2] > target[2])
    offsets, radii = _interp_model(arrays, u, reverse=reverse)
    endpoint_span = float(np.asarray(arrays.get("mean_endpoint_span_mm", np.asarray([1.0]))).reshape(-1)[0])
    target_vector = target - source
    target_span = float(np.linalg.norm(target_vector))
    if minimum_target_span_mm is not None and target_span < float(minimum_target_span_mm):
        if target_span <= 1e-6:
            target_vector = np.asarray([0.0, 0.0, -1.0], dtype=float)
            if source[2] < target[2]:
                target_vector *= -1.0
            target_span = 1.0
        target = source + target_vector / target_span * float(minimum_target_span_mm)
        target_span = float(minimum_target_span_mm)
    scale = target_span / max(endpoint_span, 1e-6)
    line = (1.0 - u[:, None]) * source[None, :] + u[:, None] * target[None, :]
    points = line + offsets * scale
    points[0] = source
    points[-1] = target
    learned_radii = np.maximum(radii * float(radius_scale), 0.1)
    if max_radius_mm is not None:
        learned_radii = np.minimum(learned_radii, float(max_radius_mm))
    return points, learned_radii


def _write_graph_nodes_csv(path: Path, nodes: list[dict[str, Any]]) -> None:
    fields = ["id", "kind", "role", "boundary_role", "radius_mm", "x_mm", "y_mm", "z_mm", "source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node in nodes:
            position = node.get("position_mm", [0.0, 0.0, 0.0])
            notes = tuple(str(note) for note in node.get("notes", []))
            writer.writerow(
                {
                    "id": node.get("id", ""),
                    "kind": node.get("kind", ""),
                    "role": node.get("role", ""),
                    "boundary_role": node.get("boundary_role", ""),
                    "radius_mm": f"{float(node.get('radius_mm', 0.0)):.6f}",
                    "x_mm": f"{float(position[0]):.6f}",
                    "y_mm": f"{float(position[1]):.6f}",
                    "z_mm": f"{float(position[2]):.6f}",
                    "source": "learned_aorta_model" if "position_replaced_from_population_learned_aorta_model" in notes else "retained",
                }
            )


def _write_graph_edges_csv(path: Path, edges: list[dict[str, Any]]) -> None:
    fields = ["id", "source", "target", "vessel_type", "flow_role", "length_mm", "point_count", "radius_start_mm", "radius_end_mm", "source_kind"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for edge in edges:
            notes = tuple(str(note) for note in edge.get("notes", []))
            writer.writerow(
                {
                    "id": edge.get("id", ""),
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "vessel_type": edge.get("vessel_type", ""),
                    "flow_role": edge.get("flow_role", ""),
                    "length_mm": f"{float(edge.get('length_mm', 0.0)):.6f}",
                    "point_count": len(edge.get("polyline_mm", [])),
                    "radius_start_mm": f"{float(edge.get('radius_start_mm', 0.0)):.6f}",
                    "radius_end_mm": f"{float(edge.get('radius_end_mm', 0.0)):.6f}",
                    "source_kind": "learned_aorta_model" if "polyline_replaced_from_population_learned_aorta_model" in notes else "retained",
                }
            )


def _write_learned_graph_preview(path: Path, source_graph: dict[str, Any], learned_graph: dict[str, Any], learned_points: np.ndarray) -> None:
    plt, _, _ = _import_dependencies()
    fig = plt.figure(figsize=(10, 8), dpi=170)
    fig.patch.set_facecolor("#f7f1e3")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f7f1e3")
    for edge in source_graph.get("edges", []):
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if points.ndim == 2 and len(points) >= 2:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#94a3b8", alpha=0.22, linewidth=0.8)
    for edge in learned_graph.get("edges", []):
        points = np.asarray(edge.get("polyline_mm", []), dtype=float)
        if points.ndim != 2 or len(points) < 2:
            continue
        notes = tuple(str(note) for note in edge.get("notes", []))
        color = "#dc2626" if "polyline_replaced_from_population_learned_aorta_model" in notes else "#2563eb"
        width = 2.8 if color == "#dc2626" else 1.1
        alpha = 0.95 if color == "#dc2626" else 0.45
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=alpha, linewidth=width)
    if len(learned_points):
        ax.scatter(learned_points[:, 0], learned_points[:, 1], learned_points[:, 2], s=9, color="#111827", alpha=0.55)
    all_points = [learned_points] + [
        np.asarray(edge.get("polyline_mm", []), dtype=float)
        for edge in learned_graph.get("edges", [])
        if len(edge.get("polyline_mm", [])) >= 2
    ]
    cloud = np.vstack([points[:, :3] for points in all_points if points.ndim == 2 and len(points)])
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2.0
    radius = float((cloud.max(axis=0) - cloud.min(axis=0)).max() / 2.0) * 1.15
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_title("Population-Learned Aorta Graph\nred = learned aorta trunk, blue = retained vessels")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_graph_report(path: Path, result: LearnedAortaGraphResult) -> None:
    lines = [
        "# Population-Learned Aorta Vascular Graph",
        "",
        f"Case ID: `{result.case_id}`",
        f"Readiness: `{result.readiness_status}`",
        f"Source graph: `{result.source_graph_path}`",
        f"Aorta model: `{result.model_yaml_path}`",
        "",
        "## Summary",
        "",
        f"- Replaced aorta nodes: {result.replaced_node_count}",
        f"- Replaced aorta edges: {result.replaced_edge_count}",
        f"- Learned centerline points: {result.learned_centerline_points}",
        f"- Learned centerline length: {result.learned_centerline_length_mm:.3f} mm",
        f"- Source benchmark mean LOO RMSE: {result.mean_model_loo_rmse_mm:.3f} mm",
        "",
        "## Outputs",
        "",
        f"- Graph YAML: `{result.graph_yaml_path}`",
        f"- Nodes CSV: `{result.nodes_csv_path}`",
        f"- Edges CSV: `{result.edges_csv_path}`",
        f"- Learned centerline CSV: `{result.centerline_csv_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        "",
        "## Interpretation",
        "",
        "- Aorta-trunk geometry is now generated from the 20-case population aorta model and registered to the target graph endpoints.",
        "- Existing graph topology, source/target node IDs, vessel type, flow role, and non-aorta edges are preserved.",
        "- This improves the aorta trunk generation, but branch-rich arteries and venous paths still require CTA/CTV labels.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def apply_learned_aorta_to_vascular_graph(
    *,
    graph_yaml_path: str | Path,
    aorta_model_path: str | Path,
    output_dir: str | Path = "outputs/digital/vascular_network_learned_aorta",
    case_id: str = "population_learned_aorta_graph",
    source_node_id: str = "aorta_inlet",
    target_node_id: str = "descending_aorta_mid",
    edge_flow_role: str = "aorta_trunk",
    point_count: int = 64,
    radius_scale: float = 1.0,
    max_radius_mm: float | None = None,
    minimum_target_span_mm: float | None = None,
    report_path: str | Path | None = "outputs/reports/population_learned_aorta_graph_stage001.md",
) -> LearnedAortaGraphResult:
    model_spec, model_arrays = _load_model(aorta_model_path)
    source_graph_path = Path(graph_yaml_path)
    source_graph = _load_yaml(source_graph_path)
    graph = copy.deepcopy(source_graph)
    nodes = [copy.deepcopy(node) for node in graph.get("nodes", [])]
    edges = [copy.deepcopy(edge) for edge in graph.get("edges", [])]
    lookup = _node_lookup({"nodes": nodes})
    if source_node_id not in lookup or target_node_id not in lookup:
        raise ValueError(f"Graph must contain source/target nodes: {source_node_id}, {target_node_id}")
    source = np.asarray(lookup[source_node_id].get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
    target = np.asarray(lookup[target_node_id].get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
    learned_points, learned_radii = _learned_segment(
        model_arrays,
        source,
        target,
        point_count=point_count,
        radius_scale=radius_scale,
        max_radius_mm=max_radius_mm,
        minimum_target_span_mm=minimum_target_span_mm,
    )
    u_full = np.linspace(0.0, 1.0, len(learned_points), dtype=float)
    axis = target - source
    axis_denominator = float(np.dot(axis, axis))
    if axis_denominator <= 0.0:
        raise ValueError("Aorta source and target nodes have identical coordinates")

    def fraction_for_node(node: dict[str, Any]) -> float:
        position = np.asarray(node.get("position_mm", [0.0, 0.0, 0.0]), dtype=float)
        return float(np.clip(np.dot(position - source, axis) / axis_denominator, 0.0, 1.0))

    def interpolate_learned(u_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(u_values, dtype=float)
        points = np.column_stack([np.interp(values, u_full, learned_points[:, axis_index]) for axis_index in range(3)])
        radii = np.interp(values, u_full, learned_radii)
        return points, radii

    trunk_edges = [
        edge
        for edge in edges
        if str(edge.get("flow_role", "")) == edge_flow_role
        and str(edge.get("source", "")) in lookup
        and str(edge.get("target", "")) in lookup
    ]
    if not trunk_edges:
        raise ValueError(f"No edge found with flow_role={edge_flow_role!r}")
    trunk_node_ids: set[str] = {source_node_id, target_node_id}
    for edge in trunk_edges:
        trunk_node_ids.add(str(edge.get("source")))
        trunk_node_ids.add(str(edge.get("target")))

    replaced_nodes = 0
    node_fractions: dict[str, float] = {}
    for node_id in sorted(trunk_node_ids):
        node = lookup[node_id]
        fraction = fraction_for_node(node)
        node_fractions[node_id] = fraction
        point, radius = interpolate_learned(np.asarray([fraction], dtype=float))
        node["position_mm"] = [float(value) for value in point[0]]
        node["radius_mm"] = float(radius[0])
        _append_note(node, "position_replaced_from_population_learned_aorta_model")
        replaced_nodes += 1

    replaced_edges = 0
    for edge in trunk_edges:
        source_id = str(edge.get("source"))
        target_id = str(edge.get("target"))
        start_fraction = node_fractions[source_id]
        end_fraction = node_fractions[target_id]
        segment_count = max(2, int(round(abs(end_fraction - start_fraction) * max(point_count - 1, 1))) + 1)
        segment_u = np.linspace(start_fraction, end_fraction, segment_count, dtype=float)
        segment_points, segment_radii = interpolate_learned(segment_u)
        segment_points[0] = np.asarray(lookup[source_id].get("position_mm", segment_points[0]), dtype=float)
        segment_points[-1] = np.asarray(lookup[target_id].get("position_mm", segment_points[-1]), dtype=float)
        edge["polyline_mm"] = [[float(value) for value in point] for point in segment_points]
        edge["radius_start_mm"] = float(segment_radii[0])
        edge["radius_end_mm"] = float(segment_radii[-1])
        edge["radius_profile_mm"] = [float(value) for value in segment_radii]
        edge["length_mm"] = _line_length(segment_points)
        _append_note(edge, "polyline_replaced_from_population_learned_aorta_model")
        _append_note(edge, "radius_profile_replaced_from_population_aorta_model")
        replaced_edges += 1

    graph["case_id"] = case_id
    graph["nodes"] = nodes
    graph["edges"] = edges
    metadata = dict(graph.get("graph_metadata", {}))
    metadata.update(
        {
            "source_graph": str(source_graph_path),
            "population_learned_aorta_model": str(aorta_model_path),
            "learned_aorta_model_dataset_id": model_spec.get("dataset_id", ""),
            "learned_aorta_replaced_node_count": replaced_nodes,
            "learned_aorta_replaced_edge_count": replaced_edges,
            "learned_aorta_centerline_points": int(len(learned_points)),
            "learned_aorta_centerline_length_mm": _line_length(learned_points),
            "learned_aorta_mean_loo_rmse_mm": float(model_spec.get("mean_loo_rmse_mm", 0.0)),
            "learned_aorta_radius_cap_mm": None if max_radius_mm is None else float(max_radius_mm),
            "learned_aorta_minimum_target_span_mm": None if minimum_target_span_mm is None else float(minimum_target_span_mm),
            "geometry_status": (
                "population_learned_aorta_trunk_registered_with_minimum_span_floor"
                if minimum_target_span_mm is not None
                else "population_learned_aorta_trunk_registered_to_target_graph_endpoints"
            ),
        }
    )
    graph["graph_metadata"] = metadata
    provenance = [str(note) for note in graph.get("provenance_notes", [])]
    provenance.append("aorta_trunk_replaced_from_20_case_population_learned_aorta_model")
    graph["provenance_notes"] = sorted(set(provenance))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_yaml = output / f"{case_id}_learned_aorta_vascular_graph_v001.yaml"
    nodes_csv = output / f"{case_id}_learned_aorta_nodes_v001.csv"
    edges_csv = output / f"{case_id}_learned_aorta_edges_v001.csv"
    centerline_csv = output / f"{case_id}_learned_aorta_centerline_v001.csv"
    preview_png = output / f"{case_id}_learned_aorta_graph_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_learned_aorta_graph_report_v001.md"

    graph_yaml.write_text(yaml.safe_dump(graph, sort_keys=False))
    _write_graph_nodes_csv(nodes_csv, nodes)
    _write_graph_edges_csv(edges_csv, edges)
    learned_sample = np.column_stack([np.linspace(0.0, 1.0, len(learned_points)), learned_points, learned_radii])
    _write_centerline_csv(centerline_csv, learned_sample)
    _write_learned_graph_preview(preview_png, source_graph, graph, learned_points)
    readiness = "learned_aorta_graph_ready" if replaced_edges > 0 else "learned_aorta_graph_review_required"
    notes = (
        "aorta_geometry_is_population_learned_and_endpoint_registered",
        (
            "minimum_target_span_floor_applied_to_prevent_endpoint_compression"
            if minimum_target_span_mm is not None
            else "no_minimum_target_span_floor_applied"
        ),
        "non_aorta_edges_preserved_from_source_graph",
        "branch_rich_vessels_still_require_real_CTA_or_CTV_label_replacement",
    )
    result = LearnedAortaGraphResult(
        case_id=case_id,
        output_dir=str(output),
        graph_yaml_path=str(graph_yaml),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        centerline_csv_path=str(centerline_csv),
        preview_png_path=str(preview_png),
        report_path=str(report),
        source_graph_path=str(source_graph_path),
        model_yaml_path=str(aorta_model_path),
        replaced_node_count=replaced_nodes,
        replaced_edge_count=replaced_edges,
        learned_centerline_points=int(len(learned_points)),
        learned_centerline_length_mm=_line_length(learned_points),
        mean_model_loo_rmse_mm=float(model_spec.get("mean_loo_rmse_mm", 0.0)),
        readiness_status=readiness,
        notes=notes,
    )
    _write_graph_report(report, result)
    return result


def format_aorta_registration_benchmark_result(result: AortaRegistrationBenchmarkResult) -> str:
    return "\n".join(
        [
            "Aorta registration benchmark built",
            f"Dataset ID: {result.dataset_id}",
            f"Readiness: {result.readiness_status}",
            f"Cases: {result.case_count}",
            f"Mean/max LOO RMSE: {result.mean_loo_rmse_mm:.3f}/{result.max_loo_rmse_mm:.3f} mm",
            f"Mean radius MAE: {result.mean_loo_radius_mae_mm:.3f} mm",
            f"Model YAML: {result.model_yaml_path}",
            f"Atlas PNG: {result.atlas_png_path}",
        ]
    )


def format_learned_aorta_graph_result(result: LearnedAortaGraphResult) -> str:
    return "\n".join(
        [
            "Population-learned aorta graph built",
            f"Case ID: {result.case_id}",
            f"Readiness: {result.readiness_status}",
            f"Replaced nodes/edges: {result.replaced_node_count}/{result.replaced_edge_count}",
            f"Learned centerline: {result.learned_centerline_points} points, {result.learned_centerline_length_mm:.3f} mm",
            f"Graph YAML: {result.graph_yaml_path}",
            f"Report: {result.report_path}",
        ]
    )
