from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np


def _import_core_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RT planning bundle export requires matplotlib, nibabel, and PyYAML.") from exc
    return plt, nib, yaml


def _import_dicom_dependencies():
    try:
        import pydicom  # type: ignore
        from pydicom.dataset import Dataset, FileDataset, FileMetaDataset  # type: ignore
        from pydicom.sequence import Sequence  # type: ignore
        from pydicom.tag import Tag  # type: ignore
        from pydicom.uid import (  # type: ignore
            CTImageStorage,
            ExplicitVRLittleEndian,
            RTDoseStorage,
            RTStructureSetStorage,
            generate_uid,
        )
    except ImportError as exc:
        raise RuntimeError("DICOM export requires pydicom. Install with: python -m pip install pydicom") from exc
    return {
        "pydicom": pydicom,
        "Dataset": Dataset,
        "FileDataset": FileDataset,
        "FileMetaDataset": FileMetaDataset,
        "Sequence": Sequence,
        "Tag": Tag,
        "CTImageStorage": CTImageStorage,
        "ExplicitVRLittleEndian": ExplicitVRLittleEndian,
        "RTDoseStorage": RTDoseStorage,
        "RTStructureSetStorage": RTStructureSetStorage,
        "generate_uid": generate_uid,
    }


@dataclass(frozen=True)
class DoseMetric:
    mask_id: str
    label: str
    role: str
    state: str
    volume_cm3: float
    min_dose_gy: float
    mean_dose_gy: float
    max_dose_gy: float
    d2_gy: float
    d50_gy: float
    d95_gy: float
    v95_percent: float
    v100_percent: float


@dataclass(frozen=True)
class DoseComparison:
    mask_id: str
    label: str
    role: str
    comparison_state: str
    static_mean_gy: float
    evaluated_mean_gy: float
    delta_mean_gy: float
    delta_mean_percent: float
    static_d95_gy: float
    evaluated_d95_gy: float
    delta_d95_gy: float
    static_v95_percent: float
    evaluated_v95_percent: float
    delta_v95_percentage_points: float


@dataclass(frozen=True)
class RTPlanningBundleResult:
    case_id: str
    output_dir: str
    bundle_spec_yaml_path: str
    report_path: str
    preview_png_path: str
    dose_metrics_csv_path: str
    dose_comparison_csv_path: str
    pymedphys_eval_config_yaml_path: str
    static_dose_nifti_path: str
    pulsatile_mean_dose_nifti_path: str
    pulsatile_peak_dose_nifti_path: str
    pulsatile_trough_dose_nifti_path: str
    pulsatile_delta_dose_nifti_path: str
    dicom_ct_dir: str | None
    dicom_rtstruct_path: str | None
    dicom_rtdose_paths: tuple[str, ...]
    prescription_dose_gy: float
    target_mask_id: str
    flow_amplitude_fraction: float
    vascular_dose_sensitivity: float
    metrics: tuple[DoseMetric, ...]
    comparisons: tuple[DoseComparison, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SpatialRTFlowEdgeCoupling:
    edge_id: str
    vessel_type: str
    flow_role: str
    length_mm: float
    mean_radius_mm: float
    sample_count: int
    valid_sample_fraction: float
    mean_flow_ml_s: float
    min_flow_ml_s: float
    max_flow_ml_s: float
    pulsatility_fraction: float
    mean_velocity_cm_s: float
    mean_pressure_drop_mmhg: float
    min_distance_to_ptv_mm: float
    min_distance_to_gtv_mm: float
    effective_distance_to_ptv_mm: float
    mean_static_dose_gy: float
    mean_peak_delta_gy: float
    mean_trough_delta_gy: float
    peak_to_trough_delta_gy: float
    ptv_sample_fraction: float
    gtv_sample_fraction: float
    vascular_fluid_sample_fraction: float
    vessel_wall_sample_fraction: float
    body_sample_fraction: float
    dominant_region: str
    coupling_score: float


@dataclass(frozen=True)
class SpatialRTFlowCouplingResult:
    case_id: str
    output_dir: str
    edge_coupling_csv_path: str
    coupling_spec_yaml_path: str
    preview_png_path: str
    report_path: str
    rt_package_spec_path: str
    rt_planning_spec_path: str
    vascular_graph_path: str
    edge_timeseries_csv_path: str
    coordinate_mode: str
    sample_step_mm: float
    influence_radius_mm: float
    edge_count: int
    top_edges: tuple[SpatialRTFlowEdgeCoupling, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SpatialRTFlowDoseResult:
    case_id: str
    output_dir: str
    dose_model_spec_yaml_path: str
    report_path: str
    preview_png_path: str
    dose_metrics_csv_path: str
    dose_comparison_csv_path: str
    edge_contribution_csv_path: str
    phase_summary_csv_path: str
    pymedphys_eval_config_yaml_path: str
    static_dose_nifti_path: str
    spatial_mean_dose_nifti_path: str
    spatial_peak_dose_nifti_path: str
    spatial_trough_dose_nifti_path: str
    spatial_delta_dose_nifti_path: str
    spatial_influence_nifti_path: str
    rt_package_spec_path: str
    rt_planning_spec_path: str
    vascular_graph_path: str
    edge_timeseries_csv_path: str
    edge_coupling_csv_path: str
    coordinate_mode: str
    sample_step_mm: float
    influence_falloff_mm: float
    vascular_dose_sensitivity: float
    max_fractional_perturbation: float
    selected_edge_count: int
    peak_phase: float
    trough_phase: float
    peak_time_s: float
    trough_time_s: float
    max_abs_peak_delta_gy: float
    max_abs_trough_delta_gy: float
    metrics: tuple[DoseMetric, ...]
    comparisons: tuple[DoseComparison, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _CTSliceRef:
    path: str
    sop_instance_uid: str
    sop_class_uid: str
    image_position: tuple[float, float, float]
    slice_index: int


def _load_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_core_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(raw_path: str | Path, reference_yaml_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    candidate = reference_yaml_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _spacing_from_affine(image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _voxel_volume_cm3(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(spacing_mm) / 1000.0)


def _write_nifti(path: Path, data: np.ndarray, reference_image) -> None:
    _, nib, _ = _import_core_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _mask_by_id(rt_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    masks = rt_spec.get("masks", [])
    if not isinstance(masks, list):
        raise ValueError("RT package spec is missing a masks list")
    result: dict[str, dict[str, Any]] = {}
    for mask in masks:
        if isinstance(mask, dict) and "mask_id" in mask:
            result[str(mask["mask_id"])] = mask
    return result


def _load_masks(rt_spec: dict[str, Any], spec_path: Path, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    _, nib, _ = _import_core_dependencies()
    masks: dict[str, np.ndarray] = {}
    for mask_id, payload in _mask_by_id(rt_spec).items():
        mask_path = _resolve_path(str(payload["path"]), spec_path)
        image = nib.load(str(mask_path))
        data = np.asanyarray(image.dataobj) > 0
        if data.shape != shape:
            raise ValueError(f"Mask {mask_id} shape {data.shape} does not match RT volume shape {shape}")
        masks[mask_id] = data
    return masks


def _target_center(rt_spec: dict[str, Any], shape: tuple[int, int, int]) -> tuple[int, int, int]:
    target = rt_spec.get("synthetic_target", {})
    center = target.get("center_ijk") if isinstance(target, dict) else None
    if isinstance(center, list | tuple) and len(center) == 3:
        return tuple(int(round(float(value))) for value in center)  # type: ignore[return-value]
    return tuple(int((value - 1) // 2) for value in shape)


def _target_radius_from_mask(
    mask: np.ndarray,
    center_ijk: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    fallback_radius_mm: float,
) -> float:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return fallback_radius_mm
    center = np.array(center_ijk, dtype=float)
    spacing = np.array(spacing_mm, dtype=float)
    deltas = (coords.astype(float) - center) * spacing
    return float(max(np.sqrt(np.sum(deltas**2, axis=1)).max(), fallback_radius_mm))


def _distance_grid_mm(
    shape: tuple[int, int, int],
    center_ijk: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    x2 = ((np.arange(shape[0], dtype=np.float32) - float(center_ijk[0])) * spacing_mm[0]) ** 2
    y2 = ((np.arange(shape[1], dtype=np.float32) - float(center_ijk[1])) * spacing_mm[1]) ** 2
    z2 = ((np.arange(shape[2], dtype=np.float32) - float(center_ijk[2])) * spacing_mm[2]) ** 2
    return np.sqrt(x2[:, None, None] + y2[None, :, None] + z2[None, None, :]).astype(np.float32)


def _build_static_dose(
    red: np.ndarray,
    body_mask: np.ndarray,
    gtv_mask: np.ndarray,
    ptv_mask: np.ndarray,
    center_ijk: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    prescription_dose_gy: float,
) -> np.ndarray:
    distance = _distance_grid_mm(red.shape, center_ijk, spacing_mm)
    ptv_radius_mm = _target_radius_from_mask(
        ptv_mask,
        center_ijk,
        spacing_mm,
        fallback_radius_mm=18.0,
    )
    high_dose_radius_mm = max(ptv_radius_mm * 0.70, 8.0)
    falloff_sigma_mm = max(ptv_radius_mm * 0.42, 9.0)
    shoulder = np.maximum(distance - high_dose_radius_mm, 0.0)
    base = prescription_dose_gy * (0.12 + 0.91 * np.exp(-(shoulder**2) / (2.0 * falloff_sigma_mm**2)))
    heterogeneity = 1.0 - np.clip((red.astype(np.float32) - 1.0) * 0.035, -0.035, 0.060)
    dose = (base * heterogeneity).astype(np.float32)
    dose[ptv_mask] = np.maximum(dose[ptv_mask], np.float32(prescription_dose_gy * 0.95))
    dose[gtv_mask] = np.maximum(dose[gtv_mask], np.float32(prescription_dose_gy * 1.02))
    dose[~body_mask] = 0.0
    return dose.astype(np.float32)


def _read_flow_amplitude_fraction(coupled_flow_model_path: str | Path | None) -> tuple[float, str | None, tuple[str, ...]]:
    if coupled_flow_model_path is None:
        return 0.0, None, ("no_coupled_flow_model_supplied",)
    path = Path(coupled_flow_model_path)
    if not path.exists():
        raise FileNotFoundError(path)
    flow_spec = _load_yaml(path)
    summary = flow_spec.get("summary", {})
    if not isinstance(summary, dict):
        return 0.0, str(path), ("coupled_flow_summary_missing",)
    mean_flow = float(summary.get("arterial_inlet_flow_mean_ml_s", 0.0) or 0.0)
    min_flow = float(summary.get("arterial_inlet_flow_min_ml_s", mean_flow) or mean_flow)
    max_flow = float(summary.get("arterial_inlet_flow_max_ml_s", mean_flow) or mean_flow)
    if mean_flow <= 0.0:
        return 0.0, str(path), ("coupled_flow_mean_nonpositive",)
    amplitude = max(abs(max_flow - mean_flow), abs(mean_flow - min_flow)) / mean_flow
    return float(amplitude), str(path), ("flow_amplitude_from_coupled_model_summary",)


def _vascular_proximity_field(
    vascular_mask: np.ndarray,
    body_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    decay_mm: float = 7.5,
) -> np.ndarray:
    if not np.any(vascular_mask):
        return np.zeros(vascular_mask.shape, dtype=np.float32)
    try:
        from scipy import ndimage as ndi  # type: ignore
    except ImportError:
        proximity = vascular_mask.astype(np.float32)
        proximity[~body_mask] = 0.0
        return proximity
    distance = ndi.distance_transform_edt(~vascular_mask, sampling=spacing_mm).astype(np.float32)
    proximity = np.exp(-distance / np.float32(max(decay_mm, 0.1))).astype(np.float32)
    proximity[distance > decay_mm * 4.0] = 0.0
    proximity[~body_mask] = 0.0
    proximity[vascular_mask] = 1.0
    return proximity


def _build_pulsatile_dose_states(
    static_dose: np.ndarray,
    vascular_mask: np.ndarray,
    body_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    flow_amplitude_fraction: float,
    vascular_dose_sensitivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proximity = _vascular_proximity_field(vascular_mask, body_mask, spacing_mm)
    amplitude = max(flow_amplitude_fraction, 0.0)
    sensitivity = max(vascular_dose_sensitivity, 0.0)
    peak_fraction = min(sensitivity * amplitude, 0.05)
    trough_fraction = -0.55 * peak_fraction
    pulsatile_peak = static_dose * (1.0 - peak_fraction * proximity)
    pulsatile_trough = static_dose * (1.0 - trough_fraction * proximity)
    pulsatile_mean = (0.45 * pulsatile_peak + 0.55 * pulsatile_trough).astype(np.float32)
    delta = (pulsatile_mean - static_dose).astype(np.float32)
    return (
        pulsatile_mean.astype(np.float32),
        pulsatile_peak.astype(np.float32),
        pulsatile_trough.astype(np.float32),
        delta,
    )


def _dose_metric(
    mask_id: str,
    mask_payload: dict[str, Any],
    state: str,
    dose: np.ndarray,
    mask: np.ndarray,
    voxel_volume_cm3: float,
    prescription_dose_gy: float,
) -> DoseMetric:
    values = dose[mask].astype(np.float64)
    volume_cm3 = float(mask.sum() * voxel_volume_cm3)
    if values.size == 0:
        return DoseMetric(
            mask_id=mask_id,
            label=str(mask_payload.get("label", mask_id)),
            role=str(mask_payload.get("role", "")),
            state=state,
            volume_cm3=0.0,
            min_dose_gy=0.0,
            mean_dose_gy=0.0,
            max_dose_gy=0.0,
            d2_gy=0.0,
            d50_gy=0.0,
            d95_gy=0.0,
            v95_percent=0.0,
            v100_percent=0.0,
        )
    return DoseMetric(
        mask_id=mask_id,
        label=str(mask_payload.get("label", mask_id)),
        role=str(mask_payload.get("role", "")),
        state=state,
        volume_cm3=volume_cm3,
        min_dose_gy=float(np.min(values)),
        mean_dose_gy=float(np.mean(values)),
        max_dose_gy=float(np.max(values)),
        d2_gy=float(np.percentile(values, 98.0)),
        d50_gy=float(np.percentile(values, 50.0)),
        d95_gy=float(np.percentile(values, 5.0)),
        v95_percent=float(100.0 * np.mean(values >= prescription_dose_gy * 0.95)),
        v100_percent=float(100.0 * np.mean(values >= prescription_dose_gy)),
    )


def _compute_metrics(
    masks: dict[str, np.ndarray],
    mask_payloads: dict[str, dict[str, Any]],
    dose_states: dict[str, np.ndarray],
    voxel_volume_cm3: float,
    prescription_dose_gy: float,
) -> tuple[DoseMetric, ...]:
    metrics: list[DoseMetric] = []
    for state, dose in dose_states.items():
        for mask_id, mask in masks.items():
            metrics.append(
                _dose_metric(
                    mask_id=mask_id,
                    mask_payload=mask_payloads.get(mask_id, {"label": mask_id, "role": ""}),
                    state=state,
                    dose=dose,
                    mask=mask,
                    voxel_volume_cm3=voxel_volume_cm3,
                    prescription_dose_gy=prescription_dose_gy,
                )
            )
    return tuple(metrics)


def _compute_comparisons(metrics: tuple[DoseMetric, ...]) -> tuple[DoseComparison, ...]:
    by_key = {(metric.mask_id, metric.state): metric for metric in metrics}
    comparisons: list[DoseComparison] = []
    states = sorted({metric.state for metric in metrics if metric.state != "static"})
    mask_ids = sorted({metric.mask_id for metric in metrics})
    for mask_id in mask_ids:
        static = by_key.get((mask_id, "static"))
        if static is None:
            continue
        for state in states:
            evaluated = by_key.get((mask_id, state))
            if evaluated is None:
                continue
            delta_mean = evaluated.mean_dose_gy - static.mean_dose_gy
            delta_mean_percent = 0.0 if static.mean_dose_gy == 0.0 else 100.0 * delta_mean / static.mean_dose_gy
            comparisons.append(
                DoseComparison(
                    mask_id=mask_id,
                    label=static.label,
                    role=static.role,
                    comparison_state=state,
                    static_mean_gy=static.mean_dose_gy,
                    evaluated_mean_gy=evaluated.mean_dose_gy,
                    delta_mean_gy=delta_mean,
                    delta_mean_percent=delta_mean_percent,
                    static_d95_gy=static.d95_gy,
                    evaluated_d95_gy=evaluated.d95_gy,
                    delta_d95_gy=evaluated.d95_gy - static.d95_gy,
                    static_v95_percent=static.v95_percent,
                    evaluated_v95_percent=evaluated.v95_percent,
                    delta_v95_percentage_points=evaluated.v95_percent - static.v95_percent,
                )
            )
    return tuple(comparisons)


def _write_metrics_csv(path: Path, metrics: tuple[DoseMetric, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mask_id",
        "label",
        "role",
        "state",
        "volume_cm3",
        "min_dose_gy",
        "mean_dose_gy",
        "max_dose_gy",
        "d2_gy",
        "d50_gy",
        "d95_gy",
        "v95_percent",
        "v100_percent",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "mask_id": metric.mask_id,
                    "label": metric.label,
                    "role": metric.role,
                    "state": metric.state,
                    "volume_cm3": f"{metric.volume_cm3:.6f}",
                    "min_dose_gy": f"{metric.min_dose_gy:.6f}",
                    "mean_dose_gy": f"{metric.mean_dose_gy:.6f}",
                    "max_dose_gy": f"{metric.max_dose_gy:.6f}",
                    "d2_gy": f"{metric.d2_gy:.6f}",
                    "d50_gy": f"{metric.d50_gy:.6f}",
                    "d95_gy": f"{metric.d95_gy:.6f}",
                    "v95_percent": f"{metric.v95_percent:.6f}",
                    "v100_percent": f"{metric.v100_percent:.6f}",
                }
            )


def _write_comparison_csv(path: Path, comparisons: tuple[DoseComparison, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mask_id",
        "label",
        "role",
        "comparison_state",
        "static_mean_gy",
        "evaluated_mean_gy",
        "delta_mean_gy",
        "delta_mean_percent",
        "static_d95_gy",
        "evaluated_d95_gy",
        "delta_d95_gy",
        "static_v95_percent",
        "evaluated_v95_percent",
        "delta_v95_percentage_points",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(
                {
                    "mask_id": comparison.mask_id,
                    "label": comparison.label,
                    "role": comparison.role,
                    "comparison_state": comparison.comparison_state,
                    "static_mean_gy": f"{comparison.static_mean_gy:.6f}",
                    "evaluated_mean_gy": f"{comparison.evaluated_mean_gy:.6f}",
                    "delta_mean_gy": f"{comparison.delta_mean_gy:.6f}",
                    "delta_mean_percent": f"{comparison.delta_mean_percent:.6f}",
                    "static_d95_gy": f"{comparison.static_d95_gy:.6f}",
                    "evaluated_d95_gy": f"{comparison.evaluated_d95_gy:.6f}",
                    "delta_d95_gy": f"{comparison.delta_d95_gy:.6f}",
                    "static_v95_percent": f"{comparison.static_v95_percent:.6f}",
                    "evaluated_v95_percent": f"{comparison.evaluated_v95_percent:.6f}",
                    "delta_v95_percentage_points": f"{comparison.delta_v95_percentage_points:.6f}",
                }
            )


def _read_edge_timeseries_stats(path: str | Path) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, float]]]]:
    stats: dict[str, dict[str, float]] = {}
    rows_by_edge: dict[str, list[dict[str, float]]] = {}
    with Path(path).open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = str(row.get("edge_id", "")).strip()
            if not edge_id:
                continue
            parsed = {
                "time_s": float(row.get("time_s", 0.0) or 0.0),
                "phase": float(row.get("phase", 0.0) or 0.0),
                "flow_ml_s": float(row.get("flow_ml_s", 0.0) or 0.0),
                "mean_velocity_cm_s": float(row.get("mean_velocity_cm_s", 0.0) or 0.0),
                "pressure_drop_mmhg": float(row.get("pressure_drop_pa", 0.0) or 0.0) / 133.32236842105263,
            }
            rows_by_edge.setdefault(edge_id, []).append(parsed)
    for edge_id, rows in rows_by_edge.items():
        flows = np.array([row["flow_ml_s"] for row in rows], dtype=float)
        velocities = np.array([row["mean_velocity_cm_s"] for row in rows], dtype=float)
        pressure_drops = np.array([row["pressure_drop_mmhg"] for row in rows], dtype=float)
        mean_flow = float(np.mean(flows)) if flows.size else 0.0
        min_flow = float(np.min(flows)) if flows.size else 0.0
        max_flow = float(np.max(flows)) if flows.size else 0.0
        pulsatility = 0.0 if abs(mean_flow) <= 1e-9 else float((max_flow - min_flow) / abs(mean_flow))
        stats[edge_id] = {
            "mean_flow_ml_s": mean_flow,
            "min_flow_ml_s": min_flow,
            "max_flow_ml_s": max_flow,
            "pulsatility_fraction": pulsatility,
            "mean_velocity_cm_s": float(np.mean(velocities)) if velocities.size else 0.0,
            "mean_pressure_drop_mmhg": float(np.mean(pressure_drops)) if pressure_drops.size else 0.0,
        }
    return stats, rows_by_edge


def _read_edge_coupling_rows(path: str | Path) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    with Path(path).open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = str(row.get("edge_id", "")).strip()
            if not edge_id:
                continue
            parsed: dict[str, float | str] = {"edge_id": edge_id}
            for key, value in row.items():
                if key is None:
                    continue
                if key in {"edge_id", "vessel_type", "flow_role", "dominant_region"}:
                    parsed[key] = str(value or "")
                    continue
                try:
                    parsed[key] = float(value) if value not in (None, "") else 0.0
                except ValueError:
                    parsed[key] = str(value or "")
            rows[edge_id] = parsed
    return rows


def _edge_flow_waveforms(
    rows_by_edge: dict[str, list[dict[str, float]]],
    selected_edge_ids: tuple[str, ...],
    edge_weights: dict[str, float],
) -> tuple[list[dict[str, float]], dict[str, np.ndarray], int, int]:
    if not selected_edge_ids:
        return [], {}, 0, 0
    row_count = min((len(rows_by_edge.get(edge_id, [])) for edge_id in selected_edge_ids), default=0)
    if row_count <= 0:
        return [], {}, 0, 0
    normalised_by_edge: dict[str, np.ndarray] = {}
    for edge_id in selected_edge_ids:
        rows = rows_by_edge.get(edge_id, [])[:row_count]
        flows = np.array([row["flow_ml_s"] for row in rows], dtype=float)
        mean_flow = float(np.mean(flows)) if flows.size else 0.0
        if abs(mean_flow) <= 1e-9:
            normalised_by_edge[edge_id] = np.zeros(row_count, dtype=float)
        else:
            normalised_by_edge[edge_id] = (flows - mean_flow) / abs(mean_flow)

    total_weight = float(sum(max(edge_weights.get(edge_id, 0.0), 0.0) for edge_id in selected_edge_ids))
    if total_weight <= 0.0:
        total_weight = float(len(selected_edge_ids))
        edge_weights = {edge_id: 1.0 for edge_id in selected_edge_ids}

    summary: list[dict[str, float]] = []
    for index in range(row_count):
        weighted_norm = 0.0
        weighted_flow = 0.0
        total = 0.0
        times: list[float] = []
        phases: list[float] = []
        for edge_id in selected_edge_ids:
            weight = max(edge_weights.get(edge_id, 0.0), 0.0)
            rows = rows_by_edge.get(edge_id, [])
            if index >= len(rows):
                continue
            row = rows[index]
            weighted_norm += weight * float(normalised_by_edge[edge_id][index])
            weighted_flow += weight * float(row["flow_ml_s"])
            total += weight
            times.append(float(row["time_s"]))
            phases.append(float(row["phase"]))
        denom = total if total > 0.0 else total_weight
        summary.append(
            {
                "index": float(index),
                "time_s": float(np.mean(times)) if times else 0.0,
                "phase": float(np.mean(phases)) if phases else 0.0,
                "weighted_normalized_flow": weighted_norm / max(denom, 1e-9),
                "weighted_flow_ml_s": weighted_flow / max(denom, 1e-9),
            }
        )
    global_wave = np.array([row["weighted_normalized_flow"] for row in summary], dtype=float)
    peak_index = int(np.argmax(global_wave)) if global_wave.size else 0
    trough_index = int(np.argmin(global_wave)) if global_wave.size else 0
    return summary, normalised_by_edge, peak_index, trough_index


def _polyline_samples(polyline_mm: list[Any], sample_step_mm: float) -> tuple[np.ndarray, float]:
    points = np.array(polyline_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Edge polyline_mm must be a list of 3D coordinates")
    if points.shape[0] < 2:
        return points.reshape((-1, 3)), 0.0
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        return points[:1], 0.0
    sample_count = max(int(math.ceil(total_length / max(sample_step_mm, 0.1))) + 1, 2)
    distances = np.linspace(0.0, total_length, sample_count)
    samples = np.column_stack([np.interp(distances, cumulative, points[:, axis]) for axis in range(3)])
    return samples.astype(float), total_length


def _sample_ijk_from_mm(
    samples_mm: np.ndarray,
    reference_image,
    coordinate_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    shape = np.array(reference_image.shape[:3], dtype=int)
    if coordinate_mode == "nifti-affine":
        inverse_affine = np.linalg.inv(reference_image.affine)
        homog = np.column_stack([samples_mm, np.ones(samples_mm.shape[0], dtype=float)])
        ijk_float = (inverse_affine @ homog.T).T[:, :3]
    elif coordinate_mode == "voxel-mm":
        spacing = np.array(_spacing_from_affine(reference_image), dtype=float)
        ijk_float = samples_mm / spacing
    else:
        raise ValueError(f"Unsupported coordinate mode: {coordinate_mode}")
    ijk = np.rint(ijk_float).astype(int)
    valid = np.all((ijk >= 0) & (ijk < shape), axis=1)
    return ijk, valid


def _mask_coordinates_mm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.empty((0, 3), dtype=float)
    return coords.astype(float) * np.array(spacing_mm, dtype=float)


def _minimum_distance_mm(samples_mm: np.ndarray, target_coords_mm: np.ndarray) -> float:
    if samples_mm.size == 0 or target_coords_mm.size == 0:
        return float("nan")
    best = float("inf")
    for start in range(0, samples_mm.shape[0], 128):
        chunk = samples_mm[start : start + 128]
        distances = np.sqrt(np.sum((chunk[:, None, :] - target_coords_mm[None, :, :]) ** 2, axis=2))
        if distances.size:
            best = min(best, float(np.min(distances)))
    return best


def _sample_fraction(mask: np.ndarray, ijk: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        return 0.0
    values = mask[tuple(ijk[valid].T)]
    return float(np.mean(values)) if values.size else 0.0


def _sample_mean(volume: np.ndarray, ijk: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        return 0.0
    values = volume[tuple(ijk[valid].T)].astype(float)
    return float(np.mean(values)) if values.size else 0.0


def _dominant_region(
    masks: dict[str, np.ndarray],
    ijk: np.ndarray,
    valid: np.ndarray,
    preferred_order: tuple[str, ...],
) -> str:
    if not np.any(valid):
        return "outside_volume"
    best_region = "unassigned"
    best_fraction = 0.0
    for mask_id in preferred_order:
        mask = masks.get(mask_id)
        if mask is None:
            continue
        fraction = _sample_fraction(mask, ijk, valid)
        if fraction > best_fraction:
            best_region = mask_id
            best_fraction = fraction
    return best_region if best_fraction > 0.0 else "background"


def _load_selected_masks(
    rt_spec: dict[str, Any],
    spec_path: Path,
    shape: tuple[int, ...],
    mask_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    _, nib, _ = _import_core_dependencies()
    payloads = _mask_by_id(rt_spec)
    masks: dict[str, np.ndarray] = {}
    for mask_id in mask_ids:
        payload = payloads.get(mask_id)
        if payload is None:
            continue
        mask_path = _resolve_path(str(payload["path"]), spec_path)
        image = nib.load(str(mask_path))
        data = np.asanyarray(image.dataobj) > 0
        if data.shape != shape:
            raise ValueError(f"Mask {mask_id} shape {data.shape} does not match RT dose shape {shape}")
        masks[mask_id] = data
    return masks


def _edge_to_spatial_coupling(
    edge: dict[str, Any],
    flow_stats: dict[str, float],
    static_dose: np.ndarray,
    peak_delta_dose: np.ndarray,
    trough_delta_dose: np.ndarray,
    peak_to_trough_dose: np.ndarray,
    masks: dict[str, np.ndarray],
    ptv_coords_mm: np.ndarray,
    gtv_coords_mm: np.ndarray,
    reference_image,
    coordinate_mode: str,
    sample_step_mm: float,
    influence_radius_mm: float,
    prescription_dose_gy: float,
) -> SpatialRTFlowEdgeCoupling:
    edge_id = str(edge.get("id", "unknown_edge"))
    samples_mm, measured_length_mm = _polyline_samples(edge.get("polyline_mm", []), sample_step_mm)
    ijk, valid = _sample_ijk_from_mm(samples_mm, reference_image, coordinate_mode)
    valid_fraction = float(np.mean(valid)) if valid.size else 0.0
    radius_start = float(edge.get("radius_start_mm", edge.get("radius_mm", 0.0)) or 0.0)
    radius_end = float(edge.get("radius_end_mm", edge.get("radius_mm", radius_start)) or radius_start)
    mean_radius_mm = 0.5 * (radius_start + radius_end)
    distance_to_ptv = _minimum_distance_mm(samples_mm, ptv_coords_mm)
    distance_to_gtv = _minimum_distance_mm(samples_mm, gtv_coords_mm)
    effective_distance = max((0.0 if math.isnan(distance_to_ptv) else distance_to_ptv) - mean_radius_mm, 0.0)
    mean_static = _sample_mean(static_dose, ijk, valid)
    mean_peak_delta = _sample_mean(peak_delta_dose, ijk, valid)
    mean_trough_delta = _sample_mean(trough_delta_dose, ijk, valid)
    peak_to_trough_delta = _sample_mean(peak_to_trough_dose, ijk, valid)
    empty_mask = np.zeros(static_dose.shape, dtype=bool)
    ptv_fraction = _sample_fraction(masks.get("target_ptv_synthetic_vertebral", empty_mask), ijk, valid)
    gtv_fraction = _sample_fraction(masks.get("target_gtv_synthetic_vertebral", empty_mask), ijk, valid)
    vascular_fraction = _sample_fraction(masks.get("vascular_fluid", empty_mask), ijk, valid)
    wall_fraction = _sample_fraction(masks.get("vessel_wall", empty_mask), ijk, valid)
    body_fraction = _sample_fraction(masks.get("body", empty_mask), ijk, valid)
    dose_weight = 0.0 if prescription_dose_gy <= 0.0 else float(np.clip(mean_static / prescription_dose_gy, 0.0, 2.5))
    proximity_weight = math.exp(-effective_distance / max(influence_radius_mm, 0.1))
    coupling_score = (
        float(flow_stats.get("pulsatility_fraction", 0.0))
        * dose_weight
        * proximity_weight
        * max(valid_fraction, 0.0)
    )
    dominant_region = _dominant_region(
        masks,
        ijk,
        valid,
        (
            "target_gtv_synthetic_vertebral",
            "target_ptv_synthetic_vertebral",
            "vascular_fluid",
            "vessel_wall",
            "oar_liver",
            "oar_kidneys",
            "oar_lungs",
            "oar_bone",
            "body",
        ),
    )
    return SpatialRTFlowEdgeCoupling(
        edge_id=edge_id,
        vessel_type=str(edge.get("vessel_type", "")),
        flow_role=str(edge.get("flow_role", "")),
        length_mm=float(edge.get("length_mm", measured_length_mm) or measured_length_mm),
        mean_radius_mm=mean_radius_mm,
        sample_count=int(samples_mm.shape[0]),
        valid_sample_fraction=valid_fraction,
        mean_flow_ml_s=float(flow_stats.get("mean_flow_ml_s", 0.0)),
        min_flow_ml_s=float(flow_stats.get("min_flow_ml_s", 0.0)),
        max_flow_ml_s=float(flow_stats.get("max_flow_ml_s", 0.0)),
        pulsatility_fraction=float(flow_stats.get("pulsatility_fraction", 0.0)),
        mean_velocity_cm_s=float(flow_stats.get("mean_velocity_cm_s", 0.0)),
        mean_pressure_drop_mmhg=float(flow_stats.get("mean_pressure_drop_mmhg", 0.0)),
        min_distance_to_ptv_mm=float(distance_to_ptv),
        min_distance_to_gtv_mm=float(distance_to_gtv),
        effective_distance_to_ptv_mm=float(effective_distance),
        mean_static_dose_gy=mean_static,
        mean_peak_delta_gy=mean_peak_delta,
        mean_trough_delta_gy=mean_trough_delta,
        peak_to_trough_delta_gy=peak_to_trough_delta,
        ptv_sample_fraction=ptv_fraction,
        gtv_sample_fraction=gtv_fraction,
        vascular_fluid_sample_fraction=vascular_fraction,
        vessel_wall_sample_fraction=wall_fraction,
        body_sample_fraction=body_fraction,
        dominant_region=dominant_region,
        coupling_score=float(coupling_score),
    )


def _edge_influence_field(
    edge: dict[str, Any],
    reference_image,
    coordinate_mode: str,
    sample_step_mm: float,
    influence_falloff_mm: float,
    body_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    shape = tuple(int(value) for value in reference_image.shape[:3])
    spacing = np.array(_spacing_from_affine(reference_image), dtype=float)
    samples_mm, measured_length_mm = _polyline_samples(edge.get("polyline_mm", []), sample_step_mm)
    ijk, valid = _sample_ijk_from_mm(samples_mm, reference_image, coordinate_mode)
    valid_ijk = ijk[valid]
    field = np.zeros(shape, dtype=np.float32)
    radius_start = float(edge.get("radius_start_mm", edge.get("radius_mm", 0.0)) or 0.0)
    radius_end = float(edge.get("radius_end_mm", edge.get("radius_mm", radius_start)) or radius_start)
    mean_radius_mm = 0.5 * (radius_start + radius_end)
    falloff = max(float(influence_falloff_mm), 0.1)
    support_radius_mm = mean_radius_mm + 4.0 * falloff
    support_voxels = np.maximum(np.ceil(support_radius_mm / spacing).astype(int), 1)
    for center in np.unique(valid_ijk, axis=0):
        starts = np.maximum(center - support_voxels, 0)
        stops = np.minimum(center + support_voxels + 1, np.array(shape, dtype=int))
        slices = tuple(slice(int(starts[axis]), int(stops[axis])) for axis in range(3))
        axes = [
            (np.arange(starts[axis], stops[axis], dtype=np.float32) - float(center[axis])) * np.float32(spacing[axis])
            for axis in range(3)
        ]
        distance = np.sqrt(
            axes[0][:, None, None] ** 2
            + axes[1][None, :, None] ** 2
            + axes[2][None, None, :] ** 2
        ).astype(np.float32)
        local = np.exp(-np.maximum(distance - np.float32(mean_radius_mm), 0.0) / np.float32(falloff)).astype(np.float32)
        local[distance > np.float32(support_radius_mm)] = 0.0
        field[slices] = np.maximum(field[slices], local)
    if body_mask is not None:
        field[~body_mask] = 0.0
    stats = {
        "sample_count": float(samples_mm.shape[0]),
        "valid_sample_fraction": float(np.mean(valid)) if valid.size else 0.0,
        "field_voxel_count": float(np.count_nonzero(field > 1e-4)),
        "field_volume_cm3": float(np.count_nonzero(field > 1e-4) * _voxel_volume_cm3(tuple(float(v) for v in spacing))),
        "max_influence": float(np.max(field)) if field.size else 0.0,
        "mean_radius_mm": float(mean_radius_mm),
        "length_mm": float(edge.get("length_mm", measured_length_mm) or measured_length_mm),
    }
    return field, stats


def _unit_vector(vector: np.ndarray, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return fallback
    return tuple(float(value / norm) for value in vector)


def _world_from_ijk(affine: np.ndarray, i: float, j: float, k: float) -> tuple[float, float, float]:
    xyz = affine @ np.array([i, j, k, 1.0], dtype=float)
    return tuple(float(value) for value in xyz[:3])


def _base_dicom_dataset(
    sop_class_uid: str,
    sop_instance_uid: str,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    frame_of_reference_uid: str,
    modality: str,
    series_description: str,
):
    deps = _import_dicom_dependencies()
    FileDataset = deps["FileDataset"]
    FileMetaDataset = deps["FileMetaDataset"]
    ExplicitVRLittleEndian = deps["ExplicitVRLittleEndian"]
    generate_uid = deps["generate_uid"]

    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    now = datetime.now()
    dataset = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SpecificCharacterSet = "ISO_IR 100"
    dataset.PatientName = "PHANTOM^DIGITAL"
    dataset.PatientID = patient_id
    dataset.PatientBirthDate = ""
    dataset.PatientSex = "O"
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = frame_of_reference_uid
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Modality = modality
    dataset.StudyDate = now.strftime("%Y%m%d")
    dataset.StudyTime = now.strftime("%H%M%S")
    dataset.SeriesDate = dataset.StudyDate
    dataset.SeriesTime = dataset.StudyTime
    dataset.ContentDate = dataset.StudyDate
    dataset.ContentTime = dataset.StudyTime
    dataset.StudyDescription = "Synthetic digital phantom RT planning bundle"
    dataset.SeriesDescription = series_description
    dataset.Manufacturer = "phantom-digital-twin"
    dataset.SoftwareVersions = "phantom-digital-twin-0.1"
    return dataset


def _write_dicom_file(pydicom_module, path: Path, dataset) -> None:
    try:
        pydicom_module.dcmwrite(str(path), dataset, enforce_file_format=True)
    except TypeError:
        pydicom_module.dcmwrite(str(path), dataset, write_like_original=False)


def _write_ct_dicom_series(
    output_dir: Path,
    case_id: str,
    hu: np.ndarray,
    reference_image,
    study_uid: str,
    frame_of_reference_uid: str,
) -> tuple[str, str, tuple[_CTSliceRef, ...]]:
    deps = _import_dicom_dependencies()
    CTImageStorage = deps["CTImageStorage"]
    generate_uid = deps["generate_uid"]
    pydicom = deps["pydicom"]

    output_dir.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    affine = np.asarray(reference_image.affine, dtype=float)
    row_vec = affine[:3, 1]
    col_vec = affine[:3, 0]
    slice_vec = affine[:3, 2]
    row_spacing = max(float(np.linalg.norm(row_vec)), 1e-6)
    col_spacing = max(float(np.linalg.norm(col_vec)), 1e-6)
    slice_spacing = max(float(np.linalg.norm(slice_vec)), 1e-6)
    row_dir = _unit_vector(row_vec, (0.0, 1.0, 0.0))
    col_dir = _unit_vector(col_vec, (1.0, 0.0, 0.0))

    refs: list[_CTSliceRef] = []
    for k in range(hu.shape[2]):
        sop_uid = generate_uid()
        ds = _base_dicom_dataset(
            sop_class_uid=str(CTImageStorage),
            sop_instance_uid=sop_uid,
            patient_id=case_id,
            study_uid=study_uid,
            series_uid=series_uid,
            frame_of_reference_uid=frame_of_reference_uid,
            modality="CT",
            series_description="Synthetic HU CT series",
        )
        ds.ImageType = ["DERIVED", "SECONDARY", "SYNTHETIC"]
        ds.InstanceNumber = k + 1
        ds.Rows = hu.shape[1]
        ds.Columns = hu.shape[0]
        ds.PixelSpacing = [f"{row_spacing:.6f}", f"{col_spacing:.6f}"]
        ds.SliceThickness = f"{slice_spacing:.6f}"
        ds.SpacingBetweenSlices = f"{slice_spacing:.6f}"
        image_position = _world_from_ijk(affine, 0.0, 0.0, float(k))
        ds.ImagePositionPatient = [f"{value:.6f}" for value in image_position]
        ds.ImageOrientationPatient = [f"{value:.8f}" for value in (*row_dir, *col_dir)]
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"
        ds.RescaleType = "HU"
        ds.KVP = "120"
        ds.ConvolutionKernel = "SYNTHETIC"
        pixels = np.clip(np.rint(hu[:, :, k].T), -32768, 32767).astype("<i2")
        ds.PixelData = pixels.tobytes()
        path = output_dir / f"{case_id}_ct_{k + 1:04d}.dcm"
        _write_dicom_file(pydicom, path, ds)
        refs.append(
            _CTSliceRef(
                path=str(path),
                sop_instance_uid=sop_uid,
                sop_class_uid=str(CTImageStorage),
                image_position=image_position,
                slice_index=k,
            )
        )
    return str(output_dir), series_uid, tuple(refs)


def _contours_for_slice(
    mask_slice: np.ndarray,
    affine: np.ndarray,
    slice_index: int,
    max_contours_per_slice: int = 8,
    max_points_per_contour: int = 220,
) -> list[list[float]]:
    try:
        from skimage import measure  # type: ignore
    except ImportError:
        measure = None
    if not np.any(mask_slice):
        return []
    if measure is None:
        rows, cols = np.where(mask_slice)
        if rows.size == 0:
            return []
        row_min, row_max = float(rows.min()), float(rows.max())
        col_min, col_max = float(cols.min()), float(cols.max())
        contour = np.array(
            [
                [row_min, col_min],
                [row_min, col_max],
                [row_max, col_max],
                [row_max, col_min],
                [row_min, col_min],
            ],
            dtype=float,
        )
        contours = [contour]
    else:
        contours = measure.find_contours(mask_slice.astype(np.float32), 0.5)
        contours = sorted(contours, key=lambda values: values.shape[0], reverse=True)[:max_contours_per_slice]

    contour_data: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        if contour.shape[0] > max_points_per_contour:
            step = int(math.ceil(contour.shape[0] / max_points_per_contour))
            contour = contour[::step]
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        data: list[float] = []
        for row, col in contour:
            x, y, z = _world_from_ijk(affine, float(col), float(row), float(slice_index))
            data.extend([round(x, 3), round(y, 3), round(z, 3)])
        if len(data) >= 9:
            contour_data.append(data)
    return contour_data


def _write_rtstruct(
    path: Path,
    case_id: str,
    masks: dict[str, np.ndarray],
    mask_payloads: dict[str, dict[str, Any]],
    reference_image,
    study_uid: str,
    ct_series_uid: str,
    frame_of_reference_uid: str,
    ct_refs: tuple[_CTSliceRef, ...],
) -> str:
    deps = _import_dicom_dependencies()
    Dataset = deps["Dataset"]
    Sequence = deps["Sequence"]
    RTStructureSetStorage = deps["RTStructureSetStorage"]
    generate_uid = deps["generate_uid"]
    pydicom = deps["pydicom"]

    sop_uid = generate_uid()
    ds = _base_dicom_dataset(
        sop_class_uid=str(RTStructureSetStorage),
        sop_instance_uid=sop_uid,
        patient_id=case_id,
        study_uid=study_uid,
        series_uid=generate_uid(),
        frame_of_reference_uid=frame_of_reference_uid,
        modality="RTSTRUCT",
        series_description="Synthetic RT structure set",
    )
    ds.StructureSetLabel = f"{case_id[:13]}RS"[:16]
    ds.StructureSetName = "Synthetic phantom structures"
    ds.StructureSetDate = ds.ContentDate
    ds.StructureSetTime = ds.ContentTime

    contour_image_sequence = Sequence()
    for ref in ct_refs:
        image_ref = Dataset()
        image_ref.ReferencedSOPClassUID = ref.sop_class_uid
        image_ref.ReferencedSOPInstanceUID = ref.sop_instance_uid
        contour_image_sequence.append(image_ref)

    rt_referenced_series = Dataset()
    rt_referenced_series.SeriesInstanceUID = ct_series_uid
    rt_referenced_series.ContourImageSequence = contour_image_sequence
    rt_referenced_study = Dataset()
    rt_referenced_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    rt_referenced_study.ReferencedSOPInstanceUID = study_uid
    rt_referenced_study.RTReferencedSeriesSequence = Sequence([rt_referenced_series])
    referenced_frame = Dataset()
    referenced_frame.FrameOfReferenceUID = frame_of_reference_uid
    referenced_frame.RTReferencedStudySequence = Sequence([rt_referenced_study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([referenced_frame])

    structure_sequence = Sequence()
    roi_contour_sequence = Sequence()
    observation_sequence = Sequence()
    affine = np.asarray(reference_image.affine, dtype=float)
    color_cycle = [
        [230, 57, 70],
        [69, 123, 157],
        [42, 157, 143],
        [244, 162, 97],
        [233, 196, 106],
        [131, 56, 236],
        [255, 0, 110],
        [6, 214, 160],
        [17, 138, 178],
    ]

    for roi_number, (mask_id, mask) in enumerate(masks.items(), start=1):
        payload = mask_payloads.get(mask_id, {})
        roi = Dataset()
        roi.ROINumber = roi_number
        roi.ReferencedFrameOfReferenceUID = frame_of_reference_uid
        roi.ROIName = str(payload.get("label", mask_id))[:64]
        roi.ROIGenerationAlgorithm = "AUTOMATIC"
        structure_sequence.append(roi)

        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = roi_number
        roi_contour.ROIDisplayColor = color_cycle[(roi_number - 1) % len(color_cycle)]
        contour_sequence = Sequence()
        for ref in ct_refs:
            slice_mask = mask[:, :, ref.slice_index].T
            for contour_values in _contours_for_slice(slice_mask, affine, ref.slice_index):
                contour = Dataset()
                contour.ContourGeometricType = "CLOSED_PLANAR"
                contour.NumberOfContourPoints = len(contour_values) // 3
                contour.ContourData = [f"{value:.3f}" for value in contour_values]
                image_ref = Dataset()
                image_ref.ReferencedSOPClassUID = ref.sop_class_uid
                image_ref.ReferencedSOPInstanceUID = ref.sop_instance_uid
                contour.ContourImageSequence = Sequence([image_ref])
                contour_sequence.append(contour)
        roi_contour.ContourSequence = contour_sequence
        roi_contour_sequence.append(roi_contour)

        observation = Dataset()
        observation.ObservationNumber = roi_number
        observation.ReferencedROINumber = roi_number
        observation.RTROIInterpretedType = "ORGAN" if str(payload.get("role", "")).startswith("oar") else "CONTROL"
        observation.ROIInterpreter = ""
        observation_sequence.append(observation)

    ds.StructureSetROISequence = structure_sequence
    ds.ROIContourSequence = roi_contour_sequence
    ds.RTROIObservationsSequence = observation_sequence
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_dicom_file(pydicom, path, ds)
    return str(path)


def _write_rtdose(
    path: Path,
    case_id: str,
    dose: np.ndarray,
    reference_image,
    study_uid: str,
    frame_of_reference_uid: str,
    state_label: str,
) -> str:
    deps = _import_dicom_dependencies()
    RTDoseStorage = deps["RTDoseStorage"]
    Tag = deps["Tag"]
    generate_uid = deps["generate_uid"]
    pydicom = deps["pydicom"]
    sop_uid = generate_uid()
    ds = _base_dicom_dataset(
        sop_class_uid=str(RTDoseStorage),
        sop_instance_uid=sop_uid,
        patient_id=case_id,
        study_uid=study_uid,
        series_uid=generate_uid(),
        frame_of_reference_uid=frame_of_reference_uid,
        modality="RTDOSE",
        series_description=f"Synthetic RT dose {state_label}",
    )
    affine = np.asarray(reference_image.affine, dtype=float)
    row_vec = affine[:3, 1]
    col_vec = affine[:3, 0]
    slice_vec = affine[:3, 2]
    row_spacing = max(float(np.linalg.norm(row_vec)), 1e-6)
    col_spacing = max(float(np.linalg.norm(col_vec)), 1e-6)
    slice_spacing = max(float(np.linalg.norm(slice_vec)), 1e-6)
    row_dir = _unit_vector(row_vec, (0.0, 1.0, 0.0))
    col_dir = _unit_vector(col_vec, (1.0, 0.0, 0.0))
    max_dose = max(float(np.nanmax(dose)), 1e-6)
    scaling = max_dose / 65535.0
    pixels = np.clip(np.rint(dose / scaling), 0, 65535).astype("<u2")
    frames = np.stack([pixels[:, :, k].T for k in range(dose.shape[2])], axis=0)
    ds.ImageType = ["DERIVED", "SECONDARY", "SYNTHETIC"]
    ds.Rows = dose.shape[1]
    ds.Columns = dose.shape[0]
    ds.NumberOfFrames = dose.shape[2]
    ds.PixelSpacing = [f"{row_spacing:.6f}", f"{col_spacing:.6f}"]
    ds.SliceThickness = f"{slice_spacing:.6f}"
    ds.ImagePositionPatient = [f"{value:.6f}" for value in _world_from_ijk(affine, 0.0, 0.0, 0.0)]
    ds.ImageOrientationPatient = [f"{value:.8f}" for value in (*row_dir, *col_dir)]
    ds.FrameIncrementPointer = Tag(0x3004, 0x000C)
    ds.GridFrameOffsetVector = [f"{k * slice_spacing:.6f}" for k in range(dose.shape[2])]
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.DoseComment = f"Synthetic {state_label} dose for digital phantom QA"
    ds.DoseGridScaling = f"{scaling:.8g}"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = frames.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_dicom_file(pydicom, path, ds)
    return str(path)


def _write_preview(
    path: Path,
    static_dose: np.ndarray,
    pulsatile_mean: np.ndarray,
    delta_dose: np.ndarray,
    target_mask: np.ndarray,
    vascular_mask: np.ndarray,
    center_ijk: tuple[int, int, int],
    metrics: tuple[DoseMetric, ...],
) -> None:
    plt, _, _ = _import_core_dependencies()
    z = int(center_ijk[2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160)
    panels = [
        (static_dose[:, :, z].T, "Static synthetic dose", "magma", 0.0, max(float(static_dose.max()), 1.0)),
        (pulsatile_mean[:, :, z].T, "Pulsatile mean dose", "magma", 0.0, max(float(static_dose.max()), 1.0)),
        (delta_dose[:, :, z].T, "Pulsatile mean - static", "coolwarm", -0.08, 0.08),
    ]
    for ax, (image, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        if np.any(target_mask[:, :, z]):
            ax.contour(target_mask[:, :, z].T.astype(float), levels=[0.5], colors=["#00f5d4"], linewidths=1.4)
        if np.any(vascular_mask[:, :, z]):
            ax.contour(vascular_mask[:, :, z].T.astype(float), levels=[0.5], colors=["#48cae4"], linewidths=1.0)
        ax.set_title(title)
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    target_metrics = {
        metric.state: metric
        for metric in metrics
        if metric.mask_id == "target_ptv_synthetic_vertebral"
    }
    static = target_metrics.get("static")
    mean = target_metrics.get("pulsatile_mean")
    if static and mean:
        fig.suptitle(
            f"PTV D95 static {static.d95_gy:.2f} Gy | pulsatile mean {mean.d95_gy:.2f} Gy",
            y=1.02,
            fontsize=12,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_pymedphys_eval_config(
    path: Path,
    rt_spec: dict[str, Any],
    rt_spec_path: Path,
    result: RTPlanningBundleResult,
) -> None:
    *_, yaml = _import_core_dependencies()
    placeholder = rt_spec.get("outputs", {}).get("pymedphys_placeholder_yaml")
    placeholder_path = None
    placeholder_payload: dict[str, Any] | None = None
    if placeholder:
        candidate = _resolve_path(str(placeholder), rt_spec_path)
        if candidate.exists():
            placeholder_path = str(candidate)
            loaded = yaml.safe_load(candidate.read_text())
            placeholder_payload = loaded if isinstance(loaded, dict) else None
    try:
        import pymedphys  # type: ignore  # noqa: F401

        pymedphys_status = "available_not_executed_by_bundle_builder"
    except ImportError:
        pymedphys_status = "not_installed_locally"
    payload = {
        "case_id": result.case_id,
        "package_role": "pymedphys_ready_dose_evaluation_config",
        "source_placeholder_yaml": placeholder_path,
        "pymedphys_status": pymedphys_status,
        "reference_dose": {
            "state": "static",
            "nifti": result.static_dose_nifti_path,
            "dicom": next((path for path in result.dicom_rtdose_paths if "_static_" in Path(path).name), None),
        },
        "evaluated_doses": [
            {
                "state": "pulsatile_mean",
                "nifti": result.pulsatile_mean_dose_nifti_path,
                "dicom": next((path for path in result.dicom_rtdose_paths if "_pulsatile_mean_" in Path(path).name), None),
            },
            {
                "state": "pulsatile_peak",
                "nifti": result.pulsatile_peak_dose_nifti_path,
                "dicom": next((path for path in result.dicom_rtdose_paths if "_pulsatile_peak_" in Path(path).name), None),
            },
            {
                "state": "pulsatile_trough",
                "nifti": result.pulsatile_trough_dose_nifti_path,
                "dicom": next((path for path in result.dicom_rtdose_paths if "_pulsatile_trough_" in Path(path).name), None),
            },
        ],
        "gamma_defaults": (
            placeholder_payload.get("gamma_defaults", {})
            if isinstance(placeholder_payload, dict)
            else {
                "dose_percent_threshold": 3.0,
                "distance_mm_threshold": 3.0,
                "lower_percent_dose_cutoff": 10.0,
                "local_gamma": False,
            }
        ),
        "dose_metric_outputs": {
            "dvh_metrics_csv": result.dose_metrics_csv_path,
            "static_vs_pulsatile_comparison_csv": result.dose_comparison_csv_path,
        },
        "structure_inputs": {
            "rtstruct_dicom": result.dicom_rtstruct_path,
            "mask_manifest": rt_spec.get("outputs", {}).get("mask_manifest_csv"),
            "masks": [
                {"mask_id": mask.get("mask_id"), "role": mask.get("role"), "path": mask.get("path")}
                for mask in rt_spec.get("masks", [])
                if isinstance(mask, dict)
            ],
        },
        "notes": [
            "DVH metrics were calculated directly from NIfTI dose grids and masks by this project.",
            "Install pymedphys to run gamma comparisons against these DICOM RTDOSE or NIfTI dose grids.",
            "Synthetic dose grids are engineering test patterns, not clinically commissioned calculations.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_bundle_spec(
    path: Path,
    rt_package_spec_path: Path,
    coupled_flow_model_path: str | None,
    result: RTPlanningBundleResult,
) -> None:
    *_, yaml = _import_core_dependencies()
    payload = {
        "case_id": result.case_id,
        "bundle_type": "dicom_rt_style_planning_and_dose_metric_bundle",
        "inputs": {
            "rt_package_spec": str(rt_package_spec_path),
            "coupled_flow_model": coupled_flow_model_path,
        },
        "outputs": {
            "static_dose_nifti": result.static_dose_nifti_path,
            "pulsatile_mean_dose_nifti": result.pulsatile_mean_dose_nifti_path,
            "pulsatile_peak_dose_nifti": result.pulsatile_peak_dose_nifti_path,
            "pulsatile_trough_dose_nifti": result.pulsatile_trough_dose_nifti_path,
            "pulsatile_delta_dose_nifti": result.pulsatile_delta_dose_nifti_path,
            "dose_metrics_csv": result.dose_metrics_csv_path,
            "dose_comparison_csv": result.dose_comparison_csv_path,
            "pymedphys_eval_config_yaml": result.pymedphys_eval_config_yaml_path,
            "dicom_ct_dir": result.dicom_ct_dir,
            "dicom_rtstruct": result.dicom_rtstruct_path,
            "dicom_rtdose": list(result.dicom_rtdose_paths),
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "dose_model": {
            "prescription_dose_gy": result.prescription_dose_gy,
            "target_mask_id": result.target_mask_id,
            "flow_amplitude_fraction": result.flow_amplitude_fraction,
            "vascular_dose_sensitivity": result.vascular_dose_sensitivity,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: RTPlanningBundleResult) -> str:
    target_rows = [metric for metric in result.metrics if metric.mask_id == result.target_mask_id]
    vascular_rows = [metric for metric in result.metrics if metric.mask_id == "vascular_fluid"]
    comparison_rows = [
        comparison
        for comparison in result.comparisons
        if comparison.mask_id in {result.target_mask_id, "vascular_fluid", "vessel_wall"}
    ]
    lines = [
        "# RT Planning Bundle Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Prescription dose: {result.prescription_dose_gy:.2f} Gy",
        f"- Target metric mask: `{result.target_mask_id}`",
        f"- Coupled-flow amplitude fraction: {result.flow_amplitude_fraction:.4f}",
        f"- Vascular dose sensitivity: {result.vascular_dose_sensitivity:.4f}",
        f"- DICOM export: {'enabled' if result.dicom_rtstruct_path else 'disabled'}",
        "",
        "## Outputs",
        "",
        f"- Bundle spec: `{Path(result.bundle_spec_yaml_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Dose metrics CSV: `{Path(result.dose_metrics_csv_path).name}`",
        f"- Dose comparison CSV: `{Path(result.dose_comparison_csv_path).name}`",
        f"- PyMedPhys evaluation config: `{Path(result.pymedphys_eval_config_yaml_path).name}`",
        f"- RTSTRUCT DICOM: `{Path(result.dicom_rtstruct_path).name if result.dicom_rtstruct_path else 'not_exported'}`",
        f"- RTDOSE count: {len(result.dicom_rtdose_paths)}",
        "",
        "## Target DVH Metrics",
        "",
        "| state | mean Gy | D95 Gy | D50 Gy | D2 Gy | V95 % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in target_rows:
        lines.append(
            f"| `{metric.state}` | {metric.mean_dose_gy:.3f} | {metric.d95_gy:.3f} | "
            f"{metric.d50_gy:.3f} | {metric.d2_gy:.3f} | {metric.v95_percent:.2f} |"
        )
    if vascular_rows:
        lines.extend(
            [
                "",
                "## Vascular-Fluid Dose Metrics",
                "",
                "| state | mean Gy | D95 Gy | max Gy | V95 % |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in vascular_rows:
            lines.append(
                f"| `{metric.state}` | {metric.mean_dose_gy:.3f} | {metric.d95_gy:.3f} | "
                f"{metric.max_dose_gy:.3f} | {metric.v95_percent:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Static vs Pulsatile Comparison",
            "",
            "| mask | state | delta mean Gy | delta mean % | delta D95 Gy | delta V95 pp |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in comparison_rows:
        lines.append(
            f"| `{comparison.mask_id}` | `{comparison.comparison_state}` | "
            f"{comparison.delta_mean_gy:.5f} | {comparison.delta_mean_percent:.4f} | "
            f"{comparison.delta_d95_gy:.5f} | {comparison.delta_v95_percentage_points:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a DICOM-RT-style engineering handoff bundle, not a commissioned clinical plan.",
            "- RTSTRUCT and RTDOSE files are exported so the digital phantom can be loaded into DICOM-aware QA tools.",
            "- Static dose is a synthetic conformal test pattern based on the target mask, RED map, and body mask.",
            "- Pulsatile states perturb the dose locally around vascular-fluid voxels using the coupled-flow amplitude.",
            "- PyMedPhys is not vendored by this repo; the generated config wires the reference and evaluated dose grids for later gamma analysis.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _write_spatial_edge_coupling_csv(path: Path, edges: tuple[SpatialRTFlowEdgeCoupling, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SpatialRTFlowEdgeCoupling.__dataclass_fields__.keys())
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    field: (
                        f"{getattr(edge, field):.9g}"
                        if isinstance(getattr(edge, field), float)
                        else getattr(edge, field)
                    )
                    for field in fields
                }
            )


def _write_spatial_rt_flow_preview(
    path: Path,
    edges: tuple[SpatialRTFlowEdgeCoupling, ...],
    rows_by_edge: dict[str, list[dict[str, float]]],
) -> None:
    plt, *_ = _import_core_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not edges:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No edge couplings available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    top = tuple(sorted(edges, key=lambda edge: edge.coupling_score, reverse=True)[:10])
    ordered = top[::-1]
    vessel_colors = {"arterial": "#c0392b", "venous": "#2471a3"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Spatial RT-Flow Coupling: Edge-Level Ranking", fontsize=15, fontweight="bold")

    axes[0, 0].barh(
        [edge.edge_id for edge in ordered],
        [edge.coupling_score for edge in ordered],
        color=[vessel_colors.get(edge.vessel_type, "#566573") for edge in ordered],
    )
    axes[0, 0].set_xlabel("Coupling score")
    axes[0, 0].set_title("Top vessel segments")
    axes[0, 0].grid(axis="x", alpha=0.25)

    colors = [vessel_colors.get(edge.vessel_type, "#566573") for edge in edges]
    sizes = [40.0 + 180.0 * max(edge.pulsatility_fraction, 0.0) for edge in edges]
    axes[0, 1].scatter(
        [edge.effective_distance_to_ptv_mm for edge in edges],
        [edge.mean_static_dose_gy for edge in edges],
        s=sizes,
        c=colors,
        alpha=0.78,
        edgecolor="#1f2933",
        linewidth=0.4,
    )
    axes[0, 1].set_xlabel("Effective distance to PTV (mm)")
    axes[0, 1].set_ylabel("Mean static dose along edge (Gy)")
    axes[0, 1].set_title("Dose proximity vs. vessel motion")
    axes[0, 1].grid(alpha=0.25)

    for edge in top[:4]:
        rows = rows_by_edge.get(edge.edge_id, [])
        if not rows:
            continue
        axes[1, 0].plot(
            [row["time_s"] for row in rows],
            [row["flow_ml_s"] for row in rows],
            label=edge.edge_id,
            linewidth=1.8,
        )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Flow (mL/s)")
    axes[1, 0].set_title("Pulsatile flow traces for top edges")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=7, loc="best")

    axes[1, 1].barh(
        [edge.edge_id for edge in ordered],
        [1000.0 * edge.peak_to_trough_delta_gy for edge in ordered],
        color=[vessel_colors.get(edge.vessel_type, "#566573") for edge in ordered],
    )
    axes[1, 1].axvline(0.0, color="#1f2933", linewidth=0.8)
    axes[1, 1].set_xlabel("Mean peak-trough dose delta along edge (mGy)")
    axes[1, 1].set_title("Local pulsatile dose swing")
    axes[1, 1].grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _write_spatial_rt_flow_spec(path: Path, result: SpatialRTFlowCouplingResult) -> None:
    *_, yaml = _import_core_dependencies()
    top_edge = result.top_edges[0] if result.top_edges else None
    payload = {
        "case_id": result.case_id,
        "analysis_type": "spatial_rt_flow_edge_coupling",
        "inputs": {
            "rt_package_spec": result.rt_package_spec_path,
            "rt_planning_spec": result.rt_planning_spec_path,
            "vascular_graph": result.vascular_graph_path,
            "edge_timeseries_csv": result.edge_timeseries_csv_path,
        },
        "outputs": {
            "edge_coupling_csv": result.edge_coupling_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "parameters": {
            "coordinate_mode": result.coordinate_mode,
            "sample_step_mm": result.sample_step_mm,
            "influence_radius_mm": result.influence_radius_mm,
        },
        "summary": {
            "edge_count": result.edge_count,
            "top_edge_id": top_edge.edge_id if top_edge else None,
            "top_edge_coupling_score": top_edge.coupling_score if top_edge else None,
            "top_edge_effective_distance_to_ptv_mm": top_edge.effective_distance_to_ptv_mm if top_edge else None,
            "top_edge_mean_static_dose_gy": top_edge.mean_static_dose_gy if top_edge else None,
        },
        "top_edges": [
            {
                "rank": rank,
                "edge_id": edge.edge_id,
                "vessel_type": edge.vessel_type,
                "flow_role": edge.flow_role,
                "coupling_score": edge.coupling_score,
                "effective_distance_to_ptv_mm": edge.effective_distance_to_ptv_mm,
                "mean_static_dose_gy": edge.mean_static_dose_gy,
                "pulsatility_fraction": edge.pulsatility_fraction,
                "peak_to_trough_delta_gy": edge.peak_to_trough_delta_gy,
                "dominant_region": edge.dominant_region,
            }
            for rank, edge in enumerate(result.top_edges, start=1)
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_spatial_rt_flow_report(result: SpatialRTFlowCouplingResult) -> str:
    lines = [
        "# Spatial RT-Flow Coupling Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Edges analyzed: {result.edge_count}",
        f"- Coordinate mode: `{result.coordinate_mode}`",
        f"- Centerline sample spacing: {result.sample_step_mm:.2f} mm",
        f"- PTV influence radius: {result.influence_radius_mm:.2f} mm",
        "",
        "## Outputs",
        "",
        f"- Edge coupling CSV: `{Path(result.edge_coupling_csv_path).name}`",
        f"- Analysis spec: `{Path(result.coupling_spec_yaml_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        "",
        "## Top Coupled Vessel Segments",
        "",
        "| rank | edge | type | role | score | PTV distance mm | static Gy | pulsatility | peak-trough mGy | dominant region |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, edge in enumerate(result.top_edges, start=1):
        lines.append(
            f"| {rank} | `{edge.edge_id}` | `{edge.vessel_type}` | `{edge.flow_role}` | "
            f"{edge.coupling_score:.5f} | {edge.effective_distance_to_ptv_mm:.2f} | "
            f"{edge.mean_static_dose_gy:.3f} | {edge.pulsatility_fraction:.3f} | "
            f"{1000.0 * edge.peak_to_trough_delta_gy:.3f} | `{edge.dominant_region}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This analysis connects the pulsatile edge time-series back to the RT dose grid by sampling each graph centerline in the phantom volume.",
            "- The coupling score combines flow pulsatility, local static dose, and distance to the synthetic PTV; it is a ranking signal, not a clinical dose recalculation.",
            "- The current graph is a synthetic major-vessel scaffold, so edge ranks should guide the next spatial dose-coupling implementation and not be interpreted as patient-specific anatomy.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def analyze_spatial_rt_flow_coupling(
    rt_package_spec_path: str | Path,
    rt_planning_spec_path: str | Path,
    vascular_graph_path: str | Path,
    edge_timeseries_csv_path: str | Path,
    output_dir: str | Path = "outputs/experiments/spatial_rt_flow_coupling",
    case_id: str = "ct_org_case0_imagetbad_case125",
    sample_step_mm: float = 2.0,
    influence_radius_mm: float = 25.0,
    coordinate_mode: str = "voxel-mm",
    report_path: str | Path | None = "outputs/reports/spatial_rt_flow_coupling_stage001.md",
) -> SpatialRTFlowCouplingResult:
    _, nib, _ = _import_core_dependencies()
    rt_spec_path = Path(rt_package_spec_path)
    plan_spec_path = Path(rt_planning_spec_path)
    graph_path = Path(vascular_graph_path)
    edge_ts_path = Path(edge_timeseries_csv_path)
    rt_spec = _load_yaml(rt_spec_path)
    plan_spec = _load_yaml(plan_spec_path)
    graph_spec = _load_yaml(graph_path)
    plan_outputs = plan_spec.get("outputs", {})
    dose_model = plan_spec.get("dose_model", {})
    if not isinstance(plan_outputs, dict) or not isinstance(dose_model, dict):
        raise ValueError("RT planning spec must contain outputs and dose_model mappings")
    static_path = _resolve_path(str(plan_outputs["static_dose_nifti"]), plan_spec_path)
    peak_path = _resolve_path(str(plan_outputs["pulsatile_peak_dose_nifti"]), plan_spec_path)
    trough_path = _resolve_path(str(plan_outputs["pulsatile_trough_dose_nifti"]), plan_spec_path)
    static_image = nib.load(str(static_path))
    peak_image = nib.load(str(peak_path))
    trough_image = nib.load(str(trough_path))
    static_dose = np.asanyarray(static_image.dataobj).astype(np.float32)
    peak_dose = np.asanyarray(peak_image.dataobj).astype(np.float32)
    trough_dose = np.asanyarray(trough_image.dataobj).astype(np.float32)
    if static_dose.shape != peak_dose.shape or static_dose.shape != trough_dose.shape:
        raise ValueError("Static, peak, and trough dose volumes must have identical shapes")
    peak_delta_dose = (peak_dose - static_dose).astype(np.float32)
    trough_delta_dose = (trough_dose - static_dose).astype(np.float32)
    peak_to_trough_dose = (peak_dose - trough_dose).astype(np.float32)

    mask_ids = (
        "body",
        "oar_lungs",
        "oar_liver",
        "oar_kidneys",
        "oar_bone",
        "vascular_fluid",
        "vessel_wall",
        "target_gtv_synthetic_vertebral",
        "target_ptv_synthetic_vertebral",
    )
    masks = _load_selected_masks(rt_spec, rt_spec_path, static_dose.shape, mask_ids)
    spacing_mm = _spacing_from_affine(static_image)
    empty_mask = np.zeros(static_dose.shape, dtype=bool)
    ptv_coords_mm = _mask_coordinates_mm(
        masks.get("target_ptv_synthetic_vertebral", empty_mask),
        spacing_mm,
    )
    gtv_coords_mm = _mask_coordinates_mm(
        masks.get("target_gtv_synthetic_vertebral", empty_mask),
        spacing_mm,
    )
    flow_stats_by_edge, rows_by_edge = _read_edge_timeseries_stats(edge_ts_path)
    raw_edges = graph_spec.get("edges", [])
    if not isinstance(raw_edges, list):
        raise ValueError("Vascular graph spec is missing an edges list")
    prescription_dose_gy = float(dose_model.get("prescription_dose_gy", 20.0) or 20.0)
    couplings = tuple(
        sorted(
            (
                _edge_to_spatial_coupling(
                    edge=edge,
                    flow_stats=flow_stats_by_edge.get(str(edge.get("id", "")), {}),
                    static_dose=static_dose,
                    peak_delta_dose=peak_delta_dose,
                    trough_delta_dose=trough_delta_dose,
                    peak_to_trough_dose=peak_to_trough_dose,
                    masks=masks,
                    ptv_coords_mm=ptv_coords_mm,
                    gtv_coords_mm=gtv_coords_mm,
                    reference_image=static_image,
                    coordinate_mode=coordinate_mode,
                    sample_step_mm=sample_step_mm,
                    influence_radius_mm=influence_radius_mm,
                    prescription_dose_gy=prescription_dose_gy,
                )
                for edge in raw_edges
                if isinstance(edge, dict)
            ),
            key=lambda edge: edge.coupling_score,
            reverse=True,
        )
    )

    output = Path(output_dir)
    edge_csv = output / f"{case_id}_spatial_rt_flow_edge_coupling_v001.csv"
    preview_png = output / f"{case_id}_spatial_rt_flow_coupling_preview_v001.png"
    spec_yaml = output / f"{case_id}_spatial_rt_flow_coupling_spec_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_spatial_rt_flow_coupling_report_v001.md"
    notes = (
        "graph_coordinates_interpreted_as_project_voxel_mm_coordinates" if coordinate_mode == "voxel-mm" else "graph_coordinates_interpreted_with_nifti_affine",
        "analysis_samples_graph_centerlines_not_full_lumen_cross_sections",
        "coupling_score_is_for_edge_ranking_not_clinical_dose_recalculation",
        "synthetic_major_vessel_graph_limits_patient_specific_interpretation",
    )
    result = SpatialRTFlowCouplingResult(
        case_id=case_id,
        output_dir=str(output),
        edge_coupling_csv_path=str(edge_csv),
        coupling_spec_yaml_path=str(spec_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        rt_package_spec_path=str(rt_spec_path),
        rt_planning_spec_path=str(plan_spec_path),
        vascular_graph_path=str(graph_path),
        edge_timeseries_csv_path=str(edge_ts_path),
        coordinate_mode=coordinate_mode,
        sample_step_mm=sample_step_mm,
        influence_radius_mm=influence_radius_mm,
        edge_count=len(couplings),
        top_edges=tuple(couplings[:10]),
        notes=notes,
    )
    _write_spatial_edge_coupling_csv(edge_csv, couplings)
    _write_spatial_rt_flow_preview(preview_png, couplings, rows_by_edge)
    _write_spatial_rt_flow_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_spatial_rt_flow_report(result))
    return result


def _write_phase_summary_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "time_s", "phase", "weighted_normalized_flow", "weighted_flow_ml_s"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": int(row["index"]),
                    "time_s": f"{row['time_s']:.9g}",
                    "phase": f"{row['phase']:.9g}",
                    "weighted_normalized_flow": f"{row['weighted_normalized_flow']:.9g}",
                    "weighted_flow_ml_s": f"{row['weighted_flow_ml_s']:.9g}",
                }
            )


def _write_edge_contribution_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "edge_id",
        "vessel_type",
        "flow_role",
        "coupling_score",
        "model_weight",
        "peak_normalized_flow",
        "trough_normalized_flow",
        "mean_normalized_flow",
        "field_voxel_count",
        "field_volume_cm3",
        "mean_radius_mm",
        "length_mm",
        "peak_mean_delta_gy",
        "trough_mean_delta_gy",
        "max_abs_peak_delta_gy",
        "max_abs_trough_delta_gy",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        f"{row.get(field, 0.0):.9g}"
                        if isinstance(row.get(field, 0.0), float)
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def _write_spatial_dose_preview(
    path: Path,
    static_dose: np.ndarray,
    peak_dose: np.ndarray,
    trough_dose: np.ndarray,
    influence: np.ndarray,
    ptv_mask: np.ndarray,
    vascular_mask: np.ndarray,
    center_ijk: tuple[int, int, int],
    edge_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, float]],
) -> None:
    plt, *_ = _import_core_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    z_index = int(np.clip(center_ijk[2], 0, static_dose.shape[2] - 1))
    peak_delta_mgy = (peak_dose - static_dose) * 1000.0
    trough_delta_mgy = (trough_dose - static_dose) * 1000.0
    vmax_delta = float(max(np.percentile(np.abs(peak_delta_mgy), 99.8), np.percentile(np.abs(trough_delta_mgy), 99.8), 1e-3))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Spatial RT-Flow Dose Model", fontsize=16, fontweight="bold")

    panels = [
        (axes[0, 0], static_dose[:, :, z_index], "Static dose (Gy)", "magma", None, None),
        (axes[0, 1], influence[:, :, z_index], "Spatial vessel influence", "viridis", 0.0, 1.0),
        (axes[0, 2], peak_delta_mgy[:, :, z_index], "Systolic high-flow delta (mGy)", "coolwarm", -vmax_delta, vmax_delta),
        (axes[1, 0], trough_delta_mgy[:, :, z_index], "Diastolic low-flow delta (mGy)", "coolwarm", -vmax_delta, vmax_delta),
    ]
    for ax, image, title, cmap, vmin, vmax in panels:
        im = ax.imshow(image.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        if np.any(ptv_mask[:, :, z_index]):
            ax.contour(ptv_mask[:, :, z_index].T.astype(float), levels=[0.5], colors=["#00ff9f"], linewidths=1.1)
        if np.any(vascular_mask[:, :, z_index]):
            ax.contour(vascular_mask[:, :, z_index].T.astype(float), levels=[0.5], colors=["#38bdf8"], linewidths=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    top_edges = sorted(edge_rows, key=lambda row: float(row.get("model_weight", 0.0)), reverse=True)[:8]
    axes[1, 1].barh(
        [str(row["edge_id"]) for row in top_edges[::-1]],
        [float(row.get("model_weight", 0.0)) for row in top_edges[::-1]],
        color="#c0392b",
    )
    axes[1, 1].set_title("Modeled edge weights")
    axes[1, 1].set_xlabel("Relative weight")
    axes[1, 1].grid(axis="x", alpha=0.25)

    if phase_rows:
        axes[1, 2].plot(
            [row["time_s"] for row in phase_rows],
            [row["weighted_normalized_flow"] for row in phase_rows],
            color="#1f77b4",
            linewidth=2.0,
        )
        peak = max(phase_rows, key=lambda row: row["weighted_normalized_flow"])
        trough = min(phase_rows, key=lambda row: row["weighted_normalized_flow"])
        axes[1, 2].scatter([peak["time_s"]], [peak["weighted_normalized_flow"]], color="#c0392b", label="peak")
        axes[1, 2].scatter([trough["time_s"]], [trough["weighted_normalized_flow"]], color="#2471a3", label="trough")
    axes[1, 2].axhline(0.0, color="#1f2933", linewidth=0.8)
    axes[1, 2].set_title("Weighted vascular flow waveform")
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("Normalized flow")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _write_spatial_dose_eval_config(path: Path, rt_spec: dict[str, Any], rt_spec_path: Path, result: SpatialRTFlowDoseResult) -> None:
    *_, yaml = _import_core_dependencies()
    placeholder_path = rt_spec.get("outputs", {}).get("pymedphys_placeholder_yaml") if isinstance(rt_spec.get("outputs", {}), dict) else None
    placeholder_payload = _load_yaml(_resolve_path(str(placeholder_path), rt_spec_path)) if placeholder_path else {}
    payload = {
        "case_id": result.case_id,
        "package_role": "pymedphys_ready_spatial_rt_flow_dose_evaluation_config",
        "source_placeholder_yaml": placeholder_path,
        "pymedphys_status": "available_not_executed_by_spatial_model_builder",
        "reference_dose": {"state": "static", "nifti": result.static_dose_nifti_path, "dicom": None},
        "evaluated_doses": [
            {"state": "spatial_pulsatile_mean", "nifti": result.spatial_mean_dose_nifti_path, "dicom": None},
            {"state": "spatial_pulsatile_peak", "nifti": result.spatial_peak_dose_nifti_path, "dicom": None},
            {"state": "spatial_pulsatile_trough", "nifti": result.spatial_trough_dose_nifti_path, "dicom": None},
        ],
        "gamma_defaults": (
            placeholder_payload.get("gamma_defaults", {})
            if isinstance(placeholder_payload, dict)
            else {
                "dose_percent_threshold": 3.0,
                "distance_mm_threshold": 3.0,
                "lower_percent_dose_cutoff": 10.0,
                "local_gamma": False,
            }
        ),
        "dose_metric_outputs": {
            "dvh_metrics_csv": result.dose_metrics_csv_path,
            "static_vs_spatial_flow_comparison_csv": result.dose_comparison_csv_path,
        },
        "structure_inputs": {
            "rtstruct_dicom": None,
            "mask_manifest": rt_spec.get("outputs", {}).get("mask_manifest_csv") if isinstance(rt_spec.get("outputs", {}), dict) else None,
            "masks": [
                {"mask_id": mask.get("mask_id"), "role": mask.get("role"), "path": mask.get("path")}
                for mask in rt_spec.get("masks", [])
                if isinstance(mask, dict)
            ],
        },
        "notes": [
            "Spatial RT-flow dose states are generated from graph edge time-series and centerline influence fields.",
            "This is an engineering dose perturbation model, not a commissioned clinical dose calculation.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_spatial_dose_spec(path: Path, result: SpatialRTFlowDoseResult) -> None:
    *_, yaml = _import_core_dependencies()
    payload = {
        "case_id": result.case_id,
        "bundle_type": "spatial_rt_flow_dose_model",
        "inputs": {
            "rt_package_spec": result.rt_package_spec_path,
            "rt_planning_spec": result.rt_planning_spec_path,
            "vascular_graph": result.vascular_graph_path,
            "edge_timeseries_csv": result.edge_timeseries_csv_path,
            "edge_coupling_csv": result.edge_coupling_csv_path,
        },
        "outputs": {
            "spatial_mean_dose_nifti": result.spatial_mean_dose_nifti_path,
            "spatial_peak_dose_nifti": result.spatial_peak_dose_nifti_path,
            "spatial_trough_dose_nifti": result.spatial_trough_dose_nifti_path,
            "spatial_delta_dose_nifti": result.spatial_delta_dose_nifti_path,
            "spatial_influence_nifti": result.spatial_influence_nifti_path,
            "dose_metrics_csv": result.dose_metrics_csv_path,
            "dose_comparison_csv": result.dose_comparison_csv_path,
            "edge_contribution_csv": result.edge_contribution_csv_path,
            "phase_summary_csv": result.phase_summary_csv_path,
            "pymedphys_eval_config_yaml": result.pymedphys_eval_config_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "dose_model": {
            "coordinate_mode": result.coordinate_mode,
            "sample_step_mm": result.sample_step_mm,
            "influence_falloff_mm": result.influence_falloff_mm,
            "vascular_dose_sensitivity": result.vascular_dose_sensitivity,
            "max_fractional_perturbation": result.max_fractional_perturbation,
            "selected_edge_count": result.selected_edge_count,
            "peak_time_s": result.peak_time_s,
            "peak_phase": result.peak_phase,
            "trough_time_s": result.trough_time_s,
            "trough_phase": result.trough_phase,
            "max_abs_peak_delta_gy": result.max_abs_peak_delta_gy,
            "max_abs_trough_delta_gy": result.max_abs_trough_delta_gy,
        },
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_spatial_dose_report(result: SpatialRTFlowDoseResult) -> str:
    target_rows = [metric for metric in result.metrics if metric.mask_id == "target_ptv_synthetic_vertebral"]
    vascular_rows = [metric for metric in result.metrics if metric.mask_id == "vascular_fluid"]
    lines = [
        "# Spatial RT-Flow Dose Model Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Selected vessel edges: {result.selected_edge_count}",
        f"- Coordinate mode: `{result.coordinate_mode}`",
        f"- Centerline sample spacing: {result.sample_step_mm:.2f} mm",
        f"- Influence falloff: {result.influence_falloff_mm:.2f} mm",
        f"- Vascular dose sensitivity: {result.vascular_dose_sensitivity:.4f}",
        f"- Peak phase/time: {result.peak_phase:.3f} / {result.peak_time_s:.3f} s",
        f"- Trough phase/time: {result.trough_phase:.3f} / {result.trough_time_s:.3f} s",
        f"- Max absolute spatial peak delta: {1000.0 * result.max_abs_peak_delta_gy:.3f} mGy",
        f"- Max absolute spatial trough delta: {1000.0 * result.max_abs_trough_delta_gy:.3f} mGy",
        "",
        "## Outputs",
        "",
        f"- Model spec: `{Path(result.dose_model_spec_yaml_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Edge contribution CSV: `{Path(result.edge_contribution_csv_path).name}`",
        f"- Phase summary CSV: `{Path(result.phase_summary_csv_path).name}`",
        f"- PyMedPhys eval config: `{Path(result.pymedphys_eval_config_yaml_path).name}`",
        "",
        "## PTV Metrics",
        "",
        "| state | mean Gy | D95 Gy | D50 Gy | D2 Gy | V95 % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in target_rows:
        lines.append(
            f"| `{metric.state}` | {metric.mean_dose_gy:.4f} | {metric.d95_gy:.4f} | "
            f"{metric.d50_gy:.4f} | {metric.d2_gy:.4f} | {metric.v95_percent:.2f} |"
        )
    if vascular_rows:
        lines.extend(["", "## Vascular-Fluid Metrics", "", "| state | mean Gy | D95 Gy | max Gy |", "| --- | ---: | ---: | ---: |"])
        for metric in vascular_rows:
            lines.append(f"| `{metric.state}` | {metric.mean_dose_gy:.4f} | {metric.d95_gy:.4f} | {metric.max_dose_gy:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Unlike the earlier scalar perturbation, this model uses each selected graph edge's centerline, relative coupling score, and pulsatile waveform.",
            "- The generated peak/trough dose volumes are spatially heterogeneous around the modeled vascular paths.",
            "- This remains a research surrogate model; it is not a Monte Carlo, TPS, or clinically commissioned dose engine.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_spatial_rt_flow_dose_model(
    rt_package_spec_path: str | Path,
    rt_planning_spec_path: str | Path,
    vascular_graph_path: str | Path,
    edge_timeseries_csv_path: str | Path,
    edge_coupling_csv_path: str | Path,
    output_dir: str | Path = "outputs/radiotherapy/spatial_rt_flow_dose",
    case_id: str = "ct_org_case0_imagetbad_case125",
    sample_step_mm: float = 2.0,
    influence_falloff_mm: float = 7.5,
    vascular_dose_sensitivity: float | None = None,
    max_fractional_perturbation: float = 0.05,
    max_edges: int = 12,
    min_coupling_score: float = 0.0,
    coordinate_mode: str = "voxel-mm",
    report_path: str | Path | None = "outputs/reports/spatial_rt_flow_dose_model_stage001.md",
) -> SpatialRTFlowDoseResult:
    _, nib, _ = _import_core_dependencies()
    rt_spec_path = Path(rt_package_spec_path)
    plan_spec_path = Path(rt_planning_spec_path)
    graph_path = Path(vascular_graph_path)
    edge_ts_path = Path(edge_timeseries_csv_path)
    edge_coupling_path = Path(edge_coupling_csv_path)
    rt_spec = _load_yaml(rt_spec_path)
    plan_spec = _load_yaml(plan_spec_path)
    graph_spec = _load_yaml(graph_path)
    plan_outputs = plan_spec.get("outputs", {})
    dose_model = plan_spec.get("dose_model", {})
    if not isinstance(plan_outputs, dict) or not isinstance(dose_model, dict):
        raise ValueError("RT planning spec must contain outputs and dose_model mappings")
    static_path = _resolve_path(str(plan_outputs["static_dose_nifti"]), plan_spec_path)
    static_image = nib.load(str(static_path))
    static_dose = np.asanyarray(static_image.dataobj).astype(np.float32)
    mask_ids = (
        "body",
        "vascular_fluid",
        "vessel_wall",
        "target_gtv_synthetic_vertebral",
        "target_ptv_synthetic_vertebral",
        "oar_lungs",
        "oar_liver",
        "oar_kidneys",
        "oar_bone",
    )
    masks = _load_selected_masks(rt_spec, rt_spec_path, static_dose.shape, mask_ids)
    body_mask = masks.get("body", static_dose > 0.0)
    vascular_mask = masks.get("vascular_fluid", np.zeros(static_dose.shape, dtype=bool))
    ptv_mask = masks.get("target_ptv_synthetic_vertebral", np.zeros(static_dose.shape, dtype=bool))
    gtv_mask = masks.get("target_gtv_synthetic_vertebral", ptv_mask)
    flow_stats_by_edge, rows_by_edge = _read_edge_timeseries_stats(edge_ts_path)
    coupling_rows = _read_edge_coupling_rows(edge_coupling_path)
    raw_edges = graph_spec.get("edges", [])
    if not isinstance(raw_edges, list):
        raise ValueError("Vascular graph spec is missing an edges list")
    edges_by_id = {str(edge.get("id", "")): edge for edge in raw_edges if isinstance(edge, dict)}
    ranked_edge_ids = [
        edge_id
        for edge_id, row in sorted(
            coupling_rows.items(),
            key=lambda item: float(item[1].get("coupling_score", 0.0) or 0.0),
            reverse=True,
        )
        if edge_id in edges_by_id and float(row.get("coupling_score", 0.0) or 0.0) >= min_coupling_score
    ][: max(max_edges, 1)]
    if not ranked_edge_ids:
        raise ValueError("No vascular edges selected for spatial RT-flow dose modeling")
    max_score = max(float(coupling_rows[edge_id].get("coupling_score", 0.0) or 0.0) for edge_id in ranked_edge_ids)
    edge_weights = {
        edge_id: float(math.sqrt(max(float(coupling_rows[edge_id].get("coupling_score", 0.0) or 0.0), 0.0) / max(max_score, 1e-12)))
        for edge_id in ranked_edge_ids
    }
    phase_rows, norm_by_edge, peak_index, trough_index = _edge_flow_waveforms(rows_by_edge, tuple(ranked_edge_ids), edge_weights)
    sensitivity = float(vascular_dose_sensitivity) if vascular_dose_sensitivity is not None else float(dose_model.get("vascular_dose_sensitivity", 0.015) or 0.015)
    peak_fraction = np.zeros(static_dose.shape, dtype=np.float32)
    trough_fraction = np.zeros(static_dose.shape, dtype=np.float32)
    mean_fraction = np.zeros(static_dose.shape, dtype=np.float32)
    influence = np.zeros(static_dose.shape, dtype=np.float32)
    contribution_rows: list[dict[str, Any]] = []
    for edge_id in ranked_edge_ids:
        edge = edges_by_id[edge_id]
        weight = float(edge_weights[edge_id])
        waveform = norm_by_edge.get(edge_id, np.zeros(len(phase_rows), dtype=float))
        peak_norm = float(waveform[peak_index]) if waveform.size else 0.0
        trough_norm = float(waveform[trough_index]) if waveform.size else 0.0
        mean_norm = float(np.mean(waveform)) if waveform.size else 0.0
        field, field_stats = _edge_influence_field(
            edge=edge,
            reference_image=static_image,
            coordinate_mode=coordinate_mode,
            sample_step_mm=sample_step_mm,
            influence_falloff_mm=influence_falloff_mm,
            body_mask=body_mask,
        )
        edge_peak_fraction = (-sensitivity * peak_norm * weight * field).astype(np.float32)
        edge_trough_fraction = (-sensitivity * trough_norm * weight * field).astype(np.float32)
        edge_mean_fraction = (-sensitivity * mean_norm * weight * field).astype(np.float32)
        peak_fraction += edge_peak_fraction
        trough_fraction += edge_trough_fraction
        mean_fraction += edge_mean_fraction
        influence = np.maximum(influence, (weight * field).astype(np.float32))
        active = field > 1e-4
        contribution_rows.append(
            {
                "edge_id": edge_id,
                "vessel_type": str(edge.get("vessel_type", "")),
                "flow_role": str(edge.get("flow_role", "")),
                "coupling_score": float(coupling_rows[edge_id].get("coupling_score", 0.0) or 0.0),
                "model_weight": weight,
                "peak_normalized_flow": peak_norm,
                "trough_normalized_flow": trough_norm,
                "mean_normalized_flow": mean_norm,
                "field_voxel_count": float(field_stats["field_voxel_count"]),
                "field_volume_cm3": float(field_stats["field_volume_cm3"]),
                "mean_radius_mm": float(field_stats["mean_radius_mm"]),
                "length_mm": float(field_stats["length_mm"]),
                "peak_mean_delta_gy": float(np.mean(static_dose[active] * edge_peak_fraction[active])) if np.any(active) else 0.0,
                "trough_mean_delta_gy": float(np.mean(static_dose[active] * edge_trough_fraction[active])) if np.any(active) else 0.0,
                "max_abs_peak_delta_gy": float(np.max(np.abs(static_dose[active] * edge_peak_fraction[active]))) if np.any(active) else 0.0,
                "max_abs_trough_delta_gy": float(np.max(np.abs(static_dose[active] * edge_trough_fraction[active]))) if np.any(active) else 0.0,
            }
        )
    limit = max(float(max_fractional_perturbation), 0.0)
    if limit > 0.0:
        peak_fraction = np.clip(peak_fraction, -limit, limit).astype(np.float32)
        trough_fraction = np.clip(trough_fraction, -limit, limit).astype(np.float32)
        mean_fraction = np.clip(mean_fraction, -limit, limit).astype(np.float32)
    spatial_peak = (static_dose * (1.0 + peak_fraction)).astype(np.float32)
    spatial_trough = (static_dose * (1.0 + trough_fraction)).astype(np.float32)
    spatial_mean = (static_dose * (1.0 + mean_fraction)).astype(np.float32)
    spatial_delta = (spatial_mean - static_dose).astype(np.float32)
    influence = np.clip(influence, 0.0, 1.0).astype(np.float32)

    output = Path(output_dir)
    dose_dir = output / "dose"
    spatial_mean_nifti = dose_dir / f"{case_id}_rt_dose_spatial_pulsatile_mean_v001.nii.gz"
    spatial_peak_nifti = dose_dir / f"{case_id}_rt_dose_spatial_pulsatile_peak_v001.nii.gz"
    spatial_trough_nifti = dose_dir / f"{case_id}_rt_dose_spatial_pulsatile_trough_v001.nii.gz"
    spatial_delta_nifti = dose_dir / f"{case_id}_rt_dose_spatial_pulsatile_mean_minus_static_v001.nii.gz"
    influence_nifti = dose_dir / f"{case_id}_rt_spatial_vascular_influence_v001.nii.gz"
    _write_nifti(spatial_mean_nifti, spatial_mean, static_image)
    _write_nifti(spatial_peak_nifti, spatial_peak, static_image)
    _write_nifti(spatial_trough_nifti, spatial_trough, static_image)
    _write_nifti(spatial_delta_nifti, spatial_delta, static_image)
    _write_nifti(influence_nifti, influence, static_image)

    dose_states = {
        "static": static_dose,
        "spatial_pulsatile_mean": spatial_mean,
        "spatial_pulsatile_peak": spatial_peak,
        "spatial_pulsatile_trough": spatial_trough,
    }
    spacing_mm = _spacing_from_affine(static_image)
    voxel_volume_cm3 = _voxel_volume_cm3(spacing_mm)
    mask_payloads = _mask_by_id(rt_spec)
    prescription_dose_gy = float(dose_model.get("prescription_dose_gy", 20.0) or 20.0)
    metrics = _compute_metrics(masks, mask_payloads, dose_states, voxel_volume_cm3, prescription_dose_gy)
    comparisons = _compute_comparisons(metrics)
    metrics_csv = output / f"{case_id}_rt_spatial_flow_dose_metrics_v001.csv"
    comparison_csv = output / f"{case_id}_rt_static_vs_spatial_flow_dose_comparison_v001.csv"
    edge_contrib_csv = output / f"{case_id}_rt_spatial_flow_edge_contributions_v001.csv"
    phase_summary_csv = output / f"{case_id}_rt_spatial_flow_phase_summary_v001.csv"
    spec_yaml = output / f"{case_id}_rt_spatial_flow_dose_model_spec_v001.yaml"
    preview_png = output / f"{case_id}_rt_spatial_flow_dose_model_preview_v001.png"
    pymedphys_config = output / f"{case_id}_pymedphys_spatial_flow_dose_eval_config_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_rt_spatial_flow_dose_model_report_v001.md"
    _write_metrics_csv(metrics_csv, metrics)
    _write_comparison_csv(comparison_csv, comparisons)
    _write_edge_contribution_csv(edge_contrib_csv, contribution_rows)
    _write_phase_summary_csv(phase_summary_csv, phase_rows)

    peak_row = phase_rows[peak_index] if phase_rows else {"phase": 0.0, "time_s": 0.0}
    trough_row = phase_rows[trough_index] if phase_rows else {"phase": 0.0, "time_s": 0.0}
    notes = (
        "spatial_dose_perturbation_uses_edge_specific_flow_waveforms",
        "edge_influence_fields_are_centerline_distance_surrogates_not_cfd_voxel_fields",
        "dose_states_are_research_surrogates_not_clinically_commissioned_calculations",
        "graph_coordinates_interpreted_as_project_voxel_mm_coordinates" if coordinate_mode == "voxel-mm" else "graph_coordinates_interpreted_with_nifti_affine",
    )
    result = SpatialRTFlowDoseResult(
        case_id=case_id,
        output_dir=str(output),
        dose_model_spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        preview_png_path=str(preview_png),
        dose_metrics_csv_path=str(metrics_csv),
        dose_comparison_csv_path=str(comparison_csv),
        edge_contribution_csv_path=str(edge_contrib_csv),
        phase_summary_csv_path=str(phase_summary_csv),
        pymedphys_eval_config_yaml_path=str(pymedphys_config),
        static_dose_nifti_path=str(static_path),
        spatial_mean_dose_nifti_path=str(spatial_mean_nifti),
        spatial_peak_dose_nifti_path=str(spatial_peak_nifti),
        spatial_trough_dose_nifti_path=str(spatial_trough_nifti),
        spatial_delta_dose_nifti_path=str(spatial_delta_nifti),
        spatial_influence_nifti_path=str(influence_nifti),
        rt_package_spec_path=str(rt_spec_path),
        rt_planning_spec_path=str(plan_spec_path),
        vascular_graph_path=str(graph_path),
        edge_timeseries_csv_path=str(edge_ts_path),
        edge_coupling_csv_path=str(edge_coupling_path),
        coordinate_mode=coordinate_mode,
        sample_step_mm=sample_step_mm,
        influence_falloff_mm=influence_falloff_mm,
        vascular_dose_sensitivity=sensitivity,
        max_fractional_perturbation=float(max_fractional_perturbation),
        selected_edge_count=len(ranked_edge_ids),
        peak_phase=float(peak_row["phase"]),
        trough_phase=float(trough_row["phase"]),
        peak_time_s=float(peak_row["time_s"]),
        trough_time_s=float(trough_row["time_s"]),
        max_abs_peak_delta_gy=float(np.max(np.abs(spatial_peak - static_dose))),
        max_abs_trough_delta_gy=float(np.max(np.abs(spatial_trough - static_dose))),
        metrics=metrics,
        comparisons=comparisons,
        notes=notes,
    )
    _write_spatial_dose_preview(
        preview_png,
        static_dose,
        spatial_peak,
        spatial_trough,
        influence,
        ptv_mask,
        vascular_mask,
        _target_center(rt_spec, static_dose.shape),
        contribution_rows,
        phase_rows,
    )
    _write_spatial_dose_eval_config(pymedphys_config, rt_spec, rt_spec_path, result)
    _write_spatial_dose_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_spatial_dose_report(result))
    return result


def build_rt_planning_bundle(
    rt_package_spec_path: str | Path,
    coupled_flow_model_path: str | Path | None = None,
    output_dir: str | Path = "outputs/radiotherapy/planning_bundle",
    case_id: str = "ct_org_case0_imagetbad_case125",
    prescription_dose_gy: float = 20.0,
    vascular_dose_sensitivity: float = 0.015,
    export_dicom: bool = True,
    report_path: str | Path | None = "outputs/reports/rt_planning_bundle_stage001.md",
) -> RTPlanningBundleResult:
    _, nib, _ = _import_core_dependencies()
    rt_spec_path = Path(rt_package_spec_path)
    rt_spec = _load_yaml(rt_spec_path)
    outputs = rt_spec.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("RT package spec is missing outputs")
    hu_path = _resolve_path(str(outputs["rt_synthetic_hu"]), rt_spec_path)
    red_path = _resolve_path(str(outputs["rt_relative_electron_density"]), rt_spec_path)
    label_path = _resolve_path(str(outputs["rt_material_labels"]), rt_spec_path)
    hu_image = nib.load(str(hu_path))
    red_image = nib.load(str(red_path))
    label_image = nib.load(str(label_path))
    hu = np.asanyarray(hu_image.dataobj).astype(np.float32)
    red = np.asanyarray(red_image.dataobj).astype(np.float32)
    labels = np.rint(np.asanyarray(label_image.dataobj)).astype(np.int16)
    if hu.shape != red.shape or hu.shape != labels.shape:
        raise ValueError("RT HU, RED, and label maps must have identical shapes")

    masks = _load_masks(rt_spec, rt_spec_path, hu.shape)
    mask_payloads = _mask_by_id(rt_spec)
    body_mask = masks.get("body", labels > 0)
    gtv_mask = masks.get("target_gtv_synthetic_vertebral", np.zeros(hu.shape, dtype=bool))
    ptv_mask = masks.get("target_ptv_synthetic_vertebral", gtv_mask)
    vascular_mask = masks.get("vascular_fluid", np.isin(labels, (14, 15)))
    center_ijk = _target_center(rt_spec, hu.shape)
    spacing_mm = _spacing_from_affine(hu_image)
    voxel_volume_cm3 = _voxel_volume_cm3(spacing_mm)
    flow_amplitude, resolved_flow_model_path, flow_notes = _read_flow_amplitude_fraction(coupled_flow_model_path)

    static_dose = _build_static_dose(
        red=red,
        body_mask=body_mask,
        gtv_mask=gtv_mask,
        ptv_mask=ptv_mask,
        center_ijk=center_ijk,
        spacing_mm=spacing_mm,
        prescription_dose_gy=prescription_dose_gy,
    )
    pulsatile_mean, pulsatile_peak, pulsatile_trough, pulsatile_delta = _build_pulsatile_dose_states(
        static_dose=static_dose,
        vascular_mask=vascular_mask,
        body_mask=body_mask,
        spacing_mm=spacing_mm,
        flow_amplitude_fraction=flow_amplitude,
        vascular_dose_sensitivity=vascular_dose_sensitivity,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dose_dir = output / "dose"
    static_nifti = dose_dir / f"{case_id}_rt_dose_static_v001.nii.gz"
    pulsatile_mean_nifti = dose_dir / f"{case_id}_rt_dose_pulsatile_mean_v001.nii.gz"
    pulsatile_peak_nifti = dose_dir / f"{case_id}_rt_dose_pulsatile_peak_v001.nii.gz"
    pulsatile_trough_nifti = dose_dir / f"{case_id}_rt_dose_pulsatile_trough_v001.nii.gz"
    pulsatile_delta_nifti = dose_dir / f"{case_id}_rt_dose_pulsatile_mean_minus_static_v001.nii.gz"
    _write_nifti(static_nifti, static_dose, hu_image)
    _write_nifti(pulsatile_mean_nifti, pulsatile_mean, hu_image)
    _write_nifti(pulsatile_peak_nifti, pulsatile_peak, hu_image)
    _write_nifti(pulsatile_trough_nifti, pulsatile_trough, hu_image)
    _write_nifti(pulsatile_delta_nifti, pulsatile_delta, hu_image)

    dose_states = {
        "static": static_dose,
        "pulsatile_mean": pulsatile_mean,
        "pulsatile_peak": pulsatile_peak,
        "pulsatile_trough": pulsatile_trough,
    }
    metrics = _compute_metrics(
        masks=masks,
        mask_payloads=mask_payloads,
        dose_states=dose_states,
        voxel_volume_cm3=voxel_volume_cm3,
        prescription_dose_gy=prescription_dose_gy,
    )
    comparisons = _compute_comparisons(metrics)
    metrics_csv = output / f"{case_id}_rt_dose_metrics_v001.csv"
    comparison_csv = output / f"{case_id}_rt_static_vs_pulsatile_dose_comparison_v001.csv"
    _write_metrics_csv(metrics_csv, metrics)
    _write_comparison_csv(comparison_csv, comparisons)

    dicom_ct_dir: str | None = None
    dicom_rtstruct: str | None = None
    dicom_rtdose_paths: list[str] = []
    if export_dicom:
        deps = _import_dicom_dependencies()
        generate_uid = deps["generate_uid"]
        study_uid = generate_uid()
        frame_uid = generate_uid()
        dicom_root = output / "dicom"
        ct_dir, ct_series_uid, ct_refs = _write_ct_dicom_series(
            output_dir=dicom_root / "ct",
            case_id=case_id,
            hu=hu,
            reference_image=hu_image,
            study_uid=study_uid,
            frame_of_reference_uid=frame_uid,
        )
        dicom_ct_dir = ct_dir
        dicom_rtstruct = _write_rtstruct(
            path=dicom_root / "rtstruct" / f"{case_id}_rtstruct_v001.dcm",
            case_id=case_id,
            masks=masks,
            mask_payloads=mask_payloads,
            reference_image=hu_image,
            study_uid=study_uid,
            ct_series_uid=ct_series_uid,
            frame_of_reference_uid=frame_uid,
            ct_refs=ct_refs,
        )
        for state, dose in dose_states.items():
            dicom_rtdose_paths.append(
                _write_rtdose(
                    path=dicom_root / "rtdose" / f"{case_id}_rtdose_{state}_v001.dcm",
                    case_id=case_id,
                    dose=dose,
                    reference_image=hu_image,
                    study_uid=study_uid,
                    frame_of_reference_uid=frame_uid,
                    state_label=state,
                )
            )

    bundle_spec = output / f"{case_id}_rt_planning_bundle_spec_v001.yaml"
    preview_png = output / f"{case_id}_rt_planning_bundle_preview_v001.png"
    pymedphys_config = output / f"{case_id}_pymedphys_dose_eval_config_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_rt_planning_bundle_report_v001.md"
    notes = (
        "dicom_rt_outputs_are_synthetic_research_handoff_files",
        "static_dose_is_synthetic_not_tps_or_monte_carlo_calculated",
        "pulsatile_dose_states_use_vascular_proximity_surrogate_perturbation",
        "dose_metrics_use_nifti_masks_from_rt_qa_package",
        *flow_notes,
    )

    result = RTPlanningBundleResult(
        case_id=case_id,
        output_dir=str(output),
        bundle_spec_yaml_path=str(bundle_spec),
        report_path=str(report),
        preview_png_path=str(preview_png),
        dose_metrics_csv_path=str(metrics_csv),
        dose_comparison_csv_path=str(comparison_csv),
        pymedphys_eval_config_yaml_path=str(pymedphys_config),
        static_dose_nifti_path=str(static_nifti),
        pulsatile_mean_dose_nifti_path=str(pulsatile_mean_nifti),
        pulsatile_peak_dose_nifti_path=str(pulsatile_peak_nifti),
        pulsatile_trough_dose_nifti_path=str(pulsatile_trough_nifti),
        pulsatile_delta_dose_nifti_path=str(pulsatile_delta_nifti),
        dicom_ct_dir=dicom_ct_dir,
        dicom_rtstruct_path=dicom_rtstruct,
        dicom_rtdose_paths=tuple(dicom_rtdose_paths),
        prescription_dose_gy=prescription_dose_gy,
        target_mask_id="target_ptv_synthetic_vertebral",
        flow_amplitude_fraction=flow_amplitude,
        vascular_dose_sensitivity=vascular_dose_sensitivity,
        metrics=metrics,
        comparisons=comparisons,
        notes=notes,
    )
    _write_preview(preview_png, static_dose, pulsatile_mean, pulsatile_delta, ptv_mask, vascular_mask, center_ijk, metrics)
    _write_pymedphys_eval_config(pymedphys_config, rt_spec, rt_spec_path, result)
    _write_bundle_spec(bundle_spec, rt_spec_path, resolved_flow_model_path, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_rt_planning_bundle_result(result: RTPlanningBundleResult) -> str:
    target_static = next(
        (metric for metric in result.metrics if metric.mask_id == result.target_mask_id and metric.state == "static"),
        None,
    )
    target_mean = next(
        (
            metric
            for metric in result.metrics
            if metric.mask_id == result.target_mask_id and metric.state == "pulsatile_mean"
        ),
        None,
    )
    lines = [
        "RT planning bundle created",
        f"Case ID: {result.case_id}",
        f"Prescription: {result.prescription_dose_gy:.2f} Gy",
        f"Flow amplitude fraction: {result.flow_amplitude_fraction:.4f}",
        f"Static dose: {result.static_dose_nifti_path}",
        f"Pulsatile mean dose: {result.pulsatile_mean_dose_nifti_path}",
        f"Dose metrics CSV: {result.dose_metrics_csv_path}",
        f"Dose comparison CSV: {result.dose_comparison_csv_path}",
        f"PyMedPhys eval config: {result.pymedphys_eval_config_yaml_path}",
        f"DICOM CT dir: {result.dicom_ct_dir or 'not_exported'}",
        f"DICOM RTSTRUCT: {result.dicom_rtstruct_path or 'not_exported'}",
        f"DICOM RTDOSE count: {len(result.dicom_rtdose_paths)}",
        f"Preview PNG: {result.preview_png_path}",
        f"Spec YAML: {result.bundle_spec_yaml_path}",
        f"Report: {result.report_path}",
    ]
    if target_static and target_mean:
        lines.extend(
            [
                f"Target static D95: {target_static.d95_gy:.3f} Gy",
                f"Target pulsatile mean D95: {target_mean.d95_gy:.3f} Gy",
            ]
        )
    return "\n".join(lines)


def format_spatial_rt_flow_coupling_result(result: SpatialRTFlowCouplingResult) -> str:
    top_edge = result.top_edges[0] if result.top_edges else None
    lines = [
        "Spatial RT-flow coupling analysis created",
        f"Case ID: {result.case_id}",
        f"Edges analyzed: {result.edge_count}",
        f"Coordinate mode: {result.coordinate_mode}",
        f"Edge coupling CSV: {result.edge_coupling_csv_path}",
        f"Preview PNG: {result.preview_png_path}",
        f"Spec YAML: {result.coupling_spec_yaml_path}",
        f"Report: {result.report_path}",
    ]
    if top_edge:
        lines.extend(
            [
                f"Top edge: {top_edge.edge_id}",
                f"Top score: {top_edge.coupling_score:.5f}",
                f"Top edge effective PTV distance: {top_edge.effective_distance_to_ptv_mm:.2f} mm",
                f"Top edge mean static dose: {top_edge.mean_static_dose_gy:.3f} Gy",
            ]
        )
    return "\n".join(lines)


def format_spatial_rt_flow_dose_result(result: SpatialRTFlowDoseResult) -> str:
    target_peak = next(
        (
            metric
            for metric in result.metrics
            if metric.mask_id == "target_ptv_synthetic_vertebral" and metric.state == "spatial_pulsatile_peak"
        ),
        None,
    )
    target_trough = next(
        (
            metric
            for metric in result.metrics
            if metric.mask_id == "target_ptv_synthetic_vertebral" and metric.state == "spatial_pulsatile_trough"
        ),
        None,
    )
    lines = [
        "Spatial RT-flow dose model created",
        f"Case ID: {result.case_id}",
        f"Selected edges: {result.selected_edge_count}",
        f"Peak phase/time: {result.peak_phase:.3f} / {result.peak_time_s:.3f} s",
        f"Trough phase/time: {result.trough_phase:.3f} / {result.trough_time_s:.3f} s",
        f"Max peak delta: {1000.0 * result.max_abs_peak_delta_gy:.3f} mGy",
        f"Max trough delta: {1000.0 * result.max_abs_trough_delta_gy:.3f} mGy",
        f"Spatial peak dose: {result.spatial_peak_dose_nifti_path}",
        f"Spatial trough dose: {result.spatial_trough_dose_nifti_path}",
        f"Metrics CSV: {result.dose_metrics_csv_path}",
        f"PyMedPhys eval config: {result.pymedphys_eval_config_yaml_path}",
        f"Preview PNG: {result.preview_png_path}",
        f"Spec YAML: {result.dose_model_spec_yaml_path}",
        f"Report: {result.report_path}",
    ]
    if target_peak and target_trough:
        lines.extend(
            [
                f"PTV peak mean dose: {target_peak.mean_dose_gy:.4f} Gy",
                f"PTV trough mean dose: {target_trough.mean_dose_gy:.4f} Gy",
            ]
        )
    return "\n".join(lines)
