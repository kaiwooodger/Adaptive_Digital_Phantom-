from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import numpy as np


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Radiotherapy QA packaging requires matplotlib, nibabel, and PyYAML.") from exc
    return plt, nib, yaml


@dataclass(frozen=True)
class RTMaskStats:
    mask_id: str
    label: str
    role: str
    path: str
    voxel_count: int
    volume_cm3: float
    mean_hu: float | None
    mean_density_g_cm3: float | None
    mean_red: float | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RadiotherapyQAPackageResult:
    case_id: str
    scenario: str
    output_dir: str
    rt_hu_path: str
    rt_material_labels_path: str
    rt_density_path: str
    rt_red_path: str
    mask_manifest_csv_path: str
    material_calibration_csv_path: str
    pymedphys_placeholder_yaml_path: str
    package_spec_yaml_path: str
    preview_png_path: str
    report_path: str
    mask_stats: tuple[RTMaskStats, ...]
    target_center_mm: tuple[float, float, float]
    target_center_ijk: tuple[int, int, int]
    target_gtv_volume_cm3: float
    target_ptv_volume_cm3: float
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _spacing(image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _write_nifti(path: Path, data: np.ndarray, reference_image) -> None:
    _, nib, _ = _import_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _scenario_outputs(spec: dict[str, Any], scenario: str) -> tuple[str | None, str, str | None, str | None]:
    outputs = spec.get("outputs", {})
    prefix = scenario.lower()
    labels_key = f"{prefix}_material_labels"
    if labels_key not in outputs:
        raise ValueError(f"Combined spec is missing {scenario} output: {labels_key}")
    return (
        None if f"{prefix}_synthetic_hu" not in outputs else str(outputs[f"{prefix}_synthetic_hu"]),
        str(outputs[labels_key]),
        None if f"{prefix}_mass_density_g_cm3" not in outputs else str(outputs[f"{prefix}_mass_density_g_cm3"]),
        None
        if f"{prefix}_relative_electron_density" not in outputs
        else str(outputs[f"{prefix}_relative_electron_density"]),
    )


def _maps_from_material_labels(
    labels: np.ndarray,
    regions: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not regions:
        raise ValueError("Label-derived RT maps require regions in the combined spec")
    region_by_index = {int(region["index"]): region for region in regions}
    present_labels = {int(value) for value in np.unique(labels)}
    missing_labels = sorted(value for value in present_labels if value not in region_by_index)
    if missing_labels:
        raise ValueError(f"Label-derived RT maps are missing region definitions for labels: {missing_labels}")
    hu = np.zeros(labels.shape, dtype=np.float32)
    density = np.zeros(labels.shape, dtype=np.float32)
    red = np.zeros(labels.shape, dtype=np.float32)
    for label_index, region in region_by_index.items():
        mask = labels == label_index
        if not np.any(mask):
            continue
        hu[mask] = float(region["target_hu_midpoint"])
        density[mask] = float(region["mass_density_g_cm3"])
        red[mask] = float(region["relative_electron_density"])
    return hu, density, red


def _voxel_volume_cm3(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(spacing_mm) / 1000.0)


def _nearest_central_bone_voxel(labels: np.ndarray, spacing_mm: tuple[float, float, float]) -> tuple[int, int, int]:
    bone = np.isin(labels, (10, 11))
    coords = np.argwhere(bone)
    if coords.size == 0:
        coords = np.argwhere(labels > 0)
    if coords.size == 0:
        return tuple(int(value // 2) for value in labels.shape)
    target = (np.array(labels.shape, dtype=float) - 1.0) / 2.0
    scaled_delta = (coords.astype(float) - target) * np.array(spacing_mm, dtype=float)
    index = int(np.argmin(np.sum(scaled_delta**2, axis=1)))
    return tuple(int(value) for value in coords[index])


def _sphere_mask(
    shape: tuple[int, int, int],
    center_ijk: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    center = np.array(center_ijk, dtype=int)
    spacing = np.array(spacing_mm, dtype=float)
    radius_voxels = np.ceil(radius_mm / spacing).astype(int) + 1
    mins = np.maximum(center - radius_voxels, 0)
    maxs = np.minimum(center + radius_voxels + 1, np.array(shape, dtype=int))
    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    grids = np.meshgrid(
        *[np.arange(mins[axis], maxs[axis], dtype=float) for axis in range(3)],
        indexing="ij",
    )
    distance_sq = np.zeros(tuple(int(maxs[axis] - mins[axis]) for axis in range(3)), dtype=float)
    for axis, grid in enumerate(grids):
        distance_sq += ((grid - float(center[axis])) * spacing[axis]) ** 2
    mask[slices] = distance_sq <= radius_mm**2
    return mask


def _target_masks(
    labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    target_radius_mm: float,
    ptv_margin_mm: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], tuple[float, float, float], tuple[str, ...]]:
    center_ijk = _nearest_central_bone_voxel(labels, spacing_mm)
    center_mm = tuple(float(center_ijk[axis] * spacing_mm[axis]) for axis in range(3))
    body = labels > 0
    bone = np.isin(labels, (10, 11))
    gtv_sphere = _sphere_mask(labels.shape, center_ijk, spacing_mm, target_radius_mm)
    gtv_bone = gtv_sphere & bone
    notes = [f"target_radius_mm={target_radius_mm}", f"ptv_margin_mm={ptv_margin_mm}"]
    if int(gtv_bone.sum()) > 0:
        gtv = gtv_bone
        notes.append("gtv_constrained_to_bone_labels_10_11")
    else:
        gtv = gtv_sphere & body
        notes.append("gtv_fallback_constrained_to_body")
    ptv = _sphere_mask(labels.shape, center_ijk, spacing_mm, target_radius_mm + ptv_margin_mm) & body
    return gtv, ptv, center_ijk, center_mm, tuple(notes)


def _mask_stats(
    mask_id: str,
    label: str,
    role: str,
    path: str,
    mask: np.ndarray,
    hu: np.ndarray,
    density: np.ndarray,
    red: np.ndarray,
    voxel_volume_cm3: float,
    notes: tuple[str, ...] = (),
) -> RTMaskStats:
    voxel_count = int(mask.sum())
    volume_cm3 = float(voxel_count * voxel_volume_cm3)
    if voxel_count == 0:
        return RTMaskStats(
            mask_id=mask_id,
            label=label,
            role=role,
            path=path,
            voxel_count=0,
            volume_cm3=0.0,
            mean_hu=None,
            mean_density_g_cm3=None,
            mean_red=None,
            notes=notes,
        )
    return RTMaskStats(
        mask_id=mask_id,
        label=label,
        role=role,
        path=path,
        voxel_count=voxel_count,
        volume_cm3=volume_cm3,
        mean_hu=float(np.mean(hu[mask])),
        mean_density_g_cm3=float(np.mean(density[mask])),
        mean_red=float(np.mean(red[mask])),
        notes=notes,
    )


def _write_mask_manifest(path: Path, stats: tuple[RTMaskStats, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "mask_id",
                "label",
                "role",
                "path",
                "voxel_count",
                "volume_cm3",
                "mean_hu",
                "mean_density_g_cm3",
                "mean_relative_electron_density",
                "notes",
            ]
        )
        for stat in stats:
            writer.writerow(
                [
                    stat.mask_id,
                    stat.label,
                    stat.role,
                    stat.path,
                    stat.voxel_count,
                    f"{stat.volume_cm3:.6f}",
                    "" if stat.mean_hu is None else f"{stat.mean_hu:.6f}",
                    "" if stat.mean_density_g_cm3 is None else f"{stat.mean_density_g_cm3:.6f}",
                    "" if stat.mean_red is None else f"{stat.mean_red:.6f}",
                    ";".join(stat.notes),
                ]
            )


def _write_material_calibration(path: Path, regions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "label_index",
                "region_name",
                "material_id",
                "target_hu_midpoint",
                "mass_density_g_cm3",
                "relative_electron_density",
                "color",
            ]
        )
        for region in regions:
            writer.writerow(
                [
                    int(region["index"]),
                    region["name"],
                    region["material_id"],
                    f"{float(region['target_hu_midpoint']):.6f}",
                    f"{float(region['mass_density_g_cm3']):.6f}",
                    f"{float(region['relative_electron_density']):.6f}",
                    region.get("color", ""),
                ]
            )


def _write_pymedphys_placeholder(path: Path, result: RadiotherapyQAPackageResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "package_role": "pymedphys_ready_placeholder",
        "status": "dose_grid_not_yet_generated",
        "suggested_next_inputs": {
            "reference_dose_dicom_or_nifti": "TODO",
            "evaluated_dose_dicom_or_nifti": "TODO",
            "rtplan_or_beam_metadata": "TODO",
        },
        "gamma_defaults": {
            "dose_percent_threshold": 3.0,
            "distance_mm_threshold": 3.0,
            "lower_percent_dose_cutoff": 10.0,
            "local_gamma": False,
        },
        "dvh_masks": [
            {"mask_id": stat.mask_id, "role": stat.role, "path": stat.path}
            for stat in result.mask_stats
        ],
        "package_outputs": {
            "hu": result.rt_hu_path,
            "density": result.rt_density_path,
            "relative_electron_density": result.rt_red_path,
            "material_labels": result.rt_material_labels_path,
            "mask_manifest": result.mask_manifest_csv_path,
        },
        "notes": [
            "Use this file as a wiring manifest once calculated dose grids exist.",
            "The current package contains geometry and material inputs, not a treatment plan or dose calculation.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_spec(path: Path, result: RadiotherapyQAPackageResult, regions: list[dict[str, Any]]) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "scenario": result.scenario,
        "package_type": "radiotherapy_qa_geometry_material_package",
        "outputs": {
            "rt_synthetic_hu": result.rt_hu_path,
            "rt_material_labels": result.rt_material_labels_path,
            "rt_mass_density_g_cm3": result.rt_density_path,
            "rt_relative_electron_density": result.rt_red_path,
            "mask_manifest_csv": result.mask_manifest_csv_path,
            "material_calibration_csv": result.material_calibration_csv_path,
            "pymedphys_placeholder_yaml": result.pymedphys_placeholder_yaml_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
        },
        "synthetic_target": {
            "center_ijk": list(result.target_center_ijk),
            "center_mm": list(result.target_center_mm),
            "gtv_volume_cm3": result.target_gtv_volume_cm3,
            "ptv_volume_cm3": result.target_ptv_volume_cm3,
        },
        "masks": [
            {
                "mask_id": stat.mask_id,
                "label": stat.label,
                "role": stat.role,
                "path": stat.path,
                "voxel_count": stat.voxel_count,
                "volume_cm3": stat.volume_cm3,
                "mean_hu": stat.mean_hu,
                "mean_density_g_cm3": stat.mean_density_g_cm3,
                "mean_relative_electron_density": stat.mean_red,
                "notes": list(stat.notes),
            }
            for stat in result.mask_stats
        ],
        "regions": regions,
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_preview(
    path: Path,
    hu: np.ndarray,
    labels: np.ndarray,
    gtv: np.ndarray,
    ptv: np.ndarray,
    center_ijk: tuple[int, int, int],
) -> None:
    plt, _, _ = _import_dependencies()
    z = int(center_ijk[2])
    image = hu[:, :, z].T
    gtv_slice = gtv[:, :, z].T
    ptv_slice = ptv[:, :, z].T
    liver = (labels[:, :, z] == 6).T
    kidneys = (labels[:, :, z] == 7).T
    lungs = (labels[:, :, z] == 8).T
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.imshow(image, cmap="gray", vmin=-1000, vmax=1000, origin="lower")
    for mask, color, label in [
        (lungs, "#48cae4", "lungs"),
        (liver, "#9d4edd", "liver"),
        (kidneys, "#f72585", "kidneys"),
        (ptv_slice, "#ffd166", "synthetic PTV"),
        (gtv_slice, "#ef476f", "synthetic GTV"),
    ]:
        if np.any(mask):
            ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.4)
            ax.plot([], [], color=color, label=label)
    ax.set_title(f"RT QA masks at target axial slice z={z}")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _format_report(result: RadiotherapyQAPackageResult) -> str:
    lines = [
        "# Radiotherapy QA Package Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        f"Scenario: `{result.scenario}`",
        "",
        "## Summary",
        "",
        f"- Synthetic target center ijk: {result.target_center_ijk}",
        f"- Synthetic target center mm: ({result.target_center_mm[0]:.2f}, {result.target_center_mm[1]:.2f}, {result.target_center_mm[2]:.2f})",
        f"- GTV volume: {result.target_gtv_volume_cm3:.3f} cm3",
        f"- PTV volume: {result.target_ptv_volume_cm3:.3f} cm3",
        f"- Mask count: {len(result.mask_stats)}",
        "",
        "## Outputs",
        "",
        f"- RT synthetic HU: `{Path(result.rt_hu_path).name}`",
        f"- RT material labels: `{Path(result.rt_material_labels_path).name}`",
        f"- RT mass density: `{Path(result.rt_density_path).name}`",
        f"- RT relative electron density: `{Path(result.rt_red_path).name}`",
        f"- Mask manifest: `{Path(result.mask_manifest_csv_path).name}`",
        f"- Material calibration table: `{Path(result.material_calibration_csv_path).name}`",
        f"- PyMedPhys placeholder: `{Path(result.pymedphys_placeholder_yaml_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Package spec: `{Path(result.package_spec_yaml_path).name}`",
        "",
        "## DVH-Ready Masks",
        "",
        "| mask | role | volume cm3 | mean HU | mean RED |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for stat in result.mask_stats:
        mean_hu = "n/a" if stat.mean_hu is None else f"{stat.mean_hu:.2f}"
        mean_red = "n/a" if stat.mean_red is None else f"{stat.mean_red:.3f}"
        lines.append(f"| `{stat.mask_id}` | {stat.role} | {stat.volume_cm3:.3f} | {mean_hu} | {mean_red} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This package is RT-planning and QA scaffolding for the digital phantom, not a clinical treatment plan.",
            "- HU, mass density, and RED maps are synthetic material-target volumes from the phantom material library.",
            "- The target is synthetic and anchored to a central bone voxel to approximate a vertebral-body QA target.",
            "- Dose grids are not present yet; the PyMedPhys YAML is a wiring placeholder for future gamma/DVH evaluation.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_radiotherapy_qa_package(
    combined_spec_path: str | Path,
    output_dir: str | Path = "outputs/radiotherapy/qa_package",
    case_id: str = "ct_org_case0_imagetbad_case125",
    scenario: str = "blood",
    target_radius_mm: float = 12.0,
    ptv_margin_mm: float = 5.0,
    report_path: str | Path | None = "outputs/reports/radiotherapy_qa_package_stage001.md",
) -> RadiotherapyQAPackageResult:
    _, nib, _ = _import_dependencies()
    spec = _load_yaml(combined_spec_path)
    hu_path, labels_path, density_path, red_path = _scenario_outputs(spec, scenario)
    labels_image = nib.load(str(labels_path))
    labels = np.rint(np.asanyarray(labels_image.dataobj)).astype(np.int16)
    regions = list(spec.get("regions", []))
    maps_are_label_derived = hu_path is None or density_path is None or red_path is None
    if maps_are_label_derived:
        hu_image = labels_image
        density_image = labels_image
        red_image = labels_image
        hu, density, red = _maps_from_material_labels(labels, regions)
    else:
        hu_image = nib.load(str(hu_path))
        density_image = nib.load(str(density_path))
        red_image = nib.load(str(red_path))
        hu = np.asanyarray(hu_image.dataobj).astype(np.float32)
        density = np.asanyarray(density_image.dataobj).astype(np.float32)
        red = np.asanyarray(red_image.dataobj).astype(np.float32)

    if labels.shape != hu.shape or density.shape != hu.shape or red.shape != hu.shape:
        raise ValueError("RT QA input maps must have identical shapes")

    output = Path(output_dir)
    masks_dir = output / "masks"
    output.mkdir(parents=True, exist_ok=True)
    spacing = _spacing(hu_image)
    voxel_volume = _voxel_volume_cm3(spacing)

    gtv, ptv, target_center_ijk, target_center_mm, target_notes = _target_masks(
        labels,
        spacing,
        target_radius_mm=target_radius_mm,
        ptv_margin_mm=ptv_margin_mm,
    )
    mask_definitions = [
        ("body", "Body contour", "body", labels > 0, ("material_labels_gt_0",)),
        ("oar_lungs", "Lungs", "oar", labels == 8, ("material_label_8",)),
        ("oar_liver", "Liver", "oar", labels == 6, ("material_label_6",)),
        ("oar_kidneys", "Kidneys", "oar", labels == 7, ("material_label_7",)),
        ("oar_bone", "Bone", "oar_reference", np.isin(labels, (10, 11)), ("material_labels_10_11",)),
        ("vascular_fluid", "Vascular fluid", "flow_reference", np.isin(labels, (14, 15)), ("material_labels_14_15",)),
        ("vessel_wall", "Vessel wall", "flow_reference", labels == 13, ("material_label_13",)),
        ("target_gtv_synthetic_vertebral", "Synthetic vertebral GTV", "target_gtv", gtv, target_notes),
        ("target_ptv_synthetic_vertebral", "Synthetic vertebral PTV", "target_ptv", ptv, target_notes),
    ]

    mask_stats: list[RTMaskStats] = []
    for mask_id, label, role, mask, notes in mask_definitions:
        mask_path = masks_dir / f"{case_id}_{mask_id}_mask_v001.nii.gz"
        _write_nifti(mask_path, mask.astype(np.uint8), labels_image)
        mask_stats.append(
            _mask_stats(
                mask_id=mask_id,
                label=label,
                role=role,
                path=str(mask_path),
                mask=mask,
                hu=hu,
                density=density,
                red=red,
                voxel_volume_cm3=voxel_volume,
                notes=notes,
            )
        )

    rt_hu = output / f"{case_id}_rt_synthetic_hu_{scenario}_v001.nii.gz"
    rt_labels = output / f"{case_id}_rt_material_labels_{scenario}_v001.nii.gz"
    rt_density = output / f"{case_id}_rt_mass_density_g_cm3_{scenario}_v001.nii.gz"
    rt_red = output / f"{case_id}_rt_relative_electron_density_{scenario}_v001.nii.gz"
    _write_nifti(rt_hu, hu.astype(np.float32), hu_image)
    _write_nifti(rt_labels, labels.astype(np.int16), labels_image)
    _write_nifti(rt_density, density.astype(np.float32), density_image)
    _write_nifti(rt_red, red.astype(np.float32), red_image)

    mask_manifest = output / f"{case_id}_rt_mask_manifest_v001.csv"
    material_calibration = output / f"{case_id}_rt_material_calibration_v001.csv"
    pymedphys_yaml = output / f"{case_id}_pymedphys_eval_placeholder_v001.yaml"
    spec_yaml = output / f"{case_id}_radiotherapy_qa_package_spec_v001.yaml"
    preview_png = output / f"{case_id}_radiotherapy_qa_preview_v001.png"
    report = Path(report_path) if report_path else output / f"{case_id}_radiotherapy_qa_package_report_v001.md"

    _write_mask_manifest(mask_manifest, tuple(mask_stats))
    _write_material_calibration(material_calibration, regions)
    _write_preview(preview_png, hu, labels, gtv, ptv, target_center_ijk)

    gtv_stat = next(stat for stat in mask_stats if stat.mask_id == "target_gtv_synthetic_vertebral")
    ptv_stat = next(stat for stat in mask_stats if stat.mask_id == "target_ptv_synthetic_vertebral")
    result = RadiotherapyQAPackageResult(
        case_id=case_id,
        scenario=scenario,
        output_dir=str(output),
        rt_hu_path=str(rt_hu),
        rt_material_labels_path=str(rt_labels),
        rt_density_path=str(rt_density),
        rt_red_path=str(rt_red),
        mask_manifest_csv_path=str(mask_manifest),
        material_calibration_csv_path=str(material_calibration),
        pymedphys_placeholder_yaml_path=str(pymedphys_yaml),
        package_spec_yaml_path=str(spec_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        mask_stats=tuple(mask_stats),
        target_center_mm=target_center_mm,
        target_center_ijk=target_center_ijk,
        target_gtv_volume_cm3=gtv_stat.volume_cm3,
        target_ptv_volume_cm3=ptv_stat.volume_cm3,
        notes=(
            "rt_package_uses_combined_digital_phantom_material_maps",
            "rt_maps_synthesized_from_material_labels_and_region_table" if maps_are_label_derived else "rt_maps_loaded_from_combined_spec_outputs",
            f"scenario={scenario}",
            "synthetic_target_is_not_clinician_contoured",
            "dose_grid_placeholders_require_future_TPS_or_MC_export",
        ),
    )
    _write_pymedphys_placeholder(pymedphys_yaml, result)
    _write_spec(spec_yaml, result, regions)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_radiotherapy_qa_package_result(result: RadiotherapyQAPackageResult) -> str:
    lines = [
        "Radiotherapy QA package created",
        f"Case ID: {result.case_id}",
        f"Scenario: {result.scenario}",
        f"Masks: {len(result.mask_stats)}",
        f"GTV volume: {result.target_gtv_volume_cm3:.3f} cm3",
        f"PTV volume: {result.target_ptv_volume_cm3:.3f} cm3",
        f"RT HU: {result.rt_hu_path}",
        f"RT material labels: {result.rt_material_labels_path}",
        f"RT density: {result.rt_density_path}",
        f"RT RED: {result.rt_red_path}",
        f"Mask manifest: {result.mask_manifest_csv_path}",
        f"Material calibration: {result.material_calibration_csv_path}",
        f"PyMedPhys placeholder: {result.pymedphys_placeholder_yaml_path}",
        f"Preview PNG: {result.preview_png_path}",
        f"Spec YAML: {result.package_spec_yaml_path}",
        f"Report: {result.report_path}",
    ]
    return "\n".join(lines)
