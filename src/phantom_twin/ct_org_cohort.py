from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import time
import urllib.request

import numpy as np


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from matplotlib.colors import ListedColormap  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
        from scipy import ndimage  # type: ignore
    except ImportError as exc:
        raise RuntimeError("CT-ORG cohort staging requires matplotlib, nibabel, and scipy.") from exc
    return plt, ListedColormap, Patch, nib, ndimage


DEFAULT_LABEL_BASE_URL = "https://huggingface.co/datasets/Angelou0516/ct-org/resolve/main/labels"


@dataclass(frozen=True)
class CtOrgStagedCase:
    case_index: int
    case_id: str
    source_url: str
    raw_label_path: str
    material_label_path: str
    preview_png_path: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    body_voxels: int
    label_values: tuple[int, ...]


@dataclass(frozen=True)
class CtOrgLabelCohortResult:
    output_dir: str
    raw_label_dir: str
    manifest_csv_path: str
    report_path: str
    case_results: tuple[CtOrgStagedCase, ...]
    notes: tuple[str, ...]


def _download_file(url: str, output_path: Path, timeout_s: int = 180, attempts: int = 4) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "phantom-digital-twin/0.1"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response, temp_path.open("wb") as outfile:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    outfile.write(chunk)
            temp_path.replace(output_path)
            return
        except Exception as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            if attempt < attempts:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_error


def _write_nifti(path: Path, data: np.ndarray, reference_image, nib) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, reference_image.affine, reference_image.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _largest_component(mask: np.ndarray, ndimage) -> np.ndarray:
    connected, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if count <= 1:
        return mask
    sizes = np.bincount(connected.ravel(), minlength=count + 1)
    sizes[0] = 0
    return connected == int(sizes.argmax())


def _infer_body_proxy(
    ct_org_labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    padding_mm: float,
    ndimage,
) -> np.ndarray:
    support = ct_org_labels > 0
    body = np.zeros(support.shape, dtype=bool)
    pad_i = max(2, int(round(padding_mm / max(spacing_mm[0], 1e-6))))
    pad_j = max(2, int(round(padding_mm / max(spacing_mm[1], 1e-6))))
    ii = np.arange(support.shape[0])[:, None]
    jj = np.arange(support.shape[1])[None, :]

    for z_index in range(support.shape[2]):
        coords = np.argwhere(support[:, :, z_index])
        if coords.size == 0:
            continue
        i_min = max(0, int(coords[:, 0].min()) - pad_i)
        i_max = min(support.shape[0] - 1, int(coords[:, 0].max()) + pad_i)
        j_min = max(0, int(coords[:, 1].min()) - pad_j)
        j_max = min(support.shape[1] - 1, int(coords[:, 1].max()) + pad_j)
        center_i = 0.5 * (i_min + i_max)
        center_j = 0.5 * (j_min + j_max)
        radius_i = max(1.0, 0.5 * (i_max - i_min + 1))
        radius_j = max(1.0, 0.5 * (j_max - j_min + 1))
        ellipse = ((ii - center_i) / radius_i) ** 2 + ((jj - center_j) / radius_j) ** 2 <= 1.0
        body[:, :, z_index] = ellipse

    body = ndimage.binary_closing(body, structure=np.ones((5, 5, 3), dtype=bool), iterations=1)
    filled = np.zeros_like(body, dtype=bool)
    for z_index in range(body.shape[2]):
        if np.any(body[:, :, z_index]):
            filled[:, :, z_index] = ndimage.binary_fill_holes(body[:, :, z_index])
    return _largest_component(filled | support, ndimage)


def _skin_layer_2d(body: np.ndarray, spacing_mm: tuple[float, float, float], adipose_layer_mm: float, ndimage) -> np.ndarray:
    skin = np.zeros(body.shape, dtype=bool)
    for z_index in range(body.shape[2]):
        slice_mask = body[:, :, z_index]
        if not np.any(slice_mask):
            continue
        distance_to_skin = ndimage.distance_transform_edt(slice_mask, sampling=spacing_mm[:2])
        skin[:, :, z_index] = slice_mask & (distance_to_skin <= adipose_layer_mm)
    return skin


def _materialize_ct_org_labels(
    ct_org_labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    body_padding_mm: float,
    adipose_layer_mm: float,
    ndimage,
) -> np.ndarray:
    material = np.zeros(ct_org_labels.shape, dtype=np.int16)
    body = _infer_body_proxy(ct_org_labels, spacing_mm, body_padding_mm, ndimage)
    skin_layer = _skin_layer_2d(body, spacing_mm, adipose_layer_mm, ndimage)
    material[skin_layer] = 3
    material[body & ~skin_layer] = 4

    material[ct_org_labels == 1] = 6
    material[ct_org_labels == 2] = 9
    material[ct_org_labels == 3] = 8
    material[ct_org_labels == 4] = 7
    material[ct_org_labels == 5] = 10
    material[ct_org_labels == 6] = 12
    material[~body] = 0
    return material


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(value // 2 for value in mask.shape)
    return tuple(int(round(float(np.median(coords[:, axis])))) for axis in range(3))


def _write_preview(path: Path, material_labels: np.ndarray, case_id: str) -> None:
    plt, ListedColormap, Patch, *_ = _import_dependencies()
    colors = [
        "#0d1b2a",
        "#6c757d",
        "#8ecae6",
        "#ffd166",
        "#d95d39",
        "#4cc9f0",
        "#9d4edd",
        "#f72585",
        "#48cae4",
        "#f4d35e",
        "#e9ecef",
        "#ffffff",
        "#80ed99",
    ]
    cmap = ListedColormap(colors)
    body = material_labels > 0
    x_index, y_index, z_index = _slice_indices(body)
    views = [
        ("Axial", material_labels[:, :, z_index]),
        ("Coronal", material_labels[:, y_index, :]),
        ("Sagittal", material_labels[x_index, :, :]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150)
    fig.patch.set_facecolor("#f6f1e8")
    for ax, (title, view) in zip(axes, views, strict=True):
        ax.imshow(np.rot90(view), cmap=cmap, vmin=0, vmax=12, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=9, color="#13202a")
        ax.axis("off")
    handles = [
        Patch(facecolor="#ffd166", label="inferred adipose envelope"),
        Patch(facecolor="#d95d39", label="inferred soft tissue"),
        Patch(facecolor="#48cae4", label="lungs"),
        Patch(facecolor="#9d4edd", label="liver"),
        Patch(facecolor="#f72585", label="kidneys"),
        Patch(facecolor="#e9ecef", label="bone"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle(f"CT-ORG Label-Only Materialization: {case_id}", fontsize=12, color="#13202a")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_manifest(path: Path, cases: tuple[CtOrgStagedCase, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "case_id",
                "case_index",
                "source_url",
                "raw_label_path",
                "material_label_path",
                "preview_png_path",
                "shape_x",
                "shape_y",
                "shape_z",
                "spacing_x_mm",
                "spacing_y_mm",
                "spacing_z_mm",
                "body_voxels",
                "label_values",
            ]
        )
        for item in cases:
            writer.writerow(
                [
                    item.case_id,
                    item.case_index,
                    item.source_url,
                    item.raw_label_path,
                    item.material_label_path,
                    item.preview_png_path,
                    *item.shape,
                    *[f"{value:.6f}" for value in item.spacing_mm],
                    item.body_voxels,
                    " ".join(str(value) for value in item.label_values),
                ]
            )


def _format_report(result: CtOrgLabelCohortResult) -> str:
    lines = [
        "# CT-ORG Label-Only Population Staging Stage 001",
        "",
        "## Summary",
        "",
        f"- Cases staged: {len(result.case_results)}",
        f"- Raw label directory: `{result.raw_label_dir}`",
        f"- Material-label output directory: `{result.output_dir}`",
        f"- Manifest CSV: `{Path(result.manifest_csv_path).name}`",
        "",
        "## Cases",
        "",
        "| case | shape | spacing mm | body voxels | labels |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for item in result.case_results:
        lines.append(
            f"| {item.case_id} | {item.shape[0]} x {item.shape[1]} x {item.shape[2]} | "
            f"{', '.join(f'{value:.3g}' for value in item.spacing_mm)} | "
            f"{item.body_voxels} | {', '.join(str(value) for value in item.label_values)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a disk-light staging path that uses CT-ORG segmentation labels without downloading every full CT volume.",
            "- Body, adipose, and generic soft-tissue labels are inferred from organ/bone support, so they are suitable for population-shape experiments but not subject-specific HU validation.",
            "- For final density-equivalent population phantoms, rerun selected cases through `build-digital-torso` with the matching CT volumes.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def stage_ct_org_label_cohort(
    case_indices: tuple[int, ...] = tuple(range(10)),
    raw_label_dir: str | Path = "data/raw/ct_org/labels",
    output_dir: str | Path = "data/processed/ct_org_label_population",
    case_id_prefix: str = "ct_org_case",
    label_base_url: str = DEFAULT_LABEL_BASE_URL,
    body_padding_mm: float = 35.0,
    adipose_layer_mm: float = 18.0,
    force_download: bool = False,
    report_path: str | Path | None = "outputs/reports/ct_org_label_population_stage001.md",
) -> CtOrgLabelCohortResult:
    _, _, _, nib, ndimage = _import_dependencies()
    raw_dir = Path(raw_label_dir)
    output = Path(output_dir)
    preview_dir = output / "previews"
    case_results: list[CtOrgStagedCase] = []

    for case_index in case_indices:
        case_id = f"{case_id_prefix}{case_index}"
        source_url = f"{label_base_url}/labels-{case_index}.nii.gz"
        raw_path = raw_dir / f"labels-{case_index}.nii.gz"
        if force_download or not raw_path.exists():
            _download_file(source_url, raw_path)
        image = nib.load(str(raw_path))
        ct_org_labels = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
        spacing_mm = tuple(float(value) for value in image.header.get_zooms()[:3])
        material_path = output / f"{case_id}_label_only_material_labels_v001.nii.gz"
        preview_path = preview_dir / f"{case_id}_label_only_material_preview_v001.png"
        if material_path.exists() and preview_path.exists() and not force_download:
            material = np.rint(np.asanyarray(nib.load(str(material_path)).dataobj)).astype(np.int16)
        else:
            material = _materialize_ct_org_labels(
                ct_org_labels,
                spacing_mm=spacing_mm,
                body_padding_mm=body_padding_mm,
                adipose_layer_mm=adipose_layer_mm,
                ndimage=ndimage,
            )
            _write_nifti(material_path, material, image, nib)
            _write_preview(preview_path, material, case_id)
        label_values = tuple(int(value) for value in np.unique(ct_org_labels))
        case_results.append(
            CtOrgStagedCase(
                case_index=case_index,
                case_id=case_id,
                source_url=source_url,
                raw_label_path=str(raw_path),
                material_label_path=str(material_path),
                preview_png_path=str(preview_path),
                shape=tuple(int(value) for value in material.shape),
                spacing_mm=spacing_mm,
                body_voxels=int((material > 0).sum()),
                label_values=label_values,
            )
        )

    manifest_out = output / "ct_org_label_population_manifest_v001.csv"
    notes = (
        "ct_org_label_only_population_staging",
        "full_ct_volumes_not_downloaded_for_disk_light_pca_bootstrap",
        "body_soft_tissue_adipose_are_inferred_from_segmentation_support",
    )
    result = CtOrgLabelCohortResult(
        output_dir=str(output),
        raw_label_dir=str(raw_dir),
        manifest_csv_path=str(manifest_out),
        report_path=str(report_path) if report_path is not None else str(output / "ct_org_label_population_report_v001.md"),
        case_results=tuple(case_results),
        notes=notes,
    )
    _write_manifest(manifest_out, result.case_results)
    report = _format_report(result)
    Path(result.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result.report_path).write_text(report)
    return result


def format_ct_org_label_cohort_result(result: CtOrgLabelCohortResult) -> str:
    lines = [
        "# CT-ORG Label-Only Population Staging",
        "",
        f"Cases staged: {len(result.case_results)}",
        f"Manifest: `{result.manifest_csv_path}`",
        "",
        "## Material-label outputs",
    ]
    for item in result.case_results:
        lines.append(f"- {item.case_id}: `{item.material_label_path}`")
    return "\n".join(lines)
