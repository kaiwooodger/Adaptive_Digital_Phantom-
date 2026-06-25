from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import gzip
import math
import re
import zipfile
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class AvtKitsAortaCaseResult:
    case_id: str
    source_case_id: str
    ct_nifti_path: str
    aorta_mask_nifti_path: str
    preview_png_path: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    aorta_voxels: int
    aorta_volume_ml: float
    aorta_z_span_mm: float
    mean_aorta_radius_mm: float
    segment_name: str
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class AvtKitsAortaStagingResult:
    dataset_id: str
    zip_path: str
    output_dir: str
    labelmap_yaml_path: str
    manifest_csv_path: str
    intake_csv_path: str
    manifest_yaml_path: str
    atlas_png_path: str
    report_path: str
    discovered_case_count: int
    staged_case_count: int
    failed_case_count: int
    total_aorta_volume_ml: float
    readiness_status: str
    case_results: tuple[AvtKitsAortaCaseResult, ...]
    notes: tuple[str, ...]


NRRD_TYPE_TO_DTYPE = {
    "signed char": np.int8,
    "int8": np.int8,
    "uchar": np.uint8,
    "unsigned char": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "short int": np.int16,
    "signed short": np.int16,
    "signed short int": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "unsigned short": np.uint16,
    "unsigned short int": np.uint16,
    "uint16": np.uint16,
    "int": np.int32,
    "signed int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "unsigned int": np.uint32,
    "uint32": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("AVT/KiTS aorta staging requires nibabel.") from exc
    return nib


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("AVT/KiTS aorta staging previews require matplotlib.") from exc
    return plt


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "case"


def _natural_case_key(case_id: str) -> tuple[str, int, str]:
    match = re.search(r"(\d+)$", case_id)
    if match:
        return case_id[: match.start()], int(match.group(1)), case_id
    return case_id, -1, case_id


def _read_nrrd_header(stream) -> dict[str, str]:
    header_lines: list[str] = []
    magic = stream.readline()
    if not magic.startswith(b"NRRD"):
        raise ValueError("NRRD member does not start with NRRD magic header")
    header_lines.append(magic.decode("latin1", errors="replace").rstrip("\r\n"))
    while True:
        line = stream.readline()
        if line in (b"", b"\n", b"\r\n"):
            break
        text = line.decode("latin1", errors="replace").rstrip("\r\n")
        header_lines.append(text)

    fields: dict[str, str] = {}
    for line in header_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("NRRD"):
            continue
        if ":=" in stripped:
            key, value = stripped.split(":=", 1)
        elif ":" in stripped:
            key, value = stripped.split(":", 1)
        else:
            continue
        fields[key.strip()] = value.strip()
    return fields


def _nrrd_dtype(fields: dict[str, str]) -> np.dtype:
    type_name = fields.get("type", "").strip().lower()
    if type_name not in NRRD_TYPE_TO_DTYPE:
        raise ValueError(f"Unsupported NRRD type: {fields.get('type', '')}")
    dtype = np.dtype(NRRD_TYPE_TO_DTYPE[type_name])
    endian = fields.get("endian", "").strip().lower()
    if dtype.itemsize > 1 and endian in {"little", "big"}:
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    return dtype


def _parse_sizes(fields: dict[str, str]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in fields.get("sizes", "").split())
    if len(values) != 3:
        raise ValueError(f"Expected 3D NRRD sizes, got: {fields.get('sizes', '')}")
    return values  # type: ignore[return-value]


def _parse_vector(raw: str) -> tuple[float, float, float]:
    values = [float(value.strip()) for value in raw.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3-vector in NRRD field, got: {raw}")
    return float(values[0]), float(values[1]), float(values[2])


def _parse_space_directions(fields: dict[str, str]) -> tuple[tuple[float, float, float], ...]:
    raw = fields.get("space directions", "")
    tokens = re.findall(r"\([^)]*\)|none", raw, flags=re.IGNORECASE)
    if len(tokens) != 3:
        spacings = tuple(float(value) for value in fields.get("spacings", "1 1 1").split())
        if len(spacings) == 3:
            return ((spacings[0], 0.0, 0.0), (0.0, spacings[1], 0.0), (0.0, 0.0, spacings[2]))
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    vectors: list[tuple[float, float, float]] = []
    for token in tokens:
        if token.lower() == "none":
            vectors.append((0.0, 0.0, 0.0))
        else:
            vectors.append(_parse_vector(token.strip("()")))
    return tuple(vectors)


def _parse_space_origin(fields: dict[str, str]) -> tuple[float, float, float]:
    raw = fields.get("space origin", "")
    if raw.startswith("(") and raw.endswith(")"):
        return _parse_vector(raw.strip("()"))
    return (0.0, 0.0, 0.0)


def _affine_from_nrrd(fields: dict[str, str]) -> np.ndarray:
    affine = np.eye(4, dtype=float)
    for axis, vector in enumerate(_parse_space_directions(fields)):
        affine[:3, axis] = np.asarray(vector, dtype=float)
    affine[:3, 3] = np.asarray(_parse_space_origin(fields), dtype=float)
    return affine


def _load_nrrd_from_zip(zip_file: zipfile.ZipFile, member: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    with zip_file.open(member) as stream:
        fields = _read_nrrd_header(stream)
        dtype = _nrrd_dtype(fields)
        sizes = _parse_sizes(fields)
        encoding = fields.get("encoding", "raw").strip().lower()
        if encoding in {"gzip", "gz"}:
            raw = gzip.GzipFile(fileobj=stream).read()
        elif encoding in {"raw", ""}:
            raw = stream.read()
        else:
            raise ValueError(f"Unsupported NRRD encoding: {fields.get('encoding', '')}")
    expected = int(np.prod(sizes)) * dtype.itemsize
    if len(raw) < expected:
        raise ValueError(f"NRRD payload shorter than expected for {member}: {len(raw)} < {expected}")
    array = np.frombuffer(raw[:expected], dtype=dtype).reshape(sizes, order="F")
    native_array = np.asarray(array, dtype=dtype.newbyteorder("="))
    return native_array, _affine_from_nrrd(fields), fields


def _discover_cases(zip_file: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    cases: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"^KiTS/([^/]+)/([^/]+)$")
    for member in zip_file.namelist():
        match = pattern.match(member)
        if not match:
            continue
        source_case, file_name = match.groups()
        if file_name == f"{source_case}.nrrd":
            cases.setdefault(source_case, {})["ct"] = member
        elif file_name == f"{source_case}.seg.nrrd":
            cases.setdefault(source_case, {})["seg"] = member
    return cases


def _case_selection(all_case_ids: list[str], requested: tuple[str, ...] | None, max_cases: int | None) -> list[str]:
    if requested:
        wanted = {case_id.strip() for case_id in requested if case_id.strip()}
        selected = [case_id for case_id in all_case_ids if case_id in wanted]
    else:
        selected = list(all_case_ids)
    if max_cases is not None:
        selected = selected[: max(0, int(max_cases))]
    return selected


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))  # type: ignore[return-value]


def _aorta_metrics(label: np.ndarray, spacing_mm: tuple[float, float, float]) -> tuple[int, float, float, float]:
    mask = label > 0
    voxels = int(np.count_nonzero(mask))
    if voxels == 0:
        return 0, 0.0, 0.0, 0.0
    voxel_volume_mm3 = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2])
    volume_ml = float(voxels * voxel_volume_mm3 / 1000.0)
    slice_counts = np.count_nonzero(mask, axis=(0, 1))
    occupied = np.flatnonzero(slice_counts)
    z_span_mm = float((occupied[-1] - occupied[0] + 1) * spacing_mm[2])
    areas_mm2 = slice_counts[occupied].astype(float) * spacing_mm[0] * spacing_mm[1]
    radii = np.sqrt(np.maximum(areas_mm2, 0.0) / math.pi)
    return voxels, volume_ml, z_span_mm, float(np.mean(radii))


def _segment_name(fields: dict[str, str]) -> str:
    for key, value in fields.items():
        if key.endswith("_Name"):
            return value
    return ""


def _write_preview(ct: np.ndarray, label: np.ndarray, path: Path, title: str) -> None:
    plt = _import_plotting()
    mask = label > 0
    occupied = np.flatnonzero(np.count_nonzero(mask, axis=(0, 1)))
    z_index = int(occupied[len(occupied) // 2]) if occupied.size else ct.shape[2] // 2
    ct_slice = np.asarray(ct[:, :, z_index], dtype=float)
    mask_slice = mask[:, :, z_index]
    finite = ct_slice[np.isfinite(ct_slice)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [1.0, 99.0])
        if math.isclose(float(vmin), float(vmax)):
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
    else:
        vmin, vmax = 0.0, 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(ct_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].imshow(np.ma.masked_where(~mask_slice.T, mask_slice.T), cmap="autumn", alpha=0.55, origin="lower")
    axes[0].set_title(f"{title}\naxial aorta slice {z_index}")
    axes[0].axis("off")
    mip = np.max(mask.astype(np.uint8), axis=2)
    axes[1].imshow(mip.T, cmap="Reds", origin="lower")
    axes[1].set_title("Aorta mask axial MIP")
    axes[1].axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_atlas(case_results: tuple[AvtKitsAortaCaseResult, ...], atlas_path: Path) -> None:
    plt = _import_plotting()
    if not case_results:
        atlas_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No staged AVT/KiTS aorta cases", ha="center", va="center")
        ax.axis("off")
        fig.savefig(atlas_path, dpi=150)
        plt.close(fig)
        return
    cols = min(5, max(1, len(case_results)))
    rows = int(math.ceil(len(case_results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes_array.ravel():
        ax.axis("off")
    for ax, result in zip(axes_array.ravel(), case_results):
        image = plt.imread(result.preview_png_path)
        ax.imshow(image)
        ax.set_title(
            f"{result.source_case_id}: {result.aorta_volume_ml:.1f} mL\n"
            f"{result.shape[0]}x{result.shape[1]}x{result.shape[2]}, dz={result.spacing_mm[2]:.2f} mm",
            fontsize=8,
        )
        ax.axis("off")
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(atlas_path, dpi=150)
    plt.close(fig)


def _write_labelmap(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": "avt_kits_aorta",
        "description": "Aorta-only masks staged from the AVT KiTS.zip archive.",
        "labels": {
            0: {"name": "background", "role": "background"},
            1: {"name": "aorta", "role": "arterial_lumen", "phase1_role": "major_vessel_registration_target"},
        },
        "graph_edge_mapping": {
            "aorta_trunk": {
                "labels": [1],
                "vessel_type": "arterial",
                "flow_role": "aorta_trunk",
            }
        },
        "notes": [
            "This labelmap supports aorta-trunk registration and centerline extraction.",
            "It does not provide renal, iliac, hepatic, splenic, portal, or venous branch labels.",
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_manifest_yaml(path: Path, result: AvtKitsAortaStagingResult) -> None:
    data = {
        "dataset_id": result.dataset_id,
        "package_type": "avt_kits_aorta_registration_cohort",
        "zip_path": result.zip_path,
        "output_dir": result.output_dir,
        "readiness_status": result.readiness_status,
        "discovered_case_count": result.discovered_case_count,
        "staged_case_count": result.staged_case_count,
        "failed_case_count": result.failed_case_count,
        "labelmap_yaml_path": result.labelmap_yaml_path,
        "manifest_csv_path": result.manifest_csv_path,
        "intake_csv_path": result.intake_csv_path,
        "atlas_png_path": result.atlas_png_path,
        "report_path": result.report_path,
        "cases": [
            {
                "case_id": item.case_id,
                "source_case_id": item.source_case_id,
                "ct_nifti_path": item.ct_nifti_path,
                "aorta_mask_nifti_path": item.aorta_mask_nifti_path,
                "preview_png_path": item.preview_png_path,
                "shape": list(item.shape),
                "spacing_mm": list(item.spacing_mm),
                "aorta_volume_ml": item.aorta_volume_ml,
                "mean_aorta_radius_mm": item.mean_aorta_radius_mm,
                "status": item.status,
                "notes": list(item.notes),
            }
            for item in result.case_results
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_report(path: Path, result: AvtKitsAortaStagingResult) -> None:
    lines = [
        "# AVT/KiTS Aorta Staging Report",
        "",
        f"Dataset ID: `{result.dataset_id}`",
        f"Readiness: `{result.readiness_status}`",
        f"Source zip: `{result.zip_path}`",
        f"Cases discovered/staged/failed: {result.discovered_case_count} / {result.staged_case_count} / {result.failed_case_count}",
        f"Total staged aorta volume: {result.total_aorta_volume_ml:.3f} mL",
        "",
        "## Outputs",
        "",
        f"- Labelmap: `{result.labelmap_yaml_path}`",
        f"- Manifest CSV: `{result.manifest_csv_path}`",
        f"- Validation-intake CSV: `{result.intake_csv_path}`",
        f"- Atlas PNG: `{result.atlas_png_path}`",
        f"- Manifest YAML: `{result.manifest_yaml_path}`",
        "",
        "## Case Summary",
        "",
        "| Case | Shape | Spacing mm | Aorta volume mL | Mean radius mm | Segment name | Status |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in result.case_results:
        shape = "x".join(str(value) for value in item.shape)
        spacing = ", ".join(f"{value:.3f}" for value in item.spacing_mm)
        lines.append(
            f"| {item.source_case_id} | {shape} | {spacing} | {item.aorta_volume_ml:.3f} | "
            f"{item.mean_aorta_radius_mm:.3f} | {item.segment_name or 'n/a'} | {item.status} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This archive is useful for practicing aorta CT/CTA registration, aorta centerline extraction, and major-vessel deformation QA.",
            "- It is not enough to replace the full synthetic vascular scaffold because it lacks renal, iliac, hepatic, splenic, portal, and venous branch labels.",
            "- The staged masks can improve the current coarse-vessel product and provide a multi-case aorta registration benchmark.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result.notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def stage_avt_kits_aorta_zip(
    *,
    zip_path: str | Path,
    output_dir: str | Path = "data/processed/avt_kits_aorta",
    dataset_id: str = "avt_kits_aorta_stage001",
    case_ids: tuple[str, ...] | None = None,
    max_cases: int | None = None,
    case_id_prefix: str = "avt_kits",
    report_path: str | Path = "outputs/reports/avt_kits_aorta_stage001.md",
) -> AvtKitsAortaStagingResult:
    nib = _import_nibabel()
    source_zip = Path(zip_path)
    if not source_zip.exists():
        raise FileNotFoundError(source_zip)

    output = Path(output_dir)
    image_dir = output / "images"
    label_dir = output / "labels"
    preview_dir = output / "previews"
    labelmap_path = output / "avt_kits_aorta_labelmap_v001.yaml"
    manifest_csv = output / f"{dataset_id}_manifest_v001.csv"
    intake_csv = output / f"{dataset_id}_validation_intake_cases_v001.csv"
    manifest_yaml = output / f"{dataset_id}_manifest_v001.yaml"
    atlas_png = output / f"{dataset_id}_preview_atlas_v001.png"
    report = Path(report_path)

    case_results: list[AvtKitsAortaCaseResult] = []
    notes = [
        "source_archive_contains_ct_or_cta_nrrd_files_and_binary_aorta_seg_nrrds",
        "staged_outputs_are_nifti_to_match_the_existing_phantom_pipeline",
        "branch_rich_vascular_labels_are_not_present_in_this_archive",
    ]
    failures = 0

    with zipfile.ZipFile(source_zip) as zip_file:
        discovered = _discover_cases(zip_file)
        all_case_ids = sorted(discovered, key=_natural_case_key)
        selected_case_ids = _case_selection(all_case_ids, case_ids, max_cases)
        for source_case_id in selected_case_ids:
            members = discovered[source_case_id]
            case_notes: list[str] = []
            if "ct" not in members or "seg" not in members:
                failures += 1
                continue
            case_id = f"{_slug(case_id_prefix)}_{_slug(source_case_id)}"
            try:
                ct, ct_affine, _ = _load_nrrd_from_zip(zip_file, members["ct"])
                seg, seg_affine, seg_fields = _load_nrrd_from_zip(zip_file, members["seg"])
                if tuple(ct.shape) != tuple(seg.shape):
                    raise ValueError(f"CT/seg shape mismatch for {source_case_id}: {ct.shape} vs {seg.shape}")
                if not np.allclose(ct_affine, seg_affine, atol=1e-3):
                    case_notes.append("ct_and_seg_affines_differ_slightly")
                label = (seg > 0).astype(np.uint8)
                spacing = _spacing_from_affine(ct_affine)
                voxels, volume_ml, z_span_mm, mean_radius_mm = _aorta_metrics(label, spacing)
                segment_name = _segment_name(seg_fields)
                if segment_name.lower().startswith("segment_"):
                    case_notes.append("generic_segment_name_treated_as_aorta_from_archive_context")
                if voxels <= 0:
                    case_notes.append("aorta_mask_empty")
                ct_path = image_dir / f"{case_id}_ct.nii.gz"
                label_path = label_dir / f"{case_id}_aorta_mask.nii.gz"
                preview_path = preview_dir / f"{case_id}_aorta_preview_v001.png"
                ct_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(ct, ct_affine), str(ct_path))
                nib.save(nib.Nifti1Image(label, ct_affine), str(label_path))
                _write_preview(ct, label, preview_path, source_case_id)
                case_results.append(
                    AvtKitsAortaCaseResult(
                        case_id=case_id,
                        source_case_id=source_case_id,
                        ct_nifti_path=str(ct_path),
                        aorta_mask_nifti_path=str(label_path),
                        preview_png_path=str(preview_path),
                        shape=tuple(int(value) for value in ct.shape),
                        spacing_mm=spacing,
                        aorta_voxels=voxels,
                        aorta_volume_ml=volume_ml,
                        aorta_z_span_mm=z_span_mm,
                        mean_aorta_radius_mm=mean_radius_mm,
                        segment_name=segment_name,
                        status="staged" if voxels > 0 else "review_empty_mask",
                        notes=tuple(case_notes),
                    )
                )
            except Exception as exc:
                failures += 1
                case_results.append(
                    AvtKitsAortaCaseResult(
                        case_id=f"{_slug(case_id_prefix)}_{_slug(source_case_id)}",
                        source_case_id=source_case_id,
                        ct_nifti_path="",
                        aorta_mask_nifti_path="",
                        preview_png_path="",
                        shape=(0, 0, 0),
                        spacing_mm=(0.0, 0.0, 0.0),
                        aorta_voxels=0,
                        aorta_volume_ml=0.0,
                        aorta_z_span_mm=0.0,
                        mean_aorta_radius_mm=0.0,
                        segment_name="",
                        status="failed",
                        notes=(f"{type(exc).__name__}: {exc}",),
                    )
                )

    staged = tuple(item for item in case_results if item.status == "staged")
    _write_labelmap(labelmap_path)
    _write_atlas(staged, atlas_png)
    _write_csv(
        manifest_csv,
        [
            {
                "case_id": item.case_id,
                "source_case_id": item.source_case_id,
                "ct_nifti_path": item.ct_nifti_path,
                "aorta_mask_nifti_path": item.aorta_mask_nifti_path,
                "preview_png_path": item.preview_png_path,
                "shape": "x".join(str(value) for value in item.shape),
                "spacing_mm": ",".join(f"{value:.8g}" for value in item.spacing_mm),
                "aorta_voxels": item.aorta_voxels,
                "aorta_volume_ml": f"{item.aorta_volume_ml:.8g}",
                "aorta_z_span_mm": f"{item.aorta_z_span_mm:.8g}",
                "mean_aorta_radius_mm": f"{item.mean_aorta_radius_mm:.8g}",
                "segment_name": item.segment_name,
                "status": item.status,
                "notes": ";".join(item.notes),
            }
            for item in case_results
        ],
        [
            "case_id",
            "source_case_id",
            "ct_nifti_path",
            "aorta_mask_nifti_path",
            "preview_png_path",
            "shape",
            "spacing_mm",
            "aorta_voxels",
            "aorta_volume_ml",
            "aorta_z_span_mm",
            "mean_aorta_radius_mm",
            "segment_name",
            "status",
            "notes",
        ],
    )
    _write_csv(
        intake_csv,
        [
            {
                "case_id": item.case_id,
                "source_dataset": "avt_kits_aorta",
                "ct_path": item.ct_nifti_path,
                "cta_path": item.ct_nifti_path,
                "ctv_path": "",
                "organ_seg_path": "",
                "vessel_seg_path": item.aorta_mask_nifti_path,
                "vessel_label_config": str(labelmap_path),
                "required_vessel_labels": "1",
                "access_status": "approved",
                "notes": "aorta_only_cta_registration_practice_case",
            }
            for item in staged
        ],
        [
            "case_id",
            "source_dataset",
            "ct_path",
            "cta_path",
            "ctv_path",
            "organ_seg_path",
            "vessel_seg_path",
            "vessel_label_config",
            "required_vessel_labels",
            "access_status",
            "notes",
        ],
    )

    failed_count = sum(1 for item in case_results if item.status == "failed")
    staged_count = len(staged)
    readiness = "aorta_registration_practice_ready" if staged_count > 0 and failed_count == 0 else "aorta_registration_review_required"
    result = AvtKitsAortaStagingResult(
        dataset_id=dataset_id,
        zip_path=str(source_zip),
        output_dir=str(output),
        labelmap_yaml_path=str(labelmap_path),
        manifest_csv_path=str(manifest_csv),
        intake_csv_path=str(intake_csv),
        manifest_yaml_path=str(manifest_yaml),
        atlas_png_path=str(atlas_png),
        report_path=str(report),
        discovered_case_count=len(all_case_ids),
        staged_case_count=staged_count,
        failed_case_count=failures,
        total_aorta_volume_ml=float(sum(item.aorta_volume_ml for item in staged)),
        readiness_status=readiness,
        case_results=tuple(case_results),
        notes=tuple(notes),
    )
    _write_manifest_yaml(manifest_yaml, result)
    _write_report(report, result)
    return result


def format_avt_kits_aorta_result(result: AvtKitsAortaStagingResult) -> str:
    return "\n".join(
        [
            "AVT/KiTS aorta cohort staged",
            f"Dataset ID: {result.dataset_id}",
            f"Readiness: {result.readiness_status}",
            f"Cases discovered/staged/failed: {result.discovered_case_count}/{result.staged_case_count}/{result.failed_case_count}",
            f"Total staged aorta volume: {result.total_aorta_volume_ml:.3f} mL",
            f"Manifest CSV: {result.manifest_csv_path}",
            f"Validation intake CSV: {result.intake_csv_path}",
            f"Atlas PNG: {result.atlas_png_path}",
            "Scope: aorta-only registration practice; not branch-rich CTA/CTV vascular replacement.",
        ]
    )
