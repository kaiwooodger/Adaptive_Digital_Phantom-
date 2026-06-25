from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np

from .mesh_clean import _decimate, _fix_normals


def _import_dependencies():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
        from scipy import ndimage  # type: ignore
        from skimage import measure  # type: ignore
        import trimesh  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Printable cartridge generation requires nibabel, scipy, scikit-image, "
            "matplotlib, trimesh, and PyYAML."
        ) from exc
    return plt, Patch, Poly3DCollection, nib, ndimage, measure, trimesh, yaml


@dataclass(frozen=True)
class CylinderSpec:
    name: str
    start_mm: tuple[float, float, float]
    end_mm: tuple[float, float, float]
    radius_mm: float
    role: str


@dataclass(frozen=True)
class PrintableCartridgeResult:
    case_id: str
    output_dir: str
    cartridge_paths: tuple[str, ...]
    fluid_core_paths: tuple[str, ...]
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    voxel_size_mm: float
    wall_thickness_mm: float
    bore_clearance_mm: float
    solid_voxels: int
    fluid_voxels: int
    grid_shape: tuple[int, int, int]
    cartridge_faces: int
    fluid_core_faces: int
    cylinders: tuple[CylinderSpec, ...]
    notes: tuple[str, ...]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = _unit(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(w, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(w, helper))
    v = _unit(np.cross(w, u))
    return u, v, w


def _load_label_mask(labels_path: Path, label_id: int):
    _, _, _, nib, ndimage, *_ = _import_dependencies()
    image = nib.load(str(labels_path))
    labels = np.asanyarray(image.dataobj)
    rounded = np.rint(labels).astype(np.int16)
    mask = rounded == label_id
    if not mask.any():
        raise ValueError(f"Label {label_id} was not found in {labels_path}")

    structure = np.ones((3, 3, 3), dtype=bool)
    connected, count = ndimage.label(mask, structure=structure)
    if count <= 1:
        return mask, image, 0

    component_sizes = np.bincount(connected.ravel())
    component_sizes[0] = 0
    largest = int(component_sizes.argmax())
    return connected == largest, image, int(count - 1)


def _read_centerline_csv(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None

    rows: list[list[float]] = []
    with Path(path).open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(
                [
                    float(row["x_mm"]),
                    float(row["y_mm"]),
                    float(row["z_mm"]),
                    float(row["radius_equiv_mm"]),
                ]
            )

    if not rows:
        return None
    return np.array(rows, dtype=float)


def _load_flow_loop_spec(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if "case_id" not in data or "adapters" not in data:
        raise ValueError(f"Flow-loop spec is missing required fields: {path}")
    return data


def _adapter_cylinders(
    adapter: dict[str, Any],
    wall_thickness_mm: float,
    bore_clearance_mm: float,
    pressure_tap_wall_mm: float,
) -> tuple[list[CylinderSpec], list[CylinderSpec]]:
    center = np.array(adapter["center_mm"], dtype=float)
    axis = _unit(np.array(adapter["axis"], dtype=float))
    u, _, _ = _orthonormal_basis(axis)

    tube_id = float(adapter["tube_inner_diameter_mm"])
    sleeve_od = float(adapter["sleeve_outer_diameter_mm"])
    sleeve_length = float(adapter["sleeve_length_mm"])
    flange_od = float(adapter["flange_outer_diameter_mm"])
    flange_thickness = float(adapter["flange_thickness_mm"])
    barb_count = int(adapter.get("barb_count", 0))
    barb_od = float(adapter.get("barb_outer_diameter_mm", sleeve_od))
    barb_width = float(adapter.get("barb_width_mm", 0.0))
    barb_spacing = float(adapter.get("barb_spacing_mm", 0.0))
    tap_diameter = float(adapter["pressure_tap_diameter_mm"])
    adapter_id = str(adapter["id"])

    solids: list[CylinderSpec] = []
    fluids: list[CylinderSpec] = []

    def add_solid(name: str, start: np.ndarray, end: np.ndarray, radius: float) -> None:
        solids.append(
            CylinderSpec(
                name=f"{adapter_id}_{name}",
                start_mm=tuple(float(value) for value in start),
                end_mm=tuple(float(value) for value in end),
                radius_mm=float(radius),
                role="solid",
            )
        )

    def add_fluid(name: str, start: np.ndarray, end: np.ndarray, radius: float) -> None:
        fluids.append(
            CylinderSpec(
                name=f"{adapter_id}_{name}",
                start_mm=tuple(float(value) for value in start),
                end_mm=tuple(float(value) for value in end),
                radius_mm=float(radius),
                role="fluid_cut",
            )
        )

    add_solid(
        "sleeve_body",
        center - axis * 1.5,
        center + axis * sleeve_length,
        sleeve_od / 2.0,
    )
    add_solid(
        "flange_body",
        center - axis * 0.5,
        center + axis * flange_thickness,
        flange_od / 2.0,
    )

    first_barb_distance = max(flange_thickness + 5.0, sleeve_length * 0.48)
    for index in range(barb_count):
        distance = first_barb_distance + index * barb_spacing
        if distance > sleeve_length - max(barb_width / 2.0, 0.5):
            continue
        add_solid(
            f"barb_{index + 1}",
            center + axis * (distance - barb_width / 2.0),
            center + axis * (distance + barb_width / 2.0),
            barb_od / 2.0,
        )

    inward_overlap = max(wall_thickness_mm + 4.0, tube_id * 0.6)
    outward_extra = max(6.0, barb_width + 4.0)
    add_fluid(
        "main_bore",
        center - axis * inward_overlap,
        center + axis * (sleeve_length + outward_extra),
        tube_id / 2.0 + bore_clearance_mm,
    )

    sleeve_anchor = center + axis * (sleeve_length * 0.55)
    sleeve_outer_radius = sleeve_od / 2.0
    boss_length = max(12.0, sleeve_outer_radius * 1.8)
    boss_radius = max(tap_diameter / 2.0 + pressure_tap_wall_mm, 3.5)
    add_solid(
        "pressure_tap_boss",
        sleeve_anchor + u * (sleeve_outer_radius * 0.55),
        sleeve_anchor + u * (sleeve_outer_radius + boss_length),
        boss_radius,
    )
    add_fluid(
        "pressure_tap_bore",
        sleeve_anchor - u * (tube_id / 2.0 + 1.5),
        sleeve_anchor + u * (sleeve_outer_radius + boss_length + 3.0),
        tap_diameter / 2.0 + bore_clearance_mm,
    )

    return solids, fluids


def _cylinder_bounds(cylinder: CylinderSpec) -> tuple[np.ndarray, np.ndarray]:
    start = np.array(cylinder.start_mm, dtype=float)
    end = np.array(cylinder.end_mm, dtype=float)
    radius = cylinder.radius_mm
    return np.minimum(start, end) - radius, np.maximum(start, end) + radius


def _grid_from_bounds(
    lumen_mask: np.ndarray,
    spacing: tuple[float, float, float],
    cylinders: tuple[CylinderSpec, ...],
    wall_thickness_mm: float,
    voxel_size_mm: float,
    margin_mm: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    coords = np.argwhere(lumen_mask)
    physical = coords * np.array(spacing, dtype=float)
    mins = physical.min(axis=0) - wall_thickness_mm
    maxs = physical.max(axis=0) + wall_thickness_mm

    for cylinder in cylinders:
        cmin, cmax = _cylinder_bounds(cylinder)
        mins = np.minimum(mins, cmin)
        maxs = np.maximum(maxs, cmax)

    mins -= margin_mm
    maxs += margin_mm
    origin = np.floor(mins / voxel_size_mm) * voxel_size_mm
    shape_array = np.ceil((maxs - origin) / voxel_size_mm).astype(int) + 3
    return origin.astype(float), tuple(int(value) for value in shape_array)


def _sample_lumen_mask(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    origin: np.ndarray,
    shape: tuple[int, int, int],
    voxel_size_mm: float,
) -> np.ndarray:
    axes = [
        origin[axis] + np.arange(shape[axis]) * voxel_size_mm
        for axis in range(3)
    ]
    indices = [
        np.rint(axis_values / spacing[axis]).astype(int)
        for axis, axis_values in enumerate(axes)
    ]
    valid = [
        (axis_indices >= 0) & (axis_indices < mask.shape[axis])
        for axis, axis_indices in enumerate(indices)
    ]

    sampled = np.zeros(shape, dtype=bool)
    if not all(np.any(axis_valid) for axis_valid in valid):
        return sampled

    source = mask[
        np.ix_(
            indices[0][valid[0]],
            indices[1][valid[1]],
            indices[2][valid[2]],
        )
    ]
    sampled[np.ix_(valid[0], valid[1], valid[2])] = source
    return sampled


def _apply_cylinder(
    volume: np.ndarray,
    cylinder: CylinderSpec,
    origin: np.ndarray,
    voxel_size_mm: float,
    add: bool,
    chunk_size: int = 32,
) -> None:
    start = np.array(cylinder.start_mm, dtype=np.float32)
    end = np.array(cylinder.end_mm, dtype=np.float32)
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 0:
        return

    radius_squared = float(cylinder.radius_mm * cylinder.radius_mm)
    y = (origin[1] + np.arange(volume.shape[1], dtype=np.float32) * voxel_size_mm)[None, :, None]
    z = (origin[2] + np.arange(volume.shape[2], dtype=np.float32) * voxel_size_mm)[None, None, :]

    for x0 in range(0, volume.shape[0], chunk_size):
        x1 = min(x0 + chunk_size, volume.shape[0])
        x = (origin[0] + np.arange(x0, x1, dtype=np.float32) * voxel_size_mm)[:, None, None]
        px = x - start[0]
        py = y - start[1]
        pz = z - start[2]
        t = (px * direction[0] + py * direction[1] + pz * direction[2]) / length_squared
        t = np.clip(t, 0.0, 1.0)
        distance_squared = px * px + py * py + pz * pz - t * t * length_squared
        selection = distance_squared <= radius_squared
        block = volume[x0:x1]
        if add:
            block |= selection
        else:
            block[selection] = False


def _mesh_from_mask(mask: np.ndarray, origin: np.ndarray, voxel_size_mm: float):
    _, _, _, _, _, measure, trimesh, _ = _import_dependencies()
    if not mask.any():
        raise ValueError("Cannot mesh an empty mask")

    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    vertices, faces, normals, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
    )
    vertices += origin - voxel_size_mm
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    _fix_normals(mesh)
    return mesh


def _maybe_decimate_preserving_watertight(mesh, target_max_faces: int, notes: list[str]):
    if target_max_faces <= 0 or len(mesh.faces) <= target_max_faces:
        return mesh

    original = mesh.copy()
    candidate, decimated, decimation_note = _decimate(mesh, target_max_faces)
    if decimation_note:
        notes.append(decimation_note)
        return original
    if decimated and original.is_watertight and not candidate.is_watertight:
        notes.append("decimation_skipped_to_preserve_watertight_mesh")
        return original
    if decimated:
        notes.append(f"decimated_to_target_max_faces={target_max_faces}")
    return candidate


def _write_meshes(mesh, output_base: Path, formats: tuple[str, ...]) -> tuple[str, ...]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        normalized_fmt = fmt.lower().lstrip(".")
        path = output_base.with_suffix(f".{normalized_fmt}")
        mesh.export(path)
        paths.append(str(path))
    return tuple(paths)


def _render_cartridge_preview(path: Path, cartridge_mesh, fluid_core_mesh) -> None:
    plt, Patch, Poly3DCollection, *_ = _import_dependencies()
    rng = np.random.default_rng(17)
    fig = plt.figure(figsize=(11, 8), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")

    cartridge_triangles = cartridge_mesh.triangles
    if len(cartridge_triangles) > 80_000:
        cartridge_triangles = cartridge_triangles[
            rng.choice(len(cartridge_triangles), size=80_000, replace=False)
        ]
    fluid_triangles = fluid_core_mesh.triangles
    if len(fluid_triangles) > 60_000:
        fluid_triangles = fluid_triangles[rng.choice(len(fluid_triangles), size=60_000, replace=False)]

    ax.add_collection3d(
        Poly3DCollection(
            cartridge_triangles,
            facecolors="#d9b06f",
            edgecolors="none",
            alpha=0.28,
            rasterized=True,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            fluid_triangles,
            facecolors="#2f80ed",
            edgecolors="none",
            alpha=0.54,
            rasterized=True,
        )
    )

    bounds = [cartridge_mesh.bounds, fluid_core_mesh.bounds]
    mins = np.vstack([item[0] for item in bounds]).min(axis=0)
    maxs = np.vstack([item[1] for item in bounds]).max(axis=0)
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2) * 1.15
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-52)
    ax.set_title("Printable Vascular Flow Cartridge: Solid Body + Fluid Path Core", pad=18)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.legend(
        handles=[
            Patch(facecolor="#d9b06f", alpha=0.28, label="printable cartridge body"),
            Patch(facecolor="#2f80ed", alpha=0.54, label="subtracted fluid path core"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_cartridge_spec(path: Path, result: PrintableCartridgeResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "coordinate_units": "mm",
        "construction_method": "voxel_boolean_outer_shell_plus_adapter_solids_minus_lumen_bores_and_pressure_tap_bores",
        "voxel_size_mm": result.voxel_size_mm,
        "wall_thickness_mm": result.wall_thickness_mm,
        "bore_clearance_mm": result.bore_clearance_mm,
        "grid_shape": list(result.grid_shape),
        "solid_voxels": result.solid_voxels,
        "fluid_voxels": result.fluid_voxels,
        "outputs": {
            "cartridge": list(result.cartridge_paths),
            "fluid_core": list(result.fluid_core_paths),
            "preview_png": result.preview_png_path,
        },
        "cylinders": [
            {
                "name": cylinder.name,
                "role": cylinder.role,
                "start_mm": list(cylinder.start_mm),
                "end_mm": list(cylinder.end_mm),
                "radius_mm": cylinder.radius_mm,
            }
            for cylinder in result.cylinders
        ],
        "manufacturing_notes": [
            "Inspect the fluid core mesh before printing to verify channel continuity.",
            "Prototype in clear resin or a transparent material if available so bubbles and leaks are visible.",
            "Use the STL as a first engineering prototype; convert to parametric CAD before adding production threads, O-rings, or luer hardware.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_cartridge_report(result: PrintableCartridgeResult) -> str:
    lines = [
        "# Printable Vascular Flow Cartridge Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Outputs",
        "",
        f"- Printable cartridge: {', '.join(f'`{Path(path).name}`' for path in result.cartridge_paths)}",
        f"- Fluid path core / negative tool: {', '.join(f'`{Path(path).name}`' for path in result.fluid_core_paths)}",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Build Method",
        "",
        "- Built as a voxel-boolean prototype because no local mesh boolean engine was available.",
        "- Solid region = dilated anatomical true-lumen shell + inlet/outlet adapter bodies + pressure tap bosses.",
        "- Fluid region subtracted from the solid = original true-lumen mask + inlet/outlet bores + pressure tap bores.",
        "- The exported `fluid_core` mesh is the negative tool used for inspection of channel continuity.",
        "",
        "## Geometry Summary",
        "",
        f"- Voxel size: {result.voxel_size_mm:.2f} mm",
        f"- Nominal wall thickness: {result.wall_thickness_mm:.2f} mm",
        f"- Bore clearance: {result.bore_clearance_mm:.2f} mm radial",
        f"- Voxel grid shape: {result.grid_shape[0]} x {result.grid_shape[1]} x {result.grid_shape[2]}",
        f"- Solid voxels after subtraction: {result.solid_voxels}",
        f"- Fluid-path voxels: {result.fluid_voxels}",
        f"- Cartridge mesh faces: {result.cartridge_faces}",
        f"- Fluid-core mesh faces: {result.fluid_core_faces}",
        "",
        "## Modeled Bores And Bodies",
        "",
        "| name | role | radius mm | start mm | end mm |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for cylinder in result.cylinders:
        start = ", ".join(f"{value:.2f}" for value in cylinder.start_mm)
        end = ", ".join(f"{value:.2f}" for value in cylinder.end_mm)
        lines.append(f"| {cylinder.name} | {cylinder.role} | {cylinder.radius_mm:.2f} | {start} | {end} |")

    lines.extend(
        [
            "",
            "## Prototype Notes",
            "",
            "- This is the first watertight printable body candidate, not the final product CAD.",
            "- The voxel resolution intentionally rounds sharp barb/flange details; use it for first fit/leak tests, then rebuild final fittings parametrically in FreeCAD or similar CAD.",
            "- Pressure taps are modeled as real through-bores into the adapter sleeves. Add threaded inserts, luer bosses, or bonded tubing in the next mechanical pass.",
            "- Before wet testing, inspect the `fluid_core` mesh and run a dry air/water leak check at low pressure.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_printable_vascular_cartridge(
    labels_path: str | Path,
    flow_loop_spec_path: str | Path,
    output_dir: str | Path,
    label_id: int = 1,
    centerline_csv_path: str | Path | None = None,
    formats: tuple[str, ...] = ("stl", "ply"),
    voxel_size_mm: float = 1.0,
    wall_thickness_mm: float = 2.5,
    bore_clearance_mm: float = 0.3,
    pressure_tap_wall_mm: float = 2.0,
    target_max_faces: int = 250_000,
    report_path: str | Path | None = None,
) -> PrintableCartridgeResult:
    _, _, _, _, ndimage, *_ = _import_dependencies()
    if voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be positive")
    if wall_thickness_mm <= 0:
        raise ValueError("wall_thickness_mm must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    mask, image, removed_components = _load_label_mask(Path(labels_path), label_id)
    if removed_components:
        notes.append(f"kept_largest_lumen_component_removed={removed_components}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    centerline = _read_centerline_csv(centerline_csv_path)
    if centerline is not None:
        notes.append(f"centerline_loaded_points={len(centerline)}")

    spec = _load_flow_loop_spec(flow_loop_spec_path)
    case_id = str(spec["case_id"])

    solid_cylinders: list[CylinderSpec] = []
    fluid_cylinders: list[CylinderSpec] = []
    for adapter in spec["adapters"]:
        solids, fluids = _adapter_cylinders(
            adapter=adapter,
            wall_thickness_mm=wall_thickness_mm,
            bore_clearance_mm=bore_clearance_mm,
            pressure_tap_wall_mm=pressure_tap_wall_mm,
        )
        solid_cylinders.extend(solids)
        fluid_cylinders.extend(fluids)

    all_cylinders = tuple([*solid_cylinders, *fluid_cylinders])
    margin_mm = max(8.0, wall_thickness_mm * 2.5)
    origin, shape = _grid_from_bounds(
        lumen_mask=mask,
        spacing=spacing,
        cylinders=all_cylinders,
        wall_thickness_mm=wall_thickness_mm,
        voxel_size_mm=voxel_size_mm,
        margin_mm=margin_mm,
    )

    lumen = _sample_lumen_mask(mask, spacing=spacing, origin=origin, shape=shape, voxel_size_mm=voxel_size_mm)
    if not lumen.any():
        raise ValueError("Sampled lumen mask is empty; check voxel size and input geometry")

    # Smooth tiny sampling gaps without intentionally closing the inlet/outlet channels.
    lumen = ndimage.binary_closing(lumen, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    fluid = lumen.copy()
    for cylinder in fluid_cylinders:
        _apply_cylinder(fluid, cylinder, origin=origin, voxel_size_mm=voxel_size_mm, add=True)

    distance_to_lumen = ndimage.distance_transform_edt(~lumen, sampling=(voxel_size_mm,) * 3)
    solid = distance_to_lumen <= wall_thickness_mm
    for cylinder in solid_cylinders:
        _apply_cylinder(solid, cylinder, origin=origin, voxel_size_mm=voxel_size_mm, add=True)

    solid &= ~fluid
    solid = ndimage.binary_closing(solid, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)

    cartridge_mesh = _mesh_from_mask(solid, origin=origin, voxel_size_mm=voxel_size_mm)
    fluid_core_mesh = _mesh_from_mask(fluid, origin=origin, voxel_size_mm=voxel_size_mm)
    cartridge_mesh = _maybe_decimate_preserving_watertight(cartridge_mesh, target_max_faces, notes)
    fluid_core_mesh = _maybe_decimate_preserving_watertight(fluid_core_mesh, target_max_faces, notes)

    cartridge_base = output / f"{case_id}_printable_flow_cartridge_v001"
    fluid_core_base = output / f"{case_id}_fluid_path_core_v001"
    cartridge_paths = _write_meshes(cartridge_mesh, cartridge_base, formats)
    fluid_core_paths = _write_meshes(fluid_core_mesh, fluid_core_base, formats)

    preview_png = output / f"{case_id}_printable_flow_cartridge_preview_v001.png"
    spec_yaml = output / f"{case_id}_printable_flow_cartridge_spec_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_printable_flow_cartridge_report_v001.md"

    result = PrintableCartridgeResult(
        case_id=case_id,
        output_dir=str(output),
        cartridge_paths=cartridge_paths,
        fluid_core_paths=fluid_core_paths,
        preview_png_path=str(preview_png),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        voxel_size_mm=voxel_size_mm,
        wall_thickness_mm=wall_thickness_mm,
        bore_clearance_mm=bore_clearance_mm,
        solid_voxels=int(solid.sum()),
        fluid_voxels=int(fluid.sum()),
        grid_shape=shape,
        cartridge_faces=int(len(cartridge_mesh.faces)),
        fluid_core_faces=int(len(fluid_core_mesh.faces)),
        cylinders=all_cylinders,
        notes=tuple(notes),
    )

    _render_cartridge_preview(preview_png, cartridge_mesh, fluid_core_mesh)
    _write_cartridge_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_cartridge_report(result))
    return result


def format_printable_cartridge_result(result: PrintableCartridgeResult) -> str:
    return _format_cartridge_report(result)
