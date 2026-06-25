from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np

from .mesh_clean import _decimate, _fill_holes, _fix_normals


def _import_dependencies():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
        from scipy import ndimage  # type: ignore
        from scipy.sparse import csr_matrix  # type: ignore
        from scipy.sparse.csgraph import dijkstra  # type: ignore
        from scipy.signal import savgol_filter  # type: ignore
        from skimage import measure  # type: ignore
        from skimage.morphology import skeletonize  # type: ignore
        import trimesh  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Vascular module preparation requires nibabel, scipy, scikit-image, "
            "matplotlib, trimesh, and PyYAML."
        ) from exc
    return (
        plt,
        Patch,
        Poly3DCollection,
        nib,
        ndimage,
        csr_matrix,
        dijkstra,
        savgol_filter,
        measure,
        skeletonize,
        trimesh,
        yaml,
    )


@dataclass(frozen=True)
class VascularPort:
    id: str
    center_mm: tuple[float, float, float]
    outward_normal: tuple[float, float, float]
    equivalent_radius_mm: float
    equivalent_diameter_mm: float
    suggested_tube_inner_diameter_mm: float
    source_slice_index: int


@dataclass(frozen=True)
class VascularModuleResult:
    case_id: str
    output_dir: str
    smoothed_mesh_paths: tuple[str, ...]
    centerline_csv_path: str
    centerline_obj_path: str
    ports_yaml_path: str
    port_plane_mesh_paths: tuple[str, ...]
    preview_png_path: str
    report_path: str
    mask_voxels: int
    centerline_points: int
    centerline_length_mm: float
    ports: tuple[VascularPort, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PortAdapter:
    id: str
    center_mm: tuple[float, float, float]
    axis: tuple[float, float, float]
    tube_inner_diameter_mm: float
    lumen_transition_diameter_mm: float
    sleeve_outer_diameter_mm: float
    sleeve_length_mm: float
    barb_count: int
    barb_outer_diameter_mm: float
    barb_width_mm: float
    barb_spacing_mm: float
    flange_outer_diameter_mm: float
    flange_thickness_mm: float
    port_plane_radius_mm: float
    pressure_tap_center_mm: tuple[float, float, float]
    pressure_tap_diameter_mm: float
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class FlowLoopDesignResult:
    case_id: str
    adapters: tuple[PortAdapter, ...]
    adapter_paths: tuple[str, ...]
    port_plane_paths: tuple[str, ...]
    assembly_paths: tuple[str, ...]
    preview_png_path: str
    spec_yaml_path: str
    report_path: str
    notes: tuple[str, ...]


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
    removed = int(count - 1)
    return connected == largest, image, removed


def _smooth_mesh_from_mask(mask: np.ndarray, spacing: tuple[float, float, float], sigma: float):
    _, _, _, _, ndimage, _, _, _, measure, _, trimesh, _ = _import_dependencies()
    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    vertices, faces, normals, _ = measure.marching_cubes(smooth, level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    _fix_normals(mesh)
    _fill_holes(mesh, "single")
    _fix_normals(mesh)
    return mesh


def _savgol(values: np.ndarray):
    _, _, _, _, _, _, _, savgol_filter, *_ = _import_dependencies()
    n = len(values)
    if n < 7:
        return values
    window = min(21, n if n % 2 == 1 else n - 1)
    if window < 7:
        return values
    return savgol_filter(values, window_length=window, polyorder=3, mode="interp")


def _axial_centerline_from_mask(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    points: list[tuple[float, float, float, float, float, int]] = []
    voxel_area = spacing[0] * spacing[1]

    for z_index in range(mask.shape[2]):
        coords = np.argwhere(mask[:, :, z_index])
        if len(coords) == 0:
            continue

        centroid_xy = coords.mean(axis=0)
        area = len(coords) * voxel_area
        radius = math.sqrt(area / math.pi)
        points.append(
            (
                float(centroid_xy[0] * spacing[0]),
                float(centroid_xy[1] * spacing[1]),
                float(z_index * spacing[2]),
                float(radius),
                float(area),
                int(z_index),
            )
        )

    if len(points) < 2:
        raise ValueError("At least two occupied slices are required to estimate a centerline")

    centerline = np.array(points, dtype=float)
    centerline[:, 0] = _savgol(centerline[:, 0])
    centerline[:, 1] = _savgol(centerline[:, 1])
    centerline[:, 3] = np.maximum(_savgol(centerline[:, 3]), 0.1)
    centerline[:, 4] = math.pi * centerline[:, 3] ** 2
    return centerline


def _smooth_centerline(centerline: np.ndarray) -> np.ndarray:
    smoothed = centerline.copy()
    for column in (0, 1, 2, 3):
        smoothed[:, column] = _savgol(smoothed[:, column])
    smoothed[:, 3] = np.maximum(smoothed[:, 3], 0.1)
    smoothed[:, 4] = math.pi * smoothed[:, 3] ** 2

    # Preserve exact endpoints for port placement.
    smoothed[0, :3] = centerline[0, :3]
    smoothed[-1, :3] = centerline[-1, :3]
    smoothed[0, 5] = centerline[0, 5]
    smoothed[-1, 5] = centerline[-1, 5]
    return smoothed


def _reconstruct_path(predecessors: np.ndarray, start: int, end: int) -> list[int]:
    path = [int(end)]
    current = int(end)
    while current != start:
        current = int(predecessors[current])
        if current < 0:
            raise ValueError("Could not reconstruct skeleton path")
        path.append(current)
    path.reverse()
    return path


def _skeleton_centerline_from_mask(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, str]:
    _, _, _, _, ndimage, csr_matrix, dijkstra, _, _, skeletonize, *_ = _import_dependencies()

    skeleton = skeletonize(mask).astype(bool)
    coords = np.argwhere(skeleton)
    if len(coords) < 2:
        raise ValueError("Skeleton centerline failed: fewer than two skeleton voxels")

    flat = np.ravel_multi_index(coords.T, mask.shape)
    flat_to_node = {int(value): index for index, value in enumerate(flat)}
    offsets = [
        np.array([dx, dy, dz], dtype=int)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    shape = np.array(mask.shape)
    spacing_array = np.array(spacing, dtype=float)

    for node, coord in enumerate(coords):
        for offset in offsets:
            neighbor = coord + offset
            if np.any(neighbor < 0) or np.any(neighbor >= shape):
                continue
            neighbor_flat = int(np.ravel_multi_index(neighbor, mask.shape))
            neighbor_node = flat_to_node.get(neighbor_flat)
            if neighbor_node is None:
                continue
            rows.append(node)
            cols.append(neighbor_node)
            weights.append(float(np.linalg.norm(offset * spacing_array)))

    graph = csr_matrix((weights, (rows, cols)), shape=(len(coords), len(coords)))
    degrees = np.diff(graph.indptr)
    endpoints = np.where(degrees <= 1)[0]
    if len(endpoints) < 2:
        endpoints = np.arange(len(coords))

    start = int(endpoints[np.argmin(coords[endpoints, 2])])
    preferred_end = int(endpoints[np.argmax(coords[endpoints, 2])])
    distances, predecessors = dijkstra(
        graph,
        directed=False,
        indices=start,
        return_predecessors=True,
    )

    if np.isfinite(distances[preferred_end]):
        end = preferred_end
    else:
        reachable_endpoints = endpoints[np.isfinite(distances[endpoints])]
        if len(reachable_endpoints) == 0:
            raise ValueError("Skeleton centerline failed: no reachable endpoint")
        end = int(reachable_endpoints[np.argmax(distances[reachable_endpoints])])

    path_nodes = _reconstruct_path(predecessors, start, end)
    path_coords = coords[path_nodes]
    distance_map = ndimage.distance_transform_edt(mask, sampling=spacing)
    radii = distance_map[tuple(path_coords.T)]
    physical = path_coords * spacing_array
    areas = math.pi * np.maximum(radii, 0.1) ** 2

    centerline = np.column_stack(
        [
            physical[:, 0],
            physical[:, 1],
            physical[:, 2],
            np.maximum(radii, 0.1),
            areas,
            path_coords[:, 2],
        ]
    )
    return _smooth_centerline(centerline), (
        f"Skeleton centerline used {len(path_nodes)} path points from "
        f"{len(coords)} skeleton voxels."
    )


def _centerline_from_mask(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    method: str,
) -> tuple[np.ndarray, str]:
    if method == "axial":
        return _axial_centerline_from_mask(mask, spacing), "Axial centroid centerline was used."
    if method == "skeleton":
        return _skeleton_centerline_from_mask(mask, spacing)
    raise ValueError("centerline method must be one of: skeleton, axial")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm


def _suggest_tube_id(diameter_mm: float) -> float:
    return math.ceil(max(6.0, diameter_mm * 1.1) * 2.0) / 2.0


def _ports_from_centerline(centerline: np.ndarray, lookahead: int = 8) -> tuple[VascularPort, VascularPort]:
    coords = centerline[:, :3]
    radius = centerline[:, 3]
    n = len(centerline)
    step = min(max(1, lookahead), n - 1)

    start_tangent = _unit(coords[step] - coords[0])
    end_tangent = _unit(coords[-1] - coords[-1 - step])
    start_radius = float(np.median(radius[: step + 1]))
    end_radius = float(np.median(radius[-step - 1 :]))

    start_diameter = 2 * start_radius
    end_diameter = 2 * end_radius

    return (
        VascularPort(
            id="z_min_port",
            center_mm=tuple(float(value) for value in coords[0]),
            outward_normal=tuple(float(value) for value in -start_tangent),
            equivalent_radius_mm=start_radius,
            equivalent_diameter_mm=start_diameter,
            suggested_tube_inner_diameter_mm=_suggest_tube_id(start_diameter),
            source_slice_index=int(centerline[0, 5]),
        ),
        VascularPort(
            id="z_max_port",
            center_mm=tuple(float(value) for value in coords[-1]),
            outward_normal=tuple(float(value) for value in end_tangent),
            equivalent_radius_mm=end_radius,
            equivalent_diameter_mm=end_diameter,
            suggested_tube_inner_diameter_mm=_suggest_tube_id(end_diameter),
            source_slice_index=int(centerline[-1, 5]),
        ),
    )


def _rotation_from_z(normal: np.ndarray) -> np.ndarray:
    source = np.array([0.0, 0.0, 1.0])
    target = _unit(normal)
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))

    if np.linalg.norm(cross) < 1e-9:
        if dot > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = _unit(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(w, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(w, helper))
    v = _unit(np.cross(w, u))
    return u, v, w


def _disk_mesh(center: tuple[float, float, float], normal: tuple[float, float, float], radius: float):
    *_, trimesh, _ = _import_dependencies()
    segments = 96
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    vertices = np.zeros((segments + 1, 3), dtype=float)
    vertices[1:, 0] = np.cos(angles) * radius
    vertices[1:, 1] = np.sin(angles) * radius
    faces = [[0, i, 1 + (i % segments)] for i in range(1, segments + 1)]
    rotation = _rotation_from_z(np.array(normal, dtype=float))
    vertices = vertices @ rotation.T + np.array(center, dtype=float)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _cylinder_between(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    length: float,
    sections: int = 96,
):
    *_, trimesh, _ = _import_dependencies()
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    rotation = np.eye(4)
    rotation[:3, :3] = _rotation_from_z(axis)
    mesh.apply_transform(rotation)
    mesh.apply_translation(center)
    return mesh


def _ring_mesh(
    center: np.ndarray,
    axis: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    thickness: float,
    sections: int = 128,
):
    *_, trimesh, _ = _import_dependencies()
    angles = np.linspace(0, 2 * math.pi, sections, endpoint=False)
    u, v, w = _orthonormal_basis(axis)
    half = thickness / 2.0
    vertices: list[np.ndarray] = []
    for z_offset in (-half, half):
        for radius in (outer_radius, inner_radius):
            for angle in angles:
                vertices.append(center + w * z_offset + radius * (math.cos(angle) * u + math.sin(angle) * v))

    vertices_array = np.array(vertices)

    def idx(layer: int, radius_index: int, i: int) -> int:
        return layer * sections * 2 + radius_index * sections + (i % sections)

    faces: list[list[int]] = []
    for i in range(sections):
        j = i + 1
        faces.extend(
            [
                [idx(0, 0, i), idx(0, 0, j), idx(1, 0, j)],
                [idx(0, 0, i), idx(1, 0, j), idx(1, 0, i)],
                [idx(0, 1, i), idx(1, 1, j), idx(0, 1, j)],
                [idx(0, 1, i), idx(1, 1, i), idx(1, 1, j)],
                [idx(1, 0, i), idx(1, 0, j), idx(1, 1, j)],
                [idx(1, 0, i), idx(1, 1, j), idx(1, 1, i)],
                [idx(0, 0, i), idx(0, 1, j), idx(0, 0, j)],
                [idx(0, 0, i), idx(0, 1, i), idx(0, 1, j)],
            ]
        )

    mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces, process=True)
    _fix_normals(mesh)
    return mesh


def _side_tap_marker(
    port: VascularPort,
    sleeve_center: np.ndarray,
    sleeve_axis: np.ndarray,
    sleeve_outer_radius: float,
    tap_diameter: float,
):
    u, _, _ = _orthonormal_basis(sleeve_axis)
    tap_axis = u
    tap_length = max(10.0, sleeve_outer_radius * 1.6)
    tap_center = sleeve_center + tap_axis * (sleeve_outer_radius + tap_length / 2.0)
    mesh = _cylinder_between(
        center=tap_center,
        axis=tap_axis,
        radius=tap_diameter / 2.0,
        length=tap_length,
        sections=32,
    )
    return mesh, tap_center


def _write_centerline_csv(path: Path, centerline: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "x_mm", "y_mm", "z_mm", "radius_equiv_mm", "area_mm2", "source_slice_index"])
        for index, row in enumerate(centerline):
            writer.writerow([index, *[f"{value:.6f}" for value in row[:5]], int(row[5])])


def _write_centerline_obj(path: Path, centerline: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as obj:
        obj.write("# Smoothed centerline polyline\n")
        for point in centerline[:, :3]:
            obj.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        obj.write("l " + " ".join(str(index) for index in range(1, len(centerline) + 1)) + "\n")


def _write_ports_yaml(path: Path, ports: tuple[VascularPort, ...], case_id: str) -> None:
    *_, yaml = _import_dependencies()
    payload: dict[str, Any] = {
        "case_id": case_id,
        "coordinate_units": "mm",
        "ports": [
            {
                "id": port.id,
                "center_mm": list(port.center_mm),
                "outward_normal": list(port.outward_normal),
                "equivalent_radius_mm": port.equivalent_radius_mm,
                "equivalent_diameter_mm": port.equivalent_diameter_mm,
                "suggested_tube_inner_diameter_mm": port.suggested_tube_inner_diameter_mm,
                "source_slice_index": port.source_slice_index,
                "cad_note": "Use this as an initial cut-plane normal. Verify in CAD before machining or printing.",
            }
            for port in ports
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _read_ports_yaml(path: str | Path) -> tuple[str, tuple[VascularPort, ...]]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    ports = []
    for item in data["ports"]:
        ports.append(
            VascularPort(
                id=str(item["id"]),
                center_mm=tuple(float(value) for value in item["center_mm"]),
                outward_normal=tuple(float(value) for value in item["outward_normal"]),
                equivalent_radius_mm=float(item["equivalent_radius_mm"]),
                equivalent_diameter_mm=float(item["equivalent_diameter_mm"]),
                suggested_tube_inner_diameter_mm=float(item["suggested_tube_inner_diameter_mm"]),
                source_slice_index=int(item["source_slice_index"]),
            )
        )
    return str(data["case_id"]), tuple(ports)


def _line_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1).sum())


def _render_preview(path: Path, mesh, centerline: np.ndarray, ports: tuple[VascularPort, ...]) -> None:
    plt, Patch, Poly3DCollection, *_ = _import_dependencies()
    rng = np.random.default_rng(7)
    triangles = mesh.triangles
    if len(triangles) > 60_000:
        triangles = triangles[rng.choice(len(triangles), size=60_000, replace=False)]

    fig = plt.figure(figsize=(10, 8), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")

    mesh_collection = Poly3DCollection(
        triangles,
        facecolors="#63a7d0",
        edgecolors="none",
        alpha=0.30,
        rasterized=True,
    )
    ax.add_collection3d(mesh_collection)
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color="#d33f2f", linewidth=3)

    for port in ports:
        center = np.array(port.center_mm)
        normal = np.array(port.outward_normal)
        ax.scatter(*center, s=60, color="#1d3557")
        ax.quiver(*center, *(normal * port.suggested_tube_inner_diameter_mm), color="#1d3557", linewidth=2)
        ax.text(*(center + normal * 8), port.id, color="#1d3557", fontsize=8)

    bounds = mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2) * 1.1
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-52)
    ax.set_title("ImageTBAD Case 125 True Lumen: Smoothed Surface + Centerline + Ports", pad=18)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.legend(
        handles=[
            Patch(facecolor="#63a7d0", alpha=0.30, label="smoothed lumen"),
            Patch(facecolor="#d33f2f", label="centerline"),
            Patch(facecolor="#1d3557", label="port normals"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _render_flow_loop_preview(
    path: Path,
    lumen_mesh,
    adapter_meshes: tuple[object, ...],
    port_plane_meshes: tuple[object, ...],
    adapters: tuple[PortAdapter, ...],
) -> None:
    plt, Patch, Poly3DCollection, *_ = _import_dependencies()
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(10, 8), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")

    lumen_triangles = lumen_mesh.triangles
    if len(lumen_triangles) > 50_000:
        lumen_triangles = lumen_triangles[rng.choice(len(lumen_triangles), size=50_000, replace=False)]
    ax.add_collection3d(
        Poly3DCollection(
            lumen_triangles,
            facecolors="#63a7d0",
            edgecolors="none",
            alpha=0.28,
            rasterized=True,
        )
    )

    for mesh in adapter_meshes:
        triangles = mesh.triangles
        ax.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolors="#f28e2b",
                edgecolors="none",
                alpha=0.55,
                rasterized=True,
            )
        )

    for mesh in port_plane_meshes:
        triangles = mesh.triangles
        ax.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolors="#7f3c8d",
                edgecolors="none",
                alpha=0.35,
                rasterized=True,
            )
        )

    bounds = [lumen_mesh.bounds] + [mesh.bounds for mesh in adapter_meshes] + [mesh.bounds for mesh in port_plane_meshes]
    mins = np.vstack([item[0] for item in bounds]).min(axis=0)
    maxs = np.vstack([item[1] for item in bounds]).max(axis=0)
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2) * 1.15
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-52)
    ax.set_title("Vascular Flow Module: Lumen + Port Adapter Reference Geometry", pad=18)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")

    for adapter in adapters:
        point = np.array(adapter.pressure_tap_center_mm)
        ax.scatter(*point, s=25, color="#2ca02c")
        ax.text(*point, f"{adapter.id} tap", fontsize=7, color="#2ca02c")

    ax.legend(
        handles=[
            Patch(facecolor="#63a7d0", alpha=0.28, label="smoothed lumen"),
            Patch(facecolor="#7f3c8d", alpha=0.35, label="port planes"),
            Patch(facecolor="#f28e2b", alpha=0.55, label="adapter cylinders/barbs/flanges"),
            Patch(facecolor="#2ca02c", label="pressure tap markers"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _format_report(result: VascularModuleResult) -> str:
    lines = [
        "# Vascular Module Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Outputs",
        "",
        f"- Smoothed mesh files: {', '.join(f'`{Path(path).name}`' for path in result.smoothed_mesh_paths)}",
        f"- Centerline CSV: `{Path(result.centerline_csv_path).name}`",
        f"- Centerline OBJ: `{Path(result.centerline_obj_path).name}`",
        f"- Port specification: `{Path(result.ports_yaml_path).name}`",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        "",
        "## Geometry Summary",
        "",
        f"- Mask voxels: {result.mask_voxels}",
        f"- Centerline points: {result.centerline_points}",
        f"- Centerline length: {result.centerline_length_mm:.2f} mm",
        "",
        "## Port Planes",
        "",
        "| port | center mm | outward normal | equiv diameter mm | suggested tube ID mm | source slice |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]

    for port in result.ports:
        center = ", ".join(f"{value:.2f}" for value in port.center_mm)
        normal = ", ".join(f"{value:.4f}" for value in port.outward_normal)
        lines.append(
            f"| {port.id} | {center} | {normal} | {port.equivalent_diameter_mm:.2f} | "
            f"{port.suggested_tube_inner_diameter_mm:.1f} | {port.source_slice_index} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The centerline is an engineering estimate for CAD setup, not a validated computational-fluid-dynamics centerline extraction.",
            "- The port planes are initial CAD references. They should be visually checked, trimmed, and converted into proper tube/flange geometry.",
            "- Open vessel ends are intentionally preserved; final flow-loop CAD should cap or sleeve these with printed/mechanical port adapters.",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")

    return "\n".join(lines)


def prepare_vascular_module(
    labels_path: str | Path,
    output_dir: str | Path,
    case_id: str,
    label_id: int = 1,
    smooth_sigma: float = 1.0,
    target_max_faces: int = 60_000,
    centerline_method: str = "skeleton",
    formats: tuple[str, ...] = ("stl", "ply"),
    report_path: str | Path | None = None,
) -> VascularModuleResult:
    _import_dependencies()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    mask, image, removed_components = _load_label_mask(Path(labels_path), label_id)
    if removed_components:
        notes.append(f"Kept largest connected component; removed {removed_components} smaller mask components.")

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    mesh = _smooth_mesh_from_mask(mask, spacing=spacing, sigma=smooth_sigma)
    mesh, decimated, decimation_note = _decimate(mesh, target_max_faces)
    if decimation_note:
        notes.append(decimation_note)
    if decimated:
        notes.append(f"Decimated smoothed lumen to target max faces {target_max_faces}.")
    _fix_normals(mesh)

    smoothed_mesh_paths: list[str] = []
    for fmt in formats:
        normalized_fmt = fmt.lower().lstrip(".")
        mesh_path = output / f"{case_id}_smoothed_lumen_v001.{normalized_fmt}"
        mesh.export(mesh_path)
        smoothed_mesh_paths.append(str(mesh_path))

    centerline, centerline_note = _centerline_from_mask(mask, spacing=spacing, method=centerline_method)
    notes.append(centerline_note)
    ports = _ports_from_centerline(centerline)

    centerline_csv = output / f"{case_id}_centerline_v001.csv"
    centerline_obj = output / f"{case_id}_centerline_v001.obj"
    ports_yaml = output / f"{case_id}_ports_v001.yaml"
    preview_png = output / f"{case_id}_vascular_preview_v001.png"

    _write_centerline_csv(centerline_csv, centerline)
    _write_centerline_obj(centerline_obj, centerline)
    _write_ports_yaml(ports_yaml, ports, case_id=case_id)

    port_plane_paths: list[str] = []
    for port in ports:
        plane_radius = max(port.suggested_tube_inner_diameter_mm * 0.75, port.equivalent_radius_mm * 1.25)
        plane_mesh = _disk_mesh(port.center_mm, port.outward_normal, plane_radius)
        for fmt in formats:
            normalized_fmt = fmt.lower().lstrip(".")
            plane_path = output / f"{case_id}_{port.id}_cut_plane_v001.{normalized_fmt}"
            plane_mesh.export(plane_path)
            port_plane_paths.append(str(plane_path))

    _render_preview(preview_png, mesh, centerline, ports)

    report = Path(report_path) if report_path else output / f"{case_id}_vascular_module_report_v001.md"
    result = VascularModuleResult(
        case_id=case_id,
        output_dir=str(output),
        smoothed_mesh_paths=tuple(smoothed_mesh_paths),
        centerline_csv_path=str(centerline_csv),
        centerline_obj_path=str(centerline_obj),
        ports_yaml_path=str(ports_yaml),
        port_plane_mesh_paths=tuple(port_plane_paths),
        preview_png_path=str(preview_png),
        report_path=str(report),
        mask_voxels=int(mask.sum()),
        centerline_points=int(len(centerline)),
        centerline_length_mm=_line_length(centerline),
        ports=ports,
        notes=tuple(notes),
    )

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_vascular_result(result: VascularModuleResult) -> str:
    return _format_report(result)


def _create_adapter_for_port(
    case_id: str,
    port: VascularPort,
    output_dir: Path,
    formats: tuple[str, ...],
    wall_thickness_mm: float,
    sleeve_length_mm: float,
    barb_count: int,
    barb_height_mm: float,
    barb_width_mm: float,
    barb_spacing_mm: float,
    flange_thickness_mm: float,
    flange_extra_radius_mm: float,
    pressure_tap_diameter_mm: float,
) -> tuple[PortAdapter, object]:
    *_, trimesh, _ = _import_dependencies()
    center = np.array(port.center_mm, dtype=float)
    axis = _unit(np.array(port.outward_normal, dtype=float))
    tube_id = port.suggested_tube_inner_diameter_mm
    sleeve_outer_diameter = tube_id + 2 * wall_thickness_mm
    barb_outer_diameter = sleeve_outer_diameter + 2 * barb_height_mm
    flange_outer_diameter = sleeve_outer_diameter + 2 * flange_extra_radius_mm
    sleeve_center = center + axis * (sleeve_length_mm / 2.0)
    flange_center = center + axis * (flange_thickness_mm / 2.0)

    sleeve = _cylinder_between(
        center=sleeve_center,
        axis=axis,
        radius=sleeve_outer_diameter / 2.0,
        length=sleeve_length_mm,
        sections=96,
    )
    transition = _cylinder_between(
        center=center + axis * 1.0,
        axis=axis,
        radius=max(port.equivalent_diameter_mm, tube_id) / 2.0,
        length=2.0,
        sections=96,
    )
    flange = _ring_mesh(
        center=flange_center,
        axis=axis,
        inner_radius=tube_id / 2.0,
        outer_radius=flange_outer_diameter / 2.0,
        thickness=flange_thickness_mm,
        sections=128,
    )
    barb_meshes = []
    first_barb_distance = max(flange_thickness_mm + 5.0, sleeve_length_mm * 0.48)
    for index in range(barb_count):
        distance = first_barb_distance + index * barb_spacing_mm
        if distance > sleeve_length_mm - barb_width_mm:
            continue
        barb_meshes.append(
            _ring_mesh(
                center=center + axis * distance,
                axis=axis,
                inner_radius=sleeve_outer_diameter / 2.0,
                outer_radius=barb_outer_diameter / 2.0,
                thickness=barb_width_mm,
                sections=96,
            )
        )
    tap_marker, tap_center = _side_tap_marker(
        port=port,
        sleeve_center=center + axis * (sleeve_length_mm * 0.55),
        sleeve_axis=axis,
        sleeve_outer_radius=sleeve_outer_diameter / 2.0,
        tap_diameter=pressure_tap_diameter_mm,
    )

    adapter_mesh = trimesh.util.concatenate([sleeve, transition, flange, *barb_meshes, tap_marker])
    adapter_mesh.merge_vertices()
    adapter_mesh.remove_unreferenced_vertices()
    _fix_normals(adapter_mesh)

    outputs: list[str] = []
    for fmt in formats:
        normalized_fmt = fmt.lower().lstrip(".")
        path = output_dir / f"{case_id}_{port.id}_adapter_reference_v001.{normalized_fmt}"
        adapter_mesh.export(path)
        outputs.append(str(path))

    adapter = PortAdapter(
        id=port.id,
        center_mm=port.center_mm,
        axis=tuple(float(value) for value in axis),
        tube_inner_diameter_mm=tube_id,
        lumen_transition_diameter_mm=max(port.equivalent_diameter_mm, tube_id),
        sleeve_outer_diameter_mm=sleeve_outer_diameter,
        sleeve_length_mm=sleeve_length_mm,
        barb_count=len(barb_meshes),
        barb_outer_diameter_mm=barb_outer_diameter,
        barb_width_mm=barb_width_mm,
        barb_spacing_mm=barb_spacing_mm,
        flange_outer_diameter_mm=flange_outer_diameter,
        flange_thickness_mm=flange_thickness_mm,
        port_plane_radius_mm=max(tube_id * 0.75, port.equivalent_radius_mm * 1.25),
        pressure_tap_center_mm=tuple(float(value) for value in tap_center),
        pressure_tap_diameter_mm=pressure_tap_diameter_mm,
        outputs=tuple(outputs),
    )
    return adapter, adapter_mesh


def _format_flow_loop_report(result: FlowLoopDesignResult) -> str:
    lines = [
        "# Vascular Flow-Loop Design Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## CAD Outputs",
        "",
    ]
    for adapter in result.adapters:
        lines.append(f"- `{adapter.id}` adapter: {', '.join(f'`{Path(path).name}`' for path in adapter.outputs)}")

    lines.extend(
        [
            f"- Reference assembly: {', '.join(f'`{Path(path).name}`' for path in result.assembly_paths)}",
            f"- Port plane references: {', '.join(f'`{Path(path).name}`' for path in result.port_plane_paths)}",
            f"- Preview PNG: `{Path(result.preview_png_path).name}`",
            f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
            "",
            "## Adapter Dimensions",
            "",
            "| adapter | tube ID mm | transition ID mm | sleeve OD mm | sleeve length mm | barbs | barb OD mm | flange OD mm | pressure tap ID mm |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for adapter in result.adapters:
        lines.append(
            f"| {adapter.id} | {adapter.tube_inner_diameter_mm:.1f} | "
            f"{adapter.lumen_transition_diameter_mm:.1f} | {adapter.sleeve_outer_diameter_mm:.1f} | "
            f"{adapter.sleeve_length_mm:.1f} | {adapter.barb_count} | "
            f"{adapter.barb_outer_diameter_mm:.1f} | {adapter.flange_outer_diameter_mm:.1f} | "
            f"{adapter.pressure_tap_diameter_mm:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Flow-Loop Bench Spec",
            "",
            "- Recommended fluid path: reservoir -> bubble trap -> pump -> flow sensor -> `z_min_port` inlet -> true-lumen module -> `z_max_port` outlet -> downstream pressure sensor -> reservoir.",
            "- Pump direction: start at `z_min_port` as inlet and `z_max_port` as outlet. Reverse-flow tests can be added later for sensitivity checks.",
            "- Pump type: programmable peristaltic pump for first wet tests because the fluid path is disposable and leak cleanup is easier. Use a gear pump later if low-pulsation steady flow is required.",
            "- Flow range: start at 0.5, 1.0, 2.0, 3.0, and 5.0 L/min steady-flow setpoints. Record actual flow, pressure drop, temperature, and visible leaks at each point.",
            "- Pressure sensors: two inline or luer-compatible pressure transducers, initially 0 to 300 mmHg or 0 to 40 kPa range, placed at the adapter tap markers.",
            "- Flow sensor: non-invasive ultrasonic clamp-on if available, otherwise inline turbine/Coriolis compatible with glycerol-water mixtures.",
            "- First fluid recipe: 40 percent glycerol / 60 percent water by volume as a starting blood-mimic viscosity candidate; verify viscosity with temperature because glycerol mixtures are temperature sensitive.",
            "- CT baseline fluid: water or saline for low-HU flow checks.",
            "- CT contrast option: diluted iodinated contrast in circulating fluid; tune concentration to roughly 200 to 450 HU at the intended CT kVp.",
            "- Leak-test pressure: start with low-flow visual leak checks, then static pressure hold at 1.5x expected operating pressure or 40 kPa, whichever is lower for the first printed prototype.",
            "- Manufacturing note: this is a CAD-reference assembly. Final CAD should boolean-union adapters, subtract tube bores, add proper barbs or threaded/luer ports, and define gasket/O-ring interfaces.",
            "",
            "## CAD Assembly Intent",
            "",
            "- The exported reference assembly contains the smoothed lumen, port-plane discs, adapter cylinders, flange collars, barb collars, and pressure-tap markers.",
            "- Import into FreeCAD or Blender as reference geometry for boolean joining and detailed mechanical design.",
            "- Do not treat the current assembly as directly printable/watertight; QA intentionally remains `review` until bores and unions are modeled.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _write_flow_loop_spec(path: Path, result: FlowLoopDesignResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "coordinate_units": "mm",
        "adapters": [
            {
                "id": adapter.id,
                "center_mm": list(adapter.center_mm),
                "axis": list(adapter.axis),
                "tube_inner_diameter_mm": adapter.tube_inner_diameter_mm,
                "lumen_transition_diameter_mm": adapter.lumen_transition_diameter_mm,
                "sleeve_outer_diameter_mm": adapter.sleeve_outer_diameter_mm,
                "sleeve_length_mm": adapter.sleeve_length_mm,
                "barb_count": adapter.barb_count,
                "barb_outer_diameter_mm": adapter.barb_outer_diameter_mm,
                "barb_width_mm": adapter.barb_width_mm,
                "barb_spacing_mm": adapter.barb_spacing_mm,
                "flange_outer_diameter_mm": adapter.flange_outer_diameter_mm,
                "flange_thickness_mm": adapter.flange_thickness_mm,
                "port_plane_radius_mm": adapter.port_plane_radius_mm,
                "pressure_tap_center_mm": list(adapter.pressure_tap_center_mm),
                "pressure_tap_diameter_mm": adapter.pressure_tap_diameter_mm,
                "outputs": list(adapter.outputs),
            }
            for adapter in result.adapters
        ],
        "bench_test": {
            "recommended_fluid_path": [
                "reservoir",
                "bubble_trap",
                "pump",
                "flow_sensor",
                "z_min_port_inlet",
                "true_lumen_module",
                "z_max_port_outlet",
                "downstream_pressure_sensor",
                "reservoir",
            ],
            "initial_pump_direction": "z_min_port_to_z_max_port",
            "flow_setpoints_l_min": [0.5, 1.0, 2.0, 3.0, 5.0],
            "first_test_fluid": "40_percent_glycerol_60_percent_water_by_volume_verify_viscosity_and_temperature",
            "ct_baseline_fluid": "water_or_saline",
            "contrast_fluid_option": "diluted_iodinated_contrast",
            "target_contrast_hu_range": [200, 450],
            "pressure_sensors": {
                "count": 2,
                "initial_range": "0_to_300_mmHg_or_0_to_40_kPa",
                "locations": "adapter_tap_markers_convert_to_threaded_or_luer_in_final_CAD",
            },
            "flow_sensor": "ultrasonic_clamp_on_preferred_or_inline_flow_sensor_compatible_with_glycerol_water",
            "leak_test": {
                "static_hold_pressure_kpa": 40,
                "or_multiplier": "1.5x_expected_operating_pressure_if_lower",
                "hold_time_minutes": 10,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def design_vascular_flow_loop(
    ports_yaml_path: str | Path,
    lumen_mesh_path: str | Path,
    output_dir: str | Path,
    formats: tuple[str, ...] = ("stl", "ply"),
    wall_thickness_mm: float = 2.0,
    sleeve_length_mm: float = 24.0,
    barb_count: int = 2,
    barb_height_mm: float = 1.0,
    barb_width_mm: float = 2.0,
    barb_spacing_mm: float = 5.0,
    flange_thickness_mm: float = 4.0,
    flange_extra_radius_mm: float = 5.0,
    pressure_tap_diameter_mm: float = 3.0,
    report_path: str | Path | None = None,
) -> FlowLoopDesignResult:
    *_, trimesh, _ = _import_dependencies()
    case_id, ports = _read_ports_yaml(ports_yaml_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    adapters: list[PortAdapter] = []
    adapter_meshes: list[object] = []
    for port in ports:
        adapter, adapter_mesh = _create_adapter_for_port(
            case_id=case_id,
            port=port,
            output_dir=output,
            formats=formats,
            wall_thickness_mm=wall_thickness_mm,
            sleeve_length_mm=sleeve_length_mm,
            barb_count=barb_count,
            barb_height_mm=barb_height_mm,
            barb_width_mm=barb_width_mm,
            barb_spacing_mm=barb_spacing_mm,
            flange_thickness_mm=flange_thickness_mm,
            flange_extra_radius_mm=flange_extra_radius_mm,
            pressure_tap_diameter_mm=pressure_tap_diameter_mm,
        )
        adapters.append(adapter)
        adapter_meshes.append(adapter_mesh)

    lumen = trimesh.load_mesh(lumen_mesh_path, process=True)

    port_plane_meshes = []
    port_plane_paths: list[str] = []
    for adapter, port in zip(adapters, ports):
        plane_mesh = _disk_mesh(port.center_mm, port.outward_normal, adapter.port_plane_radius_mm)
        port_plane_meshes.append(plane_mesh)
        for fmt in formats:
            normalized_fmt = fmt.lower().lstrip(".")
            port_plane_path = output / f"{case_id}_{port.id}_port_plane_reference_v001.{normalized_fmt}"
            plane_mesh.export(port_plane_path)
            port_plane_paths.append(str(port_plane_path))

    assembly = trimesh.util.concatenate([lumen, *port_plane_meshes, *adapter_meshes])
    assembly.merge_vertices()
    assembly.remove_unreferenced_vertices()
    _fix_normals(assembly)

    assembly_paths: list[str] = []
    for fmt in formats:
        normalized_fmt = fmt.lower().lstrip(".")
        assembly_path = output / f"{case_id}_flow_loop_reference_assembly_v001.{normalized_fmt}"
        assembly.export(assembly_path)
        assembly_paths.append(str(assembly_path))

    preview = output / f"{case_id}_flow_loop_preview_v001.png"
    spec_yaml = output / f"{case_id}_flow_loop_spec_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_flow_loop_design_report_v001.md"

    result = FlowLoopDesignResult(
        case_id=case_id,
        adapters=tuple(adapters),
        adapter_paths=tuple(path for adapter in adapters for path in adapter.outputs),
        port_plane_paths=tuple(port_plane_paths),
        assembly_paths=tuple(assembly_paths),
        preview_png_path=str(preview),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        notes=(
            "Adapter geometry is intentionally non-boolean reference geometry for CAD alignment.",
            "Final CAD should boolean-union sleeves/flanges, subtract lumen/tube bores, and add printable thread/barb details.",
        ),
    )

    _render_flow_loop_preview(preview, lumen, tuple(adapter_meshes), tuple(port_plane_meshes), result.adapters)
    _write_flow_loop_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_flow_loop_report(result))
    return result


def format_flow_loop_result(result: FlowLoopDesignResult) -> str:
    return _format_flow_loop_report(result)
