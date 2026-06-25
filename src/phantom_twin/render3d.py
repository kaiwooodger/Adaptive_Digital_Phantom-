from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .mesh_clean import _decimate, _fix_normals


def _import_dependencies():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
        import nibabel as nib  # type: ignore
        from skimage import measure  # type: ignore
        import trimesh  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "3D phantom rendering requires nibabel, scikit-image, matplotlib, trimesh, and PyYAML."
        ) from exc
    return plt, Patch, Poly3DCollection, nib, measure, trimesh, yaml


@dataclass(frozen=True)
class RenderGroup:
    id: str
    label: str
    label_ids: tuple[int, ...]
    color: str
    alpha: float
    mask_rule: str


@dataclass(frozen=True)
class RenderMeshResult:
    group_id: str
    label: str
    label_ids: tuple[int, ...]
    output_paths: tuple[str, ...]
    voxel_count: int
    volume_cm3: float
    vertices: int
    faces: int
    connected_components: int
    watertight: bool
    decimated: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Render3DResult:
    case_id: str
    output_dir: str
    preview_png_path: str
    report_path: str
    spec_yaml_path: str
    spacing_mm: tuple[float, float, float]
    groups: tuple[RenderGroup, ...]
    meshes: tuple[RenderMeshResult, ...]
    notes: tuple[str, ...]
    vessel_visible_preview_png_path: str | None = None
    vessel_visible_report_path: str | None = None


@dataclass(frozen=True)
class AtlasView:
    id: str
    label: str
    elevation: float
    azimuth: float
    zoom_group_ids: tuple[str, ...] = ()
    zoom_scale: float = 1.15


@dataclass(frozen=True)
class RenderAtlasResult:
    case_id: str
    output_dir: str
    atlas_png_path: str
    view_paths: tuple[str, ...]
    report_path: str
    spec_yaml_path: str
    views: tuple[AtlasView, ...]
    notes: tuple[str, ...]


def _default_groups() -> tuple[RenderGroup, ...]:
    return (
        RenderGroup(
            id="body_envelope",
            label="Body envelope",
            label_ids=tuple(range(1, 32)),
            color="#f2c078",
            alpha=0.10,
            mask_rule="labels > 0",
        ),
        RenderGroup(
            id="bone",
            label="Trabecular + cortical bone",
            label_ids=(10, 11),
            color="#f8f9fa",
            alpha=0.42,
            mask_rule="labels in {10, 11}",
        ),
        RenderGroup(
            id="lungs",
            label="Lungs",
            label_ids=(8,),
            color="#48cae4",
            alpha=0.38,
            mask_rule="labels == 8",
        ),
        RenderGroup(
            id="liver",
            label="Liver",
            label_ids=(6,),
            color="#9d4edd",
            alpha=0.58,
            mask_rule="labels == 6",
        ),
        RenderGroup(
            id="kidneys",
            label="Kidneys",
            label_ids=(7,),
            color="#f72585",
            alpha=0.70,
            mask_rule="labels == 7",
        ),
        RenderGroup(
            id="vessel_wall",
            label="Vessel wall",
            label_ids=(13,),
            color="#ff9f1c",
            alpha=0.82,
            mask_rule="labels == 13",
        ),
        RenderGroup(
            id="vascular_fluid",
            label="Vascular fluid",
            label_ids=(14, 15),
            color="#0077b6",
            alpha=0.92,
            mask_rule="labels in {14, 15}",
        ),
        RenderGroup(id="spleen", label="Spleen", label_ids=(16,), color="#7b2cbf", alpha=0.58, mask_rule="labels == 16"),
        RenderGroup(id="stomach_bowel_wall", label="Stomach/bowel wall", label_ids=(17,), color="#ffb703", alpha=0.50, mask_rule="labels == 17"),
        RenderGroup(id="gallbladder", label="Gallbladder / bile", label_ids=(18,), color="#84a59d", alpha=0.70, mask_rule="labels == 18"),
        RenderGroup(id="esophagus", label="Esophagus wall", label_ids=(19,), color="#f77f00", alpha=0.62, mask_rule="labels == 19"),
        RenderGroup(id="pancreas", label="Pancreas", label_ids=(20,), color="#f28482", alpha=0.62, mask_rule="labels == 20"),
        RenderGroup(id="adrenal_glands", label="Adrenal glands", label_ids=(21,), color="#b08968", alpha=0.72, mask_rule="labels == 21"),
        RenderGroup(id="gi_lumen", label="GI gas/fluid lumen placeholder", label_ids=(22, 23), color="#80ffdb", alpha=0.74, mask_rule="labels in {22, 23}"),
        RenderGroup(id="duodenum", label="Duodenum wall placeholder", label_ids=(24,), color="#f4a261", alpha=0.58, mask_rule="labels == 24"),
        RenderGroup(id="small_bowel", label="Small bowel wall placeholder", label_ids=(25,), color="#e9c46a", alpha=0.50, mask_rule="labels == 25"),
        RenderGroup(id="colon", label="Colon wall placeholder", label_ids=(26,), color="#2a9d8f", alpha=0.50, mask_rule="labels == 26"),
        RenderGroup(id="rectum", label="Rectum wall placeholder", label_ids=(27,), color="#8d6a9f", alpha=0.54, mask_rule="labels == 27"),
        RenderGroup(
            id="specific_gi_lumen",
            label="Specific bowel/colon/rectum lumen placeholders",
            label_ids=(28, 29, 30, 31),
            color="#4cc9f0",
            alpha=0.76,
            mask_rule="labels in {28, 29, 30, 31}",
        ),
    )


def _vascular_network_groups() -> tuple[RenderGroup, ...]:
    return (
        RenderGroup(
            id="body_envelope",
            label="Body envelope",
            label_ids=tuple(range(1, 32)),
            color="#f2c078",
            alpha=0.08,
            mask_rule="context labels > 0",
        ),
        RenderGroup(
            id="bone",
            label="Trabecular + cortical bone",
            label_ids=(10, 11),
            color="#f8f9fa",
            alpha=0.36,
            mask_rule="context labels in {10, 11}",
        ),
        RenderGroup(
            id="lungs",
            label="Lungs",
            label_ids=(8,),
            color="#48cae4",
            alpha=0.26,
            mask_rule="context labels == 8",
        ),
        RenderGroup(
            id="liver",
            label="Liver",
            label_ids=(6,),
            color="#9d4edd",
            alpha=0.44,
            mask_rule="context labels == 6",
        ),
        RenderGroup(
            id="kidneys",
            label="Kidneys",
            label_ids=(7,),
            color="#f72585",
            alpha=0.56,
            mask_rule="context labels == 7",
        ),
        RenderGroup(id="spleen", label="Spleen", label_ids=(16,), color="#7b2cbf", alpha=0.44, mask_rule="context labels == 16"),
        RenderGroup(id="stomach_bowel_wall", label="Stomach/bowel wall", label_ids=(17,), color="#ffb703", alpha=0.40, mask_rule="context labels == 17"),
        RenderGroup(id="gallbladder", label="Gallbladder / bile", label_ids=(18,), color="#84a59d", alpha=0.58, mask_rule="context labels == 18"),
        RenderGroup(id="esophagus", label="Esophagus wall", label_ids=(19,), color="#f77f00", alpha=0.50, mask_rule="context labels == 19"),
        RenderGroup(id="pancreas", label="Pancreas", label_ids=(20,), color="#f28482", alpha=0.50, mask_rule="context labels == 20"),
        RenderGroup(id="adrenal_glands", label="Adrenal glands", label_ids=(21,), color="#b08968", alpha=0.60, mask_rule="context labels == 21"),
        RenderGroup(id="gi_lumen", label="GI gas/fluid lumen placeholder", label_ids=(22, 23), color="#80ffdb", alpha=0.68, mask_rule="context labels in {22, 23}"),
        RenderGroup(id="duodenum", label="Duodenum wall placeholder", label_ids=(24,), color="#f4a261", alpha=0.46, mask_rule="context labels == 24"),
        RenderGroup(id="small_bowel", label="Small bowel wall placeholder", label_ids=(25,), color="#e9c46a", alpha=0.42, mask_rule="context labels == 25"),
        RenderGroup(id="colon", label="Colon wall placeholder", label_ids=(26,), color="#2a9d8f", alpha=0.42, mask_rule="context labels == 26"),
        RenderGroup(id="rectum", label="Rectum wall placeholder", label_ids=(27,), color="#8d6a9f", alpha=0.46, mask_rule="context labels == 27"),
        RenderGroup(
            id="specific_gi_lumen",
            label="Specific bowel/colon/rectum lumen placeholders",
            label_ids=(28, 29, 30, 31),
            color="#4cc9f0",
            alpha=0.66,
            mask_rule="context labels in {28, 29, 30, 31}",
        ),
        RenderGroup(
            id="flow_domains",
            label="Cleaned arterial + venous flow domains",
            label_ids=(1, 2),
            color="#2f4858",
            alpha=0.16,
            mask_rule="flow-domain labels in {1, 2}",
        ),
        RenderGroup(
            id="network_vessel_wall",
            label="Cleaned network vessel wall",
            label_ids=(5,),
            color="#ff9f1c",
            alpha=0.62,
            mask_rule="network vessel-wall mask > 0",
        ),
        RenderGroup(
            id="arterial_lumen",
            label="Cleaned arterial lumen",
            label_ids=(1,),
            color="#dc3b2a",
            alpha=0.94,
            mask_rule="arterial lumen mask > 0",
        ),
        RenderGroup(
            id="venous_lumen",
            label="Cleaned venous return lumen",
            label_ids=(2,),
            color="#2878b8",
            alpha=0.88,
            mask_rule="venous lumen mask > 0",
        ),
    )


def _render_draw_order(groups: tuple[RenderGroup, ...]) -> list[str]:
    preferred = [
        "body_envelope",
        "lungs",
        "bone",
        "liver",
        "kidneys",
        "spleen",
        "stomach_bowel_wall",
        "gallbladder",
        "esophagus",
        "pancreas",
        "adrenal_glands",
        "gi_lumen",
        "duodenum",
        "small_bowel",
        "colon",
        "rectum",
        "specific_gi_lumen",
        "vessel_wall",
        "vascular_fluid",
        "flow_domains",
        "network_vessel_wall",
        "arterial_lumen",
        "venous_lumen",
    ]
    group_ids = [group.id for group in groups]
    ordered = [group_id for group_id in preferred if group_id in group_ids]
    ordered.extend(group_id for group_id in group_ids if group_id not in ordered)
    return ordered


def _line_length(points: np.ndarray | list[list[float]]) -> float:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or len(array) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array[:, :3], axis=0), axis=1).sum())


def _mesh_from_mask(mask: np.ndarray, spacing: tuple[float, float, float]):
    *_, measure, trimesh, _ = _import_dependencies()
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    vertices, faces, normals, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=spacing,
    )
    vertices -= np.array(spacing, dtype=float)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    _fix_normals(mesh)
    return mesh


def _load_mesh(path: str | Path):
    *_, trimesh, _ = _import_dependencies()
    mesh = trimesh.load_mesh(str(path), process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _component_count(mesh) -> int:
    return int(len(mesh.split(only_watertight=False)))


def _export_mesh(mesh, output_base: Path, formats: tuple[str, ...]) -> tuple[str, ...]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        normalized = fmt.lower().lstrip(".")
        path = output_base.with_suffix(f".{normalized}")
        mesh.export(path)
        paths.append(str(path))
    return tuple(paths)


def _mesh_path_for_result(result: RenderMeshResult) -> str:
    return next(
        (path for path in result.output_paths if Path(path).suffix.lower() == ".stl"),
        result.output_paths[0],
    )


def _draw_scene(
    ax,
    render_items: tuple[dict[str, object], ...],
    bounds: np.ndarray,
    view: AtlasView,
    show_legend: bool,
) -> None:
    plt, Patch, Poly3DCollection, *_ = _import_dependencies()
    for item in render_items:
        group_id = str(item["group_id"])
        edge_color = "#111111" if group_id == "bone" else "none"
        line_width = 0.035 if group_id == "bone" else 0.0
        ax.add_collection3d(
            Poly3DCollection(
                item["triangles"],
                facecolors=str(item["color"]),
                edgecolors=edge_color,
                linewidths=line_width,
                alpha=float(item["alpha"]),
                rasterized=True,
            )
        )

    mins = bounds[0]
    maxs = bounds[1]
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0) * view.zoom_scale
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view.elevation, azim=view.azimuth)
    ax.set_title(view.label, pad=12, color="#13202a")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.set_facecolor("#f6f1e8")
    if show_legend:
        ax.legend(
            handles=[
                Patch(
                    facecolor=str(item["color"]),
                    alpha=max(float(item["alpha"]), 0.32),
                    label=str(item["label"]),
                )
                for item in render_items
            ],
            loc="upper left",
            fontsize=7,
        )


def _load_render_items(
    mesh_results: tuple[RenderMeshResult, ...],
    groups: tuple[RenderGroup, ...],
    seed: int,
) -> tuple[tuple[dict[str, object], ...], np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    group_by_id = {group.id: group for group in groups}
    result_by_id = {result.group_id: result for result in mesh_results}
    draw_order = _render_draw_order(groups)
    items: list[dict[str, object]] = []
    bounds_by_id: dict[str, np.ndarray] = {}
    all_bounds: list[np.ndarray] = []
    for group_id in draw_order:
        result = result_by_id.get(group_id)
        if result is None or not result.output_paths:
            continue
        group = group_by_id[group_id]
        mesh = _load_mesh(_mesh_path_for_result(result))
        triangles = mesh.triangles
        max_triangles = 80_000 if group_id == "body_envelope" else 60_000
        if len(triangles) > max_triangles:
            triangles = triangles[rng.choice(len(triangles), size=max_triangles, replace=False)]
        items.append(
            {
                "group_id": group_id,
                "label": group.label,
                "color": group.color,
                "alpha": group.alpha,
                "triangles": triangles,
            }
        )
        bounds_by_id[group_id] = mesh.bounds
        all_bounds.append(mesh.bounds)

    if not all_bounds:
        raise ValueError("No meshes were available to render")

    scene_bounds = np.array(
        [
            np.vstack([bounds[0] for bounds in all_bounds]).min(axis=0),
            np.vstack([bounds[1] for bounds in all_bounds]).max(axis=0),
        ]
    )
    return tuple(items), scene_bounds, bounds_by_id


def _make_group_mask(labels: np.ndarray, group: RenderGroup) -> np.ndarray:
    if group.id == "body_envelope":
        return labels > 0
    return np.isin(labels, group.label_ids)


def _render_preview(
    path: Path,
    mesh_results: tuple[RenderMeshResult, ...],
    groups: tuple[RenderGroup, ...],
) -> None:
    plt, Patch, Poly3DCollection, *_ = _import_dependencies()
    rng = np.random.default_rng(31)
    group_by_id = {group.id: group for group in groups}
    fig = plt.figure(figsize=(12, 9), dpi=180)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")

    bounds: list[np.ndarray] = []
    draw_order = _render_draw_order(groups)
    result_by_id = {result.group_id: result for result in mesh_results}
    for group_id in draw_order:
        result = result_by_id.get(group_id)
        if result is None or not result.output_paths:
            continue
        group = group_by_id[group_id]
        mesh_path = next((path for path in result.output_paths if Path(path).suffix.lower() == ".stl"), result.output_paths[0])
        mesh = _load_mesh(mesh_path)
        triangles = mesh.triangles
        max_triangles = 80_000 if group_id == "body_envelope" else 60_000
        if len(triangles) > max_triangles:
            triangles = triangles[rng.choice(len(triangles), size=max_triangles, replace=False)]
        edge_color = "#111111" if group_id == "bone" else "none"
        line_width = 0.035 if group_id == "bone" else 0.0
        ax.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolors=group.color,
                edgecolors=edge_color,
                linewidths=line_width,
                alpha=group.alpha,
                rasterized=True,
            )
        )
        bounds.append(mesh.bounds)

    if not bounds:
        raise ValueError("No meshes were available to render")

    mins = np.vstack([item[0] for item in bounds]).min(axis=0)
    maxs = np.vstack([item[1] for item in bounds]).max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0) * 1.08
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_title("Digital Phantom: 3D Renderable Mesh Scene", pad=18, color="#13202a")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.legend(
        handles=[
            Patch(facecolor=group.color, alpha=max(group.alpha, 0.32), label=group.label)
            for group in groups
            if group.id in result_by_id
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_spec(path: Path, result: Render3DResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "coordinate_units": "mm",
        "spacing_mm": list(result.spacing_mm),
        "preview_png": result.preview_png_path,
        "vessel_visible_preview_png": result.vessel_visible_preview_png_path,
        "vessel_visible_report": result.vessel_visible_report_path,
        "groups": [
            {
                "id": group.id,
                "label": group.label,
                "label_ids": list(group.label_ids),
                "color": group.color,
                "alpha": group.alpha,
                "mask_rule": group.mask_rule,
            }
            for group in result.groups
        ],
        "meshes": [
            {
                "group_id": mesh.group_id,
                "label": mesh.label,
                "label_ids": list(mesh.label_ids),
                "outputs": list(mesh.output_paths),
                "voxel_count": mesh.voxel_count,
                "volume_cm3": mesh.volume_cm3,
                "vertices": mesh.vertices,
                "faces": mesh.faces,
                "connected_components": mesh.connected_components,
                "watertight": mesh.watertight,
                "decimated": mesh.decimated,
                "notes": list(mesh.notes),
            }
            for mesh in result.meshes
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: Render3DResult) -> str:
    lines = [
        "# Combined Digital Phantom 3D Mesh Render Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Outputs",
        "",
        f"- Preview PNG: `{Path(result.preview_png_path).name}`",
        *(
            [f"- Vessel-visible preview PNG: `{Path(result.vessel_visible_preview_png_path).name}`"]
            if result.vessel_visible_preview_png_path
            else []
        ),
        f"- Machine-readable scene spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Mesh Groups",
        "",
        "| group | label IDs | voxels | volume cm3 | faces | components | watertight | outputs |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for mesh in result.meshes:
        outputs = "<br>".join(Path(path).name for path in mesh.output_paths)
        label_ids = ", ".join(str(value) for value in mesh.label_ids)
        lines.append(
            f"| {mesh.label} | {label_ids} | {mesh.voxel_count} | {mesh.volume_cm3:.2f} | "
            f"{mesh.faces} | {mesh.connected_components} | {mesh.watertight} | {outputs} |"
        )

    lines.extend(
        [
            "",
            "## Render Style",
            "",
            "- Body envelope is intentionally transparent so internal organs and vascular structures remain visible.",
            "- Bone, lungs, liver, kidneys, preserved abdominal organs, GI lumen placeholders, and specific bowel/colon/rectum placeholder organs are exported as separate renderable surfaces when present.",
            "- Vessel wall and vascular fluid are highlighted as the primary digital flow module.",
            "- Vessel-visible mode, when written, uses display-enlarged centerlines and lumen voxels so small internal aorta/IVC structures remain visible.",
            "- Meshes are generated from the combined material-label NIfTI using marching cubes in physical millimeter spacing.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def generate_combined_3d_render(
    combined_labels_path: str | Path,
    output_dir: str | Path,
    case_id: str = "ct_org_case0_imagetbad_case125",
    formats: tuple[str, ...] = ("stl", "ply", "obj"),
    target_max_faces: int = 140_000,
    report_path: str | Path | None = None,
) -> Render3DResult:
    _, _, _, nib, *_ = _import_dependencies()
    image = nib.load(str(combined_labels_path))
    labels = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    voxel_volume_cm3 = float(np.prod(spacing) / 1000.0)
    output = Path(output_dir)
    mesh_dir = output / "meshes"
    output.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    groups = _default_groups()
    results: list[RenderMeshResult] = []
    notes: list[str] = []
    for group in groups:
        mask = _make_group_mask(labels, group)
        voxel_count = int(mask.sum())
        if voxel_count == 0:
            notes.append(f"skipped_empty_group={group.id}")
            continue

        mesh = _mesh_from_mask(mask, spacing)
        decimated = False
        decimation_notes: list[str] = []
        if target_max_faces > 0 and len(mesh.faces) > target_max_faces:
            decimated_mesh, did_decimate, decimation_note = _decimate(mesh, target_max_faces)
            if decimation_note:
                decimation_notes.append(decimation_note)
            elif did_decimate:
                mesh = decimated_mesh
                decimated = True
                decimation_notes.append(f"decimated_to_target_max_faces={target_max_faces}")
        _fix_normals(mesh)
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()

        output_base = mesh_dir / f"{case_id}_{group.id}_v001"
        paths = _export_mesh(mesh, output_base, formats)
        results.append(
            RenderMeshResult(
                group_id=group.id,
                label=group.label,
                label_ids=group.label_ids,
                output_paths=paths,
                voxel_count=voxel_count,
                volume_cm3=voxel_count * voxel_volume_cm3,
                vertices=int(len(mesh.vertices)),
                faces=int(len(mesh.faces)),
                connected_components=_component_count(mesh),
                watertight=bool(mesh.is_watertight),
                decimated=decimated,
                notes=tuple(decimation_notes),
            )
        )

    preview_png = output / f"{case_id}_3d_render_preview_v001.png"
    spec_yaml = output / f"{case_id}_3d_render_scene_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_3d_render_report_v001.md"
    result = Render3DResult(
        case_id=case_id,
        output_dir=str(output),
        preview_png_path=str(preview_png),
        report_path=str(report),
        spec_yaml_path=str(spec_yaml),
        spacing_mm=spacing,
        groups=groups,
        meshes=tuple(results),
        notes=tuple(notes),
    )

    _render_preview(preview_png, result.meshes, result.groups)
    _write_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def _load_mask_nifti(path: str | Path, reference_shape: tuple[int, int, int], reference_spacing: tuple[float, float, float]) -> np.ndarray:
    _, _, _, nib, *_ = _import_dependencies()
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    shape = tuple(int(value) for value in data.shape)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if shape != reference_shape:
        raise ValueError(f"Mask shape differs from context labels: {shape} vs {reference_shape} for {path}")
    if any(abs(spacing[index] - reference_spacing[index]) > 1e-6 for index in range(3)):
        raise ValueError(f"Mask spacing differs from context labels: {spacing} vs {reference_spacing} for {path}")
    return data


def _points_mm_from_indices(indices: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0, 3), dtype=float)
    return indices.astype(float) * np.asarray(spacing, dtype=float)[None, :]


def _sample_label_points(
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    label_ids: tuple[int, ...],
    *,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.argwhere(np.isin(labels, np.asarray(label_ids, dtype=labels.dtype)))
    if len(indices) > max_points:
        indices = indices[rng.choice(len(indices), size=max_points, replace=False)]
    return _points_mm_from_indices(indices, spacing)


def _sample_binary_points(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    max_points: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.argwhere(mask)
    if max_points is not None and len(indices) > max_points:
        indices = indices[rng.choice(len(indices), size=max_points, replace=False)]
    return _points_mm_from_indices(indices, spacing)


def _edge_points(edge: dict[str, object]) -> np.ndarray:
    for key in ("polyline_mm", "points_mm", "centerline_mm"):
        points = edge.get(key)
        if isinstance(points, list) and len(points) >= 2:
            array = np.asarray(points, dtype=float)
            if array.ndim == 2 and array.shape[1] >= 3:
                return array[:, :3]
    return np.empty((0, 3), dtype=float)


def _edge_mean_radius(edge: dict[str, object]) -> float:
    values: list[float] = []
    profile = edge.get("radius_profile")
    if isinstance(profile, list):
        for item in profile:
            if isinstance(item, dict) and item.get("radius_mm") is not None:
                values.append(float(item["radius_mm"]))
    profile_mm = edge.get("radius_profile_mm")
    if isinstance(profile_mm, list):
        values.extend(float(value) for value in profile_mm if value is not None)
    for key in ("radius_start_mm", "radius_end_mm", "radius_mm"):
        value = edge.get(key)
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else 1.0


def _edge_visibility_class(edge: dict[str, object]) -> str:
    text = " ".join(str(edge.get(key, "")) for key in ("id", "label", "flow_role", "vessel_type")).lower()
    if "aorta" in text or "arterial_inlet" in text:
        return "aorta"
    if "ivc" in text or "venous_return" in text:
        return "ivc"
    if "arterial" in text or "artery" in text:
        return "arterial"
    if "venous" in text or "vein" in text:
        return "venous"
    return "other"


def _load_graph_edges(path: str | Path | None) -> tuple[dict[str, object], ...]:
    if path is None or str(path) == "" or not Path(path).exists():
        return ()
    *_, yaml = _import_dependencies()
    graph = yaml.safe_load(Path(path).read_text())
    if not isinstance(graph, dict):
        return ()
    return tuple(edge for edge in graph.get("edges", []) if isinstance(edge, dict))


def _write_vessel_visible_report(
    path: Path,
    *,
    case_id: str,
    preview_png: Path,
    vascular_graph_path: str | Path | None,
    arterial_voxels: int,
    arterial_volume_cm3: float,
    venous_voxels: int,
    venous_volume_cm3: float,
    highlighted_edges: tuple[tuple[str, str, float, float], ...],
    notes: tuple[str, ...],
) -> None:
    lines = [
        "# Vessel-Visible 3D Render Mode",
        "",
        f"Case ID: `{case_id}`",
        "",
        "## Outputs",
        "",
        f"- Vessel-visible PNG: `{preview_png}`",
        f"- Vascular graph: `{vascular_graph_path or 'not_supplied'}`",
        "",
        "## Vascular Volumes",
        "",
        f"- Arterial lumen: {arterial_voxels} voxels, {arterial_volume_cm3:.6f} cm3",
        f"- Venous lumen: {venous_voxels} voxels, {venous_volume_cm3:.6f} cm3",
        "",
        "## Highlighted Graph Edges",
        "",
        "| edge | class | centerline length mm | mean radius mm |",
        "| --- | --- | ---: | ---: |",
    ]
    if highlighted_edges:
        for edge_id, edge_class, length_mm, radius_mm in highlighted_edges:
            lines.append(f"| `{edge_id}` | {edge_class} | {length_mm:.2f} | {radius_mm:.2f} |")
    else:
        lines.append("| none | n/a | 0.00 | 0.00 |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This mode is a visualization aid: graph centerlines are display-enlarged and labelled, while voxelized lumen points are shown at their actual locations.",
            "- It does not change the STL meshes, NIfTI masks, flow domain labels, or simulation geometry.",
            "- Use this PNG when the true-scale full-torso render hides small internal vessels behind organs or transparency ordering.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _render_vessel_visible_preview(
    *,
    path: Path,
    report_path: Path,
    case_id: str,
    context_labels: np.ndarray,
    arterial_mask: np.ndarray,
    venous_mask: np.ndarray,
    spacing: tuple[float, float, float],
    vascular_graph_path: str | Path | None,
    vessel_display_scale: float,
) -> tuple[str, str]:
    plt, *_ = _import_dependencies()
    rng = np.random.default_rng(72)
    voxel_volume_cm3 = float(np.prod(spacing) / 1000.0)
    arterial_voxels = int(np.count_nonzero(arterial_mask))
    venous_voxels = int(np.count_nonzero(venous_mask))
    arterial_volume_cm3 = arterial_voxels * voxel_volume_cm3
    venous_volume_cm3 = venous_voxels * voxel_volume_cm3

    liver_points = _sample_label_points(context_labels, spacing, (6,), max_points=12_000, rng=rng)
    kidney_points = _sample_label_points(context_labels, spacing, (7,), max_points=8_000, rng=rng)
    arterial_points = _sample_binary_points(arterial_mask, spacing, max_points=None, rng=rng)
    venous_points = _sample_binary_points(venous_mask, spacing, max_points=8_000, rng=rng)
    graph_edges = _load_graph_edges(vascular_graph_path)

    fig = plt.figure(figsize=(12, 10), dpi=220)
    fig.patch.set_facecolor("#f7f2e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f7f2e8")

    if len(liver_points):
        ax.scatter(liver_points[:, 0], liver_points[:, 1], liver_points[:, 2], s=1.0, c="#9d4edd", alpha=0.035, depthshade=False, label="liver context")
    if len(kidney_points):
        ax.scatter(kidney_points[:, 0], kidney_points[:, 1], kidney_points[:, 2], s=1.8, c="#f72585", alpha=0.08, depthshade=False, label="kidney context")
    if len(venous_points):
        ax.scatter(venous_points[:, 0], venous_points[:, 1], venous_points[:, 2], s=13, c="#2878b8", alpha=0.58, depthshade=True, label="actual venous lumen voxels")
    if len(arterial_points):
        ax.scatter(arterial_points[:, 0], arterial_points[:, 1], arterial_points[:, 2], s=42, c="#dc3b2a", alpha=0.95, depthshade=True, label="actual arterial lumen voxels")

    style_by_class = {
        "aorta": ("#d00000", "AORTA centerline", 10.0),
        "ivc": ("#0057b8", "IVC/venous return centerline", 8.5),
        "arterial": ("#f97316", "other arterial centerline", 4.5),
        "venous": ("#0891b2", "other venous centerline", 4.2),
        "other": ("#334155", "other vessel centerline", 3.0),
    }
    focus_points: list[np.ndarray] = []
    highlighted: list[tuple[str, str, float, float]] = []
    plotted_labels: set[str] = set()
    for edge in graph_edges:
        points = _edge_points(edge)
        if len(points) < 2:
            continue
        edge_class = _edge_visibility_class(edge)
        if edge_class not in {"aorta", "ivc", "arterial", "venous"}:
            continue
        color, label, base_width = style_by_class[edge_class]
        radius_mm = _edge_mean_radius(edge)
        width = max(base_width, min(14.0, radius_mm * max(vessel_display_scale, 0.1)))
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            linewidth=width,
            alpha=0.98 if edge_class in {"aorta", "ivc"} else 0.42,
            solid_capstyle="round",
            label=label if label not in plotted_labels else None,
        )
        plotted_labels.add(label)
        if edge_class in {"aorta", "ivc"}:
            ax.scatter(points[0:1, 0], points[0:1, 1], points[0:1, 2], s=130, c=color, marker="o", edgecolor="black", linewidth=0.8, depthshade=False)
            ax.scatter(points[-1:, 0], points[-1:, 1], points[-1:, 2], s=130, c=color, marker="s", edgecolor="black", linewidth=0.8, depthshade=False)
            mid = points[len(points) // 2]
            ax.text(mid[0], mid[1], mid[2] + 10, "AORTA" if edge_class == "aorta" else "IVC", color=color, fontsize=14, fontweight="bold")
            focus_points.append(points)
            highlighted.append((str(edge.get("id", "")), edge_class, _line_length(points), radius_mm))

    if not focus_points:
        if len(arterial_points):
            focus_points.append(arterial_points)
        if len(venous_points):
            focus_points.append(venous_points)
    if not focus_points:
        focus_points.extend(points for points in (liver_points, kidney_points) if len(points))
    if not focus_points:
        raise ValueError("No vessel or context points were available for vessel-visible rendering")

    cloud = np.vstack(focus_points)
    mins = cloud.min(axis=0)
    maxs = cloud.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = max(float((maxs - mins).max()), 1.0)
    pad = max(35.0, extent * 0.22)
    ranges = np.array([extent + 2 * pad] * 3)
    ranges[0] *= 1.12
    ranges[1] *= 1.12
    ax.set_xlim(center[0] - ranges[0] / 2, center[0] + ranges[0] / 2)
    ax.set_ylim(center[1] - ranges[1] / 2, center[1] + ranges[1] / 2)
    ax.set_zlim(center[2] - ranges[2] / 2, center[2] + ranges[2] / 2)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=20, azim=-58)
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.set_title(
        "Vessel-Visible 3D Render inside Digital Phantom\nDisplay-enlarged centerlines + actual lumen voxels; simulation geometry unchanged",
        fontsize=15,
        pad=20,
    )
    handles, labels = ax.get_legend_handles_labels()
    unique_handles: list[object] = []
    unique_labels: list[str] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels):
        if label not in seen:
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=True, framealpha=0.88, fontsize=9)

    annotation = f"Arterial lumen: {arterial_voxels} voxels, {arterial_volume_cm3:.4f} cm3; venous lumen: {venous_voxels} voxels, {venous_volume_cm3:.3f} cm3"
    fig.text(0.5, 0.035, annotation, ha="center", va="center", fontsize=9.5, color="#333333")
    plt.tight_layout(rect=(0, 0.055, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    notes = (
        "display_enlarged_centerlines_are_for_visibility_only",
        "actual_lumen_voxel_points_are_sampled_from_render_input_masks",
        "organ_context_is_downsampled_to_keep_the_preview_fast_and_readable",
    )
    _write_vessel_visible_report(
        report_path,
        case_id=case_id,
        preview_png=path,
        vascular_graph_path=vascular_graph_path,
        arterial_voxels=arterial_voxels,
        arterial_volume_cm3=arterial_volume_cm3,
        venous_voxels=venous_voxels,
        venous_volume_cm3=venous_volume_cm3,
        highlighted_edges=tuple(highlighted),
        notes=notes,
    )
    return str(path), str(report_path)


def generate_vascular_network_3d_render(
    context_labels_path: str | Path,
    arterial_lumen_mask_path: str | Path,
    venous_lumen_mask_path: str | Path,
    flow_domain_labels_path: str | Path,
    vessel_wall_mask_path: str | Path,
    output_dir: str | Path,
    case_id: str = "ct_org_case0_imagetbad_case125",
    formats: tuple[str, ...] = ("stl", "ply", "obj"),
    target_max_faces: int = 140_000,
    vascular_graph_path: str | Path | None = None,
    render_vessel_visible_preview: bool = True,
    vessel_display_scale: float = 4.0,
    report_path: str | Path | None = None,
) -> Render3DResult:
    _, _, _, nib, *_ = _import_dependencies()
    image = nib.load(str(context_labels_path))
    context_labels = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    voxel_volume_cm3 = float(np.prod(spacing) / 1000.0)
    output = Path(output_dir)
    mesh_dir = output / "meshes"
    output.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    reference_shape = tuple(int(value) for value in context_labels.shape)
    arterial_mask = _load_mask_nifti(arterial_lumen_mask_path, reference_shape, spacing) > 0
    venous_mask = _load_mask_nifti(venous_lumen_mask_path, reference_shape, spacing) > 0
    flow_domain_labels = np.rint(_load_mask_nifti(flow_domain_labels_path, reference_shape, spacing)).astype(np.int16)
    vessel_wall_mask = _load_mask_nifti(vessel_wall_mask_path, reference_shape, spacing) > 0

    groups = _vascular_network_groups()
    masks_by_group: dict[str, np.ndarray] = {
        "body_envelope": context_labels > 0,
        "bone": np.isin(context_labels, (10, 11)),
        "lungs": context_labels == 8,
        "liver": context_labels == 6,
        "kidneys": context_labels == 7,
        "spleen": context_labels == 16,
        "stomach_bowel_wall": context_labels == 17,
        "gallbladder": context_labels == 18,
        "esophagus": context_labels == 19,
        "pancreas": context_labels == 20,
        "adrenal_glands": context_labels == 21,
        "gi_lumen": np.isin(context_labels, (22, 23)),
        "duodenum": context_labels == 24,
        "small_bowel": context_labels == 25,
        "colon": context_labels == 26,
        "rectum": context_labels == 27,
        "specific_gi_lumen": np.isin(context_labels, (28, 29, 30, 31)),
        "flow_domains": np.isin(flow_domain_labels, (1, 2)),
        "network_vessel_wall": vessel_wall_mask,
        "arterial_lumen": arterial_mask,
        "venous_lumen": venous_mask,
    }

    overlap = int((arterial_mask & venous_mask).sum())
    notes: list[str] = [
        f"context_labels={context_labels_path}",
        f"arterial_lumen_mask={arterial_lumen_mask_path}",
        f"venous_lumen_mask={venous_lumen_mask_path}",
        f"flow_domain_labels={flow_domain_labels_path}",
        f"network_vessel_wall_mask={vessel_wall_mask_path}",
        f"arterial_venous_overlap_voxels={overlap}",
    ]
    if overlap == 0:
        notes.append("cleaned_arterial_and_venous_domains_are_non_overlapping")

    results: list[RenderMeshResult] = []
    for group in groups:
        mask = masks_by_group[group.id]
        voxel_count = int(mask.sum())
        if voxel_count == 0:
            notes.append(f"skipped_empty_group={group.id}")
            continue

        mesh = _mesh_from_mask(mask, spacing)
        decimated = False
        decimation_notes: list[str] = []
        if target_max_faces > 0 and len(mesh.faces) > target_max_faces:
            decimated_mesh, did_decimate, decimation_note = _decimate(mesh, target_max_faces)
            if decimation_note:
                decimation_notes.append(decimation_note)
            elif did_decimate:
                mesh = decimated_mesh
                decimated = True
                decimation_notes.append(f"decimated_to_target_max_faces={target_max_faces}")
        _fix_normals(mesh)
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()

        output_base = mesh_dir / f"{case_id}_{group.id}_v001"
        paths = _export_mesh(mesh, output_base, formats)
        results.append(
            RenderMeshResult(
                group_id=group.id,
                label=group.label,
                label_ids=group.label_ids,
                output_paths=paths,
                voxel_count=voxel_count,
                volume_cm3=voxel_count * voxel_volume_cm3,
                vertices=int(len(mesh.vertices)),
                faces=int(len(mesh.faces)),
                connected_components=_component_count(mesh),
                watertight=bool(mesh.is_watertight),
                decimated=decimated,
                notes=tuple(decimation_notes),
            )
        )

    preview_png = output / f"{case_id}_vascular_network_3d_render_preview_v001.png"
    vessel_visible_preview_png = output / f"{case_id}_vessel_visible_3d_render_preview_v001.png"
    vessel_visible_report = output / f"{case_id}_vessel_visible_3d_render_report_v001.md"
    spec_yaml = output / f"{case_id}_vascular_network_3d_render_scene_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_vascular_network_3d_render_report_v001.md"
    vessel_visible_preview_path: str | None = None
    vessel_visible_report_path: str | None = None
    if render_vessel_visible_preview:
        try:
            vessel_visible_preview_path, vessel_visible_report_path = _render_vessel_visible_preview(
                path=vessel_visible_preview_png,
                report_path=vessel_visible_report,
                case_id=case_id,
                context_labels=context_labels,
                arterial_mask=arterial_mask,
                venous_mask=venous_mask,
                spacing=spacing,
                vascular_graph_path=vascular_graph_path,
                vessel_display_scale=vessel_display_scale,
            )
            notes.append(f"vessel_visible_preview={vessel_visible_preview_path}")
        except Exception as exc:
            notes.append(f"vessel_visible_preview_failed={type(exc).__name__}: {exc}")
    result = Render3DResult(
        case_id=case_id,
        output_dir=str(output),
        preview_png_path=str(preview_png),
        report_path=str(report),
        spec_yaml_path=str(spec_yaml),
        spacing_mm=spacing,
        groups=groups,
        meshes=tuple(results),
        notes=tuple(notes),
        vessel_visible_preview_png_path=vessel_visible_preview_path,
        vessel_visible_report_path=vessel_visible_report_path,
    )

    _render_preview(preview_png, result.meshes, result.groups)
    _write_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_render3d_result(result: Render3DResult) -> str:
    return _format_report(result)


def _default_atlas_views() -> tuple[AtlasView, ...]:
    return (
        AtlasView(id="anterior", label="Anterior / Front View", elevation=0, azimuth=-90),
        AtlasView(id="posterior", label="Posterior / Back View", elevation=0, azimuth=90),
        AtlasView(id="left_lateral", label="Left Lateral View", elevation=0, azimuth=180),
        AtlasView(id="right_lateral", label="Right Lateral View", elevation=0, azimuth=0),
        AtlasView(id="superior", label="Superior / Top View", elevation=90, azimuth=-90),
        AtlasView(id="inferior", label="Inferior / Bottom View", elevation=-90, azimuth=-90),
        AtlasView(id="oblique", label="Oblique Transparent View", elevation=18, azimuth=-58),
        AtlasView(
            id="vascular_zoom",
            label="Vascular-Focused Zoom View",
            elevation=18,
            azimuth=-58,
            zoom_group_ids=(
                "vessel_wall",
                "vascular_fluid",
                "network_vessel_wall",
                "flow_domains",
                "arterial_lumen",
                "venous_lumen",
            ),
            zoom_scale=1.35,
        ),
    )


def _read_scene_spec(path: str | Path) -> tuple[str, tuple[RenderGroup, ...], tuple[RenderMeshResult, ...]]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    groups = tuple(
        RenderGroup(
            id=str(item["id"]),
            label=str(item["label"]),
            label_ids=tuple(int(value) for value in item["label_ids"]),
            color=str(item["color"]),
            alpha=float(item["alpha"]),
            mask_rule=str(item["mask_rule"]),
        )
        for item in data["groups"]
    )
    meshes = tuple(
        RenderMeshResult(
            group_id=str(item["group_id"]),
            label=str(item["label"]),
            label_ids=tuple(int(value) for value in item["label_ids"]),
            output_paths=tuple(str(value) for value in item["outputs"]),
            voxel_count=int(item["voxel_count"]),
            volume_cm3=float(item["volume_cm3"]),
            vertices=int(item["vertices"]),
            faces=int(item["faces"]),
            connected_components=int(item["connected_components"]),
            watertight=bool(item["watertight"]),
            decimated=bool(item["decimated"]),
            notes=tuple(str(value) for value in item.get("notes", [])),
        )
        for item in data["meshes"]
    )
    return str(data["case_id"]), groups, meshes


def _bounds_for_view(
    view: AtlasView,
    scene_bounds: np.ndarray,
    bounds_by_id: dict[str, np.ndarray],
) -> np.ndarray:
    if not view.zoom_group_ids:
        return scene_bounds

    bounds = [bounds_by_id[group_id] for group_id in view.zoom_group_ids if group_id in bounds_by_id]
    if not bounds:
        return scene_bounds

    return np.array(
        [
            np.vstack([item[0] for item in bounds]).min(axis=0),
            np.vstack([item[1] for item in bounds]).max(axis=0),
        ]
    )


def _render_atlas_view(
    path: Path,
    render_items: tuple[dict[str, object], ...],
    bounds: np.ndarray,
    view: AtlasView,
    show_legend: bool,
) -> None:
    plt, *_ = _import_dependencies()
    fig = plt.figure(figsize=(9, 8), dpi=170)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    _draw_scene(ax, render_items, bounds, view, show_legend=show_legend)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _render_contact_sheet(path: Path, view_paths: tuple[str, ...], views: tuple[AtlasView, ...]) -> None:
    plt, *_ = _import_dependencies()
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), dpi=170)
    fig.patch.set_facecolor("#f6f1e8")
    for ax, view_path, view in zip(axes.ravel(), view_paths, views):
        image = plt.imread(view_path)
        ax.imshow(image)
        ax.set_title(view.label, color="#13202a", fontsize=10)
        ax.axis("off")
    fig.suptitle("Combined Digital Phantom: Multi-View 3D Render Atlas", fontsize=16, color="#13202a")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_atlas_spec(path: Path, result: RenderAtlasResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "atlas_png": result.atlas_png_path,
        "views": [
            {
                "id": view.id,
                "label": view.label,
                "elevation": view.elevation,
                "azimuth": view.azimuth,
                "zoom_group_ids": list(view.zoom_group_ids),
                "zoom_scale": view.zoom_scale,
                "output_png": result.view_paths[index],
            }
            for index, view in enumerate(result.views)
        ],
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_atlas_report(result: RenderAtlasResult) -> str:
    lines = [
        "# Combined Digital Phantom Multi-View 3D Render Atlas",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Outputs",
        "",
        f"- Atlas contact sheet: `{Path(result.atlas_png_path).name}`",
        f"- Machine-readable atlas spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Views",
        "",
        "| view | elevation | azimuth | zoom groups | output |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for view, output_path in zip(result.views, result.view_paths):
        zoom_groups = ", ".join(view.zoom_group_ids) if view.zoom_group_ids else "full scene"
        lines.append(
            f"| {view.label} | {view.elevation:.1f} | {view.azimuth:.1f} | "
            f"{zoom_groups} | `{Path(output_path).name}` |"
        )

    lines.extend(
        [
            "",
            "## Render Notes",
            "",
            "- Each view uses the same transparent body/organ/vascular mesh scene generated from the combined digital phantom.",
            "- Bone is rendered with black mesh outlines for visibility.",
            "- The vascular-focused view zooms to the vessel wall and vascular fluid bounds.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def generate_3d_view_atlas(
    scene_spec_path: str | Path,
    output_dir: str | Path,
    case_id: str | None = None,
    report_path: str | Path | None = None,
) -> RenderAtlasResult:
    spec_case_id, groups, meshes = _read_scene_spec(scene_spec_path)
    resolved_case_id = case_id or spec_case_id
    output = Path(output_dir)
    views_dir = output / "views"
    output.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)

    render_items, scene_bounds, bounds_by_id = _load_render_items(meshes, groups, seed=41)
    views = _default_atlas_views()
    view_paths: list[str] = []
    for view in views:
        view_path = views_dir / f"{resolved_case_id}_{view.id}_view_v001.png"
        bounds = _bounds_for_view(view, scene_bounds, bounds_by_id)
        _render_atlas_view(
            view_path,
            render_items,
            bounds,
            view,
            show_legend=view.id in {"oblique", "vascular_zoom"},
        )
        view_paths.append(str(view_path))

    atlas_png = output / f"{resolved_case_id}_3d_render_atlas_v001.png"
    spec_yaml = output / f"{resolved_case_id}_3d_render_atlas_v001.yaml"
    report = Path(report_path) if report_path else output / f"{resolved_case_id}_3d_render_atlas_report_v001.md"
    result = RenderAtlasResult(
        case_id=resolved_case_id,
        output_dir=str(output),
        atlas_png_path=str(atlas_png),
        view_paths=tuple(view_paths),
        report_path=str(report),
        spec_yaml_path=str(spec_yaml),
        views=views,
        notes=("scene_spec=" + str(scene_spec_path),),
    )
    _render_contact_sheet(atlas_png, result.view_paths, result.views)
    _write_atlas_spec(spec_yaml, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_atlas_report(result))
    return result


def format_render_atlas_result(result: RenderAtlasResult) -> str:
    return _format_atlas_report(result)
