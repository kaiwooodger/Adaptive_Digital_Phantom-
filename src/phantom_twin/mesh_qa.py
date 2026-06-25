from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _import_trimesh():
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Mesh QA requires trimesh. Install with: python3 -m pip install trimesh"
        ) from exc
    return trimesh


@dataclass(frozen=True)
class MeshQaResult:
    path: str
    file_size_mb: float
    vertices: int
    faces: int
    connected_components: int
    watertight: bool
    winding_consistent: bool
    euler_number: int
    bounds_min_mm: tuple[float, float, float]
    bounds_max_mm: tuple[float, float, float]
    extents_mm: tuple[float, float, float]
    surface_area_mm2: float
    volume_mm3: float | None
    nondegenerate_face_fraction: float
    boundary_edge_count: int
    nonmanifold_edge_count: int
    warnings: tuple[str, ...]

    @property
    def status(self) -> str:
        if self.warnings:
            return "review"
        return "pass"


def _load_single_mesh(path: Path):
    trimesh = _import_trimesh()
    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if len(geometry.faces) > 0]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)

    loaded.merge_vertices()
    loaded.remove_unreferenced_vertices()
    return loaded


def _edge_counts(mesh) -> tuple[int, int]:
    if len(mesh.faces) == 0:
        return 0, 0

    edges = np.sort(mesh.edges_sorted, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts > 2))
    return boundary_edges, nonmanifold_edges


def analyze_mesh(path: str | Path) -> MeshQaResult:
    mesh_path = Path(path)
    mesh = _load_single_mesh(mesh_path)
    trimesh = _import_trimesh()

    components = mesh.split(only_watertight=False)
    boundary_edge_count, nonmanifold_edge_count = _edge_counts(mesh)
    nondegenerate = trimesh.triangles.nondegenerate(mesh.triangles)
    nondegenerate_fraction = float(np.mean(nondegenerate)) if len(nondegenerate) else 0.0

    extents = tuple(float(value) for value in mesh.extents)
    bounds_min = tuple(float(value) for value in mesh.bounds[0])
    bounds_max = tuple(float(value) for value in mesh.bounds[1])
    volume = float(mesh.volume) if mesh.is_watertight else None

    warnings: list[str] = []
    if not mesh.is_watertight:
        warnings.append("not_watertight")
    if boundary_edge_count:
        warnings.append(f"boundary_edges={boundary_edge_count}")
    if nonmanifold_edge_count:
        warnings.append(f"nonmanifold_edges={nonmanifold_edge_count}")
    if len(components) > 1:
        warnings.append(f"connected_components={len(components)}")
    if nondegenerate_fraction < 0.999:
        warnings.append(f"degenerate_faces_fraction={1 - nondegenerate_fraction:.4f}")
    if min(extents) <= 0:
        warnings.append("zero_extent")
    if max(extents) > 1000:
        warnings.append("extent_over_1000mm_check_units")
    if len(mesh.faces) > 500_000:
        warnings.append("high_face_count_decimate_for_cad")
    if volume is not None and volume < 0:
        warnings.append("negative_signed_volume_flip_normals")

    return MeshQaResult(
        path=str(mesh_path),
        file_size_mb=mesh_path.stat().st_size / (1024 * 1024),
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        connected_components=int(len(components)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        bounds_min_mm=bounds_min,
        bounds_max_mm=bounds_max,
        extents_mm=extents,
        surface_area_mm2=float(mesh.area),
        volume_mm3=abs(volume) if volume is not None else None,
        nondegenerate_face_fraction=nondegenerate_fraction,
        boundary_edge_count=boundary_edge_count,
        nonmanifold_edge_count=nonmanifold_edge_count,
        warnings=tuple(warnings),
    )


def analyze_meshes(paths: Iterable[str | Path]) -> list[MeshQaResult]:
    return [analyze_mesh(path) for path in sorted(Path(p) for p in paths)]


def _fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_tuple(values: tuple[float, float, float], digits: int = 1) -> str:
    return ", ".join(f"{value:.{digits}f}" for value in values)


def format_mesh_qa_markdown(results: list[MeshQaResult], title: str = "Mesh QA Report") -> str:
    lines = [
        f"# {title}",
        "",
        f"Meshes analyzed: {len(results)}",
        "",
        "## Summary",
        "",
        "| mesh | status | watertight | components | faces | extents mm | warnings |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]

    for result in results:
        warnings = ", ".join(result.warnings) if result.warnings else "none"
        lines.append(
            "| "
            f"{Path(result.path).name} | {result.status} | {result.watertight} | "
            f"{result.connected_components} | {result.faces} | "
            f"{_fmt_tuple(result.extents_mm)} | {warnings} |"
        )

    lines.extend(["", "## Detailed Metrics", ""])

    for result in results:
        lines.extend(
            [
                f"### {Path(result.path).name}",
                "",
                f"- Path: `{result.path}`",
                f"- File size: {_fmt_number(result.file_size_mb)} MB",
                f"- Vertices/faces: {result.vertices} / {result.faces}",
                f"- Bounds min mm: {_fmt_tuple(result.bounds_min_mm)}",
                f"- Bounds max mm: {_fmt_tuple(result.bounds_max_mm)}",
                f"- Extents mm: {_fmt_tuple(result.extents_mm)}",
                f"- Surface area: {_fmt_number(result.surface_area_mm2)} mm^2",
                f"- Absolute volume: {_fmt_number(result.volume_mm3)} mm^3",
                f"- Watertight: {result.watertight}",
                f"- Winding consistent: {result.winding_consistent}",
                f"- Euler number: {result.euler_number}",
                f"- Connected components: {result.connected_components}",
                f"- Boundary edges: {result.boundary_edge_count}",
                f"- Non-manifold edges: {result.nonmanifold_edge_count}",
                f"- Nondegenerate face fraction: {_fmt_number(result.nondegenerate_face_fraction, 5)}",
                f"- Warnings: {', '.join(result.warnings) if result.warnings else 'none'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `pass` means the mesh has no automated QA warnings; it still needs visual inspection.",
            "- `review` means the mesh is usable as a first export, but should be cleaned, split, smoothed, decimated, or repaired before CAD/manufacturing.",
            "- Multiple connected components are expected for structures such as ribs/spine, lungs, kidneys, or multi-lumen vascular labels, but they still need deliberate CAD handling.",
        ]
    )
    return "\n".join(lines)


def write_mesh_qa_report(
    results: list[MeshQaResult],
    output_path: str | Path,
    title: str = "Mesh QA Report",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_mesh_qa_markdown(results, title=title))
    return output
