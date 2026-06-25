from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _import_trimesh():
    try:
        import trimesh  # type: ignore
        import trimesh.repair as repair  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Mesh cleaning requires trimesh. Install with: python3 -m pip install trimesh"
        ) from exc
    return trimesh, repair


@dataclass(frozen=True)
class MeshCleanConfig:
    output_dir: Path
    formats: tuple[str, ...] = ("stl", "ply")
    suffix: str = "cleaned_v001"
    min_component_faces: int = 100
    target_max_faces: int = 80_000
    fill_holes: str = "single"


@dataclass(frozen=True)
class MeshCleanResult:
    source_path: str
    output_paths: tuple[str, ...]
    source_faces: int
    cleaned_faces: int
    source_vertices: int
    cleaned_vertices: int
    source_components: int
    cleaned_components: int
    removed_components: int
    decimated: bool
    notes: tuple[str, ...]


def _load_mesh(path: Path):
    trimesh, _ = _import_trimesh()
    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values() if len(geometry.faces) > 0]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geometries)

    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    return mesh


def _component_count(mesh) -> int:
    return int(len(mesh.split(only_watertight=False)))


def _filter_components(mesh, min_component_faces: int):
    trimesh, _ = _import_trimesh()
    components = mesh.split(only_watertight=False)
    kept = [component for component in components if len(component.faces) >= min_component_faces]

    if not kept:
        kept = [max(components, key=lambda component: len(component.faces))]

    cleaned = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
    cleaned.merge_vertices()
    cleaned.remove_unreferenced_vertices()
    cleaned.process(validate=True)
    return cleaned, len(components) - len(kept)


def _fill_holes(mesh, fill_holes: str) -> None:
    _, repair = _import_trimesh()
    if fill_holes == "none":
        return
    if fill_holes == "single":
        mesh.fill_holes()
        return
    if fill_holes == "fan":
        repair.fill_holes(mesh, use_fan=True)
        return
    raise ValueError("fill_holes must be one of: none, single, fan")


def _fix_normals(mesh) -> None:
    _, repair = _import_trimesh()
    repair.fix_winding(mesh)
    repair.fix_normals(mesh, multibody=True)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()


def _decimate(mesh, target_max_faces: int) -> tuple[object, bool, str | None]:
    if target_max_faces <= 0 or len(mesh.faces) <= target_max_faces:
        return mesh, False, None

    try:
        decimated = mesh.simplify_quadric_decimation(face_count=target_max_faces)
    except Exception as exc:  # pragma: no cover - depends on optional backend
        return mesh, False, f"decimation_failed={type(exc).__name__}: {exc}"

    decimated.merge_vertices()
    decimated.remove_unreferenced_vertices()
    decimated.process(validate=True)
    _fix_normals(decimated)
    return decimated, True, None


def clean_mesh(path: str | Path, config: MeshCleanConfig) -> MeshCleanResult:
    mesh_path = Path(path)
    mesh = _load_mesh(mesh_path)
    source_faces = int(len(mesh.faces))
    source_vertices = int(len(mesh.vertices))
    source_components = _component_count(mesh)
    notes: list[str] = []

    cleaned, removed_components = _filter_components(mesh, config.min_component_faces)
    if removed_components:
        notes.append(f"removed_components={removed_components}")

    _fix_normals(cleaned)
    _fill_holes(cleaned, config.fill_holes)
    _fix_normals(cleaned)

    cleaned, decimated, decimation_note = _decimate(cleaned, config.target_max_faces)
    if decimation_note:
        notes.append(decimation_note)
    if decimated:
        notes.append(f"decimated_to_target_max_faces={config.target_max_faces}")

    _fill_holes(cleaned, config.fill_holes)
    _fix_normals(cleaned)
    cleaned.merge_vertices()
    cleaned.remove_unreferenced_vertices()
    cleaned.process(validate=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = mesh_path.stem
    output_paths: list[str] = []
    for fmt in config.formats:
        normalized_fmt = fmt.lower().lstrip(".")
        output_path = config.output_dir / f"{stem}_{config.suffix}.{normalized_fmt}"
        cleaned.export(output_path)
        output_paths.append(str(output_path))

    return MeshCleanResult(
        source_path=str(mesh_path),
        output_paths=tuple(output_paths),
        source_faces=source_faces,
        cleaned_faces=int(len(cleaned.faces)),
        source_vertices=source_vertices,
        cleaned_vertices=int(len(cleaned.vertices)),
        source_components=source_components,
        cleaned_components=_component_count(cleaned),
        removed_components=removed_components,
        decimated=decimated,
        notes=tuple(notes),
    )


def clean_meshes(paths: Iterable[str | Path], config: MeshCleanConfig) -> list[MeshCleanResult]:
    return [clean_mesh(path, config) for path in sorted(Path(p) for p in paths)]


def format_cleaning_report(results: Sequence[MeshCleanResult]) -> str:
    lines = [
        "# Mesh Cleaning Report",
        "",
        f"Meshes cleaned: {len(results)}",
        "",
        "| source | outputs | faces before | faces after | components before | components after | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for result in results:
        outputs = "<br>".join(Path(path).name for path in result.output_paths)
        notes = ", ".join(result.notes) if result.notes else "none"
        lines.append(
            "| "
            f"{Path(result.source_path).name} | {outputs} | "
            f"{result.source_faces} | {result.cleaned_faces} | "
            f"{result.source_components} | {result.cleaned_components} | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Cleaned files are engineering-prep outputs, not final manufacturing geometry.",
            "- `fill_holes=single` is conservative and preserves most large open boundaries, including likely vascular inlet/outlet ends.",
            "- Meshes with remaining boundary edges should be explicitly capped, trimmed, or ported during CAD design.",
            "- Decimated meshes are easier to open in CAD tools, but should be visually checked against the raw meshes before printing.",
        ]
    )
    return "\n".join(lines)


def write_cleaning_report(results: Sequence[MeshCleanResult], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_cleaning_report(results))
    return output
