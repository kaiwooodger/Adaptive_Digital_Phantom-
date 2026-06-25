from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import numpy as np

from .flow_pulsatile import PA_PER_MMHG


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm as cm  # type: ignore
        import matplotlib.colors as colors  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("4D flow visualization requires matplotlib and PyYAML.") from exc
    return cm, colors, plt, Line2D, Poly3DCollection, yaml


def _import_optional_mesh_dependencies():
    try:
        import trimesh  # type: ignore
    except ImportError:
        return None
    return trimesh


@dataclass(frozen=True)
class Flow4DVisualizationResult:
    case_id: str
    output_dir: str
    frame_dir: str
    frame_count: int
    frame_manifest_csv_path: str
    animation_gif_path: str | None
    contact_sheet_png_path: str
    spec_yaml_path: str
    report_path: str
    graph_yaml_path: str
    edge_timeseries_csv_path: str
    node_timeseries_csv_path: str
    context_scene_spec_path: str | None
    color_by: str
    color_min: float
    color_max: float
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as csvfile:
        return list(csv.DictReader(csvfile))


def _edge_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(edge["id"]): edge for edge in graph.get("edges", [])}


def _node_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in graph.get("nodes", [])}


def _edge_points(edge: dict[str, Any], node_lookup: dict[str, dict[str, Any]]) -> np.ndarray:
    raw_points = edge.get("polyline_mm")
    if raw_points:
        points = np.array(raw_points, dtype=float)
        if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 3:
            return points
    return np.array(
        [
            node_lookup[str(edge["source"])]["position_mm"],
            node_lookup[str(edge["target"])]["position_mm"],
        ],
        dtype=float,
    )


def _rows_by_time(rows: list[dict[str, str]]) -> tuple[list[float], dict[float, list[dict[str, str]]]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        time = float(row["time_s"])
        grouped.setdefault(time, []).append(row)
    times = sorted(grouped)
    return times, grouped


def _select_frame_times(times: list[float], frame_count: int) -> list[float]:
    if not times:
        raise ValueError("No time samples were available for visualization")
    if frame_count >= len(times):
        return times
    indices = np.linspace(0, len(times) - 1, frame_count, dtype=int)
    return [times[int(index)] for index in indices]


def _metric_value(edge_row: dict[str, str], color_by: str) -> float:
    if color_by == "velocity":
        return abs(float(edge_row["mean_velocity_cm_s"]))
    if color_by == "pressure":
        return (float(edge_row["pressure_source_mmhg"]) + float(edge_row["pressure_target_mmhg"])) / 2.0
    if color_by == "flow":
        return abs(float(edge_row["flow_ml_s"]))
    raise ValueError(f"Unsupported color_by value: {color_by}")


def _metric_label(color_by: str) -> str:
    return {
        "velocity": "Speed (cm/s)",
        "pressure": "Pressure (mmHg)",
        "flow": "Flow magnitude (mL/s)",
    }[color_by]


def _metric_values(rows: list[dict[str, str]], color_by: str) -> np.ndarray:
    values = np.array([_metric_value(row, color_by) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _safe_metric_limits(rows: list[dict[str, str]], color_by: str) -> tuple[float, float]:
    values = _metric_values(rows, color_by)
    if values.size == 0:
        return 0.0, 1.0
    lower = float(np.percentile(values, 2))
    upper = float(np.percentile(values, 98))
    if not math.isfinite(lower) or not math.isfinite(upper) or abs(upper - lower) < 1e-9:
        lower = float(values.min())
        upper = float(values.max())
    if abs(upper - lower) < 1e-9:
        upper = lower + 1.0
    return lower, upper


def _scene_mesh_items(
    scene_spec_path: str | Path | None,
    context_group_ids: tuple[str, ...],
    max_triangles_per_group: int,
) -> tuple[tuple[dict[str, object], ...], list[np.ndarray], tuple[str, ...]]:
    if scene_spec_path is None:
        return (), [], ("no_context_scene_spec",)
    trimesh = _import_optional_mesh_dependencies()
    if trimesh is None:
        return (), [], ("trimesh_unavailable_context_meshes_skipped",)

    spec = _load_yaml(scene_spec_path)
    group_by_id = {str(group["id"]): group for group in spec.get("groups", [])}
    items: list[dict[str, object]] = []
    bounds: list[np.ndarray] = []
    notes: list[str] = []
    rng = np.random.default_rng(42)

    for mesh_result in spec.get("meshes", []):
        group_id = str(mesh_result.get("group_id", ""))
        if group_id not in context_group_ids:
            continue
        outputs = [str(path) for path in mesh_result.get("outputs", [])]
        mesh_path = next((path for path in outputs if Path(path).suffix.lower() == ".stl"), outputs[0] if outputs else None)
        if mesh_path is None or not Path(mesh_path).exists():
            notes.append(f"context_mesh_missing={group_id}")
            continue
        mesh = trimesh.load_mesh(mesh_path, process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        triangles = mesh.triangles
        if len(triangles) > max_triangles_per_group:
            triangles = triangles[rng.choice(len(triangles), size=max_triangles_per_group, replace=False)]
        group = group_by_id.get(group_id, {})
        alpha = float(group.get("alpha", 0.12))
        if group_id == "body_envelope":
            alpha = min(alpha, 0.07)
        elif group_id == "bone":
            alpha = min(alpha, 0.18)
        else:
            alpha = min(alpha, 0.22)
        items.append(
            {
                "group_id": group_id,
                "label": str(group.get("label", group_id)),
                "color": str(group.get("color", "#dddddd")),
                "alpha": alpha,
                "triangles": triangles,
            }
        )
        bounds.append(np.array(mesh.bounds, dtype=float))
    if not items:
        notes.append("no_context_meshes_loaded")
    return tuple(items), bounds, tuple(notes)


def _combined_bounds(
    graph_edges: dict[str, dict[str, Any]],
    node_lookup: dict[str, dict[str, Any]],
    context_bounds: list[np.ndarray],
) -> np.ndarray:
    edge_points = [_edge_points(edge, node_lookup) for edge in graph_edges.values()]
    points = np.vstack(edge_points) if edge_points else np.zeros((1, 3), dtype=float)
    graph_bounds = np.array([points.min(axis=0), points.max(axis=0)])
    if not context_bounds:
        return graph_bounds
    return np.array(
        [
            np.vstack([graph_bounds[0], *[bounds[0] for bounds in context_bounds]]).min(axis=0),
            np.vstack([graph_bounds[1], *[bounds[1] for bounds in context_bounds]]).max(axis=0),
        ]
    )


def _set_axes_bounds(ax, bounds: np.ndarray, zoom: float) -> None:
    mins = bounds[0]
    maxs = bounds[1]
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0) * zoom
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _draw_context(ax, context_items: tuple[dict[str, object], ...]) -> None:
    *_, Poly3DCollection, _ = _import_dependencies()
    for item in context_items:
        edge_color = "#111111" if item["group_id"] == "bone" else "none"
        line_width = 0.02 if item["group_id"] == "bone" else 0.0
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


def _label_nodes(ax, graph: dict[str, Any], node_ids: tuple[str, ...]) -> None:
    node_lookup = _node_by_id(graph)
    for node_id in node_ids:
        node = node_lookup.get(node_id)
        if not node:
            continue
        x, y, z = [float(value) for value in node["position_mm"]]
        label = str(node.get("label", node_id)).replace(" placeholder", "")
        ax.text(x, y, z, label, fontsize=6, color="#16202a", zorder=10)


def _render_frame(
    path: Path,
    graph: dict[str, Any],
    edge_rows: list[dict[str, str]],
    node_rows: list[dict[str, str]],
    context_items: tuple[dict[str, object], ...],
    bounds: np.ndarray,
    color_by: str,
    color_min: float,
    color_max: float,
    view_elev: float,
    view_azim: float,
    zoom: float,
    label_boundary_nodes: bool,
) -> None:
    cm, colors, plt, Line2D, *_ = _import_dependencies()
    edge_lookup = _edge_by_id(graph)
    node_lookup = _node_by_id(graph)
    cmap = plt.get_cmap("turbo")
    norm = colors.Normalize(vmin=color_min, vmax=color_max)

    time_s = float(edge_rows[0]["time_s"])
    phase = float(edge_rows[0]["phase"])
    fig = plt.figure(figsize=(11, 9), dpi=150)
    fig.patch.set_facecolor("#f6f1e8")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f1e8")
    _draw_context(ax, context_items)

    for row in edge_rows:
        edge = edge_lookup.get(str(row["edge_id"]))
        if edge is None:
            continue
        points = _edge_points(edge, node_lookup)
        value = _metric_value(row, color_by)
        vessel_type = str(edge.get("vessel_type", ""))
        base_radius = (float(edge.get("radius_start_mm", 2.0)) + float(edge.get("radius_end_mm", 2.0))) / 2.0
        line_width = max(1.4, min(7.5, base_radius * 0.72))
        linestyle = "-" if vessel_type == "arterial" else "--"
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=cmap(norm(value)),
            linewidth=line_width,
            linestyle=linestyle,
            solid_capstyle="round",
            alpha=0.96,
        )

    boundary_ids = tuple(
        str(node["id"])
        for node in graph.get("nodes", [])
        if str(node.get("boundary_role", ""))
    )
    boundary_positions = []
    for node_id in boundary_ids:
        node = node_lookup[node_id]
        boundary_positions.append(node["position_mm"])
    if boundary_positions:
        positions = np.array(boundary_positions, dtype=float)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=18, c="#101820", depthshade=True)
    if label_boundary_nodes:
        _label_nodes(ax, graph, boundary_ids)

    _set_axes_bounds(ax, bounds, zoom=zoom)
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.set_title(
        f"4D Vascular Flow In Digital Phantom\nphase={phase:.3f}, time={time_s:.3f}s, colored by {_metric_label(color_by)}",
        color="#13202a",
        pad=14,
    )
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, shrink=0.62, pad=0.08)
    colorbar.set_label(_metric_label(color_by))
    ax.legend(
        handles=[
            Line2D([0], [0], color="#111111", lw=3, linestyle="-", label="Arterial graph"),
            Line2D([0], [0], color="#111111", lw=3, linestyle="--", label="Venous graph"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#101820", markersize=5, label="Boundary node"),
        ],
        loc="upper left",
        fontsize=7,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_frame_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame_index", "time_s", "phase", "png_path"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_contact_sheet(path: Path, frame_paths: list[Path], columns: int = 4) -> None:
    _, _, plt, *_ = _import_dependencies()
    if not frame_paths:
        raise ValueError("No frame paths were available for contact sheet")
    rows = math.ceil(len(frame_paths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.0, rows * 3.4), dpi=140)
    axes_array = np.array(axes).reshape(rows, columns)
    for ax in axes_array.ravel():
        ax.axis("off")
    for index, frame_path in enumerate(frame_paths):
        image = plt.imread(frame_path)
        ax = axes_array[index // columns, index % columns]
        ax.imshow(image)
        ax.set_title(f"Frame {index:03d}", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_gif(path: Path, frame_paths: list[Path], duration_ms: int) -> str | None:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    images = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
    if not images:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    for image in images:
        image.close()
    return str(path)


def _write_spec(path: Path, result: Flow4DVisualizationResult, frame_rows: list[dict[str, Any]]) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "visualization_type": "time_resolved_4d_vascular_flow_centerline_render",
        "graph_yaml": result.graph_yaml_path,
        "edge_timeseries_csv": result.edge_timeseries_csv_path,
        "node_timeseries_csv": result.node_timeseries_csv_path,
        "context_scene_spec": result.context_scene_spec_path,
        "color_by": result.color_by,
        "color_range": {
            "min": result.color_min,
            "max": result.color_max,
            "label": _metric_label(result.color_by),
        },
        "outputs": {
            "frame_dir": result.frame_dir,
            "frame_manifest_csv": result.frame_manifest_csv_path,
            "animation_gif": result.animation_gif_path,
            "contact_sheet_png": result.contact_sheet_png_path,
            "report": result.report_path,
        },
        "frames": frame_rows,
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_report(result: Flow4DVisualizationResult) -> str:
    lines = [
        "# 4D Vascular Flow Visualization Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        "",
        "## Summary",
        "",
        f"- Frames rendered: {result.frame_count}",
        f"- Color metric: {result.color_by} ({_metric_label(result.color_by)})",
        f"- Color range: {result.color_min:.3f} to {result.color_max:.3f}",
        f"- Context scene: `{result.context_scene_spec_path}`" if result.context_scene_spec_path else "- Context scene: none",
        "",
        "## Outputs",
        "",
        f"- Frame manifest CSV: `{Path(result.frame_manifest_csv_path).name}`",
        f"- Contact sheet PNG: `{Path(result.contact_sheet_png_path).name}`",
        f"- Animation GIF: `{Path(result.animation_gif_path).name}`" if result.animation_gif_path else "- Animation GIF: not written",
        f"- Machine-readable spec: `{Path(result.spec_yaml_path).name}`",
        "",
        "## Interpretation",
        "",
        "- The coupled flow values are mapped back onto the vascular graph polylines in the phantom coordinate system.",
        "- Line width follows scaffold vessel radius; color follows the selected time-varying flow metric.",
        "- The transparent torso context is for spatial orientation and is not recomputed at each phase.",
        "- This visualization is a digital-twin QA view, not a CFD pathline/particle simulation.",
        "",
        "## Notes",
    ]
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_4d_flow_visualization(
    graph_yaml_path: str | Path,
    edge_timeseries_csv_path: str | Path,
    node_timeseries_csv_path: str | Path,
    output_dir: str | Path = "outputs/sim/flow_4d_visualization",
    case_id: str = "ct_org_case0_imagetbad_case125",
    context_scene_spec_path: str | Path | None = None,
    color_by: str = "velocity",
    frame_count: int = 32,
    view_elev: float = 18.0,
    view_azim: float = -58.0,
    zoom: float = 1.08,
    context_group_ids: tuple[str, ...] = ("body_envelope", "bone", "lungs", "liver", "kidneys"),
    max_context_triangles_per_group: int = 3500,
    label_boundary_nodes: bool = True,
    gif_duration_ms: int = 110,
    report_path: str | Path | None = "outputs/reports/flow_4d_visualization_stage001.md",
) -> Flow4DVisualizationResult:
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if color_by not in {"velocity", "pressure", "flow"}:
        raise ValueError("color_by must be one of: velocity, pressure, flow")

    graph_path = Path(graph_yaml_path)
    edge_csv_path = Path(edge_timeseries_csv_path)
    node_csv_path = Path(node_timeseries_csv_path)
    output = Path(output_dir)
    frame_dir = output / "frames"
    output.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    graph = _load_yaml(graph_path)
    edge_rows = _read_csv_rows(edge_csv_path)
    node_rows = _read_csv_rows(node_csv_path)
    edge_times, edge_rows_by_time = _rows_by_time(edge_rows)
    _, node_rows_by_time = _rows_by_time(node_rows)
    selected_times = _select_frame_times(edge_times, frame_count)
    color_min, color_max = _safe_metric_limits(edge_rows, color_by)

    context_items, context_bounds, context_notes = _scene_mesh_items(
        context_scene_spec_path,
        context_group_ids=context_group_ids,
        max_triangles_per_group=max_context_triangles_per_group,
    )
    graph_bounds = _combined_bounds(_edge_by_id(graph), _node_by_id(graph), context_bounds)
    frame_rows: list[dict[str, Any]] = []
    frame_paths: list[Path] = []

    for index, time in enumerate(selected_times):
        rows_for_time = edge_rows_by_time[time]
        node_rows_for_time = node_rows_by_time.get(time, [])
        phase = float(rows_for_time[0]["phase"])
        frame_path = frame_dir / f"{case_id}_flow4d_{color_by}_frame_{index:03d}.png"
        _render_frame(
            frame_path,
            graph=graph,
            edge_rows=rows_for_time,
            node_rows=node_rows_for_time,
            context_items=context_items,
            bounds=graph_bounds,
            color_by=color_by,
            color_min=color_min,
            color_max=color_max,
            view_elev=view_elev,
            view_azim=view_azim,
            zoom=zoom,
            label_boundary_nodes=label_boundary_nodes,
        )
        frame_paths.append(frame_path)
        frame_rows.append(
            {
                "frame_index": index,
                "time_s": f"{time:.6f}",
                "phase": f"{phase:.6f}",
                "png_path": str(frame_path),
            }
        )

    frame_manifest = output / f"{case_id}_flow4d_frame_manifest_v001.csv"
    contact_sheet = output / f"{case_id}_flow4d_{color_by}_contact_sheet_v001.png"
    gif_path = output / f"{case_id}_flow4d_{color_by}_animation_v001.gif"
    spec_yaml = output / f"{case_id}_flow4d_visualization_spec_v001.yaml"
    report = Path(report_path) if report_path else output / f"{case_id}_flow4d_visualization_report_v001.md"

    _write_frame_manifest(frame_manifest, frame_rows)
    _write_contact_sheet(contact_sheet, frame_paths[: min(len(frame_paths), 16)])
    animation_gif = _write_gif(gif_path, frame_paths, gif_duration_ms)

    notes = [
        "coupled_flow_timeseries_mapped_to_vascular_graph_polylines",
        f"rendered_metric={color_by}",
        f"context_groups={','.join(context_group_ids)}",
        "line_width_scaled_by_scaffold_radius",
    ]
    notes.extend(context_notes)
    if animation_gif is None:
        notes.append("gif_not_written_pillow_unavailable")

    result = Flow4DVisualizationResult(
        case_id=case_id,
        output_dir=str(output),
        frame_dir=str(frame_dir),
        frame_count=len(frame_paths),
        frame_manifest_csv_path=str(frame_manifest),
        animation_gif_path=animation_gif,
        contact_sheet_png_path=str(contact_sheet),
        spec_yaml_path=str(spec_yaml),
        report_path=str(report),
        graph_yaml_path=str(graph_path),
        edge_timeseries_csv_path=str(edge_csv_path),
        node_timeseries_csv_path=str(node_csv_path),
        context_scene_spec_path=None if context_scene_spec_path is None else str(context_scene_spec_path),
        color_by=color_by,
        color_min=color_min,
        color_max=color_max,
        notes=tuple(notes),
    )
    _write_spec(spec_yaml, result, frame_rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_4d_flow_visualization_result(result: Flow4DVisualizationResult) -> str:
    lines = [
        "4D vascular flow visualization created",
        f"Case ID: {result.case_id}",
        f"Frames: {result.frame_count}",
        f"Color metric: {result.color_by}",
        f"Color range: {result.color_min:.3f} to {result.color_max:.3f}",
        f"Frame manifest: {result.frame_manifest_csv_path}",
        f"Contact sheet: {result.contact_sheet_png_path}",
        f"Animation GIF: {result.animation_gif_path or 'not written'}",
        f"Spec YAML: {result.spec_yaml_path}",
        f"Report: {result.report_path}",
    ]
    return "\n".join(lines)
