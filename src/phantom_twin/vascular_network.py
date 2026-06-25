from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any, Iterable

import numpy as np

from .mesh_clean import _fix_normals


def _import_dependencies():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
        import trimesh  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Vascular network scaffold generation requires matplotlib, trimesh, and PyYAML."
        ) from exc
    return plt, Line2D, Poly3DCollection, trimesh, yaml


@dataclass(frozen=True)
class NetworkNode:
    id: str
    label: str
    kind: str
    role: str
    position_mm: tuple[float, float, float]
    radius_mm: float
    boundary_role: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NetworkEdge:
    id: str
    label: str
    source: str
    target: str
    vessel_type: str
    flow_role: str
    radius_start_mm: float
    radius_end_mm: float
    polyline_mm: tuple[tuple[float, float, float], ...]
    length_mm: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VascularNetworkScaffoldResult:
    case_id: str
    include_venous_return: bool
    nodes: tuple[NetworkNode, ...]
    edges: tuple[NetworkEdge, ...]
    graph_yaml_path: str
    nodes_csv_path: str
    edges_csv_path: str
    centerline_obj_path: str
    mesh_paths: tuple[str, ...]
    arterial_mesh_paths: tuple[str, ...]
    venous_mesh_paths: tuple[str, ...]
    preview_png_path: str
    report_path: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GraphDeformationStats:
    group_id: str
    voxel_count: int
    volume_cm3: float
    centroid_mm: tuple[float, float, float]
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]
    extent_mm: tuple[float, float, float]


@dataclass(frozen=True)
class VariantGraphDeformationResult:
    case_id: str
    variant_id: str
    graph_yaml_path: str
    nodes_csv_path: str
    edges_csv_path: str
    preview_png_path: str
    report_path: str
    baseline_graph_path: str
    baseline_labels_path: str
    variant_labels_path: str
    node_count: int
    edge_count: int
    mean_node_displacement_mm: float
    max_node_displacement_mm: float
    body_volume_delta_percent: float
    radius_scale: float
    notes: tuple[str, ...]


_DEFORM_GROUP_LABELS = {
    "body": tuple(range(1, 16)),
    "lungs": (8,),
    "liver": (6,),
    "kidneys": (7,),
    "bone": (10, 11),
    "vessel_wall": (13,),
    "vascular_fluid": (14, 15),
}


def _as_point(values: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = values
    return (float(x), float(y), float(z))


def _add(point: tuple[float, float, float], dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    return (point[0] + dx, point[1] + dy, point[2] + dz)


def _line_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm


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


def _catmull_rom(points: list[tuple[float, float, float]], samples_per_segment: int = 8) -> tuple[tuple[float, float, float], ...]:
    if len(points) < 2:
        raise ValueError("At least two points are required for a scaffold edge")
    if len(points) == 2:
        p0 = np.array(points[0], dtype=float)
        p1 = np.array(points[1], dtype=float)
        samples = [p0 + (p1 - p0) * t for t in np.linspace(0.0, 1.0, samples_per_segment + 1)]
        return tuple(_as_point(sample) for sample in samples)

    controls = [np.array(point, dtype=float) for point in points]
    samples: list[np.ndarray] = []
    for index in range(len(controls) - 1):
        p0 = controls[max(index - 1, 0)]
        p1 = controls[index]
        p2 = controls[index + 1]
        p3 = controls[min(index + 2, len(controls) - 1)]
        t_values = np.linspace(0.0, 1.0, samples_per_segment + 1)
        if index > 0:
            t_values = t_values[1:]
        for t in t_values:
            t2 = t * t
            t3 = t2 * t
            sample = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            samples.append(sample)
    samples[0] = controls[0]
    samples[-1] = controls[-1]
    return tuple(_as_point(sample) for sample in samples)


def _boundary_by_name(spec: dict[str, Any], name: str) -> dict[str, Any] | None:
    for boundary in spec.get("flow_boundary_labels", []):
        if str(boundary.get("name")) == name:
            return boundary
    return None


def _load_combined_spec(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    spec_path = Path(path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Combined phantom spec was not found: {spec_path}")
    data = yaml.safe_load(spec_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Combined phantom spec is not a YAML mapping: {spec_path}")
    return data


def _default_body_mesh_path(case_id: str) -> Path | None:
    candidate = Path("outputs") / "render3d" / "combined" / "meshes" / f"{case_id}_body_envelope_v001.stl"
    return candidate if candidate.exists() else None


def _build_nodes(spec: dict[str, Any], include_venous_return: bool) -> dict[str, NetworkNode]:
    placement = spec.get("placement", {})
    target_center = _as_point(placement.get("target_torso_center_mm", (180.0, 160.0, 204.0)))
    proximal = _boundary_by_name(spec, "z_max_port")
    distal = _boundary_by_name(spec, "z_min_port")
    aorta_inlet = _as_point(proximal["center_mm"]) if proximal else _add(target_center, -3.0, -32.0, 58.0)
    distal_anchor = _as_point(distal["center_mm"]) if distal else _add(target_center, 10.0, 24.0, -66.0)
    inlet_radius = float(proximal.get("radius_mm", 8.5)) if proximal else 8.5
    distal_radius = float(distal.get("radius_mm", 5.75)) if distal else 5.75

    nodes = {
        "aorta_inlet": NetworkNode(
            id="aorta_inlet",
            label="Aorta trunk inlet",
            kind="arterial",
            role="inlet",
            position_mm=aorta_inlet,
            radius_mm=max(inlet_radius, 8.0),
            boundary_role="arterial_inlet",
            notes=("anchored_to_existing_combined_z_max_port",),
        ),
        "descending_aorta_mid": NetworkNode(
            id="descending_aorta_mid",
            label="Descending aorta midline control",
            kind="arterial",
            role="junction",
            position_mm=_add(target_center, 0.0, -16.0, 38.0),
            radius_mm=7.5,
        ),
        "visceral_branch_origin": NetworkNode(
            id="visceral_branch_origin",
            label="Visceral branch origin placeholder",
            kind="arterial",
            role="junction",
            position_mm=_add(target_center, 2.0, -15.0, 32.0),
            radius_mm=7.2,
            notes=("synthetic_branch_origin",),
        ),
        "renal_branch_origin": NetworkNode(
            id="renal_branch_origin",
            label="Renal branch origin placeholder",
            kind="arterial",
            role="junction",
            position_mm=_add(target_center, 5.0, 0.0, 2.0),
            radius_mm=6.8,
            notes=("synthetic_branch_origin",),
        ),
        "aorta_distal_anchor": NetworkNode(
            id="aorta_distal_anchor",
            label="Distal aorta anchor",
            kind="arterial",
            role="junction",
            position_mm=distal_anchor,
            radius_mm=max(distal_radius, 5.5),
            notes=("anchored_to_existing_combined_z_min_port",),
        ),
        "aortic_bifurcation": NetworkNode(
            id="aortic_bifurcation",
            label="Aortic bifurcation",
            kind="arterial",
            role="junction",
            position_mm=_add(distal_anchor, 0.0, 6.0, -18.0),
            radius_mm=5.4,
        ),
        "left_common_iliac_outlet": NetworkNode(
            id="left_common_iliac_outlet",
            label="Left common iliac outlet",
            kind="arterial",
            role="outlet",
            position_mm=_add(target_center, -32.0, 45.0, -114.0),
            radius_mm=4.5,
            boundary_role="arterial_outlet",
        ),
        "right_common_iliac_outlet": NetworkNode(
            id="right_common_iliac_outlet",
            label="Right common iliac outlet",
            kind="arterial",
            role="outlet",
            position_mm=_add(target_center, 48.0, 44.0, -114.0),
            radius_mm=4.5,
            boundary_role="arterial_outlet",
        ),
        "left_renal_outlet": NetworkNode(
            id="left_renal_outlet",
            label="Left renal artery outlet",
            kind="arterial",
            role="outlet",
            position_mm=_add(target_center, -65.0, -8.0, 2.0),
            radius_mm=3.2,
            boundary_role="arterial_outlet",
        ),
        "right_renal_outlet": NetworkNode(
            id="right_renal_outlet",
            label="Right renal artery outlet",
            kind="arterial",
            role="outlet",
            position_mm=_add(target_center, 66.0, -8.0, 2.0),
            radius_mm=3.2,
            boundary_role="arterial_outlet",
        ),
        "hepatic_placeholder_outlet": NetworkNode(
            id="hepatic_placeholder_outlet",
            label="Hepatic artery placeholder outlet",
            kind="arterial",
            role="placeholder_outlet",
            position_mm=_add(target_center, -55.0, -48.0, 34.0),
            radius_mm=2.6,
            boundary_role="arterial_outlet",
            notes=("placeholder_not_segmented_from_cta",),
        ),
        "splenic_placeholder_outlet": NetworkNode(
            id="splenic_placeholder_outlet",
            label="Splenic artery placeholder outlet",
            kind="arterial",
            role="placeholder_outlet",
            position_mm=_add(target_center, 60.0, -46.0, 32.0),
            radius_mm=2.6,
            boundary_role="arterial_outlet",
            notes=("placeholder_not_segmented_from_cta",),
        ),
    }

    if include_venous_return:
        nodes.update(
            {
                "ivc_lower_return_inlet": NetworkNode(
                    id="ivc_lower_return_inlet",
                    label="Lower venous return inlet",
                    kind="venous",
                    role="inlet",
                    position_mm=_add(target_center, -28.0, 58.0, -115.0),
                    radius_mm=5.5,
                    boundary_role="venous_inlet",
                    notes=("optional_synthetic_venous_return",),
                ),
                "ivc_bifurcation_return": NetworkNode(
                    id="ivc_bifurcation_return",
                    label="Iliocaval return junction",
                    kind="venous",
                    role="junction",
                    position_mm=_add(target_center, -28.0, 48.0, -78.0),
                    radius_mm=5.8,
                    notes=("optional_synthetic_venous_return",),
                ),
                "ivc_renal_junction": NetworkNode(
                    id="ivc_renal_junction",
                    label="IVC renal vein junction",
                    kind="venous",
                    role="junction",
                    position_mm=_add(target_center, -31.0, -30.0, 12.0),
                    radius_mm=6.0,
                    notes=("optional_synthetic_venous_return",),
                ),
                "ivc_hepatic_junction": NetworkNode(
                    id="ivc_hepatic_junction",
                    label="IVC hepatic return junction",
                    kind="venous",
                    role="junction",
                    position_mm=_add(target_center, -31.0, -58.0, 45.0),
                    radius_mm=6.2,
                    notes=("optional_synthetic_venous_return",),
                ),
                "ivc_outlet": NetworkNode(
                    id="ivc_outlet",
                    label="Superior venous return outlet",
                    kind="venous",
                    role="outlet",
                    position_mm=_add(target_center, -31.0, -61.0, 65.0),
                    radius_mm=6.5,
                    boundary_role="venous_outlet",
                    notes=("optional_synthetic_venous_return",),
                ),
                "left_renal_vein_inlet": NetworkNode(
                    id="left_renal_vein_inlet",
                    label="Left renal vein inlet",
                    kind="venous",
                    role="inlet",
                    position_mm=_add(target_center, -58.0, -36.0, 12.0),
                    radius_mm=3.0,
                    boundary_role="venous_inlet",
                    notes=("optional_synthetic_venous_return",),
                ),
                "right_renal_vein_inlet": NetworkNode(
                    id="right_renal_vein_inlet",
                    label="Right renal vein inlet",
                    kind="venous",
                    role="inlet",
                    position_mm=_add(target_center, 60.0, -36.0, 12.0),
                    radius_mm=3.0,
                    boundary_role="venous_inlet",
                    notes=("optional_synthetic_venous_return",),
                ),
                "hepatic_venous_placeholder_inlet": NetworkNode(
                    id="hepatic_venous_placeholder_inlet",
                    label="Hepatic venous placeholder inlet",
                    kind="venous",
                    role="placeholder_inlet",
                    position_mm=_add(target_center, -42.0, -57.0, 50.0),
                    radius_mm=3.5,
                    boundary_role="venous_inlet",
                    notes=("placeholder_not_segmented_from_cta", "optional_synthetic_venous_return"),
                ),
                "splenic_venous_placeholder_inlet": NetworkNode(
                    id="splenic_venous_placeholder_inlet",
                    label="Splenic/portal return placeholder inlet",
                    kind="venous",
                    role="placeholder_inlet",
                    position_mm=_add(target_center, 52.0, -58.0, 47.0),
                    radius_mm=2.8,
                    boundary_role="venous_inlet",
                    notes=("placeholder_not_segmented_from_cta", "optional_synthetic_venous_return"),
                ),
            }
        )

    return nodes


def _make_edge(
    nodes: dict[str, NetworkNode],
    edge_id: str,
    label: str,
    source: str,
    target: str,
    vessel_type: str,
    flow_role: str,
    via: tuple[tuple[float, float, float], ...] = (),
    radius_start_mm: float | None = None,
    radius_end_mm: float | None = None,
    notes: tuple[str, ...] = (),
) -> NetworkEdge:
    start = nodes[source]
    end = nodes[target]
    controls = [start.position_mm, *via, end.position_mm]
    polyline = _catmull_rom(controls)
    length = _line_length(np.array(polyline, dtype=float))
    return NetworkEdge(
        id=edge_id,
        label=label,
        source=source,
        target=target,
        vessel_type=vessel_type,
        flow_role=flow_role,
        radius_start_mm=float(radius_start_mm if radius_start_mm is not None else start.radius_mm),
        radius_end_mm=float(radius_end_mm if radius_end_mm is not None else end.radius_mm),
        polyline_mm=polyline,
        length_mm=length,
        notes=notes,
    )


def _build_edges(nodes: dict[str, NetworkNode], include_venous_return: bool) -> tuple[NetworkEdge, ...]:
    center = nodes["renal_branch_origin"].position_mm
    edges = [
        _make_edge(
            nodes,
            "aorta_inlet_to_descending",
            "Aorta trunk: inlet to descending control",
            "aorta_inlet",
            "descending_aorta_mid",
            "arterial",
            "aorta_trunk",
            notes=("trunk",),
        ),
        _make_edge(
            nodes,
            "descending_to_visceral_origin",
            "Aorta trunk: descending to visceral origin",
            "descending_aorta_mid",
            "visceral_branch_origin",
            "arterial",
            "aorta_trunk",
            notes=("trunk",),
        ),
        _make_edge(
            nodes,
            "visceral_to_renal_origin",
            "Aorta trunk: visceral origin to renal origin",
            "visceral_branch_origin",
            "renal_branch_origin",
            "arterial",
            "aorta_trunk",
            via=(_add(center, -1.0, -5.0, 18.0),),
            notes=("trunk",),
        ),
        _make_edge(
            nodes,
            "renal_origin_to_distal_aorta",
            "Aorta trunk: renal origin to distal anchor",
            "renal_branch_origin",
            "aorta_distal_anchor",
            "arterial",
            "aorta_trunk",
            via=(_add(center, 6.0, 12.0, -38.0),),
            notes=("trunk",),
        ),
        _make_edge(
            nodes,
            "distal_aorta_to_bifurcation",
            "Aorta trunk: distal anchor to bifurcation",
            "aorta_distal_anchor",
            "aortic_bifurcation",
            "arterial",
            "aorta_trunk",
            notes=("trunk",),
        ),
        _make_edge(
            nodes,
            "bifurcation_to_left_common_iliac",
            "Left common iliac branch",
            "aortic_bifurcation",
            "left_common_iliac_outlet",
            "arterial",
            "iliac_branch",
            via=(_add(nodes["aortic_bifurcation"].position_mm, -16.0, 14.0, -16.0),),
        ),
        _make_edge(
            nodes,
            "bifurcation_to_right_common_iliac",
            "Right common iliac branch",
            "aortic_bifurcation",
            "right_common_iliac_outlet",
            "arterial",
            "iliac_branch",
            via=(_add(nodes["aortic_bifurcation"].position_mm, 20.0, 14.0, -16.0),),
        ),
        _make_edge(
            nodes,
            "renal_origin_to_left_renal",
            "Left renal artery branch",
            "renal_branch_origin",
            "left_renal_outlet",
            "arterial",
            "renal_branch",
            via=(_add(nodes["renal_branch_origin"].position_mm, -33.0, -7.0, 0.0),),
        ),
        _make_edge(
            nodes,
            "renal_origin_to_right_renal",
            "Right renal artery branch",
            "renal_branch_origin",
            "right_renal_outlet",
            "arterial",
            "renal_branch",
            via=(_add(nodes["renal_branch_origin"].position_mm, 34.0, -7.0, 0.0),),
        ),
        _make_edge(
            nodes,
            "visceral_origin_to_hepatic_placeholder",
            "Hepatic artery placeholder branch",
            "visceral_branch_origin",
            "hepatic_placeholder_outlet",
            "arterial",
            "hepatic_placeholder_branch",
            via=(_add(nodes["visceral_branch_origin"].position_mm, -25.0, -20.0, 5.0),),
            notes=("placeholder_not_patient_specific",),
        ),
        _make_edge(
            nodes,
            "visceral_origin_to_splenic_placeholder",
            "Splenic artery placeholder branch",
            "visceral_branch_origin",
            "splenic_placeholder_outlet",
            "arterial",
            "splenic_placeholder_branch",
            via=(_add(nodes["visceral_branch_origin"].position_mm, 29.0, -19.0, 4.0),),
            notes=("placeholder_not_patient_specific",),
        ),
    ]

    if include_venous_return:
        edges.extend(
            [
                _make_edge(
                    nodes,
                    "ivc_lower_to_bifurcation_return",
                    "Lower venous return to iliocaval junction",
                    "ivc_lower_return_inlet",
                    "ivc_bifurcation_return",
                    "venous",
                    "venous_return_trunk",
                    notes=("optional_return_path",),
                ),
                _make_edge(
                    nodes,
                    "ivc_bifurcation_to_renal_junction",
                    "IVC trunk to renal junction",
                    "ivc_bifurcation_return",
                    "ivc_renal_junction",
                    "venous",
                    "venous_return_trunk",
                    notes=("optional_return_path",),
                ),
                _make_edge(
                    nodes,
                    "ivc_renal_to_hepatic_junction",
                    "IVC trunk to hepatic junction",
                    "ivc_renal_junction",
                    "ivc_hepatic_junction",
                    "venous",
                    "venous_return_trunk",
                    notes=("optional_return_path",),
                ),
                _make_edge(
                    nodes,
                    "ivc_hepatic_to_outlet",
                    "IVC trunk to superior outlet",
                    "ivc_hepatic_junction",
                    "ivc_outlet",
                    "venous",
                    "venous_return_trunk",
                    notes=("optional_return_path",),
                ),
                _make_edge(
                    nodes,
                    "left_renal_vein_to_ivc",
                    "Left renal vein return",
                    "left_renal_vein_inlet",
                    "ivc_renal_junction",
                    "venous",
                    "renal_venous_return",
                    via=(_add(nodes["ivc_renal_junction"].position_mm, -24.0, -3.0, 0.0),),
                    notes=("optional_return_path",),
                ),
                _make_edge(
                    nodes,
                    "right_renal_vein_to_ivc",
                    "Right renal vein return",
                    "right_renal_vein_inlet",
                    "ivc_renal_junction",
                    "venous",
                    "renal_venous_return",
                    via=(_add(nodes["ivc_renal_junction"].position_mm, 25.0, -3.0, 0.0),),
                    notes=("optional_return_path",),
                ),
                _make_edge(
                    nodes,
                    "hepatic_venous_placeholder_to_ivc",
                    "Hepatic venous placeholder return",
                    "hepatic_venous_placeholder_inlet",
                    "ivc_hepatic_junction",
                    "venous",
                    "hepatic_venous_placeholder_return",
                    via=(_add(nodes["ivc_hepatic_junction"].position_mm, -19.0, -5.0, 5.0),),
                    notes=("placeholder_not_patient_specific", "optional_return_path"),
                ),
                _make_edge(
                    nodes,
                    "splenic_venous_placeholder_to_ivc",
                    "Splenic/portal placeholder return",
                    "splenic_venous_placeholder_inlet",
                    "ivc_hepatic_junction",
                    "venous",
                    "splenic_venous_placeholder_return",
                    via=(_add(nodes["ivc_hepatic_junction"].position_mm, 22.0, -6.0, 3.0),),
                    notes=("placeholder_not_patient_specific", "optional_return_path"),
                ),
            ]
        )

    return tuple(edges)


def _boundary_ids(nodes: Iterable[NetworkNode]) -> dict[str, list[str]]:
    buckets = {
        "arterial_inlet_ids": [],
        "arterial_outlet_ids": [],
        "venous_inlet_ids": [],
        "venous_outlet_ids": [],
    }
    for node in nodes:
        if node.boundary_role == "arterial_inlet":
            buckets["arterial_inlet_ids"].append(node.id)
        elif node.boundary_role == "arterial_outlet":
            buckets["arterial_outlet_ids"].append(node.id)
        elif node.boundary_role == "venous_inlet":
            buckets["venous_inlet_ids"].append(node.id)
        elif node.boundary_role == "venous_outlet":
            buckets["venous_outlet_ids"].append(node.id)
    return buckets


def _node_payload(node: NetworkNode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": node.id,
        "label": node.label,
        "kind": node.kind,
        "role": node.role,
        "position_mm": [float(value) for value in node.position_mm],
        "radius_mm": float(node.radius_mm),
    }
    if node.boundary_role:
        payload["boundary_role"] = node.boundary_role
    if node.notes:
        payload["notes"] = list(node.notes)
    return payload


def _edge_payload(edge: NetworkEdge) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": edge.id,
        "label": edge.label,
        "source": edge.source,
        "target": edge.target,
        "vessel_type": edge.vessel_type,
        "flow_role": edge.flow_role,
        "radius_start_mm": float(edge.radius_start_mm),
        "radius_end_mm": float(edge.radius_end_mm),
        "length_mm": float(edge.length_mm),
        "polyline_mm": [[float(value) for value in point] for point in edge.polyline_mm],
    }
    if edge.notes:
        payload["notes"] = list(edge.notes)
    return payload


def _write_graph_yaml(
    path: Path,
    case_id: str,
    combined_spec_path: Path,
    include_venous_return: bool,
    nodes: tuple[NetworkNode, ...],
    edges: tuple[NetworkEdge, ...],
) -> None:
    *_, yaml = _import_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "coordinate_units": "mm",
        "scaffold_type": "synthetic_major_vessel_network",
        "source_combined_spec": str(combined_spec_path),
        "include_venous_return": include_venous_return,
        "graph_metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "boundary_ids": _boundary_ids(nodes),
            "radius_units": "mm",
            "edge_length_units": "mm",
            "coordinate_note": (
                "Coordinates are in the combined digital phantom NIfTI physical millimeter space. "
                "Left/right labels are scaffold conventions until a patient-specific orientation "
                "transform is finalized."
            ),
        },
        "provenance_notes": [
            "This is a synthetic engineering scaffold, not a full segmented vascular tree.",
            "Aorta anchors use the existing embedded ImageTBAD flow-boundary centers where available.",
            "Renal, iliac, hepatic, splenic, and venous-return branches are editable placeholders for simulation setup.",
        ],
        "nodes": [_node_payload(node) for node in nodes],
        "edges": [_edge_payload(edge) for edge in edges],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_nodes_csv(path: Path, nodes: tuple[NetworkNode, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "id",
                "label",
                "kind",
                "role",
                "x_mm",
                "y_mm",
                "z_mm",
                "radius_mm",
                "boundary_role",
                "notes",
            ]
        )
        for node in nodes:
            writer.writerow(
                [
                    node.id,
                    node.label,
                    node.kind,
                    node.role,
                    *[f"{value:.6f}" for value in node.position_mm],
                    f"{node.radius_mm:.6f}",
                    node.boundary_role or "",
                    ";".join(node.notes),
                ]
            )


def _write_edges_csv(path: Path, edges: tuple[NetworkEdge, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "id",
                "label",
                "source",
                "target",
                "vessel_type",
                "flow_role",
                "radius_start_mm",
                "radius_end_mm",
                "length_mm",
                "polyline_point_count",
                "notes",
            ]
        )
        for edge in edges:
            writer.writerow(
                [
                    edge.id,
                    edge.label,
                    edge.source,
                    edge.target,
                    edge.vessel_type,
                    edge.flow_role,
                    f"{edge.radius_start_mm:.6f}",
                    f"{edge.radius_end_mm:.6f}",
                    f"{edge.length_mm:.6f}",
                    len(edge.polyline_mm),
                    ";".join(edge.notes),
                ]
            )


def _write_centerline_obj(path: Path, edges: tuple[NetworkEdge, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    next_vertex = 1
    with path.open("w") as obj:
        obj.write("# Synthetic vascular network scaffold centerlines\n")
        obj.write("# Units: mm\n")
        for edge in edges:
            obj.write(f"g {edge.id}\n")
            indices = []
            for point in edge.polyline_mm:
                obj.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                indices.append(next_vertex)
                next_vertex += 1
            obj.write("l " + " ".join(str(index) for index in indices) + "\n")


def _cylinder_between_points(p0: np.ndarray, p1: np.ndarray, radius: float, sections: int):
    _, _, _, trimesh, _ = _import_dependencies()
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 1e-6:
        return None
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    transform = np.eye(4)
    transform[:3, :3] = _rotation_from_z(axis)
    mesh.apply_transform(transform)
    mesh.apply_translation((p0 + p1) / 2.0)
    return mesh


def _sphere_at(point: tuple[float, float, float], radius: float):
    _, _, _, trimesh, _ = _import_dependencies()
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.apply_translation(np.array(point, dtype=float))
    return mesh


def _mesh_for_edges(
    nodes_by_id: dict[str, NetworkNode],
    edges: tuple[NetworkEdge, ...],
    vessel_type: str | None,
    sections: int = 32,
):
    _, _, _, trimesh, _ = _import_dependencies()
    filtered = [edge for edge in edges if vessel_type is None or edge.vessel_type == vessel_type]
    meshes = []
    used_nodes = set()

    for edge in filtered:
        points = np.array(edge.polyline_mm, dtype=float)
        if len(points) < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total = float(segment_lengths.sum())
        running = 0.0
        for index, length in enumerate(segment_lengths):
            if length <= 1e-6:
                continue
            t0 = running / total if total > 0 else 0.0
            t1 = (running + float(length)) / total if total > 0 else 1.0
            radius0 = edge.radius_start_mm + (edge.radius_end_mm - edge.radius_start_mm) * t0
            radius1 = edge.radius_start_mm + (edge.radius_end_mm - edge.radius_start_mm) * t1
            segment = _cylinder_between_points(points[index], points[index + 1], (radius0 + radius1) / 2.0, sections)
            if segment is not None:
                meshes.append(segment)
            running += float(length)
        used_nodes.add(edge.source)
        used_nodes.add(edge.target)

    for node_id in sorted(used_nodes):
        node = nodes_by_id[node_id]
        meshes.append(_sphere_at(node.position_mm, max(node.radius_mm * 0.92, 1.0)))

    if not meshes:
        raise ValueError("No scaffold mesh primitives were generated")

    mesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    _fix_normals(mesh)
    return mesh


def _export_mesh(mesh, output_base: Path, formats: tuple[str, ...]) -> tuple[str, ...]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        normalized = fmt.lower().lstrip(".")
        path = output_base.with_suffix(f".{normalized}")
        mesh.export(path)
        paths.append(str(path))
    return tuple(paths)


def _load_body_mesh(path: str | Path | None):
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    _, _, _, trimesh, _ = _import_dependencies()
    mesh = trimesh.load_mesh(candidate, process=True)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values() if len(geometry.faces) > 0]
        if not geometries:
            return None
        mesh = trimesh.util.concatenate(geometries)
    return mesh


def _edge_color(vessel_type: str) -> str:
    if vessel_type == "arterial":
        return "#d43d2f"
    if vessel_type == "venous":
        return "#2878b8"
    return "#4d4d4d"


def _render_preview(
    path: Path,
    nodes: tuple[NetworkNode, ...],
    edges: tuple[NetworkEdge, ...],
    arterial_mesh,
    venous_mesh,
    body_mesh_path: str | Path | None,
) -> None:
    plt, Line2D, Poly3DCollection, *_ = _import_dependencies()
    rng = np.random.default_rng(17)
    body_mesh = _load_body_mesh(body_mesh_path)

    fig = plt.figure(figsize=(12, 9), dpi=180)
    fig.patch.set_facecolor("#f4efe6")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f4efe6")

    if body_mesh is not None:
        triangles = body_mesh.triangles
        if len(triangles) > 18_000:
            triangles = triangles[rng.choice(len(triangles), size=18_000, replace=False)]
        collection = Poly3DCollection(
            triangles,
            facecolors="#d8c7aa",
            edgecolors="none",
            alpha=0.045,
            rasterized=True,
        )
        ax.add_collection3d(collection)

    for mesh, color, alpha in (
        (arterial_mesh, "#dc3b2a", 0.38),
        (venous_mesh, "#2878b8", 0.30),
    ):
        if mesh is None:
            continue
        triangles = mesh.triangles
        if len(triangles) > 50_000:
            triangles = triangles[rng.choice(len(triangles), size=50_000, replace=False)]
        collection = Poly3DCollection(
            triangles,
            facecolors=color,
            edgecolors="none",
            alpha=alpha,
            rasterized=True,
        )
        ax.add_collection3d(collection)

    for edge in edges:
        points = np.array(edge.polyline_mm, dtype=float)
        linewidth = 1.4 + max(edge.radius_start_mm, edge.radius_end_mm) * 0.22
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=_edge_color(edge.vessel_type),
            linewidth=linewidth,
            solid_capstyle="round",
        )

    label_nodes = {
        "aorta_inlet",
        "left_common_iliac_outlet",
        "right_common_iliac_outlet",
        "left_renal_outlet",
        "right_renal_outlet",
        "hepatic_placeholder_outlet",
        "splenic_placeholder_outlet",
        "ivc_lower_return_inlet",
        "ivc_outlet",
    }
    for node in nodes:
        point = np.array(node.position_mm)
        ax.scatter(*point, s=18 + node.radius_mm * 8, color=_edge_color(node.kind), depthshade=False)
        if node.id in label_nodes:
            ax.text(
                point[0] + 3.0,
                point[1] + 3.0,
                point[2] + 3.0,
                node.id.replace("_", "\n"),
                color="#1b252b",
                fontsize=7,
            )

    points = np.array([node.position_mm for node in nodes], dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0) * 1.30
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)
    ax.set_title("Synthetic Vascular Network Scaffold: Aorta, Branches, and Optional Venous Return", pad=18)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#dc3b2a", lw=4, label="arterial scaffold"),
            Line2D([0], [0], color="#2878b8", lw=4, label="venous return scaffold"),
            Line2D([0], [0], color="#d8c7aa", lw=4, alpha=0.35, label="body envelope reference"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _format_node_table(nodes: tuple[NetworkNode, ...]) -> list[str]:
    lines = [
        "| node | kind | role | radius mm | position mm | boundary |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for node in nodes:
        position = ", ".join(f"{value:.1f}" for value in node.position_mm)
        lines.append(
            f"| `{node.id}` | {node.kind} | {node.role} | {node.radius_mm:.1f} | "
            f"{position} | {node.boundary_role or ''} |"
        )
    return lines


def _format_edge_table(edges: tuple[NetworkEdge, ...]) -> list[str]:
    lines = [
        "| edge | vessel | flow role | source -> target | radii mm | length mm |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for edge in edges:
        lines.append(
            f"| `{edge.id}` | {edge.vessel_type} | {edge.flow_role} | "
            f"`{edge.source}` -> `{edge.target}` | "
            f"{edge.radius_start_mm:.1f} -> {edge.radius_end_mm:.1f} | {edge.length_mm:.1f} |"
        )
    return lines


def _write_report(
    path: Path,
    case_id: str,
    combined_spec_path: Path,
    include_venous_return: bool,
    nodes: tuple[NetworkNode, ...],
    edges: tuple[NetworkEdge, ...],
    graph_yaml_path: Path,
    nodes_csv_path: Path,
    edges_csv_path: Path,
    centerline_obj_path: Path,
    mesh_paths: tuple[str, ...],
    preview_png_path: Path,
) -> None:
    boundary_ids = _boundary_ids(nodes)
    arterial_edges = [edge for edge in edges if edge.vessel_type == "arterial"]
    venous_edges = [edge for edge in edges if edge.vessel_type == "venous"]
    total_length = sum(edge.length_mm for edge in edges)
    arterial_length = sum(edge.length_mm for edge in arterial_edges)
    venous_length = sum(edge.length_mm for edge in venous_edges)

    lines = [
        "# Vascular Network Scaffold Stage 001",
        "",
        f"Case ID: `{case_id}`",
        "",
        "## Summary",
        "",
        f"- Nodes: {len(nodes)}",
        f"- Edges: {len(edges)}",
        f"- Include venous return: {include_venous_return}",
        f"- Arterial scaffold length: {arterial_length:.1f} mm",
        f"- Venous scaffold length: {venous_length:.1f} mm",
        f"- Total scaffold centerline length: {total_length:.1f} mm",
        f"- Combined phantom source spec: `{combined_spec_path}`",
        "",
        "## Boundary IDs",
        "",
        f"- Arterial inlet IDs: {', '.join(f'`{item}`' for item in boundary_ids['arterial_inlet_ids'])}",
        f"- Arterial outlet IDs: {', '.join(f'`{item}`' for item in boundary_ids['arterial_outlet_ids'])}",
        f"- Venous inlet IDs: {', '.join(f'`{item}`' for item in boundary_ids['venous_inlet_ids']) or 'none'}",
        f"- Venous outlet IDs: {', '.join(f'`{item}`' for item in boundary_ids['venous_outlet_ids']) or 'none'}",
        "",
        "## Outputs",
        "",
        f"- Graph YAML: `{graph_yaml_path}`",
        f"- Nodes CSV: `{nodes_csv_path}`",
        f"- Edges CSV: `{edges_csv_path}`",
        f"- Centerline OBJ: `{centerline_obj_path}`",
        f"- Preview PNG: `{preview_png_path}`",
        f"- Mesh files: {', '.join(f'`{Path(path).name}`' for path in mesh_paths)}",
        "",
        "## Nodes",
        "",
        *_format_node_table(nodes),
        "",
        "## Edges",
        "",
        *_format_edge_table(edges),
        "",
        "## Interpretation",
        "",
        "- This scaffold is a digital engineering graph, not a patient-specific full vascular segmentation.",
        "- The aortic inlet and distal anchor are tied to the existing combined phantom flow-boundary centers when available.",
        "- Hepatic, splenic, and venous paths are placeholders so we can start defining boundary conditions before full branch segmentation exists.",
        "- The exported tube meshes are render/CAD reference geometry. They are overlapping primitives, not a boolean-unioned CFD volume.",
        "",
        "## Next Build Step",
        "",
        "- Use this graph to voxelize arterial/venous lumen masks into the combined NIfTI and assign distinct material labels for flowing blood and optional contrast.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_vascular_network_scaffold(
    combined_spec_path: str | Path,
    output_dir: str | Path = "outputs/digital/vascular_network",
    case_id: str = "ct_org_case0_imagetbad_case125",
    include_venous_return: bool = True,
    formats: tuple[str, ...] = ("stl", "ply", "obj"),
    body_mesh_path: str | Path | None = None,
    report_path: str | Path | None = "outputs/reports/vascular_network_scaffold_stage001.md",
) -> VascularNetworkScaffoldResult:
    combined_spec = Path(combined_spec_path)
    spec = _load_combined_spec(combined_spec)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    nodes_by_id = _build_nodes(spec, include_venous_return=include_venous_return)
    nodes = tuple(nodes_by_id[key] for key in sorted(nodes_by_id))
    edges = _build_edges(nodes_by_id, include_venous_return=include_venous_return)

    graph_yaml = output / f"{case_id}_vascular_network_graph_v001.yaml"
    nodes_csv = output / f"{case_id}_vascular_network_nodes_v001.csv"
    edges_csv = output / f"{case_id}_vascular_network_edges_v001.csv"
    centerline_obj = output / f"{case_id}_vascular_network_centerlines_v001.obj"
    preview_png = output / f"{case_id}_vascular_network_preview_v001.png"
    report = Path(report_path) if report_path else output / f"{case_id}_vascular_network_report_v001.md"

    _write_graph_yaml(graph_yaml, case_id, combined_spec, include_venous_return, nodes, edges)
    _write_nodes_csv(nodes_csv, nodes)
    _write_edges_csv(edges_csv, edges)
    _write_centerline_obj(centerline_obj, edges)

    arterial_mesh = _mesh_for_edges(nodes_by_id, edges, "arterial")
    arterial_paths = _export_mesh(
        arterial_mesh,
        output / f"{case_id}_vascular_network_arterial_tubes_v001",
        formats,
    )

    venous_mesh = None
    venous_paths: tuple[str, ...] = ()
    if include_venous_return:
        venous_mesh = _mesh_for_edges(nodes_by_id, edges, "venous")
        venous_paths = _export_mesh(
            venous_mesh,
            output / f"{case_id}_vascular_network_venous_tubes_v001",
            formats,
        )

    combined_mesh = _mesh_for_edges(nodes_by_id, edges, None)
    combined_paths = _export_mesh(
        combined_mesh,
        output / f"{case_id}_vascular_network_all_tubes_v001",
        formats,
    )
    mesh_paths = (*arterial_paths, *venous_paths, *combined_paths)

    preview_body_mesh = body_mesh_path
    if preview_body_mesh is None:
        preview_body_mesh = _default_body_mesh_path(case_id)
    _render_preview(preview_png, nodes, edges, arterial_mesh, venous_mesh, preview_body_mesh)

    _write_report(
        report,
        case_id,
        combined_spec,
        include_venous_return,
        nodes,
        edges,
        graph_yaml,
        nodes_csv,
        edges_csv,
        centerline_obj,
        mesh_paths,
        preview_png,
    )

    notes = (
        "synthetic_major_vessel_scaffold_not_patient_specific",
        "aorta_anchors_use_existing_combined_flow_boundary_centers",
        "tube_meshes_are_overlapping_reference_primitives_not_boolean_unioned_cfd_lumens",
    )

    return VascularNetworkScaffoldResult(
        case_id=case_id,
        include_venous_return=include_venous_return,
        nodes=nodes,
        edges=edges,
        graph_yaml_path=str(graph_yaml),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        centerline_obj_path=str(centerline_obj),
        mesh_paths=tuple(mesh_paths),
        arterial_mesh_paths=tuple(arterial_paths),
        venous_mesh_paths=tuple(venous_paths),
        preview_png_path=str(preview_png),
        report_path=str(report),
        notes=notes,
    )


def format_vascular_network_scaffold_result(result: VascularNetworkScaffoldResult) -> str:
    boundary_ids = _boundary_ids(result.nodes)
    total_length = sum(edge.length_mm for edge in result.edges)
    lines = [
        "Vascular network scaffold created",
        f"Case ID: {result.case_id}",
        f"Nodes: {len(result.nodes)}",
        f"Edges: {len(result.edges)}",
        f"Venous return included: {result.include_venous_return}",
        f"Total centerline length: {total_length:.1f} mm",
        f"Arterial inlets: {', '.join(boundary_ids['arterial_inlet_ids'])}",
        f"Arterial outlets: {', '.join(boundary_ids['arterial_outlet_ids'])}",
    ]
    if result.include_venous_return:
        lines.extend(
            [
                f"Venous inlets: {', '.join(boundary_ids['venous_inlet_ids'])}",
                f"Venous outlets: {', '.join(boundary_ids['venous_outlet_ids'])}",
            ]
        )
    lines.extend(
        [
            f"Graph YAML: {result.graph_yaml_path}",
            f"Nodes CSV: {result.nodes_csv_path}",
            f"Edges CSV: {result.edges_csv_path}",
            f"Centerline OBJ: {result.centerline_obj_path}",
            f"Preview PNG: {result.preview_png_path}",
            f"Report: {result.report_path}",
        ]
    )
    return "\n".join(lines)


def _import_deformation_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Variant vascular graph deformation requires matplotlib, nibabel, and PyYAML.") from exc
    return plt, nib, yaml


def _load_graph_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_deformation_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _image_spacing(image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _deform_mask(labels: np.ndarray, group_id: str) -> np.ndarray:
    if group_id in {"left_kidney", "right_kidney"}:
        kidney = labels == 7
        coords = np.argwhere(kidney)
        if len(coords) == 0:
            return kidney
        x_grid = np.indices(labels.shape)[0]
        split_x = float(np.median(coords[:, 0]))
        return kidney & (x_grid <= split_x if group_id == "left_kidney" else x_grid > split_x)
    label_ids = _DEFORM_GROUP_LABELS[group_id]
    return labels == label_ids[0] if len(label_ids) == 1 else np.isin(labels, label_ids)


def _deform_stats_for_mask(
    group_id: str,
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    fallback: GraphDeformationStats | None = None,
) -> GraphDeformationStats:
    spacing = np.array(spacing_mm, dtype=float)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        if fallback is not None:
            return GraphDeformationStats(
                group_id,
                0,
                0.0,
                fallback.centroid_mm,
                fallback.bbox_min_mm,
                fallback.bbox_max_mm,
                fallback.extent_mm,
            )
        zeros = (0.0, 0.0, 0.0)
        return GraphDeformationStats(group_id, 0, 0.0, zeros, zeros, zeros, zeros)
    coords_mm = coords.astype(float) * spacing
    bbox_min = coords_mm.min(axis=0)
    bbox_max = coords_mm.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, spacing)
    centroid = coords_mm.mean(axis=0)
    return GraphDeformationStats(
        group_id=group_id,
        voxel_count=int(len(coords)),
        volume_cm3=float(len(coords) * np.prod(spacing) / 1000.0),
        centroid_mm=tuple(float(value) for value in centroid),
        bbox_min_mm=tuple(float(value) for value in bbox_min),
        bbox_max_mm=tuple(float(value) for value in bbox_max),
        extent_mm=tuple(float(value) for value in extent),
    )


def _deform_anatomy_stats(labels: np.ndarray, spacing_mm: tuple[float, float, float]) -> dict[str, GraphDeformationStats]:
    body = _deform_stats_for_mask("body", _deform_mask(labels, "body"), spacing_mm)
    stats = {"body": body}
    for group_id in ("lungs", "liver", "kidneys", "left_kidney", "right_kidney", "bone", "vessel_wall", "vascular_fluid"):
        stats[group_id] = _deform_stats_for_mask(group_id, _deform_mask(labels, group_id), spacing_mm, body)
    return stats


def _deform_group_for_identifier(identifier: str) -> tuple[str | None, float]:
    lowered = identifier.lower()
    if "left_renal" in lowered:
        return "left_kidney", 0.72
    if "right_renal" in lowered:
        return "right_kidney", 0.72
    if "renal_origin" in lowered or "renal_branch" in lowered or "ivc_renal" in lowered:
        return "kidneys", 0.38
    if "hepatic" in lowered:
        return "liver", 0.58
    if "visceral" in lowered:
        return "liver", 0.30
    return None, 0.0


def _deform_bbox_map(point: np.ndarray, baseline: GraphDeformationStats, variant: GraphDeformationStats) -> np.ndarray:
    fraction = (point - np.array(baseline.bbox_min_mm, dtype=float)) / np.maximum(np.array(baseline.extent_mm, dtype=float), 1e-6)
    return np.array(variant.bbox_min_mm, dtype=float) + fraction * np.array(variant.extent_mm, dtype=float)


def _deform_point(
    point_mm: Iterable[float],
    identifier: str,
    baseline_stats: dict[str, GraphDeformationStats],
    variant_stats: dict[str, GraphDeformationStats],
) -> np.ndarray:
    point = np.array(tuple(point_mm), dtype=float)
    body_mapped = _deform_bbox_map(point, baseline_stats["body"], variant_stats["body"])
    group_id, weight = _deform_group_for_identifier(identifier)
    if group_id is None or baseline_stats[group_id].voxel_count == 0 or variant_stats[group_id].voxel_count == 0:
        return body_mapped
    baseline_group = baseline_stats[group_id]
    variant_group = variant_stats[group_id]
    scale = np.clip(
        np.array(variant_group.extent_mm, dtype=float) / np.maximum(np.array(baseline_group.extent_mm, dtype=float), 1e-6),
        0.65,
        1.55,
    )
    landmark = np.array(variant_group.centroid_mm, dtype=float) + (point - np.array(baseline_group.centroid_mm, dtype=float)) * scale
    return (1.0 - weight) * body_mapped + weight * landmark


def _deform_length(points: list[list[float]]) -> float:
    array = np.array(points, dtype=float)
    return 0.0 if len(array) < 2 else float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def _deform_write_node_csv(
    path: Path,
    rows: list[tuple[str, str, str, tuple[float, float, float], tuple[float, float, float], float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "node_id",
                "label",
                "boundary_role",
                "baseline_x_mm",
                "baseline_y_mm",
                "baseline_z_mm",
                "variant_x_mm",
                "variant_y_mm",
                "variant_z_mm",
                "displacement_mm",
            ]
        )
        for node_id, label, role, baseline, variant, displacement in rows:
            writer.writerow([node_id, label, role, *[f"{value:.6f}" for value in baseline], *[f"{value:.6f}" for value in variant], f"{displacement:.6f}"])


def _deform_write_edge_csv(path: Path, edges: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["edge_id", "source", "target", "vessel_type", "flow_role", "length_mm", "radius_start_mm", "radius_end_mm"])
        for edge in edges:
            writer.writerow(
                [
                    edge.get("id"),
                    edge.get("source"),
                    edge.get("target"),
                    edge.get("vessel_type"),
                    edge.get("flow_role"),
                    f"{float(edge.get('length_mm', 0.0)):.6f}",
                    f"{float(edge.get('radius_start_mm', 0.0)):.6f}",
                    f"{float(edge.get('radius_end_mm', 0.0)):.6f}",
                ]
            )


def _deform_write_preview(path: Path, baseline_graph: dict[str, Any], variant_graph: dict[str, Any]) -> None:
    plt, _, _ = _import_deformation_dependencies()
    fig = plt.figure(figsize=(9, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f8f5ef")
    fig.patch.set_facecolor("#f8f5ef")
    for edge in baseline_graph.get("edges", []):
        points = np.array(edge.get("polyline_mm", []), dtype=float)
        if points.ndim == 2 and len(points) >= 2:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#9ca3af", linewidth=0.8, alpha=0.35)
    for edge in variant_graph.get("edges", []):
        points = np.array(edge.get("polyline_mm", []), dtype=float)
        if points.ndim == 2 and len(points) >= 2:
            color = "#dc2626" if str(edge.get("vessel_type")) == "arterial" else "#2563eb"
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=1.9, alpha=0.95)
    ax.set_title("Variant-Deformed Vascular Graph")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.view_init(elev=20, azim=-58)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _deform_write_report(
    path: Path,
    result: VariantGraphDeformationResult,
    node_rows: list[tuple[str, str, str, tuple[float, float, float], tuple[float, float, float], float]],
    baseline_stats: dict[str, GraphDeformationStats],
    variant_stats: dict[str, GraphDeformationStats],
) -> None:
    lines = [
        "# Variant Vascular Graph Deformation Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        f"Variant ID: `{result.variant_id}`",
        "",
        "## Summary",
        "",
        f"- Nodes deformed: {result.node_count}",
        f"- Edges deformed: {result.edge_count}",
        f"- Mean node displacement: {result.mean_node_displacement_mm:.2f} mm",
        f"- Max node displacement: {result.max_node_displacement_mm:.2f} mm",
        f"- Body volume delta: {result.body_volume_delta_percent:+.2f}%",
        f"- Radius scale: {result.radius_scale:.4f}",
        "",
        "## Largest Node Displacements",
        "",
        "| node | boundary role | displacement mm | variant position mm |",
        "| --- | --- | ---: | --- |",
    ]
    for node_id, _, role, _, variant, displacement in sorted(node_rows, key=lambda item: item[-1], reverse=True)[:8]:
        lines.append(f"| `{node_id}` | {role or 'internal'} | {displacement:.2f} | {', '.join(f'{value:.1f}' for value in variant)} |")
    lines.extend(["", "## Anatomy Anchors", "", "| group | baseline cm3 | variant cm3 | delta % | centroid shift mm |", "| --- | ---: | ---: | ---: | ---: |"])
    for group_id in ("body", "lungs", "liver", "kidneys", "bone", "vessel_wall", "vascular_fluid"):
        baseline = baseline_stats[group_id]
        variant = variant_stats[group_id]
        delta = 0.0 if baseline.volume_cm3 == 0 else (variant.volume_cm3 - baseline.volume_cm3) / baseline.volume_cm3 * 100.0
        shift = float(np.linalg.norm(np.array(variant.centroid_mm) - np.array(baseline.centroid_mm)))
        lines.append(f"| {group_id} | {baseline.volume_cm3:.2f} | {variant.volume_cm3:.2f} | {delta:+.2f} | {shift:.2f} |")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- Deformed graph YAML: `{Path(result.graph_yaml_path).name}`")
    lines.append(f"- Node displacement CSV: `{Path(result.nodes_csv_path).name}`")
    lines.append(f"- Edge table CSV: `{Path(result.edges_csv_path).name}`")
    lines.append(f"- Preview PNG: `{Path(result.preview_png_path).name}`")
    lines.extend(["", "## Notes"])
    for note in result.notes:
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def deform_vascular_graph_for_variant(
    baseline_graph_path: str | Path,
    baseline_labels_path: str | Path,
    variant_labels_path: str | Path,
    output_dir: str | Path = "outputs/digital/variant_vascular_graph",
    case_id: str = "variant_flow",
    variant_id: str = "mode01_pos",
    report_path: str | Path | None = None,
) -> VariantGraphDeformationResult:
    _, nib, yaml = _import_deformation_dependencies()
    graph = _load_graph_yaml(baseline_graph_path)
    baseline_image = nib.load(str(baseline_labels_path))
    variant_image = nib.load(str(variant_labels_path))
    baseline_labels = np.rint(np.asanyarray(baseline_image.dataobj)).astype(np.int16)
    variant_labels = np.rint(np.asanyarray(variant_image.dataobj)).astype(np.int16)
    shape_changed = baseline_labels.shape != variant_labels.shape
    baseline_spacing = _image_spacing(baseline_image)
    variant_spacing = _image_spacing(variant_image)
    if any(abs(baseline_spacing[index] - variant_spacing[index]) > 1e-6 for index in range(3)):
        raise ValueError(f"Baseline and variant labels differ in spacing: {baseline_spacing} vs {variant_spacing}")

    baseline_stats = _deform_anatomy_stats(baseline_labels, baseline_spacing)
    variant_stats = _deform_anatomy_stats(variant_labels, variant_spacing)
    body_delta = 0.0
    if baseline_stats["body"].volume_cm3:
        body_delta = (variant_stats["body"].volume_cm3 - baseline_stats["body"].volume_cm3) / baseline_stats["body"].volume_cm3 * 100.0
    radius_scale = float(np.clip((variant_stats["body"].volume_cm3 / max(baseline_stats["body"].volume_cm3, 1e-9)) ** (1 / 3), 0.85, 1.15))

    deformed_nodes: list[dict[str, Any]] = []
    node_positions: dict[str, tuple[float, float, float]] = {}
    node_rows: list[tuple[str, str, str, tuple[float, float, float], tuple[float, float, float], float]] = []
    for node in graph.get("nodes", []):
        node_id = str(node["id"])
        original = tuple(float(value) for value in node["position_mm"])
        mapped = tuple(float(value) for value in _deform_point(original, node_id, baseline_stats, variant_stats))
        displacement = float(np.linalg.norm(np.array(mapped) - np.array(original)))
        payload = dict(node)
        payload["position_mm"] = list(mapped)
        notes = list(payload.get("notes", []))
        notes.append("position_deformed_to_variant_anatomy")
        payload["notes"] = notes
        deformed_nodes.append(payload)
        node_positions[node_id] = mapped
        node_rows.append((node_id, str(node.get("label", node_id)), str(node.get("boundary_role", "")), original, mapped, displacement))

    deformed_edges: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        edge_id = str(edge["id"])
        source_id = str(edge["source"])
        target_id = str(edge["target"])
        source_target = np.array(node_positions[source_id], dtype=float)
        target_target = np.array(node_positions[target_id], dtype=float)
        raw_points = [np.array(point, dtype=float) for point in edge.get("polyline_mm", [])]
        if len(raw_points) < 2:
            raw_points = [source_target, target_target]
        mapped_points = [_deform_point(point, edge_id, baseline_stats, variant_stats) for point in raw_points]
        source_correction = source_target - mapped_points[0]
        target_correction = target_target - mapped_points[-1]
        corrected: list[list[float]] = []
        denominator = max(len(mapped_points) - 1, 1)
        for index, point in enumerate(mapped_points):
            t = index / denominator
            adjusted = point + (1.0 - t) * source_correction + t * target_correction
            corrected.append([float(value) for value in adjusted])
        payload = dict(edge)
        payload["polyline_mm"] = corrected
        payload["length_mm"] = _deform_length(corrected)
        payload["radius_start_mm"] = float(edge.get("radius_start_mm", 1.0)) * radius_scale
        payload["radius_end_mm"] = float(edge.get("radius_end_mm", edge.get("radius_start_mm", 1.0))) * radius_scale
        notes = list(payload.get("notes", []))
        notes.append("polyline_deformed_to_variant_anatomy")
        payload["notes"] = notes
        deformed_edges.append(payload)

    output = Path(output_dir)
    graph_yaml = output / f"{case_id}_variant_deformed_vascular_graph_v001.yaml"
    nodes_csv = output / f"{case_id}_variant_deformed_vascular_nodes_v001.csv"
    edges_csv = output / f"{case_id}_variant_deformed_vascular_edges_v001.csv"
    preview_png = output / f"{case_id}_variant_deformed_vascular_graph_preview_v001.png"
    report = Path(report_path) if report_path is not None else output / f"{case_id}_variant_deformed_vascular_graph_report_v001.md"

    payload = dict(graph)
    payload["case_id"] = case_id
    payload["variant_id"] = variant_id
    payload["source_baseline_graph"] = str(baseline_graph_path)
    payload["source_baseline_labels"] = str(baseline_labels_path)
    payload["source_variant_labels"] = str(variant_labels_path)
    payload["nodes"] = deformed_nodes
    payload["edges"] = deformed_edges
    metadata = dict(payload.get("graph_metadata", {}))
    metadata["node_count"] = len(deformed_nodes)
    metadata["edge_count"] = len(deformed_edges)
    metadata["deformation_method"] = "body_bbox_map_with_organ_centroid_blends"
    metadata["mean_node_displacement_mm"] = float(np.mean([row[-1] for row in node_rows])) if node_rows else 0.0
    metadata["max_node_displacement_mm"] = max((row[-1] for row in node_rows), default=0.0)
    metadata["body_volume_delta_percent"] = body_delta
    metadata["radius_scale"] = radius_scale
    metadata["baseline_label_shape"] = list(baseline_labels.shape)
    metadata["variant_label_shape"] = list(variant_labels.shape)
    payload["graph_metadata"] = metadata
    provenance = list(payload.get("provenance_notes", []))
    provenance.extend(
        [
            "variant_graph_deformed_from_baseline_scaffold_not_segmented_from_cta",
            "body_bbox_mapping_preserves_graph_topology_while_following_variant_envelope",
            "renal_and_hepatic_nodes_receive_organ_centroid_blended_offsets",
        ]
    )
    payload["provenance_notes"] = provenance

    graph_yaml.parent.mkdir(parents=True, exist_ok=True)
    graph_yaml.write_text(yaml.safe_dump(payload, sort_keys=False))
    _deform_write_node_csv(nodes_csv, node_rows)
    _deform_write_edge_csv(edges_csv, deformed_edges)
    _deform_write_preview(preview_png, graph, payload)

    notes = (
        "graph_deformation_is_a_surrogate_registration_not_patient_specific_vessel_segmentation",
        "topology_and_boundary_ids_are_preserved_for_solver_compatibility",
        "use_as_first_true_variant_flow_domain_before_higher_order_registration_or_cfd",
    )
    if shape_changed:
        notes = notes + ("variant_label_grid_shape_differs_from_baseline_but_spacing_matches",)
    result = VariantGraphDeformationResult(
        case_id=case_id,
        variant_id=variant_id,
        graph_yaml_path=str(graph_yaml),
        nodes_csv_path=str(nodes_csv),
        edges_csv_path=str(edges_csv),
        preview_png_path=str(preview_png),
        report_path=str(report),
        baseline_graph_path=str(baseline_graph_path),
        baseline_labels_path=str(baseline_labels_path),
        variant_labels_path=str(variant_labels_path),
        node_count=len(deformed_nodes),
        edge_count=len(deformed_edges),
        mean_node_displacement_mm=metadata["mean_node_displacement_mm"],
        max_node_displacement_mm=metadata["max_node_displacement_mm"],
        body_volume_delta_percent=body_delta,
        radius_scale=radius_scale,
        notes=notes,
    )
    _deform_write_report(report, result, node_rows, baseline_stats, variant_stats)
    return result


def format_variant_graph_deformation_result(result: VariantGraphDeformationResult) -> str:
    lines = [
        "Variant vascular graph deformed",
        f"Case ID: {result.case_id}",
        f"Variant ID: {result.variant_id}",
        f"Nodes/edges: {result.node_count}/{result.edge_count}",
        f"Mean node displacement: {result.mean_node_displacement_mm:.2f} mm",
        f"Max node displacement: {result.max_node_displacement_mm:.2f} mm",
        f"Deformed graph YAML: {result.graph_yaml_path}",
        f"Node CSV: {result.nodes_csv_path}",
        f"Edge CSV: {result.edges_csv_path}",
        f"Preview PNG: {result.preview_png_path}",
        f"Report: {result.report_path}",
    ]
    return "\n".join(lines)
