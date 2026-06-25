from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.coarse_vessel_graph import build_btcv_coarse_vessel_graph


def _node(node_id: str, boundary_role: str | None = None) -> dict:
    node = {
        "id": node_id,
        "label": node_id,
        "kind": "arterial" if "aorta" in node_id or "renal" in node_id else "venous",
        "role": "junction",
        "position_mm": [10.0, 12.0, 14.0],
        "radius_mm": 6.0,
        "notes": [],
    }
    if boundary_role:
        node["boundary_role"] = boundary_role
    return node


def _edge(edge_id: str, source: str, target: str, vessel_type: str) -> dict:
    return {
        "id": edge_id,
        "label": edge_id,
        "source": source,
        "target": target,
        "vessel_type": vessel_type,
        "flow_role": "renal_branch" if "renal" in edge_id else "aorta_trunk" if vessel_type == "arterial" else "venous_return_trunk",
        "radius_start_mm": 6.0,
        "radius_end_mm": 6.0,
        "length_mm": 10.0,
        "polyline_mm": [[10.0, 12.0, 14.0], [12.0, 12.0, 18.0]],
        "notes": [],
    }


class CoarseVesselGraphTests(unittest.TestCase):
    def test_btcv_coarse_graph_preserves_full_source_aorta_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "case_id": "source",
                "nodes": [
                    {**_node("aorta_inlet", "arterial_inlet"), "position_mm": [0.0, 0.0, 100.0]},
                    {**_node("descending_aorta_mid"), "position_mm": [0.0, 0.0, 80.0]},
                    {**_node("renal_branch_origin"), "position_mm": [0.0, 0.0, 40.0]},
                    {**_node("aorta_distal_anchor"), "position_mm": [0.0, 0.0, 0.0]},
                    _node("ivc_lower_return_inlet", "venous_inlet"),
                    _node("ivc_bifurcation_return"),
                ],
                "edges": [
                    {
                        **_edge("aorta_inlet_to_descending", "aorta_inlet", "descending_aorta_mid", "arterial"),
                        "polyline_mm": [[0.0, 0.0, 100.0], [0.0, 0.0, 80.0]],
                        "length_mm": 20.0,
                    },
                    {
                        **_edge("descending_to_renal", "descending_aorta_mid", "renal_branch_origin", "arterial"),
                        "flow_role": "aorta_trunk",
                        "polyline_mm": [[0.0, 0.0, 80.0], [0.0, 0.0, 40.0]],
                        "length_mm": 40.0,
                    },
                    {
                        **_edge("renal_to_distal", "renal_branch_origin", "aorta_distal_anchor", "arterial"),
                        "flow_role": "aorta_trunk",
                        "polyline_mm": [[0.0, 0.0, 40.0], [0.0, 0.0, 0.0]],
                        "length_mm": 40.0,
                    },
                    _edge("ivc_lower_to_bifurcation_return", "ivc_lower_return_inlet", "ivc_bifurcation_return", "venous"),
                ],
                "graph_metadata": {"edge_count": 4, "node_count": 6},
            }
            source = root / "graph.yaml"
            source.write_text(yaml.safe_dump(graph))

            result = build_btcv_coarse_vessel_graph(
                graph_yaml_path=source,
                output_dir=root / "coarse",
                case_id="coarse_case",
                report_path=root / "coarse.md",
            )

            output = yaml.safe_load(Path(result.graph_yaml_path).read_text())
            aorta_edge = next(edge for edge in output["edges"] if edge["id"] == "coarse_aorta_trunk")
            aorta_outlet = next(node for node in output["nodes"] if node["id"] == "descending_aorta_mid")
            self.assertEqual(aorta_edge["length_mm"], 100.0)
            self.assertEqual(aorta_outlet["position_mm"], [0.0, 0.0, 0.0])
            self.assertIn("full_source_aorta_trunk_chain_preserved_for_coarse_mode", aorta_edge["notes"])
            self.assertEqual(
                output["graph_metadata"]["coarse_aorta_source_chain_edge_ids"],
                ["aorta_inlet_to_descending", "descending_to_renal", "renal_to_distal"],
            )

    def test_btcv_coarse_graph_prunes_unsupported_branches_and_sets_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "case_id": "source",
                "nodes": [
                    _node("aorta_inlet", "arterial_inlet"),
                    _node("descending_aorta_mid"),
                    _node("ivc_lower_return_inlet", "venous_inlet"),
                    _node("ivc_bifurcation_return"),
                    _node("left_renal_outlet", "arterial_outlet"),
                ],
                "edges": [
                    _edge("aorta_inlet_to_descending", "aorta_inlet", "descending_aorta_mid", "arterial"),
                    _edge("ivc_lower_to_bifurcation_return", "ivc_lower_return_inlet", "ivc_bifurcation_return", "venous"),
                    _edge("renal_origin_to_left_renal", "aorta_inlet", "left_renal_outlet", "arterial"),
                ],
                "graph_metadata": {"edge_count": 3, "node_count": 5},
            }
            source = root / "graph.yaml"
            source.write_text(yaml.safe_dump(graph))

            result = build_btcv_coarse_vessel_graph(
                graph_yaml_path=source,
                output_dir=root / "coarse",
                case_id="coarse_case",
                report_path=root / "coarse.md",
            )

            output = yaml.safe_load(Path(result.graph_yaml_path).read_text())
            edge_ids = {edge["id"] for edge in output["edges"]}
            boundaries = {node["id"]: node.get("boundary_role") for node in output["nodes"]}
            self.assertEqual(edge_ids, {"coarse_aorta_trunk", "coarse_ivc_return"})
            self.assertEqual(boundaries["aorta_inlet"], "arterial_inlet")
            self.assertEqual(boundaries["descending_aorta_mid"], "arterial_outlet")
            self.assertEqual(boundaries["ivc_lower_return_inlet"], "venous_inlet")
            self.assertEqual(boundaries["ivc_bifurcation_return"], "venous_outlet")
            self.assertEqual(output["graph_metadata"]["expected_lumen_components"], 2)
            self.assertEqual(result.dropped_edge_count, 1)
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
