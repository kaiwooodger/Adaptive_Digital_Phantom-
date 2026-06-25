from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.flow_1d import build_flow_1d_model


class Flow1DModelTests(unittest.TestCase):
    def test_branch_flow_allocation_balances_signed_boundaries(self) -> None:
        graph = {
            "nodes": [
                {"id": "inlet", "label": "Inlet", "kind": "arterial", "role": "inlet", "boundary_role": "arterial_inlet"},
                {"id": "branch", "label": "Branch", "kind": "arterial", "role": "junction"},
                {"id": "left", "label": "Left outlet", "kind": "arterial", "role": "outlet", "boundary_role": "arterial_outlet"},
                {"id": "right", "label": "Right outlet", "kind": "arterial", "role": "outlet", "boundary_role": "arterial_outlet"},
            ],
            "edges": [
                {
                    "id": "inlet_to_branch",
                    "label": "Inlet to branch",
                    "source": "inlet",
                    "target": "branch",
                    "vessel_type": "arterial",
                    "flow_role": "trunk",
                    "radius_start_mm": 5.0,
                    "radius_end_mm": 5.0,
                    "polyline_mm": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                },
                {
                    "id": "branch_to_left",
                    "label": "Branch to left",
                    "source": "branch",
                    "target": "left",
                    "vessel_type": "arterial",
                    "flow_role": "outlet",
                    "radius_start_mm": 4.0,
                    "radius_end_mm": 3.0,
                    "polyline_mm": [[10.0, 0.0, 0.0], [20.0, 5.0, 0.0]],
                },
                {
                    "id": "branch_to_right",
                    "label": "Branch to right",
                    "source": "branch",
                    "target": "right",
                    "vessel_type": "arterial",
                    "flow_role": "outlet",
                    "radius_start_mm": 4.0,
                    "radius_end_mm": 3.0,
                    "polyline_mm": [[10.0, 0.0, 0.0], [20.0, -5.0, 0.0]],
                },
            ],
        }
        boundaries = {
            "global_placeholders": {"venous_outlet_pressure_pa": 667.0},
            "boundaries": [
                {"node_id": "inlet", "role": "arterial_inlet", "assigned_flow_ml_s": 10.0},
                {"node_id": "left", "role": "arterial_outlet", "assigned_flow_ml_s": -4.0},
                {"node_id": "right", "role": "arterial_outlet", "assigned_flow_ml_s": -6.0},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            graph_path = temp_path / "graph.yaml"
            boundary_path = temp_path / "boundaries.yaml"
            graph_path.write_text(yaml.safe_dump(graph))
            boundary_path.write_text(yaml.safe_dump(boundaries))

            result = build_flow_1d_model(
                graph_yaml_path=graph_path,
                boundary_config_path=boundary_path,
                output_dir=temp_path / "outputs",
                report_path=temp_path / "report.md",
                case_id="toy",
            )

        flows = {segment.edge_id: segment.flow_ml_s for segment in result.segments}
        self.assertAlmostEqual(flows["inlet_to_branch"], 10.0, places=6)
        self.assertAlmostEqual(flows["branch_to_left"], 4.0, places=6)
        self.assertAlmostEqual(flows["branch_to_right"], 6.0, places=6)
        self.assertLess(result.max_abs_mass_balance_residual_ml_s, 1e-6)
        self.assertLess(result.max_abs_pressure_equation_residual_pa, 1e-6)

    def test_venous_outlet_pressure_override_beats_configured_boundary_pressure(self) -> None:
        graph = {
            "nodes": [
                {"id": "return_inlet", "label": "Return inlet", "kind": "venous", "role": "inlet", "boundary_role": "venous_inlet"},
                {"id": "return_outlet", "label": "Return outlet", "kind": "venous", "role": "outlet", "boundary_role": "venous_outlet"},
            ],
            "edges": [
                {
                    "id": "return_path",
                    "label": "Return path",
                    "source": "return_inlet",
                    "target": "return_outlet",
                    "vessel_type": "venous",
                    "flow_role": "return",
                    "radius_start_mm": 4.0,
                    "radius_end_mm": 4.0,
                    "polyline_mm": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                }
            ],
        }
        boundaries = {
            "global_placeholders": {"venous_outlet_pressure_pa": 667.0},
            "boundaries": [
                {"node_id": "return_inlet", "role": "venous_inlet", "assigned_flow_ml_s": 5.0},
                {"node_id": "return_outlet", "role": "venous_outlet", "assigned_flow_ml_s": -5.0, "pressure_pa": 667.0},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            graph_path = temp_path / "graph.yaml"
            boundary_path = temp_path / "boundaries.yaml"
            graph_path.write_text(yaml.safe_dump(graph))
            boundary_path.write_text(yaml.safe_dump(boundaries))

            result = build_flow_1d_model(
                graph_yaml_path=graph_path,
                boundary_config_path=boundary_path,
                output_dir=temp_path / "outputs",
                report_path=temp_path / "report.md",
                case_id="toy_venous",
                venous_outlet_pressure_pa=900.0,
            )

        outlet = next(node for node in result.nodes if node.node_id == "return_outlet")
        self.assertAlmostEqual(outlet.pressure_pa, 900.0, places=6)


if __name__ == "__main__":
    unittest.main()
