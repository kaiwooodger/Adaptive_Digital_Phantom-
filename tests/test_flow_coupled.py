from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.flow_1d import build_flow_1d_model
from phantom_twin.flow_coupled import build_coupled_pulsatile_flow_model


class CoupledPulsatileFlowModelTests(unittest.TestCase):
    def test_coupled_model_exports_dynamic_outlet_splits_for_branch(self) -> None:
        graph = {
            "nodes": [
                {"id": "aorta_inlet", "label": "Aorta inlet", "kind": "arterial", "role": "inlet", "boundary_role": "arterial_inlet"},
                {"id": "branch", "label": "Branch", "kind": "arterial", "role": "junction"},
                {"id": "left_outlet", "label": "Left outlet", "kind": "arterial", "role": "outlet", "boundary_role": "arterial_outlet"},
                {"id": "right_outlet", "label": "Right outlet", "kind": "arterial", "role": "outlet", "boundary_role": "arterial_outlet"},
            ],
            "edges": [
                {
                    "id": "aorta_to_branch",
                    "label": "Aorta to branch",
                    "source": "aorta_inlet",
                    "target": "branch",
                    "vessel_type": "arterial",
                    "flow_role": "trunk",
                    "radius_start_mm": 6.0,
                    "radius_end_mm": 5.0,
                    "polyline_mm": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                },
                {
                    "id": "branch_to_left",
                    "label": "Branch to left",
                    "source": "branch",
                    "target": "left_outlet",
                    "vessel_type": "arterial",
                    "flow_role": "outlet",
                    "radius_start_mm": 5.0,
                    "radius_end_mm": 4.0,
                    "polyline_mm": [[20.0, 0.0, 0.0], [40.0, 8.0, 0.0]],
                },
                {
                    "id": "branch_to_right",
                    "label": "Branch to right",
                    "source": "branch",
                    "target": "right_outlet",
                    "vessel_type": "arterial",
                    "flow_role": "outlet",
                    "radius_start_mm": 5.0,
                    "radius_end_mm": 3.0,
                    "polyline_mm": [[20.0, 0.0, 0.0], [40.0, -8.0, 0.0]],
                },
            ],
        }
        boundaries = {
            "global_placeholders": {"venous_outlet_pressure_pa": 667.0},
            "waveforms": {
                "arterial_inlet_unit_cycle": [
                    {"phase": 0.0, "normalized_flow_multiplier": 0.7},
                    {"phase": 0.25, "normalized_flow_multiplier": 1.7},
                    {"phase": 0.5, "normalized_flow_multiplier": 0.9},
                    {"phase": 0.75, "normalized_flow_multiplier": 0.7},
                ]
            },
            "boundaries": [
                {"boundary_id": 1, "node_id": "aorta_inlet", "label": "Aorta inlet", "role": "arterial_inlet", "assigned_flow_ml_s": 12.0},
                {
                    "boundary_id": 2,
                    "node_id": "left_outlet",
                    "label": "Left outlet",
                    "role": "arterial_outlet",
                    "assigned_flow_ml_s": -7.0,
                    "resistance_pa_s_per_m3": 700000000.0,
                },
                {
                    "boundary_id": 3,
                    "node_id": "right_outlet",
                    "label": "Right outlet",
                    "role": "arterial_outlet",
                    "assigned_flow_ml_s": -5.0,
                    "resistance_pa_s_per_m3": 1000000000.0,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            graph_path = temp_path / "graph.yaml"
            boundary_path = temp_path / "boundaries.yaml"
            graph_path.write_text(yaml.safe_dump(graph))
            boundary_path.write_text(yaml.safe_dump(boundaries))
            steady = build_flow_1d_model(
                graph_yaml_path=graph_path,
                boundary_config_path=boundary_path,
                output_dir=temp_path / "steady",
                report_path=temp_path / "steady.md",
                case_id="toy_coupled",
            )
            coupled = build_coupled_pulsatile_flow_model(
                flow_1d_model_path=steady.model_yaml_path,
                boundary_config_path=boundary_path,
                output_dir=temp_path / "coupled",
                report_path=temp_path / "coupled.md",
                case_id="toy_coupled",
                samples_per_cycle=24,
                settling_cycles=1,
            )

            self.assertTrue(Path(coupled.edge_timeseries_csv_path).exists())
            self.assertTrue(Path(coupled.outlet_split_csv_path).exists())
            self.assertTrue(Path(coupled.qa_timeseries_csv_path).exists())
            for plot_path in coupled.plot_paths:
                self.assertTrue(Path(plot_path).exists())

            with Path(coupled.outlet_split_csv_path).open(newline="") as csvfile:
                split_rows = list(csv.DictReader(csvfile))

        self.assertEqual(coupled.terminal_rcr_count, 2)
        self.assertAlmostEqual(coupled.arterial_inlet_flow_mean_ml_s, 12.0, places=6)
        self.assertLess(coupled.max_abs_mass_balance_residual_ml_s, 1e-6)
        self.assertGreater(coupled.max_outlet_split_range_percentage_points, 0.01)
        self.assertTrue(any(row["node_id"] == "left_outlet" for row in split_rows))


if __name__ == "__main__":
    unittest.main()
