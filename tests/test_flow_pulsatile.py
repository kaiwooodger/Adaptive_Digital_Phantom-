from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.flow_1d import build_flow_1d_model
from phantom_twin.flow_pulsatile import build_pulsatile_flow_model


class PulsatileFlowModelTests(unittest.TestCase):
    def test_pulsatile_model_exports_timeseries_and_balances_single_outlet(self) -> None:
        graph = {
            "nodes": [
                {"id": "aorta_inlet", "label": "Aorta inlet", "kind": "arterial", "role": "inlet", "boundary_role": "arterial_inlet"},
                {"id": "outlet", "label": "Outlet", "kind": "arterial", "role": "outlet", "boundary_role": "arterial_outlet"},
            ],
            "edges": [
                {
                    "id": "aorta_inlet_to_outlet",
                    "label": "Aorta inlet to outlet",
                    "source": "aorta_inlet",
                    "target": "outlet",
                    "vessel_type": "arterial",
                    "flow_role": "trunk",
                    "radius_start_mm": 5.0,
                    "radius_end_mm": 4.0,
                    "polyline_mm": [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
                }
            ],
        }
        boundaries = {
            "global_placeholders": {
                "venous_outlet_pressure_pa": 667.0,
                "nominal_arterial_outlet_pressure_drop_pa": 8000.0,
            },
            "waveforms": {
                "arterial_inlet_unit_cycle": [
                    {"phase": 0.0, "normalized_flow_multiplier": 0.7},
                    {"phase": 0.25, "normalized_flow_multiplier": 1.6},
                    {"phase": 0.5, "normalized_flow_multiplier": 0.9},
                    {"phase": 0.75, "normalized_flow_multiplier": 0.8},
                ]
            },
            "boundaries": [
                {"boundary_id": 1, "node_id": "aorta_inlet", "label": "Aorta inlet", "role": "arterial_inlet", "assigned_flow_ml_s": 10.0},
                {
                    "boundary_id": 2,
                    "node_id": "outlet",
                    "label": "Outlet",
                    "role": "arterial_outlet",
                    "assigned_flow_ml_s": -10.0,
                    "resistance_pa_s_per_m3": 800000000.0,
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
                case_id="toy",
            )
            pulsatile = build_pulsatile_flow_model(
                flow_1d_model_path=steady.model_yaml_path,
                boundary_config_path=boundary_path,
                output_dir=temp_path / "pulsatile",
                report_path=temp_path / "pulsatile.md",
                case_id="toy",
                samples_per_cycle=16,
                settling_cycles=1,
            )

            self.assertTrue(Path(pulsatile.edge_timeseries_csv_path).exists())
            self.assertTrue(Path(pulsatile.node_timeseries_csv_path).exists())
            self.assertTrue(Path(pulsatile.boundary_timeseries_csv_path).exists())
            self.assertTrue(Path(pulsatile.model_yaml_path).exists())
            for plot_path in pulsatile.plot_paths:
                self.assertTrue(Path(plot_path).exists())

        self.assertEqual(pulsatile.terminal_rcr_count, 1)
        self.assertAlmostEqual(pulsatile.arterial_inlet_flow_mean_ml_s, 10.0, places=6)
        self.assertLess(pulsatile.max_abs_mass_balance_residual_ml_s, 1e-6)


if __name__ == "__main__":
    unittest.main()
