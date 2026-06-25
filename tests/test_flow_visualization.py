from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.flow_visualization import build_4d_flow_visualization


class Flow4DVisualizationTests(unittest.TestCase):
    def test_4d_visualization_exports_frames_and_manifest(self) -> None:
        graph = {
            "nodes": [
                {"id": "aorta_inlet", "label": "Aorta inlet", "role": "inlet", "position_mm": [0.0, 0.0, 0.0], "boundary_role": "arterial_inlet"},
                {"id": "outlet", "label": "Outlet", "role": "outlet", "position_mm": [20.0, 0.0, 0.0], "boundary_role": "arterial_outlet"},
            ],
            "edges": [
                {
                    "id": "aorta_to_outlet",
                    "source": "aorta_inlet",
                    "target": "outlet",
                    "vessel_type": "arterial",
                    "radius_start_mm": 5.0,
                    "radius_end_mm": 4.0,
                    "polyline_mm": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            graph_path = temp_path / "graph.yaml"
            edge_csv = temp_path / "edges.csv"
            node_csv = temp_path / "nodes.csv"
            graph_path.write_text(yaml.safe_dump(graph))
            with edge_csv.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "time_s",
                        "phase",
                        "edge_id",
                        "vessel_type",
                        "flow_role",
                        "flow_ml_s",
                        "mean_velocity_cm_s",
                        "pressure_source_pa",
                        "pressure_target_pa",
                        "pressure_drop_pa",
                        "pressure_source_mmhg",
                        "pressure_target_mmhg",
                        "pressure_equation_residual_pa",
                    ]
                )
                writer.writerow(["0.0", "0.0", "aorta_to_outlet", "arterial", "trunk", "10", "20", "13000", "12900", "100", "97.5", "96.8", "0"])
                writer.writerow(["0.5", "0.5", "aorta_to_outlet", "arterial", "trunk", "16", "32", "13200", "13080", "120", "99.0", "98.1", "0"])
            with node_csv.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "time_s",
                        "phase",
                        "node_id",
                        "vessel_type",
                        "role",
                        "boundary_role",
                        "pressure_pa",
                        "pressure_mmhg",
                        "boundary_flow_ml_s",
                        "incoming_edge_flow_ml_s",
                        "outgoing_edge_flow_ml_s",
                        "mass_balance_residual_ml_s",
                    ]
                )
                writer.writerow(["0.0", "0.0", "aorta_inlet", "arterial", "inlet", "arterial_inlet", "13000", "97.5", "10", "0", "10", "0"])
                writer.writerow(["0.0", "0.0", "outlet", "arterial", "outlet", "arterial_outlet", "12900", "96.8", "-10", "10", "0", "0"])
                writer.writerow(["0.5", "0.5", "aorta_inlet", "arterial", "inlet", "arterial_inlet", "13200", "99.0", "16", "0", "16", "0"])
                writer.writerow(["0.5", "0.5", "outlet", "arterial", "outlet", "arterial_outlet", "13080", "98.1", "-16", "16", "0", "0"])

            result = build_4d_flow_visualization(
                graph_yaml_path=graph_path,
                edge_timeseries_csv_path=edge_csv,
                node_timeseries_csv_path=node_csv,
                output_dir=temp_path / "flow4d",
                case_id="toy",
                frame_count=2,
                context_scene_spec_path=None,
                label_boundary_nodes=False,
                color_by="velocity",
                report_path=temp_path / "report.md",
            )

            self.assertTrue(Path(result.frame_manifest_csv_path).exists())
            self.assertTrue(Path(result.contact_sheet_png_path).exists())
            self.assertTrue(Path(result.spec_yaml_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertEqual(len(list(Path(result.frame_dir).glob("*.png"))), 2)
            self.assertEqual(result.frame_count, 2)


if __name__ == "__main__":
    unittest.main()
