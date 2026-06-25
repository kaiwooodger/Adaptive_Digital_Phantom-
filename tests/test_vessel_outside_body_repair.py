from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from phantom_twin.vessel_outside_body_repair import repair_vessel_outside_body_margin


class VesselOutsideBodyRepairTests(unittest.TestCase):
    def test_repair_trims_degree_one_boundary_edge_outside_body(self) -> None:
        import nibabel as nib

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            labels = np.zeros((32, 32, 16), dtype=np.int16)
            labels[8:24, 8:24, 2:14] = 1
            labels_path = temp / "labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), labels_path)

            graph_path = temp / "graph.yaml"
            graph_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy",
                        "nodes": [
                            {
                                "id": "inlet",
                                "position_mm": [16.0, 16.0, 15.0],
                                "radius_mm": 2.0,
                                "boundary_role": "arterial_inlet",
                            },
                            {
                                "id": "junction",
                                "position_mm": [16.0, 16.0, 6.0],
                                "radius_mm": 2.0,
                            },
                        ],
                        "edges": [
                            {
                                "id": "inlet_to_junction",
                                "source": "inlet",
                                "target": "junction",
                                "vessel_type": "arterial",
                                "flow_role": "test_trunk",
                                "radius_start_mm": 2.0,
                                "radius_end_mm": 2.0,
                                "polyline_mm": [[16.0, 16.0, 15.0], [16.0, 16.0, 6.0]],
                            }
                        ],
                    },
                    sort_keys=False,
                )
            )

            result = repair_vessel_outside_body_margin(
                graph_yaml_path=graph_path,
                anatomy_labels_path=labels_path,
                output_dir=temp / "repair",
                case_id="toy_repaired",
                sample_step_mm=1.0,
                body_margin_mm=0.5,
                min_radius_mm=0.25,
                report_path=temp / "report.md",
            )

            self.assertEqual(result.status, "outside_body_margin_repaired")
            self.assertEqual(result.targeted_edge_count, 1)
            self.assertEqual(result.outside_voxels_after, 0)
            self.assertEqual(result.moved_node_count, 1)
            repaired = yaml.safe_load(Path(result.repaired_graph_yaml_path).read_text())
            inlet = next(node for node in repaired["nodes"] if node["id"] == "inlet")
            self.assertLess(inlet["position_mm"][2], 15.0)


if __name__ == "__main__":
    unittest.main()
