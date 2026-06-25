from pathlib import Path
import csv
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vessel_anatomy_correction import correct_vessel_bone_conflicts


class VesselAnatomyCorrectionTests(unittest.TestCase):
    def test_correction_reduces_edge_centerline_bone_fraction_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = np.zeros((48, 48, 24), dtype=np.int16)
            labels[4:44, 4:44, 4:22] = 4
            labels[20:30, 18:24, 10:16] = 10
            labels_path = tmp_path / "labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), str(labels_path))

            graph = {
                "case_id": "tiny_bone_conflict",
                "graph_metadata": {},
                "nodes": [
                    {"id": "source_node", "position_mm": [12.0, 20.0, 13.0], "radius_mm": 2.0, "boundary_role": "arterial_inlet"},
                    {"id": "target_node", "position_mm": [38.0, 20.0, 13.0], "radius_mm": 2.0, "boundary_role": "arterial_outlet"},
                ],
                "edges": [
                    {
                        "id": "renal_origin_to_left_renal",
                        "source": "source_node",
                        "target": "target_node",
                        "vessel_type": "arterial",
                        "flow_role": "renal_branch",
                        "radius_start_mm": 2.0,
                        "radius_end_mm": 2.0,
                        "length_mm": 26.0,
                        "polyline_mm": [[12.0, 20.0, 13.0], [24.0, 20.0, 13.0], [38.0, 20.0, 13.0]],
                    }
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            metrics_path = tmp_path / "edge_metrics.csv"
            with metrics_path.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["edge_id", "status", "inside_bone_fraction"])
                writer.writeheader()
                writer.writerow({"edge_id": "renal_origin_to_left_renal", "status": "review", "inside_bone_fraction": "0.5"})

            result = correct_vessel_bone_conflicts(
                graph_yaml_path=graph_path,
                anatomy_labels_path=labels_path,
                edge_metrics_csv_path=metrics_path,
                output_dir=tmp_path / "out",
                case_id="tiny_bone_corrected",
                clearance_mm=5.0,
                max_node_shift_mm=8.0,
                max_point_shift_mm=12.0,
                smooth_iterations=0,
                report_path=tmp_path / "report.md",
            )

            self.assertEqual(result.corrected_edge_count, 1)
            corrected = yaml.safe_load(Path(result.corrected_graph_yaml_path).read_text())
            self.assertEqual(corrected["edges"][0]["id"], "renal_origin_to_left_renal")
            self.assertEqual(corrected["edges"][0]["source"], "source_node")
            self.assertEqual(corrected["edges"][0]["target"], "target_node")
            self.assertIn("polyline_adjusted_by_organ_aware_bone_conflict_correction", corrected["edges"][0]["notes"])

            with Path(result.edge_corrections_csv_path).open(newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))
            self.assertEqual(rows[0]["corrected"], "1")
            self.assertLess(float(rows[0]["bone_fraction_after"]), float(rows[0]["bone_fraction_before"]))
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertIn("Organ-Aware Vascular Bone-Conflict Correction", Path(result.report_path).read_text())


if __name__ == "__main__":
    unittest.main()
