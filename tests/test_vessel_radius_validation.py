from pathlib import Path
import csv
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vessel_radius_validation import validate_vessel_radius_anatomy


class VesselRadiusValidationTests(unittest.TestCase):
    def test_radius_validation_flags_tube_overlap_when_centerline_is_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = np.zeros((48, 48, 24), dtype=np.int16)
            labels[4:44, 4:44, 4:22] = 4
            labels[20:28, 20:28, 10:16] = 10
            labels_path = tmp_path / "labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), str(labels_path))

            graph = {
                "case_id": "tiny_radius_overlap",
                "nodes": [
                    {"id": "source_node", "position_mm": [12.0, 30.0, 13.0], "radius_mm": 4.0},
                    {"id": "target_node", "position_mm": [38.0, 30.0, 13.0], "radius_mm": 4.0},
                ],
                "edges": [
                    {
                        "id": "renal_origin_to_left_renal",
                        "source": "source_node",
                        "target": "target_node",
                        "vessel_type": "arterial",
                        "flow_role": "renal_branch",
                        "radius_start_mm": 4.0,
                        "radius_end_mm": 4.0,
                        "length_mm": 26.0,
                        "polyline_mm": [[12.0, 30.0, 13.0], [38.0, 30.0, 13.0]],
                    }
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))
            spec = {
                "case_id": "tiny_radius_overlap",
                "voxelization": {
                    "source_graph": str(graph_path),
                    "source_combined_labels": str(labels_path),
                },
                "outputs": {},
            }
            spec_path = tmp_path / "spec.yaml"
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))

            result = validate_vessel_radius_anatomy(
                voxelized_spec_path=spec_path,
                output_dir=tmp_path / "out",
                case_id="tiny_radius_overlap",
                sample_step_mm=1.0,
                scaled_radius_factor=0.5,
                review_lumen_bone_fraction=0.01,
                fail_lumen_bone_fraction=0.5,
                report_path=tmp_path / "report.md",
            )

            self.assertEqual(result.edge_count, 1)
            self.assertEqual(result.review_count, 1)
            with Path(result.edge_metrics_csv_path).open(newline="") as csvfile:
                row = next(csv.DictReader(csvfile))
            self.assertEqual(float(row["centerline_bone_fraction"]), 0.0)
            self.assertGreater(float(row["lumen_bone_fraction"]), 0.0)
            self.assertLess(float(row["scaled_lumen_bone_fraction"]), float(row["lumen_bone_fraction"]))
            self.assertIn("radius_tuning", row["recommendation"])
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertIn("Vessel Radius-Aware Anatomy Validation", Path(result.report_path).read_text())


if __name__ == "__main__":
    unittest.main()
