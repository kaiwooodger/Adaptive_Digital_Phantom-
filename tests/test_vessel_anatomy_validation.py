from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vessel_anatomy_validation import validate_vessel_organ_anatomy


class VesselAnatomyValidationTests(unittest.TestCase):
    def test_left_renal_edge_passes_when_it_reaches_left_kidney_and_avoids_bone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = np.zeros((64, 64, 32), dtype=np.int16)
            labels[5:58, 5:58, 4:30] = 4
            labels[10:18, 28:36, 14:20] = 7
            labels[38:50, 26:42, 18:26] = 6
            labels[26:34, 18:26, 10:24] = 10
            label_path = tmp_path / "labels.nii.gz"
            reference = nib.Nifti1Image(labels, np.eye(4))
            nib.save(reference, str(label_path))

            arterial = np.zeros(labels.shape, dtype=np.uint8)
            arterial[14:23, 29:32, 15:18] = 1
            venous = np.zeros(labels.shape, dtype=np.uint8)
            wall = np.zeros(labels.shape, dtype=np.uint8)
            wall[13:24, 28:33, 14:19] = 1
            arterial_path = tmp_path / "arterial.nii.gz"
            venous_path = tmp_path / "venous.nii.gz"
            wall_path = tmp_path / "wall.nii.gz"
            nib.save(nib.Nifti1Image(arterial, np.eye(4)), str(arterial_path))
            nib.save(nib.Nifti1Image(venous, np.eye(4)), str(venous_path))
            nib.save(nib.Nifti1Image(wall, np.eye(4)), str(wall_path))

            graph = {
                "case_id": "tiny_vessel_anatomy",
                "nodes": [
                    {"id": "renal_branch_origin", "position_mm": [22.0, 30.0, 16.0], "radius_mm": 3.0},
                    {"id": "left_renal_outlet", "position_mm": [14.0, 31.0, 16.0], "radius_mm": 2.0},
                ],
                "edges": [
                    {
                        "id": "renal_origin_to_left_renal",
                        "source": "renal_branch_origin",
                        "target": "left_renal_outlet",
                        "vessel_type": "arterial",
                        "flow_role": "renal_branch",
                        "length_mm": 8.1,
                        "polyline_mm": [[22.0, 30.0, 16.0], [18.0, 31.0, 16.0], [14.0, 31.0, 16.0]],
                    }
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            spec = {
                "case_id": "tiny_vessel_anatomy",
                "voxelization": {
                    "source_graph": str(graph_path),
                    "source_combined_labels": str(label_path),
                },
                "outputs": {
                    "arterial_lumen_mask": str(arterial_path),
                    "venous_lumen_mask": str(venous_path),
                    "vessel_wall_mask": str(wall_path),
                },
            }
            spec_path = tmp_path / "voxelized_spec.yaml"
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))

            result = validate_vessel_organ_anatomy(
                voxelized_spec_path=spec_path,
                output_dir=tmp_path / "out",
                case_id="tiny_vessel_anatomy",
                sample_step_mm=1.0,
                report_path=tmp_path / "report.md",
            )

            self.assertEqual(result.edge_count, 1)
            self.assertEqual(result.pass_count, 1)
            edge_csv = Path(result.edge_metrics_csv_path).read_text()
            self.assertIn("renal_origin_to_left_renal", edge_csv)
            self.assertIn("left_kidney", edge_csv)
            self.assertIn("pass", edge_csv)
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertIn("Organ-Aware Vascular Anatomy Validation", Path(result.report_path).read_text())


if __name__ == "__main__":
    unittest.main()
