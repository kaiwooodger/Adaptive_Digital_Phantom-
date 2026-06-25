from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vessel_anatomy_validation import _edge_samples, _mask_for_group, _points_to_indices, _sample_mask_fraction
from phantom_twin.vessel_edge_reroute import reroute_vessel_edge_around_bone


class VesselEdgeRerouteTests(unittest.TestCase):
    def test_targeted_edge_reroute_reduces_bone_fraction_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = np.zeros((80, 80, 40), dtype=np.int16)
            labels[4:76, 4:76, 4:36] = 4
            labels[35:45, 30:50, 18:22] = 10
            labels_path = tmp_path / "labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), str(labels_path))

            graph = {
                "case_id": "tiny_reroute",
                "graph_metadata": {},
                "nodes": [
                    {"id": "source_node", "position_mm": [20.0, 40.0, 20.0], "radius_mm": 2.0},
                    {"id": "target_node", "position_mm": [60.0, 40.0, 20.0], "radius_mm": 2.0},
                ],
                "edges": [
                    {
                        "id": "visceral_to_renal_origin",
                        "source": "source_node",
                        "target": "target_node",
                        "vessel_type": "arterial",
                        "flow_role": "aorta_trunk",
                        "radius_start_mm": 2.0,
                        "radius_end_mm": 3.0,
                        "length_mm": 40.0,
                        "polyline_mm": [[20.0, 40.0, 20.0], [40.0, 40.0, 20.0], [60.0, 40.0, 20.0]],
                    }
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            result = reroute_vessel_edge_around_bone(
                graph_yaml_path=graph_path,
                anatomy_labels_path=labels_path,
                edge_id="visceral_to_renal_origin",
                output_dir=tmp_path / "out",
                case_id="tiny_reroute_corrected",
                clearance_mm=4.0,
                max_detour_mm=35.0,
                detour_step_mm=5.0,
                sample_step_mm=1.0,
                resample_step_mm=2.0,
                max_point_shift_mm=8.0,
                smooth_iterations=1,
                report_path=tmp_path / "report.md",
            )

            corrected = yaml.safe_load(Path(result.corrected_graph_yaml_path).read_text())
            corrected_edge = corrected["edges"][0]
            self.assertEqual(corrected_edge["id"], "visceral_to_renal_origin")
            self.assertEqual(corrected_edge["source"], "source_node")
            self.assertEqual(corrected_edge["target"], "target_node")
            self.assertEqual(corrected_edge["vessel_type"], "arterial")
            self.assertEqual(corrected_edge["flow_role"], "aorta_trunk")
            self.assertEqual(corrected_edge["radius_start_mm"], 2.0)
            self.assertEqual(corrected_edge["radius_end_mm"], 3.0)
            self.assertLess(result.bone_fraction_after, result.bone_fraction_before)
            self.assertLessEqual(result.outside_body_fraction_after, 0.02)
            self.assertIn("polyline_rerouted_by_targeted_bone_avoidance", corrected_edge["notes"])
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertIn("Targeted Vessel Edge Reroute", Path(result.report_path).read_text())

            bone = _mask_for_group(labels, "bone")
            samples = _edge_samples(corrected_edge, 1.0)
            indices, valid = _points_to_indices(samples, (1.0, 1.0, 1.0), labels.shape)
            self.assertLess(_sample_mask_fraction(bone, indices, valid), 0.05)


if __name__ == "__main__":
    unittest.main()
