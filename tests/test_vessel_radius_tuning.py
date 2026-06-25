from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vessel_radius_tuning import tune_vessel_radii_against_bone
from phantom_twin.vessel_radius_validation import validate_vessel_radius_anatomy


class VesselRadiusTuningTests(unittest.TestCase):
    def test_radius_tuning_adds_profile_and_reduces_tube_bone_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = np.zeros((72, 72, 36), dtype=np.int16)
            labels[4:68, 4:68, 4:32] = 4
            labels[24:48, 32:36, 15:23] = 10
            labels_path = tmp_path / "labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), str(labels_path))

            graph = {
                "case_id": "tiny_radius_tuning",
                "graph_metadata": {},
                "nodes": [
                    {"id": "source_node", "position_mm": [10.0, 36.0, 19.0], "radius_mm": 8.0},
                    {"id": "target_node", "position_mm": [62.0, 36.0, 19.0], "radius_mm": 8.0},
                ],
                "edges": [
                    {
                        "id": "descending_to_visceral_origin",
                        "source": "source_node",
                        "target": "target_node",
                        "vessel_type": "arterial",
                        "flow_role": "aorta_trunk",
                        "radius_start_mm": 8.0,
                        "radius_end_mm": 8.0,
                        "length_mm": 52.0,
                        "polyline_mm": [[10.0, 36.0, 19.0], [36.0, 36.0, 19.0], [62.0, 36.0, 19.0]],
                    }
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))
            spec_path = tmp_path / "spec.yaml"
            spec_path.write_text(
                yaml.safe_dump(
                    {
                        "voxelization": {
                            "source_graph": str(graph_path),
                            "source_combined_labels": str(labels_path),
                        }
                    },
                    sort_keys=False,
                )
            )

            before = validate_vessel_radius_anatomy(
                voxelized_spec_path=spec_path,
                graph_yaml_path=graph_path,
                anatomy_labels_path=labels_path,
                output_dir=tmp_path / "before",
                case_id="before",
                sample_step_mm=1.0,
                review_lumen_bone_fraction=0.05,
                report_path=tmp_path / "before.md",
            )
            self.assertEqual(before.review_count, 1)

            result = tune_vessel_radii_against_bone(
                graph_yaml_path=graph_path,
                anatomy_labels_path=labels_path,
                output_dir=tmp_path / "tuned",
                case_id="tiny_radius_tuned",
                tune_review_edges_only=False,
                bone_clearance_mm=0.5,
                sample_step_mm=1.0,
                max_profile_points=16,
                smooth_iterations=1,
                arterial_trunk_min_radius_mm=2.0,
                arterial_trunk_max_radius_mm=8.0,
                report_path=tmp_path / "tuning.md",
            )

            self.assertEqual(result.tuned_edge_count, 1)
            tuned = yaml.safe_load(Path(result.tuned_graph_yaml_path).read_text())
            edge = tuned["edges"][0]
            self.assertEqual(edge["source"], "source_node")
            self.assertEqual(edge["target"], "target_node")
            self.assertEqual(edge["flow_role"], "aorta_trunk")
            self.assertIn("radius_profile", edge)
            self.assertLess(min(item["radius_mm"] for item in edge["radius_profile"]), 8.0)
            self.assertIn("radius_profile_tuned_by_anatomy_aware_bone_clearance", edge["notes"])

            tuned_spec_path = tmp_path / "tuned_spec.yaml"
            tuned_spec_path.write_text(
                yaml.safe_dump(
                    {
                        "voxelization": {
                            "source_graph": result.tuned_graph_yaml_path,
                            "source_combined_labels": str(labels_path),
                        }
                    },
                    sort_keys=False,
                )
            )
            after = validate_vessel_radius_anatomy(
                voxelized_spec_path=tuned_spec_path,
                graph_yaml_path=result.tuned_graph_yaml_path,
                anatomy_labels_path=labels_path,
                output_dir=tmp_path / "after",
                case_id="after",
                sample_step_mm=1.0,
                review_lumen_bone_fraction=0.05,
                report_path=tmp_path / "after.md",
            )
            self.assertEqual(after.review_count, 0)
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertIn("Anatomy-Aware Vessel Radius Tuning", Path(result.report_path).read_text())


if __name__ == "__main__":
    unittest.main()
