from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.labeled_vessel_graph import build_labeled_vessel_vascular_graph, build_registered_labeled_vessel_vascular_graph


class LabeledVesselGraphTests(unittest.TestCase):
    def test_replaces_edge_from_labeled_vessel_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mask = np.zeros((32, 32, 8), dtype=np.uint8)
            mask[14:18, 14:18, :] = 4
            for x in range(12, 22):
                mask[x, 23:26, 3:5] = 2
            mask_path = tmp_path / "labels.nii.gz"
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

            label_config = {
                "dataset": "tiny_labeled_vessels",
                "labels": {2: "Left common iliac artery", 4: "Aorta"},
                "graph_edge_mapping": {
                    "bifurcation_to_left_common_iliac": {
                        "labels": [2],
                        "replacement_role": "left_common_iliac_artery",
                    }
                },
            }
            config_path = tmp_path / "labels.yaml"
            config_path.write_text(yaml.safe_dump(label_config, sort_keys=False))

            graph = {
                "case_id": "tiny",
                "graph_metadata": {},
                "nodes": [
                    {"id": "aortic_bifurcation", "position_mm": [10.0, 10.0, 0.0], "radius_mm": 4.0},
                    {"id": "left_common_iliac_outlet", "position_mm": [25.0, 15.0, 0.0], "radius_mm": 3.0},
                ],
                "edges": [
                    {
                        "id": "bifurcation_to_left_common_iliac",
                        "source": "aortic_bifurcation",
                        "target": "left_common_iliac_outlet",
                        "vessel_type": "arterial",
                        "flow_role": "iliac_branch",
                        "radius_start_mm": 4.0,
                        "radius_end_mm": 3.0,
                        "length_mm": 15.0,
                        "polyline_mm": [[10.0, 10.0, 0.0], [25.0, 15.0, 0.0]],
                    }
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            result = build_labeled_vessel_vascular_graph(
                baseline_graph_path=graph_path,
                labeled_mask_path=mask_path,
                label_config_path=config_path,
                output_dir=tmp_path / "out",
                case_id="tiny_replaced",
                report_path=tmp_path / "report.md",
            )

            self.assertEqual(result.successful_replacements, 1)
            derived = yaml.safe_load(Path(result.graph_yaml_path).read_text())
            edge = derived["edges"][0]
            self.assertIn("polyline_replaced_from_labeled_vessel_template", edge["notes"])
            self.assertGreater(len(edge["polyline_mm"]), 2)
            self.assertIn("replaced_from_labeled_vessel_template", Path(result.replacements_csv_path).read_text())

    def test_registered_builder_applies_local_deformable_landmark_warp(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mask = np.zeros((32, 32, 16), dtype=np.uint8)
            mask[15:18, 15:18, 2:15] = 4
            for offset, x in enumerate(range(10, 17)):
                mask[x, 20 + offset // 3 : 23 + offset // 3, 3:5] = 2
            mask_path = tmp_path / "labels.nii.gz"
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

            label_config = {
                "dataset": "tiny_registered_labeled_vessels",
                "labels": {2: "Left common iliac artery", 4: "Aorta"},
                "graph_edge_mapping": {
                    "bifurcation_to_left_common_iliac": {
                        "labels": [2],
                        "replacement_role": "left_common_iliac_artery",
                    }
                },
            }
            config_path = tmp_path / "labels.yaml"
            config_path.write_text(yaml.safe_dump(label_config, sort_keys=False))

            graph = {
                "case_id": "tiny_registered",
                "graph_metadata": {},
                "nodes": [
                    {"id": "aorta_inlet", "position_mm": [31.0, 41.0, 30.0], "radius_mm": 4.0},
                    {"id": "descending_aorta_mid", "position_mm": [29.0, 42.0, 24.0], "radius_mm": 4.0},
                    {"id": "aorta_distal_anchor", "position_mm": [27.0, 45.0, 12.0], "radius_mm": 4.0},
                    {"id": "aortic_bifurcation", "position_mm": [25.0, 48.0, 10.0], "radius_mm": 3.0},
                    {"id": "left_common_iliac_outlet", "position_mm": [14.0, 57.0, 9.0], "radius_mm": 2.5},
                ],
                "edges": [
                    {
                        "id": "bifurcation_to_left_common_iliac",
                        "source": "aortic_bifurcation",
                        "target": "left_common_iliac_outlet",
                        "vessel_type": "arterial",
                        "flow_role": "iliac_branch",
                        "radius_start_mm": 3.0,
                        "radius_end_mm": 2.5,
                        "length_mm": 15.0,
                        "polyline_mm": [[25.0, 48.0, 10.0], [14.0, 57.0, 9.0]],
                    }
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            result = build_registered_labeled_vessel_vascular_graph(
                target_graph_path=graph_path,
                labeled_mask_path=mask_path,
                label_config_path=config_path,
                output_dir=tmp_path / "out",
                case_id="tiny_registered_replaced",
                report_path=tmp_path / "registered_report.md",
            )

            self.assertEqual(result.successful_replacements, 1)
            self.assertLessEqual(result.deformable_registration_rms_error_mm, 1e-6)
            derived = yaml.safe_load(Path(result.graph_yaml_path).read_text())
            edge = derived["edges"][0]
            self.assertIn("polyline_replaced_from_registered_labeled_vessel_centerline", edge["notes"])
            self.assertIn("registration_model=landmark_affine_plus_local_residual_warp_with_endpoint_snap", edge["notes"])
            landmarks_csv = Path(result.landmarks_csv_path).read_text()
            self.assertIn("deformed_x_mm", landmarks_csv)
            self.assertIn("deformable_error_mm", landmarks_csv)


if __name__ == "__main__":
    unittest.main()
