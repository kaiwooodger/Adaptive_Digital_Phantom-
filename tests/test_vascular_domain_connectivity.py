from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vascular_domain_connectivity import repair_vascular_domain_connectivity


class VascularDomainConnectivityRepairTests(unittest.TestCase):
    def test_repair_prunes_unseeded_arterial_islands_and_writes_compatible_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = np.zeros((48, 48, 24), dtype=np.int16)
            labels[4:44, 4:44, 4:22] = 4
            labels_path = tmp_path / "labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), str(labels_path))

            arterial = np.zeros(labels.shape, dtype=np.uint8)
            arterial[10:36, 24, 12] = 1
            arterial[3, 3, 3] = 1
            arterial[40:42, 40:42, 18:20] = 1
            arterial_path = tmp_path / "arterial.nii.gz"
            nib.save(nib.Nifti1Image(arterial, np.eye(4)), str(arterial_path))

            venous = np.zeros(labels.shape, dtype=np.uint8)
            venous[20:26, 28, 12] = 1
            venous_path = tmp_path / "venous.nii.gz"
            nib.save(nib.Nifti1Image(venous, np.eye(4)), str(venous_path))

            graph = {
                "case_id": "tiny_domain_repair",
                "nodes": [
                    {"id": "aorta_inlet", "position_mm": [10.0, 24.0, 12.0], "radius_mm": 2.0, "boundary_role": "arterial_inlet"},
                    {"id": "arterial_outlet", "position_mm": [35.0, 24.0, 12.0], "radius_mm": 2.0, "boundary_role": "arterial_outlet"},
                    {"id": "venous_inlet", "position_mm": [20.0, 28.0, 12.0], "radius_mm": 2.0, "boundary_role": "venous_inlet"},
                    {"id": "venous_outlet", "position_mm": [25.0, 28.0, 12.0], "radius_mm": 2.0, "boundary_role": "venous_outlet"},
                ],
                "edges": [
                    {
                        "id": "aorta_inlet_to_outlet",
                        "source": "aorta_inlet",
                        "target": "arterial_outlet",
                        "vessel_type": "arterial",
                        "flow_role": "aorta_trunk",
                        "radius_start_mm": 1.0,
                        "radius_end_mm": 1.0,
                        "polyline_mm": [[10.0, 24.0, 12.0], [35.0, 24.0, 12.0]],
                    },
                    {
                        "id": "venous_inlet_to_outlet",
                        "source": "venous_inlet",
                        "target": "venous_outlet",
                        "vessel_type": "venous",
                        "flow_role": "venous_return_trunk",
                        "radius_start_mm": 1.0,
                        "radius_end_mm": 1.0,
                        "polyline_mm": [[20.0, 28.0, 12.0], [25.0, 28.0, 12.0]],
                    },
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
                            "sample_step_mm": 1.0,
                            "vessel_wall_thickness_mm": 1.0,
                            "contrast_mode": "arterial",
                            "collision_cleanup": "nearest-centerline",
                            "connected_components": 1,
                            "arterial_components": 3,
                            "venous_components": 1,
                            "arterial_venous_overlap_voxels_before_cleanup": 0,
                            "arterial_venous_overlap_voxels_after_cleanup": 0,
                        },
                        "outputs": {
                            "arterial_lumen_mask": str(arterial_path),
                            "venous_lumen_mask": str(venous_path),
                        },
                    },
                    sort_keys=False,
                )
            )

            result = repair_vascular_domain_connectivity(
                voxelized_spec_path=spec_path,
                output_dir=tmp_path / "out",
                case_id="tiny_domain_repaired",
                max_unseeded_component_voxels=20,
                seed_search_radius_voxels=1,
                connect_seeded_components=False,
                write_material_volumes=False,
                report_path=tmp_path / "report.md",
            )

            self.assertEqual(result.arterial_summary.components_before, 3)
            self.assertEqual(result.arterial_summary.components_after, 1)
            self.assertEqual(result.arterial_summary.pruned_component_count, 2)
            self.assertEqual(result.overlap_after_repair, 0)
            repaired_spec = yaml.safe_load(Path(result.repaired_spec_yaml_path).read_text())
            self.assertEqual(repaired_spec["voxelization"]["arterial_components"], 1)
            self.assertTrue(Path(repaired_spec["outputs"]["arterial_lumen_mask"]).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertIn("Vascular Domain Connectivity Repair", Path(result.report_path).read_text())


if __name__ == "__main__":
    unittest.main()
