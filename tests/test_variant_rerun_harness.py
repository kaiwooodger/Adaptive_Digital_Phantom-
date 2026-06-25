from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.variant_rerun_harness import build_variant_rerun_harness
from phantom_twin.vascular_network import deform_vascular_graph_for_variant


def _regions() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "name": f"label_{index}",
            "material_id": f"label_{index}",
            "target_hu_midpoint": float(index * 10),
            "mass_density_g_cm3": 1.0 + index * 0.01,
            "relative_electron_density": 1.0 + index * 0.005,
            "color": "#ffffff",
        }
        for index in range(16)
    ]


class VariantRerunHarnessTests(unittest.TestCase):
    def test_deform_vascular_graph_for_variant_moves_nodes_and_preserves_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            baseline = np.zeros((16, 16, 8), dtype=np.int16)
            variant = np.zeros((16, 16, 8), dtype=np.int16)
            baseline[2:14, 2:14, 1:7] = 4
            variant[1:15, 2:15, 1:7] = 4
            baseline[4:6, 6:8, 3:5] = 7
            variant[5:8, 7:10, 3:5] = 7
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            baseline_labels = temp_path / "baseline.nii.gz"
            variant_labels = temp_path / "variant.nii.gz"
            nib.save(nib.Nifti1Image(baseline, affine), baseline_labels)
            nib.save(nib.Nifti1Image(variant, affine), variant_labels)
            graph_path = temp_path / "graph.yaml"
            graph_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy",
                        "graph_metadata": {"node_count": 2, "edge_count": 1},
                        "nodes": [
                            {"id": "renal_branch_origin", "label": "origin", "position_mm": [14.0, 14.0, 12.0], "radius_mm": 2.0},
                            {
                                "id": "left_renal_outlet",
                                "label": "left renal",
                                "position_mm": [8.0, 14.0, 12.0],
                                "radius_mm": 1.5,
                                "boundary_role": "arterial_outlet",
                            },
                        ],
                        "edges": [
                            {
                                "id": "renal_origin_to_left_renal",
                                "source": "renal_branch_origin",
                                "target": "left_renal_outlet",
                                "vessel_type": "arterial",
                                "flow_role": "renal_branch",
                                "radius_start_mm": 2.0,
                                "radius_end_mm": 1.5,
                                "polyline_mm": [[14.0, 14.0, 12.0], [11.0, 14.0, 12.0], [8.0, 14.0, 12.0]],
                            }
                        ],
                    },
                    sort_keys=False,
                )
            )

            result = deform_vascular_graph_for_variant(
                baseline_graph_path=graph_path,
                baseline_labels_path=baseline_labels,
                variant_labels_path=variant_labels,
                output_dir=temp_path / "deformed",
                case_id="toy_variant",
            )
            deformed = yaml.safe_load(Path(result.graph_yaml_path).read_text())

            self.assertEqual(result.node_count, 2)
            self.assertEqual(result.edge_count, 1)
            self.assertGreater(result.max_node_displacement_mm, 0.0)
            self.assertEqual(deformed["edges"][0]["polyline_mm"][0], deformed["nodes"][0]["position_mm"])
            self.assertTrue(Path(result.preview_png_path).exists())

    def test_build_variant_rerun_harness_stages_material_maps_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            labels = np.zeros((12, 12, 8), dtype=np.int16)
            labels[2:10, 2:10, 1:7] = 4
            labels[3:5, 3:5, 2:5] = 8
            labels[6:8, 3:5, 2:5] = 6
            labels[4:6, 6:8, 2:5] = 7
            labels[5:7, 5:7, 3:6] = 10
            labels[7:9, 7:9, 2:5] = 13
            labels[8:10, 8:10, 2:5] = 14
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            baseline_labels = temp_path / "baseline_labels.nii.gz"
            variant_labels = temp_path / "variant_labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, affine), baseline_labels)
            nib.save(nib.Nifti1Image(labels, affine), variant_labels)

            combined_spec = temp_path / "combined.yaml"
            combined_spec.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_baseline",
                        "outputs": {"blood_material_labels": str(baseline_labels)},
                        "regions": _regions(),
                    },
                    sort_keys=False,
                )
            )
            atlas_spec = temp_path / "atlas.yaml"
            atlas_spec.write_text(yaml.safe_dump({"source_combined_spec": str(combined_spec)}))
            approved_manifest = temp_path / "approved.yaml"
            approved_manifest.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_modes",
                        "source_atlas_spec": str(atlas_spec),
                        "variants": [
                            {
                                "variant_id": "mode01_neg",
                                "material_labels": str(variant_labels),
                                "release_role": "approved_with_warning",
                                "warning_status": "warning",
                            }
                        ],
                    },
                    sort_keys=False,
                )
            )

            result = build_variant_rerun_harness(
                approved_set_manifest_path=approved_manifest,
                variant_id="mode01_neg",
                output_dir=temp_path / "harness",
                case_id="toy_mode01_neg",
            )

            self.assertTrue(result.rt_ready)
            self.assertTrue(result.flow_ready)
            self.assertEqual(len(result.material_map_paths), 3)
            self.assertTrue(Path(result.variant_combined_spec_path).exists())
            self.assertTrue(Path(result.harness_yaml_path).exists())
            self.assertTrue(Path(result.preflight_csv_path).exists())
            self.assertTrue(Path(result.commands_script_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertIn("build-radiotherapy-qa-package", Path(result.commands_script_path).read_text())

    def test_build_variant_rerun_harness_can_skip_material_maps_for_label_derived_rt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            labels = np.zeros((12, 12, 8), dtype=np.int16)
            labels[2:10, 2:10, 1:7] = 4
            labels[3:5, 3:5, 2:5] = 8
            labels[6:8, 3:5, 2:5] = 6
            labels[4:6, 6:8, 2:5] = 7
            labels[5:7, 5:7, 3:6] = 10
            labels[7:9, 7:9, 2:5] = 13
            labels[8:10, 8:10, 2:5] = 14
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            baseline_labels = temp_path / "baseline_labels.nii.gz"
            variant_labels = temp_path / "variant_labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, affine), baseline_labels)
            nib.save(nib.Nifti1Image(labels, affine), variant_labels)

            combined_spec = temp_path / "combined.yaml"
            combined_spec.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_baseline",
                        "outputs": {
                            "blood_material_labels": str(baseline_labels),
                            "contrast_material_labels": str(baseline_labels),
                            "blood_synthetic_hu": "old_hu.nii.gz",
                            "contrast_synthetic_hu": "old_contrast_hu.nii.gz",
                        },
                        "regions": _regions(),
                    },
                    sort_keys=False,
                )
            )
            atlas_spec = temp_path / "atlas.yaml"
            atlas_spec.write_text(yaml.safe_dump({"source_combined_spec": str(combined_spec)}))
            approved_manifest = temp_path / "approved.yaml"
            approved_manifest.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_modes",
                        "source_atlas_spec": str(atlas_spec),
                        "variants": [
                            {
                                "variant_id": "mode01_neg",
                                "material_labels": str(variant_labels),
                                "release_role": "approved_with_warning",
                                "warning_status": "warning",
                            }
                        ],
                    },
                    sort_keys=False,
                )
            )

            result = build_variant_rerun_harness(
                approved_set_manifest_path=approved_manifest,
                variant_id="mode01_neg",
                output_dir=temp_path / "harness",
                case_id="toy_mode01_neg",
                stage_material_maps=False,
            )
            variant_spec = yaml.safe_load(Path(result.variant_combined_spec_path).read_text())

            self.assertTrue(result.rt_ready)
            self.assertFalse(result.material_maps_staged)
            self.assertNotIn("blood_synthetic_hu", variant_spec["outputs"])
            self.assertNotIn("contrast_material_labels", variant_spec["outputs"])
            self.assertIn("synthesize", Path(result.commands_script_path).read_text())


if __name__ == "__main__":
    unittest.main()
