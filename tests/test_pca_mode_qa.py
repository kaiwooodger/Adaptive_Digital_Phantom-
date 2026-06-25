from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.pca_mode_qa import rank_pca_modes


class PcaModeQaTests(unittest.TestCase):
    def test_rank_pca_modes_approves_and_rejects_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics_csv = temp_path / "metrics.csv"
            atlas_spec = temp_path / "atlas.yaml"
            fieldnames = [
                "variant_id",
                "label",
                "mode_index",
                "mode_weight",
                "body_volume_cm3",
                "waist_cm",
                "bbox_x_mm",
                "bbox_y_mm",
                "bbox_z_mm",
                "vascular_components",
                "group_lungs_volume_cm3",
                "group_liver_volume_cm3",
                "group_kidneys_volume_cm3",
                "group_bladder_volume_cm3",
                "group_bone_volume_cm3",
                "group_vessel_wall_volume_cm3",
                "group_vascular_fluid_volume_cm3",
                "material_labels_path",
                "preview_png_path",
            ]
            rows = [
                ("mean", "Mean", "", 0.0, 10000.0, 80.0, 1000.0, 1200.0, 200.0, 50.0, 900.0, 25.0, 100.0, 1),
                ("mode01_neg", "-1 Mode 1", 1, -1.0, 9900.0, 79.5, 980.0, 1190.0, 210.0, 52.0, 905.0, 25.5, 101.0, 1),
                ("mode01_pos", "+1 Mode 1", 1, 1.0, 10100.0, 80.5, 1020.0, 1210.0, 190.0, 48.0, 895.0, 24.5, 99.0, 1),
                ("mode02_neg", "-1 Mode 2", 2, -1.0, 13500.0, 86.0, 450.0, 2500.0, 50.0, 10.0, 400.0, 60.0, 180.0, 2),
                ("mode02_pos", "+1 Mode 2", 2, 1.0, 7000.0, 73.0, 2000.0, 500.0, 400.0, 100.0, 1500.0, 5.0, 20.0, 2),
            ]
            with metrics_csv.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    (
                        variant_id,
                        label,
                        mode_index,
                        mode_weight,
                        body,
                        waist,
                        lungs,
                        liver,
                        kidneys,
                        bladder,
                        bone,
                        wall,
                        fluid,
                        components,
                    ) = row
                    writer.writerow(
                        {
                            "variant_id": variant_id,
                            "label": label,
                            "mode_index": mode_index,
                            "mode_weight": mode_weight,
                            "body_volume_cm3": body,
                            "waist_cm": waist,
                            "bbox_x_mm": 100.0,
                            "bbox_y_mm": 100.0,
                            "bbox_z_mm": 100.0,
                            "vascular_components": components,
                            "group_lungs_volume_cm3": lungs,
                            "group_liver_volume_cm3": liver,
                            "group_kidneys_volume_cm3": kidneys,
                            "group_bladder_volume_cm3": bladder,
                            "group_bone_volume_cm3": bone,
                            "group_vessel_wall_volume_cm3": wall,
                            "group_vascular_fluid_volume_cm3": fluid,
                            "material_labels_path": f"{variant_id}.nii.gz",
                            "preview_png_path": f"{variant_id}.png",
                        }
                    )
            atlas_spec.write_text(yaml.safe_dump({"case_id": "toy_pca_modes", "package_type": "pca_mode_variant_atlas"}))

            result = rank_pca_modes(
                metrics_csv_path=metrics_csv,
                atlas_spec_path=atlas_spec,
                output_dir=temp_path / "qa",
                report_path=temp_path / "qa.md",
            )

            self.assertEqual(result.approved_modes, (1,))
            self.assertEqual(result.rejected_modes, (2,))
            self.assertTrue(Path(result.ranking_csv_path).exists())
            self.assertTrue(Path(result.decisions_yaml_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertIn("vascular_component_count_mismatch", result.decisions[1].issues[0])


if __name__ == "__main__":
    unittest.main()
