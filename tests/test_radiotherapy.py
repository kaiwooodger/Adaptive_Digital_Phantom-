from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.radiotherapy import build_radiotherapy_qa_package


class RadiotherapyQAPackageTests(unittest.TestCase):
    def test_radiotherapy_package_exports_masks_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shape = (16, 16, 8)
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            labels = np.zeros(shape, dtype=np.int16)
            labels[2:14, 2:14, 1:7] = 4
            labels[4:7, 4:7, 2:5] = 8
            labels[9:13, 4:9, 2:5] = 6
            labels[5:8, 10:13, 2:5] = 7
            labels[7:10, 7:10, 3:5] = 10
            labels[8, 8, 4] = 11
            labels[10:12, 10:12, 4:6] = 14
            labels[10:12, 9:10, 4:6] = 13

            hu = np.full(shape, -975.0, dtype=np.float32)
            density = np.full(shape, 0.0012, dtype=np.float32)
            red = np.full(shape, 0.001, dtype=np.float32)
            for value, hu_value, density_value, red_value in [
                (4, 45.0, 1.05, 1.04),
                (6, 60.0, 1.06, 1.05),
                (7, 42.5, 1.05, 1.04),
                (8, -750.0, 0.2, 0.2),
                (10, 300.0, 1.2, 1.15),
                (11, 1050.0, 1.85, 1.65),
                (13, 100.0, 1.1, 1.08),
                (14, 50.0, 1.06, 1.05),
            ]:
                hu[labels == value] = hu_value
                density[labels == value] = density_value
                red[labels == value] = red_value

            paths = {}
            for name, data in [
                ("hu", hu),
                ("labels", labels),
                ("density", density),
                ("red", red),
            ]:
                path = temp_path / f"{name}.nii.gz"
                nib.save(nib.Nifti1Image(data, affine), path)
                paths[name] = path

            spec = {
                "case_id": "toy",
                "outputs": {
                    "blood_synthetic_hu": str(paths["hu"]),
                    "blood_material_labels": str(paths["labels"]),
                    "blood_mass_density_g_cm3": str(paths["density"]),
                    "blood_relative_electron_density": str(paths["red"]),
                },
                "regions": [
                    {
                        "index": 4,
                        "name": "generic_muscle_soft_tissue",
                        "material_id": "muscle",
                        "target_hu_midpoint": 45.0,
                        "mass_density_g_cm3": 1.05,
                        "relative_electron_density": 1.04,
                        "color": "#d95d39",
                    },
                    {
                        "index": 10,
                        "name": "trabecular_bone",
                        "material_id": "trabecular_bone",
                        "target_hu_midpoint": 300.0,
                        "mass_density_g_cm3": 1.2,
                        "relative_electron_density": 1.15,
                        "color": "#e9ecef",
                    },
                ],
            }
            spec_path = temp_path / "combined_spec.yaml"
            spec_path.write_text(yaml.safe_dump(spec))

            result = build_radiotherapy_qa_package(
                combined_spec_path=spec_path,
                output_dir=temp_path / "rt",
                case_id="toy",
                target_radius_mm=4.0,
                ptv_margin_mm=2.0,
                report_path=temp_path / "rt_report.md",
            )

            self.assertTrue(Path(result.rt_hu_path).exists())
            self.assertTrue(Path(result.mask_manifest_csv_path).exists())
            self.assertTrue(Path(result.material_calibration_csv_path).exists())
            self.assertTrue(Path(result.pymedphys_placeholder_yaml_path).exists())
            self.assertTrue(Path(result.package_spec_yaml_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertGreater(result.target_gtv_volume_cm3, 0.0)
            self.assertGreater(result.target_ptv_volume_cm3, result.target_gtv_volume_cm3)
            self.assertEqual(len(result.mask_stats), 9)

    def test_radiotherapy_package_derives_maps_from_material_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shape = (16, 16, 8)
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            labels = np.zeros(shape, dtype=np.int16)
            labels[2:14, 2:14, 1:7] = 4
            labels[4:7, 4:7, 2:5] = 8
            labels[9:13, 4:9, 2:5] = 6
            labels[5:8, 10:13, 2:5] = 7
            labels[7:10, 7:10, 3:5] = 10
            labels[8, 8, 4] = 11
            labels[10:12, 10:12, 4:6] = 14
            labels[10:12, 9:10, 4:6] = 13
            labels_path = temp_path / "labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, affine), labels_path)

            regions = []
            for index in range(16):
                regions.append(
                    {
                        "index": index,
                        "name": f"label_{index}",
                        "material_id": f"label_{index}",
                        "target_hu_midpoint": float(index * 10),
                        "mass_density_g_cm3": 1.0 + index * 0.01,
                        "relative_electron_density": 1.0 + index * 0.005,
                        "color": "#ffffff",
                    }
                )
            spec_path = temp_path / "combined_spec.yaml"
            spec_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_label_only",
                        "outputs": {"blood_material_labels": str(labels_path)},
                        "regions": regions,
                    }
                )
            )

            result = build_radiotherapy_qa_package(
                combined_spec_path=spec_path,
                output_dir=temp_path / "rt_label_only",
                case_id="toy_label_only",
                target_radius_mm=4.0,
                ptv_margin_mm=2.0,
                report_path=temp_path / "rt_label_only_report.md",
            )

            self.assertTrue(Path(result.rt_hu_path).exists())
            self.assertTrue(Path(result.rt_density_path).exists())
            self.assertTrue(Path(result.rt_red_path).exists())
            self.assertIn("rt_maps_synthesized_from_material_labels_and_region_table", result.notes)
            self.assertGreater(result.target_gtv_volume_cm3, 0.0)


if __name__ == "__main__":
    unittest.main()
