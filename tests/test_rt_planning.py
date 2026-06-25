from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.radiotherapy import build_radiotherapy_qa_package
from phantom_twin.rt_planning import build_rt_planning_bundle


class RTPlanningBundleTests(unittest.TestCase):
    def test_rt_planning_bundle_exports_dicom_and_dose_metrics(self) -> None:
        try:
            import pydicom  # noqa: F401
        except ImportError:
            self.skipTest("pydicom is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shape = (18, 18, 8)
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            labels = np.zeros(shape, dtype=np.int16)
            labels[2:16, 2:16, 1:7] = 4
            labels[4:7, 4:7, 2:5] = 8
            labels[10:14, 4:9, 2:5] = 6
            labels[5:8, 11:14, 2:5] = 7
            labels[7:11, 7:11, 3:5] = 10
            labels[8, 8, 4] = 11
            labels[11:13, 11:13, 4:6] = 14
            labels[10:11, 11:13, 4:6] = 13

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

            combined_spec = {
                "case_id": "toy",
                "outputs": {
                    "blood_synthetic_hu": str(paths["hu"]),
                    "blood_material_labels": str(paths["labels"]),
                    "blood_mass_density_g_cm3": str(paths["density"]),
                    "blood_relative_electron_density": str(paths["red"]),
                },
                "regions": [],
            }
            combined_spec_path = temp_path / "combined_spec.yaml"
            combined_spec_path.write_text(yaml.safe_dump(combined_spec))
            rt_result = build_radiotherapy_qa_package(
                combined_spec_path=combined_spec_path,
                output_dir=temp_path / "rt_package",
                case_id="toy",
                target_radius_mm=4.0,
                ptv_margin_mm=2.0,
                report_path=temp_path / "rt_package.md",
            )

            flow_model_path = temp_path / "coupled_flow.yaml"
            flow_model_path.write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "arterial_inlet_flow_mean_ml_s": 80.0,
                            "arterial_inlet_flow_min_ml_s": 55.0,
                            "arterial_inlet_flow_max_ml_s": 155.0,
                        }
                    }
                )
            )
            planning = build_rt_planning_bundle(
                rt_package_spec_path=rt_result.package_spec_yaml_path,
                coupled_flow_model_path=flow_model_path,
                output_dir=temp_path / "planning",
                case_id="toy",
                prescription_dose_gy=20.0,
                report_path=temp_path / "planning.md",
            )

            self.assertTrue(Path(planning.static_dose_nifti_path).exists())
            self.assertTrue(Path(planning.pulsatile_mean_dose_nifti_path).exists())
            self.assertTrue(Path(planning.dose_metrics_csv_path).exists())
            self.assertTrue(Path(planning.dose_comparison_csv_path).exists())
            self.assertTrue(Path(planning.pymedphys_eval_config_yaml_path).exists())
            self.assertTrue(Path(planning.dicom_rtstruct_path or "").exists())
            self.assertEqual(len(list(Path(planning.dicom_ct_dir or "").glob("*.dcm"))), shape[2])
            self.assertEqual(len(planning.dicom_rtdose_paths), 4)
            self.assertGreater(planning.flow_amplitude_fraction, 0.0)
            target_static = [
                metric
                for metric in planning.metrics
                if metric.mask_id == "target_ptv_synthetic_vertebral" and metric.state == "static"
            ][0]
            self.assertGreater(target_static.d95_gy, 18.0)
            self.assertGreater(len(planning.comparisons), 0)


if __name__ == "__main__":
    unittest.main()
