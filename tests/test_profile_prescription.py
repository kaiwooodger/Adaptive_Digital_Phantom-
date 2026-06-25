from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.profile_prescription import build_profile_operating_prescription


class ProfilePrescriptionTests(unittest.TestCase):
    def test_prescription_interpolates_inside_sweep_and_selects_high_bmi_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics = temp_path / "sweep_metrics.csv"
            fieldnames = [
                "profile_id",
                "target_bmi",
                "target_waist_cm",
                "target_height_cm",
                "body_volume_l",
                "ptv_peak_v95_percent",
                "gamma_min_pass_rate_percent",
                "overall_status",
            ]
            rows = [
                ("bmi22_waist85_height175", 22.0, 85.0, 175.0, 20.0, 86.1, 100.0, "pass"),
                ("bmi27_waist95_height175", 27.0, 95.0, 175.0, 23.5, 86.0, 100.0, "pass"),
                ("bmi32_waist110_height175", 32.0, 110.0, 175.0, 35.0, 85.2, 100.0, "pass"),
                ("bmi38_waist125_height175", 38.0, 125.0, 175.0, 38.0, 85.7, 100.0, "pass"),
            ]
            with metrics.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(zip(fieldnames, row)))

            result = build_profile_operating_prescription(
                metrics_csv_path=metrics,
                output_dir=temp_path / "prescriptions",
                profile_id="BMI 35 Waist 118",
                case_id="toy_prescription",
                target_height_cm=175.0,
                target_bmi=35.0,
                target_waist_cm=118.0,
                report_path=temp_path / "prescription.md",
            )

            self.assertEqual(result.profile_id, "bmi_35_waist_118")
            self.assertEqual(result.fit_status, "interpolated_inside_validated_sweep_envelope")
            self.assertEqual(result.morph_mode, "high-bmi")
            self.assertEqual(result.xy_padding_voxels, 96)
            self.assertEqual(result.lower_profile_id, "bmi32_waist110_height175")
            self.assertEqual(result.upper_profile_id, "bmi38_waist125_height175")
            self.assertGreater(result.interpolated_body_volume_l, 35.0)
            self.assertTrue(Path(result.prescription_yaml_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            payload = yaml.safe_load(Path(result.prescription_yaml_path).read_text())
            self.assertEqual(payload["prescription"]["morph_mode"], "high-bmi")
            self.assertIn("build-profile-sweep", "\n".join(payload["recommended_commands"]))


if __name__ == "__main__":
    unittest.main()
