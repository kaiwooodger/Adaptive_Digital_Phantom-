from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.profile_envelope import build_profile_operating_envelope


class ProfileEnvelopeTests(unittest.TestCase):
    def _write_metrics(self, path: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = [
            "profile_id",
            "case_id",
            "target_bmi",
            "target_waist_cm",
            "target_height_cm",
            "morph_mode",
            "xy_padding_voxels",
            "achieved_waist_cm",
            "body_volume_l",
            "ptv_peak_v95_percent",
            "gamma_min_pass_rate_percent",
            "overall_status",
        ]
        with path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_envelope_merges_rows_and_compares_prescription_to_actual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base = temp_path / "base.csv"
            extra = temp_path / "extra.csv"
            prescription = temp_path / "prescription.yaml"
            self._write_metrics(
                base,
                [
                    {
                        "profile_id": "bmi32",
                        "case_id": "case32",
                        "target_bmi": 32,
                        "target_waist_cm": 110,
                        "target_height_cm": 175,
                        "morph_mode": "standard",
                        "xy_padding_voxels": 0,
                        "achieved_waist_cm": 108.5,
                        "body_volume_l": 35.0,
                        "ptv_peak_v95_percent": 85.2,
                        "gamma_min_pass_rate_percent": 100.0,
                        "overall_status": "pass",
                    },
                    {
                        "profile_id": "bmi38",
                        "case_id": "case38",
                        "target_bmi": 38,
                        "target_waist_cm": 125,
                        "target_height_cm": 175,
                        "morph_mode": "high-bmi",
                        "xy_padding_voxels": 96,
                        "achieved_waist_cm": 125.0,
                        "body_volume_l": 38.0,
                        "ptv_peak_v95_percent": 85.6,
                        "gamma_min_pass_rate_percent": 100.0,
                        "overall_status": "pass",
                    },
                ],
            )
            self._write_metrics(
                extra,
                [
                    {
                        "profile_id": "bmi35",
                        "case_id": "case35",
                        "target_bmi": 35,
                        "target_waist_cm": 118,
                        "target_height_cm": 175,
                        "morph_mode": "high-bmi",
                        "xy_padding_voxels": 96,
                        "achieved_waist_cm": 117.95,
                        "body_volume_l": 34.25,
                        "ptv_peak_v95_percent": 85.7,
                        "gamma_min_pass_rate_percent": 100.0,
                        "overall_status": "pass",
                    }
                ],
            )
            prescription.write_text(
                yaml.safe_dump(
                    {
                        "profile_id": "bmi35",
                        "target": {"waist_cm": 118.0},
                        "interpolated_expectations": {
                            "body_volume_l": 36.5,
                            "ptv_peak_v95_percent": 85.4,
                            "gamma_min_pass_rate_percent": 100.0,
                        },
                    }
                )
            )

            result = build_profile_operating_envelope(
                metrics_csv_paths=(base, extra),
                output_dir=temp_path / "envelope",
                envelope_id="toy_envelope",
                prescription_yaml_path=prescription,
                report_path=temp_path / "envelope.md",
            )

            self.assertEqual(result.profile_count, 3)
            self.assertEqual(result.pass_count, 3)
            self.assertEqual(result.high_bmi_profile_count, 2)
            self.assertIsNotNone(result.prediction_delta)
            self.assertAlmostEqual(result.prediction_delta.body_volume_error_l, -2.25)
            self.assertTrue(Path(result.metrics_csv_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            manifest = yaml.safe_load(Path(result.manifest_yaml_path).read_text())
            self.assertEqual(manifest["summary"]["profile_count"], 3)
            self.assertEqual(manifest["prediction_delta"]["profile_id"], "bmi35")


if __name__ == "__main__":
    unittest.main()
