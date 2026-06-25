from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.profile_planner import plan_next_profile_validations


class ProfilePlannerTests(unittest.TestCase):
    def test_planner_prioritizes_high_bmi_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics = temp_path / "envelope.csv"
            fieldnames = [
                "profile_id",
                "case_id",
                "target_bmi",
                "target_waist_cm",
                "target_height_cm",
                "morph_mode",
                "xy_padding_voxels",
                "body_volume_l",
                "ptv_peak_v95_percent",
                "overall_status",
            ]
            rows = [
                ("bmi32_waist110_height175", "case32", 32.0, 110.0, 175.0, "standard", 0, 35.0, 85.2, "pass"),
                ("bmi35_waist118_height175", "case35", 35.0, 118.0, 175.0, "high-bmi", 96, 34.2, 85.7, "pass"),
                ("bmi38_waist125_height175", "case38", 38.0, 125.0, 175.0, "high-bmi", 96, 38.0, 85.6, "pass"),
            ]
            with metrics.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(zip(fieldnames, row)))

            result = plan_next_profile_validations(
                metrics_csv_path=metrics,
                output_dir=temp_path / "plan",
                plan_id="toy_plan",
                max_candidates=4,
                report_path=temp_path / "plan.md",
            )

            self.assertGreaterEqual(result.candidate_count, 3)
            self.assertEqual(result.top_candidates[0].morph_mode, "high-bmi")
            self.assertIn("transition", result.top_candidates[0].reason)
            self.assertTrue(any(item.target_waist_cm == 114.0 and item.morph_mode == "standard" for item in result.top_candidates))
            self.assertTrue(any(item.target_waist_cm == 116.0 and item.morph_mode == "high-bmi" for item in result.top_candidates))
            self.assertIn("build-profile-sweep", result.top_candidates[0].recommended_command)
            self.assertTrue(Path(result.plan_yaml_path).exists())
            self.assertTrue(Path(result.candidates_csv_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            payload = yaml.safe_load(Path(result.plan_yaml_path).read_text())
            self.assertEqual(payload["summary"]["candidate_count"], result.candidate_count)


if __name__ == "__main__":
    unittest.main()
