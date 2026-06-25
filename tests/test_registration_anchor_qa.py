from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.registration_anchor_qa import rank_registration_anchors


class RegistrationAnchorQaTests(unittest.TestCase):
    def test_rank_registration_anchors_approves_reviews_and_rejects_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_metrics = root / "label_metrics.csv"
            spec = root / "benchmark.yaml"
            fields = [
                "target_case_id",
                "label_id",
                "label_name",
                "consensus_volume_ml",
                "mean_propagated_volume_ml",
                "volume_cv",
                "mean_dice_to_consensus",
                "median_dice_to_consensus",
                "min_dice_to_consensus",
                "mean_centroid_dispersion_mm",
                "max_centroid_dispersion_mm",
            ]
            rows = [
                ("t1", 1, "liver", 1000, 990, 0.20, 0.82, 0.84, 0.70, 15, 50),
                ("t2", 1, "liver", 1100, 1110, 0.25, 0.78, 0.80, 0.65, 20, 70),
                ("t1", 2, "aorta", 80, 95, 0.50, 0.55, 0.60, 0.20, 30, 95),
                ("t2", 2, "aorta", 75, 90, 0.55, 0.47, 0.51, 0.15, 35, 120),
                ("t1", 3, "gallbladder", 0.0, 20, 1.10, 0.05, 0.00, 0.00, 40, 140),
                ("t2", 3, "gallbladder", 0.2, 18, 0.90, 0.08, 0.02, 0.00, 45, 160),
            ]
            with label_metrics.open("w", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(zip(fields, row, strict=True)))
            spec.write_text(yaml.safe_dump({"outputs": {"label_metrics_csv": str(label_metrics)}}))

            result = rank_registration_anchors(
                benchmark_spec_path=spec,
                output_dir=root / "qa",
                case_id="toy_anchor_qa",
                report_path=root / "qa.md",
            )

            self.assertEqual(result.approved_anchor_labels, (1,))
            self.assertEqual(result.review_anchor_labels, (2,))
            self.assertEqual(result.rejected_anchor_labels, (3,))
            self.assertTrue(Path(result.ranking_csv_path).exists())
            self.assertTrue(Path(result.decisions_yaml_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertIn("primary_deformation_anchor", Path(result.decisions_yaml_path).read_text())


if __name__ == "__main__":
    unittest.main()
