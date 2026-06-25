from pathlib import Path
import csv
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.reg_training_benchmark import build_reg_training_testing_benchmark


def _write_nifti(path: Path, data: np.ndarray) -> None:
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


class RegTrainingBenchmarkTests(unittest.TestCase):
    def test_build_reg_training_testing_benchmark_scores_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_csv = root / "staged_manifest.csv"
            staged_manifest = root / "staged_manifest.yaml"
            labelmap = root / "labelmap.yaml"
            rows = []
            for moving_case, shift in [("0001", 0), ("0002", 1), ("0003", 0)]:
                image = np.zeros((18, 18, 8), dtype=np.int16)
                image[4:12, 4 + shift : 12 + shift, 2:6] = 100
                label = np.zeros(image.shape, dtype=np.uint8)
                label[4:10, 4 + shift : 10 + shift, 2:6] = 1
                label[11:15, 11:15, 3:6] = 2
                image_path = root / "images" / "0061" / f"img{moving_case}-0061.nii.gz"
                label_path = root / "labels" / "0061" / f"label{moving_case}-0061.nii.gz"
                _write_nifti(image_path, image)
                _write_nifti(label_path, label)
                rows.append(
                    {
                        "target_case_id": "0061",
                        "moving_case_id": moving_case,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "image_size_mb": "0.1",
                        "label_size_mb": "0.1",
                    }
                )
            with manifest_csv.open("w", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            staged_manifest.write_text(yaml.safe_dump({"outputs": {"manifest_csv": str(manifest_csv)}}))
            labelmap.write_text(yaml.safe_dump({"labels": {1: {"name": "organ_a"}, 2: {"name": "organ_b"}}}))

            result = build_reg_training_testing_benchmark(
                staged_manifest_path=staged_manifest,
                output_dir=root / "benchmark",
                dataset_id="toy_reg_benchmark",
                labelmap_path=labelmap,
                min_mean_ncc=-1.0,
                intensity_sample_stride=2,
                report_path=root / "benchmark.md",
            )

            self.assertEqual(result.target_count, 1)
            self.assertEqual(result.pair_count, 3)
            self.assertIn(result.readiness_status, {"registration_benchmark_pass", "registration_benchmark_review_required"})
            self.assertTrue(Path(result.target_summary_csv_path).exists())
            self.assertTrue(Path(result.label_metrics_csv_path).exists())
            self.assertTrue(Path(result.pair_metrics_csv_path).exists())
            self.assertTrue(Path(result.spec_yaml_path).exists())
            self.assertTrue(Path(result.atlas_png_path).exists())
            self.assertTrue(Path(result.target_results[0].consensus_label_path).exists())
            self.assertGreater(result.mean_label_dice_to_consensus, 0.75)


if __name__ == "__main__":
    unittest.main()
