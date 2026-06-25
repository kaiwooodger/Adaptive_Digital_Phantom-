from pathlib import Path
import tempfile
import unittest
import zipfile

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.reg_training_testing import stage_reg_training_testing_zip


def _write_nifti(path: Path, data: np.ndarray) -> None:
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))


class RegTrainingTestingStagingTests(unittest.TestCase):
    def test_stage_reg_training_testing_subset_extracts_paired_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "Reg-Training-Testing.zip"
            image = np.zeros((8, 9, 5), dtype=np.int16)
            image[2:6, 2:7, :] = 100
            label = np.zeros(image.shape, dtype=np.uint8)
            label[3:5, 4:6, 1:4] = 1
            image_path = root / "img0001-0061.nii.gz"
            label_path = root / "label0001-0061.nii.gz"
            _write_nifti(image_path, image)
            _write_nifti(label_path, label)
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(image_path, "Training-Testing/img/0061/img0001-0061.nii.gz")
                zf.write(label_path, "Training-Testing/label/0061/label0001-0061.nii.gz")

            result = stage_reg_training_testing_zip(
                zip_path=zip_path,
                output_dir=root / "processed",
                dataset_id="toy_reg_training",
                max_targets=1,
                report_path=root / "report.md",
            )

            manifest = yaml.safe_load(Path(result.manifest_yaml_path).read_text())
            self.assertEqual(result.discovered_target_count, 1)
            self.assertEqual(result.staged_target_count, 1)
            self.assertEqual(result.staged_pair_count, 1)
            self.assertEqual(result.readiness_status, "registration_subset_ready")
            self.assertTrue(Path(result.manifest_csv_path).exists())
            self.assertTrue(Path(result.atlas_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertEqual(manifest["targets"][0]["target_case_id"], "0061")
            self.assertEqual(manifest["targets"][0]["shape"], [8, 9, 5])


if __name__ == "__main__":
    unittest.main()
