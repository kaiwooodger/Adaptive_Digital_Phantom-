from pathlib import Path
import csv
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vessel_label_anatomy_qa import qa_vessel_label_anatomy


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


class VesselLabelAnatomyQATests(unittest.TestCase):
    def test_vessel_label_anatomy_qa_flags_bone_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            anatomy = np.zeros((12, 12, 8), dtype=np.int16)
            anatomy[1:11, 1:11, 1:7] = 1
            anatomy[7:10, 5:8, 3:6] = 7
            anatomy[2:5, 5:8, 3:6] = 7
            anatomy[5:7, 5:7, 1:7] = 10
            anatomy[3:7, 2:5, 3:6] = 6
            vessel = np.zeros_like(anatomy)
            vessel[8:10, 6:8, 4:6] = 27
            vessel[5:7, 5:7, 2:5] = 4
            anatomy_path = temp_path / "anatomy.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            config_path = temp_path / "vessel_labels.yaml"
            _write_nifti(anatomy_path, anatomy)
            _write_nifti(vessel_path, vessel)
            config_path.write_text(yaml.safe_dump({"labels": {4: "Aorta", 27: "Right renal artery"}}, sort_keys=False))

            result = qa_vessel_label_anatomy(
                anatomy_labels_path=anatomy_path,
                vessel_labels_path=vessel_path,
                vessel_label_config=config_path,
                case_id="toy_vessel_qa",
                output_dir=temp_path / "qa",
                required_vessel_labels=(4, 27),
                report_path=temp_path / "qa.md",
            )

            self.assertEqual(result.geometry_status, "co_registered_to_ct_grid")
            self.assertEqual(result.label_count, 2)
            self.assertEqual(result.present_required_label_count, 2)
            self.assertGreaterEqual(result.fail_count, 1)
            self.assertGreaterEqual(result.bone_overlap_label_count, 1)
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            with Path(result.label_metrics_csv_path).open(newline="") as csvfile:
                rows = {int(row["label_id"]): row for row in csv.DictReader(csvfile)}
            self.assertEqual(rows[27]["status"], "pass")
            self.assertEqual(rows[4]["status"], "fail")

    def test_vessel_label_anatomy_qa_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            anatomy_path = temp_path / "anatomy.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            _write_nifti(anatomy_path, np.zeros((8, 8, 5), dtype=np.int16))
            _write_nifti(vessel_path, np.zeros((6, 8, 5), dtype=np.int16))

            with self.assertRaises(ValueError):
                qa_vessel_label_anatomy(
                    anatomy_labels_path=anatomy_path,
                    vessel_labels_path=vessel_path,
                    case_id="mismatch",
                    output_dir=temp_path / "qa",
                    required_vessel_labels=(4,),
                    report_path=temp_path / "mismatch.md",
                )


if __name__ == "__main__":
    unittest.main()
