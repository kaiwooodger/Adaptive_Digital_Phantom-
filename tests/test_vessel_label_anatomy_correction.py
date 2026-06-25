from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vessel_label_anatomy_correction import correct_vessel_label_anatomy
from phantom_twin.vessel_label_anatomy_qa import qa_vessel_label_anatomy


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))


class VesselLabelAnatomyCorrectionTests(unittest.TestCase):
    def test_correction_clears_bone_and_outside_body_while_preserving_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            anatomy = np.zeros((14, 14, 8), dtype=np.int16)
            anatomy[1:13, 1:13, 1:7] = 1
            anatomy[5:8, 5:8, 1:7] = 10
            anatomy[8:11, 6:9, 3:6] = 7
            vessel = np.zeros_like(anatomy)
            vessel[5:8, 5:8, 2:5] = 4
            vessel[8:10, 7:9, 3:5] = 27
            vessel[0, 0, 0] = 4
            anatomy_path = temp_path / "anatomy.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            config_path = temp_path / "labels.yaml"
            _write_nifti(anatomy_path, anatomy)
            _write_nifti(vessel_path, vessel)
            config_path.write_text(yaml.safe_dump({"labels": {4: "Aorta", 27: "Right renal artery"}}, sort_keys=False))

            result = correct_vessel_label_anatomy(
                anatomy_labels_path=anatomy_path,
                vessel_labels_path=vessel_path,
                vessel_label_config=config_path,
                case_id="toy_correction",
                output_dir=temp_path / "correction",
                max_regrow_iterations=6,
                report_path=temp_path / "correction.md",
            )

            corrected = np.rint(np.asanyarray(nib.load(result.corrected_vessel_path).dataobj)).astype(np.int16)
            bone = anatomy == 10
            body = anatomy != 0
            self.assertEqual(int(np.count_nonzero((corrected != 0) & bone)), 0)
            self.assertEqual(int(np.count_nonzero((corrected != 0) & ~body)), 0)
            self.assertIn(4, set(np.unique(corrected).astype(int).tolist()))
            self.assertIn(27, set(np.unique(corrected).astype(int).tolist()))
            self.assertEqual(result.corrected_invalid_voxels, 0)
            self.assertTrue(Path(result.label_correction_csv_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            qa = qa_vessel_label_anatomy(
                anatomy_labels_path=anatomy_path,
                vessel_labels_path=result.corrected_vessel_path,
                vessel_label_config=config_path,
                case_id="toy_correction_qa",
                output_dir=temp_path / "qa",
                required_vessel_labels=(4, 27),
                report_path=temp_path / "qa.md",
            )
            self.assertEqual(qa.bone_overlap_label_count, 0)

    def test_correction_rejects_geometry_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            anatomy_path = temp_path / "anatomy.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            _write_nifti(anatomy_path, np.zeros((8, 8, 4), dtype=np.int16))
            _write_nifti(vessel_path, np.zeros((8, 8, 4), dtype=np.int16), np.diag([2.0, 1.0, 1.0, 1.0]))

            with self.assertRaises(ValueError):
                correct_vessel_label_anatomy(
                    anatomy_labels_path=anatomy_path,
                    vessel_labels_path=vessel_path,
                    case_id="mismatch",
                    output_dir=temp_path / "correction",
                    report_path=temp_path / "mismatch.md",
                )


if __name__ == "__main__":
    unittest.main()
