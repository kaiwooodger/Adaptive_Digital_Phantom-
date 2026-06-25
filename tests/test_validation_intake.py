from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from phantom_twin.validation_intake import build_validation_intake_package


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


class ValidationIntakeTests(unittest.TestCase):
    def test_validation_intake_scores_coregistered_branch_label_case_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([1.5, 1.5, 2.0, 1.0])
            ct = np.zeros((12, 12, 8), dtype=np.float32)
            ct[2:10, 2:10, 1:7] = 40.0
            organ = np.zeros(ct.shape, dtype=np.int16)
            organ[2:10, 2:10, 1:7] = 4
            organ[3:5, 3:5, 2:5] = 7
            vessel = np.zeros(ct.shape, dtype=np.int16)
            vessel[5:7, 5:7, 1:7] = 4
            vessel[4:6, 6:8, 3:5] = 1
            vessel[6:8, 4:6, 3:5] = 27

            ct_path = temp_path / "ct.nii.gz"
            cta_path = temp_path / "cta.nii.gz"
            organ_path = temp_path / "organ.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(cta_path, ct + 100.0, affine)
            _write_nifti(organ_path, organ, affine)
            _write_nifti(vessel_path, vessel, affine)

            cases_csv = temp_path / "cases.csv"
            cases_csv.write_text(
                "case_id,source_dataset,ct_path,cta_path,ctv_path,organ_seg_path,vessel_seg_path,vessel_label_config,required_vessel_labels,access_status,notes\n"
                f"toy_case,local,{ct_path},{cta_path},,{organ_path},{vessel_path},,\"1,4,27\",approved,\n"
            )

            result = build_validation_intake_package(
                cases_csv_path=cases_csv,
                output_dir=temp_path / "intake",
                intake_id="toy_p1_intake",
                required_vessel_labels=(1, 4, 27),
                report_path=temp_path / "intake.md",
            )

            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.ready_case_count, 1)
            self.assertEqual(result.review_case_count, 0)
            self.assertEqual(result.cases[0].readiness_status, "ready_for_p1_patient_specific_validation")
            self.assertEqual(result.cases[0].vessel_label_coverage_percent, 100.0)
            self.assertTrue(Path(result.template_csv_path).exists())
            self.assertTrue(Path(result.case_summary_csv_path).exists())
            self.assertTrue(Path(result.dataset_requirements_csv_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())

    def test_validation_intake_without_cases_writes_blank_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = build_validation_intake_package(
                output_dir=temp_path / "intake",
                intake_id="blank_p1_intake",
                required_vessel_labels=(1, 4),
                report_path=temp_path / "blank.md",
            )

            self.assertEqual(result.case_count, 0)
            self.assertTrue(Path(result.template_csv_path).exists())
            self.assertIn("candidate_case_001", Path(result.template_csv_path).read_text())
            self.assertIn("No candidate cases", Path(result.report_path).read_text())

    def test_validation_intake_defers_anatomy_only_case_without_ct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            organ_path = temp_path / "organ_only.nii.gz"
            organ_path.write_text("not a nifti but should not be loaded without primary CT")
            cases_csv = temp_path / "cases.csv"
            cases_csv.write_text(
                "case_id,source_dataset,ct_path,cta_path,ctv_path,organ_seg_path,vessel_seg_path,vessel_label_config,required_vessel_labels,access_status,notes\n"
                f"organ_only,local,,,,{organ_path},,,\"1,4\",approved,\n"
            )

            result = build_validation_intake_package(
                cases_csv_path=cases_csv,
                output_dir=temp_path / "intake",
                intake_id="organ_only_intake",
                required_vessel_labels=(1, 4),
                report_path=temp_path / "organ_only.md",
            )

            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.missing_case_count, 1)
            self.assertEqual(result.cases[0].organ_seg_status, "metadata_deferred")
            self.assertIn("Stage primary CT", result.cases[0].recommended_action)

    def test_validation_intake_requires_vascular_image_for_ready_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([1.5, 1.5, 2.0, 1.0])
            ct = np.zeros((10, 10, 6), dtype=np.float32)
            ct[2:8, 2:8, 1:5] = 50.0
            organ = np.zeros(ct.shape, dtype=np.int16)
            organ[2:8, 2:8, 1:5] = 4
            vessel = np.zeros(ct.shape, dtype=np.int16)
            vessel[4:6, 4:6, 1:5] = 1
            vessel[6:8, 4:6, 2:4] = 4

            ct_path = temp_path / "ct.nii.gz"
            organ_path = temp_path / "organ.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(organ_path, organ, affine)
            _write_nifti(vessel_path, vessel, affine)
            cases_csv = temp_path / "cases.csv"
            cases_csv.write_text(
                "case_id,source_dataset,ct_path,cta_path,ctv_path,organ_seg_path,vessel_seg_path,vessel_label_config,required_vessel_labels,access_status,notes\n"
                f"no_vascular_image,local,{ct_path},,,{organ_path},{vessel_path},,\"1,4\",approved,\n"
            )

            result = build_validation_intake_package(
                cases_csv_path=cases_csv,
                output_dir=temp_path / "intake",
                intake_id="no_vascular_image_intake",
                required_vessel_labels=(1, 4),
                report_path=temp_path / "no_vascular_image.md",
            )

            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.ready_case_count, 0)
            self.assertEqual(result.review_case_count, 1)
            self.assertEqual(result.cases[0].readiness_status, "registration_or_label_review_required")
            self.assertIn("Stage CTA", result.cases[0].recommended_action)


if __name__ == "__main__":
    unittest.main()
