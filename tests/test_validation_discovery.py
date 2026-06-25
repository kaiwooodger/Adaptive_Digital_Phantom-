from pathlib import Path
import csv
import tempfile
import unittest

import nibabel as nib
import numpy as np

from phantom_twin.validation_discovery import discover_validation_candidates
from phantom_twin.validation_intake import build_validation_intake_package


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


class ValidationDiscoveryTests(unittest.TestCase):
    def test_known_manifest_discovery_writes_candidate_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            organ = temp_path / "ct_org_case0_material.nii.gz"
            cta = temp_path / "medseg_img.nii.gz"
            vessel = temp_path / "medseg_msk.nii.gz"
            ct = temp_path / "adapter_ct.nii.gz"
            adapter_organ = temp_path / "adapter_organ.nii.gz"
            adapter_vessel = temp_path / "adapter_vessel.nii.gz"
            for path in (organ, cta, vessel, ct, adapter_organ, adapter_vessel):
                path.touch()

            ct_org_manifest = temp_path / "ct_org_label_population_manifest_v001.csv"
            ct_org_manifest.write_text(
                "case_id,raw_label_path,material_label_path\n"
                f"ct_org_case0,,{organ}\n"
            )
            medseg_manifest = temp_path / "medseg_abdominal_vasculature_case001_manifest_v001.yaml"
            medseg_manifest.write_text(
                "\n".join(
                    [
                        "case_id: medseg_case",
                        "dataset: medseg_abdominal_vasculature",
                        f"image_path: {cta}",
                        f"mask_path: {vessel}",
                        "label_config_path: configs/labelmaps/medseg_abdominal_vasculature.yaml",
                    ]
                )
            )
            patient_manifest = temp_path / "toy_patient_input_manifest_v001.yaml"
            patient_manifest.write_text(
                "\n".join(
                    [
                        "case_id: adapter_case",
                        "package_type: patient_phantom_input_adapter",
                        "inputs:",
                        "  - role: ct",
                        f"    staged_path: {ct}",
                        "  - role: organ_seg",
                        f"    staged_path: {adapter_organ}",
                        "  - role: vessel_seg",
                        f"    staged_path: {adapter_vessel}",
                    ]
                )
            )

            result = discover_validation_candidates(
                search_roots=(temp_path,),
                output_dir=temp_path / "discovery",
                discovery_id="toy_discovery",
                required_vessel_labels=(1, 4),
                max_ct_org_cases=1,
                max_loose_nifti_cases=0,
                report_path=temp_path / "discovery.md",
            )

            self.assertEqual(result.candidate_count, 3)
            self.assertEqual(result.complete_candidate_count, 0)
            self.assertEqual(result.partial_candidate_count, 3)
            statuses = {candidate.case_id: candidate.discovery_status for candidate in result.candidates}
            self.assertEqual(statuses["ct_org_case0_label_only"], "partial_local_candidate")
            self.assertEqual(statuses["medseg_case"], "partial_local_candidate")
            self.assertEqual(statuses["adapter_case"], "partial_local_candidate")
            with Path(result.candidates_csv_path).open(newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))
            self.assertEqual(len(rows), 3)
            self.assertIn("ct_path", rows[0])
            self.assertIn("discovery_status", rows[0])
            self.assertTrue(Path(result.summary_csv_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())

    def test_loose_nifti_discovery_csv_feeds_intake_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            case_dir = temp_path / "case_complete"
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            ct = np.zeros((8, 8, 6), dtype=np.float32)
            ct[2:6, 2:6, 1:5] = 42.0
            organ = np.zeros(ct.shape, dtype=np.int16)
            organ[2:6, 2:6, 1:5] = 4
            vessel = np.zeros(ct.shape, dtype=np.int16)
            vessel[3:5, 3:5, 1:5] = 1
            vessel[4:6, 4:6, 3:5] = 4

            _write_nifti(case_dir / "toy_ct.nii.gz", ct, affine)
            _write_nifti(case_dir / "toy_cta.nii.gz", ct + 100.0, affine)
            _write_nifti(case_dir / "toy_organ_seg.nii.gz", organ, affine)
            _write_nifti(case_dir / "toy_vessel_seg.nii.gz", vessel, affine)

            discovery = discover_validation_candidates(
                search_roots=(case_dir,),
                output_dir=temp_path / "discovery",
                discovery_id="loose_discovery",
                required_vessel_labels=(1, 4),
                max_loose_nifti_cases=5,
                report_path=temp_path / "discovery.md",
            )
            self.assertEqual(discovery.candidate_count, 1)
            self.assertEqual(discovery.complete_candidate_count, 1)

            intake = build_validation_intake_package(
                cases_csv_path=discovery.candidates_csv_path,
                output_dir=temp_path / "intake",
                intake_id="loose_intake",
                required_vessel_labels=(1, 4),
                report_path=temp_path / "intake.md",
            )
            self.assertEqual(intake.case_count, 1)
            self.assertEqual(intake.ready_case_count, 1)
            self.assertEqual(intake.cases[0].readiness_status, "ready_for_p1_patient_specific_validation")


if __name__ == "__main__":
    unittest.main()
