from pathlib import Path
import csv
import tempfile
import unittest

from phantom_twin.validation_case_staging import stage_validation_case
from phantom_twin.validation_discovery import discover_validation_candidates


class ValidationCaseStagingTests(unittest.TestCase):
    def test_stage_validation_case_references_inputs_and_writes_intake_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ct = temp_path / "ct.nii.gz"
            cta = temp_path / "cta.nii.gz"
            organ = temp_path / "organ.nii.gz"
            vessel = temp_path / "vessel.nii.gz"
            for path in (ct, cta, organ, vessel):
                path.write_text("placeholder")

            result = stage_validation_case(
                case_id="Toy Case 01",
                source_dataset="toy_public_case",
                ct_path=ct,
                cta_path=cta,
                organ_seg_path=organ,
                vessel_seg_path=vessel,
                output_dir=temp_path / "cases",
                required_vessel_labels=(1, 4),
                access_status="open",
                notes="unit test",
                report_path=temp_path / "stage.md",
            )

            self.assertEqual(result.case_id, "toy_case_01")
            self.assertEqual(result.completeness_status, "complete_ready_for_intake_qa")
            self.assertIn("ct", result.present_roles)
            self.assertIn("cta_or_ctv", result.present_roles)
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.intake_case_csv_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            with Path(result.intake_case_csv_path).open(newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))
            self.assertEqual(rows[0]["case_id"], "toy_case_01")
            self.assertEqual(rows[0]["ct_path"], str(ct))
            self.assertEqual(rows[0]["cta_path"], str(cta))
            self.assertEqual(rows[0]["required_vessel_labels"], "1,4")

    def test_discovery_recognizes_staged_validation_case_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ct = temp_path / "ct.nii.gz"
            ctv = temp_path / "ctv.nii.gz"
            organ = temp_path / "organ.nii.gz"
            vessel = temp_path / "vessel.nii.gz"
            for path in (ct, ctv, organ, vessel):
                path.write_text("placeholder")

            stage = stage_validation_case(
                case_id="discoverable_case",
                source_dataset="toy_case",
                ct_path=ct,
                ctv_path=ctv,
                organ_seg_path=organ,
                vessel_seg_path=vessel,
                output_dir=temp_path / "cases",
                required_vessel_labels=(1, 2),
                report_path=temp_path / "stage.md",
            )
            discovery = discover_validation_candidates(
                search_roots=(Path(stage.output_dir),),
                output_dir=temp_path / "discovery",
                discovery_id="discover_staged_case",
                required_vessel_labels=(1, 2),
                max_loose_nifti_cases=0,
                report_path=temp_path / "discovery.md",
            )

            self.assertEqual(discovery.candidate_count, 1)
            self.assertEqual(discovery.complete_candidate_count, 1)
            self.assertEqual(discovery.candidates[0].case_id, "discoverable_case")
            self.assertEqual(discovery.candidates[0].source_dataset, "toy_case")
            self.assertIn("vascular_image", discovery.candidates[0].present_roles)


if __name__ == "__main__":
    unittest.main()
