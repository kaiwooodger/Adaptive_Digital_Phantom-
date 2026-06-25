from pathlib import Path
import csv
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.validation_case_promotion import promote_harmonized_vessel_case
from phantom_twin.validation_case_staging import stage_validation_case
from phantom_twin.vessel_label_harmonizer import harmonize_vessel_labels


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


class ValidationCasePromotionTests(unittest.TestCase):
    def test_promote_harmonized_vessel_replaces_staged_vessel_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ct = np.zeros((8, 8, 4), dtype=np.float32)
            ct[2:6, 2:6, 1:3] = 40.0
            organ = np.zeros(ct.shape, dtype=np.int16)
            organ[2:6, 2:6, 1:3] = 4
            cta = ct + 100.0
            vessel = np.zeros(ct.shape, dtype=np.int16)
            vessel[3:5, 3:5, 1:3] = 10

            ct_path = temp_path / "ct.nii.gz"
            cta_path = temp_path / "cta.nii.gz"
            organ_path = temp_path / "organ.nii.gz"
            vessel_path = temp_path / "vessel_source.nii.gz"
            _write_nifti(ct_path, ct)
            _write_nifti(cta_path, cta)
            _write_nifti(organ_path, organ)
            _write_nifti(vessel_path, vessel)

            label_config = temp_path / "labels.yaml"
            label_config.write_text(yaml.safe_dump({"labels": {4: "Aorta"}}, sort_keys=False))
            mapping = temp_path / "mapping.csv"
            mapping.write_text("source_label,target_label\n10,4\n")

            staged = stage_validation_case(
                case_id="toy_base",
                source_dataset="toy",
                ct_path=ct_path,
                cta_path=cta_path,
                organ_seg_path=organ_path,
                vessel_seg_path=vessel_path,
                vessel_label_config=label_config,
                output_dir=temp_path / "cases",
                required_vessel_labels=(4,),
                report_path=temp_path / "stage.md",
            )
            harmonized = harmonize_vessel_labels(
                vessel_seg_path=vessel_path,
                case_id="toy_harmonized",
                output_dir=temp_path / "harmonized",
                target_label_config=label_config,
                mapping_csv_path=mapping,
                required_vessel_labels=(4,),
                report_path=temp_path / "harmonized.md",
            )

            promoted = promote_harmonized_vessel_case(
                staged_case_manifest_path=staged.manifest_yaml_path,
                vessel_harmonization_manifest_path=harmonized.manifest_yaml_path,
                promoted_case_id="toy_promoted",
                output_dir=temp_path / "cases",
                report_path=temp_path / "promoted.md",
            )

            self.assertEqual(promoted.promoted_case_id, "toy_promoted")
            self.assertEqual(promoted.geometry_status, "co_registered_to_ct_grid")
            self.assertEqual(promoted.promoted_completeness_status, "complete_ready_for_intake_qa")
            self.assertTrue(Path(promoted.promoted_manifest_yaml_path).exists())
            self.assertTrue(Path(promoted.promoted_intake_case_csv_path).exists())
            with Path(promoted.promoted_intake_case_csv_path).open(newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))
            self.assertEqual(rows[0]["vessel_seg_path"], harmonized.harmonized_nifti_path)
            self.assertEqual(rows[0]["required_vessel_labels"], "4")
            self.assertTrue(Path(promoted.report_path).exists())


if __name__ == "__main__":
    unittest.main()
