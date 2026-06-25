from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from phantom_twin.ct_org_cohort import stage_ct_org_label_cohort


class CtOrgCohortStagingTests(unittest.TestCase):
    def test_stage_ct_org_label_cohort_materializes_labels_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_dir = temp_path / "raw"
            raw_dir.mkdir()
            labels = np.zeros((32, 32, 10), dtype=np.int16)
            labels[10:20, 9:14, 3:7] = 3
            labels[12:18, 18:24, 4:8] = 1
            labels[15:17, 15:19, 2:9] = 5
            nib.save(nib.Nifti1Image(labels, np.diag([2.0, 2.0, 5.0, 1.0])), raw_dir / "labels-0.nii.gz")

            result = stage_ct_org_label_cohort(
                case_indices=(0,),
                raw_label_dir=raw_dir,
                output_dir=temp_path / "processed",
                body_padding_mm=12.0,
                adipose_layer_mm=6.0,
                report_path=temp_path / "report.md",
            )

            self.assertEqual(len(result.case_results), 1)
            self.assertTrue(Path(result.case_results[0].material_label_path).exists())
            self.assertTrue(Path(result.case_results[0].preview_png_path).exists())
            self.assertTrue(Path(result.manifest_csv_path).exists())
            self.assertGreater(result.case_results[0].body_voxels, int((labels > 0).sum()))


if __name__ == "__main__":
    unittest.main()
