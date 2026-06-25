from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from phantom_twin.vessel_ctgrid_resample import resample_vessel_to_ct_grid


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))


class VesselCtGridResampleTests(unittest.TestCase):
    def test_header_affine_resample_preserves_aligned_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ct = np.zeros((8, 8, 5), dtype=np.float32)
            ct[2:6, 2:6, 1:4] = 40.0
            vessel = np.zeros(ct.shape, dtype=np.int16)
            vessel[3:5, 3:5, 1:4] = 4
            ct_path = temp_path / "ct.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            _write_nifti(ct_path, ct)
            _write_nifti(vessel_path, vessel)

            result = resample_vessel_to_ct_grid(
                ct_path=ct_path,
                vessel_seg_path=vessel_path,
                case_id="aligned",
                output_dir=temp_path / "resample",
                alignment_mode="header-affine",
                required_vessel_labels=(4,),
                report_path=temp_path / "aligned.md",
            )

            output = np.asanyarray(nib.load(result.resampled_vessel_path).dataobj)
            self.assertEqual(output.shape, ct.shape)
            self.assertEqual(result.source_geometry_status, "co_registered_to_ct_grid")
            self.assertEqual(result.output_geometry_status, "co_registered_to_ct_grid")
            self.assertEqual(result.vessel_label_coverage_percent, 100.0)
            self.assertEqual(set(np.unique(output).astype(int).tolist()), {0, 4})
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())

    def test_centered_bbox_places_mismatched_source_on_ct_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ct = np.full((12, 12, 8), -1000.0, dtype=np.float32)
            ct[2:10, 2:10, 1:7] = 30.0
            target_mask = np.zeros(ct.shape, dtype=np.int16)
            target_mask[2:10, 2:10, 1:7] = 1
            vessel = np.zeros((6, 6, 4), dtype=np.int16)
            vessel[2:4, 2:4, 1:3] = 4
            ct_path = temp_path / "ct.nii.gz"
            target_mask_path = temp_path / "organ.nii.gz"
            vessel_path = temp_path / "source_vessel.nii.gz"
            _write_nifti(ct_path, ct)
            _write_nifti(target_mask_path, target_mask)
            _write_nifti(vessel_path, vessel, np.diag([2.0, 2.0, 2.0, 1.0]))

            result = resample_vessel_to_ct_grid(
                ct_path=ct_path,
                vessel_seg_path=vessel_path,
                target_mask_path=target_mask_path,
                case_id="centered",
                output_dir=temp_path / "resample",
                alignment_mode="centered-bbox",
                required_vessel_labels=(4,),
                report_path=temp_path / "centered.md",
            )

            output = np.asanyarray(nib.load(result.resampled_vessel_path).dataobj)
            self.assertEqual(output.shape, ct.shape)
            self.assertGreater(result.output_nonzero_voxels, 0)
            self.assertEqual(result.output_geometry_status, "co_registered_to_ct_grid")
            self.assertEqual(result.vessel_label_coverage_percent, 100.0)
            self.assertIn("centered-bbox_mode_is_template_placement", ";".join(result.notes))


if __name__ == "__main__":
    unittest.main()
