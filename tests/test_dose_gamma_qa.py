from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.dose_gamma_qa import build_dose_gamma_qa


class DoseGammaQATests(unittest.TestCase):
    def test_dose_gamma_qa_exports_summary_and_maps(self) -> None:
        try:
            import pymedphys  # noqa: F401
        except ImportError:
            self.skipTest("pymedphys is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            reference = np.zeros((8, 8, 4), dtype=np.float32)
            reference[2:6, 2:6, 1:3] = 10.0
            evaluated = reference.copy()
            evaluated[3:5, 3:5, 1:3] -= 0.15
            reference_path = temp_path / "reference.nii.gz"
            evaluated_path = temp_path / "evaluated.nii.gz"
            nib.save(nib.Nifti1Image(reference, affine), reference_path)
            nib.save(nib.Nifti1Image(evaluated, affine), evaluated_path)

            config_path = temp_path / "pymedphys_eval.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy",
                        "reference_dose": {"state": "static", "nifti": str(reference_path)},
                        "evaluated_doses": [
                            {"state": "evaluated", "nifti": str(evaluated_path)},
                        ],
                        "gamma_defaults": {
                            "dose_percent_threshold": 3.0,
                            "distance_mm_threshold": 3.0,
                            "lower_percent_dose_cutoff": 10.0,
                            "local_gamma": False,
                        },
                    }
                )
            )

            result = build_dose_gamma_qa(
                pymedphys_eval_config_path=config_path,
                output_dir=temp_path / "qa",
                case_id="toy",
                random_subset=None,
                interp_fraction=2.0,
                report_path=temp_path / "qa.md",
            )

            self.assertTrue(Path(result.summary_csv_path).exists())
            self.assertTrue(Path(result.spec_yaml_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertEqual(len(result.state_results), 1)
            self.assertGreater(result.state_results[0].finite_gamma_points, 0)
            self.assertGreater(result.state_results[0].pass_rate_percent, 99.0)
            self.assertTrue(Path(result.state_results[0].gamma_map_path).exists())
            self.assertTrue(Path(result.state_results[0].dose_difference_path).exists())

    def test_dose_gamma_qa_can_skip_heavy_volume_outputs(self) -> None:
        try:
            import pymedphys  # noqa: F401
        except ImportError:
            self.skipTest("pymedphys is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            reference = np.zeros((8, 8, 4), dtype=np.float32)
            reference[2:6, 2:6, 1:3] = 10.0
            evaluated = reference.copy()
            evaluated[3:5, 3:5, 1:3] -= 0.15
            reference_path = temp_path / "reference.nii.gz"
            evaluated_path = temp_path / "evaluated.nii.gz"
            nib.save(nib.Nifti1Image(reference, affine), reference_path)
            nib.save(nib.Nifti1Image(evaluated, affine), evaluated_path)

            config_path = temp_path / "pymedphys_eval.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy",
                        "reference_dose": {"state": "static", "nifti": str(reference_path)},
                        "evaluated_doses": [
                            {"state": "evaluated", "nifti": str(evaluated_path)},
                        ],
                    }
                )
            )

            result = build_dose_gamma_qa(
                pymedphys_eval_config_path=config_path,
                output_dir=temp_path / "qa",
                case_id="toy",
                random_subset=None,
                interp_fraction=2.0,
                write_volume_outputs=False,
                report_path=temp_path / "qa.md",
            )
            spec = yaml.safe_load(Path(result.spec_yaml_path).read_text())

            self.assertFalse(result.volume_outputs_written)
            self.assertIsNone(result.state_results[0].gamma_map_path)
            self.assertIsNone(result.state_results[0].dose_difference_path)
            self.assertFalse((temp_path / "qa" / "gamma_maps").exists())
            self.assertFalse((temp_path / "qa" / "dose_difference").exists())
            self.assertFalse(spec["outputs"]["volume_outputs_written"])


if __name__ == "__main__":
    unittest.main()
