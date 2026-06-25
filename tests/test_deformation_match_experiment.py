from pathlib import Path
import csv
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.deformation_match_experiment import run_deformation_match_experiment


def _write_nifti(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([2.0, 2.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(labels.astype(np.int16), affine), str(path))


class DeformationMatchExperimentTests(unittest.TestCase):
    def test_run_deformation_match_experiment_scores_scans_against_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ct_org_label = root / "ct_org_case0.nii.gz"
            labels = np.zeros((24, 24, 10), dtype=np.int16)
            labels[5:19, 6:18, 2:8] = 4
            labels[10:17, 10:17, 4:8] = 6
            labels[7:10, 13:16, 4:7] = 7
            _write_nifti(ct_org_label, labels)
            ct_org_manifest = root / "ct_org_manifest.csv"
            with ct_org_manifest.open("w", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=["case_id", "material_label_path"])
                writer.writeheader()
                writer.writerow({"case_id": "ct_org_case0", "material_label_path": str(ct_org_label)})

            reg_label = root / "reg_label.nii.gz"
            _write_nifti(reg_label, labels)
            reg_manifest_csv = root / "reg_manifest.csv"
            with reg_manifest_csv.open("w", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=["target_case_id", "moving_case_id", "label_path", "image_path"])
                writer.writeheader()
                for moving in ["0001", "0002"]:
                    writer.writerow({"target_case_id": "0061", "moving_case_id": moving, "label_path": str(reg_label), "image_path": ""})
            reg_manifest = root / "reg_manifest.yaml"
            reg_manifest.write_text(yaml.safe_dump({"outputs": {"manifest_csv": str(reg_manifest_csv)}}))

            avt_manifest = root / "aorta_manifest.csv"
            with avt_manifest.open("w", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=["case_id", "aorta_mask_nifti_path", "aorta_volume_ml", "aorta_z_span_mm"])
                writer.writeheader()
                writer.writerow({"case_id": "aorta_case", "aorta_mask_nifti_path": "aorta.nii.gz", "aorta_volume_ml": 12.0, "aorta_z_span_mm": 100.0})

            profile_dir = root / "profiles" / "bmi27" / "anthropometry"
            profile_dir.mkdir(parents=True)
            profile_spec = profile_dir / "profile_anthro_morph_spec_v001.yaml"
            profile_spec.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_profile",
                        "anthropometry": {
                            "target_bmi": 27.0,
                            "target_height_cm": 175.0,
                            "target_waist_cm": 80.0,
                            "achieved_waist_cm": 80.0,
                        },
                        "quality_summary": {
                            "morphed_body_volume_cm3": 8_000.0,
                            "morphed_bbox_mm": [48.0, 48.0, 40.0],
                        },
                        "region_stats": [
                            {"name": "liver", "morphed_volume_cm3": 120.0},
                            {"name": "kidneys", "morphed_volume_cm3": 20.0},
                            {"name": "blood_equivalent_fluid", "morphed_volume_cm3": 12.0},
                        ],
                    }
                )
            )

            result = run_deformation_match_experiment(
                experiment_id="toy_deformation_match",
                output_dir=root / "experiment",
                ct_org_manifest_csv=ct_org_manifest,
                btcv_label_path=root / "missing_btcv.nii.gz",
                reg_training_manifest_path=reg_manifest,
                avt_aorta_manifest_csv=avt_manifest,
                profile_spec_glob=str(profile_spec),
                include_metric_scaled_height_grid=False,
                report_path=root / "report.md",
            )

            self.assertEqual(result.scan_count, 3)
            self.assertEqual(result.profile_count, 1)
            self.assertEqual(result.match_count, 3)
            self.assertTrue(Path(result.scan_metrics_csv_path).exists())
            self.assertTrue(Path(result.profile_metrics_csv_path).exists())
            self.assertTrue(Path(result.match_matrix_csv_path).exists())
            self.assertTrue(Path(result.top_matches_csv_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.atlas_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
