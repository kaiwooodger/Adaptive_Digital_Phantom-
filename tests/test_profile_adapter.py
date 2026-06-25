from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.profile_adapter import build_user_profile_adapter


class UserProfileAdapterTests(unittest.TestCase):
    def test_profile_adapter_scores_and_selects_matching_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            combined_spec = temp_path / "combined.yaml"
            atlas_spec = temp_path / "atlas.yaml"
            manifest_path = temp_path / "approved.yaml"
            combined_spec.write_text(yaml.safe_dump({"case_id": "toy_combined"}))
            atlas_spec.write_text(yaml.safe_dump({"case_id": "toy_modes", "source_combined_spec": str(combined_spec)}))
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_modes",
                        "source_atlas_spec": str(atlas_spec),
                        "variants": [
                            {
                                "variant_id": "mean",
                                "label": "Mean",
                                "release_role": "baseline",
                                "warning_status": "clean",
                                "body_volume_cm3": 30000.0,
                                "waist_cm": 100.0,
                                "material_labels": "mean.nii.gz",
                                "preview_png": "mean.png",
                            },
                            {
                                "variant_id": "small",
                                "label": "Small",
                                "release_role": "approved_primary",
                                "warning_status": "clean",
                                "body_volume_cm3": 28000.0,
                                "waist_cm": 95.0,
                                "material_labels": "small.nii.gz",
                                "preview_png": "small.png",
                            },
                            {
                                "variant_id": "large",
                                "label": "Large",
                                "release_role": "approved_primary",
                                "warning_status": "clean",
                                "body_volume_cm3": 36000.0,
                                "waist_cm": 110.0,
                                "material_labels": "large.nii.gz",
                                "preview_png": "large.png",
                            },
                        ],
                    },
                    sort_keys=False,
                )
            )

            result = build_user_profile_adapter(
                approved_set_manifest_path=manifest_path,
                output_dir=temp_path / "profile",
                profile_id="waist110",
                case_id="toy_profile",
                target_height_cm=170.0,
                target_bmi=29.0,
                target_waist_cm=110.0,
                baseline_height_cm=170.0,
                baseline_bmi=24.0,
                report_path=temp_path / "profile.md",
            )

            self.assertEqual(result.selected_variant_id, "large")
            self.assertEqual(result.fit_status, "matched_to_approved_variant")
            self.assertTrue(Path(result.profile_yaml_path).exists())
            self.assertTrue(Path(result.score_csv_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            profile = yaml.safe_load(Path(result.profile_yaml_path).read_text())
            self.assertEqual(profile["selection"]["selected_variant_id"], "large")
            self.assertIn("build-anthropometric-torso-morph", "\n".join(result.recommended_commands))


if __name__ == "__main__":
    unittest.main()
