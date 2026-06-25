from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.pca_phantom_set import build_approved_pca_phantom_set


class ApprovedPcaPhantomSetTests(unittest.TestCase):
    def test_build_approved_set_references_only_approved_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            atlas_spec = temp_path / "atlas.yaml"
            qa_decisions = temp_path / "qa.yaml"
            metrics_csv = temp_path / "metrics.csv"

            variants = [
                ("mean", None, 0.0),
                ("mode01_neg", 1, -1.0),
                ("mode01_pos", 1, 1.0),
                ("mode02_neg", 2, -1.0),
                ("mode02_pos", 2, 1.0),
            ]
            atlas_spec.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_modes",
                        "variants": [
                            {
                                "variant_id": variant_id,
                                "label": variant_id,
                                "mode_index": mode_index,
                                "mode_weight": mode_weight,
                                "material_labels": f"{variant_id}.nii.gz",
                                "preview_png": f"{variant_id}.png",
                                "body_volume_cm3": 1000.0,
                                "waist_cm": 50.0,
                                "vascular_components": 1,
                            }
                            for variant_id, mode_index, mode_weight in variants
                        ],
                    },
                    sort_keys=False,
                )
            )
            qa_decisions.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_modes",
                        "source_metrics_csv": str(metrics_csv),
                        "approved_mode_indices": [1],
                        "rejected_mode_indices": [2],
                        "decisions": [
                            {
                                "rank": 1,
                                "mode_index": 1,
                                "decision": "approved",
                                "score": 92.0,
                                "interpretation": "toy mode",
                                "variant_ids": ["mode01_neg", "mode01_pos"],
                                "issues": [],
                                "notes": ["mode_within_current_stage001_guardrails"],
                            },
                            {
                                "rank": 2,
                                "mode_index": 2,
                                "decision": "rejected",
                                "score": 40.0,
                                "interpretation": "bad mode",
                                "variant_ids": ["mode02_neg", "mode02_pos"],
                                "issues": ["score_below_approval_threshold"],
                                "notes": [],
                            },
                        ],
                    },
                    sort_keys=False,
                )
            )
            with metrics_csv.open("w", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=["variant_id", "body_volume_cm3", "waist_cm", "vascular_components"],
                )
                writer.writeheader()
                for variant_id, _, _ in variants:
                    writer.writerow(
                        {
                            "variant_id": variant_id,
                            "body_volume_cm3": 1000.0,
                            "waist_cm": 50.0,
                            "vascular_components": 1,
                        }
                    )

            result = build_approved_pca_phantom_set(
                qa_decisions_path=qa_decisions,
                atlas_spec_path=atlas_spec,
                output_dir=temp_path / "release",
                report_path=temp_path / "release.md",
            )

            self.assertEqual([item.variant_id for item in result.variants], ["mean", "mode01_neg", "mode01_pos"])
            self.assertEqual(result.approved_modes, (1,))
            self.assertEqual(result.rejected_modes, (2,))
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.metrics_csv_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
