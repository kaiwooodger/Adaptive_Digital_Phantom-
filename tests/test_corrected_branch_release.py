from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.corrected_branch_release import build_corrected_branch_release_package


class CorrectedBranchReleaseTests(unittest.TestCase):
    def test_release_copies_small_artifacts_and_indexes_referenced_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small_spec = root / "spec.yaml"
            volume = root / "dose.nii.gz"
            figure = root / "preview.png"
            report = root / "report.md"
            volume.write_bytes(b"fake nifti volume")
            figure.write_bytes(b"fake png")
            report.write_text("# Report\n")
            small_spec.write_text(
                yaml.safe_dump(
                    {
                        "outputs": {
                            "dose": str(volume),
                            "preview": str(figure),
                        }
                    },
                    sort_keys=False,
                )
            )
            status_manifest = root / "status.yaml"
            status_manifest.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_corrected",
                        "outputs": {
                            "report": str(report),
                            "manifest_yaml": str(status_manifest),
                        },
                        "artifacts": {
                            "small_spec": {"path": str(small_spec), "status": "exists"},
                            "preview": {"path": str(figure), "status": "exists"},
                        },
                        "summary": {
                            "flow": {"aorta_flow_mean_ml_s": 80.0},
                            "rt_flow": {"selected_edge_count": 2},
                            "gamma": {"min_pass_rate_percent": 100.0},
                        },
                    },
                    sort_keys=False,
                )
            )

            result = build_corrected_branch_release_package(
                status_manifest_path=status_manifest,
                output_dir=root / "release",
                case_id="toy_corrected",
                large_file_threshold_mb=1.0,
                report_path=root / "release_report.md",
            )

            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.artifact_index_csv_path).exists())
            self.assertTrue(Path(result.readme_markdown_path).exists())
            self.assertGreaterEqual(result.summary["artifact_count"], 4)
            self.assertGreaterEqual(result.summary["copied_artifact_count"], 3)
            self.assertGreaterEqual(result.summary["indexed_large_or_volume_artifact_count"], 1)
            with Path(result.artifact_index_csv_path).open(newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))
            volume_rows = [row for row in rows if row["source_path"] == str(volume)]
            self.assertEqual(volume_rows[0]["copy_policy"], "indexed_only_large_or_volume")
            copied_rows = [row for row in rows if row["source_path"] == str(small_spec)]
            self.assertTrue(copied_rows[0]["packaged_path"])


if __name__ == "__main__":
    unittest.main()
