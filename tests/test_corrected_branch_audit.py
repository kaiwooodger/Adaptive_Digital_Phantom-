from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.corrected_branch_audit import (
    EXPECTED_EVIDENCE_GROUPS,
    audit_corrected_branch_release_package,
)


class CorrectedBranchAuditTests(unittest.TestCase):
    def test_audit_scores_corrected_release_with_clinical_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_index = root / "artifact_index.csv"
            readme = root / "README.md"
            limitations = root / "limitations.md"
            command_log = root / "commands.md"
            release_report = root / "release_report.md"
            manifest = root / "release_manifest.yaml"
            readme.write_text("# Toy corrected release\n")
            limitations.write_text(
                "This is not for clinical use. RT outputs are synthetic and not TPS or Monte Carlo dose. "
                "Flow is not calibrated 3D CFD."
            )
            command_log.write_text(
                "phantom-twin build-corrected-branch-status-report\n"
                "phantom-twin build-corrected-branch-release-package\n"
            )
            release_report.write_text("# Release report\n")

            rows: list[dict[str, object]] = []
            for index, group in enumerate(EXPECTED_EVIDENCE_GROUPS):
                if group == "corrected_vascular_domain":
                    source = root / f"{group}.nii.gz"
                    source.write_bytes(b"toy nifti")
                    file_type = "nifti_volume"
                    copy_policy = "indexed_only_large_or_volume"
                    sha256 = ""
                else:
                    source = root / f"{group}.yaml"
                    source.write_text("ok: true\n")
                    file_type = "yaml"
                    copy_policy = "copied"
                    sha256 = f"toysha{index}"
                rows.append(
                    {
                        "group": group,
                        "role": f"{group}_role",
                        "file_type": file_type,
                        "source_path": str(source),
                        "exists": True,
                        "size_bytes": source.stat().st_size,
                        "sha256": sha256,
                        "copy_policy": copy_policy,
                        "packaged_path": str(root / "package" / source.name) if copy_policy == "copied" else "",
                        "notes": "",
                    }
                )

            with artifact_index.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            manifest.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_corrected_flow",
                        "release_id": "toy_corrected_rc1",
                        "outputs": {
                            "manifest_yaml": str(manifest),
                            "artifact_index_csv": str(artifact_index),
                            "readme_markdown": str(readme),
                            "limitations_markdown": str(limitations),
                            "command_log": str(command_log),
                            "report": str(release_report),
                        },
                        "summary": {
                            "artifact_count": len(rows),
                            "copied_artifact_count": len([row for row in rows if row["copy_policy"] == "copied"]),
                            "indexed_large_or_volume_artifact_count": 1,
                            "missing_artifact_count": 0,
                            "copied_total_size_bytes": 2048,
                            "status_summary": {
                                "vascular_domain": {
                                    "arterial_voxels": 10,
                                    "venous_voxels": 12,
                                    "vessel_wall_voxels": 5,
                                    "snapped_boundary_nodes": 3,
                                    "unclassified_labels": [],
                                },
                                "rt_material_package": {
                                    "vascular_fluid_volume_cm3": 10.0,
                                    "vessel_wall_volume_cm3": 1.0,
                                    "ptv_volume_cm3": 2.0,
                                },
                                "flow": {
                                    "node_count": 4,
                                    "edge_count": 3,
                                    "boundary_count": 3,
                                    "aorta_flow_mean_ml_s": 80.0,
                                    "aorta_flow_min_ml_s": 55.0,
                                    "aorta_flow_max_ml_s": 130.0,
                                    "max_mass_balance_residual_ml_s": 1e-8,
                                    "max_outlet_split_range_pp": 1.0,
                                },
                                "flow4d": {
                                    "frame_count": 24,
                                    "color_min": 1.0,
                                    "color_max": 40.0,
                                    "animation_gif": str(root / "flow.gif"),
                                },
                                "rt_flow": {
                                    "selected_edge_count": 3,
                                    "top_coupled_edge": "aorta",
                                    "max_peak_delta_mgy": 100.0,
                                    "max_trough_delta_mgy": 50.0,
                                    "ptv_peak_v95_percent": 99.0,
                                },
                                "gamma": {
                                    "min_pass_rate_percent": 100.0,
                                    "max_p95_gamma": 0.1,
                                },
                            },
                        },
                    },
                    sort_keys=False,
                )
            )

            result = audit_corrected_branch_release_package(
                release_manifest_path=manifest,
                output_dir=root / "audit",
                audit_id="toy_corrected_readiness",
                report_path=root / "audit.md",
            )

            self.assertEqual(result.readiness_tier, "research_ready_clinical_validation_required")
            self.assertGreaterEqual(result.clinical_blocker_count, 5)
            self.assertEqual(sum(check.status == "fail" for check in result.checks), 0)
            self.assertGreaterEqual(result.research_score_percent, 99.0)
            self.assertTrue(Path(result.audit_yaml_path).exists())
            self.assertTrue(Path(result.scorecard_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
