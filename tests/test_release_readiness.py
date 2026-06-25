from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.release_readiness import audit_research_release_package


class ReleaseReadinessAuditTests(unittest.TestCase):
    def test_audit_scores_research_ready_release_with_clinical_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            release_dir = temp_path / "release"
            output_dir = release_dir / "audit"
            release_dir.mkdir()
            report = temp_path / "release_report.md"
            atlas = release_dir / "atlas.png"
            atlas.write_bytes(b"png")
            report.write_text("# Release report\n")
            command_log = release_dir / "commands.md"
            command_log.write_text(
                "validate-vessel-organ-anatomy\n"
                "validate-vessel-radius-anatomy\n"
                "build-flow-boundary-package\n"
                "build-rt-planning-bundle\n"
                "build-dose-gamma-qa\n"
            )
            limitations = release_dir / "limitations.md"
            limitations.write_text(
                "This is not clinical. Synthetic dose is not TPS or Monte Carlo. "
                "Flow is not 3D CFD and is not patient-specific."
            )

            graph = temp_path / "graph.yaml"
            graph.write_text("case_id: graph\n")
            preview = temp_path / "vascular_network_voxelized" / "preview.png"
            preview.parent.mkdir()
            preview.write_bytes(b"png")
            volume = temp_path / "vascular_network_voxelized" / "lumen.nii.gz"
            volume.write_bytes(b"nii")
            artifact_index = release_dir / "artifacts.csv"
            artifact_index.write_text(
                "group,role,file_type,source_path,exists,size_bytes,sha256,copy_policy,packaged_path,notes\n"
                f"vascular_voxelization,preview,png,{preview},True,3,abc,copied,{preview},\n"
                f"vascular_voxelization,lumen,nifti_volume,{volume},True,3,,indexed_only_large_or_volume,,not_copied\n"
                f"vessel_organ_validation,organ,csv,{graph},True,10,abc,copied,{graph},\n"
                f"vessel_radius_validation,radius,csv,{graph},True,10,abc,copied,{graph},\n"
                f"coupled_pulsatile_flow,flow,yaml,{graph},True,10,abc,copied,{graph},\n"
                f"radiotherapy_qa,rtqa,yaml,{graph},True,10,abc,copied,{graph},\n"
                f"rt_planning,rtplan,yaml,{graph},True,10,abc,copied,{graph},\n"
                f"dose_gamma_qa,gamma,yaml,{graph},True,10,abc,copied,{graph},\n"
                f"reports,report,markdown,{report},True,10,abc,copied,{report},\n"
            )

            qa_summary = release_dir / "qa.csv"
            qa_summary.write_text(
                "category,metric,value,threshold,status,source_path,notes\n"
                "vascular_domain,connected_lumen_components,1,== 1,pass,,\n"
                "vascular_domain,arterial_components,1,== 1,pass,,\n"
                "vascular_domain,venous_components,1,== 1,pass,,\n"
                "vascular_domain,arterial_venous_overlap_after_cleanup_voxels,0,== 0,pass,,\n"
                "vascular_domain,outside_body_fraction_before_clip,0.0,<= 0,pass,,\n"
                "organ_aware_vascular_anatomy,review_count,0,== 0,pass,,\n"
                "organ_aware_vascular_anatomy,fail_count,0,== 0,pass,,\n"
                "radius_aware_vascular_anatomy,review_count,0,== 0,pass,,\n"
                "radius_aware_vascular_anatomy,fail_count,0,== 0,pass,,\n"
                "coupled_pulsatile_flow,max_abs_mass_balance_residual_ml_s,0.0,<= 1e-4,pass,,\n"
                "radiotherapy_planning,ptv_static_d95_gy,19.0,>= 19,pass,,\n"
                "dose_gamma_qa,minimum_gamma_pass_rate_percent,100.0,>= 95,pass,,\n"
            )

            manifest = release_dir / "manifest.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_case",
                        "release_id": "toy_rc1",
                        "outputs": {
                            "manifest_yaml": str(manifest),
                            "artifact_index_csv": str(artifact_index),
                            "qa_summary_csv": str(qa_summary),
                            "command_log": str(command_log),
                            "limitations_markdown": str(limitations),
                            "atlas_png": str(atlas),
                            "report": str(report),
                        },
                    },
                    sort_keys=False,
                )
            )

            result = audit_research_release_package(
                release_manifest_path=manifest,
                output_dir=output_dir,
                audit_id="toy_readiness",
                report_path=temp_path / "audit.md",
            )

            self.assertEqual(result.readiness_tier, "research_ready_clinical_validation_required")
            self.assertGreater(result.research_score_percent, 90.0)
            self.assertGreaterEqual(result.clinical_blocker_count, 4)
            self.assertTrue(Path(result.checks_csv_path).exists())
            self.assertTrue(Path(result.roadmap_csv_path).exists())
            self.assertTrue(Path(result.audit_yaml_path).exists())
            self.assertTrue(Path(result.scorecard_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
