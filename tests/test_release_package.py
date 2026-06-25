from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.release_package import build_research_release_package


class ResearchReleasePackageTests(unittest.TestCase):
    def test_build_research_release_package_indexes_outputs_and_writes_qa(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            case_id = "toy_stage007"
            stage_root = temp_path / "stage"
            reports_dir = temp_path / "reports"
            output_dir = temp_path / "release"
            reports_dir.mkdir()

            graph_path = temp_path / "source_graph.yaml"
            graph_path.write_text("case_id: toy_graph\n")
            volume_path = stage_root / "vascular_network_voxelized" / f"{case_id}_lumen_mask_v001.nii.gz"
            volume_path.parent.mkdir(parents=True)
            volume_path.write_bytes(b"not-a-real-nifti-but-indexed-as-volume")
            voxel_spec = volume_path.parent / f"{case_id}_vascular_network_voxelized_spec_v001.yaml"
            voxel_spec.write_text(
                yaml.safe_dump(
                    {
                        "case_id": case_id,
                        "voxelization": {
                            "source_graph": str(graph_path),
                            "source_combined_labels": str(volume_path),
                            "connected_components": 1,
                            "arterial_components": 1,
                            "venous_components": 1,
                            "arterial_venous_overlap_voxels_after_cleanup": 0,
                            "outside_body_fraction_before_clip": 0.0,
                        },
                        "outputs": {
                            "combined_lumen_mask": str(volume_path),
                            "preview_png": str(stage_root / "vascular_network_voxelized" / f"{case_id}_preview_v001.png"),
                        },
                    },
                    sort_keys=False,
                )
            )
            (stage_root / "vascular_network_voxelized" / f"{case_id}_vascular_domain_connectivity_summary_v001.csv").write_text(
                "domain,components_before,components_after,voxels_before,voxels_after,seeded_component_count_before,pruned_component_count,pruned_voxel_count,connector_voxel_count\n"
                "arterial,2,1,12,10,1,1,2,0\n"
                "venous,1,1,8,8,1,0,0,0\n"
            )

            organ_dir = stage_root / "validation" / "vessel_organ_anatomy"
            organ_dir.mkdir(parents=True)
            (organ_dir / f"{case_id}_vessel_organ_validation_spec_v001.yaml").write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "edge_count": 3,
                            "pass_count": 3,
                            "review_count": 0,
                            "fail_count": 0,
                            "bone_intersection_edge_count": 0,
                        }
                    },
                    sort_keys=False,
                )
            )

            radius_dir = stage_root / "validation" / "vessel_radius_anatomy"
            radius_dir.mkdir(parents=True)
            (radius_dir / f"{case_id}_vessel_radius_validation_spec_v001.yaml").write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "edge_count": 3,
                            "pass_count": 3,
                            "review_count": 0,
                            "fail_count": 0,
                            "radius_tuning_candidate_count": 0,
                            "reroute_candidate_count": 0,
                        }
                    },
                    sort_keys=False,
                )
            )

            flow_1d_dir = stage_root / "flow_1d"
            flow_1d_dir.mkdir()
            (flow_1d_dir / f"{case_id}_flow_1d_model_v001.yaml").write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "max_abs_mass_balance_residual_ml_s": 0.0,
                            "arterial_total_flow_ml_s": 80.0,
                        }
                    },
                    sort_keys=False,
                )
            )

            flow_dir = stage_root / "flow_coupled_pulsatile"
            flow_dir.mkdir()
            (flow_dir / f"{case_id}_coupled_pulsatile_flow_model_v001.yaml").write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "max_abs_mass_balance_residual_ml_s": 0.0,
                            "arterial_inlet_flow_mean_ml_s": 80.0,
                            "arterial_inlet_flow_min_ml_s": 56.0,
                            "arterial_inlet_flow_max_ml_s": 158.0,
                        }
                    },
                    sort_keys=False,
                )
            )

            rt_qa_dir = stage_root / "radiotherapy_qa_package"
            rt_qa_dir.mkdir()
            (rt_qa_dir / f"{case_id}_radiotherapy_qa_package_spec_v001.yaml").write_text(
                yaml.safe_dump(
                    {"synthetic_target": {"gtv_volume_cm3": 3.0, "ptv_volume_cm3": 21.0}},
                    sort_keys=False,
                )
            )

            rt_plan_dir = stage_root / "rt_planning_bundle"
            rt_plan_dir.mkdir()
            dose_metrics_csv = rt_plan_dir / f"{case_id}_rt_dose_metrics_v001.csv"
            dose_metrics_csv.write_text(
                "mask_id,state,d95_gy\n"
                "target_ptv_synthetic_vertebral,static,19.0\n"
            )
            dose_comparison_csv = rt_plan_dir / f"{case_id}_rt_static_vs_pulsatile_dose_comparison_v001.csv"
            dose_comparison_csv.write_text(
                "mask_id,comparison_state,delta_d95_gy\n"
                "target_ptv_synthetic_vertebral,pulsatile_peak,0.01\n"
            )
            (rt_plan_dir / f"{case_id}_rt_planning_bundle_spec_v001.yaml").write_text(
                yaml.safe_dump(
                    {"outputs": {"dose_metrics_csv": str(dose_metrics_csv), "dose_comparison_csv": str(dose_comparison_csv)}},
                    sort_keys=False,
                )
            )

            gamma_dir = stage_root / "dose_gamma_qa"
            gamma_dir.mkdir()
            (gamma_dir / f"{case_id}_dose_gamma_qa_spec_v001.yaml").write_text(
                yaml.safe_dump(
                    {
                        "state_results": [
                            {"state": "pulsatile_mean", "pass_rate_percent": 100.0, "p95_gamma": 0.01},
                            {"state": "pulsatile_peak", "pass_rate_percent": 99.0, "p95_gamma": 0.2},
                        ]
                    },
                    sort_keys=False,
                )
            )

            (reports_dir / f"{case_id}.md").write_text("# Toy report\n")

            result = build_research_release_package(
                stage_root=stage_root,
                reports_dir=reports_dir,
                output_dir=output_dir,
                case_id=case_id,
                release_id="toy_rc1",
                report_path=reports_dir / "toy_release.md",
            )

            self.assertEqual(result.readiness_status, "research_release_candidate")
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.artifact_index_csv_path).exists())
            self.assertTrue(Path(result.qa_summary_csv_path).exists())
            self.assertTrue(Path(result.command_log_path).exists())
            self.assertTrue(Path(result.limitations_markdown_path).exists())
            self.assertTrue(Path(result.atlas_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertGreater(result.summary["artifact_count"], 0)
            self.assertGreater(result.summary["copied_artifact_count"], 0)
            self.assertEqual(result.summary["qa_status_counts"]["fail"], 0)
            manifest = yaml.safe_load(Path(result.manifest_yaml_path).read_text())
            self.assertEqual(manifest["release_id"], "toy_rc1")
            self.assertTrue(manifest["large_or_volume_artifacts"])

    def test_release_package_marks_nonzero_vessel_reviews_as_review_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            case_id = "toy_review"
            stage_root = temp_path / "stage"
            reports_dir = temp_path / "reports"
            output_dir = temp_path / "release"
            reports_dir.mkdir()

            radius_dir = stage_root / "validation" / "vessel_radius_anatomy"
            radius_dir.mkdir(parents=True)
            (radius_dir / f"{case_id}_vessel_radius_validation_spec_v001.yaml").write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "edge_count": 3,
                            "pass_count": 2,
                            "review_count": 1,
                            "fail_count": 0,
                            "radius_tuning_candidate_count": 1,
                            "reroute_candidate_count": 1,
                        }
                    },
                    sort_keys=False,
                )
            )

            result = build_research_release_package(
                stage_root=stage_root,
                reports_dir=reports_dir,
                output_dir=output_dir,
                case_id=case_id,
                release_id="toy_review_rc1",
                report_path=reports_dir / "toy_review_release.md",
            )

            self.assertEqual(result.readiness_status, "research_release_candidate_review_required")
            self.assertEqual(result.summary["qa_status_counts"]["fail"], 0)
            self.assertGreater(result.summary["qa_status_counts"]["review"], 0)


if __name__ == "__main__":
    unittest.main()
