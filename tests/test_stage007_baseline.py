from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.stage007_baseline import (
    promote_stage007_clean_baseline,
    resolve_stage007_active_baseline,
    run_stage007_acceptance_smoke,
)


def _write_stage007_toy_release(temp_path: Path) -> dict[str, Path]:
    release_dir = temp_path / "release"
    release_dir.mkdir()
    graph = temp_path / "clean_radius_tuned_vascular_graph_v001.yaml"
    graph.write_text("case_id: toy\nedges: []\n")
    voxel_spec = temp_path / "toy_vascular_network_voxelized_spec_v001.yaml"
    voxel_spec.write_text(
        yaml.safe_dump(
            {
                "case_id": "toy",
                "voxelization": {
                    "connected_components": 1,
                    "arterial_components": 1,
                    "venous_components": 1,
                    "arterial_venous_overlap_voxels_after_cleanup": 0,
                    "outside_body_fraction_before_clip": 0.0,
                },
            },
            sort_keys=False,
        )
    )
    boundary = temp_path / "toy_flow_boundary_conditions_v001.yaml"
    boundary.write_text(
        yaml.safe_dump(
            {
                "case_id": "toy",
                "boundaries": [
                    {"boundary_id": 1, "status": "mapped"},
                    {"boundary_id": 2, "status": "mapped"},
                ],
            },
            sort_keys=False,
        )
    )
    flow_1d = temp_path / "toy_flow_1d_model_v001.yaml"
    flow_1d.write_text(
        yaml.safe_dump({"summary": {"max_abs_mass_balance_residual_ml_s": 0.0}}, sort_keys=False)
    )
    coupled = temp_path / "toy_coupled_pulsatile_flow_model_v001.yaml"
    coupled.write_text(
        yaml.safe_dump(
            {
                "summary": {
                    "max_abs_mass_balance_residual_ml_s": 0.0,
                    "max_outlet_split_range_percentage_points": 8.5,
                }
            },
            sort_keys=False,
        )
    )
    atlas = release_dir / "atlas.png"
    atlas.write_bytes(b"png")
    report = release_dir / "report.md"
    report.write_text("# report\n")
    archive = temp_path / "toy_release.tar.gz"
    archive.write_bytes(b"tgz")

    artifact_index = release_dir / "artifact_index.csv"
    artifact_index.write_text(
        "group,role,file_type,source_path,exists,size_bytes,sha256,copy_policy,packaged_path,notes\n"
        f"source_dependency,clean_radius_tuned_vascular_graph_v001,yaml,{graph},True,10,abc,copied,{graph},\n"
        f"vascular_voxelization,toy_vascular_network_voxelized_spec_v001,yaml,{voxel_spec},True,10,abc,copied,{voxel_spec},\n"
        f"flow_boundary_conditions,toy_flow_boundary_conditions_v001,yaml,{boundary},True,10,abc,copied,{boundary},\n"
        f"steady_1d_flow,toy_flow_1d_model_v001,yaml,{flow_1d},True,10,abc,copied,{flow_1d},\n"
        f"coupled_pulsatile_flow,toy_coupled_pulsatile_flow_model_v001,yaml,{coupled},True,10,abc,copied,{coupled},\n"
    )
    qa = release_dir / "qa.csv"
    qa.write_text(
        "category,metric,value,threshold,status,source_path,notes\n"
        f"vascular_domain,connected_lumen_components,1,== 1,pass,{voxel_spec},\n"
        f"vascular_domain,arterial_components,1,== 1,pass,{voxel_spec},\n"
        f"vascular_domain,venous_components,1,== 1,pass,{voxel_spec},\n"
        f"vascular_domain,arterial_venous_overlap_after_cleanup_voxels,0,== 0,pass,{voxel_spec},\n"
        f"vascular_domain,outside_body_fraction_before_clip,0.0,<= 0,pass,{voxel_spec},\n"
        f"organ_aware_vascular_anatomy,review_count,0,== 0,pass,{voxel_spec},\n"
        f"organ_aware_vascular_anatomy,fail_count,0,== 0,pass,{voxel_spec},\n"
        f"organ_aware_vascular_anatomy,bone_intersection_edge_count,0,== 0,pass,{voxel_spec},\n"
        f"radius_aware_vascular_anatomy,review_count,0,== 0,pass,{voxel_spec},\n"
        f"radius_aware_vascular_anatomy,fail_count,0,== 0,pass,{voxel_spec},\n"
        f"radius_aware_vascular_anatomy,radius_tuning_candidate_count,0,== 0,pass,{voxel_spec},\n"
        f"radius_aware_vascular_anatomy,reroute_candidate_count,0,== 0,pass,{voxel_spec},\n"
        f"steady_1d_flow,max_abs_mass_balance_residual_ml_s,0.0,<= 1e-4,pass,{flow_1d},\n"
        f"coupled_pulsatile_flow,max_abs_mass_balance_residual_ml_s,0.0,<= 1e-4,pass,{coupled},\n"
    )
    manifest = release_dir / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "case_id": "toy_stage007",
                "release_id": "toy_stage007_rc1",
                "readiness_status": "research_release_candidate",
                "summary": {"qa_status_counts": {"pass": 14, "review": 0, "fail": 0}},
                "outputs": {
                    "manifest_yaml": str(manifest),
                    "artifact_index_csv": str(artifact_index),
                    "qa_summary_csv": str(qa),
                    "atlas_png": str(atlas),
                    "report": str(report),
                },
                "key_artifacts": {
                    "toy_vascular_network_voxelized_spec_v001": str(voxel_spec),
                    "toy_coupled_pulsatile_flow_model_v001": str(coupled),
                },
            },
            sort_keys=False,
        )
    )
    return {
        "release_manifest": manifest,
        "stage_root": temp_path / "stage007",
        "graph": graph,
        "voxel_spec": voxel_spec,
        "coupled": coupled,
        "archive": archive,
    }


class Stage007BaselineTests(unittest.TestCase):
    def test_resolve_stage007_active_baseline_reads_pointer_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_root = root / "stage007"
            stage_root.mkdir()
            graph = root / "active_graph.yaml"
            voxel = root / "active_voxelized_spec.yaml"
            release = root / "release_manifest.yaml"
            flow = root / "coupled_flow.yaml"
            graph.write_text("case_id: graph\n")
            voxel.write_text("case_id: voxel\n")
            release.write_text("case_id: release\n")
            flow.write_text("case_id: flow\n")
            (stage_root / "latest_stage007_active_graph_path.txt").write_text(f"{graph}\n")
            (stage_root / "latest_stage007_active_voxelized_spec_path.txt").write_text(f"{voxel}\n")
            (stage_root / "latest_stage007_active_release_manifest_path.txt").write_text(f"{release}\n")
            (stage_root / "latest_stage007_active_coupled_flow_model_path.txt").write_text(f"{flow}\n")

            result = resolve_stage007_active_baseline(stage_root=stage_root)

            self.assertEqual(result.status, "ready")
            self.assertEqual(result.graph_path, str(graph))
            self.assertEqual(result.voxelized_spec_path, str(voxel))
            self.assertEqual(result.release_manifest_path, str(release))
            self.assertEqual(result.coupled_flow_model_path, str(flow))

    def test_promote_stage007_clean_baseline_writes_active_and_accepted_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _write_stage007_toy_release(Path(temp_dir))

            result = promote_stage007_clean_baseline(
                release_manifest_path=paths["release_manifest"],
                stage_root=paths["stage_root"],
                release_archive_path=paths["archive"],
                report_path=Path(temp_dir) / "promotion.md",
            )

            self.assertEqual(result.status, "active_research_release_candidate")
            self.assertTrue(Path(result.active_manifest_path).exists())
            self.assertTrue(Path(result.accepted_manifest_path).exists())
            self.assertEqual(
                (paths["stage_root"] / "latest_stage007_active_graph_path.txt").read_text().strip(),
                str(paths["graph"]),
            )
            self.assertEqual(
                (paths["stage_root"] / "latest_stage007_accepted_graph_path.txt").read_text().strip(),
                str(paths["graph"]),
            )
            manifest = yaml.safe_load(Path(result.active_manifest_path).read_text())
            self.assertEqual(manifest["active_artifacts"]["graph"], str(paths["graph"]))

    def test_run_stage007_acceptance_smoke_passes_clean_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _write_stage007_toy_release(Path(temp_dir))

            result = run_stage007_acceptance_smoke(
                release_manifest_path=paths["release_manifest"],
                release_archive_path=paths["archive"],
                output_dir=Path(temp_dir) / "acceptance",
                report_path=Path(temp_dir) / "acceptance.md",
                min_boundary_count=2,
            )

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.fail_count, 0)
            self.assertEqual(result.review_count, 0)
            self.assertGreater(result.pass_count, 8)
            self.assertAlmostEqual(result.flow_split_range_percentage_points, 8.5)
            self.assertTrue(Path(result.checks_csv_path).exists())
            self.assertTrue(Path(result.smoke_yaml_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
