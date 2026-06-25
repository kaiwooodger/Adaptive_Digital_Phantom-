from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.product_case_runner import build_product_case
from phantom_twin.stage007_baseline import Stage007ActiveBaselineResolution


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n")


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _write_completed_build_fixture(root: Path, *, organ_fail_count: int = 0) -> Path:
    case_id = "product_fixture_case"
    build_manifest = root / f"{case_id}_patient_build_manifest_v001.yaml"
    outputs = {
        "torso_spec": root / "torso.yaml",
        "vascular_graph": root / "graph.yaml",
        "voxelized_spec": root / "vascular_network_voxelized" / "voxelized.yaml",
        "flow_boundary_config": root / "flow_boundary.yaml",
        "flow_1d_model": root / "flow_1d.yaml",
        "coupled_flow_model": root / "flow.yaml",
        "rt_package_spec": root / "rt_package.yaml",
        "rt_planning_spec": root / "rt_plan.yaml",
    }
    for key, path in outputs.items():
        if key not in {"voxelized_spec", "coupled_flow_model", "rt_planning_spec"}:
            _touch(path)

    edge_metrics = root / "validation" / "organ" / "product_fixture_vessel_organ_edge_metrics_v001.csv"
    edge_metrics.parent.mkdir(parents=True, exist_ok=True)
    edge_metrics.write_text(
        "edge_id,status,status_note,outside_body_fraction,inside_bone_fraction\n"
        + (
            "aorta,fail,centerline_samples_outside_body,0.25,0.0\n"
            if organ_fail_count
            else "aorta,pass,,0.0,0.0\n"
        )
    )

    organ_spec = root / "validation" / "organ" / "product_fixture_vessel_organ_validation_spec_v001.yaml"
    organ_spec.write_text(
        yaml.safe_dump(
            {
                "summary": {
                    "edge_count": 1,
                    "pass_count": 0 if organ_fail_count else 1,
                    "review_count": 0,
                    "fail_count": organ_fail_count,
                    "outside_body_edge_count": organ_fail_count,
                },
                "outputs": {"edge_metrics_csv": str(edge_metrics)},
            }
        )
    )

    radius_spec = root / "validation" / "radius" / "product_fixture_vessel_radius_validation_spec_v001.yaml"
    radius_spec.parent.mkdir(parents=True, exist_ok=True)
    radius_spec.write_text(
        yaml.safe_dump(
            {
                "summary": {
                    "edge_count": 1,
                    "pass_count": 1,
                    "review_count": 0,
                    "fail_count": 0,
                }
            }
        )
    )

    outputs["voxelized_spec"].parent.mkdir(parents=True, exist_ok=True)
    outputs["voxelized_spec"].write_text(
        yaml.safe_dump(
            {
                "voxelization": {
                    "connected_components": 1,
                    "arterial_venous_overlap_voxels_after_cleanup": 0,
                    "outside_body_fraction_before_clip": 0.0,
                },
                "outputs": {},
            }
        )
    )
    outputs["coupled_flow_model"].write_text(yaml.safe_dump({"summary": {"max_abs_mass_balance_residual_ml_s": 1e-8}}))

    rt_outputs = {
        "static_dose_nifti": root / "dose_static.nii.gz",
        "pulsatile_mean_dose_nifti": root / "dose_mean.nii.gz",
        "pulsatile_peak_dose_nifti": root / "dose_peak.nii.gz",
        "pulsatile_trough_dose_nifti": root / "dose_trough.nii.gz",
        "pulsatile_delta_dose_nifti": root / "dose_delta.nii.gz",
        "dose_metrics_csv": root / "dose_metrics.csv",
        "dose_comparison_csv": root / "dose_compare.csv",
    }
    for path in rt_outputs.values():
        _touch(path)
    outputs["rt_planning_spec"].write_text(yaml.safe_dump({"outputs": {key: str(path) for key, path in rt_outputs.items()}}))

    build_manifest.write_text(
        yaml.safe_dump(
            {
                "case_id": case_id,
                "patient_id": "product_fixture_patient",
                "overall_status": "completed",
                "source_patient_manifest": str(root / "adapter.yaml"),
                "outputs": {key: str(path) for key, path in outputs.items()},
            }
        )
    )
    return build_manifest


class ProductCaseRunnerTests(unittest.TestCase):
    def test_fresh_input_dry_run_defaults_to_active_stage007_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.full((18, 18, 8), 35.0, dtype=np.float32)
            organ = np.ones(ct.shape, dtype=np.int16)
            vessel = np.zeros(ct.shape, dtype=np.uint8)
            vessel[8:10, 8:10, 2:7] = 1
            ct_path = root / "ct.nii.gz"
            organ_path = root / "organ.nii.gz"
            vessel_path = root / "vessel.nii.gz"
            graph_path = root / "stage007_active_graph.yaml"
            spec_path = root / "stage007_active_voxelized.yaml"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(organ_path, organ, affine)
            _write_nifti(vessel_path, vessel, affine)
            graph_path.write_text(yaml.safe_dump({"case_id": "stage007", "nodes": [], "edges": []}))
            spec_path.write_text("case_id: stage007_voxelized\n")
            active = Stage007ActiveBaselineResolution(
                status="ready",
                stage_root=str(root / "stage007"),
                active_manifest_path=None,
                graph_path=str(graph_path),
                voxelized_spec_path=str(spec_path),
                release_manifest_path=None,
                flow_boundary_config_path=None,
                flow_1d_model_path=None,
                coupled_flow_model_path=None,
                release_archive_path=None,
                notes=("test_active_baseline",),
            )

            with patch("phantom_twin.product_case_runner.resolve_stage007_active_baseline", return_value=active):
                result = build_product_case(
                    input_ct_path=ct_path,
                    organ_seg_path=organ_path,
                    vessel_seg_path=vessel_path,
                    output_dir=root / "product",
                    patient_id="auto_stage007_patient",
                    case_id="auto_stage007_product_case",
                    dry_run=True,
                    run_qa=False,
                    render_3d=False,
                )

            self.assertEqual(result.final_status, "build_planned_only")
            self.assertIn("baseline_graph_auto_resolved_from_stage007_active_baseline", result.notes)
            adapter_manifest = yaml.safe_load(Path(result.adapter_manifest_path).read_text())
            self.assertEqual(adapter_manifest["configuration"]["baseline_graph"], str(graph_path))
            self.assertEqual(adapter_manifest["configuration"]["baseline_combined_spec"], str(spec_path))
            build_manifest = yaml.safe_load(Path(result.build_manifest_path).read_text())
            self.assertEqual(build_manifest["overall_status"], "planned_only")

    def test_existing_build_generates_user_facing_product_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_manifest = _write_completed_build_fixture(root / "build")
            render_preview = root / "existing_render.png"
            render_scene = root / "existing_render_scene.yaml"
            render_preview.write_bytes(b"png-placeholder")
            render_scene.write_text("case_id: render\n")

            result = build_product_case(
                existing_build_manifest_path=build_manifest,
                output_dir=root / "product",
                case_id="product_case",
                render_3d=False,
                existing_render_preview_path=render_preview,
                existing_render_scene_spec_path=render_scene,
            )

            self.assertEqual(result.final_status, "research_demo_ready")
            self.assertTrue(Path(result.product_manifest_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(Path(result.qa_yaml_path).exists())
            self.assertEqual(result.render_preview_png_path, str(render_preview))
            self.assertIn("qa_gate", [stage.stage_id for stage in result.stages])
            self.assertIn("linked_existing", [stage.status for stage in result.stages])
            self.assertIn("Phantom Twin Product Case Report", Path(result.report_path).read_text())

    def test_blocked_build_report_lists_top_qa_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_manifest = _write_completed_build_fixture(root / "build", organ_fail_count=1)

            result = build_product_case(
                existing_build_manifest_path=build_manifest,
                output_dir=root / "product",
                case_id="blocked_product_case",
                render_3d=False,
            )

            report_text = Path(result.report_path).read_text()
            self.assertEqual(result.final_status, "research_demo_needs_corrections")
            self.assertTrue(result.qa_blockers)
            self.assertIn("Top QA Blockers", report_text)
            self.assertIn("organ_aware_vessel_fail_edges", report_text)

    def test_fresh_input_dry_run_plans_product_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.full((20, 20, 8), 35.0, dtype=np.float32)
            organ = np.ones(ct.shape, dtype=np.int16)
            vessel = np.zeros(ct.shape, dtype=np.uint8)
            vessel[9:11, 9:11, 2:7] = 1
            ct_path = root / "ct.nii.gz"
            organ_path = root / "organ.nii.gz"
            vessel_path = root / "vessel.nii.gz"
            graph_path = root / "baseline_graph.yaml"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(organ_path, organ, affine)
            _write_nifti(vessel_path, vessel, affine)
            graph_path.write_text(yaml.safe_dump({"case_id": "toy", "nodes": [], "edges": []}))

            result = build_product_case(
                input_ct_path=ct_path,
                organ_seg_path=organ_path,
                vessel_seg_path=vessel_path,
                baseline_graph_path=graph_path,
                output_dir=root / "product",
                patient_id="fresh_patient",
                case_id="fresh_product_case",
                dry_run=True,
                run_qa=False,
                render_3d=False,
            )

            self.assertEqual(result.final_status, "build_planned_only")
            self.assertTrue(Path(result.adapter_manifest_path).exists())
            self.assertTrue(Path(result.build_manifest_path).exists())
            self.assertIn("phantom_build", [stage.stage_id for stage in result.stages])


if __name__ == "__main__":
    unittest.main()
