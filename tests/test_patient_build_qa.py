from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.patient_build_qa import qa_patient_phantom_build


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n")


def _write_build_fixture(root: Path, *, organ_fail_count: int) -> Path:
    case_id = "qa_case"
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

    edge_metrics = root / "validation" / "organ" / "qa_case_vessel_organ_edge_metrics_v001.csv"
    edge_metrics.parent.mkdir(parents=True, exist_ok=True)
    edge_metrics.write_text(
        "edge_id,status,status_note,outside_body_fraction,inside_bone_fraction\n"
        + ("aorta,fail,centerline_samples_outside_body,0.25,0.0\n" if organ_fail_count else "aorta,pass,,0.0,0.0\n")
    )

    organ_spec = root / "validation" / "organ" / "qa_case_vessel_organ_validation_spec_v001.yaml"
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
    radius_spec = root / "validation" / "radius" / "qa_case_vessel_radius_validation_spec_v001.yaml"
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
                }
            }
        )
    )
    outputs["coupled_flow_model"].write_text(yaml.safe_dump({"summary": {"max_abs_mass_balance_residual_ml_s": 1e-8}}))

    rt_dose_paths = {
        "static_dose_nifti": root / "dose_static.nii.gz",
        "pulsatile_mean_dose_nifti": root / "dose_mean.nii.gz",
        "pulsatile_peak_dose_nifti": root / "dose_peak.nii.gz",
        "pulsatile_trough_dose_nifti": root / "dose_trough.nii.gz",
        "pulsatile_delta_dose_nifti": root / "dose_delta.nii.gz",
        "dose_metrics_csv": root / "dose_metrics.csv",
        "dose_comparison_csv": root / "dose_compare.csv",
    }
    for path in rt_dose_paths.values():
        _touch(path)
    outputs["rt_planning_spec"].write_text(yaml.safe_dump({"outputs": {key: str(path) for key, path in rt_dose_paths.items()}}))

    build_manifest.write_text(
        yaml.safe_dump(
            {
                "case_id": case_id,
                "patient_id": "qa_patient",
                "overall_status": "completed",
                "outputs": {key: str(path) for key, path in outputs.items()},
            }
        )
    )
    return build_manifest


class PatientBuildQAGateTests(unittest.TestCase):
    def test_qa_gate_approves_clean_research_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = _write_build_fixture(root / "build", organ_fail_count=0)

            result = qa_patient_phantom_build(
                build_manifest_path=manifest,
                output_dir=root / "qa",
                report_path=root / "qa.md",
            )

            self.assertEqual(result.readiness_status, "approved_research_use")
            self.assertEqual(result.fail_count, 0)
            self.assertTrue(Path(result.qa_yaml_path).exists())
            self.assertTrue(Path(result.checks_csv_path).exists())

    def test_qa_gate_blocks_organ_aware_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = _write_build_fixture(root / "build", organ_fail_count=1)

            result = qa_patient_phantom_build(
                build_manifest_path=manifest,
                output_dir=root / "qa",
                report_path=root / "qa.md",
            )

            self.assertEqual(result.readiness_status, "blocked_anatomy_or_pipeline_qa_failed")
            self.assertGreater(result.fail_count, 0)
            self.assertIn("Correct vessel-organ relationships", result.recommended_actions[0])


if __name__ == "__main__":
    unittest.main()
