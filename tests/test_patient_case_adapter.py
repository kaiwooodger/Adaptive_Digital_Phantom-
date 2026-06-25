from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.patient_case_adapter import run_patient_case_adapter
from phantom_twin.stage007_baseline import Stage007ActiveBaselineResolution


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _write_profile_spec(root: Path, profile_id: str, *, bmi: float, waist_cm: float, body_volume_cm3: float) -> Path:
    spec_dir = root / "profiles" / profile_id / "anthropometry"
    spec_dir.mkdir(parents=True, exist_ok=True)
    path = spec_dir / f"{profile_id}_anthro_morph_spec_v001.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "case_id": f"{profile_id}_variant",
                "anthropometry": {
                    "target_bmi": bmi,
                    "target_height_cm": 175.0,
                    "target_waist_cm": waist_cm,
                    "achieved_waist_cm": waist_cm,
                },
                "quality_summary": {
                    "morphed_body_volume_cm3": body_volume_cm3,
                    "morphed_bbox_mm": [240.0, 180.0, 420.0],
                },
                "region_stats": [
                    {"name": "liver", "morphed_volume_cm3": 0.80},
                    {"name": "kidneys", "morphed_volume_cm3": 0.20},
                    {"name": "blood_equivalent_fluid", "morphed_volume_cm3": 0.05},
                ],
            },
            sort_keys=False,
        )
    )
    return path


class PatientCaseAdapterTests(unittest.TestCase):
    def test_patient_case_adapter_score_only_defaults_to_active_stage007_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.full((20, 20, 8), -1000.0, dtype=np.float32)
            ct[4:16, 4:16, 2:7] = 40.0
            ct_path = root / "ct.nii.gz"
            graph_path = root / "stage007_graph.yaml"
            spec_path = root / "stage007_voxelized.yaml"
            _write_nifti(ct_path, ct, affine)
            graph_path.write_text(yaml.safe_dump({"case_id": "stage007", "nodes": [], "edges": []}))
            spec_path.write_text("case_id: stage007_voxelized\n")
            _write_profile_spec(root, "profile", bmi=30.0, waist_cm=105.0, body_volume_cm3=30_000.0)
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

            with patch("phantom_twin.patient_case_adapter.resolve_stage007_active_baseline", return_value=active):
                result = run_patient_case_adapter(
                    input_ct_path=ct_path,
                    patient_id="score_only_patient",
                    case_id="score_only_case",
                    output_dir=root / "adapter",
                    profile_spec_glob=str(root / "profiles" / "**" / "*_anthro_morph_spec_v001.yaml"),
                    include_metric_scaled_height_grid=False,
                    score_only=True,
                )

            self.assertEqual(result.baseline_graph_path, str(graph_path))
            self.assertEqual(result.baseline_combined_spec_path, str(spec_path))
            self.assertIn("baseline_graph_auto_resolved_from_stage007_active_baseline", result.notes)
            manifest = yaml.safe_load(Path(result.manifest_yaml_path).read_text())
            self.assertEqual(manifest["configuration"]["baseline_graph"], str(graph_path))
            self.assertEqual(manifest["configuration"]["baseline_combined_spec"], str(spec_path))

    def test_patient_case_adapter_scores_library_and_runs_product_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.full((24, 22, 10), -1000.0, dtype=np.float32)
            ct[4:20, 4:18, 2:8] = 40.0
            organs = np.zeros(ct.shape, dtype=np.int16)
            organs[4:20, 4:18, 2:8] = 4
            organs[8:14, 8:14, 4:7] = 6
            organs[14:18, 8:12, 4:7] = 7
            vessels = np.zeros(ct.shape, dtype=np.uint8)
            vessels[11:13, 10:12, 2:8] = 1
            ct_path = root / "ct.nii.gz"
            organ_path = root / "organs.nii.gz"
            vessel_path = root / "vessels.nii.gz"
            graph_path = root / "baseline_graph.yaml"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(organ_path, organs, affine)
            _write_nifti(vessel_path, vessels, affine)
            graph_path.write_text(yaml.safe_dump({"case_id": "toy_graph", "nodes": [], "edges": []}))

            _write_profile_spec(root, "small_profile", bmi=22.0, waist_cm=85.0, body_volume_cm3=25_000.0)
            _write_profile_spec(root, "large_profile", bmi=32.0, waist_cm=110.0, body_volume_cm3=36_000.0)

            result = run_patient_case_adapter(
                input_ct_path=ct_path,
                organ_seg_path=organ_path,
                vessel_seg_path=vessel_path,
                patient_id="demo_patient",
                case_id="demo_patient_case",
                output_dir=root / "adapter",
                profile_spec_glob=str(root / "profiles" / "**" / "*_anthro_morph_spec_v001.yaml"),
                include_metric_scaled_height_grid=False,
                baseline_graph_path=graph_path,
                target_height_cm=175.0,
                target_bmi=32.0,
                target_waist_cm=110.0,
                dry_run=True,
                run_qa=False,
                render_3d=False,
            )

            self.assertEqual(result.selected_profile_id, "large_profile")
            self.assertEqual(result.final_status, "patient_case_profile_matched_build_planned")
            self.assertTrue(Path(result.patient_metric_yaml_path).exists())
            self.assertTrue(Path(result.match_scores_csv_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(result.product_manifest_path)
            self.assertTrue(Path(result.product_manifest_path).exists())
            self.assertIn("phantom_build:planned_only", result.stage_statuses)

            manifest = yaml.safe_load(Path(result.manifest_yaml_path).read_text())
            self.assertEqual(manifest["selected_profile"]["profile_id"], "large_profile")
            self.assertEqual(manifest["final_status"], result.final_status)


if __name__ == "__main__":
    unittest.main()
