from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.patient_adapter import build_patient_phantom_adapter
from phantom_twin.stage007_baseline import Stage007ActiveBaselineResolution


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, affine), str(path))


class PatientPhantomAdapterTests(unittest.TestCase):
    def test_patient_adapter_defaults_to_active_stage007_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.ones((12, 12, 6), dtype=np.float32)
            ct_path = temp_path / "ct.nii.gz"
            graph_path = temp_path / "stage007_graph.yaml"
            spec_path = temp_path / "stage007_voxelized.yaml"
            _write_nifti(ct_path, ct, affine)
            graph_path.write_text("case_id: stage007\n")
            spec_path.write_text("case_id: stage007_voxelized\n")
            active = Stage007ActiveBaselineResolution(
                status="ready",
                stage_root=str(temp_path / "stage007"),
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

            with patch("phantom_twin.patient_adapter.resolve_stage007_active_baseline", return_value=active):
                result = build_patient_phantom_adapter(
                    input_ct_path=ct_path,
                    patient_id="auto_baseline_patient",
                    output_dir=temp_path / "adapter",
                    report_path=temp_path / "adapter.md",
                )

            manifest = yaml.safe_load(Path(result.manifest_yaml_path).read_text())
            self.assertEqual(manifest["configuration"]["baseline_graph"], str(graph_path))
            self.assertEqual(manifest["configuration"]["baseline_combined_spec"], str(spec_path))
            self.assertIn("baseline_graph_auto_resolved_from_stage007_active_baseline", result.notes)
            self.assertIn("baseline_reference_spec_auto_resolved_from_stage007_active_voxelized_spec", result.notes)

    def test_patient_adapter_accepts_coregistered_ct_organ_and_vessel_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.full((24, 22, 12), -900.0, dtype=np.float32)
            ct[5:19, 5:17, 2:10] = 40.0
            organ = np.zeros(ct.shape, dtype=np.int16)
            organ[5:19, 5:17, 2:10] = 4
            organ[8:13, 8:13, 5:9] = 6
            vessel = np.zeros(ct.shape, dtype=np.int16)
            vessel[11:13, 10:12, 3:10] = 1

            ct_path = temp_path / "patient_ct.nii.gz"
            organ_path = temp_path / "patient_organs.nii.gz"
            vessel_path = temp_path / "patient_vessels.nii.gz"
            graph_path = temp_path / "baseline_graph.yaml"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(organ_path, organ, affine)
            _write_nifti(vessel_path, vessel, affine)
            graph_path.write_text(yaml.safe_dump({"case_id": "baseline", "nodes": [], "edges": []}))

            result = build_patient_phantom_adapter(
                input_ct_path=ct_path,
                organ_seg_path=organ_path,
                vessel_seg_path=vessel_path,
                baseline_graph_path=graph_path,
                patient_id="demo_patient",
                case_id="demo_patient_case",
                output_dir=temp_path / "adapter",
                report_path=temp_path / "adapter.md",
            )

            self.assertEqual(result.primary_ct_status, "ct_nifti_ready")
            self.assertEqual(result.anatomy_adaptation_status, "ready_for_ct_registered_anatomy_build")
            self.assertEqual(result.vascular_adaptation_status, "ready_for_patient_vessel_replacement")
            self.assertEqual(result.overall_status, "ready_for_patient_specific_anatomy_and_vascular_build")
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.input_qa_csv_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertIn("build-digital-torso", "\n".join(result.recommended_commands))
            self.assertIn("build-cta-derived-vascular-graph", "\n".join(result.recommended_commands))

            manifest = yaml.safe_load(Path(result.manifest_yaml_path).read_text())
            self.assertEqual(manifest["status"]["overall"], result.overall_status)
            self.assertEqual(len(manifest["inputs"]), 3)

    def test_patient_adapter_flags_mismatched_vessel_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ct_affine = np.diag([2.0, 2.0, 4.0, 1.0])
            vessel_affine = np.diag([1.0, 1.0, 2.0, 1.0])
            ct = np.ones((20, 20, 8), dtype=np.float32)
            vessel = np.zeros((16, 16, 6), dtype=np.int16)
            vessel[7:9, 7:9, 1:5] = 1
            ct_path = temp_path / "ct.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            _write_nifti(ct_path, ct, ct_affine)
            _write_nifti(vessel_path, vessel, vessel_affine)

            result = build_patient_phantom_adapter(
                input_ct_path=ct_path,
                vessel_seg_path=vessel_path,
                patient_id="mismatch_patient",
                output_dir=temp_path / "adapter",
                report_path=temp_path / "adapter.md",
            )

            self.assertEqual(result.vascular_adaptation_status, "vessel_segmentation_registration_required")
            vessel_summary = next(item for item in result.inputs if item.role == "vessel_seg")
            self.assertEqual(vessel_summary.geometry_status, "registration_required_to_ct_grid")
            self.assertIn("one_or_more_inputs_require_registration_to_primary_ct_grid", result.notes)


if __name__ == "__main__":
    unittest.main()
