from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.patient_adapter import build_patient_phantom_adapter
from phantom_twin.patient_build import _build_btcv_label_torso, _build_material_label_torso, _organ_label_mode, run_patient_phantom_build
from phantom_twin.stage007_baseline import Stage007ActiveBaselineResolution


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, affine), str(path))


class PatientPhantomBuildExecutorTests(unittest.TestCase):
    def test_patient_build_defaults_to_active_stage007_graph_for_legacy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.ones((16, 16, 8), dtype=np.float32)
            organ = np.ones(ct.shape, dtype=np.int16)
            vessel = np.zeros(ct.shape, dtype=np.uint8)
            vessel[7:9, 7:9, 2:7] = 1
            ct_path = temp_path / "ct.nii.gz"
            organ_path = temp_path / "organ.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            graph_path = temp_path / "stage007_graph.yaml"
            spec_path = temp_path / "stage007_voxelized.yaml"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(organ_path, organ, affine)
            _write_nifti(vessel_path, vessel, affine)
            graph_path.write_text(yaml.safe_dump({"case_id": "stage007", "nodes": [], "edges": []}))
            spec_path.write_text("case_id: stage007_voxelized\n")
            manifest_path = temp_path / "legacy_patient_manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "legacy_case",
                        "patient_id": "legacy_patient",
                        "status": {
                            "primary_ct": "ct_nifti_ready",
                            "anatomy_adaptation": "ready_for_ct_registered_anatomy_build",
                            "vascular_adaptation": "ready_for_patient_vessel_replacement",
                        },
                        "configuration": {
                            "organ_labelmap": "configs/labelmaps/ct_org.yaml",
                            "materials": "configs/materials.yaml",
                            "baseline_graph": None,
                        },
                        "inputs": [
                            {"role": "ct", "exists": True, "source_path": str(ct_path), "geometry_status": "reference_ct_grid"},
                            {
                                "role": "organ_seg",
                                "exists": True,
                                "source_path": str(organ_path),
                                "geometry_status": "co_registered_to_ct_grid",
                                "unique_labels_sample": [1],
                                "max_value": 1,
                            },
                            {
                                "role": "vessel_seg",
                                "exists": True,
                                "source_path": str(vessel_path),
                                "geometry_status": "co_registered_to_ct_grid",
                            },
                        ],
                    },
                    sort_keys=False,
                )
            )
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

            with patch("phantom_twin.patient_build.resolve_stage007_active_baseline", return_value=active):
                result = run_patient_phantom_build(
                    patient_manifest_path=manifest_path,
                    output_dir=temp_path / "builds",
                    report_path=temp_path / "build.md",
                    dry_run=True,
                )

            self.assertEqual(result.overall_status, "planned_only")
            self.assertIn("baseline_graph_auto_resolved_from_stage007_active_baseline", result.notes)
            self.assertIn("vascular_graph", [step.step_id for step in result.steps])

    def test_material_label_torso_recasts_existing_vascular_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.full((12, 12, 6), 40.0, dtype=np.float32)
            labels = np.zeros(ct.shape, dtype=np.int16)
            labels[2:10, 2:10, 1:5] = 4
            labels[4:8, 4:8, 2:4] = 14
            labels[5:7, 5:7, 2:4] = 13
            ct_path = temp_path / "ct.nii.gz"
            labels_path = temp_path / "labels.nii.gz"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(labels_path, labels, affine)

            manifest = {
                "inputs": [
                    {
                        "role": "organ_seg",
                        "unique_labels_sample": [0, 4, 13, 14],
                        "max_value": 14,
                    }
                ]
            }
            self.assertEqual(_organ_label_mode(manifest, "auto"), "material")

            result = _build_material_label_torso(
                ct_path=ct_path,
                material_labels_path=labels_path,
                materials_path=Path("configs/materials.yaml"),
                output_dir=temp_path / "torso",
                case_id="toy_material",
                report_path=temp_path / "torso.md",
            )

            output_labels = np.asanyarray(nib.load(result.material_label_path).dataobj).astype(np.int16)
            self.assertNotIn(13, set(np.unique(output_labels).tolist()))
            self.assertNotIn(14, set(np.unique(output_labels).tolist()))
            self.assertIn(4, set(np.unique(output_labels).tolist()))
            self.assertTrue(Path(result.spec_yaml_path).exists())

    def test_btcv_label_torso_converts_liver_and_kidneys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.full((12, 12, 6), 35.0, dtype=np.float32)
            labels = np.zeros(ct.shape, dtype=np.int16)
            labels[2:7, 2:7, 1:4] = 6
            labels[7:10, 2:5, 2:5] = 2
            labels[2:5, 7:10, 2:5] = 3
            labels[1:4, 1:4, 1:3] = 1
            labels[8:11, 8:11, 1:4] = 7
            labels[9:10, 9:10, 2:3] = 7
            ct[9:10, 9:10, 2:3] = -850.0
            labels[1:3, 8:10, 1:3] = 4
            labels[4:5, 10:11, 2:4] = 5
            labels[5:8, 8:10, 2:5] = 11
            labels[9:10, 5:6, 3:4] = 12
            labels[10:11, 5:6, 3:4] = 13
            labels[6:8, 6:8, 2:5] = 8
            ct_path = temp_path / "ct.nii.gz"
            labels_path = temp_path / "btcv_labels.nii.gz"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(labels_path, labels, affine)

            manifest = {
                "configuration": {"organ_labelmap": "configs/labelmaps/btcv_abdomen.yaml"},
                "inputs": [
                    {
                        "role": "organ_seg",
                        "unique_labels_sample": [0, 2, 3, 6, 8],
                        "max_value": 8,
                    }
                ],
            }
            self.assertEqual(_organ_label_mode(manifest, "auto"), "btcv")

            result = _build_btcv_label_torso(
                ct_path=ct_path,
                btcv_labels_path=labels_path,
                materials_path=Path("configs/materials.yaml"),
                output_dir=temp_path / "torso",
                case_id="toy_btcv",
                report_path=temp_path / "torso.md",
            )

            output_labels = np.asanyarray(nib.load(result.material_label_path).dataobj).astype(np.int16)
            self.assertIn(6, set(np.unique(output_labels).tolist()))
            self.assertIn(7, set(np.unique(output_labels).tolist()))
            self.assertIn(16, set(np.unique(output_labels).tolist()))
            self.assertIn(17, set(np.unique(output_labels).tolist()))
            self.assertIn(18, set(np.unique(output_labels).tolist()))
            self.assertIn(19, set(np.unique(output_labels).tolist()))
            self.assertIn(20, set(np.unique(output_labels).tolist()))
            self.assertIn(21, set(np.unique(output_labels).tolist()))
            self.assertIn(22, set(np.unique(output_labels).tolist()))
            self.assertIn(24, set(np.unique(output_labels).tolist()))
            self.assertIn(25, set(np.unique(output_labels).tolist()))
            self.assertIn(26, set(np.unique(output_labels).tolist()))
            self.assertIn(27, set(np.unique(output_labels).tolist()))
            self.assertIn(28, set(np.unique(output_labels).tolist()))
            self.assertIn(29, set(np.unique(output_labels).tolist()))
            self.assertIn(30, set(np.unique(output_labels).tolist()))
            self.assertIn(31, set(np.unique(output_labels).tolist()))
            self.assertNotIn(8, set(np.unique(output_labels).tolist()))
            self.assertTrue(Path(result.spec_yaml_path).exists())
            spec = yaml.safe_load(Path(result.spec_yaml_path).read_text())
            self.assertTrue(Path(spec["outputs"]["abdominal_organ_qa_yaml"]).exists())
            self.assertTrue(Path(spec["outputs"]["abdominal_organ_metrics_csv"]).exists())
            self.assertTrue(Path(spec["outputs"]["gi_tract_placeholder_labels"]).exists())
            abdominal_qa = yaml.safe_load(Path(spec["outputs"]["abdominal_organ_qa_yaml"]).read_text())
            self.assertEqual(abdominal_qa["summary"]["fail_count"], 0)
            self.assertGreaterEqual(abdominal_qa["summary"]["organ_count"], 14)

    def test_patient_build_dry_run_plans_ready_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            ct = np.full((24, 24, 10), -900.0, dtype=np.float32)
            ct[5:19, 5:19, 2:8] = 35.0
            organ = np.zeros(ct.shape, dtype=np.int16)
            organ[5:19, 5:19, 2:8] = 4
            organ[9:14, 9:14, 4:7] = 6
            vessel = np.zeros(ct.shape, dtype=np.uint8)
            vessel[11:13, 11:13, 2:8] = 1
            ct_path = temp_path / "ct.nii.gz"
            organ_path = temp_path / "organ.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            graph_path = temp_path / "graph.yaml"
            _write_nifti(ct_path, ct, affine)
            _write_nifti(organ_path, organ, affine)
            _write_nifti(vessel_path, vessel, affine)
            graph_path.write_text(yaml.safe_dump({"case_id": "toy", "nodes": [], "edges": []}))

            adapter = build_patient_phantom_adapter(
                input_ct_path=ct_path,
                organ_seg_path=organ_path,
                vessel_seg_path=vessel_path,
                baseline_graph_path=graph_path,
                patient_id="ready_patient",
                case_id="ready_case",
                output_dir=temp_path / "adapter",
                report_path=temp_path / "adapter.md",
            )

            result = run_patient_phantom_build(
                patient_manifest_path=adapter.manifest_yaml_path,
                output_dir=temp_path / "builds",
                report_path=temp_path / "build.md",
                dry_run=True,
            )

            self.assertEqual(result.overall_status, "planned_only")
            self.assertTrue(Path(result.build_manifest_yaml_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertIn("torso", [step.step_id for step in result.steps])
            self.assertIsNone(result.voxelized_spec_path)

    def test_patient_build_blocks_unregistered_vessel_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ct_affine = np.diag([2.0, 2.0, 4.0, 1.0])
            vessel_affine = np.diag([1.0, 1.0, 2.0, 1.0])
            ct = np.ones((20, 20, 8), dtype=np.float32)
            organ = np.ones(ct.shape, dtype=np.int16)
            vessel = np.zeros((16, 16, 6), dtype=np.uint8)
            vessel[7:9, 7:9, 1:5] = 1
            ct_path = temp_path / "ct.nii.gz"
            organ_path = temp_path / "organ.nii.gz"
            vessel_path = temp_path / "vessel.nii.gz"
            graph_path = temp_path / "graph.yaml"
            _write_nifti(ct_path, ct, ct_affine)
            _write_nifti(organ_path, organ, ct_affine)
            _write_nifti(vessel_path, vessel, vessel_affine)
            graph_path.write_text(yaml.safe_dump({"case_id": "toy", "nodes": [], "edges": []}))

            adapter = build_patient_phantom_adapter(
                input_ct_path=ct_path,
                organ_seg_path=organ_path,
                vessel_seg_path=vessel_path,
                baseline_graph_path=graph_path,
                patient_id="blocked_patient",
                case_id="blocked_case",
                output_dir=temp_path / "adapter",
                report_path=temp_path / "adapter.md",
            )

            result = run_patient_phantom_build(
                patient_manifest_path=adapter.manifest_yaml_path,
                output_dir=temp_path / "builds",
                report_path=temp_path / "build.md",
            )

            self.assertEqual(result.overall_status, "blocked_registration_required")
            self.assertIn("vessel_segmentation_registration_required", result.notes)
            self.assertIsNone(result.torso_spec_path)


if __name__ == "__main__":
    unittest.main()
