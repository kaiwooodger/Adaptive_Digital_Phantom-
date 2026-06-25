from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.corrected_branch_status import build_corrected_branch_status_report


class CorrectedBranchStatusTests(unittest.TestCase):
    def test_status_report_collects_corrected_branch_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vessel_manifest = root / "vessel_manifest.yaml"
            vessel_manifest.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_corrected",
                        "metrics": {
                            "arterial_voxels": 12,
                            "venous_voxels": 20,
                            "vessel_wall_voxels": 4,
                            "snapped_boundary_node_count": 3,
                            "unclassified_labels": [],
                        },
                    },
                    sort_keys=False,
                )
            )
            vessel_spec = root / "vessel_spec.yaml"
            vessel_spec.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_corrected",
                        "voxelization": {
                            "connected_components": 2,
                            "arterial_components": 1,
                            "venous_components": 1,
                        },
                    },
                    sort_keys=False,
                )
            )
            rt_spec = root / "rt_spec.yaml"
            rt_spec.write_text(
                yaml.safe_dump(
                    {
                        "masks": [
                            {"mask_id": "vascular_fluid", "voxel_count": 100, "volume_cm3": 10.0},
                            {"mask_id": "vessel_wall", "voxel_count": 11, "volume_cm3": 1.1},
                            {"mask_id": "target_ptv_synthetic_vertebral", "volume_cm3": 2.0, "mean_hu": 120.0},
                        ]
                    },
                    sort_keys=False,
                )
            )
            flow_model = root / "flow.yaml"
            flow_model.write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "edge_count": 2,
                            "node_count": 3,
                            "boundary_count": 2,
                            "arterial_inlet_flow_mean_ml_s": 80.0,
                            "arterial_inlet_flow_min_ml_s": 55.0,
                            "arterial_inlet_flow_max_ml_s": 150.0,
                            "aorta_pressure_mean_pa": 13332.0,
                            "max_abs_mass_balance_residual_ml_s": 0.0,
                            "max_outlet_split_range_percentage_points": 0.5,
                        }
                    },
                    sort_keys=False,
                )
            )
            flow4d = root / "flow4d.yaml"
            flow4d.write_text(
                yaml.safe_dump(
                    {
                        "color_by": "velocity",
                        "color_range": {"min": 1.0, "max": 2.0},
                        "outputs": {"animation_gif": str(root / "anim.gif"), "contact_sheet_png": str(root / "sheet.png")},
                        "frames": [{"frame_index": 0}],
                    },
                    sort_keys=False,
                )
            )
            coupling_csv = root / "coupling.csv"
            coupling_csv.write_text(
                "edge_id,coupling_score,effective_distance_to_ptv_mm\n"
                "aorta_edge,0.25,0.0\n"
            )
            metrics_csv = root / "dose_metrics.csv"
            metrics_csv.write_text(
                "mask_id,state,d95_gy,v95_percent,mean_dose_gy\n"
                "target_ptv_synthetic_vertebral,static,19.0,100.0,20.0\n"
                "target_ptv_synthetic_vertebral,spatial_pulsatile_peak,18.8,90.0,19.9\n"
                "target_ptv_synthetic_vertebral,spatial_pulsatile_trough,19.1,100.0,20.1\n"
                "vascular_fluid,spatial_pulsatile_peak,2.0,0.0,2.5\n"
            )
            spatial_dose = root / "spatial.yaml"
            spatial_dose.write_text(
                yaml.safe_dump(
                    {
                        "outputs": {"dose_metrics_csv": str(metrics_csv)},
                        "dose_model": {
                            "selected_edge_count": 2,
                            "peak_phase": 0.2,
                            "trough_phase": 0.65,
                            "max_abs_peak_delta_gy": 0.3,
                            "max_abs_trough_delta_gy": 0.1,
                        },
                    },
                    sort_keys=False,
                )
            )
            gamma = root / "gamma.yaml"
            gamma.write_text(
                yaml.safe_dump(
                    {
                        "gamma_settings": {"dose_percent_threshold": 3, "distance_mm_threshold": 3, "random_subset": 100},
                        "state_results": [
                            {"pass_rate_percent": 99.0, "p95_gamma": 0.1, "max_gamma_value": 0.4},
                            {"pass_rate_percent": 100.0, "p95_gamma": 0.2, "max_gamma_value": 0.5},
                        ],
                    },
                    sort_keys=False,
                )
            )

            result = build_corrected_branch_status_report(
                output_dir=root / "out",
                case_id="toy_corrected",
                vessel_flow_manifest_path=vessel_manifest,
                vessel_flow_spec_path=vessel_spec,
                rt_package_spec_path=rt_spec,
                coupled_flow_model_path=flow_model,
                flow4d_spec_path=flow4d,
                spatial_coupling_csv_path=coupling_csv,
                spatial_dose_spec_path=spatial_dose,
                gamma_spec_path=gamma,
                report_path=root / "report.md",
            )

            self.assertEqual(result.summary["flow"]["edge_count"], 2)
            self.assertEqual(result.summary["rt_flow"]["top_coupled_edge"], "aorta_edge")
            self.assertEqual(result.summary["gamma"]["min_pass_rate_percent"], 99.0)
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.metrics_csv_path).exists())
            self.assertTrue(Path(result.atlas_png_path).exists())


if __name__ == "__main__":
    unittest.main()
