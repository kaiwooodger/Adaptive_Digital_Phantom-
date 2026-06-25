from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.profile_comparison import build_profile_rerun_comparison_atlas


class ProfileRerunComparisonTests(unittest.TestCase):
    def test_profile_comparison_writes_atlas_spec_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            profile_adapter = temp_path / "adapter.yaml"
            anthropometric = temp_path / "anthro.yaml"
            reference_vascular = temp_path / "ref_vascular.yaml"
            profile_vascular = temp_path / "profile_vascular.yaml"
            reference_flow = temp_path / "ref_flow.yaml"
            profile_flow = temp_path / "profile_flow.yaml"
            reference_spatial = temp_path / "ref_spatial.yaml"
            profile_spatial = temp_path / "profile_spatial.yaml"
            reference_gamma = temp_path / "ref_gamma.yaml"
            profile_gamma = temp_path / "profile_gamma.yaml"
            profile_metrics = temp_path / "profile_metrics.csv"
            reference_metrics = temp_path / "reference_metrics.csv"

            profile_adapter.write_text(
                yaml.safe_dump(
                    {
                        "target": {"bmi": 32.0, "waist_cm": 110.0, "height_cm": 175.0},
                        "selection": {
                            "selected_variant_id": "mode03_pos",
                            "fit_status": "target_outside_current_release_envelope",
                        },
                    }
                )
            )
            anthropometric.write_text(
                yaml.safe_dump(
                    {
                        "anthropometry": {
                            "target_waist_cm": 110.0,
                            "achieved_waist_cm": 108.0,
                            "body_radial_scale": 1.2,
                            "height_scale": 1.03,
                        },
                        "quality_summary": {
                            "baseline_body_volume_cm3": 23000.0,
                            "morphed_body_volume_cm3": 35000.0,
                            "body_volume_change_percent": 52.0,
                            "vascular_components": 1,
                        },
                    }
                )
            )
            for path, overlap in [(reference_vascular, 3), (profile_vascular, 0)]:
                path.write_text(
                    yaml.safe_dump(
                        {
                            "voxelization": {
                                "connected_components": 1,
                                "arterial_components": 1,
                                "venous_components": 1,
                                "arterial_venous_overlap_voxels_before_cleanup": overlap,
                                "arterial_venous_overlap_voxels_after_cleanup": 0,
                            }
                        }
                    )
                )
            for path, split_range in [(reference_flow, 0.2), (profile_flow, 0.3)]:
                path.write_text(
                    yaml.safe_dump(
                        {
                            "summary": {
                                "node_count": 21,
                                "edge_count": 19,
                                "boundary_count": 13,
                                "arterial_inlet_flow_mean_ml_s": 80.0,
                                "arterial_inlet_flow_min_ml_s": 56.0,
                                "arterial_inlet_flow_max_ml_s": 158.0,
                                "aorta_pressure_mean_pa": 13332.0,
                                "max_outlet_split_range_percentage_points": split_range,
                                "max_abs_mass_balance_residual_ml_s": 0.0,
                            }
                        }
                    )
                )
            reference_metrics.write_text(
                "mask_id,state,mean_dose_gy,v95_percent\n"
                "target_ptv_synthetic_vertebral,static,20.1,100\n"
                "target_ptv_synthetic_vertebral,spatial_pulsatile_peak,20.0,92\n"
            )
            profile_metrics.write_text(
                "mask_id,state,mean_dose_gy,v95_percent\n"
                "target_ptv_synthetic_vertebral,static,20.05,100\n"
                "target_ptv_synthetic_vertebral,spatial_pulsatile_peak,19.9,85\n"
            )
            for path, metrics, peak in [(reference_spatial, reference_metrics, 0.4), (profile_spatial, profile_metrics, 0.5)]:
                path.write_text(
                    yaml.safe_dump(
                        {
                            "outputs": {"dose_metrics_csv": str(metrics)},
                            "dose_model": {
                                "selected_edge_count": 12,
                                "peak_phase": 0.2,
                                "trough_phase": 0.65,
                                "max_abs_peak_delta_gy": peak,
                                "max_abs_trough_delta_gy": 0.1,
                            },
                        }
                    )
                )
            for path in (reference_gamma, profile_gamma):
                path.write_text(
                    yaml.safe_dump(
                        {
                            "state_results": [
                                {"state": "peak", "pass_rate_percent": 100.0, "p95_gamma": 0.001, "max_gamma_value": 0.6}
                            ]
                        }
                    )
                )

            result = build_profile_rerun_comparison_atlas(
                output_dir=temp_path / "out",
                report_path=temp_path / "report.md",
                profile_adapter_spec_path=profile_adapter,
                anthropometric_spec_path=anthropometric,
                reference_vascular_spec_path=reference_vascular,
                profile_vascular_spec_path=profile_vascular,
                reference_flow_spec_path=reference_flow,
                profile_flow_spec_path=profile_flow,
                reference_spatial_dose_spec_path=reference_spatial,
                profile_spatial_dose_spec_path=profile_spatial,
                reference_gamma_spec_path=reference_gamma,
                profile_gamma_spec_path=profile_gamma,
            )

            self.assertEqual(result.summary["profile"]["target_bmi"], 32.0)
            self.assertAlmostEqual(result.summary["delta_profile_minus_reference"]["rt"]["max_peak_delta_mgy"], 100.0)
            self.assertTrue(Path(result.atlas_png_path).exists())
            self.assertTrue(Path(result.spec_yaml_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
