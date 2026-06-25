from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.phantom_experiment import run_phantom_experiment_set


class PhantomExperimentSetTests(unittest.TestCase):
    def test_run_phantom_experiment_set_ranks_variant_impacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics_csv = temp_path / "approved_metrics.csv"
            manifest_path = temp_path / "approved_manifest.yaml"
            flow_spec = temp_path / "flow.yaml"
            rt_spec = temp_path / "rt.yaml"
            gamma_spec = temp_path / "gamma.yaml"
            dose_metrics = temp_path / "dose_metrics.csv"
            dose_comparison = temp_path / "dose_comparison.csv"
            gamma_summary = temp_path / "gamma_summary.csv"

            with metrics_csv.open("w", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "variant_id",
                        "body_volume_cm3",
                        "waist_cm",
                        "vascular_components",
                        "group_lungs_volume_cm3",
                        "group_liver_volume_cm3",
                        "group_kidneys_volume_cm3",
                        "group_bone_volume_cm3",
                        "group_vessel_wall_volume_cm3",
                        "group_vascular_fluid_volume_cm3",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "variant_id": "mean",
                        "body_volume_cm3": 10000,
                        "waist_cm": 80,
                        "vascular_components": 1,
                        "group_lungs_volume_cm3": 1000,
                        "group_liver_volume_cm3": 1200,
                        "group_kidneys_volume_cm3": 200,
                        "group_bone_volume_cm3": 900,
                        "group_vessel_wall_volume_cm3": 25,
                        "group_vascular_fluid_volume_cm3": 100,
                    }
                )
                writer.writerow(
                    {
                        "variant_id": "mode01_pos",
                        "body_volume_cm3": 11500,
                        "waist_cm": 83,
                        "vascular_components": 1,
                        "group_lungs_volume_cm3": 1000,
                        "group_liver_volume_cm3": 1200,
                        "group_kidneys_volume_cm3": 350,
                        "group_bone_volume_cm3": 900,
                        "group_vessel_wall_volume_cm3": 25,
                        "group_vascular_fluid_volume_cm3": 105,
                    }
                )
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_experiment",
                        "outputs": {"metrics_csv": str(metrics_csv)},
                        "variants": [
                            {
                                "variant_id": "mean",
                                "release_role": "baseline",
                                "warning_status": "clean",
                                "mode_index": None,
                                "mode_weight": 0.0,
                                "body_volume_cm3": 10000,
                                "waist_cm": 80,
                                "vascular_components": 1,
                                "qa_notes": [],
                                "qa_issues": [],
                            },
                            {
                                "variant_id": "mode01_pos",
                                "release_role": "approved_with_warning",
                                "warning_status": "warning",
                                "mode_index": 1,
                                "mode_weight": 1.0,
                                "qa_score": 80.0,
                                "body_volume_cm3": 11500,
                                "waist_cm": 83,
                                "vascular_components": 1,
                                "qa_notes": ["kidneys_delta_exceeds_soft_limit"],
                                "qa_issues": [],
                            },
                        ],
                    },
                    sort_keys=False,
                )
            )
            with dose_metrics.open("w", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=["mask_id", "state", "mean_dose_gy", "d95_gy", "v95_percent"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "mask_id": "target_ptv_synthetic_vertebral",
                        "state": "static",
                        "mean_dose_gy": 20.0,
                        "d95_gy": 19.0,
                        "v95_percent": 100.0,
                    }
                )
            with dose_comparison.open("w", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=["mask_id", "comparison_state", "delta_mean_gy", "delta_v95_percentage_points"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "mask_id": "target_ptv_synthetic_vertebral",
                        "comparison_state": "pulsatile_peak",
                        "delta_mean_gy": -0.01,
                        "delta_v95_percentage_points": -1.0,
                    }
                )
            rt_spec.write_text(
                yaml.safe_dump(
                    {
                        "outputs": {
                            "dose_metrics_csv": str(dose_metrics),
                            "dose_comparison_csv": str(dose_comparison),
                        },
                        "dose_model": {"prescription_dose_gy": 20.0},
                    }
                )
            )
            with gamma_summary.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["state", "pass_rate_percent", "max_gamma_value"])
                writer.writeheader()
                writer.writerow({"state": "pulsatile_peak", "pass_rate_percent": 99.0, "max_gamma_value": 0.5})
            gamma_spec.write_text(yaml.safe_dump({"outputs": {"summary_csv": str(gamma_summary)}}))
            flow_spec.write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "edge_count": 2,
                            "node_count": 3,
                            "boundary_count": 2,
                            "arterial_inlet_flow_mean_ml_s": 10.0,
                            "aorta_pressure_mean_pa": 13332.0,
                            "max_abs_mass_balance_residual_ml_s": 0.0,
                        }
                    }
                )
            )

            result = run_phantom_experiment_set(
                approved_set_manifest_path=manifest_path,
                rt_planning_spec_path=rt_spec,
                dose_gamma_spec_path=gamma_spec,
                flow_model_spec_path=flow_spec,
                output_dir=temp_path / "experiment",
                report_path=temp_path / "experiment.md",
            )

            self.assertEqual(result.variant_count, 2)
            self.assertEqual(result.high_impact_variant_count, 1)
            self.assertEqual(result.variants[1].rt_status, "needs_variant_specific_rerun")
            self.assertEqual(result.rt_summary["gamma"]["min_pass_rate_percent"], 99.0)
            self.assertTrue(Path(result.variant_metrics_csv_path).exists())
            self.assertTrue(Path(result.manifest_yaml_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
