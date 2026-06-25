from pathlib import Path
import csv
import tempfile
import unittest

import yaml

from phantom_twin.variant_rt_compare import compare_variant_rt_impact


def _write_dose_metrics(path: Path, ptv_mean: float, ptv_volume: float, vascular_mean: float) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "mask_id",
                "label",
                "role",
                "state",
                "volume_cm3",
                "mean_dose_gy",
                "d95_gy",
                "v95_percent",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "mask_id": "target_ptv_synthetic_vertebral",
                "label": "Synthetic vertebral PTV",
                "role": "target_ptv",
                "state": "static",
                "volume_cm3": ptv_volume,
                "mean_dose_gy": ptv_mean,
                "d95_gy": 19.0,
                "v95_percent": 100.0,
            }
        )
        writer.writerow(
            {
                "mask_id": "vascular_fluid",
                "label": "Vascular fluid",
                "role": "flow_reference",
                "state": "static",
                "volume_cm3": 100.0,
                "mean_dose_gy": vascular_mean,
                "d95_gy": 2.0,
                "v95_percent": 0.0,
            }
        )


def _write_pulsatile_comparison(path: Path, ptv_delta_v95: float) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "mask_id",
                "label",
                "role",
                "comparison_state",
                "delta_mean_gy",
                "delta_v95_percentage_points",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "mask_id": "target_ptv_synthetic_vertebral",
                "label": "Synthetic vertebral PTV",
                "role": "target_ptv",
                "comparison_state": "pulsatile_peak",
                "delta_mean_gy": -0.01,
                "delta_v95_percentage_points": ptv_delta_v95,
            }
        )


def _write_gamma(path: Path, p95_gamma: float, sampled_points: int) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "state",
                "finite_gamma_points",
                "pass_rate_percent",
                "p95_gamma",
                "max_gamma_value",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "state": "pulsatile_peak",
                "finite_gamma_points": sampled_points,
                "pass_rate_percent": 100.0,
                "p95_gamma": p95_gamma,
                "max_gamma_value": p95_gamma * 10.0,
            }
        )


class VariantRTCompareTests(unittest.TestCase):
    def test_compare_variant_rt_impact_writes_disk_light_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            baseline_metrics = temp_path / "baseline_metrics.csv"
            variant_metrics = temp_path / "variant_metrics.csv"
            baseline_pulsatile = temp_path / "baseline_pulsatile.csv"
            variant_pulsatile = temp_path / "variant_pulsatile.csv"
            baseline_gamma = temp_path / "baseline_gamma.csv"
            variant_gamma = temp_path / "variant_gamma.csv"
            baseline_rt_spec = temp_path / "baseline_rt.yaml"
            variant_rt_spec = temp_path / "variant_rt.yaml"
            baseline_gamma_spec = temp_path / "baseline_gamma.yaml"
            variant_gamma_spec = temp_path / "variant_gamma.yaml"

            _write_dose_metrics(baseline_metrics, ptv_mean=20.0, ptv_volume=10.0, vascular_mean=2.0)
            _write_dose_metrics(variant_metrics, ptv_mean=20.2, ptv_volume=12.0, vascular_mean=2.1)
            _write_pulsatile_comparison(baseline_pulsatile, ptv_delta_v95=-1.0)
            _write_pulsatile_comparison(variant_pulsatile, ptv_delta_v95=-2.5)
            _write_gamma(baseline_gamma, p95_gamma=0.20, sampled_points=25000)
            _write_gamma(variant_gamma, p95_gamma=0.15, sampled_points=10000)

            baseline_rt_spec.write_text(
                yaml.safe_dump(
                    {
                        "outputs": {
                            "dose_metrics_csv": str(baseline_metrics),
                            "dose_comparison_csv": str(baseline_pulsatile),
                        }
                    },
                    sort_keys=False,
                )
            )
            variant_rt_spec.write_text(
                yaml.safe_dump(
                    {
                        "outputs": {
                            "dose_metrics_csv": str(variant_metrics),
                            "dose_comparison_csv": str(variant_pulsatile),
                        }
                    },
                    sort_keys=False,
                )
            )
            baseline_gamma_spec.write_text(
                yaml.safe_dump(
                    {"gamma_settings": {"random_subset": 25000}, "outputs": {"summary_csv": str(baseline_gamma)}},
                    sort_keys=False,
                )
            )
            variant_gamma_spec.write_text(
                yaml.safe_dump(
                    {"gamma_settings": {"random_subset": 10000}, "outputs": {"summary_csv": str(variant_gamma)}},
                    sort_keys=False,
                )
            )

            result = compare_variant_rt_impact(
                baseline_rt_planning_spec_path=baseline_rt_spec,
                variant_rt_planning_spec_path=variant_rt_spec,
                baseline_gamma_spec_path=baseline_gamma_spec,
                variant_gamma_spec_path=variant_gamma_spec,
                output_dir=temp_path / "compare",
                report_path=temp_path / "compare.md",
                variant_id="mode01_neg",
            )

            self.assertAlmostEqual(result.summary["ptv_static_mean_delta_gy"], 0.2)
            self.assertAlmostEqual(result.summary["ptv_volume_delta_percent"], 20.0)
            self.assertEqual(result.summary["matched_gamma_states"], 1)
            self.assertIn("baseline_and_variant_gamma_random_subset_sizes_differ", result.notes)
            self.assertTrue(Path(result.dose_metric_comparison_csv_path).exists())
            self.assertTrue(Path(result.gamma_comparison_csv_path or "").exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
