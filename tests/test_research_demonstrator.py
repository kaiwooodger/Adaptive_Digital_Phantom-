from pathlib import Path
import base64
import tempfile
import unittest

import yaml

from phantom_twin.research_demonstrator import build_research_demonstrator_package


PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(PNG_1X1))


class ResearchDemonstratorTests(unittest.TestCase):
    def test_builds_publishable_demonstrator_from_corrected_release_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            figure = root / "figure.png"
            _write_png(figure)

            release_manifest = root / "release.yaml"
            audit_yaml = root / "audit.yaml"
            status_manifest = root / "status.yaml"
            roadmap_yaml = root / "roadmap.yaml"
            intake_yaml = root / "intake.yaml"
            harmonization_yaml = root / "harmonization.yaml"

            release_manifest.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_corrected_flow",
                        "release_id": "toy_corrected_flow_rc1",
                        "summary": {
                            "artifact_count": 20,
                            "copied_artifact_count": 12,
                            "missing_artifact_count": 0,
                            "status_summary": {
                                "vascular_domain": {
                                    "arterial_voxels": 10,
                                    "venous_voxels": 20,
                                    "vessel_wall_voxels": 5,
                                    "snapped_boundary_nodes": 3,
                                    "unclassified_labels": [],
                                },
                                "rt_material_package": {
                                    "vascular_fluid_volume_cm3": 100.0,
                                    "vessel_wall_volume_cm3": 20.0,
                                    "ptv_volume_cm3": 5.0,
                                },
                                "flow": {
                                    "node_count": 4,
                                    "edge_count": 3,
                                    "boundary_count": 3,
                                    "aorta_flow_mean_ml_s": 80.0,
                                    "aorta_flow_min_ml_s": 55.0,
                                    "aorta_flow_max_ml_s": 120.0,
                                    "max_mass_balance_residual_ml_s": 1e-8,
                                },
                                "flow4d": {"frame_count": 24},
                                "rt_flow": {
                                    "selected_edge_count": 2,
                                    "max_peak_delta_mgy": 100.0,
                                    "max_trough_delta_mgy": 40.0,
                                },
                                "gamma": {"min_pass_rate_percent": 100.0},
                            },
                        },
                    },
                    sort_keys=False,
                )
            )
            audit_yaml.write_text(
                yaml.safe_dump(
                    {
                        "scores": {
                            "overall_score_percent": 91.0,
                            "research_score_percent": 98.0,
                            "clinical_blocker_count": 5,
                        },
                        "outputs": {"scorecard_png": str(figure)},
                        "checks": [
                            {"status": "pass"},
                            {"status": "pass"},
                            {"status": "review", "clinical_blocker": True},
                        ],
                    },
                    sort_keys=False,
                )
            )
            status_manifest.write_text(
                yaml.safe_dump(
                    {
                        "outputs": {"atlas_png": str(figure)},
                        "artifacts": {
                            "flow_domain_preview": {"path": str(figure)},
                            "coupled_flow_preview": {"path": str(figure)},
                            "flow4d_contact_sheet": {"path": str(figure)},
                            "rt_qa_preview": {"path": str(figure)},
                            "spatial_coupling_preview": {"path": str(figure)},
                            "spatial_dose_preview": {"path": str(figure)},
                            "gamma_qa_preview": {"path": str(figure)},
                        },
                    },
                    sort_keys=False,
                )
            )
            roadmap_yaml.write_text(
                yaml.safe_dump(
                    {
                        "summary": {"task_count": 3},
                        "outputs": {"roadmap_png": str(figure)},
                    },
                    sort_keys=False,
                )
            )
            intake_yaml.write_text(
                yaml.safe_dump(
                    {
                        "summary": {"ready_case_count": 0, "review_case_count": 0, "missing_case_count": 1},
                        "outputs": {"preview_png": str(figure)},
                    },
                    sort_keys=False,
                )
            )
            harmonization_yaml.write_text(
                yaml.safe_dump(
                    {
                        "summary": {
                            "source_label_count": 43,
                            "mapped_source_label_count": 43,
                            "vessel_label_coverage_percent": 100.0,
                        },
                        "outputs": {"preview_png": str(figure)},
                    },
                    sort_keys=False,
                )
            )

            result = build_research_demonstrator_package(
                release_manifest_path=release_manifest,
                audit_yaml_path=audit_yaml,
                status_manifest_path=status_manifest,
                validation_roadmap_path=roadmap_yaml,
                validation_intake_path=intake_yaml,
                vessel_harmonization_path=harmonization_yaml,
                output_dir=root / "demonstrator",
                package_id="toy_demo",
                report_path=root / "demo.md",
            )

            self.assertEqual(result.readiness_status, "publishable_research_demonstrator_ready")
            self.assertEqual(result.summary["present_figure_count"], 12)
            self.assertGreaterEqual(result.summary["metric_count"], 20)
            self.assertTrue(Path(result.figure_atlas_path).exists())
            self.assertTrue(Path(result.manuscript_outline_path).exists())
            self.assertTrue(Path(result.metrics_csv_path).exists())
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
