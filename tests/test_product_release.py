from pathlib import Path
import base64
import tempfile
import unittest

import yaml

from phantom_twin.product_release import build_product_release_package


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class ProductReleasePackageTests(unittest.TestCase):
    def test_product_release_copies_small_artifacts_and_indexes_large_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            product_root = root / "product_case"
            product_root.mkdir()
            qa_dir = product_root / "qa"
            qa_dir.mkdir()
            render_dir = product_root / "render3d"
            render_dir.mkdir()
            meshes_dir = render_dir / "meshes"
            meshes_dir.mkdir()
            build_root = root / "build"
            build_root.mkdir()
            benchmark_dir = root / "aorta_registration_benchmark_stage001"
            benchmark_dir.mkdir()

            render_png = render_dir / "preview.png"
            render_png.write_bytes(TINY_PNG)
            mesh = meshes_dir / "bone.stl"
            mesh.write_text("solid bone\nendsolid bone\n")
            volume = build_root / "labels.nii.gz"
            volume.write_bytes(b"volume-placeholder")
            product_report = product_root / "product_report.md"
            product_report.write_text("# Product report\n")
            benchmark_report = benchmark_dir / "benchmark.md"
            benchmark_report.write_text("# Aorta benchmark\n")

            qa_yaml = qa_dir / "qa.yaml"
            qa_yaml.write_text(
                yaml.safe_dump(
                    {
                        "readiness_status": "approved_research_use",
                        "summary": {"pass_count": 2, "review_count": 0, "fail_count": 0},
                        "checks": [
                            {
                                "check_id": "build_completed",
                                "category": "build",
                                "status": "pass",
                                "metric": "overall_status",
                                "value": "completed",
                                "threshold": "completed",
                                "source_path": "",
                                "notes": [],
                            },
                            {
                                "check_id": "voxel_outside_body_fraction",
                                "category": "voxelization",
                                "status": "pass",
                                "metric": "outside_body_fraction_before_clip",
                                "value": "0.0",
                                "threshold": "<=0",
                                "source_path": str(volume),
                                "notes": [],
                            },
                        ],
                    },
                    sort_keys=False,
                )
            )
            render_scene = render_dir / "scene.yaml"
            render_scene.write_text(yaml.safe_dump({"outputs": [str(mesh)], "preview_png": str(render_png)}))
            build_manifest = build_root / "build_manifest.yaml"
            build_manifest.write_text(yaml.safe_dump({"outputs": {"label_volume": str(volume)}}))
            product_manifest = product_root / "product_manifest.yaml"
            product_manifest.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy_product_case",
                        "patient_id": "toy_patient",
                        "final_status": "research_demo_ready",
                        "outputs": {
                            "product_manifest": str(product_manifest),
                            "report": str(product_report),
                            "build_manifest": str(build_manifest),
                            "qa_yaml": str(qa_yaml),
                            "render_preview_png": str(render_png),
                            "render_scene_spec": str(render_scene),
                        },
                        "stages": [
                            {"stage_id": "qa_gate", "status": "approved_research_use", "primary_output_path": str(qa_yaml)}
                        ],
                    },
                    sort_keys=False,
                )
            )

            result = build_product_release_package(
                product_manifest_path=product_manifest,
                output_dir=root / "releases",
                release_id="toy_rc1",
                command_lines=("python -m phantom_twin.cli build-product-case ...",),
                supplemental_artifact_paths=(benchmark_dir,),
            )

            self.assertEqual(result.readiness_status, "product_release_ready")
            self.assertEqual(result.qa_pass_count, 2)
            self.assertEqual(result.qa_review_count, 0)
            self.assertEqual(result.qa_fail_count, 0)
            self.assertTrue(Path(result.readme_path).exists())
            self.assertTrue(Path(result.release_manifest_path).exists())
            self.assertTrue(Path(result.artifact_index_csv_path).exists())
            self.assertTrue(Path(result.validation_summary_csv_path).exists())
            self.assertTrue(Path(result.command_log_path).exists())
            self.assertTrue(Path(result.limitations_path).exists())
            self.assertTrue(Path(result.overview_png_path).exists())
            self.assertGreater(result.copied_artifact_count, 0)
            self.assertGreaterEqual(result.indexed_large_artifact_count, 2)
            self.assertIn("coarse major-vessel", Path(result.readme_path).read_text())
            self.assertIn("aorta_registration_benchmark", Path(result.artifact_index_csv_path).read_text())
            self.assertIn("aorta_registration_benchmark", Path(result.readme_path).read_text())


if __name__ == "__main__":
    unittest.main()
