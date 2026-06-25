from pathlib import Path
import base64
import tempfile
import unittest

import yaml

from phantom_twin.product_release_case_runner import build_product_release_case
from tests.test_product_case_runner import _write_completed_build_fixture


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class ProductReleaseCaseRunnerTests(unittest.TestCase):
    def test_existing_build_generates_release_ready_workflow_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_manifest = _write_completed_build_fixture(root / "build")
            render_preview = root / "existing_render.png"
            render_scene = root / "existing_render_scene.yaml"
            supplemental = root / "aorta_registration_benchmark_stage001"
            supplemental.mkdir()
            render_preview.write_bytes(TINY_PNG)
            render_scene.write_text("case_id: render\n")
            (supplemental / "benchmark.md").write_text("# benchmark\n")

            result = build_product_release_case(
                existing_build_manifest_path=build_manifest,
                output_dir=root / "workflow",
                product_output_dir=root / "product",
                release_output_dir=root / "release",
                release_id="toy_rc1",
                case_id="toy_release_case",
                render_3d=False,
                existing_render_preview_path=render_preview,
                existing_render_scene_spec_path=render_scene,
                release_command_lines=("python -m phantom_twin.cli build-product-release-case ...",),
                supplemental_artifact_paths=(supplemental,),
            )

            workflow_manifest = yaml.safe_load(Path(result.workflow_manifest_path).read_text())
            self.assertEqual(result.workflow_status, "release_ready")
            self.assertEqual(result.product_final_status, "research_demo_ready")
            self.assertEqual(result.release_readiness_status, "product_release_ready")
            self.assertTrue(Path(result.workflow_report_path).exists())
            self.assertTrue(Path(result.product_manifest_path).exists())
            self.assertTrue(Path(result.release_readme_path).exists())
            self.assertTrue(Path(result.release_overview_png_path).exists())
            self.assertEqual(workflow_manifest["summary"]["qa_fail_count"], 0)
            self.assertIn("aorta_registration_benchmark", Path(result.release_artifact_index_csv_path).read_text())


if __name__ == "__main__":
    unittest.main()
