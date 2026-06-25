from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.aorta_registration_benchmark import (
    apply_learned_aorta_to_vascular_graph,
    build_aorta_registration_benchmark,
)


def _write_mask(path: Path, *, x_offset: float, y_offset: float) -> None:
    data = np.zeros((32, 32, 18), dtype=np.uint8)
    for z in range(data.shape[2]):
        center_x = 15.0 + x_offset + 1.2 * np.sin(z / 4.0)
        center_y = 16.0 + y_offset + 0.8 * np.cos(z / 5.0)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= 3.0**2:
                    data[x, y, z] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 2.0, 1.0])), str(path))


class AortaRegistrationBenchmarkTests(unittest.TestCase):
    def test_benchmark_model_updates_aorta_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            masks = []
            for index, offsets in enumerate(((0.0, 0.0), (0.8, -0.3), (-0.6, 0.5)), start=1):
                mask_path = temp_path / f"case{index}_aorta.nii.gz"
                _write_mask(mask_path, x_offset=offsets[0], y_offset=offsets[1])
                masks.append(mask_path)

            manifest = temp_path / "manifest.csv"
            manifest.write_text(
                "case_id,aorta_mask_nifti_path\n"
                + "\n".join(f"case{index},{path}" for index, path in enumerate(masks, start=1))
                + "\n"
            )

            benchmark = build_aorta_registration_benchmark(
                manifest_csv_path=manifest,
                output_dir=temp_path / "benchmark",
                dataset_id="toy_aorta_benchmark",
                sample_count=18,
                report_path=temp_path / "benchmark.md",
            )

            self.assertEqual(benchmark.case_count, 3)
            self.assertTrue(Path(benchmark.model_yaml_path).exists())
            self.assertTrue(Path(benchmark.metrics_csv_path).exists())
            self.assertTrue(Path(benchmark.atlas_png_path).exists())

            graph = {
                "case_id": "toy_graph",
                "coordinate_units": "mm",
                "graph_metadata": {},
                "nodes": [
                    {
                        "id": "aorta_inlet",
                        "kind": "arterial",
                        "role": "inlet",
                        "position_mm": [20.0, 20.0, 34.0],
                        "radius_mm": 2.0,
                    },
                    {
                        "id": "descending_aorta_mid",
                        "kind": "arterial",
                        "role": "outlet",
                        "position_mm": [19.0, 21.0, 0.0],
                        "radius_mm": 2.0,
                    },
                    {
                        "id": "ivc",
                        "kind": "venous",
                        "role": "outlet",
                        "position_mm": [25.0, 21.0, 0.0],
                        "radius_mm": 3.0,
                    },
                ],
                "edges": [
                    {
                        "id": "coarse_aorta_trunk",
                        "source": "aorta_inlet",
                        "target": "descending_aorta_mid",
                        "vessel_type": "arterial",
                        "flow_role": "aorta_trunk",
                        "polyline_mm": [[20.0, 20.0, 34.0], [19.0, 21.0, 0.0]],
                        "radius_start_mm": 2.0,
                        "radius_end_mm": 2.0,
                    },
                    {
                        "id": "ivc_return",
                        "source": "ivc",
                        "target": "descending_aorta_mid",
                        "vessel_type": "venous",
                        "flow_role": "venous_return_trunk",
                        "polyline_mm": [[25.0, 21.0, 0.0], [19.0, 21.0, 0.0]],
                        "radius_start_mm": 3.0,
                        "radius_end_mm": 3.0,
                    },
                ],
            }
            graph_path = temp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            learned = apply_learned_aorta_to_vascular_graph(
                graph_yaml_path=graph_path,
                aorta_model_path=benchmark.model_yaml_path,
                output_dir=temp_path / "learned",
                case_id="toy_learned_aorta",
                point_count=18,
                max_radius_mm=2.5,
                report_path=temp_path / "learned.md",
            )

            self.assertEqual(learned.replaced_node_count, 2)
            self.assertEqual(learned.replaced_edge_count, 1)
            self.assertTrue(Path(learned.graph_yaml_path).exists())
            updated = yaml.safe_load(Path(learned.graph_yaml_path).read_text())
            aorta_edge = next(edge for edge in updated["edges"] if edge["id"] == "coarse_aorta_trunk")
            self.assertEqual(aorta_edge["flow_role"], "aorta_trunk")
            self.assertEqual(len(aorta_edge["polyline_mm"]), 18)
            self.assertLessEqual(max(aorta_edge["radius_profile_mm"]), 2.5)
            self.assertIn("polyline_replaced_from_population_learned_aorta_model", aorta_edge["notes"])
            self.assertEqual(updated["graph_metadata"]["geometry_status"], "population_learned_aorta_trunk_registered_to_target_graph_endpoints")


if __name__ == "__main__":
    unittest.main()
