from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.label_vessel_flow_domain import build_label_vessel_flow_domain


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


class LabelVesselFlowDomainTests(unittest.TestCase):
    def test_corrected_label_volume_becomes_solver_ready_flow_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            anatomy = np.zeros((32, 32, 16), dtype=np.int16)
            anatomy[3:29, 3:29, 2:14] = 4
            anatomy[15:18, 15:18, 2:14] = 10
            vessels = np.zeros_like(anatomy)
            vessels[8:24, 10, 8] = 4
            vessels[22, 20, 8] = 13
            vessels[10:24, 23, 8] = 1
            vessels[16, 16, 8] = 4
            anatomy_path = tmp_path / "anatomy.nii.gz"
            vessel_path = tmp_path / "vessels.nii.gz"
            _write_nifti(anatomy_path, anatomy)
            _write_nifti(vessel_path, vessels)
            config_path = tmp_path / "vessel_labels.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "labels": {1: "Inferior vena cava", 4: "Aorta", 13: "Splenic artery"},
                        "graph_edge_mapping": {"visceral_origin_to_splenic_placeholder": {"labels": [5, 13]}},
                    },
                    sort_keys=False,
                )
            )
            graph = {
                "case_id": "toy",
                "nodes": [
                    {"id": "aorta_inlet", "position_mm": [7.0, 9.0, 8.0], "radius_mm": 1.0, "boundary_role": "arterial_inlet"},
                    {"id": "aorta_outlet", "position_mm": [25.0, 11.0, 8.0], "radius_mm": 1.0, "boundary_role": "arterial_outlet"},
                    {"id": "splenic_placeholder_outlet", "position_mm": [9.0, 10.0, 8.0], "radius_mm": 1.0, "boundary_role": "arterial_outlet"},
                    {"id": "ivc_inlet", "position_mm": [9.0, 22.0, 8.0], "radius_mm": 1.0, "boundary_role": "venous_inlet"},
                    {"id": "ivc_outlet", "position_mm": [25.0, 24.0, 8.0], "radius_mm": 1.0, "boundary_role": "venous_outlet"},
                ],
                "edges": [
                    {
                        "id": "aorta",
                        "source": "aorta_inlet",
                        "target": "aorta_outlet",
                        "vessel_type": "arterial",
                        "polyline_mm": [[7.0, 9.0, 8.0], [25.0, 11.0, 8.0]],
                    },
                    {
                        "id": "ivc",
                        "source": "ivc_inlet",
                        "target": "ivc_outlet",
                        "vessel_type": "venous",
                        "polyline_mm": [[9.0, 22.0, 8.0], [25.0, 24.0, 8.0]],
                    },
                    {
                        "id": "visceral_origin_to_splenic_placeholder",
                        "source": "aorta_inlet",
                        "target": "splenic_placeholder_outlet",
                        "vessel_type": "arterial",
                        "polyline_mm": [[7.0, 9.0, 8.0], [9.0, 10.0, 8.0]],
                    },
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            result = build_label_vessel_flow_domain(
                anatomy_labels_path=anatomy_path,
                vessel_labels_path=vessel_path,
                graph_yaml_path=graph_path,
                output_dir=tmp_path / "flow_domain",
                case_id="toy_flow_domain",
                vessel_label_config=config_path,
                boundary_snap_radius_mm=25.0,
                report_path=tmp_path / "report.md",
            )

            flow_domains = np.rint(np.asanyarray(nib.load(result.flow_domain_labels_path).dataobj)).astype(np.int16)
            self.assertGreater(result.arterial_voxels, 0)
            self.assertGreater(result.venous_voxels, 0)
            self.assertGreater(result.vessel_wall_voxels, 0)
            self.assertEqual(result.snapped_boundary_node_count, 5)
            self.assertEqual(result.unclassified_labels, ())
            self.assertIn(1, set(np.unique(flow_domains).astype(int).tolist()))
            self.assertIn(2, set(np.unique(flow_domains).astype(int).tolist()))
            self.assertIn(5, set(np.unique(flow_domains).astype(int).tolist()))
            self.assertEqual(int(np.count_nonzero((flow_domains > 0) & (anatomy == 10))), 0)
            spec = yaml.safe_load(Path(result.spec_yaml_path).read_text())
            self.assertEqual(spec["voxelization"]["source_graph"], result.flow_graph_yaml_path)
            self.assertGreater(len(spec["regions"]), 0)
            flow_graph = yaml.safe_load(Path(result.flow_graph_yaml_path).read_text())
            splenic_node = next(node for node in flow_graph["nodes"] if node["id"] == "splenic_placeholder_outlet")
            splenic_index = tuple(np.rint(np.asarray(splenic_node["position_mm"]) / 1.0).astype(int))
            self.assertEqual(int(vessels[splenic_index]), 13)
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertIn("Labelled Vessel Flow-Domain Generation", Path(result.report_path).read_text())


if __name__ == "__main__":
    unittest.main()
