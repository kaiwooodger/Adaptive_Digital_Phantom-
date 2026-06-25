from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.cta_vascular_graph import build_cta_derived_vascular_graph


def _disk(mask: np.ndarray, center_i: int, center_j: int, z: int, radius: int) -> None:
    ii, jj = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[:, :, z] |= (ii - center_i) ** 2 + (jj - center_j) ** 2 <= radius**2


class CtaVascularGraphTests(unittest.TestCase):
    def test_coarse_aorta_edge_uses_full_cta_centerline_extent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mask = np.zeros((48, 48, 14), dtype=np.uint8)
            for z in range(14):
                _disk(mask, 22, 22, z, 4)
            mask_path = tmp_path / "vascular_mask.nii.gz"
            nib.save(nib.Nifti1Image(mask, np.diag([1.0, 1.0, 5.0, 1.0])), str(mask_path))

            graph = {
                "case_id": "coarse",
                "graph_metadata": {"product_scope": "coarse_major_vessels_only_not_branch_rich_vascular_network"},
                "nodes": [
                    {"id": "aorta_inlet", "position_mm": [22.0, 22.0, 65.0], "radius_mm": 4.0, "kind": "arterial", "role": "inlet"},
                    {"id": "descending_aorta_mid", "position_mm": [22.0, 22.0, 55.0], "radius_mm": 4.0, "kind": "arterial", "role": "outlet"},
                ],
                "edges": [
                    {
                        "id": "coarse_aorta_trunk",
                        "source": "aorta_inlet",
                        "target": "descending_aorta_mid",
                        "flow_role": "aorta_trunk",
                        "vessel_type": "arterial",
                        "polyline_mm": [[22.0, 22.0, 65.0], [22.0, 22.0, 55.0]],
                    },
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            result = build_cta_derived_vascular_graph(
                baseline_graph_path=graph_path,
                vascular_mask_path=mask_path,
                output_dir=tmp_path / "out",
                case_id="coarse_cta",
                report_path=tmp_path / "report.md",
            )

            derived = yaml.safe_load(Path(result.graph_yaml_path).read_text())
            edge = derived["edges"][0]
            target = next(node for node in derived["nodes"] if node["id"] == "descending_aorta_mid")
            self.assertGreater(edge["length_mm"], 55.0)
            self.assertAlmostEqual(target["position_mm"][2], 0.0)
            self.assertIn("polyline_replaced_from_full_cta_lumen_centerline_for_coarse_trunk", edge["notes"])
            self.assertEqual(derived["graph_metadata"]["full_trunk_coarse_aorta_replaced_edge_count"], 1)

    def test_longitudinal_side_channel_is_not_promoted_as_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mask = np.zeros((48, 48, 12), dtype=np.uint8)
            for z in range(12):
                _disk(mask, 22, 22, z, 4)
            for z in range(1, 11):
                _disk(mask, 22, 36, z, 3)
            mask_path = tmp_path / "vascular_mask.nii.gz"
            nib.save(nib.Nifti1Image(mask, np.diag([1.0, 1.0, 5.0, 1.0])), str(mask_path))

            graph = {
                "case_id": "tiny",
                "graph_metadata": {},
                "nodes": [
                    {"id": "aorta_inlet", "position_mm": [22.0, 22.0, 55.0], "radius_mm": 4.0, "kind": "arterial", "role": "inlet"},
                    {"id": "descending_aorta_mid", "position_mm": [22.0, 22.0, 45.0], "radius_mm": 4.0, "kind": "arterial", "role": "junction"},
                    {"id": "visceral_branch_origin", "position_mm": [22.0, 22.0, 35.0], "radius_mm": 4.0, "kind": "arterial", "role": "junction"},
                    {"id": "renal_branch_origin", "position_mm": [22.0, 22.0, 25.0], "radius_mm": 4.0, "kind": "arterial", "role": "junction"},
                    {"id": "aorta_distal_anchor", "position_mm": [22.0, 22.0, 5.0], "radius_mm": 4.0, "kind": "arterial", "role": "junction"},
                    {"id": "left_renal_outlet", "position_mm": [22.0, 36.0, 25.0], "radius_mm": 2.0, "kind": "arterial", "role": "outlet"},
                ],
                "edges": [
                    {
                        "id": "aorta_inlet_to_descending",
                        "source": "aorta_inlet",
                        "target": "descending_aorta_mid",
                        "flow_role": "aorta_trunk",
                        "vessel_type": "arterial",
                        "polyline_mm": [[22.0, 22.0, 55.0], [22.0, 22.0, 45.0]],
                    },
                    {
                        "id": "descending_to_visceral_origin",
                        "source": "descending_aorta_mid",
                        "target": "visceral_branch_origin",
                        "flow_role": "aorta_trunk",
                        "vessel_type": "arterial",
                        "polyline_mm": [[22.0, 22.0, 45.0], [22.0, 22.0, 35.0]],
                    },
                    {
                        "id": "visceral_to_renal_origin",
                        "source": "visceral_branch_origin",
                        "target": "renal_branch_origin",
                        "flow_role": "aorta_trunk",
                        "vessel_type": "arterial",
                        "polyline_mm": [[22.0, 22.0, 35.0], [22.0, 22.0, 25.0]],
                    },
                    {
                        "id": "renal_origin_to_distal_aorta",
                        "source": "renal_branch_origin",
                        "target": "aorta_distal_anchor",
                        "flow_role": "aorta_trunk",
                        "vessel_type": "arterial",
                        "polyline_mm": [[22.0, 22.0, 25.0], [22.0, 22.0, 5.0]],
                    },
                    {
                        "id": "renal_origin_to_left_renal",
                        "source": "renal_branch_origin",
                        "target": "left_renal_outlet",
                        "flow_role": "renal_branch",
                        "vessel_type": "arterial",
                        "polyline_mm": [[22.0, 22.0, 25.0], [22.0, 36.0, 25.0]],
                    },
                ],
            }
            graph_path = tmp_path / "graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))

            result = build_cta_derived_vascular_graph(
                baseline_graph_path=graph_path,
                vascular_mask_path=mask_path,
                output_dir=tmp_path / "out",
                case_id="tiny_cta",
                report_path=tmp_path / "report.md",
            )

            self.assertEqual(result.promoted_branch_count, 0)
            self.assertGreaterEqual(result.branch_candidate_count, 1)
            self.assertIn("rejected_longitudinal_lumen_channel", Path(result.branch_candidates_csv_path).read_text())

            derived = yaml.safe_load(Path(result.graph_yaml_path).read_text())
            metadata = derived["graph_metadata"]
            self.assertEqual(metadata["promoted_branch_edge_count"], 0)
            self.assertEqual(metadata["geometry_status"], "cta_derived_aorta_trunk_with_retained_placeholder_branches")


if __name__ == "__main__":
    unittest.main()
