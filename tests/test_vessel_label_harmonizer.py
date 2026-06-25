from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.vessel_label_harmonizer import harmonize_vessel_labels


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _write_label_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "labels": {
                    1: "Inferior vena cava",
                    4: "Aorta",
                    27: "Right renal artery",
                    28: "Left renal artery",
                }
            },
            sort_keys=False,
        )
    )


class VesselLabelHarmonizerTests(unittest.TestCase):
    def test_template_only_generates_mapping_csv_without_output_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            vessel = np.zeros((8, 8, 4), dtype=np.int16)
            vessel[1:3, 1:3, 1] = 10
            vessel[4:6, 4:6, 2] = 20
            vessel_path = temp_path / "source_vessels.nii.gz"
            config_path = temp_path / "labels.yaml"
            _write_nifti(vessel_path, vessel)
            _write_label_config(config_path)

            result = harmonize_vessel_labels(
                vessel_seg_path=vessel_path,
                case_id="template_case",
                output_dir=temp_path / "harmonized",
                target_label_config=config_path,
                required_vessel_labels=(4, 27),
                report_path=temp_path / "template.md",
            )

            self.assertEqual(result.status, "template_only_mapping_required")
            self.assertEqual(result.source_label_count, 2)
            self.assertEqual(result.mapped_source_label_count, 0)
            self.assertEqual(result.harmonized_nifti_path, "")
            self.assertTrue(Path(result.mapping_template_csv_path).exists())
            self.assertTrue(Path(result.mapping_summary_csv_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertIn("source_label", Path(result.mapping_template_csv_path).read_text())

    def test_mapping_csv_remaps_source_labels_to_required_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            vessel = np.zeros((8, 8, 4), dtype=np.int16)
            vessel[1:3, 1:3, 1] = 10
            vessel[4:6, 4:6, 2] = 20
            vessel[6:7, 6:7, 3] = 30
            vessel_path = temp_path / "source_vessels.nii.gz"
            config_path = temp_path / "labels.yaml"
            mapping_path = temp_path / "mapping.csv"
            _write_nifti(vessel_path, vessel)
            _write_label_config(config_path)
            mapping_path.write_text(
                "source_label,source_name,target_label,target_name,notes\n"
                "10,input_aorta,4,Aorta,\n"
                "20,input_renal,27,Right renal artery,\n"
            )

            result = harmonize_vessel_labels(
                vessel_seg_path=vessel_path,
                case_id="mapped_case",
                output_dir=temp_path / "harmonized",
                target_label_config=config_path,
                mapping_csv_path=mapping_path,
                required_vessel_labels=(4, 27),
                report_path=temp_path / "mapped.md",
            )

            self.assertEqual(result.status, "harmonized_ready_for_intake")
            self.assertEqual(result.mapped_source_label_count, 2)
            self.assertEqual(result.vessel_label_coverage_percent, 100.0)
            output = np.asanyarray(nib.load(result.harmonized_nifti_path).dataobj)
            self.assertEqual(set(np.unique(output).astype(int).tolist()), {0, 4, 27})
            self.assertTrue(Path(result.manifest_yaml_path).exists())


if __name__ == "__main__":
    unittest.main()
