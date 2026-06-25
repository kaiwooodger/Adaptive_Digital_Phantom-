from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.pca_variants import generate_pca_mode_variants


def _toy_regions() -> list[dict[str, object]]:
    colors = [
        "#0d1b2a",
        "#6c757d",
        "#8ecae6",
        "#ffd166",
        "#d95d39",
        "#4cc9f0",
        "#9d4edd",
        "#f72585",
        "#48cae4",
        "#f4d35e",
        "#e9ecef",
        "#ffffff",
        "#80ed99",
        "#ff9f1c",
        "#0077b6",
        "#ef476f",
    ]
    return [
        {
            "index": index,
            "name": f"label_{index}",
            "material_id": f"label_{index}",
            "target_hu_midpoint": float(index),
            "mass_density_g_cm3": 1.0,
            "relative_electron_density": 1.0,
            "color": colors[index],
        }
        for index in range(16)
    ]


def _toy_labels(width_pad: int = 0) -> np.ndarray:
    labels = np.zeros((28, 28, 10), dtype=np.int16)
    labels[7 - width_pad : 21 + width_pad, 8 - width_pad : 20 + width_pad, 2:8] = 4
    labels[9:14, 10:16, 4:7] = 8
    labels[16:21, 15:22, 4:8] = 6
    labels[11:16, 17:22, 4:7] = 7
    labels[13:16, 13:16, 3:8] = 10
    labels[15:17, 18:20, 4:8] = 13
    labels[16:18, 19:21, 4:8] = 14
    return labels


class PcaModeVariantTests(unittest.TestCase):
    def test_generate_pca_mode_variants_exports_atlas_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            reference = _toy_labels()
            reference_path = temp_path / "reference.nii.gz"
            nib.save(nib.Nifti1Image(reference, affine), reference_path)

            population_paths = []
            for index, labels in enumerate([_toy_labels(), _toy_labels(width_pad=2)], start=1):
                path = temp_path / f"registered_{index}.nii.gz"
                nib.save(nib.Nifti1Image(labels, affine), path)
                population_paths.append(path)

            combined_spec = temp_path / "combined_spec.yaml"
            combined_spec.write_text(
                yaml.safe_dump(
                    {
                        "outputs": {
                            "blood_material_labels": str(reference_path),
                            "contrast_material_labels": str(reference_path),
                        },
                        "regions": _toy_regions(),
                    }
                )
            )
            cohort_spec = temp_path / "cohort_spec.yaml"
            cohort_spec.write_text(
                yaml.safe_dump(
                    {
                        "outputs": {"registered_label_paths": [str(path) for path in population_paths]},
                        "cases": [{"case_id": "toy_a"}, {"case_id": "toy_b"}],
                    }
                )
            )

            result = generate_pca_mode_variants(
                combined_spec_path=combined_spec,
                cohort_spec_path=cohort_spec,
                output_dir=temp_path / "variants",
                case_id="toy_modes",
                mode_count=1,
                amplitude=0.5,
                max_modes=1,
                report_path=temp_path / "variants.md",
            )

            self.assertEqual(result.variant_count, 3)
            self.assertTrue(Path(result.metrics_csv_path).exists())
            self.assertTrue(Path(result.atlas_png_path).exists())
            self.assertTrue(Path(result.spec_yaml_path).exists())
            self.assertTrue(Path(result.variants[0].material_labels_path).exists())


if __name__ == "__main__":
    unittest.main()
