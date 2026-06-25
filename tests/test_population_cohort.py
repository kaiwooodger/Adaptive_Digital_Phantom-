from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.population_cohort import build_population_cohort


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
    names = [
        "external_air",
        "internal_air_cavity",
        "low_density_lung_like_tissue",
        "adipose_envelope",
        "generic_muscle_soft_tissue",
        "water_equivalent_fluid_or_soft_tissue",
        "liver",
        "kidneys",
        "lungs",
        "bladder",
        "trabecular_bone",
        "cortical_bone",
        "brain_or_out_of_scope_soft_tissue",
        "vessel_wall",
        "blood_equivalent_fluid",
        "contrast_filled_blood",
    ]
    return [
        {
            "index": index,
            "name": names[index],
            "material_id": names[index],
            "target_hu_midpoint": float(index),
            "mass_density_g_cm3": 1.0,
            "relative_electron_density": 1.0,
            "color": colors[index],
        }
        for index in range(16)
    ]


def _toy_labels(width_pad: int = 0, liver_shift: int = 0) -> np.ndarray:
    shape = (40, 40, 14)
    labels = np.zeros(shape, dtype=np.int16)
    labels[8 - width_pad : 32 + width_pad, 9 - width_pad : 31 + width_pad, 2:12] = 4
    labels[9 - width_pad : 31 + width_pad, 10 - width_pad : 30 + width_pad, 2:12] = 3
    labels[11:18, 12:18, 5:10] = 8
    labels[23:30, 12:18, 5:10] = 8
    labels[22 + liver_shift : 30 + liver_shift, 21:28, 5:10] = 6
    labels[13:18, 21:26, 5:9] = 7
    labels[14:26, 18:21, 4:11] = 10
    labels[20:22, 25:27, 5:10] = 13
    labels[21:23, 26:28, 5:10] = 14
    return labels


class PopulationCohortTests(unittest.TestCase):
    def test_population_cohort_exports_manifest_qc_previews_and_pca_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            reference = _toy_labels()
            reference_path = temp_path / "reference.nii.gz"
            nib.save(nib.Nifti1Image(reference, affine), reference_path)

            population_paths = []
            for index, labels in enumerate([_toy_labels(), _toy_labels(width_pad=2, liver_shift=1)], start=1):
                path = temp_path / f"population_{index}.nii.gz"
                nib.save(nib.Nifti1Image(labels, affine), path)
                population_paths.append(path)

            spec_path = temp_path / "combined_spec.yaml"
            spec_path.write_text(
                yaml.safe_dump(
                    {
                        "case_id": "toy",
                        "outputs": {"blood_material_labels": str(reference_path)},
                        "regions": _toy_regions(),
                    }
                )
            )

            result = build_population_cohort(
                combined_spec_path=spec_path,
                population_label_paths=tuple(str(path) for path in population_paths),
                output_dir=temp_path / "cohort",
                cohort_id="toy_cohort",
                population_case_ids=("toy_a", "toy_b"),
                min_body_dice=0.30,
                min_body_overlap=0.30,
                report_path=temp_path / "cohort.md",
            )

            self.assertEqual(result.case_count, 2)
            self.assertEqual(result.shape_mode_count, 1)
            self.assertEqual(len(result.registered_label_paths), 2)
            self.assertTrue(Path(result.manifest_csv_path).exists())
            self.assertTrue(Path(result.registration_qc_csv_path).exists())
            self.assertTrue(Path(result.group_metrics_csv_path).exists())
            self.assertTrue(Path(result.feature_matrix_csv_path).exists())
            self.assertTrue(Path(result.pca_loadings_csv_path).exists())
            self.assertTrue(Path(result.shape_model_npz_path).exists())
            self.assertTrue(Path(result.atlas_png_path).exists())
            self.assertTrue(Path(result.spec_yaml_path).exists())
            self.assertTrue(Path(result.case_results[0].preview_png_path).exists())


if __name__ == "__main__":
    unittest.main()
