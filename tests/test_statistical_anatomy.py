from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.statistical_anatomy import build_statistical_anatomy_morph


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
    hu = [-975, -975, -750, -100, 45, 10, 60, 42, -750, 10, 300, 1050, 10, 100, 50, 325]
    density = [0.001, 0.001, 0.2, 0.92, 1.05, 1.0, 1.06, 1.05, 0.2, 1.0, 1.2, 1.85, 1.0, 1.1, 1.06, 1.03]
    red = [0.001, 0.001, 0.2, 0.95, 1.04, 1.0, 1.05, 1.04, 0.2, 1.0, 1.15, 1.65, 1.0, 1.08, 1.05, 1.05]
    return [
        {
            "index": index,
            "name": names[index],
            "material_id": names[index],
            "target_hu_midpoint": float(hu[index]),
            "mass_density_g_cm3": float(density[index]),
            "relative_electron_density": float(red[index]),
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


class StatisticalAnatomyMorphTests(unittest.TestCase):
    def test_statistical_anatomy_morph_exports_population_model_and_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            reference = _toy_labels()
            contrast = reference.copy()
            contrast[contrast == 14] = 15

            paths = {}
            for name, data in [("reference", reference), ("contrast", contrast)]:
                path = temp_path / f"{name}.nii.gz"
                nib.save(nib.Nifti1Image(data, affine), path)
                paths[name] = path

            population_paths = []
            for index, data in enumerate([_toy_labels(), _toy_labels(width_pad=2, liver_shift=1)], start=1):
                path = temp_path / f"population_{index}.nii.gz"
                nib.save(nib.Nifti1Image(data, affine), path)
                population_paths.append(path)

            spec = {
                "case_id": "toy",
                "outputs": {
                    "blood_material_labels": str(paths["reference"]),
                    "contrast_material_labels": str(paths["contrast"]),
                },
                "regions": _toy_regions(),
            }
            spec_path = temp_path / "combined_spec.yaml"
            spec_path.write_text(yaml.safe_dump(spec))

            result = build_statistical_anatomy_morph(
                combined_spec_path=spec_path,
                population_label_paths=tuple(str(path) for path in population_paths),
                output_dir=temp_path / "statistical",
                case_id="toy_population",
                population_case_ids=("toy_a", "toy_b"),
                target_height_cm=176.0,
                target_bmi=29.0,
                target_waist_cm=24.0,
                baseline_height_cm=170.0,
                baseline_bmi=24.0,
                mode_weights=(0.25,),
                report_path=temp_path / "statistical.md",
            )

            self.assertEqual(result.population_case_count, 2)
            self.assertEqual(result.shape_mode_count, 1)
            self.assertTrue(Path(result.morphed_blood_material_labels_path).exists())
            self.assertTrue(Path(result.morphed_blood_synthetic_hu_path).exists())
            self.assertTrue(Path(result.shape_model_npz_path).exists())
            self.assertTrue(Path(result.registration_csv_path).exists())
            self.assertTrue(Path(result.deformation_transforms_csv_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertGreater(result.morphed_body_volume_cm3, 0.0)
            self.assertGreater(result.achieved_waist_cm, 0.0)
            self.assertGreaterEqual(result.vascular_components, 0)


if __name__ == "__main__":
    unittest.main()
