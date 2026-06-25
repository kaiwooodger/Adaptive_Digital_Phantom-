from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import yaml

from phantom_twin.anthropometry import build_anthropometric_torso_morph


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


class AnthropometricMorphTests(unittest.TestCase):
    def test_anthropometric_morph_exports_variant_and_increases_body_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shape = (32, 32, 12)
            affine = np.diag([2.0, 2.0, 4.0, 1.0])
            labels = np.zeros(shape, dtype=np.int16)
            labels[6:26, 7:25, 2:10] = 4
            labels[7:25, 8:24, 2:10] = 3
            labels[10:14, 10:14, 4:8] = 8
            labels[18:23, 10:16, 4:8] = 6
            labels[14:18, 15:18, 4:8] = 10
            labels[16:18, 19:21, 5:8] = 13
            labels[17:19, 20:22, 5:8] = 14
            contrast = labels.copy()
            contrast[contrast == 14] = 15
            flow = np.zeros(shape, dtype=np.int16)
            flow[17:19, 20:22, 5:8] = 1
            flow[16:18, 19:21, 5:8] = 5

            paths = {}
            for name, data in [("blood", labels), ("contrast", contrast), ("flow", flow)]:
                path = temp_path / f"{name}.nii.gz"
                nib.save(nib.Nifti1Image(data, affine), path)
                paths[name] = path

            spec = {
                "case_id": "toy",
                "outputs": {
                    "blood_material_labels": str(paths["blood"]),
                    "contrast_material_labels": str(paths["contrast"]),
                    "flow_boundary_labels": str(paths["flow"]),
                },
                "regions": _toy_regions(),
            }
            spec_path = temp_path / "combined_spec.yaml"
            spec_path.write_text(yaml.safe_dump(spec))

            result = build_anthropometric_torso_morph(
                combined_spec_path=spec_path,
                output_dir=temp_path / "morph",
                case_id="toy_bmi30",
                target_height_cm=175.0,
                target_bmi=30.0,
                target_waist_cm=20.0,
                baseline_height_cm=170.0,
                baseline_bmi=24.0,
                baseline_waist_cm=None,
                report_path=temp_path / "morph.md",
            )

            self.assertTrue(Path(result.morphed_blood_material_labels_path).exists())
            self.assertTrue(Path(result.morphed_blood_synthetic_hu_path).exists())
            self.assertTrue(Path(result.morphed_flow_boundary_labels_path or "").exists())
            self.assertTrue(Path(result.scale_profile_csv_path).exists())
            self.assertTrue(Path(result.preview_png_path).exists())
            self.assertGreater(result.morphed_body_volume_cm3, result.baseline_body_volume_cm3)
            self.assertGreater(result.achieved_waist_cm, result.baseline_waist_cm)
            self.assertEqual(result.vascular_components, 1)


if __name__ == "__main__":
    unittest.main()
