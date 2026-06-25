from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phantom_twin.materials import load_material_library


class MaterialLibraryTest(unittest.TestCase):
    def test_material_library_loads(self):
        library = load_material_library(ROOT / "configs" / "materials.yaml")
        self.assertGreaterEqual(len(library.materials), 10)
        self.assertIn("lung_inflated", library.by_id)
        self.assertIn("blood_equivalent_fluid", library.by_id)
        self.assertEqual(
            library.by_id["water_equivalent_soft_tissue"].target_relative_electron_density,
            1.0,
        )

    def test_hu_ranges_are_ordered(self):
        library = load_material_library(ROOT / "configs" / "materials.yaml")
        for material in library.materials:
            hu_min, hu_max = material.target_hu
            self.assertLessEqual(hu_min, hu_max)


if __name__ == "__main__":
    unittest.main()
