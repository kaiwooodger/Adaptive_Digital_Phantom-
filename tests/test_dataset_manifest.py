from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phantom_twin.datasets import load_dataset_manifest


class DatasetManifestTest(unittest.TestCase):
    def test_dataset_manifest_loads(self):
        manifest = load_dataset_manifest(ROOT / "configs" / "datasets.yaml")
        self.assertIn("ct_org", manifest.by_id)
        self.assertIn("imagetbad", manifest.by_id)
        self.assertEqual(manifest.selected_for_phase1["anatomy_ct"], "ct_org")

    def test_selected_sources_exist(self):
        manifest = load_dataset_manifest(ROOT / "configs" / "datasets.yaml")
        for source_id in manifest.selected_for_phase1.values():
            self.assertIn(source_id, manifest.by_id)


if __name__ == "__main__":
    unittest.main()
