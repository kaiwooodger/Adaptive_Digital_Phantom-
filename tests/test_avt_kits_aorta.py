from pathlib import Path
import gzip
import tempfile
import unittest
import zipfile

import nibabel as nib
import numpy as np

from phantom_twin.avt_kits_aorta import stage_avt_kits_aorta_zip


def _nrrd_bytes(data: np.ndarray, *, spacing: tuple[float, float, float], dtype_name: str, segment_name: str | None = None) -> bytes:
    header = [
        "NRRD0005",
        f"type: {dtype_name}",
        "dimension: 3",
        f"sizes: {data.shape[0]} {data.shape[1]} {data.shape[2]}",
        f"space directions: ({spacing[0]},0,0) (0,{spacing[1]},0) (0,0,{spacing[2]})",
        "encoding: gzip",
    ]
    if segment_name is not None:
        header.extend(
            [
                "Segment0_LabelValue:=1",
                f"Segment0_Name:={segment_name}",
            ]
        )
    payload = gzip.compress(np.asfortranarray(data).tobytes(order="F"))
    return ("\n".join(header) + "\n\n").encode("ascii") + payload


class AvtKitsAortaStagingTests(unittest.TestCase):
    def test_stage_avt_kits_aorta_zip_converts_toy_nrrd_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / "KiTS.zip"
            ct = np.zeros((8, 9, 5), dtype=np.int16)
            ct[2:6, 2:7, :] = 120
            seg = np.zeros(ct.shape, dtype=np.uint8)
            seg[3:5, 4:6, 1:4] = 1
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("KiTS/K1/K1.nrrd", _nrrd_bytes(ct, spacing=(1.2, 1.3, 2.5), dtype_name="short"))
                zf.writestr(
                    "KiTS/K1/K1.seg.nrrd",
                    _nrrd_bytes(seg, spacing=(1.2, 1.3, 2.5), dtype_name="unsigned char", segment_name="aorta"),
                )

            result = stage_avt_kits_aorta_zip(
                zip_path=zip_path,
                output_dir=temp_path / "processed",
                dataset_id="toy_avt_kits",
                report_path=temp_path / "report.md",
            )

            self.assertEqual(result.discovered_case_count, 1)
            self.assertEqual(result.staged_case_count, 1)
            self.assertEqual(result.failed_case_count, 0)
            self.assertEqual(result.readiness_status, "aorta_registration_practice_ready")
            self.assertTrue(Path(result.labelmap_yaml_path).exists())
            self.assertTrue(Path(result.manifest_csv_path).exists())
            self.assertTrue(Path(result.intake_csv_path).exists())
            self.assertTrue(Path(result.atlas_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())

            case = result.case_results[0]
            self.assertEqual(case.shape, ct.shape)
            self.assertEqual(case.segment_name, "aorta")
            self.assertGreater(case.aorta_volume_ml, 0.0)
            staged_label = np.asanyarray(nib.load(case.aorta_mask_nifti_path).dataobj)
            self.assertEqual(int(staged_label.max()), 1)


if __name__ == "__main__":
    unittest.main()
