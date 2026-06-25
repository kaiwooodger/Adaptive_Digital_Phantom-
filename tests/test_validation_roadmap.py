from pathlib import Path
import tempfile
import unittest

import yaml

from phantom_twin.validation_roadmap import build_validation_roadmap


class ValidationRoadmapTests(unittest.TestCase):
    def test_validation_roadmap_builds_protocol_from_clinical_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audit_dir = temp_path / "audit"
            audit_dir.mkdir()
            roadmap_csv = audit_dir / "roadmap.csv"
            roadmap_csv.write_text(
                "priority,domain,check_id,status,clinical_blocker,recommended_action,evidence_path\n"
                "1,vascular_anatomy,patient_specific_vascular_template_limit,review,True,Stage patient vessels,artifact.csv\n"
                "1,radiotherapy,clinical_dose_engine_gap,review,True,Connect TPS dose,limitations.md\n"
            )
            audit_yaml = audit_dir / "audit.yaml"
            audit_yaml.write_text(
                yaml.safe_dump(
                    {
                        "release_id": "toy_rc1",
                        "case_id": "toy_case",
                        "outputs": {"roadmap_csv": str(roadmap_csv)},
                        "checks": [],
                    },
                    sort_keys=False,
                )
            )

            result = build_validation_roadmap(
                readiness_audit_yaml_path=audit_yaml,
                output_dir=temp_path / "roadmap",
                roadmap_id="toy_validation_roadmap",
                report_path=temp_path / "roadmap.md",
            )

            self.assertEqual(result.release_id, "toy_rc1")
            self.assertEqual(result.task_count, 2)
            self.assertEqual(result.high_priority_task_count, 2)
            self.assertTrue(Path(result.protocol_markdown_path).exists())
            self.assertTrue(Path(result.tasks_csv_path).exists())
            self.assertTrue(Path(result.acceptance_criteria_csv_path).exists())
            self.assertTrue(Path(result.dataset_requirements_csv_path).exists())
            self.assertTrue(Path(result.roadmap_yaml_path).exists())
            self.assertTrue(Path(result.roadmap_png_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            protocol = Path(result.protocol_markdown_path).read_text()
            self.assertIn("Clinical Validation Gap-Closure Protocol", protocol)
            self.assertIn("patient-specific", protocol.lower())
            tasks = Path(result.tasks_csv_path).read_text()
            self.assertIn("P1", tasks)
            self.assertIn("P3", tasks)


if __name__ == "__main__":
    unittest.main()
