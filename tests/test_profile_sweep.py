import unittest

from phantom_twin.profile_sweep import default_profile_sweep_targets, parse_profile_sweep_target


class ProfileSweepTests(unittest.TestCase):
    def test_parse_profile_sweep_target(self) -> None:
        target = parse_profile_sweep_target("BMI 32 Waist 110:32:110:175")

        self.assertEqual(target.profile_id, "bmi_32_waist_110")
        self.assertEqual(target.target_bmi, 32.0)
        self.assertEqual(target.target_waist_cm, 110.0)
        self.assertEqual(target.target_height_cm, 175.0)

    def test_default_targets_cover_expected_envelope(self) -> None:
        targets = default_profile_sweep_targets()

        self.assertEqual(len(targets), 4)
        self.assertEqual([target.target_waist_cm for target in targets], [85.0, 95.0, 110.0, 125.0])
        self.assertEqual([target.target_bmi for target in targets], [22.0, 27.0, 32.0, 38.0])


if __name__ == "__main__":
    unittest.main()
