"""Unit tests for pipeline/extractor.py"""
import unittest
from unittest.mock import MagicMock, patch

from pipeline.extractor import RegexExtractor, extract_patient_profile
from pipeline.models import PatientProfile

SAMPLE_TEXT = (
    "58-year-old male with Stage IIIB non-small cell lung cancer, "
    "EGFR negative, residing in Tampa, FL. No prior chemotherapy."
)


class TestRegexExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = RegexExtractor()

    def test_extracts_age(self):
        profile = self.extractor.extract(SAMPLE_TEXT)
        self.assertEqual(profile.age, 58)

    def test_extracts_location(self):
        profile = self.extractor.extract(SAMPLE_TEXT)
        self.assertIsNotNone(profile.location)
        self.assertIn("Tampa", profile.location)

    def test_returns_patient_profile(self):
        profile = self.extractor.extract(SAMPLE_TEXT)
        self.assertIsInstance(profile, PatientProfile)

    def test_missing_age_returns_none(self):
        profile = self.extractor.extract("Patient with diabetes in Ohio.")
        self.assertIsNone(profile.age)


class TestExtractPatientProfile(unittest.TestCase):
    """Test extract_patient_profile (LLM-primary path)."""

    @patch("pipeline.extractor._extract_via_llm")
    def test_llm_called(self, mock_llm):
        mock_llm.return_value = {
            "conditions": ["lung cancer"],
            "stage": "Stage IIIB",
            "prior_treatments": [],
            "biomarkers": ["EGFR negative"],
            "age": 58,
            "location": "Tampa, FL",
            "exclusion_flags": [],
        }
        profile = extract_patient_profile(SAMPLE_TEXT)
        mock_llm.assert_called_once()
        self.assertIsInstance(profile, PatientProfile)
        self.assertEqual(profile.age, 58)
        self.assertIn("lung cancer", profile.conditions)


if __name__ == "__main__":
    unittest.main()
