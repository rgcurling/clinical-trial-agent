"""Tests for pipeline/extractor.py"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.extractor import RegexExtractor, extract_patient_profile
from pipeline.models import PatientProfile

SAMPLE_TEXT = (
    "58-year-old male with Stage IIIB NSCLC, EGFR negative, "
    "no prior chemotherapy, Indianapolis, IN."
)


class TestRegexExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = RegexExtractor()

    def test_age_extracted(self):
        profile = self.extractor.extract(SAMPLE_TEXT)
        self.assertEqual(profile.age, 58)

    def test_location_extracted(self):
        profile = self.extractor.extract(SAMPLE_TEXT)
        self.assertIsNotNone(profile.location)
        self.assertIn("Indianapolis", profile.location)

    def test_no_age_returns_none(self):
        profile = self.extractor.extract("Patient has NSCLC and EGFR mutation.")
        self.assertIsNone(profile.age)

    def test_no_crash_on_empty(self):
        profile = self.extractor.extract("")
        self.assertIsNone(profile.age)
        self.assertIsNone(profile.location)

    def test_returns_patient_profile(self):
        profile = self.extractor.extract(SAMPLE_TEXT)
        self.assertIsInstance(profile, PatientProfile)


class TestExtractPatientProfileSpacy(unittest.TestCase):
    """Test extract_patient_profile when scispaCy is available."""

    @patch("pipeline.extractor._load_spacy", return_value=True)
    @patch("pipeline.extractor._NLP")
    def test_conditions_extracted(self, mock_nlp, mock_load):
        # Build a mock doc with one DISEASE entity
        mock_ent = MagicMock()
        mock_ent.label_ = "DISEASE"
        mock_ent.text = "NSCLC"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc

        import pipeline.extractor as ext
        ext._NLP = mock_nlp

        profile = extract_patient_profile(SAMPLE_TEXT)
        self.assertIn("NSCLC", profile.conditions)

    @patch("pipeline.extractor._load_spacy", return_value=True)
    @patch("pipeline.extractor._NLP")
    def test_age_from_regex_in_spacy_path(self, mock_nlp, mock_load):
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc

        import pipeline.extractor as ext
        ext._NLP = mock_nlp

        profile = extract_patient_profile(SAMPLE_TEXT)
        self.assertEqual(profile.age, 58)

    @patch("pipeline.extractor._load_spacy", return_value=True)
    @patch("pipeline.extractor._NLP")
    def test_location_contains_indianapolis(self, mock_nlp, mock_load):
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc

        import pipeline.extractor as ext
        ext._NLP = mock_nlp

        profile = extract_patient_profile(SAMPLE_TEXT)
        self.assertIsNotNone(profile.location)
        self.assertIn("Indianapolis", profile.location)

    @patch("pipeline.extractor._load_spacy", return_value=True)
    @patch("pipeline.extractor._NLP")
    def test_no_age_field_is_none(self, mock_nlp, mock_load):
        no_age_text = "Patient has Stage II breast cancer, HER2 positive."
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc

        import pipeline.extractor as ext
        ext._NLP = mock_nlp

        profile = extract_patient_profile(no_age_text)
        self.assertIsNone(profile.age)


class TestExtractPatientProfileLLMFallback(unittest.TestCase):
    """Test extract_patient_profile when scispaCy is unavailable (LLM fallback)."""

    @patch("pipeline.extractor._load_spacy", return_value=False)
    @patch("pipeline.extractor._extract_via_llm")
    def test_llm_fallback_called(self, mock_llm, mock_load):
        mock_llm.return_value = {
            "conditions": ["NSCLC"],
            "stage": "Stage IIIB",
            "prior_treatments": [],
            "biomarkers": ["EGFR negative"],
            "age": 58,
            "location": "Indianapolis, IN",
            "exclusion_flags": [],
        }
        profile = extract_patient_profile(SAMPLE_TEXT)
        mock_llm.assert_called_once_with(SAMPLE_TEXT)
        self.assertEqual(profile.age, 58)
        self.assertIn("NSCLC", profile.conditions)
        self.assertIn("Indianapolis", profile.location)


if __name__ == "__main__":
    unittest.main()
