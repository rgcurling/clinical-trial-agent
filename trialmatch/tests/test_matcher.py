"""Unit tests for pipeline/matcher.py"""
import json
import unittest
from unittest.mock import MagicMock, patch

from pipeline.matcher import ClaudeMatcher, compute_match_score, parse_criteria
from pipeline.models import CriterionResult, MatchResult, PatientProfile, Trial

SAMPLE_PROFILE = PatientProfile(
    raw_text="58-year-old male with Stage IIIB NSCLC, EGFR negative, Tampa FL.",
    conditions=["non-small cell lung cancer"],
    stage="Stage IIIB",
    age=58,
    location="Tampa, FL",
    biomarkers=["EGFR negative"],
)

SAMPLE_TRIAL = Trial(
    nct_id="NCT00000001",
    title="Test NSCLC Trial",
    phase="PHASE2",
    status="RECRUITING",
    conditions=["Lung Cancer"],
    eligibility_criteria_raw=(
        "Inclusion Criteria:\n"
        "- Age 18 or older\n"
        "- Diagnosis of non-small cell lung cancer\n"
        "Exclusion Criteria:\n"
        "- Prior chemotherapy within 6 months"
    ),
)

FAKE_CLAUDE_RESPONSE = json.dumps({
    "eligible": "true",
    "confidence": 0.92,
    "reasoning": "Patient is 58 years old, meets age >= 18 requirement.",
    "relevant_patient_info": "58-year-old",
})


class TestParseCriteria(unittest.TestCase):
    def test_splits_inclusion_exclusion(self):
        result = parse_criteria(SAMPLE_TRIAL.eligibility_criteria_raw)
        self.assertIn("inclusion", result)
        self.assertIn("exclusion", result)
        self.assertGreater(len(result["inclusion"]), 0)
        self.assertGreater(len(result["exclusion"]), 0)

    def test_inclusion_contains_age(self):
        result = parse_criteria(SAMPLE_TRIAL.eligibility_criteria_raw)
        combined = " ".join(result["inclusion"]).lower()
        self.assertIn("age", combined)


class TestComputeMatchScore(unittest.TestCase):
    def test_full_inclusion_met(self):
        results = [
            CriterionResult("Age >= 18", "inclusion", "true", 0.95, "ok", "58"),
            CriterionResult("NSCLC diagnosis", "inclusion", "true", 0.9, "ok", "lung cancer"),
        ]
        score = compute_match_score(results)
        self.assertAlmostEqual(score, 1.0)

    def test_exclusion_triggered_returns_zero(self):
        results = [
            CriterionResult("Age >= 18", "inclusion", "true", 0.95, "ok", "58"),
            CriterionResult("No prior chemo", "exclusion", "false", 0.95, "had chemo", "chemo"),
        ]
        score = compute_match_score(results)
        self.assertEqual(score, 0.0)

    def test_empty_inclusion_returns_zero(self):
        score = compute_match_score([])
        self.assertEqual(score, 0.0)


class TestClaudeMatcher(unittest.TestCase):
    def _make_mock_client(self):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=FAKE_CLAUDE_RESPONSE)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        return mock_client

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_match_trial_returns_match_result(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PROFILE, SAMPLE_TRIAL)
        self.assertIsInstance(result, MatchResult)

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_match_score_positive_when_criteria_met(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PROFILE, SAMPLE_TRIAL)
        self.assertGreater(result.match_score, 0.0)

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_criterion_results_populated(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PROFILE, SAMPLE_TRIAL)
        self.assertGreater(len(result.criterion_results), 0)

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_match_trials_respects_max(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        # Pass 15 identical trials; should only match up to MAX_TRIALS_TO_MATCH (10)
        trials = [SAMPLE_TRIAL] * 15
        results = matcher.match_trials(SAMPLE_PROFILE, trials)
        self.assertLessEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main()
