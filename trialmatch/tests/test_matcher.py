"""Unit tests for pipeline/matcher.py"""
import json
import unittest
from unittest.mock import MagicMock, patch

from pipeline.matcher import (
    ClaudeMatcher,
    compute_match_score,
    compute_potential_score,
    generate_clarifying_questions,
    parse_criteria,
)
from pipeline.models import MatchResult, PatientProfile, Trial

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
    "overall_score": 0.92,
    "met_criteria": ["Age 18 or older", "Diagnosis of non-small cell lung cancer"],
    "failed_criteria": [],
    "uncertain_criteria": [],
    "hard_exclusion": False,
    "exclusion_reason": None,
    "reasoning": "Patient is 58 years old with confirmed NSCLC, meets all inclusion criteria.",
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
        score = compute_match_score(overall_score=1.0, hard_exclusion=False)
        self.assertAlmostEqual(score, 1.0)

    def test_exclusion_triggered_returns_zero(self):
        score = compute_match_score(overall_score=0.9, hard_exclusion=True)
        self.assertEqual(score, 0.0)

    def test_zero_overall_score(self):
        score = compute_match_score(overall_score=0.0, hard_exclusion=False)
        self.assertEqual(score, 0.0)


class TestComputePotentialScore(unittest.TestCase):
    def test_all_met_no_uncertain(self):
        score = compute_potential_score(["a", "b"], [], ["c"])
        self.assertAlmostEqual(score, 2 / 3)

    def test_uncertain_lifts_score(self):
        score = compute_potential_score(["a"], ["b"], [])
        self.assertAlmostEqual(score, 1.0)

    def test_empty_lists_returns_zero(self):
        self.assertEqual(compute_potential_score([], [], []), 0.0)

    def test_all_failed_returns_zero(self):
        self.assertAlmostEqual(compute_potential_score([], [], ["a", "b"]), 0.0)


class TestGenerateClarifyingQuestions(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        mock_client = MagicMock()
        result = generate_clarifying_questions(mock_client, [])
        self.assertEqual(result, [])
        mock_client.messages.create.assert_not_called()

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_returns_one_question_per_criterion(self, mock_anthropic):
        fake_response = json.dumps([
            {"criterion": "PD-L1 expression", "question": "Do you have IHC results for PD-L1?"},
        ])
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=fake_response)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        result = generate_clarifying_questions(mock_client, ["PD-L1 expression"])
        self.assertEqual(len(result), 1)
        self.assertIn("criterion", result[0])
        self.assertIn("question", result[0])

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_falls_back_on_malformed_json(self, mock_anthropic):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="not json at all")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        result = generate_clarifying_questions(mock_client, ["ECOG status"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["criterion"], "ECOG status")


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
    def test_met_criteria_populated(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PROFILE, SAMPLE_TRIAL)
        self.assertGreater(len(result.met_criteria), 0)

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_hard_exclusion_false_by_default(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PROFILE, SAMPLE_TRIAL)
        self.assertFalse(result.hard_exclusion)

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_match_trials_respects_max(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        # Pass 15 identical trials; should only match up to MAX_TRIALS_TO_MATCH (10)
        trials = [SAMPLE_TRIAL] * 15
        results = matcher.match_trials(SAMPLE_PROFILE, trials)
        self.assertLessEqual(len(results), 10)

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_potential_score_set_on_result(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PROFILE, SAMPLE_TRIAL)
        # FAKE_CLAUDE_RESPONSE: 2 met, 0 failed, 0 uncertain → potential_score = 2/2 = 1.0
        self.assertAlmostEqual(result.potential_score, 1.0, places=5)

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_clarifying_questions_empty_when_no_uncertain(self, mock_anthropic):
        mock_anthropic.return_value = self._make_mock_client()
        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PROFILE, SAMPLE_TRIAL)
        self.assertEqual(result.clarifying_questions, [])

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_potential_score_higher_than_overall_when_uncertain(self, mock_anthropic):
        uncertain_response = json.dumps({
            "overall_score": 0.5,
            "met_criteria": ["Age 18 or older"],
            "failed_criteria": [],
            "uncertain_criteria": ["Diagnosis of non-small cell lung cancer"],
            "hard_exclusion": False,
            "exclusion_reason": None,
            "reasoning": "Uncertain about diagnosis.",
            "clarifying_questions": [
                {"criterion": "Diagnosis of non-small cell lung cancer",
                 "question": "Can you confirm the histological diagnosis of NSCLC?"}
            ],
        })

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=uncertain_response)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic.return_value = mock_client

        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PROFILE, SAMPLE_TRIAL)
        self.assertGreater(result.potential_score, result.overall_score)
        self.assertEqual(len(result.clarifying_questions), 1)
        self.assertIn("criterion", result.clarifying_questions[0])


if __name__ == "__main__":
    unittest.main()
