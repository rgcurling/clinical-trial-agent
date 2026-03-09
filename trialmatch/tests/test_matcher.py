"""Tests for pipeline/matcher.py"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.matcher import (
    ClaudeMatcher,
    compute_match_score,
    parse_criteria,
)
from pipeline.models import CriterionResult, MatchResult, PatientProfile, Trial

# ── Fixtures ──────────────────────────────────────────────────────────────────

STANDARD_ELIGIBILITY = """\
Inclusion Criteria:
  * Age >= 18 years
  * ECOG performance status 0-1
  * Confirmed NSCLC diagnosis

Exclusion Criteria:
  * Prior platinum-based chemotherapy
  * Active CNS metastases
"""

SAMPLE_TRIAL = Trial(
    nct_id="NCT99999999",
    title="Test Trial",
    phase="PHASE2",
    status="RECRUITING",
    conditions=["NSCLC"],
    eligibility_criteria_raw=STANDARD_ELIGIBILITY,
)

SAMPLE_PATIENT = PatientProfile(
    raw_text="58-year-old male with Stage IIIB NSCLC, EGFR negative, no prior chemo.",
    conditions=["NSCLC"],
    stage="Stage IIIB",
    age=58,
)


# ── parse_criteria tests ──────────────────────────────────────────────────────

class TestParseCriteria(unittest.TestCase):
    def test_inclusion_list_populated(self):
        result = parse_criteria(STANDARD_ELIGIBILITY)
        self.assertIn("inclusion", result)
        self.assertGreaterEqual(len(result["inclusion"]), 2)

    def test_exclusion_list_populated(self):
        result = parse_criteria(STANDARD_ELIGIBILITY)
        self.assertIn("exclusion", result)
        self.assertGreaterEqual(len(result["exclusion"]), 1)

    def test_inclusion_contains_age_criterion(self):
        result = parse_criteria(STANDARD_ELIGIBILITY)
        combined = " ".join(result["inclusion"]).lower()
        self.assertIn("age", combined)

    def test_exclusion_contains_platinum(self):
        result = parse_criteria(STANDARD_ELIGIBILITY)
        combined = " ".join(result["exclusion"]).lower()
        self.assertIn("platinum", combined)

    def test_llm_fallback_called_on_short_text(self):
        """When regex yields < 2 criteria, LLM fallback should be called."""
        with patch("pipeline.matcher._parse_criteria_via_llm") as mock_llm:
            mock_llm.return_value = {
                "inclusion": ["Age >= 18", "Confirmed diagnosis"],
                "exclusion": ["Prior chemo"],
            }
            result = parse_criteria("Some vague eligibility text.")
            mock_llm.assert_called_once()

    def test_case_insensitive_headers(self):
        text = (
            "INCLUSION CRITERIA:\n  * Age >= 18\n  * Diagnosis confirmed\n"
            "EXCLUSION CRITERIA:\n  * Prior treatment\n"
        )
        result = parse_criteria(text)
        self.assertGreaterEqual(len(result["inclusion"]), 1)
        self.assertGreaterEqual(len(result["exclusion"]), 1)


# ── compute_match_score tests ─────────────────────────────────────────────────

def _make_criterion(ctype: str, eligible: str, confidence: float = 0.9) -> CriterionResult:
    return CriterionResult(
        criterion_text="test",
        criterion_type=ctype,
        eligible=eligible,
        confidence=confidence,
        reasoning="",
        relevant_patient_info="",
    )


class TestComputeMatchScore(unittest.TestCase):
    def test_all_inclusion_met_no_exclusion(self):
        results = [
            _make_criterion("inclusion", "true"),
            _make_criterion("inclusion", "true"),
        ]
        self.assertAlmostEqual(compute_match_score(results), 1.0)

    def test_exclusion_triggered_returns_zero(self):
        results = [
            _make_criterion("inclusion", "true"),
            _make_criterion("exclusion", "false", confidence=0.95),  # above threshold
        ]
        self.assertEqual(compute_match_score(results), 0.0)

    def test_exclusion_low_confidence_does_not_zero_score(self):
        results = [
            _make_criterion("inclusion", "true"),
            _make_criterion("exclusion", "false", confidence=0.5),  # below threshold
        ]
        self.assertGreater(compute_match_score(results), 0.0)

    def test_partial_inclusion_met(self):
        results = [
            _make_criterion("inclusion", "true"),
            _make_criterion("inclusion", "false"),
            _make_criterion("inclusion", "uncertain"),
        ]
        score = compute_match_score(results)
        self.assertAlmostEqual(score, 1 / 3)

    def test_no_inclusion_criteria_returns_zero(self):
        results = [_make_criterion("exclusion", "true")]
        self.assertEqual(compute_match_score(results), 0.0)

    def test_empty_list_returns_zero(self):
        self.assertEqual(compute_match_score([]), 0.0)

    def test_exclusion_not_triggered_below_confidence_threshold(self):
        """exclusion eligible='true' (patient doesn't trigger it) should not zero out."""
        results = [
            _make_criterion("inclusion", "true"),
            _make_criterion("exclusion", "true", confidence=0.95),
        ]
        self.assertAlmostEqual(compute_match_score(results), 1.0)


# ── ClaudeMatcher integration (mocked) ───────────────────────────────────────

class TestClaudeMatcher(unittest.TestCase):
    def _mock_claude_response(self, eligible="true", confidence=0.9):
        mock_msg = MagicMock()
        mock_msg.content = [
            MagicMock(
                text=f'{{"eligible": "{eligible}", "confidence": {confidence}, '
                     f'"reasoning": "Patient meets criterion.", '
                     f'"relevant_patient_info": "NSCLC confirmed"}}'
            )
        ]
        return mock_msg

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_match_trial_returns_match_result(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response()
        mock_anthropic_cls.return_value = mock_client

        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PATIENT, SAMPLE_TRIAL)

        self.assertIsInstance(result, MatchResult)
        self.assertEqual(result.trial.nct_id, "NCT99999999")

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_criterion_results_populated(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response()
        mock_anthropic_cls.return_value = mock_client

        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PATIENT, SAMPLE_TRIAL)

        self.assertGreater(len(result.criterion_results), 0)
        for cr in result.criterion_results:
            self.assertIsInstance(cr, CriterionResult)
            self.assertIn(cr.eligible, ("true", "false", "uncertain"))

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_one_api_call_per_criterion(self, mock_anthropic_cls):
        """Each criterion must get its own API call — never batched."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response()
        mock_anthropic_cls.return_value = mock_client

        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PATIENT, SAMPLE_TRIAL)

        # 3 inclusion + 2 exclusion = 5 criteria → 5 calls
        expected_calls = len(result.criterion_results)
        self.assertEqual(mock_client.messages.create.call_count, expected_calls)

    @patch("pipeline.matcher.anthropic.Anthropic")
    def test_score_zero_when_exclusion_triggered(self, mock_anthropic_cls):
        mock_client = MagicMock()

        def side_effect(**kwargs):
            content = kwargs.get("messages", [{}])[-1].get("content", "")
            if "Criterion Type: exclusion" in content:
                return MagicMock(
                    content=[
                        MagicMock(
                            text='{"eligible": "false", "confidence": 0.95, '
                                 '"reasoning": "Triggered.", "relevant_patient_info": "x"}'
                        )
                    ]
                )
            return self._mock_claude_response("true", 0.9)

        mock_client.messages.create.side_effect = side_effect
        mock_anthropic_cls.return_value = mock_client

        matcher = ClaudeMatcher()
        result = matcher.match_trial(SAMPLE_PATIENT, SAMPLE_TRIAL)
        self.assertEqual(result.match_score, 0.0)


if __name__ == "__main__":
    unittest.main()
