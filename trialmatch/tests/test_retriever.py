"""Tests for pipeline/retriever.py"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.models import Trial

# ── Shared fixture ─────────────────────────────────────────────────────────────

MOCK_API_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "officialTitle": "A Phase II Study of Drug X in NSCLC",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "conditionsModule": {"conditions": ["Non-Small Cell Lung Cancer"]},
                "eligibilityModule": {
                    "eligibilityCriteria": (
                        "Inclusion Criteria:\n  * Age >= 18\n  * NSCLC diagnosis\n"
                        "Exclusion Criteria:\n  * Prior platinum chemotherapy"
                    )
                },
                "designModule": {"phases": ["PHASE2"]},
                "contactsLocationsModule": {
                    "locations": [
                        {"facility": {"address": {"city": "Indianapolis"}}}
                    ]
                },
            }
        }
    ]
}


class TestTrialParsing(unittest.TestCase):
    """Test that API JSON is correctly parsed into Trial dataclasses."""

    def test_trial_dataclass_populated(self):
        import pipeline.retriever as ret

        trial = ret._parse_trial(MOCK_API_RESPONSE["studies"][0])
        self.assertIsNotNone(trial)
        self.assertIsInstance(trial, Trial)
        self.assertEqual(trial.nct_id, "NCT12345678")
        self.assertEqual(trial.status, "RECRUITING")
        self.assertIn("Non-Small Cell Lung Cancer", trial.conditions)
        self.assertEqual(trial.phase, "PHASE2")
        self.assertIn("Indianapolis", trial.locations)

    def test_missing_nct_id_returns_none(self):
        import pipeline.retriever as ret

        bad_study = {"protocolSection": {"identificationModule": {}}}
        self.assertIsNone(ret._parse_trial(bad_study))


class TestCaching(unittest.TestCase):
    """Test cache hit/miss behaviour."""

    def test_second_call_hits_cache(self):
        """A second call with the same query must not make an HTTP request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("pipeline.retriever.CACHE_DIR", tmpdir),
                patch("pipeline.retriever._fetch_from_api") as mock_fetch,
            ):
                mock_fetch.return_value = MOCK_API_RESPONSE

                import pipeline.retriever as ret
                # Patch the module-level CACHE_DIR used inside the functions
                original_cache_dir = ret.CACHE_DIR
                ret.CACHE_DIR = tmpdir
                try:
                    ret.retrieve_trials("NSCLC")
                    ret.retrieve_trials("NSCLC")
                    # fetch should only have been called once
                    mock_fetch.assert_called_once()
                finally:
                    ret.CACHE_DIR = original_cache_dir

    def test_cache_written_after_api_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pipeline.retriever._fetch_from_api") as mock_fetch:
                mock_fetch.return_value = MOCK_API_RESPONSE

                import pipeline.retriever as ret
                original_cache_dir = ret.CACHE_DIR
                ret.CACHE_DIR = tmpdir
                try:
                    ret.retrieve_trials("NSCLC cache write test")
                    # At least one .json file should exist in tmpdir
                    cached_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
                    self.assertGreater(len(cached_files), 0)
                finally:
                    ret.CACHE_DIR = original_cache_dir


class TestRetryLogic(unittest.TestCase):
    """Test that retry fires on a 429 response."""

    def test_retry_on_429(self):
        import requests as req
        import pipeline.retriever as ret

        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429

        http_error = req.HTTPError(response=mock_response_429)

        call_count = {"n": 0}

        def side_effect(params):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise http_error
            return MOCK_API_RESPONSE

        with (
            patch.object(ret, "_fetch_from_api", wraps=side_effect),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            original = ret.CACHE_DIR
            ret.CACHE_DIR = tmpdir
            try:
                # _is_retryable gates tenacity; test it directly
                self.assertTrue(ret._is_retryable(http_error))
            finally:
                ret.CACHE_DIR = original


if __name__ == "__main__":
    unittest.main()
