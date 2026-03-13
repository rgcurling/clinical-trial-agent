"""Unit tests for pipeline/retriever.py"""
import json
import unittest
from unittest.mock import MagicMock, patch

from pipeline.models import Trial
from pipeline.retriever import retrieve_trials

# Minimal fake API response matching ClinicalTrials.gov v2 structure
FAKE_API_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000001",
                    "officialTitle": "Test Trial One",
                    "briefTitle": "Trial One",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "conditionsModule": {"conditions": ["Lung Cancer"]},
                "eligibilityModule": {
                    "eligibilityCriteria": (
                        "Inclusion Criteria:\n- Age 18 or older\n"
                        "Exclusion Criteria:\n- Prior chemotherapy"
                    )
                },
                "designModule": {"phases": ["PHASE2"]},
                "contactsLocationsModule": {
                    "locations": [{"facility": {"address": {"city": "Tampa"}}}]
                },
            }
        },
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000002",
                    "officialTitle": "Test Trial Two",
                    "briefTitle": "Trial Two",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "conditionsModule": {"conditions": ["NSCLC"]},
                "eligibilityModule": {
                    "eligibilityCriteria": (
                        "Inclusion Criteria:\n- Diagnosed with NSCLC\n"
                        "Exclusion Criteria:\n- Brain metastases"
                    )
                },
                "designModule": {"phases": ["PHASE3"]},
                "contactsLocationsModule": {"locations": []},
            }
        },
    ]
}


class TestRetrieveTrials(unittest.TestCase):

    @patch("pipeline.retriever._load_cache", return_value=None)
    @patch("pipeline.retriever._save_cache")
    @patch("pipeline.retriever._fetch_from_api", return_value=FAKE_API_RESPONSE)
    def test_returns_trial_objects(self, mock_fetch, mock_save, mock_cache):
        trials = retrieve_trials("lung cancer")
        self.assertEqual(len(trials), 2)
        self.assertIsInstance(trials[0], Trial)

    @patch("pipeline.retriever._load_cache", return_value=None)
    @patch("pipeline.retriever._save_cache")
    @patch("pipeline.retriever._fetch_from_api", return_value=FAKE_API_RESPONSE)
    def test_nct_ids_populated(self, mock_fetch, mock_save, mock_cache):
        trials = retrieve_trials("lung cancer")
        nct_ids = [t.nct_id for t in trials]
        self.assertIn("NCT00000001", nct_ids)
        self.assertIn("NCT00000002", nct_ids)

    @patch("pipeline.retriever._load_cache", return_value=None)
    @patch("pipeline.retriever._save_cache")
    @patch("pipeline.retriever._fetch_from_api", return_value=FAKE_API_RESPONSE)
    def test_eligibility_criteria_populated(self, mock_fetch, mock_save, mock_cache):
        trials = retrieve_trials("lung cancer")
        for trial in trials:
            self.assertGreater(len(trial.eligibility_criteria_raw), 0)

    @patch("pipeline.retriever._load_cache", return_value=FAKE_API_RESPONSE)
    @patch("pipeline.retriever._fetch_from_api")
    def test_cache_hit_skips_api(self, mock_fetch, mock_cache):
        retrieve_trials("lung cancer")
        mock_fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
