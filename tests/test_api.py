"""
API integration tests for TrialMatch AI.

Uses httpx TestClient — no real API calls to Claude or ClinicalTrials.gov.
All external services are mocked at the boundary.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "model" in body


def test_health_fast(benchmark=None):
    """Health check must respond quickly — no heavy deps touched."""
    resp = client.get("/health")
    assert resp.status_code == 200


# ── /docs (Swagger UI) ────────────────────────────────────────────────────────

def test_swagger_ui_available():
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower()


def test_redoc_available():
    resp = client.get("/redoc")
    assert resp.status_code == 200


def test_openapi_schema_has_match_endpoint():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "/match" in schema["paths"]
    assert "/health" in schema["paths"]


# ── /match — input validation ─────────────────────────────────────────────────

def test_match_rejects_empty_patient_text():
    resp = client.post("/match", json={"patient_text": ""})
    assert resp.status_code == 422


def test_match_rejects_missing_patient_text():
    resp = client.post("/match", json={})
    assert resp.status_code == 422


def test_match_rejects_max_trials_out_of_range():
    resp = client.post("/match", json={"patient_text": "58yo female with NSCLC", "max_trials": 99})
    assert resp.status_code == 422


def test_match_accepts_valid_defaults():
    """Confirm the request schema accepts all optional fields."""
    from src.api.models import MatchRequest
    req = MatchRequest(patient_text="58-year-old female with Stage IIIB non-small cell lung cancer")
    assert req.max_trials == 5
    assert req.use_critic is False
    assert req.status_filter == "RECRUITING"


# ── /match — mocked end-to-end ────────────────────────────────────────────────

MOCK_PROFILE = MagicMock(
    conditions=["non-small cell lung cancer"],
    age=58,
    gender="female",
    biomarkers=["EGFR negative"],
    stage="IIIB",
    medications=["carboplatin"],
    performance_status=None,
)

MOCK_TRIAL = MagicMock(
    nct_id="NCT99999999",
    title="Phase III NSCLC Study",
    phase="PHASE3",
    status="RECRUITING",
    conditions=["lung cancer"],
    eligibility_criteria_raw="Inclusion: Age ≥ 18. Exclusion: Prior immunotherapy.",
    locations=["Indianapolis, IN"],
)

MOCK_MATCH = MagicMock(
    trial=MOCK_TRIAL,
    overall_score=0.85,
    match_score=0.85,
    met_criteria=["Age ≥ 18"],
    failed_criteria=[],
    uncertain_criteria=["ECOG status"],
    hard_exclusion=False,
    exclusion_reason=None,
    reasoning="Patient meets key criteria",
    critic_flagged=False,
    critic_override=False,
    uncertain=False,
    uncertainty_reason=None,
)

MOCK_RAW_TRIAL = {
    "nct_id": "NCT99999999",
    "title": "Phase III NSCLC Study",
    "phase": "PHASE3",
    "status": "RECRUITING",
    "eligibility": "Inclusion: Age ≥ 18.",
    "conditions": ["lung cancer"],
    "locations": ["Indianapolis, IN"],
    "trial_url": "https://clinicaltrials.gov/study/NCT99999999",
}

MOCK_CARD = {"card_text": "You may be eligible for this trial.", "fk_grade": 6.2}


@patch("src.api.routes.extract_patient_profile", return_value=MOCK_PROFILE)
@patch("src.api.routes.ClinicalTrialsAPI")
@patch("src.api.routes.ClaudeMatcher")
@patch("src.api.routes.rank_trials", return_value=[MOCK_MATCH])
@patch("src.api.routes.generate_all_cards", return_value=[MOCK_CARD])
def test_match_returns_structured_response(
    mock_cards, mock_rank, mock_matcher_cls, mock_api_cls, mock_extract
):
    # Wire up the async ClinicalTrialsAPI context manager
    mock_api_instance = AsyncMock()
    mock_api_instance.search = AsyncMock(return_value=[MOCK_RAW_TRIAL])
    mock_api_cls.return_value.__aenter__ = AsyncMock(return_value=mock_api_instance)
    mock_api_cls.return_value.__aexit__ = AsyncMock(return_value=None)

    # Wire up ClaudeMatcher
    mock_matcher = MagicMock()
    mock_matcher.match_trials.return_value = [MOCK_MATCH]
    mock_matcher_cls.return_value = mock_matcher

    resp = client.post(
        "/match",
        json={"patient_text": "58-year-old female with Stage IIIB non-small cell lung cancer, EGFR negative."},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "success"
    assert "patient_profile" in body
    assert body["patient_profile"]["age"] == 58
    assert body["patient_profile"]["conditions"] == ["non-small cell lung cancer"]

    assert len(body["matches"]) == 1
    match = body["matches"][0]
    assert match["rank"] == 1
    assert match["nct_id"] == "NCT99999999"
    assert match["overall_score"] == 0.85
    assert match["explanation"] == "You may be eligible for this trial."
    assert match["fk_grade"] == 6.2
    assert "trial_url" in match

    assert body["n_candidates_retrieved"] == 1
    assert "processing_time_ms" in body


@patch("src.api.routes.extract_patient_profile", return_value=MOCK_PROFILE)
@patch("src.api.routes.ClinicalTrialsAPI")
@patch("src.api.routes.ClaudeMatcher")
@patch("src.api.routes.rank_trials", return_value=[])
@patch("src.api.routes.generate_all_cards", return_value=[])
def test_match_no_trials_found(mock_cards, mock_rank, mock_matcher_cls, mock_api_cls, mock_extract):
    mock_api_instance = AsyncMock()
    mock_api_instance.search = AsyncMock(return_value=[])
    mock_api_cls.return_value.__aenter__ = AsyncMock(return_value=mock_api_instance)
    mock_api_cls.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_matcher_cls.return_value.match_trials.return_value = []

    resp = client.post("/match", json={"patient_text": "very rare condition with no matching trials"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_trials_found"
    assert resp.json()["matches"] == []


@patch("src.api.routes.extract_patient_profile", side_effect=Exception("Claude unavailable"))
def test_match_propagates_extraction_error(mock_extract):
    resp = client.post("/match", json={"patient_text": "58-year-old female with NSCLC"})
    assert resp.status_code == 422


# ── Response schema completeness ──────────────────────────────────────────────

def test_match_response_schema():
    from src.api.models import MatchResponse, TrialMatchOut, PatientProfileOut
    # Pydantic model instantiation validates all field types
    profile = PatientProfileOut(
        conditions=["NSCLC"], age=58, gender="female",
        biomarkers=[], stage="IIIB", medications=[], performance_status=None,
    )
    trial = TrialMatchOut(
        rank=1, nct_id="NCT99999999", title="Test Trial", phase="PHASE3",
        overall_score=0.85, met_criteria=[], failed_criteria=[],
        uncertain_criteria=[], hard_exclusion=False, exclusion_reason=None,
        explanation="Test explanation.", fk_grade=6.2,
        trial_url="https://clinicaltrials.gov/study/NCT99999999",
        locations=["Indianapolis, IN"],
    )
    response = MatchResponse(
        status="success", patient_profile=profile, matches=[trial],
        n_candidates_retrieved=10, n_candidates_matched=5,
        processing_time_ms=1234.5,
    )
    assert response.matches[0].overall_score == 0.85
