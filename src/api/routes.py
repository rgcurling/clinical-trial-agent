"""
/match and /health endpoints.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.models import HealthResponse, MatchRequest, MatchResponse, PatientProfileOut, TrialMatchOut
from src.data.clinicaltrials_api import ClinicalTrialsAPI

# ── Resolve trialmatch package path once at import time ───────────────────────
_TRIALMATCH = os.path.join(os.path.dirname(__file__), "..", "..", "trialmatch")
sys.path.insert(0, os.path.abspath(_TRIALMATCH))
sys.path.insert(0, "/app/trialmatch")  # Docker path

from pipeline.extractor import extract_patient_profile  # noqa: E402
from pipeline.matcher import ClaudeMatcher, critic_review, resolve_discrepancies  # noqa: E402
from pipeline.ranker import rank_trials  # noqa: E402
from pipeline.explainer import generate_all_cards  # noqa: E402
from pipeline.models import Trial  # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model="claude-sonnet-4-20250514")


@router.post("/match", response_model=MatchResponse, tags=["matching"])
async def match_patient(body: MatchRequest, background_tasks: BackgroundTasks, request: Request) -> MatchResponse:
    """
    Match a patient clinical note against live ClinicalTrials.gov trials.

    1. Extract structured patient profile (Claude)
    2. Search ClinicalTrials.gov for candidate trials
    3. Score each trial against inclusion/exclusion criteria (Claude)
    4. Optionally apply GPT-4o critic review
    5. Return top `max_trials` matches with plain-English explanations
    """
    t_start = time.perf_counter()

    # 1. Extract patient profile
    try:
        profile = extract_patient_profile(body.patient_text)
    except Exception as exc:
        logger.error(f"Profile extraction failed: {exc}")
        raise HTTPException(status_code=422, detail=f"Could not parse patient note: {exc}")

    # 2. Fetch live trials
    try:
        condition_query = " ".join(profile.conditions[:2]) if profile.conditions else body.patient_text[:200]
        async with ClinicalTrialsAPI() as api:
            raw_trials = await api.search(
                condition_query,
                location=body.location,
                status=body.status_filter,
                limit=50,
            )
        trials = [
            Trial(
                nct_id=t["nct_id"],
                title=t["title"],
                phase=t.get("phase"),
                status=t.get("status", "RECRUITING"),
                conditions=t.get("conditions", []),
                eligibility_criteria_raw=t.get("eligibility", ""),
                locations=t.get("locations", []),
            )
            for t in raw_trials
        ]
    except Exception as exc:
        logger.error(f"ClinicalTrials.gov fetch failed: {exc}")
        raise HTTPException(status_code=502, detail="Could not reach ClinicalTrials.gov")

    n_candidates = len(trials)
    if not trials:
        elapsed = (time.perf_counter() - t_start) * 1000
        return MatchResponse(
            status="no_trials_found",
            patient_profile=_profile_out(profile),
            matches=[],
            n_candidates_retrieved=0,
            n_candidates_matched=0,
            processing_time_ms=round(elapsed, 1),
        )

    # 3. Match (Claude per trial, capped at 20 for latency)
    try:
        matcher = ClaudeMatcher()
        match_results = matcher.match_trials(profile, trials[:20])
    except Exception as exc:
        logger.error(f"Matching failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Matching error: {exc}")

    # 4. Optional GPT-4o critic
    if body.use_critic:
        try:
            reviewed = []
            for mr in match_results:
                verdict = critic_review(profile, mr.trial, mr)
                reviewed.append(resolve_discrepancies(mr, verdict, topic_id="live", nct_id=mr.trial.nct_id))
            match_results = reviewed
        except Exception as exc:
            logger.warning(f"Critic review failed (continuing without): {exc}")

    # 5. Rank + explain
    ranked = rank_trials(match_results)
    top_n = ranked[: body.max_trials]
    try:
        cards = generate_all_cards(top_n)
    except Exception:
        cards = [None] * len(top_n)

    # 6. Build response
    raw_by_id = {t["nct_id"]: t for t in raw_trials}
    matches_out = [
        TrialMatchOut(
            rank=rank_i,
            nct_id=mr.trial.nct_id,
            title=mr.trial.title,
            phase=mr.trial.phase,
            overall_score=mr.overall_score,
            met_criteria=mr.met_criteria,
            failed_criteria=mr.failed_criteria,
            uncertain_criteria=mr.uncertain_criteria,
            hard_exclusion=mr.hard_exclusion,
            exclusion_reason=mr.exclusion_reason,
            explanation=card.get("card_text") if card else None,
            fk_grade=card.get("fk_grade") if card else None,
            trial_url=raw_by_id.get(mr.trial.nct_id, {}).get(
                "trial_url", f"https://clinicaltrials.gov/study/{mr.trial.nct_id}"
            ),
            locations=mr.trial.locations or [],
            critic_flagged=mr.critic_flagged,
            critic_override=mr.critic_override,
        )
        for rank_i, (mr, card) in enumerate(zip(top_n, cards), 1)
    ]

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    background_tasks.add_task(_log_request, body.patient_text, len(matches_out), elapsed_ms)

    return MatchResponse(
        status="success",
        patient_profile=_profile_out(profile),
        matches=matches_out,
        n_candidates_retrieved=n_candidates,
        n_candidates_matched=len(match_results),
        processing_time_ms=round(elapsed_ms, 1),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _profile_out(profile) -> PatientProfileOut:
    return PatientProfileOut(
        conditions=profile.conditions or [],
        age=profile.age,
        gender=profile.gender,
        biomarkers=profile.biomarkers or [],
        stage=profile.stage,
        medications=profile.medications or [],
        performance_status=getattr(profile, "performance_status", None),
    )


def _log_request(patient_text: str, n_matches: int, elapsed_ms: float) -> None:
    logger.info(f"match completed n_matches={n_matches} elapsed_ms={elapsed_ms:.1f} text_len={len(patient_text)}")
