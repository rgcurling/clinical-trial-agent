"""
Stage 2 — Trial Retrieval

Queries ClinicalTrials.gov REST API v2 for recruiting trials.
Caches responses to disk by query hash. Retries on 429/500/503.
"""

import hashlib
import json
import logging
import os
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    CACHE_DIR,
    CLINICALTRIALS_BASE_URL,
    MAX_TRIALS_TO_RETRIEVE,
)
from pipeline.models import Trial

logger = logging.getLogger(__name__)


# ── Retry predicate ───────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in (429, 500, 503)
    return isinstance(exc, requests.ConnectionError)


# ── Low-level HTTP fetch with retry ──────────────────────────────────────────

@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _fetch_from_api(params: dict) -> dict:
    response = requests.get(CLINICALTRIALS_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(query_string: str) -> str:
    return hashlib.md5(query_string.encode()).hexdigest()


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_cache(key: str) -> Optional[dict]:
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(key: str, data: dict) -> None:
    path = _cache_path(key)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── JSON → Trial dataclass ────────────────────────────────────────────────────

def _parse_trial(study: dict) -> Optional[Trial]:
    try:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        cond_mod = proto.get("conditionsModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        design_mod = proto.get("designModule", {})
        contacts_mod = proto.get("contactsLocationsModule", {})

        nct_id = id_mod.get("nctId", "")
        title = id_mod.get("officialTitle") or id_mod.get("briefTitle", "")
        status = status_mod.get("overallStatus", "")
        conditions = cond_mod.get("conditions", [])
        eligibility_raw = elig_mod.get("eligibilityCriteria", "")

        phases = design_mod.get("phases", [])
        phase = phases[0] if phases else None

        locations = [
            loc.get("facility", {}).get("address", {}).get("city", "")
            for loc in contacts_mod.get("locations", [])
            if loc.get("facility", {}).get("address", {}).get("city")
        ]

        if not nct_id:
            return None

        return Trial(
            nct_id=nct_id,
            title=title,
            phase=phase,
            status=status,
            conditions=conditions,
            eligibility_criteria_raw=eligibility_raw,
            locations=locations,
        )
    except Exception as e:
        logger.warning(f"Failed to parse trial: {e}")
        return None


# ── Public retriever ──────────────────────────────────────────────────────────

def retrieve_trials(
    condition: str,
    extra_terms: Optional[str] = None,
) -> list[Trial]:
    """
    Fetch recruiting trials for *condition* from ClinicalTrials.gov.
    Returns a list of Trial dataclasses (up to MAX_TRIALS_TO_RETRIEVE).
    Results are cached to disk by query hash.
    """
    query_condition = condition
    if extra_terms:
        query_condition = f"{condition} {extra_terms}"

    params = {
        "query.cond": query_condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": MAX_TRIALS_TO_RETRIEVE,
        "format": "json",
    }

    cache_key = _cache_key(json.dumps(params, sort_keys=True))
    cached = _load_cache(cache_key)

    if cached is not None:
        logger.info(f"Cache hit for query '{query_condition}' (key={cache_key[:8]})")
        data = cached
    else:
        logger.info(f"Fetching trials for '{query_condition}' from ClinicalTrials.gov")
        try:
            data = _fetch_from_api(params)
        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            raise
        _save_cache(cache_key, data)
        logger.info(f"Cached response as {cache_key[:8]}.json")

    studies = data.get("studies", [])
    trials = [t for s in studies if (t := _parse_trial(s)) is not None]
    logger.info(f"Retrieved {len(trials)} trials for '{query_condition}'")
    return trials
