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
from pipeline.models import PatientProfile, Trial

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
            loc.get("city", "")
            for loc in contacts_mod.get("locations", [])
            if loc.get("city")
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


# ── Query builders ────────────────────────────────────────────────────────────

def _build_params(
    condition: str,
    extra_terms: Optional[str],
    profile: Optional[PatientProfile],
) -> dict:
    """
    Build the CT.gov v2 query params dict from condition + optional profile.

    Enrichments applied when a profile is provided:
    - query.cond: condition only — kept broad to maximise recall
    - query.term: disease stage as a soft ranking signal (boosts stage-matched
      trials without hard-filtering) + any caller-supplied extra_terms
    - aggFilters: age category (drops pediatric-only trials for adult patients)

    Biomarkers are intentionally excluded from the query: negative markers
    (e.g. "EGFR negative") attract mutation-specific trials that are exactly
    wrong for the patient. Stage-based and criteria-level matching is left to
    the LLM matcher in Stage 3.
    """
    query_cond = condition  # bare condition for maximum recall

    params: dict = {
        "query.cond": query_cond,
        "filter.overallStatus": "RECRUITING",
        "pageSize": MAX_TRIALS_TO_RETRIEVE,
        "format": "json",
    }

    # Build soft-signal term: stage boosts rank without excluding trials
    term_parts = []
    if profile and profile.stage:
        term_parts.append(profile.stage)
    if extra_terms:
        term_parts.append(extra_terms)
    if term_parts:
        params["query.term"] = " ".join(term_parts)

    # Filter by age category so pediatric-only trials are excluded
    if profile and profile.age is not None and profile.age >= 18:
        params["aggFilters"] = "ages:adult"

    return params


# ── Public retriever ──────────────────────────────────────────────────────────

def retrieve_trials(
    condition: str,
    extra_terms: Optional[str] = None,
    profile: Optional[PatientProfile] = None,
) -> list[Trial]:
    """
    Fetch recruiting trials for *condition* from ClinicalTrials.gov.

    When *profile* is provided the query is enriched with disease stage,
    biomarkers, and age category to improve retrieval precision.

    Returns a list of Trial dataclasses (up to MAX_TRIALS_TO_RETRIEVE).
    Results are cached to disk by query hash.
    """
    params = _build_params(condition, extra_terms, profile)
    query_cond = params["query.cond"]

    cache_key = _cache_key(json.dumps(params, sort_keys=True))
    cached = _load_cache(cache_key)

    if cached is not None:
        logger.info(f"Cache hit for query '{query_cond}' (key={cache_key[:8]})")
        data = cached
    else:
        logger.info(f"Fetching trials for '{query_cond}' from ClinicalTrials.gov")
        try:
            data = _fetch_from_api(params)
        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            raise
        _save_cache(cache_key, data)
        logger.info(f"Cached response as {cache_key[:8]}.json")

    studies = data.get("studies", [])
    trials = [t for s in studies if (t := _parse_trial(s)) is not None]
    logger.info(f"Retrieved {len(trials)} trials for '{query_cond}'")
    return trials


# ── Corpus-based retriever (TF-IDF) ──────────────────────────────────────────

def retrieve_from_corpus(
    patient_text: str,
    corpus_path,
    top_k: int = 20,
) -> list[Trial]:
    """
    Retrieve top_k trials from a TREC-format JSONL corpus using TF-IDF cosine
    similarity. Each line should be a JSON object with fields:
      nct_id (or _id), title, condition (or metadata.conditions),
      eligibility_criteria (or text)

    Caches the fitted TfidfVectorizer and document matrix alongside corpus_path
    as a .pkl file so fitting only happens once.
    """
    import pickle
    from pathlib import Path

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_path = Path(corpus_path)
    cache_path = corpus_path.with_name(corpus_path.stem + ".tfidf_cache.pkl")

    trials: list[Trial] = []
    vectorizer: Optional[TfidfVectorizer] = None
    matrix = None

    if cache_path.exists():
        logger.info(f"Loading TF-IDF cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        trials = cached["trials"]
        vectorizer = cached["vectorizer"]
        matrix = cached["matrix"]
        logger.info(f"Loaded {len(trials):,} trials from TF-IDF cache")
    else:
        logger.info(f"Building TF-IDF index from {corpus_path} (one-time)...")
        docs: list[str] = []

        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                nct_id = record.get("nct_id") or record.get("_id", "")
                title = record.get("title", "")

                condition = record.get("condition", "")
                if not condition:
                    meta = record.get("metadata", {})
                    conditions = meta.get("conditions", [])
                    condition = " ".join(conditions) if isinstance(conditions, list) else str(conditions)

                eligibility = record.get("eligibility_criteria", "") or record.get("text", "")

                if not nct_id:
                    continue

                trial = Trial(
                    nct_id=nct_id,
                    title=title,
                    phase=None,
                    status="RECRUITING",
                    conditions=[condition] if condition else [],
                    eligibility_criteria_raw=eligibility,
                )
                trials.append(trial)
                docs.append(f"{title} {condition} {eligibility}")

        vectorizer = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
        matrix = vectorizer.fit_transform(docs)

        logger.info(f"Fitted TF-IDF over {len(trials):,} trials. Caching to {cache_path}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"trials": trials, "vectorizer": vectorizer, "matrix": matrix}, f)
        logger.info("TF-IDF cache saved.")

    query_vec = vectorizer.transform([patient_text])
    sims = cosine_similarity(query_vec, matrix)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [trials[i] for i in top_indices]
