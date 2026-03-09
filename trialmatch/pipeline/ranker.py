"""
Stage 4 — Trial Ranking

Hard-filters ineligible trials, then sorts by match_score descending
and uncertain_count ascending. Returns at most MAX_TRIALS_TO_RETURN.
"""

import logging

from config import EXCLUSION_CONFIDENCE_THRESHOLD, MAX_TRIALS_TO_RETURN
from pipeline.models import MatchResult

logger = logging.getLogger(__name__)


def rank_trials(match_results: list[MatchResult]) -> list[MatchResult]:
    """
    Filter and rank matched trials.

    Hard filter: drop any trial where an exclusion criterion was triggered
    with confidence > EXCLUSION_CONFIDENCE_THRESHOLD.

    Sort: match_score descending, then uncertain_count ascending
    (fewer unknowns is better at equal score).

    Returns at most MAX_TRIALS_TO_RETURN results.
    """
    filtered = [
        m for m in match_results
        if not any(
            r.criterion_type == "exclusion"
            and r.eligible == "false"
            and r.confidence > EXCLUSION_CONFIDENCE_THRESHOLD
            for r in m.criterion_results
        )
    ]

    logger.info(
        f"Ranking: {len(match_results)} total → "
        f"{len(filtered)} after exclusion filter"
    )

    ranked = sorted(
        filtered,
        key=lambda m: (m.match_score, -m.uncertain_count),
        reverse=True,
    )

    top = ranked[:MAX_TRIALS_TO_RETURN]
    logger.info(f"Returning top {len(top)} trials")
    return top
