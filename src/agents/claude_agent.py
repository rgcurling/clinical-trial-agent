"""
Agent 1: Claude eligibility assessor.

Thin wrapper around trialmatch.pipeline.matcher.ClaudeMatcher
that exposes a clean interface for the multi-agent resolver.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trialmatch"))

from pipeline.matcher import ClaudeMatcher  # noqa: E402
from pipeline.models import MatchResult, PatientProfile, Trial  # noqa: E402


class ClaudeAgent:
    """Wraps ClaudeMatcher for use in the multi-agent orchestration layer."""

    def __init__(self):
        self._matcher = ClaudeMatcher()

    def assess(self, profile: PatientProfile, trial: Trial) -> MatchResult:
        results = self._matcher.match_trials(profile, [trial])
        return results[0] if results else _empty_result(trial)


def _empty_result(trial: Trial) -> MatchResult:
    from pipeline.models import MatchResult
    return MatchResult(
        trial=trial,
        overall_score=0.0,
        match_score=0.0,
        met_criteria=[],
        failed_criteria=[],
        uncertain_criteria=[],
        hard_exclusion=False,
        exclusion_reason=None,
        reasoning="No assessment produced",
        uncertain=True,
        uncertainty_reason="Empty response from Claude",
        critic_flagged=False,
        critic_override=False,
    )
