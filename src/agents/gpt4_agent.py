"""
Agent 2: GPT-4o critic reviewer.

Thin wrapper around trialmatch.pipeline.matcher.critic_review
that exposes a clean interface for the multi-agent resolver.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trialmatch"))

from pipeline.matcher import critic_review  # noqa: E402
from pipeline.models import MatchResult, PatientProfile, Trial  # noqa: E402


class GPT4CriticAgent:
    """
    Independent second reviewer using GPT-4o.

    Examines the Claude match result and returns a structured verdict:
      agree=True       → accepts Agent 1's assessment
      agree=False + recommendation="override"        → GPT-4o overrides score
      agree=False + recommendation="flag_uncertain"  → result flagged for review
    """

    def review(self, profile: PatientProfile, trial: Trial, agent1_result: MatchResult) -> dict:
        return critic_review(profile, trial, agent1_result)
