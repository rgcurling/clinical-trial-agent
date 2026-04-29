"""
Discrepancy resolver for the two-agent pipeline.

Takes Agent 1 (Claude) and Agent 2 (GPT-4o) assessments and applies
the resolution protocol:
  - agree=True                            → return Agent 1 result unchanged
  - agree=False, override                 → return Agent 2 score, set critic_override
  - agree=False, flag_uncertain           → return Agent 1 score, set critic_flagged
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trialmatch"))

from pipeline.matcher import resolve_discrepancies  # noqa: E402
from pipeline.models import MatchResult  # noqa: E402


def resolve(
    agent1_result: MatchResult,
    agent2_verdict: dict,
    *,
    topic_id: str = "live",
    nct_id: str = "",
) -> MatchResult:
    """Merge Agent 1 result with Agent 2 verdict per the protocol."""
    return resolve_discrepancies(agent1_result, agent2_verdict, topic_id=topic_id, nct_id=nct_id)
