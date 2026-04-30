"""
Tool implementations for the TrialMatch orchestrator agent.

Each function is called by the Claude agent when it decides to use a tool.
All tools are synchronous wrappers so the agent loop stays simple.
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "trialmatch")))
sys.path.insert(0, "/app/trialmatch")

from pipeline.extractor import extract_patient_profile  # noqa: E402
from pipeline.matcher import ClaudeMatcher, critic_review, resolve_discrepancies  # noqa: E402
from pipeline.ranker import rank_trials  # noqa: E402
from pipeline.explainer import generate_all_cards  # noqa: E402
from pipeline.models import PatientProfile, Trial  # noqa: E402

logger = logging.getLogger(__name__)

# ── Tool schemas (passed to Claude API) ───────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "extract_patient_profile",
        "description": (
            "Parse a free-text clinical note into a structured patient profile. "
            "Extracts age, gender, conditions, cancer stage, biomarkers, medications, "
            "and performance status. Call this first before any other tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_text": {
                    "type": "string",
                    "description": "The raw patient clinical note to parse.",
                }
            },
            "required": ["patient_text"],
        },
    },
    {
        "name": "search_clinical_trials",
        "description": (
            "Search ClinicalTrials.gov for currently recruiting trials matching a condition. "
            "Returns up to 50 candidate trials with NCT IDs, titles, and eligibility criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "description": "Primary condition or disease to search for, e.g. 'non-small cell lung cancer'.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional city/state for proximity filtering, e.g. 'Indianapolis, IN'.",
                },
            },
            "required": ["condition"],
        },
    },
    {
        "name": "rank_trials_by_relevance",
        "description": (
            "Use BiomedBERT semantic embeddings to re-rank a list of trials by relevance "
            "to the patient profile. Returns the top trials ordered by semantic similarity. "
            "Call after search_clinical_trials to surface the most relevant candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_text": {
                    "type": "string",
                    "description": "Full patient clinical note used as the query.",
                },
                "trial_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of NCT IDs to re-rank.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top trials to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["patient_text", "trial_ids"],
        },
    },
    {
        "name": "evaluate_eligibility",
        "description": (
            "Use Claude to assess whether a patient meets the eligibility criteria for a specific trial. "
            "Returns an overall score (0-1), met criteria, failed criteria, uncertain criteria, "
            "and whether the patient is hard-excluded. Call this for each trial you want to evaluate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nct_id": {
                    "type": "string",
                    "description": "The NCT ID of the trial to evaluate.",
                },
            },
            "required": ["nct_id"],
        },
    },
    {
        "name": "get_critic_review",
        "description": (
            "Use GPT-4o as an independent second reviewer to validate a Claude eligibility assessment. "
            "The critic can agree, override the score, or flag the result as uncertain. "
            "Call this after evaluate_eligibility for your top candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nct_id": {
                    "type": "string",
                    "description": "The NCT ID of the trial whose assessment to review.",
                },
            },
            "required": ["nct_id"],
        },
    },
    {
        "name": "generate_patient_explanation",
        "description": (
            "Generate a plain-English explanation of trial eligibility written at an 8th-grade "
            "reading level, suitable for sharing directly with the patient. "
            "Call this for the final recommended trials."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nct_id": {
                    "type": "string",
                    "description": "NCT ID of the trial to explain.",
                },
            },
            "required": ["nct_id"],
        },
    },
]


# ── Tool executor ──────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Stateful executor that holds the patient profile and trial data
    across multiple tool calls within a single agent session.
    """

    def __init__(self):
        self._profile: PatientProfile | None = None
        self._raw_trials: list[dict] = []
        self._trials: list[Trial] = []
        self._trial_by_id: dict[str, Trial] = {}
        self._assessments: dict[str, object] = {}   # nct_id → MatchResult
        self._matcher = ClaudeMatcher()
        self._bert_model = None   # loaded once on first rank call

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch tool call and return a JSON string result."""
        try:
            if tool_name == "extract_patient_profile":
                return self._extract_profile(tool_input["patient_text"])
            elif tool_name == "search_clinical_trials":
                return self._search_trials(tool_input["condition"], tool_input.get("location"))
            elif tool_name == "rank_trials_by_relevance":
                return self._rank_trials(
                    tool_input["patient_text"],
                    tool_input["trial_ids"],
                    tool_input.get("top_k", 10),
                )
            elif tool_name == "evaluate_eligibility":
                return self._evaluate(tool_input["nct_id"])
            elif tool_name == "get_critic_review":
                return self._critic(tool_input["nct_id"])
            elif tool_name == "generate_patient_explanation":
                return self._explain(tool_input["nct_id"])
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            logger.error(f"Tool {tool_name} failed: {exc}")
            return json.dumps({"error": str(exc)})

    # ── Tool implementations ───────────────────────────────────────────────────

    def _extract_profile(self, patient_text: str) -> str:
        self._profile = extract_patient_profile(patient_text)
        return json.dumps({
            "conditions": self._profile.conditions,
            "age": self._profile.age,
            "gender": self._profile.gender,
            "stage": self._profile.stage,
            "biomarkers": self._profile.biomarkers,
            "medications": self._profile.medications,
        })

    def _search_trials(self, condition: str, location: str | None) -> str:
        import asyncio
        from src.data.clinicaltrials_api import ClinicalTrialsAPI

        async def _fetch():
            async with ClinicalTrialsAPI() as api:
                return await api.search(condition, location=location, limit=50)

        self._raw_trials = asyncio.run(_fetch())
        self._trials = [
            Trial(
                nct_id=t["nct_id"],
                title=t["title"],
                phase=t.get("phase"),
                status=t.get("status", "RECRUITING"),
                conditions=t.get("conditions", []),
                eligibility_criteria_raw=t.get("eligibility", ""),
                locations=t.get("locations", []),
            )
            for t in self._raw_trials
        ]
        self._trial_by_id = {t.nct_id: t for t in self._trials}

        return json.dumps({
            "n_found": len(self._trials),
            "trials": [
                {"nct_id": t.nct_id, "title": t.title, "phase": t.phase, "locations": t.locations[:2]}
                for t in self._trials[:20]
            ],
        })

    def _rank_trials(self, patient_text: str, trial_ids: list[str], top_k: int) -> str:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            if self._bert_model is None:
                logger.info("Loading BiomedBERT model (one-time per session)...")
                self._bert_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
            model = self._bert_model
            query_emb = model.encode(patient_text, convert_to_numpy=True)
            query_emb = query_emb / max(float(np.linalg.norm(query_emb)), 1e-12)

            candidates = [self._trial_by_id[nid] for nid in trial_ids if nid in self._trial_by_id]
            docs = [f"{t.title} {' '.join(t.conditions)} {t.eligibility_criteria_raw[:500]}" for t in candidates]
            doc_embs = model.encode(docs, convert_to_numpy=True)
            norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
            doc_embs = doc_embs / np.clip(norms, 1e-12, None)

            sims = np.dot(doc_embs, query_emb)
            top_idx = sims.argsort()[::-1][:top_k]
            ranked = [(candidates[i].nct_id, float(sims[i])) for i in top_idx]
        except ImportError:
            # Fallback: return same order if sentence-transformers unavailable
            ranked = [(nid, 1.0) for nid in trial_ids[:top_k]]

        return json.dumps({
            "ranked_trials": [{"nct_id": nid, "similarity": round(score, 4)} for nid, score in ranked]
        })

    def _evaluate(self, nct_id: str) -> str:
        if self._profile is None:
            return json.dumps({"error": "Call extract_patient_profile first."})
        trial = self._trial_by_id.get(nct_id)
        if trial is None:
            return json.dumps({"error": f"Trial {nct_id} not found. Call search_clinical_trials first."})

        results = self._matcher.match_trials(self._profile, [trial])
        if not results:
            return json.dumps({"error": "Matcher returned no results."})

        mr = results[0]
        self._assessments[nct_id] = mr
        return json.dumps({
            "nct_id": nct_id,
            "overall_score": mr.overall_score,
            "hard_exclusion": mr.hard_exclusion,
            "exclusion_reason": mr.exclusion_reason,
            "met_criteria": mr.met_criteria,
            "failed_criteria": mr.failed_criteria,
            "uncertain_criteria": mr.uncertain_criteria,
            "reasoning": mr.reasoning,
        })

    def _critic(self, nct_id: str) -> str:
        if self._profile is None:
            return json.dumps({"error": "Call extract_patient_profile first."})
        trial = self._trial_by_id.get(nct_id)
        assessment = self._assessments.get(nct_id)
        if trial is None or assessment is None:
            return json.dumps({"error": f"Call evaluate_eligibility for {nct_id} first."})

        verdict = critic_review(self._profile, trial, assessment)
        resolved = resolve_discrepancies(assessment, verdict, topic_id="live", nct_id=nct_id)
        self._assessments[nct_id] = resolved

        return json.dumps({
            "nct_id": nct_id,
            "agree": verdict.get("agree", True),
            "recommendation": verdict.get("recommendation", "accept_agent1"),
            "critic_reasoning": verdict.get("reasoning", ""),
            "final_score": resolved.overall_score,
            "critic_override": resolved.critic_override,
            "critic_flagged": resolved.critic_flagged,
        })

    def _explain(self, nct_id: str) -> str:
        assessment = self._assessments.get(nct_id)
        if assessment is None:
            return json.dumps({"error": f"Call evaluate_eligibility for {nct_id} first."})
        try:
            cards = generate_all_cards([assessment])
            card = cards[0] if cards else {}
            return json.dumps({
                "nct_id": nct_id,
                "explanation": card.get("card_text", ""),
                "fk_grade": card.get("fk_grade"),
                "trial_url": f"https://clinicaltrials.gov/study/{nct_id}",
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})
