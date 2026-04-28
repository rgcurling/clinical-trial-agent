"""
Stage 3 — Per-Trial Matching

Single Claude API call per trial returns structured JSON with overall score,
met/failed/uncertain criteria, and exclusion assessment.
"""

import json
import logging
import os
import re
import time
from dataclasses import replace
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, BASELINE_MODEL, MAX_TRIALS_TO_MATCH, PRIMARY_MODEL
from pipeline.models import (
    MatchResult,
    PatientProfile,
    Trial,
)

logger = logging.getLogger(__name__)

# ── Criterion parsing ─────────────────────────────────────────────────────────

_INCLUSION_RE = re.compile(r"inclusion criteria\s*:?", re.IGNORECASE)
_EXCLUSION_RE = re.compile(r"exclusion criteria\s*:?", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[\s*\-•]+", re.MULTILINE)


def _split_criteria_text(raw: str) -> tuple[str, str]:
    """Return (inclusion_block, exclusion_block) as raw strings."""
    inc_match = _INCLUSION_RE.search(raw)
    exc_match = _EXCLUSION_RE.search(raw)

    if inc_match and exc_match:
        inc_start = inc_match.end()
        exc_start = exc_match.start()
        exc_end = exc_match.end()
        return raw[inc_start:exc_start].strip(), raw[exc_end:].strip()
    elif inc_match:
        return raw[inc_match.end():].strip(), ""
    elif exc_match:
        return "", raw[exc_match.end():].strip()
    return raw.strip(), ""


def _extract_bullets(block: str) -> list[str]:
    """Split a criteria block into individual criterion strings."""
    lines = block.splitlines()
    criteria = []
    for line in lines:
        cleaned = _BULLET_RE.sub("", line).strip()
        if cleaned:
            criteria.append(cleaned)
    return criteria


def _parse_criteria_via_llm(raw: str) -> dict[str, list[str]]:
    """LLM fallback when regex yields < 2 total criteria."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = (
        "Parse the following clinical trial eligibility criteria text into "
        "inclusion and exclusion lists.\n"
        "Return ONLY valid JSON with this structure and no other text:\n"
        '{"inclusion": ["...", "..."], "exclusion": ["...", "..."]}\n\n'
        f"Eligibility text:\n{raw}"
    )
    msg = client.messages.create(
        model=PRIMARY_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_json = msg.content[0].text.strip()
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("LLM criteria parse returned malformed JSON; returning empty")
        return {"inclusion": [], "exclusion": []}


def parse_criteria(raw_text: str) -> dict[str, list[str]]:
    """
    Parse raw eligibility text into inclusion / exclusion criterion lists.
    Falls back to an LLM call if regex yields < 2 total criteria.
    """
    inc_block, exc_block = _split_criteria_text(raw_text)
    inclusion = _extract_bullets(inc_block)
    exclusion = _extract_bullets(exc_block)

    if len(inclusion) + len(exclusion) < 2:
        logger.info("Regex parse yielded < 2 criteria; using LLM fallback")
        return _parse_criteria_via_llm(raw_text)

    return {"inclusion": inclusion, "exclusion": exclusion}


# ── Matching prompt ───────────────────────────────────────────────────────────

_MATCH_PROMPT = """\
You are a clinical trial eligibility assessor. Given a patient profile and a trial's eligibility criteria, determine how well the patient matches the trial.

Patient Profile:
{patient_profile_text}

Inclusion Criteria:
{inclusion_criteria}

Exclusion Criteria:
{exclusion_criteria}

Assess the patient against these criteria and return ONLY valid JSON with no other text:
{{
  "overall_score": <float 0.0-1.0, fraction of inclusion criteria the patient meets>,
  "met_criteria": ["<inclusion criterion the patient clearly meets>", ...],
  "failed_criteria": ["<inclusion criterion the patient clearly does not meet>", ...],
  "uncertain_criteria": ["<criterion where patient info is missing or ambiguous>", ...],
  "hard_exclusion": <true if any exclusion criterion is clearly triggered, else false>,
  "exclusion_reason": "<which exclusion criterion was triggered, or null>",
  "reasoning": "<one short paragraph explaining the overall assessment>"
}}

Rules:
- overall_score reflects the fraction of inclusion criteria met (0.0 if none are met)
- Set hard_exclusion to true only when the patient clearly meets an exclusion criterion
- When information is missing, classify the criterion as uncertain, not failed
- If hard_exclusion is true, set overall_score to 0.0"""


def _patient_profile_to_text(profile: PatientProfile) -> str:
    parts = [f"Raw description: {profile.raw_text}"]
    if profile.age is not None:
        parts.append(f"Age: {profile.age}")
    if profile.conditions:
        parts.append(f"Conditions: {', '.join(profile.conditions)}")
    if profile.stage:
        parts.append(f"Stage: {profile.stage}")
    if profile.prior_treatments:
        parts.append(f"Prior treatments: {', '.join(profile.prior_treatments)}")
    if profile.biomarkers:
        parts.append(f"Biomarkers: {', '.join(profile.biomarkers)}")
    if profile.location:
        parts.append(f"Location: {profile.location}")
    if profile.exclusion_flags:
        parts.append(f"Exclusion flags: {', '.join(profile.exclusion_flags)}")
    return "\n".join(parts)


def _format_criteria_list(criteria: list[str]) -> str:
    if not criteria:
        return "  (none)"
    return "\n".join(f"  - {c}" for c in criteria)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present, otherwise return text as-is."""
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _safe_default() -> dict:
    return {
        "overall_score": 0.0,
        "met_criteria": [],
        "failed_criteria": [],
        "uncertain_criteria": [],
        "hard_exclusion": False,
        "exclusion_reason": None,
        "reasoning": "Could not parse Claude response.",
    }


def _call_claude_for_trial(
    client: anthropic.Anthropic,
    patient_text: str,
    inclusion_criteria: list[str],
    exclusion_criteria: list[str],
) -> dict:
    prompt = _MATCH_PROMPT.format(
        patient_profile_text=patient_text,
        inclusion_criteria=_format_criteria_list(inclusion_criteria),
        exclusion_criteria=_format_criteria_list(exclusion_criteria),
    )
    msg = client.messages.create(
        model=PRIMARY_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _strip_fences(msg.content[0].text.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude returned malformed JSON for trial; attempting repair")
        repair_prompt = (
            "The following should be valid JSON but is not. "
            "Return ONLY the corrected JSON with no other text:\n\n" + raw
        )
        repair_msg = client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": repair_prompt}],
        )
        repaired = _strip_fences(repair_msg.content[0].text.strip())
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            logger.error("Repair also failed; using safe default for this trial")
            return _safe_default()


# ── Score computation ─────────────────────────────────────────────────────────

def compute_match_score(overall_score: float, hard_exclusion: bool) -> float:
    return 0.0 if hard_exclusion else overall_score


# ── Claude matcher (primary) ──────────────────────────────────────────────────

class ClaudeMatcher:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def match_trial(self, profile: PatientProfile, trial: Trial) -> MatchResult:
        criteria = parse_criteria(trial.eligibility_criteria_raw)
        patient_text = _patient_profile_to_text(profile)

        result_dict = _call_claude_for_trial(
            self.client,
            patient_text,
            criteria.get("inclusion", []),
            criteria.get("exclusion", []),
        )

        overall_score = float(result_dict.get("overall_score", 0.0))
        hard_exclusion = bool(result_dict.get("hard_exclusion", False))
        met_criteria = result_dict.get("met_criteria", [])
        failed_criteria = result_dict.get("failed_criteria", [])
        uncertain_criteria = result_dict.get("uncertain_criteria", [])
        exclusion_reason = result_dict.get("exclusion_reason") or None
        reasoning = result_dict.get("reasoning", "")

        score = compute_match_score(overall_score, hard_exclusion)

        return MatchResult(
            trial=trial,
            overall_score=overall_score,
            met_criteria=met_criteria,
            failed_criteria=failed_criteria,
            uncertain_criteria=uncertain_criteria,
            hard_exclusion=hard_exclusion,
            exclusion_reason=exclusion_reason,
            reasoning=reasoning,
            match_score=score,
            uncertain_count=len(uncertain_criteria),
        )

    def match_trials(
        self, profile: PatientProfile, trials: list[Trial]
    ) -> list[MatchResult]:
        results = []
        for trial in trials[:MAX_TRIALS_TO_MATCH]:
            logger.info(f"Matching trial {trial.nct_id}: {trial.title[:60]}")
            results.append(self.match_trial(profile, trial))
        return results


# ── Critic agent (Part 2) ─────────────────────────────────────────────────────

def call_gpt4(prompt: str, temperature: float = 0.0) -> str:
    """
    Call OpenAI GPT-4o API with the same simple interface as Claude helpers.
    Requires OPENAI_API_KEY in the environment.
    """
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=BASELINE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


_CRITIC_PROMPT = """\
You are an independent clinical trial eligibility reviewer (Agent 2).

Patient Profile:
{patient_profile_text}

Trial Eligibility Criteria:
{eligibility_criteria}

First Reviewer's Assessment (Agent 1):
- Overall score: {agent1_score:.2f}
- Met criteria: {met_criteria}
- Hard exclusions: {hard_exclusion}

Your Task:
Independently assess this patient's eligibility for this trial WITHOUT being biased by Agent 1's reasoning.

Output JSON format:
{{
  "agree": true/false,
  "your_score": 0.0-1.0,
  "discrepancies": ["criterion X differs because...", "criterion Y..."],
  "recommendation": "accept_agent1" | "override" | "flag_uncertain"
}}

Guidelines:
- "accept_agent1": You agree, use Agent 1's assessment
- "override": You strongly disagree (2 or more criteria differ), use your score instead
- "flag_uncertain": 1 criterion differs or the case is borderline, flag for human review
"""


def _match_result_to_agent_summary(agent1_output) -> dict:
    if isinstance(agent1_output, MatchResult):
        return {
            "overall_score": agent1_output.overall_score,
            "met_criteria": agent1_output.met_criteria,
            "hard_exclusion": agent1_output.hard_exclusion,
        }
    return {
        "overall_score": float(agent1_output.get("overall_score", 0.0)),
        "met_criteria": agent1_output.get("met_criteria", []),
        "hard_exclusion": agent1_output.get("hard_exclusion", []),
    }


def _coerce_patient_text(patient_profile) -> str:
    if isinstance(patient_profile, PatientProfile):
        return _patient_profile_to_text(patient_profile)
    return str(patient_profile)


def _coerce_trial_criteria(trial_data) -> str:
    if isinstance(trial_data, Trial):
        return trial_data.eligibility_criteria_raw or ""
    return (
        trial_data.get("eligibility_criteria")
        or trial_data.get("eligibility_criteria_raw")
        or trial_data.get("text")
        or ""
    )


def _parse_json_response(raw: str) -> dict:
    raw = _strip_fences(raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if repaired:
            return json.loads(repaired.group(0))
        raise


def critic_review(
    patient_profile,
    trial_data,
    agent1_output,
) -> dict:
    """
    Independent eligibility review by GPT-4o (Agent 2).

    Agent 2 sees patient profile, full eligibility criteria, and Agent 1's
    summary only: score, met criteria, and hard exclusions. It does not see
    Agent 1's detailed reasoning.

    Returns dict: {agree, your_score, discrepancies, recommendation}.
    """
    agent1_summary = _match_result_to_agent_summary(agent1_output)

    prompt = _CRITIC_PROMPT.format(
        patient_profile_text=_coerce_patient_text(patient_profile),
        eligibility_criteria=_coerce_trial_criteria(trial_data),
        agent1_score=agent1_summary["overall_score"],
        met_criteria=agent1_summary.get("met_criteria", []),
        hard_exclusion=agent1_summary.get("hard_exclusion", []),
    )

    for attempt in range(3):
        try:
            raw = call_gpt4(prompt, temperature=0.0)
            parsed = _parse_json_response(raw)
            return {
                "agree": bool(parsed.get("agree", False)),
                "your_score": float(parsed.get("your_score", agent1_summary["overall_score"])),
                "discrepancies": parsed.get("discrepancies", []) or [],
                "recommendation": parsed.get("recommendation", "flag_uncertain"),
            }
        except Exception as exc:
            logger.warning(f"Critic (GPT-4o) attempt {attempt + 1}/3 failed: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    logger.error("All critic retry attempts failed; defaulting to accept_agent1")
    return {
        "agree": True,
        "your_score": agent1_summary["overall_score"],
        "discrepancies": [],
        "recommendation": "accept_agent1",
    }


def resolve_discrepancies(
    agent1_output: MatchResult,
    agent2_output: dict,
    topic_id: Optional[str] = None,
    nct_id: Optional[str] = None,
) -> MatchResult:
    """
    Merge Agent 1 and Agent 2 assessments per the critic protocol:
      - agree=True                    → return Agent 1 unchanged
      - recommendation=override       → use Agent 2 score, set critic_override=True
      - recommendation=flag_uncertain → keep Agent 1 score, set critic_flagged=True
    Logs ALL disagreements to results/critic_disagreements.jsonl.
    """
    agree = agent2_output.get("agree", True)
    discrepancies = agent2_output.get("discrepancies", [])
    recommendation = agent2_output.get("recommendation", "accept_agent1")
    agent2_score = float(agent2_output.get("your_score", agent1_output.overall_score))

    if agree:
        return agent1_output

    resolution = "accept"
    resolved = agent1_output

    if recommendation == "override":
        resolution = "override"
        score = compute_match_score(agent2_score, agent1_output.hard_exclusion)
        resolved = replace(
            agent1_output,
            overall_score=agent2_score,
            match_score=score,
            critic_override=True,
        )
    elif recommendation == "flag_uncertain":
        resolution = "flag"
        resolved = replace(
            agent1_output,
            critic_flagged=True,
            uncertain=True,
            uncertainty_reason=discrepancies,
        )

    os.makedirs("results", exist_ok=True)
    with open("results/critic_disagreements.jsonl", "a") as f:
        f.write(json.dumps({
            "topic_id": topic_id,
            "nct_id": nct_id or agent1_output.trial.nct_id,
            "agent1_score": agent1_output.overall_score,
            "agent2_score": agent2_score,
            "agreed": bool(agree),
            "discrepancies": discrepancies,
            "resolution": resolution,
        }) + "\n")

    return resolved
