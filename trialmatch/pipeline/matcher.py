"""
Stage 3 — Criterion Parsing + Per-Criterion Matching

Each criterion gets its own Claude API call (never batched).
Architecture follows Wornow et al. (2025) and Jin et al. (2024).
"""

import json
import logging
import re
from typing import Optional

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    EXCLUSION_CONFIDENCE_THRESHOLD,
    MAX_TRIALS_TO_MATCH,
    OPENAI_API_KEY,
    PRIMARY_MODEL,
    BASELINE_MODEL,
)
from pipeline.models import (
    CriterionResult,
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
You are a clinical trial eligibility assessor. Your job is to determine whether a specific patient meets a single eligibility criterion.

Patient Profile:
{patient_profile_text}

Criterion Type: {inclusion_or_exclusion}
Criterion: "{criterion_text}"

Does this patient meet this criterion based on the information provided?

Rules:
- If the patient profile clearly satisfies the criterion, set eligible to "true"
- If the patient profile clearly does not satisfy the criterion, set eligible to "false"
- If the patient profile does not contain enough information to assess this criterion, set eligible to "uncertain"
- Be conservative: when in doubt, prefer "uncertain" over "false"

Respond ONLY with valid JSON, no other text:
{{
  "eligible": "true" | "false" | "uncertain",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence citing specific patient information>",
  "relevant_patient_info": "<exact text from patient profile that informed this, or 'not mentioned' if absent>"
}}"""


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


def _call_claude_for_criterion(
    client: anthropic.Anthropic,
    patient_text: str,
    criterion_type: str,
    criterion_text: str,
) -> dict:
    prompt = _MATCH_PROMPT.format(
        patient_profile_text=patient_text,
        inclusion_or_exclusion=criterion_type,
        criterion_text=criterion_text,
    )
    msg = client.messages.create(
        model=PRIMARY_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude returned malformed JSON for criterion; attempting repair")
        repair_prompt = (
            "The following should be valid JSON but is not. "
            "Return ONLY the corrected JSON with no other text:\n\n" + raw
        )
        repair_msg = client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": repair_prompt}],
        )
        return json.loads(repair_msg.content[0].text.strip())


# ── Score computation ─────────────────────────────────────────────────────────

def compute_match_score(criterion_results: list[CriterionResult]) -> float:
    inclusion = [r for r in criterion_results if r.criterion_type == "inclusion"]
    exclusion = [r for r in criterion_results if r.criterion_type == "exclusion"]

    if not inclusion:
        return 0.0

    met = sum(1 for r in inclusion if r.eligible == "true")
    triggered_exclusion = any(
        r.eligible == "false" and r.confidence > EXCLUSION_CONFIDENCE_THRESHOLD
        for r in exclusion
    )

    base_score = met / len(inclusion)
    return 0.0 if triggered_exclusion else base_score


# ── Claude matcher (primary) ──────────────────────────────────────────────────

class ClaudeMatcher:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def match_trial(self, profile: PatientProfile, trial: Trial) -> MatchResult:
        criteria = parse_criteria(trial.eligibility_criteria_raw)
        patient_text = _patient_profile_to_text(profile)
        criterion_results: list[CriterionResult] = []

        for ctype in ("inclusion", "exclusion"):
            for ctext in criteria.get(ctype, []):
                result_dict = _call_claude_for_criterion(
                    self.client, patient_text, ctype, ctext
                )
                criterion_results.append(
                    CriterionResult(
                        criterion_text=ctext,
                        criterion_type=ctype,
                        eligible=result_dict.get("eligible", "uncertain"),
                        confidence=float(result_dict.get("confidence", 0.5)),
                        reasoning=result_dict.get("reasoning", ""),
                        relevant_patient_info=result_dict.get(
                            "relevant_patient_info", "not mentioned"
                        ),
                    )
                )

        score = compute_match_score(criterion_results)

        inc_results = [r for r in criterion_results if r.criterion_type == "inclusion"]
        exc_results = [r for r in criterion_results if r.criterion_type == "exclusion"]

        return MatchResult(
            trial=trial,
            criterion_results=criterion_results,
            match_score=score,
            met_inclusion=sum(1 for r in inc_results if r.eligible == "true"),
            failed_inclusion=sum(1 for r in inc_results if r.eligible == "false"),
            triggered_exclusion=sum(
                1
                for r in exc_results
                if r.eligible == "false"
                and r.confidence > EXCLUSION_CONFIDENCE_THRESHOLD
            ),
            uncertain_count=sum(1 for r in criterion_results if r.eligible == "uncertain"),
        )

    def match_trials(
        self, profile: PatientProfile, trials: list[Trial]
    ) -> list[MatchResult]:
        results = []
        for trial in trials[:MAX_TRIALS_TO_MATCH]:
            logger.info(f"Matching trial {trial.nct_id}: {trial.title[:60]}")
            results.append(self.match_trial(profile, trial))
        return results


# ── GPT-4o baseline matcher ───────────────────────────────────────────────────

class GPT4oMatcher:
    """Baseline comparison matcher using OpenAI GPT-4o. Requires OPENAI_API_KEY."""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use GPT4oMatcher")
        import openai
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def _call_for_criterion(
        self,
        patient_text: str,
        criterion_type: str,
        criterion_text: str,
    ) -> dict:
        prompt = _MATCH_PROMPT.format(
            patient_profile_text=patient_text,
            inclusion_or_exclusion=criterion_type,
            criterion_text=criterion_text,
        )
        response = self.client.chat.completions.create(
            model=BASELINE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("GPT-4o returned malformed JSON for criterion")
            return {
                "eligible": "uncertain",
                "confidence": 0.5,
                "reasoning": "Parse error",
                "relevant_patient_info": "not mentioned",
            }

    def match_trial(self, profile: PatientProfile, trial: Trial) -> MatchResult:
        criteria = parse_criteria(trial.eligibility_criteria_raw)
        patient_text = _patient_profile_to_text(profile)
        criterion_results: list[CriterionResult] = []

        for ctype in ("inclusion", "exclusion"):
            for ctext in criteria.get(ctype, []):
                result_dict = self._call_for_criterion(patient_text, ctype, ctext)
                criterion_results.append(
                    CriterionResult(
                        criterion_text=ctext,
                        criterion_type=ctype,
                        eligible=result_dict.get("eligible", "uncertain"),
                        confidence=float(result_dict.get("confidence", 0.5)),
                        reasoning=result_dict.get("reasoning", ""),
                        relevant_patient_info=result_dict.get(
                            "relevant_patient_info", "not mentioned"
                        ),
                    )
                )

        score = compute_match_score(criterion_results)
        inc_results = [r for r in criterion_results if r.criterion_type == "inclusion"]
        exc_results = [r for r in criterion_results if r.criterion_type == "exclusion"]

        return MatchResult(
            trial=trial,
            criterion_results=criterion_results,
            match_score=score,
            met_inclusion=sum(1 for r in inc_results if r.eligible == "true"),
            failed_inclusion=sum(1 for r in inc_results if r.eligible == "false"),
            triggered_exclusion=sum(
                1
                for r in exc_results
                if r.eligible == "false"
                and r.confidence > EXCLUSION_CONFIDENCE_THRESHOLD
            ),
            uncertain_count=sum(1 for r in criterion_results if r.eligible == "uncertain"),
        )

    def match_trials(
        self, profile: PatientProfile, trials: list[Trial]
    ) -> list[MatchResult]:
        results = []
        for trial in trials[:MAX_TRIALS_TO_MATCH]:
            logger.info(f"[GPT-4o] Matching trial {trial.nct_id}")
            results.append(self.match_trial(profile, trial))
        return results
