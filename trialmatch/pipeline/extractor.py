"""
Stage 1 — Entity Extraction

Primary: Claude LLM structured extraction
Baseline: RegexExtractor (age + location only, no API calls)
"""

import json
import logging
import re
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, PRIMARY_MODEL
from pipeline.models import PatientProfile

logger = logging.getLogger(__name__)

# ── Regex helpers ─────────────────────────────────────────────────────────────

_AGE_RE = re.compile(r"(\d+)[-\s]year[-\s]old", re.IGNORECASE)
_LOCATION_RE = re.compile(r"\b([A-Z][a-zA-Z\s]+),\s*([A-Z]{2})\b")

# ── LLM extraction ────────────────────────────────────────────────────────────

_LLM_EXTRACTION_PROMPT = """\
Extract structured medical information from this patient description.
Return ONLY valid JSON with these exact fields and no other text:
{{
  "conditions": [],
  "stage": null,
  "prior_treatments": [],
  "biomarkers": [],
  "age": null,
  "location": null,
  "exclusion_flags": []
}}

Rules:
- "conditions": list of diagnosed medical conditions (e.g. ["non-small cell lung cancer"])
- "stage": disease stage as a string if mentioned (e.g. "Stage IIIB"), otherwise null
- "prior_treatments": list of prior medications or therapies mentioned
- "biomarkers": list of biomarker results mentioned (e.g. ["EGFR negative", "PD-L1 positive"])
- "age": integer age if mentioned, otherwise null
- "location": "City, ST" format if a US city/state is mentioned, otherwise null
- "exclusion_flags": any self-reported contraindications or disqualifying conditions

Patient text: {text}"""


def _extract_via_llm(text: str) -> dict:
    """Call Claude to extract structured patient attributes from free text."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = _LLM_EXTRACTION_PROMPT.format(text=text)

    message = client.messages.create(
        model=PRIMARY_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned malformed JSON; requesting repair")
        repair_prompt = (
            "The following text should be valid JSON but is not. "
            "Return ONLY the corrected JSON with no other text:\n\n" + raw
        )
        repair_msg = client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": repair_prompt}],
        )
        return json.loads(repair_msg.content[0].text.strip())


# ── Public extractor ──────────────────────────────────────────────────────────

def extract_patient_profile(text: str) -> PatientProfile:
    """
    Extract a PatientProfile from free text using Claude.
    """
    data = _extract_via_llm(text)
    logger.info("Extraction completed via LLM")

    return PatientProfile(
        raw_text=text,
        conditions=data.get("conditions") or [],
        stage=data.get("stage"),
        prior_treatments=data.get("prior_treatments") or [],
        biomarkers=data.get("biomarkers") or [],
        age=data.get("age"),
        location=data.get("location"),
        exclusion_flags=data.get("exclusion_flags") or [],
    )


# ── Regex-only baseline ───────────────────────────────────────────────────────

class RegexExtractor:
    """
    Lightweight baseline extractor: age and location via regex only.
    No API calls. Used in benchmarking to compare against LLM extraction.
    """

    def extract(self, text: str) -> PatientProfile:
        age_match = _AGE_RE.search(text)
        age = int(age_match.group(1)) if age_match else None

        loc_match = _LOCATION_RE.search(text)
        location = (
            f"{loc_match.group(1).strip()}, {loc_match.group(2)}"
            if loc_match
            else None
        )

        return PatientProfile(
            raw_text=text,
            age=age,
            location=location,
        )
