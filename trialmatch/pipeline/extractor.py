"""
Stage 1 — Entity Extraction

Primary: scispaCy en_core_sci_md biomedical NER
Fallback: Claude LLM extraction if scispaCy not installed
Baseline: RegexExtractor (age + location only)
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
_LOCATION_RE = re.compile(
    r"\b([A-Z][a-zA-Z\s]+),\s*([A-Z]{2})\b"
)

# ── scispaCy loader (optional) ────────────────────────────────────────────────

_NLP = None
_SPACY_AVAILABLE = False

def _load_spacy() -> bool:
    global _NLP, _SPACY_AVAILABLE
    if _SPACY_AVAILABLE:
        return True
    try:
        import spacy
        _NLP = spacy.load("en_core_sci_md")
        _SPACY_AVAILABLE = True
        logger.info("scispaCy en_core_sci_md loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"scispaCy unavailable ({e}); will use LLM fallback")
        return False


# ── LLM fallback ──────────────────────────────────────────────────────────────

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

Patient text: {text}"""


def _extract_via_llm(text: str) -> dict:
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
        # Repair: ask Claude to fix the JSON
        logger.warning("LLM returned malformed JSON; requesting repair")
        repair_prompt = (
            f"The following text should be valid JSON but is not. "
            f"Return ONLY the corrected JSON with no other text:\n\n{raw}"
        )
        repair_msg = client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": repair_prompt}],
        )
        return json.loads(repair_msg.content[0].text.strip())


# ── scispaCy-based extraction ─────────────────────────────────────────────────

def _extract_via_spacy(text: str) -> dict:
    doc = _NLP(text)

    conditions = []
    biomarkers = []
    treatments = []

    for ent in doc.ents:
        label = ent.label_.upper()
        ent_text = ent.text.strip()
        if label in ("DISEASE", "CANCER"):
            conditions.append(ent_text)
        elif label in ("CHEMICAL", "DRUG", "SIMPLE_CHEMICAL"):
            treatments.append(ent_text)
        elif label in ("GENE_OR_GENE_PRODUCT", "PROTEIN"):
            biomarkers.append(ent_text)

    # Regex supplements
    age_match = _AGE_RE.search(text)
    age = int(age_match.group(1)) if age_match else None

    loc_match = _LOCATION_RE.search(text)
    location = f"{loc_match.group(1).strip()}, {loc_match.group(2)}" if loc_match else None

    # Stage detection
    stage_match = re.search(r"\bStage\s+([IVX]{1,4}[ABC]?)\b", text, re.IGNORECASE)
    stage = stage_match.group(0) if stage_match else None

    return {
        "conditions": conditions,
        "stage": stage,
        "prior_treatments": treatments,
        "biomarkers": biomarkers,
        "age": age,
        "location": location,
        "exclusion_flags": [],
    }


# ── Public extractor ──────────────────────────────────────────────────────────

def extract_patient_profile(text: str) -> PatientProfile:
    """
    Extract a PatientProfile from free-text.
    Uses scispaCy if available, otherwise falls back to Claude.
    """
    if _load_spacy():
        try:
            data = _extract_via_spacy(text)
            logger.info("Extraction completed via scispaCy")
        except Exception as e:
            logger.warning(f"scispaCy extraction failed ({e}); falling back to LLM")
            data = _extract_via_llm(text)
    else:
        data = _extract_via_llm(text)
        logger.info("Extraction completed via LLM fallback")

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
    """Simple baseline: extracts only age and location via regex."""

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
