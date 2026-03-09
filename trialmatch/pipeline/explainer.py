"""
Stage 5 — Plain-English Output

Generates patient-facing trial cards for each ranked result using Claude.
Measures Flesch-Kincaid grade and simplifies automatically if grade > 8.
"""

import logging

import anthropic
import textstat

from config import ANTHROPIC_API_KEY, PRIMARY_MODEL, TARGET_FK_GRADE
from pipeline.models import CriterionResult, MatchResult

logger = logging.getLogger(__name__)

_EXPLAIN_PROMPT = """\
You are writing for a patient with no medical background. Explain this clinical trial match result in plain English.

Trial Title: {title}
Match Score: {score}/10
NCT ID: {nct_id}

Criteria the patient likely meets:
{met_criteria_list}

Criteria that are uncertain (missing information):
{uncertain_criteria_list}

Write a patient-facing summary card with these exact sections:
TRIAL NAME: [simplified title, max 12 words]
MATCH SCORE: [{score}/10]
WHY YOU MAY QUALIFY:
  - [each met criterion in plain English, no jargon, max 20 words each]
THINGS TO CLARIFY WITH YOUR DOCTOR:
  - [each uncertain criterion rephrased as a question the patient can ask]
LEARN MORE:
  https://clinicaltrials.gov/study/{nct_id}

Rules:
- Use simple language. Target a grade 8 reading level.
- Never use acronyms without defining them.
- Never speculate beyond what the match data shows.
- If match score is below 0.4, add a note that this trial may not be the best fit."""

_SIMPLIFY_PROMPT = """\
The following patient-facing text has a Flesch-Kincaid grade level of {fk_grade:.1f}, \
which is above the target of grade 8. Please rewrite it to be simpler and easier to read \
while keeping all the same information. Use shorter sentences and simpler words.

Text to simplify:
{text}

Return ONLY the simplified text with no other commentary."""


def _format_criteria_list(criteria: list[CriterionResult]) -> str:
    if not criteria:
        return "  (none)"
    return "\n".join(f"  - {c.criterion_text}" for c in criteria)


def generate_trial_card(match: MatchResult) -> str:
    """
    Generate a plain-English patient card for a single MatchResult.
    Automatically simplifies if FK grade > TARGET_FK_GRADE.
    Returns the final card text.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    met = [r for r in match.criterion_results if r.criterion_type == "inclusion" and r.eligible == "true"]
    uncertain = [r for r in match.criterion_results if r.eligible == "uncertain"]

    score_display = round(match.match_score * 10, 1)

    prompt = _EXPLAIN_PROMPT.format(
        title=match.trial.title,
        score=score_display,
        nct_id=match.trial.nct_id,
        met_criteria_list=_format_criteria_list(met),
        uncertain_criteria_list=_format_criteria_list(uncertain),
    )

    msg = client.messages.create(
        model=PRIMARY_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    card_text = msg.content[0].text.strip()

    fk_grade = textstat.flesch_kincaid_grade(card_text)
    logger.info(
        f"[{match.trial.nct_id}] Initial FK grade: {fk_grade:.1f} "
        f"(target: {TARGET_FK_GRADE})"
    )

    if fk_grade > TARGET_FK_GRADE:
        logger.info(f"[{match.trial.nct_id}] Simplifying (FK {fk_grade:.1f} > {TARGET_FK_GRADE})")
        simplify_prompt = _SIMPLIFY_PROMPT.format(fk_grade=fk_grade, text=card_text)
        simplify_msg = client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": simplify_prompt}],
        )
        card_text = simplify_msg.content[0].text.strip()
        final_fk = textstat.flesch_kincaid_grade(card_text)
        logger.info(f"[{match.trial.nct_id}] Final FK grade after simplification: {final_fk:.1f}")
    else:
        logger.info(f"[{match.trial.nct_id}] FK grade within target — no simplification needed")

    return card_text


def generate_all_cards(match_results: list[MatchResult]) -> list[dict]:
    """
    Generate patient cards for all ranked results.
    Returns a list of dicts with nct_id, match_score, card_text, and fk_grade.
    """
    output = []
    for match in match_results:
        card_text = generate_trial_card(match)
        fk_grade = textstat.flesch_kincaid_grade(card_text)
        output.append(
            {
                "nct_id": match.trial.nct_id,
                "match_score": match.match_score,
                "card_text": card_text,
                "fk_grade": fk_grade,
            }
        )
    return output
