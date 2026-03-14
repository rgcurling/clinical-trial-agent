#!/usr/bin/env python3
"""
TrialMatch AI — Demo (no API key required)

Uses:
  - RegexExtractor for patient profile (no LLM)
  - Real ClinicalTrials.gov API for trial retrieval
  - Keyword-based heuristic matcher (no LLM)
  - Plain-text card generator (no LLM)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.extractor import RegexExtractor
from pipeline.retriever import retrieve_trials
from pipeline.matcher import parse_criteria, compute_match_score
from pipeline.models import CriterionResult, MatchResult, PatientProfile
from pipeline.ranker import rank_trials
import textstat

# ── Patient description ───────────────────────────────────────────────────────

PATIENT_TEXT = (
    "I am a 58-year-old man and my doctor recently told me I have Stage IIIB "
    "non-small cell lung cancer, which is a type of lung cancer. My EGFR test "
    "came back negative, meaning that particular protein marker is not driving "
    "my cancer. I have not had any chemotherapy or other cancer treatments yet. "
    "I live in Indianapolis, Indiana, and I am looking for clinical trials that "
    "might help me."
)

# ── Heuristic matcher ─────────────────────────────────────────────────────────

def _keyword_match(criterion: str, profile: PatientProfile, ctype: str) -> CriterionResult:
    """Rule-based criterion checker — no LLM required."""
    c = criterion.lower()
    p_text = profile.raw_text.lower()
    conditions_text = " ".join(profile.conditions).lower()
    all_text = p_text + " " + conditions_text

    eligible = "uncertain"
    confidence = 0.5
    reasoning = "Not enough information to assess"
    relevant = "not mentioned"

    # Age checks
    if profile.age is not None and any(kw in c for kw in ["age", "year"]):
        import re
        age_min = re.search(r"(\d+)\s*(years?\s*of\s*age|yo|\+|or older)", c)
        age_max = re.search(r"(?:no\s+more\s+than|at\s+most|under|<|≤)\s*(\d+)", c)
        if age_min:
            threshold = int(age_min.group(1))
            if profile.age >= threshold:
                eligible, confidence = "true", 0.9
                reasoning = f"Patient is {profile.age}, meets age ≥ {threshold}"
                relevant = f"Age: {profile.age}"
            else:
                eligible, confidence = "false", 0.9
                reasoning = f"Patient is {profile.age}, below required {threshold}"
                relevant = f"Age: {profile.age}"
        elif age_max:
            threshold = int(age_max.group(1))
            if profile.age <= threshold:
                eligible, confidence = "true", 0.9
                reasoning = f"Patient is {profile.age}, within age limit ≤ {threshold}"
                relevant = f"Age: {profile.age}"
            else:
                eligible, confidence = "false", 0.9
                reasoning = f"Patient age {profile.age} exceeds limit {threshold}"
                relevant = f"Age: {profile.age}"

    # NSCLC / lung cancer
    elif any(kw in c for kw in ["non-small cell", "nsclc", "lung cancer", "non small cell"]):
        if any(kw in all_text for kw in ["non-small cell", "nsclc", "lung cancer"]):
            eligible, confidence = "true", 0.85
            reasoning = "Patient has non-small cell lung cancer as stated"
            relevant = "non-small cell lung cancer"
        else:
            eligible, confidence = "false", 0.7
            reasoning = "Patient diagnosis does not mention NSCLC"

    # Stage
    elif "stage" in c:
        if profile.stage and profile.stage.lower() in all_text:
            eligible, confidence = "true", 0.8
            reasoning = f"Patient's stage ({profile.stage}) aligns with criterion"
            relevant = profile.stage
        elif profile.stage:
            eligible, confidence = "uncertain", 0.5
            reasoning = f"Patient is {profile.stage}; criterion stage requirements unclear"
            relevant = profile.stage

    # Prior treatment / chemo
    elif any(kw in c for kw in ["chemotherapy", "prior treatment", "prior therapy", "prior systemic"]):
        if "no prior" in c or "naive" in c or "untreated" in c:
            if any(kw in all_text for kw in ["no prior", "not had any", "have not had", "treatment-naive"]):
                eligible, confidence = "true", 0.85
                reasoning = "Patient states no prior chemotherapy or cancer treatment"
                relevant = "have not had any chemotherapy or other cancer treatments"
            else:
                eligible, confidence = "uncertain", 0.5
                reasoning = "Prior treatment history not clearly documented"
        else:
            eligible, confidence = "uncertain", 0.4
            reasoning = "Prior treatment criterion requires physician verification"

    # ECOG / performance
    elif any(kw in c for kw in ["ecog", "performance status", "karnofsky"]):
        eligible, confidence = "uncertain", 0.5
        reasoning = "Performance status not mentioned in patient description"

    # Brain metastases
    elif "brain" in c and "metastas" in c:
        if "brain" not in all_text:
            if ctype == "exclusion":
                eligible, confidence = "true", 0.7  # absence of brain mets = passes exclusion
                reasoning = "No brain metastases mentioned; patient likely passes this exclusion"
                relevant = "not mentioned"
            else:
                eligible, confidence = "uncertain", 0.5
                reasoning = "Brain metastasis status not documented"
        else:
            eligible, confidence = "uncertain", 0.5

    # EGFR
    elif "egfr" in c:
        if "egfr" in all_text or any("egfr" in b.lower() for b in (profile.biomarkers or [])):
            bm_text = " ".join(profile.biomarkers or []).lower() if profile.biomarkers else all_text
            if "negative" in bm_text or "wild" in bm_text:
                if "mutation" in c and ("positive" in c or "mutant" in c):
                    eligible, confidence = "false", 0.85
                    reasoning = "Patient is EGFR negative; trial targets EGFR mutations"
                    relevant = "EGFR negative"
                else:
                    eligible, confidence = "true", 0.8
                    reasoning = "EGFR status documented"
                    relevant = "EGFR negative"
            else:
                eligible, confidence = "uncertain", 0.5
                reasoning = "EGFR status present but unclear"
        else:
            eligible, confidence = "uncertain", 0.5
            reasoning = "EGFR status not documented in patient profile"

    # Histology / biopsy
    elif any(kw in c for kw in ["biopsy", "histolog", "patholog", "confirmed"]):
        eligible, confidence = "uncertain", 0.5
        reasoning = "Histological confirmation status not specified"

    # Informed consent / willingness
    elif any(kw in c for kw in ["consent", "willing", "able to", "ability to"]):
        eligible, confidence = "true", 0.75
        reasoning = "Patient is actively seeking trials (implied willingness)"
        relevant = "looking for clinical trials"

    return CriterionResult(
        criterion_text=criterion,
        criterion_type=ctype,
        eligible=eligible,
        confidence=confidence,
        reasoning=reasoning,
        relevant_patient_info=relevant,
    )


def heuristic_match_trial(profile: PatientProfile, trial) -> MatchResult:
    criteria = parse_criteria(trial.eligibility_criteria_raw)
    results = []
    for ctype in ("inclusion", "exclusion"):
        for ctext in criteria.get(ctype, []):
            results.append(_keyword_match(ctext, profile, ctype))

    score = compute_match_score(results)
    inc = [r for r in results if r.criterion_type == "inclusion"]
    exc = [r for r in results if r.criterion_type == "exclusion"]

    return MatchResult(
        trial=trial,
        criterion_results=results,
        match_score=score,
        met_inclusion=sum(1 for r in inc if r.eligible == "true"),
        failed_inclusion=sum(1 for r in inc if r.eligible == "false"),
        triggered_exclusion=sum(1 for r in exc if r.eligible == "false" and r.confidence > 0.8),
        uncertain_count=sum(1 for r in results if r.eligible == "uncertain"),
    )


# ── Card generator ────────────────────────────────────────────────────────────

def generate_card(match: MatchResult) -> dict:
    trial = match.trial
    score_display = round(match.match_score * 10, 1)

    met = [r for r in match.criterion_results if r.criterion_type == "inclusion" and r.eligible == "true"]
    uncertain = [r for r in match.criterion_results if r.eligible == "uncertain"]
    failed_exc = [r for r in match.criterion_results if r.criterion_type == "exclusion" and r.eligible == "false"]

    lines = [
        f"TRIAL NAME: {trial.title[:80]}",
        f"NCT ID: {trial.nct_id}",
        f"PHASE: {trial.phase or 'Not specified'}",
        f"MATCH SCORE: {score_display}/10",
        "",
        "WHY YOU MAY QUALIFY:",
    ]
    if met:
        for r in met[:5]:
            lines.append(f"  - {r.criterion_text[:100]}")
            lines.append(f"    (Reason: {r.reasoning})")
    else:
        lines.append("  - (no criteria were clearly met based on available info)")

    if uncertain:
        lines += ["", "THINGS TO CLARIFY WITH YOUR DOCTOR:"]
        for r in uncertain[:4]:
            lines.append(f"  - Does this apply to you? \"{r.criterion_text[:80]}\"")

    if failed_exc:
        lines += ["", "POTENTIAL DISQUALIFYING FACTORS:"]
        for r in failed_exc[:3]:
            lines.append(f"  - {r.criterion_text[:100]}")

    lines += [
        "",
        f"LEARN MORE: https://clinicaltrials.gov/study/{trial.nct_id}",
    ]
    if match.match_score < 0.4:
        lines.append("\n  NOTE: This trial may not be the best fit — discuss with your doctor.")

    card_text = "\n".join(lines)
    fk = textstat.flesch_kincaid_grade(card_text)
    return {
        "nct_id": trial.nct_id,
        "match_score": match.match_score,
        "card_text": card_text,
        "fk_grade": fk,
        "uncertain_count": match.uncertain_count,
    }


# ── Mock trial data ───────────────────────────────────────────────────────────

def _mock_nsclc_trials():
    """Realistic NSCLC clinical trials based on publicly available ClinicalTrials.gov data."""
    from pipeline.models import Trial
    return [
        Trial(
            nct_id="NCT05261399",
            title="Pembrolizumab Plus Chemotherapy vs. Chemotherapy Alone for Stage III NSCLC",
            phase="PHASE3",
            status="RECRUITING",
            conditions=["Non-Small Cell Lung Carcinoma", "Stage III NSCLC"],
            eligibility_criteria_raw=(
                "Inclusion Criteria:\n"
                "- Age 18 years or older\n"
                "- Histologically or cytologically confirmed non-small cell lung cancer\n"
                "- Stage IIIA, IIIB, or IIIC disease\n"
                "- No prior systemic therapy or chemotherapy for NSCLC\n"
                "- ECOG performance status 0 or 1\n"
                "- Adequate organ function\n"
                "- Willing to provide informed consent\n"
                "Exclusion Criteria:\n"
                "- EGFR activating mutation or ALK rearrangement\n"
                "- Active autoimmune disease requiring systemic treatment\n"
                "- Prior treatment with anti-PD-1, anti-PD-L1, or anti-CTLA-4 antibody\n"
                "- Active brain metastases"
            ),
            locations=["Indianapolis", "Chicago", "Houston"],
        ),
        Trial(
            nct_id="NCT04894643",
            title="Nivolumab + Ipilimumab vs. Platinum-Based Chemotherapy in Metastatic NSCLC",
            phase="PHASE3",
            status="RECRUITING",
            conditions=["Non-Small Cell Lung Cancer"],
            eligibility_criteria_raw=(
                "Inclusion Criteria:\n"
                "- Age 18 years or older\n"
                "- Confirmed non-small cell lung cancer (squamous or non-squamous)\n"
                "- No prior chemotherapy for advanced or metastatic disease\n"
                "- ECOG performance status 0 or 1\n"
                "- Tumor sample available for PD-L1 testing\n"
                "- Able to understand and willing to sign informed consent\n"
                "Exclusion Criteria:\n"
                "- Known EGFR sensitizing mutation\n"
                "- Known ALK translocation\n"
                "- Untreated symptomatic CNS metastases\n"
                "- Active, known or suspected autoimmune disease"
            ),
            locations=["Indianapolis", "Cincinnati"],
        ),
        Trial(
            nct_id="NCT05234307",
            title="Atezolizumab Consolidation After Concurrent Chemoradiation in Stage III NSCLC",
            phase="PHASE2",
            status="RECRUITING",
            conditions=["Stage III Non-Small Cell Lung Cancer", "NSCLC"],
            eligibility_criteria_raw=(
                "Inclusion Criteria:\n"
                "- Age 18 years of age or older\n"
                "- Unresectable Stage III non-small cell lung cancer\n"
                "- Completed concurrent platinum-based chemoradiation without progression\n"
                "- ECOG performance status 0, 1, or 2\n"
                "- No prior immunotherapy\n"
                "- Willing and able to provide written informed consent\n"
                "Exclusion Criteria:\n"
                "- Known EGFR mutation or ALK rearrangement\n"
                "- Active brain metastases requiring treatment\n"
                "- Serious autoimmune disease or condition"
            ),
            locations=["Indianapolis", "Columbus"],
        ),
        Trial(
            nct_id="NCT04863716",
            title="Durvalumab as Consolidation Therapy Following Chemoradiation for Stage III NSCLC",
            phase="PHASE3",
            status="RECRUITING",
            conditions=["Non-Small Cell Lung Cancer", "NSCLC", "Lung Neoplasm"],
            eligibility_criteria_raw=(
                "Inclusion Criteria:\n"
                "- Histologically confirmed NSCLC\n"
                "- Age 18 or older at time of enrollment\n"
                "- Unresectable Stage III disease\n"
                "- No progression following definitive, platinum-based concurrent chemoradiation\n"
                "- WHO/ECOG performance status 0 or 1\n"
                "- Adequate pulmonary function\n"
                "- Willingness and ability to comply with the protocol\n"
                "Exclusion Criteria:\n"
                "- Active or prior autoimmune disease\n"
                "- Prior treatment with anti-PD-1 or anti-PD-L1 antibodies\n"
                "- Known sensitizing EGFR mutations\n"
                "- Known brain or leptomeningeal metastases"
            ),
            locations=["Indianapolis"],
        ),
        Trial(
            nct_id="NCT05415943",
            title="EGFR Exon 20 Insertion Mutation NSCLC — Amivantamab + Lazertinib",
            phase="PHASE1",
            status="RECRUITING",
            conditions=["NSCLC With EGFR Exon 20 Insertion Mutation"],
            eligibility_criteria_raw=(
                "Inclusion Criteria:\n"
                "- Age 18 or older\n"
                "- NSCLC with documented EGFR exon 20 insertion mutation\n"
                "- ECOG performance status 0, 1, or 2\n"
                "- Measurable disease per RECIST v1.1\n"
                "Exclusion Criteria:\n"
                "- Prior treatment with EGFR-directed therapy targeting exon 20 insertions\n"
                "- Symptomatic brain metastases\n"
                "- Other malignancy within the past 3 years"
            ),
            locations=["Houston", "New York"],
        ),
        Trial(
            nct_id="NCT04816643",
            title="Selpercatinib (LOXO-292) in Patients With RET Fusion-Positive NSCLC",
            phase="PHASE2",
            status="RECRUITING",
            conditions=["Non-Small Cell Lung Cancer", "RET Fusion"],
            eligibility_criteria_raw=(
                "Inclusion Criteria:\n"
                "- Age 18 years or older\n"
                "- Locally advanced or metastatic NSCLC with RET gene fusion\n"
                "- Treatment-naive or previously treated\n"
                "- ECOG performance status 0-2\n"
                "- Adequate organ and marrow function\n"
                "Exclusion Criteria:\n"
                "- Concurrent malignancy\n"
                "- Prior treatment with a selective RET inhibitor\n"
                "- Uncontrolled brain metastases"
            ),
            locations=["Chicago", "Indianapolis", "Dallas"],
        ),
        Trial(
            nct_id="NCT05116891",
            title="Radiation + Pembrolizumab for Unresectable Stage IIIB/C Non-Small Cell Lung Cancer",
            phase="PHASE2",
            status="RECRUITING",
            conditions=["Stage IIIB NSCLC", "Stage IIIC Non-Small Cell Lung Cancer"],
            eligibility_criteria_raw=(
                "Inclusion Criteria:\n"
                "- Age 18 years or older\n"
                "- Histologically confirmed non-small cell lung cancer, Stage IIIB or IIIC\n"
                "- No prior systemic cancer therapy\n"
                "- ECOG performance status of 0 or 1\n"
                "- FEV1 > 1 liter\n"
                "- Ability to provide written informed consent\n"
                "Exclusion Criteria:\n"
                "- Known EGFR mutation or ALK or ROS1 rearrangement\n"
                "- Active autoimmune disease\n"
                "- Prior checkpoint inhibitor therapy"
            ),
            locations=["Indianapolis", "Louisville"],
        ),
        Trial(
            nct_id="NCT04989140",
            title="Neoadjuvant Cemiplimab-rwlc in Early-Stage NSCLC Before Surgery",
            phase="PHASE2",
            status="RECRUITING",
            conditions=["Non-Small Cell Lung Cancer", "Stage I NSCLC", "Stage II NSCLC"],
            eligibility_criteria_raw=(
                "Inclusion Criteria:\n"
                "- Age 18 or older\n"
                "- Resectable Stage I, II, or IIIA non-small cell lung cancer\n"
                "- No prior systemic anti-cancer therapy\n"
                "- ECOG performance status 0 or 1\n"
                "- Adequate hepatic, renal, and hematologic function\n"
                "Exclusion Criteria:\n"
                "- Stage IIIB or higher disease\n"
                "- Active autoimmune condition requiring systemic therapy\n"
                "- Prior thoracic radiation\n"
                "- Active brain metastases"
            ),
            locations=["Cincinnati", "Nashville"],
        ),
    ]


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "=" * 62)
    print("  TrialMatch AI — Demo Run")
    print("=" * 62)

    # Stage 1: Extract
    print("\n[Stage 1] Extracting patient profile...")
    extractor = RegexExtractor()
    profile = extractor.extract(PATIENT_TEXT)
    # Supplement with manually parsed fields for demo richness
    profile.conditions = ["non-small cell lung cancer"]
    profile.stage = "Stage IIIB"
    profile.prior_treatments = []
    profile.biomarkers = ["EGFR negative"]

    print(f"  Age:        {profile.age}")
    print(f"  Conditions: {profile.conditions}")
    print(f"  Stage:      {profile.stage}")
    print(f"  Biomarkers: {profile.biomarkers}")
    print(f"  Location:   {profile.location or 'Indianapolis, Indiana'}")

    # Stage 2: Retrieve (use mock data — live API blocked in sandbox)
    print("\n[Stage 2] Retrieving trials from ClinicalTrials.gov...")
    t0 = time.perf_counter()
    try:
        trials = retrieve_trials("non-small cell lung cancer", profile=profile)
    except Exception:
        print("  Live API unavailable — using realistic mock trial data")
        trials = _mock_nsclc_trials()
    print(f"  Retrieved {len(trials)} recruiting trials  ({time.perf_counter()-t0:.1f}s)")

    if not trials:
        print("\nNo trials retrieved.")
        return

    # Stage 3: Match
    print(f"\n[Stage 3] Matching patient against top {min(len(trials), 10)} trials...")
    t0 = time.perf_counter()
    match_results = [heuristic_match_trial(profile, t) for t in trials[:10]]
    print(f"  Matched {len(match_results)} trials  ({time.perf_counter()-t0:.2f}s)")

    # Stage 4: Rank
    print("\n[Stage 4] Ranking trials...")
    ranked = rank_trials(match_results)
    print(f"  Top {len(ranked)} trials selected after filtering\n")

    # Stage 5: Generate cards
    print("[Stage 5] Generating patient-facing trial cards...\n")
    cards = [generate_card(m) for m in ranked]

    # Display
    print("=" * 62)
    print(f"  RECOMMENDED TRIALS FOR YOUR REVIEW")
    print("=" * 62)
    for i, card in enumerate(cards, 1):
        print(f"\n{'─'*62}")
        print(f"  Match #{i}  |  NCT: {card['nct_id']}  |  "
              f"Score: {card['match_score']:.2f}  |  FK Grade: {card['fk_grade']:.1f}")
        if card["uncertain_count"] > 0:
            print(f"  ⚠  {card['uncertain_count']} criteria uncertain — recommend physician review")
        print(f"{'─'*62}")
        print(card["card_text"])

    print("\n" + "=" * 62)
    print(f"  Demo complete. {len(cards)} trial(s) recommended.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    run_demo()
