# TrialMatch AI

TrialMatch AI converts a free-text patient profile into a ranked list of recruiting clinical trials, each accompanied by a plain-English eligibility summary written at an 8th-grade reading level. It is designed to help patients and physicians quickly identify potentially relevant trials and understand eligibility requirements without deciphering complex medical language. TrialMatch AI is a research prototype and is **not** medical advice, nor is it a HIPAA-compliant system.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_sci_md   # optional; falls back to LLM extraction
cp .env.example .env                      # add ANTHROPIC_API_KEY
```

## Usage

```bash
# Single patient from text
python main.py --patient "58-year-old male with Stage IIIB NSCLC..."

# Single patient from file
python main.py --patient-file data/sample_patients/patient_01.txt --output results/p01.json

# Synthetic benchmark (FK grade + BERTScore across all sample patients)
python main.py --eval-synthetic

# TREC Clinical Trials 2021 benchmark (P@5, NDCG@5)
python main.py --benchmark

# Claude vs GPT-4o comparison
python main.py --patient-file data/sample_patients/patient_01.txt --compare-models
```

## Architecture

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | pipeline/extractor.py | Entity extraction from free text (scispaCy + LLM fallback) |
| 2 | pipeline/retriever.py | ClinicalTrials.gov API v2 with disk cache and retry |
| 3 | pipeline/matcher.py | Per-criterion eligibility matching via Claude |
| 4 | pipeline/ranker.py | Hard exclusion filter + score-based ranking |
| 5 | pipeline/explainer.py | FK-controlled plain-English patient cards |

## Evaluation

- **Readability:** Flesch-Kincaid grade ≤ 8 (auto-simplified if exceeded)
- **Factual consistency:** BERTScore F1 ≥ 0.85 vs. source eligibility criteria
- **Retrieval quality:** Precision@5 and NDCG@5 on TREC Clinical Trials 2021

## References

- Wornow et al. (2025). Zero-shot clinical trial patient matching with LLMs. *NEJM AI*. https://arxiv.org/abs/2402.05125
- Jin et al. (2024). Matching patients to clinical trials with LLMs. *Nature Communications*, 15, 9074. https://arxiv.org/abs/2307.15051
- ClinicalTrials.gov API v2: https://clinicaltrials.gov/data-api/api
- TREC Clinical Trials 2021: https://trec-cds.org/2021.html
