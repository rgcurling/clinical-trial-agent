# TrialMatch AI

TrialMatch AI is a 5-stage agentic LLM pipeline that accepts a patient's free-text medical history, queries the live ClinicalTrials.gov REST API, and uses Claude to match the patient against each trial's eligibility criteria one criterion at a time — returning a ranked, plain-English explanation of which trials they may qualify for.

---

## Architecture

```
Patient Free Text
       │
       ▼
┌──────────────────┐
│  Stage 1         │  pipeline/extractor.py
│  Entity          │  scispaCy NER → PatientProfile dataclass
│  Extraction      │  (LLM fallback if scispaCy unavailable)
└────────┬─────────┘
         │ PatientProfile
         ▼
┌──────────────────┐
│  Stage 2         │  pipeline/retriever.py
│  Trial           │  ClinicalTrials.gov API v2 → list[Trial]
│  Retrieval       │  (disk cache · tenacity retry on 429/5xx)
└────────┬─────────┘
         │ list[Trial]
         ▼
┌──────────────────┐
│  Stage 3         │  pipeline/matcher.py
│  Per-Criterion   │  ONE Claude call per criterion (never batched)
│  Matching        │  → list[MatchResult]
└────────┬─────────┘
         │ list[MatchResult]
         ▼
┌──────────────────┐
│  Stage 4         │  pipeline/ranker.py
│  Ranking &       │  Hard-filter excluded trials → sort by score
│  Filtering       │  → top-5 MatchResults
└────────┬─────────┘
         │ list[MatchResult]
         ▼
┌──────────────────┐
│  Stage 5         │  pipeline/explainer.py
│  Plain-English   │  Claude → patient card · FK grade check
│  Explanation     │  Auto-simplify if grade > 8
└────────┬─────────┘
         │
         ▼
   Ranked Trial Cards (stdout / JSON)
```

---

## Quickstart

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd trialmatch

# 2. Install dependencies (Python 3.11+ recommended)
pip install -r requirements.txt

# Optional: install scispaCy biomedical model for best NER quality
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# 3. Set your API key
cp .env .env.local
# Edit .env.local and replace "your_key_here" with your Anthropic API key
# (Rename to .env or export ANTHROPIC_API_KEY= directly)

# 4. Run the pipeline on a single patient
python main.py --patient "58-year-old male with Stage IIIB NSCLC, EGFR negative, Indianapolis IN"
```

Expected output: ranked trial cards printed to stdout with FK grade per card.

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `--patient "TEXT"` | Run pipeline on an inline patient description string |
| `--patient-file PATH` | Run pipeline on a .txt patient file |
| `--output PATH` | Save results to a JSON file |
| `--eval-synthetic` | Run full pipeline on all 10 patients in `data/sample_patients/` and print summary table |
| `--benchmark --data-dir DIR` | Run n2c2 2018 cohort selection benchmark; report macro-F1 and micro-F1 |
| `--compare-models` | Run both Claude (primary) and GPT-4o (baseline) and print a side-by-side score comparison |
| `--clear-cache` | Delete all cached ClinicalTrials.gov API responses in `data/cached_trials/` |

### Examples

```bash
# File input + save JSON
python main.py --patient-file data/sample_patients/patient_01.txt --output results/patient_01.json

# Synthetic eval on all 10 patients
python main.py --eval-synthetic

# n2c2 benchmark
python main.py --benchmark --data-dir data/n2c2/

# Compare Claude vs GPT-4o (requires OPENAI_API_KEY in .env)
python main.py --patient-file data/sample_patients/patient_01.txt --compare-models

# Clear cache
python main.py --clear-cache
```

---

## Running Tests

```bash
cd trialmatch
pytest tests/ -v
```

---

## Eval Results

*(To be filled in after M2/M3 runs)*

| Patient | Avg FK Grade | BERTScore |
|---------|-------------|-----------|
| patient_01 | — | — |
| patient_02 | — | — |
| patient_03 | — | — |
| patient_04 | — | — |
| patient_05 | — | — |
| patient_06 | — | — |
| patient_07 | — | — |
| patient_08 | — | — |
| patient_09 | — | — |
| patient_10 | — | — |

| Model | Macro-F1 | Micro-F1 | Avg FK Grade |
|-------|----------|----------|--------------|
| Claude (primary) | — | — | — |
| GPT-4o (baseline) | — | — | — |

---

## Key Design Decisions

**Criterion-by-criterion prompting (non-negotiable)**
Per Wornow et al. (2025) and Jin et al. (2024), matching each criterion in a separate LLM call significantly outperforms batching all criteria into one prompt. `matcher.py` never batches.

**Disk caching**
Every ClinicalTrials.gov API response is written to `data/cached_trials/{hash}.json` on first call. Subsequent identical queries hit the cache, protecting against rate limits during development and grading.

**Graceful fallbacks**
- scispaCy unavailable → LLM extraction
- API unreachable → load from cache
- Malformed LLM JSON → one repair attempt before raising

---

## References

- Wornow, M., Lozano, A., Dash, D., Jindal, J., Mahaffey, K. W., & Shah, N. H. (2025). Zero-shot clinical trial patient matching with LLMs. *NEJM AI*, 2(1). https://arxiv.org/abs/2402.05125
- Jin, Q., Wang, Z., Floudas, C. S., et al. (2024). Matching patients to clinical trials with large language models. *Nature Communications*, 15, 9074. https://arxiv.org/abs/2307.15051
- ClinicalTrials.gov API v2 documentation: https://clinicaltrials.gov/data-api/api
- n2c2 2018 Cohort Selection Task: https://n2c2.dbmi.hms.harvard.edu/2018-shared-tasks
