# TrialMatch AI

Match a patient clinical note to relevant clinical trials using a two-agent AI pipeline (Claude + GPT-4o), with plain-English explanations written at an 8th-grade reading level.

> **Research prototype.** Not medical advice. Not HIPAA-compliant.

---

## How it works

```
Patient note (free text)
        │
        ▼
┌───────────────────┐
│  Profile Extractor│  Claude extracts age, conditions, biomarkers, stage
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Trial Retrieval  │  ClinicalTrials.gov API v2 → up to 50 live recruiting trials
└────────┬──────────┘    (TREC eval: TF-IDF or BiomedBERT on static 26K corpus)
         │
         ▼
┌───────────────────┐
│  BiomedBERT Rank  │  PubMedBERT embeddings → re-rank by cosine similarity → top 10
└────────┬──────────┘
         │
         ▼
┌───────────────────┐   ┌─────────────────────┐
│  Agent 1 — Claude │──▶│ Agent 2 — GPT-4o    │  independent critic review
│  per-trial match  │   │ accepts / overrides  │
└────────┬──────────┘   └──────────┬──────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌─────────────────┐
         │  Resolver       │  merges both agents; flags uncertainty
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  Ranker         │  hard exclusion filter → score sort
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  Explainer      │  FK-controlled patient card (≤ grade 8)
         └─────────────────┘
                  │
                  ▼
         Top-5 ranked trials + explanations
```

---

## Quick start — local CLI

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys
cp .env.example trialmatch/.env
#    → edit trialmatch/.env, add ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Run on a single patient file
cd trialmatch
python main.py --patient-file data/sample_patients/patient_01.txt

# 4. Run the full synthetic benchmark (FK + BERTScore)
python main.py --eval-synthetic
```

---

## Quick start — production API (Docker)

```bash
# 1. Set API keys
cp .env.example .env
#    → edit .env, add ANTHROPIC_API_KEY and OPENAI_API_KEY

# 2. Build and start (first run downloads BiomedBERT, ~5 min)
docker-compose up --build

# 3. Health check
curl http://localhost:8000/health

# 4. Match a patient
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text": "58-year-old female with Stage IIIB non-small cell lung cancer, EGFR negative, PD-L1 50%, prior platinum chemotherapy.",
    "location": "Indianapolis, IN",
    "max_trials": 5,
    "use_critic": true
  }'
```

API docs available at `http://localhost:8000/docs` (Swagger) and `/redoc`.

---

## API reference

### `GET /health`

```json
{"status": "ok", "version": "1.0.0", "model": "claude-sonnet-4-20250514"}
```

### `POST /match`

**Request body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patient_text` | string | required | Free-text clinical note |
| `location` | string | null | City/state for proximity filtering |
| `max_trials` | int | 5 | Top-N matches to return (1–20) |
| `use_critic` | bool | false | Enable GPT-4o critic review |
| `status_filter` | string | `"RECRUITING"` | ClinicalTrials.gov status filter |

**Response** (`MatchResponse`)

```jsonc
{
  "status": "success",
  "patient_profile": {
    "conditions": ["Non-small cell lung cancer"],
    "age": 58,
    "gender": "female",
    "biomarkers": ["EGFR negative", "PD-L1 50%"],
    "stage": "IIIB",
    "medications": ["carboplatin", "pemetrexed"]
  },
  "matches": [
    {
      "rank": 1,
      "nct_id": "NCT05123456",
      "title": "Phase III Study of Pembrolizumab ...",
      "phase": "PHASE3",
      "overall_score": 0.87,
      "met_criteria": ["Age ≥ 18", "PD-L1 ≥ 1%", "Prior platinum therapy"],
      "failed_criteria": [],
      "uncertain_criteria": ["ECOG performance status"],
      "hard_exclusion": false,
      "explanation": "You may be eligible for this trial. It is looking for patients with lung cancer who have already received chemotherapy ...",
      "fk_grade": 6.4,
      "trial_url": "https://clinicaltrials.gov/study/NCT05123456",
      "locations": ["Indianapolis, IN", "Chicago, IL"]
    }
  ],
  "n_candidates_retrieved": 47,
  "n_candidates_matched": 20,
  "processing_time_ms": 8420.3
}
```

---

## M3 TREC 2021 Validation

Four ablation configurations evaluated on TREC Clinical Trials 2021 (topics 26–40, 15 topics).

| Run | Retriever | Critic | P@5 | NDCG@5 |
|-----|-----------|--------|-----|--------|
| Baseline | TF-IDF | — | — | — |
| R1 | BiomedBERT | — | — | — |
| R2 | TF-IDF | GPT-4o | — | — |
| R3 (Combined) | BiomedBERT | GPT-4o | — | — |

*Results populated after full experiment run completes.*

### Run the experiments

```bash
cd trialmatch

# Full run — all 4 configs, topics 26–40 (~2–4 h, ~$15 in API costs)
python run_experiments.py --fresh --topic-range 26 40

# If interrupted, resume without --fresh (per-config checkpoints every 5 topics)
python run_experiments.py --topic-range 26 40

# Generate slide deck artifacts after runs complete
python compare_runs.py                                    # → results/comparison_table.csv
python compute_confusion_matrix.py --run results/run_r3.json   # → results/confusion_matrix.png
python evaluate_explanations.py --run results/run_r3.json      # → results/explanation_metrics.csv
python select_audit_cases.py --run results/run_r3.json         # → results/audit_sheet.txt
```

---

## Project structure

```
clinical-trial-agent/
├── src/                          # Production service
│   ├── api/
│   │   ├── app.py                # FastAPI application (CORS, lifespan)
│   │   ├── routes.py             # POST /match, GET /health
│   │   └── models.py             # Pydantic request/response schemas
│   ├── agents/
│   │   ├── orchestrator.py       # Agentic loop — Claude drives the full pipeline via tool use
│   │   ├── tools.py              # Tool schemas + ToolExecutor (BiomedBERT rank, evaluate, explain)
│   │   ├── claude_agent.py       # Agent 1 — Claude eligibility assessor
│   │   ├── gpt4_agent.py         # Agent 2 — GPT-4o critic
│   │   └── resolver.py           # Two-agent discrepancy resolver
│   └── data/
│       └── clinicaltrials_api.py # Async ClinicalTrials.gov API v2 client
│
├── trialmatch/                   # Academic pipeline (M1–M3)
│   ├── pipeline/
│   │   ├── extractor.py          # Patient profile extraction (Claude)
│   │   ├── retriever.py          # TfidfRetriever + BiomedBERTRetriever
│   │   ├── matcher.py            # ClaudeMatcher + GPT-4o critic
│   │   ├── ranker.py             # Hard exclusion filter + score sort
│   │   └── explainer.py          # Plain-English patient card generator
│   ├── eval/
│   │   ├── benchmark.py          # TREC benchmark runner (checkpointed)
│   │   └── metrics.py            # P@5, NDCG@5, BERTScore, FK grade
│   ├── data/
│   │   ├── trec_2021/            # TREC 2021 corpus + qrels (gitignored)
│   │   └── sample_patients/      # 10 synthetic patient files
│   ├── results/                  # Experiment outputs (run_*.json, PNGs, CSVs)
│   ├── tests/                    # 32 unit tests
│   ├── run_experiments.py        # 4-config ablation runner
│   ├── compare_runs.py           # Comparison table generator
│   ├── compute_confusion_matrix.py
│   ├── evaluate_explanations.py
│   └── select_audit_cases.py
│
├── scripts/                      # Top-level wrappers for trialmatch eval scripts
├── Dockerfile                    # Multi-stage: builder + slim runtime
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Running tests

```bash
cd trialmatch
pytest tests/ -v          # 32 tests
```

---

## Deployment

### Docker (single machine)

```bash
docker-compose up --build
```

### Google Cloud Run (recommended)

```bash
# One-time setup
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev

# Deploy (builds image remotely, stores API keys as GCP secrets)
PROJECT_ID=your-gcp-project ./deploy/cloudrun.sh
```

The script creates an Artifact Registry repo, builds the image via Cloud Build, stores your API keys as GCP Secrets, and deploys to Cloud Run. Cold-start time is under 15 seconds. Cost is effectively $0 at low traffic (scales to zero).

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `OPENAI_API_KEY` | Yes | OpenAI API key (GPT-4o critic) |
| `PORT` | No (default 8000) | Service port |
| `WORKERS` | No (default 4) | Uvicorn worker count |
| `CORS_ORIGINS` | No (default `*`) | Comma-separated allowed origins |

---

## License

MIT — see [LICENSE](LICENSE).
