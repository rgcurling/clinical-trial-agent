import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

PRIMARY_MODEL = "claude-sonnet-4-20250514"
FAST_MODEL = "claude-haiku-4-5-20251001"
BASELINE_MODEL = "gpt-4o"

CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
MAX_TRIALS_TO_RETRIEVE = 20
MAX_TRIALS_TO_MATCH = 10
MAX_TRIALS_TO_RETURN = 5

CACHE_DIR = "data/cached_trials"
SAMPLE_PATIENTS_DIR = "data/sample_patients"
TARGET_FK_GRADE = 8
BERTSCORE_THRESHOLD = 0.85
EXCLUSION_CONFIDENCE_THRESHOLD = 0.8
