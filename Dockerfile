# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: production (Cloud Run / docker-compose) ─────────────────────────
# No BiomedBERT — production uses the live ClinicalTrials.gov API, not the
# TREC corpus. Keeps the image under 1 GB and startup under 10 seconds.
FROM python:3.11-slim AS production

RUN useradd -m -u 1000 trialmatch
WORKDIR /app

COPY --from=builder /install /usr/local

COPY trialmatch/ /app/trialmatch/
COPY src/ /app/src/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WORKERS=1

RUN chown -R trialmatch:trialmatch /app
USER trialmatch

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level info


# ── Stage 3: local-eval (docker-compose --profile eval) ──────────────────────
# Includes BiomedBERT for running the TREC benchmark locally.
FROM production AS local-eval

USER root
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('NeuML/pubmedbert-base-embeddings')"
USER trialmatch
