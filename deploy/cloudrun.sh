#!/usr/bin/env bash
# Deploy TrialMatch AI to Google Cloud Run
#
# Prerequisites:
#   gcloud auth login
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#
# Usage:
#   ./deploy/cloudrun.sh                        # uses PROJECT_ID env var
#   PROJECT_ID=my-project ./deploy/cloudrun.sh
#   ./deploy/cloudrun.sh --region us-east1

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID or pass it as an env var}"
REGION="${REGION:-us-central1}"
SERVICE="trialmatch-ai"
REPO="trialmatch"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"

echo "Project : ${PROJECT_ID}"
echo "Region  : ${REGION}"
echo "Image   : ${IMAGE}"
echo ""

# ── 1. Create Artifact Registry repo (idempotent) ────────────────────────────
gcloud artifacts repositories describe "${REPO}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" &>/dev/null \
  || gcloud artifacts repositories create "${REPO}" \
        --project="${PROJECT_ID}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="TrialMatch AI container images"

# ── 2. Build & push via Cloud Build (no local Docker needed) ─────────────────
gcloud builds submit . \
    --project="${PROJECT_ID}" \
    --tag="${IMAGE}" \
    --timeout=20m

# ── 3. Store API keys as secrets (only needed on first deploy) ───────────────
for SECRET in ANTHROPIC_API_KEY OPENAI_API_KEY; do
    if ! gcloud secrets describe "${SECRET}" --project="${PROJECT_ID}" &>/dev/null; then
        echo ""
        echo "Creating secret ${SECRET} — paste the value and press Ctrl-D:"
        gcloud secrets create "${SECRET}" \
            --project="${PROJECT_ID}" \
            --data-file=-
    fi
done

# ── 4. Deploy to Cloud Run ────────────────────────────────────────────────────
gcloud run deploy "${SERVICE}" \
    --project="${PROJECT_ID}" \
    --image="${IMAGE}" \
    --platform=managed \
    --region="${REGION}" \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --concurrency=10 \
    --min-instances=0 \
    --max-instances=5 \
    --set-secrets="ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest" \
    --set-env-vars="WORKERS=1,PYTHONPATH=/app"

# ── 5. Print service URL ──────────────────────────────────────────────────────
URL=$(gcloud run services describe "${SERVICE}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format="value(status.url)")

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deployed: ${URL}"
echo "  Health:   ${URL}/health"
echo "  Docs:     ${URL}/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Test it:"
echo "  curl ${URL}/health"
echo "  curl -X POST ${URL}/match \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"patient_text\": \"58yo female, Stage IIIB NSCLC, EGFR negative\"}'"
