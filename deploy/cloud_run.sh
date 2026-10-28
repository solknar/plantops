#!/usr/bin/env bash
# Deploy PlantOps to Cloud Run (scales to zero; fits in the free tier at demo traffic).
#
# Prereqs (one time):
#   brew install --cask google-cloud-sdk && gcloud auth login
#   gcloud projects create <id> && gcloud config set project <id>
#   Link a billing account: Cloud Run's free tier still requires one.
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com cloudtrace.googleapis.com
#   printf '%s' "$GOOGLE_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
#
# Then: ./deploy/cloud_run.sh [region]
set -euo pipefail

REGION="${1:-us-central1}"
PROJECT="$(gcloud config get-value project)"
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:${SA}" --role=roles/secretmanager.secretAccessor >/dev/null

gcloud run deploy plantops \
  --source . \
  --region "$REGION" \
  --no-allow-unauthenticated \
  --min-instances 0 --max-instances 1 \
  --memory 1Gi --cpu 1 \
  --set-secrets GOOGLE_API_KEY=gemini-api-key:latest \
  --set-env-vars PLANTOPS_CLOUD_TRACE=1,GOOGLE_GENAI_USE_VERTEXAI=FALSE

echo
echo "Private service deployed. Open it locally through an authenticated proxy:"
echo "  gcloud run services proxy plantops --region $REGION --port 8080"
