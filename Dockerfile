FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PLANTOPS_VAR_DIR=/app/var PORT=8080
WORKDIR /app

COPY pyproject.toml ./
COPY plantops ./plantops
RUN pip install --no-cache-dir ".[gcp]"

COPY data ./data

EXPOSE 8080
# The index is built on startup (~300 chunks, a few embedding calls). Cloud Run's filesystem is
# ephemeral, so sessions and telemetry reset on each cold start; see README "Production path".
CMD ["sh", "-c", "plantops ingest && plantops serve --host 0.0.0.0 --port ${PORT}"]
