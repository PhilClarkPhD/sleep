# Mora Sleep Scoring API - Docker Image
#
# IMPORTANT: this image must be built with the REPO ROOT as the build context
# (not backend/), because it includes the standalone `sleep_features` package
# that lives at the repo root. All COPY paths below are relative to the repo root.
#   docker build -f backend/Dockerfile -t mora-api .
# (docker-compose sets this context; on Railway set the service Root Directory to
#  the repo root and Dockerfile Path to backend/Dockerfile.)

FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app/ ./app/

# Copy the standalone feature-engineering package (shared source of truth).
# /app is on sys.path at runtime, so both `app` and `sleep_features` import cleanly.
COPY sleep_features/ ./sleep_features/

# Copy model artifacts
COPY backend/models/ ./models/

# Copy and prepare startup script
COPY backend/start.sh .
RUN chmod +x start.sh

# Expose port (informational)
EXPOSE 8000

# Run via startup script (handles Railway's PORT variable)
CMD ["./start.sh"]
