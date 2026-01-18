# Mora Sleep Scoring API - Docker Image

FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy model artifacts
COPY models/ ./models/

# Copy and prepare startup script
COPY start.sh .
RUN chmod +x start.sh

# Expose port (informational)
EXPOSE 8000

# Run via startup script (handles Railway's PORT variable)
CMD ["./start.sh"]
