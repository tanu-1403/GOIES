# ── GOIES Production Dockerfile ───────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create runtime data dirs (volumes can override)
RUN mkdir -p goies_snapshots

# Default port — Railway overrides this with $PORT
ENV PORT=8000
EXPOSE 8000

CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
