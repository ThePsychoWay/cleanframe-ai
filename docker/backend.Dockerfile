# API and worker image. Build context is the repository root:
#   docker build -f docker/backend.Dockerfile -t cleanframe-backend .
# GPU worker (a PyTorch model on this machine): add --build-arg INSTALL_AI=true
FROM python:3.13-slim

ARG INSTALL_AI=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements-ai.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_AI" = "true" ]; then pip install -r requirements-ai.txt; fi

COPY backend/ ./

# Run as an ordinary user. Workers write large temporary files under /tmp/cleanframe (mount a volume there).
RUN useradd --create-home --uid 10001 cleanframe \
    && mkdir -p /tmp/cleanframe \
    && chown -R cleanframe /app /tmp/cleanframe
USER cleanframe

ENV WORKER_TMP_DIR=/tmp/cleanframe
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
