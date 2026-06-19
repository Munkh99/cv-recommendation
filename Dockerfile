FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential: fallback for any source-only wheels (pymupdf/chromadb deps)
# curl: used by the compose healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# api on 8000, streamlit ui on 8501 — the actual command is set per-service in compose.
EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
