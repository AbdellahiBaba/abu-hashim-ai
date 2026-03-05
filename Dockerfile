FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .

RUN useradd -m -s /bin/bash appuser && \
    mkdir -p dataset_raw/qalam_exports \
    dataset_processed/qalam_processed \
    learning_buffer \
    model_base \
    model_finetune \
    model_inference && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

CMD ["uvicorn", "api_server.main:app", "--host", "0.0.0.0", "--port", "5000"]
