# Purchase Intent Engine: one image for the REST API and the dashboard.
#   docker compose up --build        (see docker-compose.yml)
# Trained runs and tables are mounted from ./results and ./data, not baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-serve.txt .
# CPU-only PyTorch keeps the image about 1.5 GB smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    && pip install -r requirements-serve.txt

COPY config ./config
COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 engine && chown -R engine /app
USER engine

ENV PIE_MODE=full
EXPOSE 8000 8501
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
