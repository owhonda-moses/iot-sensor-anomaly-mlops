FROM python:3.11-slim

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV PYTHONPATH=/app

EXPOSE 8080

# overridden for MLflow/Prefect
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "predict:app"]