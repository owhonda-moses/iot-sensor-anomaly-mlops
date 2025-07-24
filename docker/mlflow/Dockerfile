FROM python:3.11-slim AS builder

#system deps & Poetry
RUN apt-get update \
  && apt-get install -y curl build-essential git \
  && curl -sSL https://install.python-poetry.org | python3 - \
  && mv /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# install prod deps only
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi


FROM python:3.11-slim

WORKDIR /app

# Copy packages and app code
COPY --from=builder /usr/local/lib/python3.11/site-packages/ \
                    /usr/local/lib/python3.11/site-packages/

COPY . .

EXPOSE 5000 4200

# entrypoint
CMD ["bash","-lc", "\
  mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root gs://${ARTIFACT_BUCKET}/mlruns & \
  prefect orion start --host 0.0.0.0 --port 4200"]