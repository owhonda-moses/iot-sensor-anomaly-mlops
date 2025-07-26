FROM python:3.11-slim AS builder

# install system deps & Poetry
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential git \
 && curl -sSL https://install.python-poetry.org | python3 - \
 && mv /root/.local/bin/poetry /usr/local/bin/poetry \
 && apt-get purge -y --auto-remove curl build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# bring in dependency specs
COPY pyproject.toml poetry.lock ./

# install Python deps
RUN pip install --upgrade pip \
 && poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --without dev \
 && pip install psycopg2-binary

# copy app code  
COPY src/ ./src/


FROM python:3.11-slim

WORKDIR /app

# copy installed packages & executables
COPY --from=builder /usr/local/lib/python3.11/site-packages/ \
                    /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ \
                    /usr/local/bin/

# copy app code
COPY --from=builder /app/src/ ./src/

# Cloud Run 
EXPOSE 8080

ENTRYPOINT ["mlflow", "server"]
CMD ["--host","0.0.0.0","--port","8080","--backend-store-uri","$MLFLOW_TRACKING_URI","--default-artifact-root","gs://${ARTIFACT_BUCKET}/mlruns"]

