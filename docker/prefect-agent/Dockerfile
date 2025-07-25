FROM python:3.11-slim AS builder

# Install poetry and build dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl build-essential git \
  && curl -sSL https://install.python-poetry.org | python3 - \
  && mv /root/.local/bin/poetry /usr/local/bin/poetry \
  && apt-get purge -y --auto-remove curl build-essential git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install app dependencies
COPY pyproject.toml poetry.lock ./
COPY src/ src/
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --without dev

FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and code
COPY --from=builder /usr/local/lib/python3.11/site-packages/ \
                    /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ \
                    /usr/local/bin/
COPY . .

# expose port if agent metrics or health end up on HTTP
EXPOSE 8080

CMD ["bash","-lc","prefect agent start -q default --orion api://${PREFECT_ORION_API_URL}"]
