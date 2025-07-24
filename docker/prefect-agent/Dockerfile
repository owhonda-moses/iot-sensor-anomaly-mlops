FROM python:3.11-slim AS builder

# system tools & poetry
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl build-essential git \
  && curl -sSL https://install.python-poetry.org | python3 - \
  && mv /root/.local/bin/poetry /usr/local/bin/poetry \
  && apt-get purge -y --auto-remove curl build-essential git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install production dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --without dev

FROM python:3.11-slim

WORKDIR /app

# copy packages and app code
COPY --from=builder /usr/local/lib/python3.11/site-packages/ \
                    /usr/local/lib/python3.11/site-packages/

COPY . .

EXPOSE 8080

# start agent
CMD ["bash","-lc", "\
  prefect agent start -q default --orion api://${PREFECT_ORION_URL}"]
