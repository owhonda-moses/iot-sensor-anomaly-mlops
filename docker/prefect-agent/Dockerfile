FROM python:3.11-slim AS builder

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl build-essential git \
  && curl -sSL https://install.python-poetry.org | python3 - \
  && mv /root/.local/bin/poetry /usr/local/bin/poetry \
  && apt-get purge -y --auto-remove curl build-essential git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --without-dev

COPY src/ src/

FROM python:3.11-slim
WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages/ \
                    /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ \
                    /usr/local/bin/

COPY . .

# expose if agent health or metrics is served
EXPOSE 8080

ENTRYPOINT ["bash", "-lc"]
CMD ["prefect agent start -q default --orion api://${PREFECT_ORION_UI_API_URL}"]
