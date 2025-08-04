.PHONY: install format lint test

install:
	poetry install

format:
	poetry run black .
	poetry run ruff check --fix .

lint:
	poetry run ruff check .

test:
	poetry run pytest

run:
	poetry run python prefect_deploy.py
