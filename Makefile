.PHONY: install format lint test

install:
	poetry install

format:
	poetry run black .
	poetry run ruff --fix .

lint:
	poetry run ruff check .

test:
	poetry run pytest