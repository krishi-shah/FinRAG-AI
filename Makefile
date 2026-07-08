.PHONY: install run test evaluate experiments lint format clean docker-build docker-run

install:
	pip install -r requirements.txt

run:
	streamlit run ui/streamlit_app.py

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

evaluate:
	python -m evaluation.rag_evaluator

experiments:
	python -m evaluation.experiments

lint:
	ruff check .

format:
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	rm -rf data/index/*.index data/index/*_chunks.json

docker-build:
	docker compose build

docker-run:
	docker compose up -d
