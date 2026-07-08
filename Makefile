<<<<<<< HEAD
.PHONY: install run test evaluate experiments lint format clean docker-build docker-run
=======
.PHONY: install run test evaluate lint clean
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c

install:
	pip install -r requirements.txt

run:
	streamlit run ui/streamlit_app.py

test:
<<<<<<< HEAD
	pytest tests/ -v --cov=. --cov-report=term-missing
=======
	pytest tests/ -v
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c

evaluate:
	python -m evaluation.rag_evaluator

<<<<<<< HEAD
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
=======
lint:
	python -m py_compile config.py
	python -m py_compile embeddings/embedder.py
	python -m py_compile retrieval/rag_pipeline.py
	python -m py_compile sentiment/sentiment_analyzer.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c
