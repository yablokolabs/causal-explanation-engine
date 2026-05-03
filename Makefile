.PHONY: install dev lint typecheck test serve regression clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy core api models causal llm retrieval validation

test:
	pytest

serve:
	uvicorn api.main:app --reload

regression:
	python scripts/evaluate_regression.py --n 1000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache artifacts/*.json artifacts/*.joblib artifacts/*.sha256
