.PHONY: install test test-py test-js lint validate
install:
	pip install -r requirements-dev.txt
test: test-py test-js
test-py:
	python -m pytest --cov=backend --cov=research --cov-report=term-missing
test-js:
	node --test tests/frontend/*.test.js
lint:
	ruff check .
validate:   ## clinical model + GA validation report (no GPU, no X-rays needed)
	python research/clinical_pipeline.py --data data/clinical_train_800.csv --out docs/validation_report.json
