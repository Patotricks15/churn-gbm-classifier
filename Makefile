.PHONY: install test reproduce

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

reproduce:
	@test -n "$(INPUT_DIR)" || (echo "Set INPUT_DIR=/path/to/olist/csv/folder" && exit 1)
	python scripts/run_gbm.py --input-dir "$(INPUT_DIR)" --out-dir reports
