SHELL := /bin/bash
PY ?= uv run
SEED ?= 20251201
RAW_DIR ?= data/raw
INTERIM_DIR ?= data/interim
REPORTS_DIR ?= reports
GEOJSON ?= milano-grid.geojson
AREAS ?= top,4159,4556

export PYTHONHASHSEED := $(SEED)

.PHONY: bootstrap ingest eda baselines train-sarima train-lstm train-cnn train forecast eval test test-perf lint format clean

bootstrap:
	uv venv
	uv sync --all-extras
	uv run pre-commit install || true

ingest:
	$(PY) mtraffic ingest --raw-dir $(RAW_DIR) --out-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR) --measure-memory

eda:
	$(PY) mtraffic eda all --interim-dir $(INTERIM_DIR) --geojson $(GEOJSON) --areas $(AREAS) --report-dir $(REPORTS_DIR)

baselines:
	$(PY) mtraffic forecast --model naive_last --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)
	$(PY) mtraffic forecast --model naive_d144 --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)
	$(PY) mtraffic forecast --model naive_w1008 --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)

train-sarima:
	$(PY) mtraffic train sarima --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)

train-lstm:
	$(PY) mtraffic train lstm --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)

train-cnn:
	$(PY) mtraffic train cnn --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)

train: train-sarima train-lstm train-cnn

forecast:
	$(PY) mtraffic forecast --model sarima --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)
	$(PY) mtraffic forecast --model lstm   --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)
	$(PY) mtraffic forecast --model cnn    --areas $(AREAS) --interim-dir $(INTERIM_DIR) --report-dir $(REPORTS_DIR)

eval:
	$(PY) mtraffic eval task3 --report-dir $(REPORTS_DIR) --areas $(AREAS)

test:
	$(PY) pytest tests/unit tests/integration

test-perf:
	$(PY) pytest tests -m perf

lint:
	$(PY) ruff check src tests
	$(PY) black --check src tests
	$(PY) mypy

format:
	$(PY) ruff check --fix src tests
	$(PY) black src tests

clean:
	@echo "This will delete data/interim and reports. Press Ctrl-C to abort."
	@sleep 3
	rm -rf data/interim reports/runs reports/models
	find reports -maxdepth 2 -name '*.png' -delete || true
	find reports -maxdepth 2 -name '*.csv' -delete || true
