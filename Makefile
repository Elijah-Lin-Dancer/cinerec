# CineRec — common developer tasks
# Usage: make <target>

PYTHON ?= python
APP_MODE ?= full
HOST ?= http://127.0.0.1:8000

.PHONY: help setup setup-train data features train eval ablation coldstart precompute charts serve test lint loadtest docker-build clean

help:
	@echo "Targets:"
	@echo "  setup         Install runtime dependencies"
	@echo "  setup-train   Install training/research dependencies (torch, vision, ...)"
	@echo "  data          Download MovieLens 100K + enrich metadata"
	@echo "  features      Compute text/image/genre content features"
	@echo "  train         Train all 6 models and write eval_results.json"
	@echo "  ablation      Run the MultiModalNCF ablation study"
	@echo "  coldstart     Run the cold-start study"
	@echo "  precompute    Build recs_cache.json for APP_MODE=lite"
	@echo "  charts        Regenerate docs/*.png from measured results"
	@echo "  serve         Run the API locally on :8000 (APP_MODE=$(APP_MODE))"
	@echo "  test          Run the test suite"
	@echo "  lint          Run ruff"
	@echo "  loadtest      Run the Locust load test against a running server (HOST=$(HOST))"
	@echo "  docker-build  Build the Docker image (APP_MODE=$(APP_MODE))"
	@echo "  clean         Remove caches"

setup:
	$(PYTHON) -m pip install -r requirements.txt

setup-train:
	$(PYTHON) -m pip install -r requirements-train.txt

data:
	$(PYTHON) data/download.py
	$(PYTHON) data/enrich_tmdb.py

features:
	HF_ENDPOINT=$${HF_ENDPOINT:-https://hf-mirror.com} $(PYTHON) data/preprocess.py

train:
	$(PYTHON) scripts/train_all.py

eval:
	$(PYTHON) -c "from evaluation.runner import run_evaluation; run_evaluation()"

ablation:
	$(PYTHON) scripts/ablation_experiment.py

coldstart:
	$(PYTHON) scripts/coldstart_experiment.py

precompute:
	$(PYTHON) scripts/precompute.py

charts:
	$(PYTHON) -c "from evaluation.visualize import generate_all; generate_all()"

serve:
	APP_MODE=$(APP_MODE) $(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

loadtest:
	$(PYTHON) -m locust -f scripts/locustfile.py --headless -u 20 -r 5 -t 30s \
		--host $(HOST) --html reports/locust_report.html

docker-build:
	docker build --build-arg APP_MODE=$(APP_MODE) -t cinerec:$(APP_MODE) .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache