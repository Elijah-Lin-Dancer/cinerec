# CineRec — common developer tasks
# Usage: make <target>

PYTHON ?= python
APP_MODE ?= full
HOST ?= http://127.0.0.1:8000

.PHONY: help setup setup-train data features train eval eval-seeds ablation ablation-seeds coldstart coldstart-seeds precompute charts serve test lint loadtest docker-build clean

help:
	@echo "Targets:"
	@echo "  setup         Install runtime dependencies"
	@echo "  setup-train   Install training/research dependencies (torch, vision, ...)"
	@echo "  data          Download MovieLens 100K + enrich metadata"
	@echo "  features      Compute text/image/genre content features"
	@echo "  train         Train all 6 models and write eval_results.json"
	@echo "  eval-seeds    Repeat the evaluation across 5 seeds → eval_multiseed.json (mean±std)"
	@echo "  ablation      Run the MultiModalNCF ablation study"
	@echo "  ablation-seeds  Repeat the ablation across 5 seeds → ablation_multiseed.json (mean±std)"
	@echo "  coldstart     Run the cold-start study"
	@echo "  coldstart-seeds Repeat the cold-start study across 5 seeds → coldstart_multiseed.json (mean±std)"
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
	# Set HF_ENDPOINT externally if the default Hugging Face host is unreachable.
	$(PYTHON) data/preprocess.py

train:
	$(PYTHON) scripts/train_all.py

eval:
	$(PYTHON) -c "from evaluation.runner import run_evaluation; run_evaluation()"

eval-seeds:
	$(PYTHON) -c "from evaluation.runner import run_evaluation; run_evaluation(seeds=[42, 43, 44, 45, 46])"

ablation:
	$(PYTHON) scripts/ablation_experiment.py

ablation-seeds:
	$(PYTHON) scripts/ablation_experiment.py --seeds 42 43 44 45 46

coldstart:
	$(PYTHON) scripts/coldstart_experiment.py

coldstart-seeds:
	$(PYTHON) scripts/coldstart_experiment.py --seeds 42 43 44 45 46

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
	$(PYTHON) -m locust -f scripts/locustfile.py --headless -u 30 -r 10 -t 30s \
		--host $(HOST) --html reports/locust_report.html

docker-build:
	docker build --build-arg APP_MODE=$(APP_MODE) -t cinerec:$(APP_MODE) .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache