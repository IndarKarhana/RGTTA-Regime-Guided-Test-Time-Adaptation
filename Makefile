# Makefile for RGTTA — Regime-Guided Test-Time Adaptation
# Unified benchmark: 7 policies × 4 models × 4 ETT datasets

PYTHON = .venv/bin/python
PYTHONPATH_ENV = PYTHONPATH=.:src:benchmarks:benchmarks/data_loaders

.PHONY: help install test lint format benchmark benchmark-quick benchmark-clean clean-pkl

help:
	@echo "Makefile targets:"
	@echo "  make install           - install dependencies (pip)"
	@echo "  make test              - run test suite"
	@echo "  make lint              - run ruff linter"
	@echo "  make format            - format code with ruff"
	@echo "  make benchmark         - full 7-policy benchmark (4 models × 4 datasets × 3 horizons)"
	@echo "  make benchmark-quick   - smoke test (1 model × 1 dataset × 1 horizon)"
	@echo "  make benchmark-clean   - clean results and re-run benchmark"
	@echo "  make clean-pkl         - delete ALL .pkl checkpoint files from workspace"

install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install -e .

test:
	@echo "Running test suite..."
	$(PYTHON) -m pytest -v tests/

lint:
	@echo "Running ruff linter..."
	$(PYTHON) -m ruff check src/ tests/ benchmarks/

format:
	@echo "Formatting code with ruff..."
	$(PYTHON) -m ruff format src/ tests/ benchmarks/

benchmark:
	@echo "Running full 7-policy unified benchmark..."
	@echo "  Policies: Retrain, TTA, EWC, DynaTTA, TAFAS, RGTTA, RGTTA+EWC"
	@echo "  Models:   GRU-Small, LSTM, GRU-Large, Transformer"
	@echo "  Datasets: ETTh1, ETTh2, ETTm1, ETTm2"
	@echo "  Horizons: 96, 192, 336"
	@echo "  Expected runtime: ~4-6 hours"
	$(PYTHONPATH_ENV) $(PYTHON) benchmarks/run_unified_benchmark.py --seeds 1 --horizons 96 192 336

benchmark-quick:
	@echo "Running smoke test (GRU-Small, ETTh1, h=96)..."
	$(PYTHONPATH_ENV) $(PYTHON) benchmarks/run_unified_benchmark.py --quick --datasets ETTh1

benchmark-clean:
	@echo "Cleaning benchmark artifacts..."
	rm -rf benchmarks/results/unified/ckpt_*
	@echo "Running fresh benchmark..."
	$(MAKE) benchmark

clean-pkl:
	@echo "Deleting all .pkl checkpoint files..."
	find . -name "*.pkl" -not -path "./.git/*" -delete
	@echo "✅ All .pkl files removed."
