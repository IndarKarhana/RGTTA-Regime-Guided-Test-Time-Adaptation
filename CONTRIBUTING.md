# Contributing to RG-TTA

Thank you for your interest in contributing to this project.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
git clone https://github.com/IndarKarhana/RGTTA-Regime-Guided-Test-Time-Adaptation.git
cd RGTTA-Regime-Guided-Test-Time-Adaptation

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
PYTHONPATH=.:src:benchmarks:benchmarks/data_loaders pytest tests/ -v
```

### Running Linters

```bash
ruff check src/ benchmarks/ tests/      # Lint
ruff format src/ benchmarks/ tests/     # Format (auto-fix)
ruff format --check src/ benchmarks/ tests/  # Format (check only)
```

## Development Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with tests where appropriate.
3. Ensure all checks pass:
   - `ruff check` (no lint errors)
   - `ruff format --check` (formatting consistent)
   - `pytest tests/` (all tests pass)
4. Open a pull request against `main`.

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Line length limit: 120 characters.
- Follow PEP 8 conventions.
- Use type hints for function signatures.
- Keep functions focused and small.

## Project Structure

```
src/regime_forecasting/     # Core library (models, memory, regime detection)
benchmarks/                 # Benchmark runner and policy implementations
tests/                      # Test suite
paper/                      # LaTeX paper and figures
```

## Key Conventions

- **6 primary policies**: TTA, EWC, DynaTTA, RG-TTA, RG-EWC, RG-DynaTTA
- **Same base model** across all policies per experiment
- **Frozen backbone**: Only `output_projection` is trainable during adaptation
- **Streaming protocol**: 10-batch sequential evaluation (not sliding window)

## Reporting Issues

Open an issue on GitHub with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
