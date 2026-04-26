# ML Platform Toolkit

A small internal ML engineering toolkit that standardizes project scaffolding, training loops, experiment tracking, evaluation, and dataset ingestion across teams.

## What it provides

- `ml init <project_name>` to generate a new ML project scaffold
- Reusable training-loop abstraction for PyTorch-style experiments
- Reference PyTorch trainer and model template in generated projects
- Local JSON and SQLite experiment tracking
- Standardized classification reports with confusion matrices and latency benchmarking
- Dataset loader and preprocessing pipeline abstractions
- CI templates for test, config validation, and training sanity checks

## Install

```bash
pip install -e .
```

For local development and git checks:

```bash
pip install -e ".[dev]"
pre-commit install
ruff check src tests
ruff format --check src tests
make check
make test
```

In VS Code, the same checks are available as tasks under `Terminal > Run Task`.

## Validation

The current codebase has been verified locally with:

- `ruff check src tests`
- `python -m compileall src`
- `python -m unittest discover -s tests`

## CLI

```bash
ml init fraud-detector
```

Track and compare runs:

```bash
ml runs list --run-dir .ml_runs
ml runs compare run-1 run-2 --run-dir .ml_runs
```

This creates a new installable project with:

- `pyproject.toml`
- config files under `configs/`
- a training entry point
- an evaluation entry point
- `Dockerfile`
- `.github/workflows/ci.yml`

## Example usage

```python
from ml_platform.core.config import ExperimentConfig
from ml_platform.tracking.store import LocalRunStore
from ml_platform.evaluation.engine import EvaluationEngine

config = ExperimentConfig(project_name="demo", seed=42)
tracker = LocalRunStore(root_dir=".ml_runs")
engine = EvaluationEngine()
```

## Design philosophy

The toolkit favors a narrow, opinionated API at the platform layer and lightweight extension points in user projects. The goal is not to hide ML implementation details, but to make the surrounding workflow repeatable: reproducible training, comparable experiments, standard reports, and a consistent project layout.
