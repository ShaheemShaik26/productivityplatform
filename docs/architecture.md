# Architecture

## Overview

The toolkit is split into six layers:

1. `core` for shared experiment configuration and serialization.
2. `training` for reusable training-loop orchestration and reproducibility.
3. `tracking` for local or SQLite run persistence and comparison.
4. `evaluation` for standardized classification reports and latency benchmarks.
5. `data` for dataset ingestion and preprocessing pipelines.
6. `scaffold` and `cli` for project generation with `ml init <project_name>`.

## CLI Flow

`ml init <project_name>` creates a project directory containing:

- installable Python packaging metadata
- a reusable config layer
- training and evaluation entry points
- config JSON files
- a Dockerfile
- a GitHub Actions workflow
- a smoke test

## Training Framework

The platform does not hide the model implementation. Instead, it standardizes the parts that should be identical across teams:

- seeding and reproducibility
- epoch orchestration
- metric aggregation
- run record creation and persistence
- consistent output artifacts

The expectation is that teams subclass `BaseTrainer` and implement batch-level hooks for their own model and data shapes.

The generated project scaffold includes a reference PyTorch binary classifier and trainer so teams start from a consistent pattern instead of a blank `train.py`.

## Experiment Tracking

Runs are stored either:

- as JSON files in a local run directory
- in SQLite for teams that want a durable lightweight database

Every run record keeps the config, metrics, tags, runtime, timestamps, and status. That gives teams a common comparison surface across experiments.

The CLI exposes that surface directly with `ml runs list` and `ml runs compare`, which means engineers can inspect and compare experiments without writing ad hoc scripts.

## Evaluation Engine

The evaluation layer produces a single structured artifact for classification jobs:

- metrics
- confusion matrix
- latency summary
- metadata

The report can be emitted as JSON for downstream automation or rendered as Markdown for humans.

## Dataset Loading

Data ingestion uses a loader abstraction plus a reusable preprocessing pipeline. That lets teams compose project-specific transformations without changing the platform API.

## Scaling Across Teams

The design scales by keeping the platform narrow and the project templates opinionated:

- platform code stays stable and versioned centrally
- team-specific logic stays in generated project code
- the interface between teams and the platform is explicit and small
- run tracking and evaluation stay standardized, which makes cross-team comparison possible

This is the same pattern used by large internal ML platforms: centralize workflow mechanics, decentralize modeling details.
