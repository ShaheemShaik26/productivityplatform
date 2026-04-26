from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Mapping

from ..core.config import ExperimentConfig
from ..tracking.models import RunRecord
from ..tracking.store import RunStore
from .reproducibility import seed_everything


@dataclass(slots=True)
class EpochResult:
    epoch: int
    metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass(slots=True)
class TrainingResult:
    run: RunRecord
    epochs: list[EpochResult]
    metrics: dict[str, float]
    runtime_seconds: float


class BaseTrainer(ABC):
    """Reusable training loop abstraction for ML experiments.

    Subclasses implement batch-level hooks while the platform manages seeding,
    metric aggregation, persistence, and run bookkeeping.
    """

    def __init__(self, config: ExperimentConfig, tracker: RunStore | None = None) -> None:
        self.config = config
        self.tracker = tracker
        self.state: dict[str, Any] = {}

    def fit(
        self,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any] | None = None,
    ) -> TrainingResult:
        seed_everything(self.config.seed)
        started = perf_counter()
        run = self._create_run_record()
        epochs: list[EpochResult] = []
        last_metrics: dict[str, float] = {}

        total_epochs = int(self.config.training.get("epochs", 1))
        for epoch_index in range(total_epochs):
            epoch_started = perf_counter()
            train_metrics = self.train_epoch(train_loader, epoch_index)
            metrics = dict(train_metrics)
            if val_loader is not None:
                metrics.update(self.evaluate_epoch(val_loader, epoch_index))
            duration = perf_counter() - epoch_started
            epochs.append(
                EpochResult(
                    epoch=epoch_index + 1,
                    metrics=metrics,
                    duration_seconds=duration,
                )
            )
            last_metrics = metrics
            self.after_epoch(epoch_index + 1, metrics)

        runtime_seconds = perf_counter() - started
        run.metrics = last_metrics
        run.runtime_seconds = runtime_seconds
        if self.tracker is not None:
            self.tracker.save_run(run)
        return TrainingResult(
            run=run,
            epochs=epochs,
            metrics=last_metrics,
            runtime_seconds=runtime_seconds,
        )

    def train_epoch(self, train_loader: Iterable[Any], epoch_index: int) -> dict[str, float]:
        return self._aggregate_batch_metrics(
            self.train_step(batch, epoch_index) for batch in train_loader
        )

    def evaluate_epoch(self, val_loader: Iterable[Any], epoch_index: int) -> dict[str, float]:
        return self._aggregate_batch_metrics(
            self.evaluate_step(batch, epoch_index) for batch in val_loader
        )

    def after_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        del epoch, metrics

    def _aggregate_batch_metrics(
        self,
        batch_metrics: Iterable[Mapping[str, float]],
    ) -> dict[str, float]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for metrics in batch_metrics:
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        return {key: totals[key] / counts[key] for key in totals}

    def _create_run_record(self) -> RunRecord:
        return RunRecord.from_config(self.config)

    @abstractmethod
    def train_step(self, batch: Any, epoch_index: int) -> Mapping[str, float]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_step(self, batch: Any, epoch_index: int) -> Mapping[str, float]:
        raise NotImplementedError
