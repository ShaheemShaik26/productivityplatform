from __future__ import annotations

from statistics import mean
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

from .report import ClassificationReport, LatencySummary


class EvaluationEngine:
    """Create standardized evaluation reports for ML projects."""

    def classification_report(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        labels: Sequence[str] | None = None,
        latencies_ms: Iterable[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClassificationReport:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        label_values = sorted(set(y_true) | set(y_pred))
        if labels is None:
            labels = [str(value) for value in label_values]

        matrix = self._confusion_matrix(y_true, y_pred, label_values)
        precision_macro, recall_macro, f1_macro = self._macro_metrics(matrix)
        accuracy = sum(
            int(true == pred) for true, pred in zip(y_true, y_pred, strict=False)
        ) / max(len(y_true), 1)
        latency = self._latency_summary(latencies_ms) if latencies_ms is not None else None
        return ClassificationReport(
            labels=list(labels),
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            confusion_matrix=matrix,
            latency=latency,
            metadata=metadata or {},
        )

    def benchmark_latency(
        self,
        fn: Callable[[], Any],
        samples: int = 100,
        warmup: int = 10,
    ) -> LatencySummary:
        durations: list[float] = []
        for _ in range(warmup):
            fn()
        for _ in range(samples):
            started = perf_counter()
            fn()
            durations.append((perf_counter() - started) * 1000.0)
        return self._latency_summary(durations)

    def _confusion_matrix(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        label_values: Sequence[int],
    ) -> list[list[int]]:
        index_by_label = {label: index for index, label in enumerate(label_values)}
        matrix = [[0 for _ in label_values] for _ in label_values]
        for true_label, predicted_label in zip(y_true, y_pred, strict=False):
            matrix[index_by_label[true_label]][index_by_label[predicted_label]] += 1
        return matrix

    def _macro_metrics(self, matrix: list[list[int]]) -> tuple[float, float, float]:
        precisions: list[float] = []
        recalls: list[float] = []
        f1_scores: list[float] = []
        for index, row in enumerate(matrix):
            tp = row[index]
            fp = sum(
                matrix[row_index][index]
                for row_index in range(len(matrix))
                if row_index != index
            )
            fn = sum(value for column_index, value in enumerate(row) if column_index != index)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        return (
            mean(precisions) if precisions else 0.0,
            mean(recalls) if recalls else 0.0,
            mean(f1_scores) if f1_scores else 0.0,
        )

    def _latency_summary(self, latencies_ms: Iterable[float]) -> LatencySummary:
        samples = sorted(float(value) for value in latencies_ms)
        if not samples:
            return LatencySummary(samples=0, average_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0)
        return LatencySummary(
            samples=len(samples),
            average_ms=mean(samples),
            p50_ms=self._percentile(samples, 50),
            p95_ms=self._percentile(samples, 95),
            p99_ms=self._percentile(samples, 99),
        )

    def _percentile(self, samples: list[float], percentile: float) -> float:
        if len(samples) == 1:
            return samples[0]
        rank = (percentile / 100.0) * (len(samples) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(samples) - 1)
        if lower == upper:
            return samples[lower]
        weight = rank - lower
        return samples[lower] * (1 - weight) + samples[upper] * weight
