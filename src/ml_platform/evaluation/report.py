from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class LatencySummary:
    samples: int
    average_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass(slots=True)
class ClassificationReport:
    labels: list[str]
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    latency: LatencySummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.latency is None:
            payload["latency"] = None
        return payload

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return destination

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation Report",
            "",
            f"- accuracy: {self.accuracy:.6f}",
            f"- precision_macro: {self.precision_macro:.6f}",
            f"- recall_macro: {self.recall_macro:.6f}",
            f"- f1_macro: {self.f1_macro:.6f}",
            "",
            "## Confusion Matrix",
        ]
        header = "| label | " + " | ".join(self.labels) + " |"
        separator = "|---" * (len(self.labels) + 1) + "|"
        lines.extend([header, separator])
        for label, row in zip(self.labels, self.confusion_matrix):
            lines.append("| " + label + " | " + " | ".join(str(value) for value in row) + " |")
        if self.latency is not None:
            lines.extend(
                [
                    "",
                    "## Latency",
                    f"- samples: {self.latency.samples}",
                    f"- average_ms: {self.latency.average_ms:.3f}",
                    f"- p50_ms: {self.latency.p50_ms:.3f}",
                    f"- p95_ms: {self.latency.p95_ms:.3f}",
                    f"- p99_ms: {self.latency.p99_ms:.3f}",
                ]
            )
        return "\n".join(lines)
