from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.config import ExperimentConfig


@dataclass(slots=True)
class RunRecord:
    run_id: str
    project_name: str
    experiment_name: str
    created_at: str
    seed: int
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    status: str = "created"
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "RunRecord":
        return cls(
            run_id=str(uuid.uuid4()),
            project_name=config.project_name,
            experiment_name=config.experiment_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            seed=config.seed,
            config=config.to_dict(),
            tags=list(config.tags),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        return cls(**payload)


@dataclass(slots=True)
class RunComparisonRow:
    run_id: str
    experiment_name: str
    metrics: dict[str, float]
    runtime_seconds: float


@dataclass(slots=True)
class RunComparison:
    rows: list[RunComparisonRow]

    def to_table(self, metric_keys: list[str] | None = None) -> str:
        metric_keys = metric_keys or sorted({key for row in self.rows for key in row.metrics})
        header = ["run_id", "experiment_name", *metric_keys, "runtime_seconds"]
        lines = [" | ".join(header), " | ".join(["---"] * len(header))]
        for row in self.rows:
            values = [row.run_id, row.experiment_name]
            values.extend(f"{row.metrics.get(metric_key, 0.0):.6f}" for metric_key in metric_keys)
            values.append(f"{row.runtime_seconds:.3f}")
            lines.append(" | ".join(values))
        return "\n".join(lines)
