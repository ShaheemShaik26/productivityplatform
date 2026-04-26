from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    """Standard experiment configuration shared across ML projects."""

    project_name: str
    experiment_name: str = "default"
    seed: int = 42
    output_dir: str = "artifacts"
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(
        default_factory=lambda: {
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
    )
    evaluation: dict[str, Any] = field(default_factory=dict)
    tracking: dict[str, Any] = field(
        default_factory=lambda: {"backend": "local", "run_dir": ".ml_runs"}
    )
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(payload)
