from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, TypeVar

T = TypeVar("T")


class SupportsTransform(Protocol[T]):
    def transform(self, data: T) -> T:
        ...


@dataclass(slots=True)
class MaterializedDataset:
    features: list[Any]
    targets: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TransformPipeline:
    """Reusable preprocessing pipeline for datasets."""

    def __init__(
        self,
        steps: Iterable[Callable[[Any], Any] | SupportsTransform[Any]] | None = None,
    ) -> None:
        self.steps = list(steps or [])

    def fit(self, data: Any) -> "TransformPipeline":
        current = data
        for index, step in enumerate(self.steps):
            if hasattr(step, "fit"):
                maybe_fitted = step.fit(current)  # type: ignore[attr-defined]
                if maybe_fitted is not None:
                    step = maybe_fitted
                    self.steps[index] = step
            if hasattr(step, "transform"):
                current = step.transform(current)  # type: ignore[attr-defined]
            else:
                current = step(current)  # type: ignore[operator]
        return self

    def transform(self, data: Any) -> Any:
        current = data
        for step in self.steps:
            if hasattr(step, "transform"):
                current = step.transform(current)  # type: ignore[attr-defined]
            else:
                current = step(current)  # type: ignore[operator]
        return current

    def fit_transform(self, data: Any) -> Any:
        self.fit(data)
        return self.transform(data)


class DatasetLoader(ABC):
    """Unified ingestion interface used across ML teams."""

    def __init__(self, pipeline: TransformPipeline | None = None) -> None:
        self.pipeline = pipeline or TransformPipeline()

    def load(self, split: str = "train") -> MaterializedDataset:
        raw = self.load_raw(split)
        processed = (
            self.pipeline.fit_transform(raw)
            if split == "train"
            else self.pipeline.transform(raw)
        )
        return self.materialize(processed, split)

    @abstractmethod
    def load_raw(self, split: str) -> Any:
        raise NotImplementedError

    def materialize(self, data: Any, split: str) -> MaterializedDataset:
        del split
        if isinstance(data, MaterializedDataset):
            return data
        if isinstance(data, dict) and "features" in data:
            return MaterializedDataset(
                features=list(data["features"]),
                targets=list(data.get("targets", [])) if data.get("targets") is not None else None,
                metadata={k: v for k, v in data.items() if k not in {"features", "targets"}},
            )
        if isinstance(data, tuple) and len(data) == 2:
            return MaterializedDataset(features=list(data[0]), targets=list(data[1]))
        return MaterializedDataset(features=list(data))
