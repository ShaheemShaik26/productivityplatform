"""Internal ML engineering toolkit."""

from .core.config import ExperimentConfig
from .data.loader import DatasetLoader, MaterializedDataset, TransformPipeline
from .evaluation.engine import EvaluationEngine
from .tracking.models import RunComparison, RunRecord
from .tracking.store import LocalRunStore, SQLiteRunStore
from .training.base import BaseTrainer, TrainingResult

__all__ = [
    "BaseTrainer",
    "DatasetLoader",
    "EvaluationEngine",
    "ExperimentConfig",
    "LocalRunStore",
    "MaterializedDataset",
    "RunComparison",
    "RunRecord",
    "SQLiteRunStore",
    "TrainingResult",
    "TransformPipeline",
]

__version__ = "0.1.0"
