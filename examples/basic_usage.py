from ml_platform.core.config import ExperimentConfig
from ml_platform.data.loader import DatasetLoader, MaterializedDataset, TransformPipeline
from ml_platform.evaluation.engine import EvaluationEngine
from ml_platform.tracking.store import LocalRunStore


class InMemoryLoader(DatasetLoader):
    def load_raw(self, split: str):
        del split
        return MaterializedDataset(features=[[0.1, 0.9], [0.8, 0.2]], targets=[1, 0])


class NormalizeFeatures:
    def transform(self, data):
        return data


config = ExperimentConfig(project_name="demo-project", tags=["example", "smoke"])
tracker = LocalRunStore(root_dir=".ml_runs")
engine = EvaluationEngine()
loader = InMemoryLoader(pipeline=TransformPipeline([NormalizeFeatures()]))

dataset = loader.load("train")
report = engine.classification_report([1, 0], [1, 0], labels=["negative", "positive"], latencies_ms=[4.1, 4.2, 4.3])

config.save("artifacts/demo_config.json")
tracker.list_runs()
report.save_json("artifacts/demo_report.json")
print(dataset)
print(report.to_markdown())
