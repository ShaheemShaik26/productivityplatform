from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_platform.core.config import ExperimentConfig
from ml_platform.data.loader import TransformPipeline
from ml_platform.evaluation.engine import EvaluationEngine
from ml_platform.tracking.models import RunRecord
from ml_platform.tracking.store import LocalRunStore
from ml_platform.cli import main as cli_main
from ml_platform.scaffold import create_project_scaffold


@dataclass
class IncrementTransformer:
    offset: int = 1
    fitted: bool = False

    def fit(self, data):
        self.fitted = True
        return self

    def transform(self, data):
        return [value + self.offset for value in data]


class ToolkitSmokeTest(unittest.TestCase):
    def test_config_round_trip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(project_name="demo", tags=["smoke"])
            path = Path(temp_dir) / "config.json"
            config.save(path)
            loaded = ExperimentConfig.load(path)
            self.assertEqual(loaded.project_name, "demo")
            self.assertEqual(loaded.tags, ["smoke"])

    def test_pipeline_retains_fitted_state(self) -> None:
        pipeline = TransformPipeline([IncrementTransformer(offset=2)])
        result = pipeline.fit_transform([1, 2, 3])
        self.assertEqual(result, [3, 4, 5])
        self.assertTrue(pipeline.steps[0].fitted)

    def test_evaluation_report_generation(self) -> None:
        engine = EvaluationEngine()
        report = engine.classification_report([0, 1, 1], [0, 1, 0], labels=["zero", "one"], latencies_ms=[1.0, 2.0, 3.0])
        self.assertAlmostEqual(report.accuracy, 2 / 3)
        self.assertEqual(report.confusion_matrix, [[1, 0], [1, 1]])
        self.assertIsNotNone(report.latency)

    def test_scaffold_generation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            result = create_project_scaffold("Demo Project", destination_dir=temp_dir)
            self.assertTrue((result.root_dir / "pyproject.toml").exists())
            self.assertTrue((result.root_dir / "src" / "demo_project" / "model.py").exists())
            self.assertTrue((result.root_dir / "src" / "demo_project" / "trainer.py").exists())
            self.assertTrue((result.root_dir / "src" / "demo_project" / "train.py").exists())
            self.assertTrue((result.root_dir / ".github" / "workflows" / "ci.yml").exists())

    def test_runs_cli_lists_and_compares(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = LocalRunStore(root_dir=Path(temp_dir))
            first = RunRecord(
                run_id="run-1",
                project_name="demo",
                experiment_name="baseline",
                created_at="2026-04-26T00:00:00Z",
                seed=1,
                config={"project_name": "demo"},
                metrics={"accuracy": 0.9},
                runtime_seconds=10.0,
                status="completed",
                tags=["baseline"],
            )
            second = RunRecord(
                run_id="run-2",
                project_name="demo",
                experiment_name="candidate",
                created_at="2026-04-26T00:01:00Z",
                seed=2,
                config={"project_name": "demo"},
                metrics={"accuracy": 0.95},
                runtime_seconds=12.0,
                status="completed",
                tags=["candidate"],
            )
            store.save_run(first)
            store.save_run(second)

            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = cli_main(["runs", "list", "--run-dir", temp_dir])
            self.assertEqual(exit_code, 0)
            output = buffer.getvalue()
            self.assertIn("run_id\tproject_name\texperiment_name", output)
            self.assertIn("run-1", output)

            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = cli_main(["runs", "compare", "run-1", "run-2", "--run-dir", temp_dir])
            self.assertEqual(exit_code, 0)
            comparison_output = buffer.getvalue()
            self.assertIn("baseline", comparison_output)
            self.assertIn("candidate", comparison_output)


if __name__ == "__main__":
    unittest.main()
