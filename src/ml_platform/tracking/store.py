from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable
import json
import sqlite3

from .models import RunComparison, RunComparisonRow, RunRecord


class RunStore(ABC):
    @abstractmethod
    def save_run(self, run: RunRecord) -> RunRecord:
        raise NotImplementedError

    @abstractmethod
    def get_run(self, run_id: str) -> RunRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_runs(self, project_name: str | None = None) -> list[RunRecord]:
        raise NotImplementedError

    def compare_runs(self, run_ids: Iterable[str]) -> RunComparison:
        rows: list[RunComparisonRow] = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is not None:
                rows.append(
                    RunComparisonRow(
                        run_id=run.run_id,
                        experiment_name=run.experiment_name,
                        metrics=run.metrics,
                        runtime_seconds=run.runtime_seconds,
                    )
                )
        return RunComparison(rows=rows)


class LocalRunStore(RunStore):
    def __init__(self, root_dir: str | Path = ".ml_runs") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.root_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run: RunRecord) -> RunRecord:
        payload = json.dumps(run.to_dict(), indent=2, sort_keys=True)
        (self.runs_dir / f"{run.run_id}.json").write_text(payload, encoding="utf-8")
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_runs(self, project_name: str | None = None) -> list[RunRecord]:
        records: list[RunRecord] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            record = RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if project_name is None or record.project_name == project_name:
                records.append(record)
        return records


class SQLiteRunStore(RunStore):
    def __init__(self, db_path: str | Path = ".ml_runs/runs.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    experiment_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    runtime_seconds REAL NOT NULL,
                    status TEXT NOT NULL,
                    tags_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save_run(self, run: RunRecord) -> RunRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, project_name, experiment_name, created_at, seed, config_json, metrics_json, runtime_seconds, status, tags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.project_name,
                    run.experiment_name,
                    run.created_at,
                    run.seed,
                    json.dumps(run.config, sort_keys=True),
                    json.dumps(run.metrics, sort_keys=True),
                    float(run.runtime_seconds),
                    run.status,
                    json.dumps(run.tags),
                ),
            )
            connection.commit()
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_runs(self, project_name: str | None = None) -> list[RunRecord]:
        query = "SELECT * FROM runs"
        parameters: tuple[str, ...] = ()
        if project_name is not None:
            query += " WHERE project_name = ?"
            parameters = (project_name,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row | tuple[object, ...]) -> RunRecord:
        return RunRecord(
            run_id=row[0],
            project_name=row[1],
            experiment_name=row[2],
            created_at=row[3],
            seed=int(row[4]),
            config=json.loads(row[5]),
            metrics=json.loads(row[6]),
            runtime_seconds=float(row[7]),
            status=row[8],
            tags=json.loads(row[9]),
        )
