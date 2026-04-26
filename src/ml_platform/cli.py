from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .tracking.store import LocalRunStore, SQLiteRunStore
from .scaffold import create_project_scaffold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml", description="Internal ML engineering toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Generate a new ML project scaffold")
    init_parser.add_argument("project_name", help="Name of the new project")
    init_parser.add_argument("--destination", default=str(Path.cwd()), help="Directory where the project should be created")
    init_parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing project directory")

    runs_parser = subparsers.add_parser("runs", help="Inspect and compare tracked experiments")
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command", required=True)

    runs_list_parser = runs_subparsers.add_parser("list", help="List tracked runs")
    runs_list_parser.add_argument("--project", default=None, help="Filter by project name")
    _add_store_arguments(runs_list_parser)

    runs_compare_parser = runs_subparsers.add_parser("compare", help="Compare tracked runs")
    runs_compare_parser.add_argument("run_ids", nargs="+", help="Run IDs to compare")
    _add_store_arguments(runs_compare_parser)

    return parser


def _add_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", choices=("local", "sqlite"), default="local", help="Tracking backend to use")
    parser.add_argument("--run-dir", default=".ml_runs", help="Directory for local JSON run storage")
    parser.add_argument("--db-path", default=".ml_runs/runs.sqlite3", help="SQLite database path")


def _build_store(args: argparse.Namespace) -> LocalRunStore | SQLiteRunStore:
    if args.store == "sqlite":
        return SQLiteRunStore(db_path=args.db_path)
    return LocalRunStore(root_dir=args.run_dir)


def _print_run_row(run) -> None:
    metrics = json.dumps(run.metrics, sort_keys=True)
    tags = ",".join(run.tags) if run.tags else "-"
    print(f"{run.run_id}\t{run.project_name}\t{run.experiment_name}\t{run.status}\t{run.runtime_seconds:.3f}\t{metrics}\t{tags}")


def handle_init(args: argparse.Namespace) -> int:
    result = create_project_scaffold(
        project_name=args.project_name,
        destination_dir=args.destination,
        overwrite=args.overwrite,
    )
    print(f"Created {result.project_name} at {result.root_dir}")
    for created_file in result.created_files:
        print(f"- {created_file.relative_to(result.root_dir)}")
    return 0


def handle_runs_list(args: argparse.Namespace) -> int:
    store = _build_store(args)
    runs = store.list_runs(project_name=args.project)
    print("run_id\tproject_name\texperiment_name\tstatus\truntime_seconds\tmetrics\ttags")
    for run in runs:
        _print_run_row(run)
    return 0


def handle_runs_compare(args: argparse.Namespace) -> int:
    store = _build_store(args)
    comparison = store.compare_runs(args.run_ids)
    if not comparison.rows:
        print("No matching runs found.")
        return 1
    print(comparison.to_table())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "init": handle_init,
        "runs": {
            "list": handle_runs_list,
            "compare": handle_runs_compare,
        },
    }
    if args.command == "runs":
        return handlers[args.command][args.runs_command](args)
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
