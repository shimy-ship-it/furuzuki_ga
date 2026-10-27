"""Command-line entry point: python -m tsp_solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsp_solver.algorithms import (
    AntColonyOptimizer,
    GeneticAlgorithm,
    ParticleSwarmOptimizer,
)
from tsp_solver.instance import discover_instances, load_instance

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and optimize TSPLIB TSP instances.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_command = commands.add_parser("list", help="list .tsp files in a dataset directory")
    list_command.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)

    solve_command = commands.add_parser("solve", help="run a selected TSP metaheuristic")
    solve_command.add_argument("instance", type=Path, help="path to a TSPLIB .tsp file")
    solve_command.add_argument("--algorithm", choices=("ga", "aco", "pso"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        for path in discover_instances(args.dataset_dir):
            print(path)
        return 0

    instance = load_instance(args.instance)
    if args.algorithm == "ga":
        optimizer = GeneticAlgorithm()
    elif args.algorithm == "aco":
        optimizer = AntColonyOptimizer()
    else:
        optimizer = ParticleSwarmOptimizer()

    result = optimizer.solve(instance)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())