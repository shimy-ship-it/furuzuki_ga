"""Command-line entry point: python -m tsp_solver."""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from tsp_solver.algorithms import (
    AntColonyOptimizer,
    GeneticAlgorithm,
    GreedyOptimizer,
    ParticleSwarmOptimizer,
)
from tsp_solver.instance import discover_instances, load_instance
from tsp_solver.plotting import plot_convergence, plot_tour

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and optimize TSPLIB TSP instances.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_command = commands.add_parser("list", help="list .tsp files in a dataset directory")
    list_command.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)

    solve_command = commands.add_parser("solve", help="run a selected TSP metaheuristic")
    solve_command.add_argument("instance", type=Path, help="path to a TSPLIB .tsp file")
    solve_command.add_argument(
        "--algorithm", choices=("greedy", "ga", "aco", "pso"), required=True
    )
    solve_command.add_argument(
        "--plot", type=Path, default=None,
        help="save the two-dimensional route image to this path",
    )
    solve_command.add_argument(
        "--show", action="store_true",
        help="open route and convergence images in graphical windows",
    )
    solve_command.add_argument(
        "--convergence", type=Path, default=None,
        help="save the GA best-value convergence image to this path",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        for path in discover_instances(args.dataset_dir):
            print(path)
        return 0

    instance = load_instance(args.instance)
    if args.algorithm == "greedy":
        optimizer = GreedyOptimizer()
    elif args.algorithm == "ga":
        optimizer = GeneticAlgorithm()
    elif args.algorithm == "aco":
        optimizer = AntColonyOptimizer()
    else:
        optimizer = ParticleSwarmOptimizer()

    result = optimizer.solve(instance)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))

    if args.plot is not None or args.show:
        saved_path = plot_tour(instance, result, args.plot, show=args.show)
        if saved_path is not None:
            print("Plot saved to: %s" % saved_path)

    if args.convergence is not None or (args.show and result.algorithm == "ga"):
        saved_path = plot_convergence(result, args.convergence, show=args.show)
        if saved_path is not None:
            print("Convergence plot saved to: %s" % saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())