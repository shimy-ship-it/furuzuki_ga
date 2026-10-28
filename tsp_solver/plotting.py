"""Visualization helpers for two-dimensional TSP instances."""

from pathlib import Path
from typing import Optional, Union

from tsp_solver.instance import TSPInstance
from tsp_solver.result import OptimizationResult


def plot_tour(
    instance: TSPInstance,
    result: OptimizationResult,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Draw a tour using the instance's TSPLIB coordinates.

    Returns the saved file path, or None if an output path was not provided.
    """
    if instance.coordinates is None:
        raise ValueError(
            "This TSPLIB instance has no two-dimensional NODE_COORD_SECTION."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install dependencies with "
            "python -m pip install -r requirements.txt"
        ) from error

    closed_tour = result.closed_tour
    xs = []
    ys = []
    for node in closed_tour:
        if node not in instance.coordinates:
            raise ValueError("The result contains a node with no coordinates: %r" % node)
        coordinate = instance.coordinates[node]
        xs.append(coordinate[0])
        ys.append(coordinate[1])

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.plot(xs, ys, "-o", color="#2878B5", linewidth=1.2, markersize=4)

    if closed_tour:
        start_x, start_y = instance.coordinates[closed_tour[0]]
        axis.scatter([start_x], [start_y], color="#E74C3C", s=90, zorder=3, label="Start")
        axis.legend()

    axis.set_title("%s - %s route (length %.3f)" % (
        instance.name,
        result.algorithm.upper(),
        result.distance,
    ))
    axis.set_xlabel("X coordinate")
    axis.set_ylabel("Y coordinate")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path).expanduser().resolve()
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(saved_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return saved_path


def plot_convergence(
    result: OptimizationResult,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot the GA best-distance value recorded for each generation."""
    if result.algorithm.lower() != "ga":
        raise ValueError("Best-value history is only available for GA results.")
    if not result.best_history:
        raise ValueError("The GA result contains no best-value history.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install dependencies with "
            "python -m pip install -r requirements.txt"
        ) from error

    iterations = list(range(1, len(result.best_history) + 1))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(iterations, result.best_history, color="#2878B5", linewidth=1.5)
    axis.scatter(
        [iterations[-1]], [result.best_history[-1]],
        color="#E74C3C", s=45, zorder=3,
        label="Final best: %.3f" % result.best_history[-1],
    )
    axis.set_title("GA convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Best distance")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path).expanduser().resolve()
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(saved_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return saved_path
