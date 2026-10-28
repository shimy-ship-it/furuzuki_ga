"""TSPLIB instance loading and distance access."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union

import tsplib95

Node = Hashable
WeightFunction = Callable[[Node, Node], Union[int, float]]


@dataclass(frozen=True)
class TSPInstance:
    """A TSPLIB problem with its node labels and edge-weight function."""

    name: str
    nodes: Tuple[Node, ...]
    _weight: WeightFunction
    source: Optional[Path] = None
    coordinates: Optional[Dict[Node, Tuple[float, float]]] = None

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("A TSP instance must contain at least two nodes.")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("TSP node labels must be unique.")

    def distance(self, origin: Node, destination: Node) -> float:
        """Return the TSPLIB-defined distance from origin to destination."""
        weight = float(self._weight(origin, destination))
        if not isfinite(weight) or weight < 0:
            raise ValueError(
                f"Invalid edge weight for {origin!r} -> {destination!r}: {weight}"
            )
        return weight

    def tour_length(self, tour: Union[Tuple[Node, ...], List[Node]]) -> float:
        """Return the length of a closed tour after checking its permutation."""
        if len(tour) != len(self.nodes) or set(tour) != set(self.nodes):
            raise ValueError("A tour must visit every instance node exactly once.")
        return sum(
            self.distance(tour[index], tour[(index + 1) % len(tour)])
            for index in range(len(tour))
        )


def load_instance(path: Union[str, Path]) -> TSPInstance:
    """Load a .tsp file through tsplib95, preserving its distance semantics."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".tsp":
        raise ValueError(f"Expected a .tsp file, got: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"TSPLIB instance not found: {source}")

    problem = tsplib95.load(str(source))
    nodes = tuple(problem.get_nodes())
    raw_coordinates = getattr(problem, "node_coords", None)
    coordinates = None
    if raw_coordinates:
        coordinates = dict(
            (node, (float(raw_coordinates[node][0]), float(raw_coordinates[node][1])))
            for node in nodes
            if node in raw_coordinates
        )
        if len(coordinates) != len(nodes):
            coordinates = None

    return TSPInstance(
        name=problem.name or source.stem,
        nodes=nodes,
        _weight=problem.get_weight,
        source=source,
        coordinates=coordinates,
    )


def discover_instances(directory: Union[str, Path]) -> List[Path]:
    """Return all .tsp files below directory in stable path order."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".tsp"),
        key=lambda path: str(path).lower(),
    )