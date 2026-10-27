"""Common result type returned by each optimization algorithm."""

from dataclasses import dataclass
from typing import Any

from tsp_solver.instance import Node


@dataclass(frozen=True)
class OptimizationResult:
    algorithm: str
    tour: tuple[Node, ...]
    distance: float
    iterations: int

    @property
    def closed_tour(self) -> tuple[Node, ...]:
        """Return the tour with its starting node repeated at the end."""
        return self.tour + self.tour[:1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "tour": list(self.tour),
            "closed_tour": list(self.closed_tour),
            "distance": self.distance,
            "iterations": self.iterations,
        }