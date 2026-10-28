"""Multi-start nearest-neighbor greedy algorithm for TSP."""

from typing import Optional

from tsp_solver.instance import Node, TSPInstance
from tsp_solver.result import OptimizationResult


class GreedyOptimizer:
    """Build a nearest-neighbor tour from each node and keep the shortest."""

    def solve(
        self, instance: TSPInstance, start_node: Optional[Node] = None
    ) -> OptimizationResult:
        if start_node is not None and start_node not in instance.nodes:
            raise ValueError("start_node must belong to the TSP instance.")

        starts = (start_node,) if start_node is not None else instance.nodes
        best_tour = None
        best_distance = None

        for start in starts:
            unvisited = set(instance.nodes)
            unvisited.remove(start)
            tour = [start]

            while unvisited:
                current = tour[-1]
                next_node = min(
                    unvisited,
                    key=lambda node: instance.distance(current, node),
                )
                tour.append(next_node)
                unvisited.remove(next_node)

            distance = instance.tour_length(tour)
            if best_distance is None or distance < best_distance:
                best_tour = tuple(tour)
                best_distance = distance

        if best_tour is None or best_distance is None:
            raise RuntimeError("Greedy algorithm did not construct a tour.")

        return OptimizationResult(
            algorithm="greedy",
            tour=best_tour,
            distance=best_distance,
            iterations=len(starts),
        )
