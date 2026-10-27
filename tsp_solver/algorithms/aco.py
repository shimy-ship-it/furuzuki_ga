"""Ant-colony optimization for the traveling-salesperson problem."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from tsp_solver.instance import Node, TSPInstance
from tsp_solver.result import OptimizationResult


@dataclass(frozen=True)
class ACOConfig:
    ants: int = 30
    iterations: int = 500
    alpha: float = 1.0
    beta: float = 2.0
    evaporation_rate: float = 0.5
    deposit_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.ants < 1 or self.iterations < 1:
            raise ValueError("ACO ants and iterations must be positive.")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("ACO alpha and beta must be non-negative.")
        if not 0 < self.evaporation_rate <= 1:
            raise ValueError("ACO evaporation_rate must be in (0, 1].")
        if self.deposit_factor <= 0:
            raise ValueError("ACO deposit_factor must be positive.")


class AntColonyOptimizer:
    def __init__(self, config: ACOConfig | None = None) -> None:
        self.config = config or ACOConfig()

    def solve(self, instance: TSPInstance, seed: int | None = None) -> OptimizationResult:
        rng = random.Random(seed)
        size = len(instance.nodes)
        node_indices = {node: index for index, node in enumerate(instance.nodes)}
        pheromone = [[1.0] * size for _ in range(size)]
        best_tour: tuple[Node, ...] | None = None
        best_distance = math.inf

        for _ in range(self.config.iterations):
            tours: list[tuple[tuple[Node, ...], float]] = []
            for _ in range(self.config.ants):
                tour = self._construct_tour(instance, pheromone, node_indices, rng)
                distance = instance.tour_length(tour)
                tours.append((tour, distance))
                if distance < best_distance:
                    best_tour, best_distance = tour, distance

            retention = 1.0 - self.config.evaporation_rate
            for row in pheromone:
                for index in range(size):
                    row[index] = max(row[index] * retention, 1e-12)
            for tour, distance in tours:
                deposit = self.config.deposit_factor / max(distance, 1e-12)
                for index, origin in enumerate(tour):
                    destination = tour[(index + 1) % size]
                    i = node_indices[origin]
                    j = node_indices[destination]
                    pheromone[i][j] += deposit

        if best_tour is None:
            raise RuntimeError("ACO completed without constructing a tour.")
        return OptimizationResult(
            algorithm="aco",
            tour=best_tour,
            distance=best_distance,
            iterations=self.config.iterations,
        )

    def _construct_tour(
        self,
        instance: TSPInstance,
        pheromone: list[list[float]],
        node_indices: dict[Node, int],
        rng: random.Random,
    ) -> tuple[Node, ...]:
        nodes = instance.nodes
        start = rng.randrange(len(nodes))
        route = [nodes[start]]
        unvisited = set(nodes)
        unvisited.remove(nodes[start])

        while unvisited:
            current = route[-1]
            current_index = node_indices[current]
            candidates = [node for node in nodes if node in unvisited]
            log_weights = []
            for candidate in candidates:
                pheromone_value = pheromone[current_index][node_indices[candidate]]
                distance = max(instance.distance(current, candidate), 1e-12)
                log_weights.append(
                    self.config.alpha * math.log(pheromone_value)
                    - self.config.beta * math.log(distance)
                )
            offset = max(log_weights)
            weights = [math.exp(value - offset) for value in log_weights]
            selected = rng.choices(candidates, weights=weights, k=1)[0]
            route.append(selected)
            unvisited.remove(selected)
        return tuple(route)