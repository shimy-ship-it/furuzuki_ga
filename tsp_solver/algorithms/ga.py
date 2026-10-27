"""Permutation genetic algorithm for the traveling-salesperson problem."""

from __future__ import annotations

import random
from dataclasses import dataclass

from tsp_solver.instance import Node, TSPInstance
from tsp_solver.result import OptimizationResult


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    generations: int = 500
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02
    elite_size: int = 2
    tournament_size: int = 3

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.generations < 1:
            raise ValueError("GA population_size must be >= 2 and generations >= 1.")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("GA elite_size must be between 0 and population_size - 1.")
        if self.tournament_size < 1:
            raise ValueError("GA tournament_size must be positive.")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("GA crossover_rate and mutation_rate must be in [0, 1].")


class GeneticAlgorithm:
    def __init__(self, config: GAConfig | None = None) -> None:
        self.config = config or GAConfig()

    def solve(self, instance: TSPInstance, seed: int | None = None) -> OptimizationResult:
        rng = random.Random(seed)
        population = []
        for _ in range(self.config.population_size):
            candidate = list(instance.nodes)
            rng.shuffle(candidate)
            population.append(candidate)

        best_tour = min(population, key=instance.tour_length)
        best_distance = instance.tour_length(best_tour)

        for _ in range(self.config.generations):
            scored = sorted(
                ((instance.tour_length(tour), tour) for tour in population),
                key=lambda item: item[0],
            )
            if scored[0][0] < best_distance:
                best_distance, best_tour = scored[0][0], scored[0][1].copy()

            next_population = [tour.copy() for _, tour in scored[: self.config.elite_size]]
            while len(next_population) < self.config.population_size:
                parent_a = self._tournament(scored, rng)
                parent_b = self._tournament(scored, rng)
                if rng.random() < self.config.crossover_rate:
                    child_a, child_b = self._order_crossover(parent_a, parent_b, rng)
                else:
                    child_a, child_b = parent_a.copy(), parent_b.copy()
                self._mutate(child_a, rng)
                self._mutate(child_b, rng)
                next_population.append(child_a)
                if len(next_population) < self.config.population_size:
                    next_population.append(child_b)
            population = next_population

        final_distance, final_tour = min(
            ((instance.tour_length(tour), tour) for tour in population),
            key=lambda item: item[0],
        )
        if final_distance < best_distance:
            best_distance, best_tour = final_distance, final_tour
        return OptimizationResult(
            algorithm="ga",
            tour=tuple(best_tour),
            distance=best_distance,
            iterations=self.config.generations,
        )

    def _tournament(
        self,
        scored: list[tuple[float, list[Node]]],
        rng: random.Random,
    ) -> list[Node]:
        contestants = rng.sample(
            scored, k=min(self.config.tournament_size, len(scored))
        )
        return min(contestants, key=lambda item: item[0])[1]

    @staticmethod
    def _order_crossover(
        parent_a: list[Node], parent_b: list[Node], rng: random.Random
    ) -> tuple[list[Node], list[Node]]:
        size = len(parent_a)
        left, right = sorted((rng.randrange(size), rng.randrange(size)))

        def make_child(first: list[Node], second: list[Node]) -> list[Node]:
            child: list[Node | None] = [None] * size
            child[left : right + 1] = first[left : right + 1]
            remaining = [node for node in second if node not in child]
            positions = list(range(right + 1, size)) + list(range(left))
            for position, node in zip(positions, remaining):
                child[position] = node
            return [node for node in child if node is not None]

        return make_child(parent_a, parent_b), make_child(parent_b, parent_a)

    def _mutate(self, tour: list[Node], rng: random.Random) -> None:
        if len(tour) > 1 and rng.random() < self.config.mutation_rate:
            first, second = rng.sample(range(len(tour)), 2)
            tour[first], tour[second] = tour[second], tour[first]