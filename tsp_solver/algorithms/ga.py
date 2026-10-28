"""Permutation genetic algorithm for the traveling-salesperson problem."""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tsp_solver.instance import Node, TSPInstance
from tsp_solver.result import OptimizationResult


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    generations: int = 100
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02
    elite_size: int = 2
    tournament_size: int = 3
    local_search_fraction: float = 0.10
    max_depth: int = 1  # Maximum number of iterations for 2-Opt local search

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.generations < 1:
            raise ValueError("GA population_size must be >= 2 and generations >= 1.")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("GA elite_size must be between 0 and population_size - 1.")
        if self.tournament_size < 1:
            raise ValueError("GA tournament_size must be positive.")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("GA crossover_rate and mutation_rate must be in [0, 1].")
        if not 0 <= self.local_search_fraction <= 1:
            raise ValueError("GA local_search_fraction must be between 0 and 1.")


class GeneticAlgorithm:
    def __init__(self, config: Optional[GAConfig] = None) -> None:
        self.config = config or GAConfig()

    def solve(
        self, instance: TSPInstance, seed: Optional[int] = None
    ) -> OptimizationResult:
        rng = random.Random(seed)
        population = []
        for _ in range(self.config.population_size):
            candidate = list(instance.nodes)
            rng.shuffle(candidate)
            population.append(candidate)

        best_tour = min(population, key=instance.tour_length)
        best_distance = instance.tour_length(best_tour)
        best_history = []

        for _ in range(self.config.generations):
            print(f"Generation {_ + 1}/{self.config.generations}: Best distance = {best_distance:.4f}")
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

            if self.config.local_search_fraction > 0:
                # population.sort(key=instance.tour_length)
                rng.shuffle(population)
                local_search_count = max(
                    1,
                    int(math.ceil(len(population) * self.config.local_search_fraction)),
                )
                for index in range(local_search_count):
                    population[index] = self._two_opt(
                        population[index], instance, self.config.max_depth
                    )
                
            generation_distance, generation_tour = min(
                ((instance.tour_length(tour), tour) for tour in population),
                key=lambda item: item[0],
            )
            if generation_distance < best_distance:
                best_distance, best_tour = generation_distance, generation_tour.copy()
            best_history.append(best_distance)

        final_distance, final_tour = min(
            ((instance.tour_length(tour), tour) for tour in population),
            key=lambda item: item[0],
        )
        if final_distance < best_distance:
            best_distance, best_tour = final_distance, final_tour
            if best_history:
                best_history[-1] = best_distance
        return OptimizationResult(
            algorithm="ga",
            tour=tuple(best_tour),
            distance=best_distance,
            iterations=self.config.generations,
            best_history=tuple(best_history),
        )

    @staticmethod
    def _two_opt(
        tour: List[Node], instance: TSPInstance, max_depth: int
    ) -> List[Node]:
        """Apply best-improvement 2-Opt until no improving reversal remains."""
        optimized = tour.copy()
        current_distance = instance.tour_length(optimized)

        for _ in range(max_depth):
            best_distance = current_distance
            best_tour = optimized

            for left in range(len(optimized) - 1):
                for right in range(left + 1, len(optimized)):
                    candidate = (
                        optimized[:left]
                        + list(reversed(optimized[left:right + 1]))
                        + optimized[right + 1:]
                    )
                    candidate_distance = instance.tour_length(candidate)
                    if candidate_distance < best_distance:
                        best_distance = candidate_distance
                        best_tour = candidate

            if best_distance >= current_distance:
                return optimized
            optimized = best_tour
            current_distance = best_distance

        return optimized

    def _tournament(
        self,
        scored: List[Tuple[float, List[Node]]],
        rng: random.Random,
    ) -> List[Node]:
        contestants = rng.sample(
            scored, k=min(self.config.tournament_size, len(scored))
        )
        return min(contestants, key=lambda item: item[0])[1]

    @staticmethod
    def _order_crossover(
        parent_a: List[Node], parent_b: List[Node], rng: random.Random
    ) -> Tuple[List[Node], List[Node]]:
        size = len(parent_a)
        left, right = sorted((rng.randrange(size), rng.randrange(size)))

        def make_child(first: List[Node], second: List[Node]) -> List[Node]:
            child = [None] * size
            child[left : right + 1] = first[left : right + 1]
            remaining = [node for node in second if node not in child]
            positions = list(range(right + 1, size)) + list(range(left))
            for position, node in zip(positions, remaining):
                child[position] = node
            return [node for node in child if node is not None]

        return make_child(parent_a, parent_b), make_child(parent_b, parent_a)

    def _mutate(self, tour: List[Node], rng: random.Random) -> None:
        if len(tour) > 1 and rng.random() < self.config.mutation_rate:
            first, second = rng.sample(range(len(tour)), 2)
            tour[first], tour[second] = tour[second], tour[first]