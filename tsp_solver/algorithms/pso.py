"""Discrete swap-sequence particle swarm optimization for TSP tours."""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tsp_solver.instance import TSPInstance
from tsp_solver.result import OptimizationResult

Swap = Tuple[int, int]


@dataclass(frozen=True)
class PSOConfig:
    particles: int = 40
    iterations: int = 500
    inertia: float = 0.6
    cognitive: float = 1.5
    social: float = 1.5
    max_velocity: int = 20

    def __post_init__(self) -> None:
        if self.particles < 1 or self.iterations < 1:
            raise ValueError("PSO particles and iterations must be positive.")
        if not 0 <= self.inertia <= 1:
            raise ValueError("PSO inertia must be in [0, 1].")
        if self.cognitive < 0 or self.social < 0:
            raise ValueError("PSO cognitive and social coefficients must be non-negative.")
        if self.max_velocity < 1:
            raise ValueError("PSO max_velocity must be positive.")


class ParticleSwarmOptimizer:
    def __init__(self, config: Optional[PSOConfig] = None) -> None:
        self.config = config or PSOConfig()

    def solve(
        self, instance: TSPInstance, seed: Optional[int] = None
    ) -> OptimizationResult:
        rng = random.Random(seed)
        size = len(instance.nodes)
        positions: List[List[int]] = []
        personal_best: List[List[int]] = []
        personal_cost: List[float] = []
        velocities: List[List[Swap]] = []

        for _ in range(self.config.particles):
            position = list(range(size))
            rng.shuffle(position)
            positions.append(position)
            personal_best.append(position.copy())
            personal_cost.append(self._cost(instance, position))
            velocities.append([])

        best_index = min(range(len(personal_cost)), key=personal_cost.__getitem__)
        global_best = personal_best[best_index].copy()
        global_cost = personal_cost[best_index]

        for _ in range(self.config.iterations):
            for particle_index, position in enumerate(positions):
                velocity = [
                    swap
                    for swap in velocities[particle_index]
                    if rng.random() < self.config.inertia
                ]
                if rng.random() < min(self.config.cognitive, 1.0):
                    velocity.extend(self._difference(position, personal_best[particle_index]))
                if rng.random() < min(self.config.social, 1.0):
                    velocity.extend(self._difference(position, global_best))
                velocity = velocity[: self.config.max_velocity]
                self._apply_swaps(position, velocity)
                velocities[particle_index] = velocity

                cost = self._cost(instance, position)
                if cost < personal_cost[particle_index]:
                    personal_best[particle_index] = position.copy()
                    personal_cost[particle_index] = cost
                    if cost < global_cost:
                        global_best, global_cost = position.copy(), cost

        route = tuple(instance.nodes[index] for index in global_best)
        return OptimizationResult(
            algorithm="pso",
            tour=route,
            distance=global_cost,
            iterations=self.config.iterations,
        )

    @staticmethod
    def _cost(instance: TSPInstance, position: List[int]) -> float:
        tour = tuple(instance.nodes[index] for index in position)
        return instance.tour_length(tour)

    @staticmethod
    def _difference(source: List[int], target: List[int]) -> List[Swap]:
        working = source.copy()
        position_of = dict((value, index) for index, value in enumerate(working))
        swaps = []
        for index, target_value in enumerate(target):
            current_index = position_of[target_value]
            if current_index == index:
                continue
            current_value = working[index]
            working[index], working[current_index] = working[current_index], working[index]
            position_of[target_value] = index
            position_of[current_value] = current_index
            swaps.append((index, current_index))
        return swaps

    @staticmethod
    def _apply_swaps(position: List[int], swaps: List[Swap]) -> None:
        for first, second in swaps:
            position[first], position[second] = position[second], position[first]