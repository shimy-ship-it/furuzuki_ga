import unittest
from pathlib import Path

from tsp_solver.algorithms import (
    ACOConfig,
    AntColonyOptimizer,
    GAConfig,
    GeneticAlgorithm,
    PSOConfig,
    ParticleSwarmOptimizer,
)
from tsp_solver.instance import TSPInstance, discover_instances, load_instance

ROOT = Path(__file__).resolve().parents[1]


def square_instance() -> TSPInstance:
    coordinates = {
        1: (0, 0),
        2: (1, 0),
        3: (1, 1),
        4: (0, 1),
    }

    def distance(origin: int, destination: int) -> float:
        x1, y1 = coordinates[origin]
        x2, y2 = coordinates[destination]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    return TSPInstance("square", tuple(coordinates), distance)


class InstanceTests(unittest.TestCase):
    def test_loads_coordinate_and_explicit_tsplib_instances(self) -> None:
        coordinate_instance = load_instance(ROOT / "dataset" / "berlin52.tsp")
        explicit_instance = load_instance(ROOT / "dataset" / "gr17.tsp")

        self.assertEqual(len(coordinate_instance.nodes), 52)
        self.assertEqual(coordinate_instance.distance(1, 2), 666.0)
        self.assertEqual(len(explicit_instance.nodes), 17)
        self.assertEqual(
            explicit_instance.distance(
                explicit_instance.nodes[0], explicit_instance.nodes[1]
            ),
            633.0,
        )

    def test_discovery_only_returns_tsp_files(self) -> None:
        paths = discover_instances(ROOT / "dataset")
        self.assertTrue(paths)
        self.assertTrue(all(path.suffix.lower() == ".tsp" for path in paths))
        self.assertFalse(any(path.name.endswith(".opt.tour") for path in paths))

    def test_tour_length_requires_a_node_permutation(self) -> None:
        instance = square_instance()
        self.assertEqual(instance.tour_length([1, 2, 3, 4]), 4.0)
        with self.assertRaises(ValueError):
            instance.tour_length([1, 2, 2, 4])


class AlgorithmTests(unittest.TestCase):
    def test_each_algorithm_returns_a_valid_tour(self) -> None:
        instance = square_instance()
        algorithms = (
            GeneticAlgorithm(
                GAConfig(
                    population_size=8,
                    generations=3,
                    elite_size=1,
                    tournament_size=2,
                )
            ),
            AntColonyOptimizer(ACOConfig(ants=4, iterations=3)),
            ParticleSwarmOptimizer(
                PSOConfig(particles=6, iterations=3, max_velocity=8)
            ),
        )

        for optimizer in algorithms:
            with self.subTest(algorithm=type(optimizer).__name__):
                result = optimizer.solve(instance, seed=11)
                self.assertEqual(set(result.tour), set(instance.nodes))
                self.assertEqual(len(result.tour), len(instance.nodes))
                self.assertEqual(result.distance, instance.tour_length(result.tour))
                self.assertEqual(result.closed_tour[0], result.closed_tour[-1])


if __name__ == "__main__":
    unittest.main()