import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from tsp_solver.algorithms import (
    ACOConfig,
    AntColonyOptimizer,
    GAConfig,
    GeneticAlgorithm,
    GreedyOptimizer,
    PSOConfig,
    ParticleSwarmOptimizer,
)
from tsp_solver.instance import TSPInstance, discover_instances, load_instance
from tsp_solver.plotting import plot_convergence, plot_tour

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

    return TSPInstance(
        "square",
        tuple(coordinates),
        distance,
        coordinates=dict((node, coordinate) for node, coordinate in coordinates.items()),
    )


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

    def test_loads_two_dimensional_coordinates(self) -> None:
        instance = load_instance(ROOT / "dataset" / "berlin52.tsp")
        self.assertIsNotNone(instance.coordinates)
        self.assertEqual(instance.coordinates[1], (565.0, 575.0))

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
    def test_plot_tour_saves_image(self) -> None:
        instance = square_instance()
        result = GreedyOptimizer().solve(instance)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "route.png"
            saved_path = plot_tour(instance, result, output_path)
            self.assertEqual(saved_path, output_path.resolve())
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_rejects_instance_without_coordinates(self) -> None:
        instance = load_instance(ROOT / "dataset" / "gr17.tsp")
        result = GreedyOptimizer().solve(instance)
        with self.assertRaises(ValueError):
            plot_tour(instance, result)

    def test_greedy_finds_square_tour_and_accepts_start_node(self) -> None:
        instance = square_instance()

        result = GreedyOptimizer().solve(instance)
        fixed_start_result = GreedyOptimizer().solve(instance, start_node=1)

        self.assertEqual(result.distance, 4.0)
        self.assertEqual(result.distance, instance.tour_length(result.tour))
        self.assertEqual(set(result.tour), set(instance.nodes))
        self.assertEqual(fixed_start_result.tour[0], 1)
        self.assertEqual(fixed_start_result.distance, 4.0)

    def test_greedy_rejects_unknown_start_node(self) -> None:
        with self.assertRaises(ValueError):
            GreedyOptimizer().solve(square_instance(), start_node=99)

    def test_ga_two_opt_improves_or_preserves_a_tour(self) -> None:
        instance = square_instance()
        tour = [1, 3, 2, 4]

        optimized = GeneticAlgorithm._two_opt(tour, instance)

        self.assertEqual(set(optimized), set(instance.nodes))
        self.assertLessEqual(
            instance.tour_length(optimized), instance.tour_length(tour)
        )
        self.assertEqual(instance.tour_length(optimized), 4.0)

    def test_ga_rejects_invalid_local_search_fraction(self) -> None:
        with self.assertRaises(ValueError):
            GAConfig(local_search_fraction=1.5)

    def test_ga_records_one_best_value_per_iteration(self) -> None:
        result = GeneticAlgorithm(
            GAConfig(population_size=8, generations=3, elite_size=1)
        ).solve(square_instance(), seed=11)

        self.assertEqual(len(result.best_history), 3)
        self.assertEqual(result.best_history[-1], result.distance)
        self.assertEqual(
            result.as_dict()["best_history"], list(result.best_history)
        )
        for previous, current in zip(result.best_history, result.best_history[1:]):
            self.assertLessEqual(current, previous)

    def test_plot_convergence_saves_image(self) -> None:
        result = GeneticAlgorithm(
            GAConfig(population_size=8, generations=3, elite_size=1)
        ).solve(square_instance(), seed=11)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "convergence.png"
            saved_path = plot_convergence(result, output_path)
            self.assertEqual(saved_path, output_path.resolve())
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

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