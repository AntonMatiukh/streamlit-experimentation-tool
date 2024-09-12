"""Power simulation """

from src.parallel.runner import ParallelRunner
import numpy as np
from typing import Callable, List
import random
from numpy.random import SeedSequence, default_rng


class PowerSimulation:
    def __init__(self, n_iterations=2000, power=0.8, alpha=0.05, greater_or_less='greater'):
        self.n_iterations = n_iterations
        self.power = power
        self.alpha = alpha
        self.greater_or_less = greater_or_less

    def _perform_test_iteration(self, test_function: Callable, control: np.ndarray, test: np.ndarray) -> int:
        """
        1 bootstrapped iteration of the stat test
        """

        control_b = np.random.choice(control, len(control), replace=True)
        test_b = np.random.choice(test, len(test), replace=True)

        if self.greater_or_less == 'greater':
            return 1 if test_function(control=control_b, test=test_b) < self.alpha else 0
        elif self.greater_or_less == 'less':
            return 1 if test_function(control=test_b, test=control_b) <= self.alpha else 0

    def simulate_power(self, test_function: Callable, control: np.ndarray, test: np.ndarray,
                       runner: ParallelRunner = None) -> float:
        """
        Power simulation function
        """

        if runner is None:
            runner = ParallelRunner(n_iterations=self.n_iterations, n_jobs=-1)

        results = runner.run(func=self._perform_test_iteration,
                             test_function=test_function,
                             control=control,
                             test=test
                             )

        return sum(results) / len(results)


