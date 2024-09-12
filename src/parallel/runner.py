"""Class to parallelize"""

from joblib import Parallel, delayed


class ParallelRunner:
    def __init__(self, n_iterations=2000, n_jobs=-1):
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs

    def run(self, func, **kwargs):
        """
        Parallel calculations

        :param func: Custom function
        :param kwargs: Function arguments
        :return: Results
        """

        # seed_seq = np.random.SeedSequence()
        # seeds = seed_seq.spawn(self.n_iterations)
        # rng = np.random.default_rng(seed)

        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(**kwargs) for _ in range(self.n_iterations)
        )
