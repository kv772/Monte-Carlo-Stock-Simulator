import unittest

import numpy as np
import pandas as pd

from backend.montecarlo import portfolio_mean_vol, simulate_portfolio_paths


class MonteCarloTests(unittest.TestCase):
    def test_simulate_portfolio_paths_shape(self) -> None:
        paths = simulate_portfolio_paths(
            mean_return=0.08,
            volatility=0.16,
            initial_investment=10000,
            num_days=10,
            num_simulations=500,
            random_seed=7,
        )
        self.assertEqual(paths.shape, (500, 10))
        self.assertTrue(np.all(paths > 0))

    def test_portfolio_mean_vol(self) -> None:
        means = pd.Series([0.1, 0.06], index=["AAA", "BBB"])
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.0225]], columns=["AAA", "BBB"], index=["AAA", "BBB"])
        mu, sigma = portfolio_mean_vol(means, cov, np.array([0.6, 0.4]))
        self.assertGreater(mu, 0)
        self.assertGreater(sigma, 0)


if __name__ == "__main__":
    unittest.main()

