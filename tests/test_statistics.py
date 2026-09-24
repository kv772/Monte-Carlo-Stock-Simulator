import unittest

import numpy as np
import pandas as pd

from backend.statistics import annualized_mean_return, annualized_volatility, compute_daily_log_returns, covariance_matrix


class StatisticsTests(unittest.TestCase):
    def test_statistics_pipeline_shapes(self) -> None:
        prices = pd.DataFrame(
            {
                "AAA": [100, 101, 102, 104, 103],
                "BBB": [50, 49, 50, 51, 53],
            }
        )
        returns = compute_daily_log_returns(prices)
        means = annualized_mean_return(returns)
        vol = annualized_volatility(returns)
        cov = covariance_matrix(returns)

        self.assertEqual(returns.shape, (4, 2))
        self.assertEqual(set(means.index), {"AAA", "BBB"})
        self.assertEqual(set(vol.index), {"AAA", "BBB"})
        self.assertEqual(cov.shape, (2, 2))
        self.assertTrue(np.allclose(cov.values, cov.values.T))


if __name__ == "__main__":
    unittest.main()
