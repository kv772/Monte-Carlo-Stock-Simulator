import unittest

import numpy as np

from backend.portfolio import risk_metrics


class PortfolioTests(unittest.TestCase):
    def test_risk_metrics_keys(self) -> None:
        rng = np.random.default_rng(1)
        daily = rng.normal(0.0003, 0.01, size=(1000, 30))
        paths = 10000 * np.cumprod(1 + daily, axis=1)
        metrics = risk_metrics(paths, initial_investment=10000, risk_free_rate=0.02, horizon_days=30)

        expected = {"p5", "p50", "p95", "probability_of_loss", "value_at_risk_95", "expected_return", "sharpe_ratio"}
        self.assertTrue(expected.issubset(metrics.keys()))
        self.assertGreaterEqual(metrics["probability_of_loss"], 0.0)
        self.assertLessEqual(metrics["probability_of_loss"], 1.0)


if __name__ == "__main__":
    unittest.main()

