import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.api import SimulationRequest, simulate


class ApiTests(unittest.TestCase):
    @patch("backend.api.get_price_data")
    def test_simulate_endpoint(self, mock_get_price_data) -> None:
        dates = pd.date_range("2025-01-01", periods=20, freq="D")
        mock_get_price_data.return_value = pd.DataFrame(
            {"AAA": np.linspace(100, 110, 20), "BBB": np.linspace(90, 102, 20)},
            index=dates,
        )
        payload = {
            "tickers": ["AAA", "BBB"],
            "weights": [0.5, 0.5],
            "initial_investment": 10000,
            "horizon_days": 30,
            "num_simulations": 500,
            "lookback_years": 2,
            "risk_free_rate": 0.02,
            "sample_paths": 50,
        }
        body = simulate(SimulationRequest(**payload))
        self.assertIn("fan_chart", body)
        self.assertIn("sample_paths", body)
        self.assertIn("histogram", body)
        self.assertIn("risk_metrics", body)
        self.assertEqual(len(body["sample_paths"]), 50)


if __name__ == "__main__":
    unittest.main()
