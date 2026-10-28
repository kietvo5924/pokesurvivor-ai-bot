"""
tests/test_coin_tracker.py
--------------------------
Verifies CoinTracker fixes:
- coins=None is preserved as None (not converted to 0)
- Separates unobserved runs from genuine 0-coin runs
- In-game farming rate (coins/play hour) separates active play time from idle time
- Safety backup is created before overwriting
- Historical data persistence and backward compatibility
"""

import unittest
import tempfile
import os
import json
from pokesurvivor.coin_tracker import CoinTracker

class TestCoinTracker(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

    def tearDown(self):
        for ext in ["", ".bak"]:
            p = self.temp_file + ext
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def test_coins_none_not_converted_to_zero(self):
        """Verifies coins=None is recorded as None, not converted to 0."""
        tracker = CoinTracker(filepath=self.temp_file)
        tracker.record_run(coins=None, kills=100, survival_time=120.0, level=10)

        summary = tracker.get_summary()
        self.assertEqual(summary["total_runs"], 1)
        self.assertEqual(summary["observed_runs"], 0)
        self.assertEqual(summary["unobserved_runs"], 1)
        self.assertEqual(summary["zero_coin_runs"], 0)
        self.assertEqual(summary["total_coins"], 0)
        self.assertIsNone(summary["last_run_coins"])

        # Inspect raw JSON entry
        with open(self.temp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        last_run = data["runs"][-1]
        self.assertIsNone(last_run["coins"])
        self.assertFalse(last_run["coins_observed"])

    def test_genuine_zero_coins_distinguished_from_unobserved(self):
        """Verifies genuine 0 coins is counted in observed_runs and zero_coin_runs."""
        tracker = CoinTracker(filepath=self.temp_file)
        # Run 1: Earned genuine 0 coins (e.g. quick death at 10s)
        tracker.record_run(coins=0, kills=5, survival_time=10.0, level=1)
        # Run 2: Unobserved (modal closed or crash)
        tracker.record_run(coins=None, kills=20, survival_time=30.0, level=3)
        # Run 3: Earned 150 coins
        tracker.record_run(coins=150, kills=250, survival_time=180.0, level=15)

        summary = tracker.get_summary()
        self.assertEqual(summary["total_runs"], 3)
        self.assertEqual(summary["observed_runs"], 2) # Run 1 (0 coins) and Run 3 (150 coins)
        self.assertEqual(summary["unobserved_runs"], 1) # Run 2 (None)
        self.assertEqual(summary["zero_coin_runs"], 1) # Run 1 (0 coins)
        self.assertEqual(summary["total_coins"], 150)

    def test_in_game_play_rate_distinguishes_idle_time(self):
        """
        Verifies farming rate uses in-game play time (survival_time)
        rather than wall-clock time which includes hours of idle pause.
        """
        tracker = CoinTracker(filepath=self.temp_file)
        # Suppose tracker was created with "since" 10 hours ago:
        tracker.data["since"] = tracker.data["since"] - 36000.0 # 10 hours of wall time!

        # Bot played 1 run: 200 coins in 360 seconds (0.1 in-game play hours)
        tracker.record_run(coins=200, kills=300, survival_time=360.0, level=16)

        summary = tracker.get_summary()
        self.assertEqual(summary["total_coins"], 200)
        self.assertEqual(summary["total_play_time_seconds"], 360.0)
        # Play hour = 360 / 3600 = 0.1 hr. Rate = 200 / 0.1 = 2000.0 coins/hr
        self.assertAlmostEqual(summary["coins_per_play_hour"], 2000.0)
        self.assertAlmostEqual(summary["coins_per_hour"], 2000.0)

    def test_safety_backup_creation(self):
        """Verifies safety backup (.bak) is created before modifying existing file."""
        # Create initial file
        with open(self.temp_file, "w", encoding="utf-8") as f:
            json.dump({"total_coins": 50, "runs": [{"coins": 50, "survival_time": 60.0}]}, f)

        # Initialize tracker on existing file
        tracker = CoinTracker(filepath=self.temp_file)
        bak_file = self.temp_file + ".bak"
        self.assertTrue(os.path.exists(bak_file))

        with open(bak_file, "r", encoding="utf-8") as f:
            bak_data = json.load(f)
        self.assertEqual(bak_data["total_coins"], 50)

if __name__ == "__main__":
    unittest.main()
