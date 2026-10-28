import unittest
import copy
import os
import shutil
from pokesurvivor.memory_store import MemoryStore
from pokesurvivor.learning_engine import LearningEngine

class TestLearningAndRollback(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.memory = MemoryStore(data_dir=self.test_dir)
        self.engine = LearningEngine(self.memory)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    def test_no_artificial_sleep(self):
        import time
        t0 = time.time()
        run_data = {
            "duration": 50.0,
            "max_level": 3,
            "damage_events": 1,
            "damage_incidents": [],
            "trajectory_snapshots": [],
            "stuck_events": 0,
            "end_reason": "timeout"
        }
        self.engine.evaluate_run(run_data)
        elapsed = time.time() - t0
        # Must execute in under 0.25 seconds without fake sleep(2.5)
        self.assertLess(elapsed, 0.5)

    def test_automatic_rollback_on_performance_regression(self):
        # Save a known baseline best_policy
        best_p = {
            "version": "5.0",
            "movement": {"danger_threshold": 160.0, "mob_penalty_weight": 85.0}
        }
        self.engine.best_policy = copy.deepcopy(best_p)
        self.engine.learning_state["best_eval_avg"] = 120.0
        self.engine.learning_state["eval_block_runs"] = []

        # Simulate 3 poor runs on a mutated policy (averaging 40s << 120s baseline)
        poor_run = {
            "duration": 40.0,
            "max_level": 2,
            "damage_events": 3,
            "damage_incidents": [],
            "trajectory_snapshots": [],
            "stuck_events": 1,
            "end_reason": "defeated"
        }
        self.engine.evaluate_run(poor_run)
        self.engine.evaluate_run(poor_run)
        res3 = self.engine.evaluate_run(poor_run)

        # On the 3rd run of the block, rollback should trigger
        self.assertTrue(res3.get("is_rollback", False))
        # The current policy should be rolled back to match best_policy's parameters
        self.assertEqual(
            self.engine.current_policy["movement"]["danger_threshold"],
            best_p["movement"]["danger_threshold"]
        )

    def test_promotion_on_verified_improvement(self):
        best_p = {
            "version": "5.0",
            "movement": {"danger_threshold": 160.0, "mob_penalty_weight": 85.0}
        }
        self.engine.best_policy = copy.deepcopy(best_p)
        self.engine.learning_state["current_policy_version"] = "5.0"
        self.engine.learning_state["best_eval_avg"] = 80.0
        self.engine.learning_state["eval_block_runs"] = []

        # Simulate 3 superior runs (averaging 150s >> 80s baseline)
        good_run = {
            "duration": 150.0,
            "max_level": 6,
            "damage_events": 0,
            "damage_incidents": [],
            "trajectory_snapshots": [],
            "stuck_events": 0,
            "end_reason": "timeout"
        }
        self.engine.evaluate_run(good_run)
        self.engine.evaluate_run(good_run)
        res3 = self.engine.evaluate_run(good_run)

        # Should be promoted
        self.assertTrue(res3.get("is_promoted", False))
        self.assertEqual(self.engine.current_policy["version"], "5.1")

if __name__ == "__main__":
    unittest.main()
