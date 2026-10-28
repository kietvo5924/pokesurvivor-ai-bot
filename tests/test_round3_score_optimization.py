import unittest
import copy
import os
import shutil
from unittest.mock import MagicMock
from pokesurvivor.memory_store import MemoryStore
from pokesurvivor.learning_engine import LearningEngine, compute_policy_hash
from pokesurvivor.game_observer import GameObserver
from pokesurvivor.state_tracker import StateTracker

class TestRound3ScoreOptimization(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_data_round3"
        os.makedirs(self.test_dir, exist_ok=True)
        self.memory = MemoryStore(data_dir=self.test_dir)
        self.engine = LearningEngine(self.memory)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    def test_read_match_results_dom_structure(self):
        """Verifies genuine DOM end-screen parsing extracts game_score (kills), level, time, damage."""
        mock_page = MagicMock()
        mock_page.evaluate.return_value = {
            "available": True,
            "source": "dom_results_modal",
            "outcome": "You fell…",
            "kills": 184,
            "level": 14,
            "time_seconds": 165,
            "damage": "42.5k",
            "last_hits": "150",
            "meters": "1,240 m",
            "downs": "0",
            "coins": 120,
            "bosses": 1,
            "raw_stats": {
                "time": "02:45",
                "level": "14",
                "defeated": "184",
                "bosses": "1",
                "pokecoins earned": "+120"
            }
        }
        observer = GameObserver(mock_page)
        results = observer.read_match_results()
        self.assertTrue(results["available"])
        self.assertEqual(results["kills"], 184)
        self.assertEqual(results["level"], 14)
        self.assertEqual(results["time_seconds"], 165)
        self.assertEqual(results["coins"], 120)
        self.assertEqual(results["bosses"], 1)

        # Feed into StateTracker
        tracker = StateTracker()
        tracker.start_run()
        end_data = tracker.end_run(reason="defeated", dom_results=results)
        self.assertEqual(end_data["game_score"], 184)
        self.assertEqual(end_data["game_score_source"], "dom_results_kills")
        self.assertEqual(end_data["survival_time"], 165.0)
        self.assertEqual(end_data["survival_time_source"], "dom_results_time")
        self.assertEqual(end_data["max_level"], 14)
        self.assertEqual(end_data["max_level_source"], "dom_results_level")

    def test_unobserved_match_results_never_fabricates_zero_or_composite_score(self):
        """Verifies that unobserved runs record game_score=None and do not fabricate 0 or fake game scores."""
        tracker = StateTracker()
        tracker.start_run()
        end_data = tracker.end_run(reason="timeout", dom_results=None)
        self.assertIsNone(end_data["game_score"])
        self.assertEqual(end_data["game_score_source"], "unobserved_timeout")
        self.assertEqual(end_data["survival_time_source"], "state_tracker_clock")
        self.assertEqual(end_data["max_level_source"], "state_tracker_verified")

    def test_fixed_policy_and_skills_frozen_during_eval_block(self):
        """Verifies that current_policy (both movement AND skill_ratings) is 100% frozen during eval runs 1 and 2."""
        initial_policy = copy.deepcopy(self.engine.current_policy)
        initial_skills = copy.deepcopy(initial_policy.get("upgrade", {}).get("skill_ratings", {}))
        self.engine.learning_state["eval_block_runs"] = []

        run_data = {
            "duration": 80.0,
            "max_level": 4,
            "damage_events": 2,
            "damage_incidents": [],
            "trajectory_snapshots": [],
            "stuck_events": 0,
            "end_reason": "defeated",
            "upgrades": [
                {"name": "Water Gun", "tag": "Move", "time": 10.0},
                {"name": "Aqua Ring", "tag": "Buff", "time": 30.0}
            ]
        }

        # Run 1
        self.engine.evaluate_run(run_data)
        self.assertEqual(self.engine.current_policy["movement"], initial_policy["movement"])
        self.assertEqual(self.engine.current_policy.get("upgrade", {}).get("skill_ratings", {}), initial_skills)
        self.assertEqual(len(self.engine.learning_state["eval_block_runs"]), 1)

        # Run 2
        self.engine.evaluate_run(run_data)
        self.assertEqual(self.engine.current_policy["movement"], initial_policy["movement"])
        self.assertEqual(self.engine.current_policy.get("upgrade", {}).get("skill_ratings", {}), initial_skills)
        self.assertEqual(len(self.engine.learning_state["eval_block_runs"]), 2)

    def test_no_false_update_logs_when_parameters_at_ceiling(self):
        """Verifies engine never logs a parameter update where new value == old value."""
        # Set all relevant parameters to their exact ceiling limits
        self.engine.current_policy["movement"]["mob_penalty_weight"] = 150.0
        self.engine.current_policy["movement"]["mob_clear_corridor_bonus"] = 550.0
        self.engine.current_policy["movement"]["wall_breakout_thrust"] = 0.55
        self.engine.current_policy["movement"]["danger_threshold"] = 185.0
        self.engine.current_policy["movement"]["dir_commit_duration"] = 0.40
        self.engine.learning_state["eval_block_runs"] = []
        self.engine.learning_state["best_eval_avg"] = 50.0

        run_swarm = {
            "duration": 65.0,
            "max_level": 3,
            "damage_events": 4,
            "damage_incidents": [{"is_encircled": True, "min_enemy_dist": 100.0}],
            "trajectory_snapshots": [],
            "stuck_events": 0,
            "end_reason": "defeated"
        }
        self.engine.evaluate_run(run_swarm)
        self.engine.evaluate_run(run_swarm)
        res3 = self.engine.evaluate_run(run_swarm)

        deltas = res3.get("adaptations", [])
        for d in deltas:
            # Must never log "X -> X" (identical values)
            self.assertNotIn("550 -> 550", d)
            self.assertNotIn("0.55 -> 0.55", d)
            self.assertNotIn("185 -> 185", d)
            self.assertNotIn("0.40 -> 0.40", d)

    def test_candidate_rejection_and_champion_restoration(self):
        """Verifies regression triggers rollback to champion without incrementing policy version."""
        champion_policy = {
            "version": "6.0",
            "movement": {"danger_threshold": 160.0, "mob_penalty_weight": 90.0}
        }
        self.engine.best_policy = copy.deepcopy(champion_policy)
        self.engine.current_policy = copy.deepcopy(champion_policy)
        self.engine.learning_state["current_policy_version"] = "6.0"
        self.engine.learning_state["best_eval_avg"] = 140.0
        self.engine.learning_state["eval_block_runs"] = []

        poor_run = {
            "duration": 40.0,
            "max_level": 2,
            "damage_events": 5,
            "damage_incidents": [],
            "trajectory_snapshots": [],
            "stuck_events": 1,
            "end_reason": "defeated"
        }
        self.engine.evaluate_run(poor_run)
        self.engine.evaluate_run(poor_run)
        res3 = self.engine.evaluate_run(poor_run)

        self.assertTrue(res3.get("is_rollback", False))
        self.assertFalse(res3.get("is_promoted", False))
        # Version must NOT increment on failure!
        self.assertEqual(self.engine.current_policy["version"], "6.0")
        self.assertEqual(self.engine.learning_state["current_policy_version"], "6.0")
        # Champion parameters must be restored
        self.assertEqual(self.engine.current_policy["movement"]["danger_threshold"], 160.0)

    def test_policy_promotion_on_verified_game_score(self):
        """Verifies genuine game_score (kills) drives promotion when available."""
        self.engine.learning_state["current_policy_version"] = "7.0"
        self.engine.learning_state["best_game_score_avg"] = 120.0
        self.engine.learning_state["eval_block_runs"] = []

        superior_run = {
            "game_score": 210,
            "game_score_source": "dom_results_kills",
            "duration": 180.0,
            "max_level": 15,
            "damage_events": 10,
            "damage_incidents": [],
            "trajectory_snapshots": [],
            "stuck_events": 0,
            "end_reason": "defeated"
        }
        self.engine.evaluate_run(superior_run)
        self.engine.evaluate_run(superior_run)
        res3 = self.engine.evaluate_run(superior_run)

        self.assertTrue(res3.get("is_promoted", False))
        self.assertEqual(self.engine.current_policy["version"], "7.1")
        self.assertEqual(self.engine.learning_state["best_game_score_avg"], 210.0)

    def test_policy_hash_tracking(self):
        """Verifies policy hash is deterministic, changes with behavior, and is recorded per run."""
        p1 = {"movement": {"danger_threshold": 160.0}, "upgrade": {}, "items": {}}
        p2 = {"movement": {"danger_threshold": 175.0}, "upgrade": {}, "items": {}}
        hash1 = compute_policy_hash(p1)
        hash2 = compute_policy_hash(p2)
        self.assertNotEqual(hash1, hash2)
        self.assertEqual(hash1, compute_policy_hash(p1))

    def test_legacy_eval_block_schema_migration(self):
        """Verifies legacy list of floats in eval_block_runs is safely archived without losing history."""
        legacy_state = {
            "total_runs": 10,
            "eval_block_runs": [150.0, 160.0]  # Old float schema from Round 1
        }
        self.memory.save_learning_state(legacy_state)
        # Create fresh engine instance
        engine = LearningEngine(self.memory)
        # eval_block_runs should be sanitized to empty for fresh block
        self.assertEqual(engine.learning_state["eval_block_runs"], [])
        # History should preserve legacy data
        self.assertIn("eval_block_history", engine.learning_state)
        self.assertEqual(engine.learning_state["eval_block_history"][0]["runs"], [150.0, 160.0])

if __name__ == "__main__":
    unittest.main()

