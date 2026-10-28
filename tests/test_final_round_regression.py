"""
tests.test_final_round_regression
---------------------------------
Comprehensive regression test suite for Round 5 PokéSurvivor AI Bot integration:
1. World vs Screen Coordinates Consistency (no default to 640,360 when world coords exist)
2. Frame Caching across skipped frames (no vanishing entities when frame_img is None)
3. Target Direction Relative Vector Calculation (dx = tx - px, dy = ty - py, not from origin)
4. Unknown HP Cautious Policy & Rift Suppression (hp_ratio is None never assumed 100%)
5. DOM Phase Fallback to gState.gameState
6. Match Results Schema Compatibility (time_seconds, kills, coins through StateTracker & CoinTracker)
7. Terrain Boundaries & Projectile Trajectory TTC/DCA Hazard Evaluation
"""

import unittest
import os
import time
import math
from unittest.mock import MagicMock
from pokesurvivor.game_state_adapter import GameStateAdapter
from pokesurvivor.target_evaluator import TargetEvaluator, normalize_game_state
from pokesurvivor.movement_policy import MovementPolicy
from pokesurvivor.state_tracker import StateTracker
from pokesurvivor.coin_tracker import CoinTracker


class TestFinalRoundRegression(unittest.TestCase):

    def setUp(self):
        self.default_policy = {
            "movement": {
                "pattern": "kiting",
                "danger_threshold": 160.0,
                "mob_penalty_weight": 85.0,
                "mob_clear_corridor_bonus": 350.0,
                "wall_avoidance_weight": 1.4,
                "wall_breakout_thrust": 0.40,
                "dodge_hold_duration": 0.35,
                "dir_commit_duration": 0.28,
                "exp_harvest_weight": 0.45,
                "evasion_weight": 0.95,
                "step_duration": 0.24
            }
        }

    def test_world_vs_screen_coordinates_consistency(self):
        """
        Verify player_pos is included in unified_state with real world coordinates,
        and MovementPolicy uses them rather than defaulting to (640, 360).
        """
        adapter = GameStateAdapter()
        # Mock browser data with window.G world coordinates
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "player": {"x": 1200.0, "y": 800.0, "hp": 100, "maxHp": 100},
                "enemies": [{"x": 1250.0, "y": 800.0, "r": 16, "dex": 25}],
                "hazards": [],
                "projectiles": [],
                "pickups": [],
                "rift": None
            },
            "telemetry": None,
            "dom": None,
            "results": None
        })

        unified_state = adapter.read_unified_state()
        self.assertEqual(unified_state["player_pos"], (1200.0, 800.0))
        self.assertEqual(unified_state["player"]["pos"], (1200.0, 800.0))
        
        enemy = unified_state["enemies"][0]
        self.assertAlmostEqual(enemy["dist"], 50.0, places=1)
        self.assertEqual(enemy["sector"], "E")

        # In MovementPolicy, player at (1200, 800) with close enemy at (1250, 800) East
        # should steer West (repulsion away from enemy)
        policy = MovementPolicy(self.default_policy)
        chosen_dir, keys, dur = policy.select_movement(unified_state, current_level=3, elapsed_time=50.0)
        self.assertEqual(chosen_dir, "W")
        self.assertIn("KeyA", keys)

    def test_missing_frame_caching(self):
        """
        RunManager captures frames at ~80ms while looping at ~40ms.
        When frame_img is None on intermediate loops, adapter must return cached entities
        within max_frame_staleness (0.85s), not an empty list.
        """
        mock_perception = MagicMock()
        mock_perception.analyze_frame.return_value = {
            "enemies": [{"x": 600, "y": 300, "dist": 65.0, "sector": "NW"}],
            "hazards": [],
            "gems": [],
            "items_on_ground": [],
            "sector_dangers": {"NW": 80.0},
            "sector_enemy_counts": {"NW": 1},
            "min_enemy_dist": 65.0,
            "nearest_enemy_sector": "NW",
            "danger_level": "HIGH",
            "is_encircled": False,
            "is_stuck": False
        }

        adapter = GameStateAdapter(perception=mock_perception)
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False, "gState": None, "telemetry": None, "dom": None, "results": None
        })

        # Frame 1: fresh image provided
        fake_frame = "fake_image_bytes"
        state1 = adapter.read_unified_state(frame_img=fake_frame)
        self.assertEqual(len(state1["enemies"]), 1)
        self.assertEqual(state1["sources"]["enemies"], "opencv_perception")
        self.assertEqual(mock_perception.analyze_frame.call_count, 1)

        # Frame 2: intermediate loop with frame_img = None
        state2 = adapter.read_unified_state(frame_img=None)
        self.assertEqual(len(state2["enemies"]), 1)
        self.assertEqual(state2["sources"]["enemies"], "opencv_perception_cached")
        # OpenCV was NOT called again!
        self.assertEqual(mock_perception.analyze_frame.call_count, 1)

        # Simulate expiration by setting last_frame_time back by 2 seconds
        adapter.last_frame_time = time.time() - 2.0
        state3 = adapter.read_unified_state(frame_img=None)
        self.assertEqual(len(state3["enemies"]), 0)
        self.assertEqual(state3["sources"]["enemies"], "unobserved_expired")

    def test_rift_target_relative_vector(self):
        """
        Verify target evaluator computes sector & distance from player position,
        not from origin (0, 0).
        """
        evaluator = TargetEvaluator()

        # Case A: Player at (1500, 1500), Rift at (1600, 1500) -> 100px East
        state_east = {
            "player": {"world_x": 1500.0, "world_y": 1500.0, "hp_ratio": 1.0},
            "specials": {
                "rift": {"x": 1600.0, "y": 1500.0}
            },
            "enemies": []
        }
        target_a = evaluator.evaluate_targets(state_east, now=time.time())
        self.assertIsNotNone(target_a)
        self.assertEqual(target_a["type"], "rift")
        self.assertEqual(target_a["sector"], "E")
        self.assertAlmostEqual(target_a["dist"], 100.0, places=1)

        # Case B: Player at (1500, 1500), Rift at (1500, 1200) -> 300px North
        state_north = {
            "player": {"world_x": 1500.0, "world_y": 1500.0, "hp_ratio": 1.0},
            "specials": {
                "rift": {"x": 1500.0, "y": 1200.0}
            },
            "enemies": []
        }
        target_b = evaluator.evaluate_targets(state_north, now=time.time())
        self.assertIsNotNone(target_b)
        self.assertEqual(target_b["type"], "rift")
        self.assertEqual(target_b["sector"], "N")
        self.assertAlmostEqual(target_b["dist"], 300.0, places=1)

    def test_unknown_hp_cautious_policy(self):
        """
        Unobserved HP must remain None/unknown (not 100%).
        Unknown HP triggers cautious behavior:
        1. Rift entry is strictly suppressed (cannot risk boss arena with unknown health)
        2. MovementPolicy increases danger threshold and repulsion multiplier
        3. Reckless gem diving is suppressed
        """
        evaluator = TargetEvaluator()
        state_unknown_hp = {
            "player": {"world_x": 1500.0, "world_y": 1500.0, "hp_ratio": None},
            "specials": {
                "rift": {"x": 1600.0, "y": 1500.0}
            },
            "enemies": []
        }

        # 1. Normalize preserves None
        norm = normalize_game_state(state_unknown_hp)
        self.assertIsNone(norm["player"]["hp_ratio"])

        # 2. Rift entry is suppressed when HP is unknown
        target = evaluator.evaluate_targets(state_unknown_hp, now=time.time())
        self.assertIsNone(target, "Rift entry should be suppressed when HP is unknown")

        # 3. MovementPolicy caution
        policy = MovementPolicy(self.default_policy)
        vision_unknown_hp = {
            "player_pos": (640, 360),
            "hp_ratio": None,
            "enemies": [{"x": 640, "y": 190, "dist": 170.0, "sector": "N"}],  # 170px away
            "min_enemy_dist": 170.0,
            "nearest_enemy_sector": "N",
            "sector_enemy_counts": {"N": 1},
            "sector_dangers": {"N": 85.0},
            "sector_obstacles": {},
            "danger_level": "LOW"
        }
        # Baseline threshold is 160px.
        # When HP is unknown, effective threshold is 160 + 15 = 175px.
        # Since min_enemy_dist (170px) < 175px, the bot should trigger repulsion away from North!
        dir_name, keys, dur = policy.select_movement(vision_unknown_hp, current_level=2, elapsed_time=60.0)
        self.assertEqual(dir_name, "S", "Bot with unknown HP should cautiously evade enemy within 175px")

    def test_dom_phase_fallback_to_gstate(self):
        """
        When DOM does not have [data-testid="game-phase"] or returns 'none',
        adapter falls back to gState.gameState.
        """
        adapter = GameStateAdapter()

        # DOM returns phase: 'none', but engine has gameState: 'running'
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "gameState": "running",
                "player": {"x": 500, "y": 500}
            },
            "telemetry": None,
            "dom": {"phase": "none", "activeScreen": "game"},
            "results": None
        })

        state = adapter.read_unified_state()
        self.assertEqual(state["ui"]["phase"], "running")

        # Engine has gameState: 'GameOver'
        adapter._fetch_browser_data.return_value["gState"]["gameState"] = "GameOver"
        state_ended = adapter.read_unified_state()
        self.assertEqual(state_ended["ui"]["phase"], "ended")

    def test_match_results_schema_compatibility(self):
        """
        Verify match results from adapter match StateTracker.end_run and CoinTracker schemas,
        preserving kills, time_seconds, and coins without loss.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False,
            "gState": None,
            "telemetry": None,
            "dom": None,
            "results": {
                "available": True,
                "kills": 384,
                "level": 16,
                "time_str": "03:45",
                "time_seconds": 225.0,
                "coins": 180,
                "bosses": 2,
                "damage": 54000,
                "downs": 0
            }
        })

        unified_state = adapter.read_unified_state()
        metrics = unified_state["metrics"]
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["kills"], 384)
        self.assertEqual(metrics["time_seconds"], 225.0)
        self.assertEqual(metrics["coins"], 180)

        # Test StateTracker.end_run with adapter metrics
        tracker = StateTracker()
        tracker.start_run()
        run_res = tracker.end_run(reason="defeated", dom_results=metrics)
        
        self.assertEqual(run_res["game_score"], 384)
        self.assertEqual(run_res["game_score_source"], "dom_results_kills")
        self.assertEqual(run_res["survival_time"], 225.0)
        self.assertEqual(run_res["survival_time_source"], "dom_results_time")
        self.assertEqual(run_res["max_level"], 16)
        self.assertEqual(run_res["max_level_source"], "dom_results_level")

        # Test CoinTracker with adapter metrics using a temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            coin_tracker = CoinTracker(filepath=tmp_path)
            coin_tracker.record_run(
                coins=metrics.get("coins"),
                kills=run_res["game_score"],
                survival_time=run_res["survival_time"],
                level=run_res["max_level"],
                bosses=metrics.get("bosses")
            )
            summary = coin_tracker.get_summary()
            self.assertEqual(summary["total_coins"], 180)
            self.assertEqual(summary["observed_runs"], 1)
            self.assertEqual(summary["unobserved_runs"], 0)
        finally:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: pass
            if os.path.exists(tmp_path + ".bak"):
                try: os.remove(tmp_path + ".bak")
                except Exception: pass

    def test_terrain_boundaries_and_projectile_ttc(self):
        """
        Verify window.G map bounds calculate wall obstacles near map edge,
        and projectile trajectory calculates TTC and DCA.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "mapBounds": {"minX": 0, "maxX": 3000, "minY": 0, "maxY": 3000},
                # Player at (80, 1500) - 80px from West wall
                "player": {"x": 80.0, "y": 1500.0, "hp": 100, "maxHp": 100},
                "enemies": [],
                "hazards": [],
                # Incoming projectile moving East at (280, 1500) with vx = -200 (towards player at x=80)
                "projectiles": [{"x": 280.0, "y": 1500.0, "vx": -200.0, "vy": 0.0}],
                "pickups": [],
                "rift": None
            },
            "telemetry": None,
            "dom": None,
            "results": None
        })

        state = adapter.read_unified_state()
        
        # 1. West wall boundary is marked as obstacle
        obs = state["terrain"]["sector_obstacles"]
        costs = state["terrain"]["sector_obs_costs"]
        self.assertTrue(obs["W"], "West wall within 180px buffer must be marked obstacle")
        self.assertGreater(costs["W"], 0.0)

        # 2. Projectile TTC & DCA calculated
        hazards = state["hazards"]
        self.assertEqual(len(hazards), 1)
        proj = hazards[0]
        self.assertEqual(proj["type"], "bullet")
        # Distance = 280 - 80 = 200px. Speed = 200px/s -> TTC = 1.0s
        self.assertAlmostEqual(proj["ttc"], 1.0, places=2)
        self.assertAlmostEqual(proj["dca"], 0.0, places=1)
        self.assertTrue(proj["is_imminent"])


if __name__ == '__main__':
    unittest.main()
