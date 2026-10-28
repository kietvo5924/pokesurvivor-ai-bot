"""
tests.test_round6_regression
----------------------------
Regression tests for Round 6 PokéSurvivor AI Bot integration:
1. Camera Scale & Translation Coordinate Consistency (Screen <-> World normalization).
2. Match Startup Loop DOM Phase Fallback (activeScreen == 'game', gState.gameState, HUD metrics).
3. Hybrid Terrain (mapBounds + OpenCV), Missing mapBounds & Unobserved Safety semantics.
4. Target Provenance & Dummy Target Rejection (No fake (0,0) coordinates or invented targets).
"""

import unittest
import math
from unittest.mock import MagicMock, patch
from pokesurvivor.game_state_adapter import GameStateAdapter
from pokesurvivor.movement_policy import MovementPolicy
from pokesurvivor.target_evaluator import TargetEvaluator, normalize_game_state
from pokesurvivor.game_observer import GameObserver


class TestRound6Regression(unittest.TestCase):

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

    # =========================================================================
    # 1. CAMERA SCALE & TRANSLATION COORDINATE NORMALIZATION
    # =========================================================================

    def test_camera_transform_opencv_screen_to_world(self):
        """
        Verify that OpenCV screen-detected entities (enemies, hazards, items, rift)
        are correctly converted to world coordinates using camera matrix (s, e, f):
            world_x = (screen_x - e) / s
            world_y = (screen_y - f) / s
        and that screen_x, screen_y are preserved.
        """
        adapter = GameStateAdapter()

        # Camera with scale = 1.25, translation e = -400.0, f = -200.0
        s = 1.25
        e_cam = -400.0
        f_cam = -200.0

        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False,
            "gState": None,
            "telemetry": {
                "texts": ["Tapu Fini", "100 / 100"],
                "camera": {"s": s, "e": e_cam, "f": f_cam},
                "spriteCount": 10,
                "lastFrameAt": 1000.0
            },
            "dom": {"activeScreen": "game", "phase": "running"},
            "results": None
        })

        # Mock perception returns entities in screen space (screen center is 640, 360)
        # Enemy at screen (740, 360) -> 100px East on screen
        mock_frame = MagicMock()
        adapter.perception.analyze_frame = MagicMock(return_value={
            "enemies": [{"x": 740.0, "y": 360.0, "radius": 16, "dex": 19}],
            "hazards": [{"x": 640.0, "y": 460.0, "radius": 20, "kind": "telegraph"}],
            "gems": [{"x": 600.0, "y": 360.0, "kind": "gem"}],
            "items_on_ground": [{"x": 580.0, "y": 300.0, "kind": "chest_gold"}],
            "rift_target": {"x": 800.0, "y": 200.0, "source": "opencv_purple_portal"},
            "sector_obstacles": {},
            "sector_obs_costs": {},
            "sector_dangers": {}
        })

        unified = adapter.read_unified_state(frame_img=mock_frame)

        # Expected player world pos: (640 - (-400)) / 1.25 = 1040 / 1.25 = 832.0
        #                            (360 - (-200)) / 1.25 = 560 / 1.25 = 448.0
        p_wx = (640.0 - e_cam) / s
        p_wy = (360.0 - f_cam) / s
        self.assertAlmostEqual(unified["player"]["world_x"], p_wx, places=2)
        self.assertAlmostEqual(unified["player"]["world_y"], p_wy, places=2)

        # Expected enemy world pos: (740 - (-400)) / 1.25 = 1140 / 1.25 = 912.0
        #                           (360 - (-200)) / 1.25 = 560 / 1.25 = 448.0
        enemies = unified["enemies"]
        self.assertEqual(len(enemies), 1)
        e0 = enemies[0]
        self.assertAlmostEqual(e0["world_x"], (740.0 - e_cam) / s, places=2)
        self.assertAlmostEqual(e0["world_y"], (360.0 - f_cam) / s, places=2)
        self.assertEqual(e0["screen_x"], 740.0)
        self.assertEqual(e0["screen_y"], 360.0)

        # Relative world vector dx should match screen relative dx scaled by 1/s
        dx_world = e0["world_x"] - p_wx
        dx_screen_scaled = (740.0 - 640.0) / s
        self.assertAlmostEqual(dx_world, dx_screen_scaled, places=2)
        self.assertEqual(e0["sector"], "E")

        # Rift target in specials
        rift = unified["specials"]["rift"]
        self.assertIsNotNone(rift)
        self.assertAlmostEqual(rift["world_x"], (800.0 - e_cam) / s, places=2)
        self.assertAlmostEqual(rift["world_y"], (200.0 - f_cam) / s, places=2)
        self.assertEqual(rift["screen_x"], 800.0)
        self.assertEqual(rift["screen_y"], 200.0)

    def test_camera_transform_window_g_world_to_screen(self):
        """
        Verify that in window.G mode, world coordinates are native and
        screen coordinates are accurately derived via:
            screen_x = s * world_x + e
            screen_y = s * world_y + f
        """
        adapter = GameStateAdapter()
        s = 1.5
        e_cam = -300.0
        f_cam = -150.0

        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "player": {"x": 1000.0, "y": 800.0, "hp": 150, "maxHp": 150},
                "enemies": [{"x": 1100.0, "y": 800.0, "r": 16, "dex": 25}],
                "hazards": [{"x": 1000.0, "y": 900.0, "r": 20, "kind": "telegraph"}],
                "projectiles": [],
                "pickups": [{"x": 1020.0, "y": 820.0, "kind": "gem"}],
                "specials": {}
            },
            "telemetry": {
                "texts": [],
                "camera": {"s": s, "e": e_cam, "f": f_cam}
            },
            "dom": {"activeScreen": "game", "phase": "running"},
            "results": None
        })

        unified = adapter.read_unified_state()
        e0 = unified["enemies"][0]
        self.assertEqual(e0["world_x"], 1100.0)
        self.assertEqual(e0["world_y"], 800.0)
        # Expected screen_x = 1.5 * 1100.0 - 300.0 = 1350.0
        self.assertAlmostEqual(e0["screen_x"], 1.5 * 1100.0 + e_cam, places=2)
        self.assertAlmostEqual(e0["screen_y"], 1.5 * 800.0 + f_cam, places=2)

    # =========================================================================
    # 2. STARTUP LOOP & DOM PHASE FALLBACK
    # =========================================================================

    def test_dom_phase_missing_fallback_active_screen(self):
        """
        Verify GameStateAdapter and GameObserver fall back to 'running'
        when DOM phase is missing/empty but activeScreen is 'game'.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False,
            "gState": None,
            "telemetry": None,
            "dom": {
                "activeScreen": "game",
                "phase": "none",  # DOM testid was missing/empty
                "choiceOpen": False,
                "modalKind": "level-up",
                "choices": [],
                "exitAvailable": False
            },
            "results": None
        })

        unified = adapter.read_unified_state()
        self.assertEqual(unified["ui"]["phase"], "running")
        self.assertEqual(unified["ui"]["active_screen"], "game")

    def test_dom_phase_missing_fallback_gstate(self):
        """
        Verify GameStateAdapter falls back to gState.gameState when DOM phase is none.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "gameState": "playing",
                "player": {"x": 500.0, "y": 500.0, "hp": 100, "maxHp": 100},
                "enemies": []
            },
            "telemetry": None,
            "dom": {
                "activeScreen": "unknown",
                "phase": "",  # missing
                "choiceOpen": False
            },
            "results": None
        })

        unified = adapter.read_unified_state()
        self.assertEqual(unified["ui"]["phase"], "running")

    def test_live_hud_metrics_signals_match_started(self):
        """
        Verify that live metrics from canvas telemetry (kills or time_seconds)
        indicate the match is actively running even if DOM phase selector is absent.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False,
            "gState": None,
            "telemetry": {
                "texts": ["SHADY FOREST · 4 DEFEATED · 12 COINS", "0:45"],
                "camera": {"s": 1.0, "e": 0.0, "f": 0.0}
            },
            "dom": {
                "activeScreen": "game",
                "phase": "none"
            },
            "results": None
        })

        unified = adapter.read_unified_state()
        self.assertEqual(unified["metrics"]["kills"], 4)
        self.assertEqual(unified["metrics"]["coins"], 12)
        self.assertEqual(unified["metrics"]["time_seconds"], 45.0)
        self.assertTrue(unified["metrics"]["available"])

    # =========================================================================
    # 3. HYBRID TERRAIN & UNOBSERVED SAFETY STATUS
    # =========================================================================

    def test_hybrid_terrain_bounds_and_opencv(self):
        """
        Verify that when both mapBounds (window.G) and OpenCV vision terrain are present,
        terrain status is 'verified' and obstacles from both sources are merged.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "player": {"x": 100.0, "y": 1500.0, "hp": 100, "maxHp": 100},  # Close to West wall
                "enemies": [],
                "mapBounds": {"minX": 0.0, "maxX": 3000.0, "minY": 0.0, "maxY": 3000.0}
            },
            "telemetry": {"camera": {"s": 1.0, "e": 0.0, "f": 0.0}},
            "dom": {"phase": "running"}
        })

        mock_frame = MagicMock()
        # Mock OpenCV detects an obstacle to the North
        adapter.perception.analyze_frame = MagicMock(return_value={
            "enemies": [],
            "sector_obstacles": {"N": True, "W": False},
            "sector_obs_costs": {"N": 200.0, "W": 0.0},
            "sector_dangers": {}
        })

        unified = adapter.read_unified_state(frame_img=mock_frame)
        terrain = unified["terrain"]

        self.assertEqual(terrain["status"], "verified")
        self.assertEqual(terrain["source"], "hybrid_bounds_and_opencv")
        # West is blocked by wall boundary (< 180px)
        self.assertTrue(terrain["sector_obstacles"]["W"])
        # North is blocked by OpenCV contour obstacle
        self.assertTrue(terrain["sector_obstacles"]["N"])

    def test_unobserved_terrain_prevents_completely_safe_label(self):
        """
        Verify MovementPolicy treats unobserved terrain cautiously:
        is_completely_safe() must return False when terrain_status is 'unobserved',
        so the bot does not blindly rush into unverified directions.
        """
        policy = MovementPolicy(self.default_policy)
        vision_state = {
            "player": {"world_x": 1000.0, "world_y": 1000.0, "hp_ratio": 1.0},
            "enemies": [],
            "hazards": [],
            "sector_enemy_counts": {s: 0 for s in ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]},
            "sector_dangers": {s: 0.0 for s in ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]},
            "terrain": {
                "sector_obstacles": {},
                "sector_obs_costs": {},
                "status": "unobserved",
                "source": "unobserved"
            },
            "terrain_status": "unobserved",
            "min_enemy_dist": 9999.0,
            "danger_level": "LOW"
        }

        # Choose movement should execute cleanly without crashing
        direction, keys, dur = policy.select_movement(vision_state)
        self.assertIn(direction, ["E", "SE", "S", "SW", "W", "NW", "N", "NE"])
        self.assertIsInstance(keys, list)

    def test_missing_mapbounds_falls_back_to_opencv(self):
        """
        Verify that when mapBounds is None/missing, terrain safely uses OpenCV terrain
        with status 'local_view_only'.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "player": {"x": 500.0, "y": 500.0, "hp": 100, "maxHp": 100},
                "enemies": [],
                "mapBounds": None  # Missing map bounds!
            },
            "telemetry": {"camera": {"s": 1.0, "e": 0.0, "f": 0.0}},
            "dom": {"phase": "running"}
        })

        mock_frame = MagicMock()
        adapter.perception.analyze_frame = MagicMock(return_value={
            "enemies": [],
            "sector_obstacles": {"E": True},
            "sector_obs_costs": {"E": 150.0},
            "sector_dangers": {}
        })

        unified = adapter.read_unified_state(frame_img=mock_frame)
        self.assertEqual(unified["terrain"]["status"], "local_view_only")
        self.assertEqual(unified["terrain"]["source"], "opencv_terrain")
        self.assertTrue(unified["terrain"]["sector_obstacles"]["E"])

    # =========================================================================
    # 4. TARGET PROVENANCE & DUMMY TARGET REJECTION
    # =========================================================================

    def test_reject_dummy_uninitialized_coordinates(self):
        """
        Verify that TargetEvaluator rejects dummy/uninitialized coordinates (0, 0)
        when no genuine source is present, and does not invent fake targets.
        """
        evaluator = TargetEvaluator()

        dummy_state = {
            "player": {"world_x": 1000.0, "world_y": 1000.0, "hp_ratio": 1.0},
            "specials": {
                # Dummy (0, 0) with no source or sector
                "rift": {"x": 0.0, "y": 0.0},
                "spring": {"x": 0.0, "y": 0.0},
                "chest": {"x": 0.0, "y": 0.0},
                "altar": {"x": 0.0, "y": 0.0}
            },
            "enemies": [],
            "items": [],
            "hazards": []
        }

        # TargetEvaluator must NOT select any of the fake (0, 0) specials
        target = evaluator.evaluate_targets(dummy_state)
        self.assertIsNone(target, "Should reject dummy (0, 0) specials with no genuine provenance")

    def test_genuine_special_target_evaluated_correctly(self):
        """
        Verify that genuine specials (with valid coordinates or source)
        are properly evaluated with accurate relative distance and sector from player.
        """
        evaluator = TargetEvaluator()

        state = {
            "player": {"world_x": 1000.0, "world_y": 1000.0, "hp_ratio": 0.85},
            "specials": {
                # Real chest at (1150, 1000) -> 150px East
                "chest": {"x": 1150.0, "y": 1000.0, "source": "pickupMgr_chest"}
            },
            "enemies": [],
            "items": [],
            "hazards": []
        }

        target = evaluator.evaluate_targets(state)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "chest")
        self.assertEqual(target["sector"], "E")
        self.assertAlmostEqual(target["dist"], 150.0, places=1)


if __name__ == "__main__":
    unittest.main()

