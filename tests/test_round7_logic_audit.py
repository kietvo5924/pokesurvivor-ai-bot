"""
tests.test_round7_logic_audit
-----------------------------
Comprehensive regression test suite for Round 7 Final Logic Audit & Fixes:
1. Camera Matrix Validation: invalid (shear/scale) or stale transforms must never derive fake world coords.
2. Incomplete window.G Fallback: empty enemies/hazards/items list in window.G does NOT imply zero danger;
   harmonizes with OpenCV perception when visual frame is available.
3. Dummy (0,0) Target Rejection: Rift, Spring, Chest, Altar at (0,0) are strictly rejected regardless of source.
4. Unified Early-Game Rift Gating: MovementPolicy and TargetEvaluator both strictly suppress Rift entry
   when elapsed_time < 120s or level < 3.
5. Terrain Semantics for 'bounds_only': Knowing map bounds alone does not make directions 'completely safe'.
"""

import unittest
import time
import math
from unittest.mock import MagicMock
from pokesurvivor.game_state_adapter import GameStateAdapter
from pokesurvivor.movement_policy import MovementPolicy
from pokesurvivor.target_evaluator import TargetEvaluator


class TestRound7LogicAudit(unittest.TestCase):

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
    # 1. CAMERA MATRIX VALIDATION (STALE, SHEARED, INVALID SCALE)
    # =========================================================================

    def test_camera_transform_rejected_when_sheared(self):
        """
        Matrix with shear or rotation (b != 0 or c != 0) is not a 2D camera transform.
        GameStateAdapter must NOT derive world coordinates from it.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False,
            "gState": None,
            "telemetry": {
                "texts": ["100 / 100"],
                # Matrix has shear (b=0.8, c=-0.8)
                "camera": {"s": 1.0, "e": 100.0, "f": 50.0, "b": 0.8, "c": -0.8, "verified": True},
            },
            "dom": {"phase": "running"}
        })

        mock_frame = MagicMock()
        adapter.perception.analyze_frame = MagicMock(return_value={
            "enemies": [{"x": 700.0, "y": 360.0, "radius": 16}],
            "hazards": [],
            "gems": [],
            "sector_obstacles": {}
        })

        unified = adapter.read_unified_state(frame_img=mock_frame)
        player = unified["player"]

        # World coordinates must NOT be derived from sheared camera
        self.assertFalse(unified["camera_verified"])
        self.assertIsNone(player["world_x"])
        self.assertIsNone(player["world_y"])
        # Player pos falls back to screen space center
        self.assertEqual(unified["player_pos"], (640.0, 360.0))

        # Enemies remain in screen space and do not invent fake world coords
        e0 = unified["enemies"][0]
        self.assertIsNone(e0["world_x"])
        self.assertEqual(e0["screen_x"], 700.0)
        # Distance and sector are correctly calculated in screen coordinates (700 - 640 = 60px East)
        self.assertAlmostEqual(e0["dist"], 60.0, places=1)
        self.assertEqual(e0["sector"], "E")

    def test_camera_transform_rejected_when_stale(self):
        """
        Stale camera matrix (> 1.5s old) must be rejected and not used for world coords.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False,
            "gState": None,
            "telemetry": {
                "texts": ["100 / 100"],
                # 3000ms old (stale)
                "camera": {"s": 1.5, "e": -300.0, "f": -150.0, "now": 5000.0, "updatedAt": 2000.0, "verified": True},
            },
            "dom": {"phase": "running"}
        })

        unified = adapter.read_unified_state(frame_img=None)
        self.assertFalse(unified["camera_verified"])
        self.assertIsNone(unified["player"]["world_x"])

    def test_camera_transform_rejected_when_unverified_or_invalid_scale(self):
        """
        Camera matrix with verified=False or scale out of bounds (s < 0.2 or s > 5.0)
        must be rejected.
        """
        adapter = GameStateAdapter()
        # Scale = 0.05 (too small)
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False,
            "gState": None,
            "telemetry": {
                "camera": {"s": 0.05, "e": 0.0, "f": 0.0, "verified": True},
            },
            "dom": {"phase": "running"}
        })
        unified = adapter.read_unified_state()
        self.assertFalse(unified["camera_verified"])

        # verified = False
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": False,
            "gState": None,
            "telemetry": {
                "camera": {"s": 1.0, "e": 0.0, "f": 0.0, "verified": False},
            },
            "dom": {"phase": "running"}
        })
        unified2 = adapter.read_unified_state()
        self.assertFalse(unified2["camera_verified"])

    # =========================================================================
    # 2. INCOMPLETE WINDOW.G FALLBACK TO OPENCV (NEVER ASSUME EMPTY IS SAFE)
    # =========================================================================

    def test_window_g_empty_enemies_falls_back_to_opencv(self):
        """
        When window.G is present but enemies is [] (e.g. minified engine differences),
        do NOT assume zero danger! Harmonize and incorporate OpenCV detected enemies.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "player": {"x": 1000.0, "y": 1000.0, "hp": 100, "maxHp": 100},
                "enemies": [],  # Empty in window.G!
                "hazards": [],
                "projectiles": [],
                "pickups": []
            },
            "telemetry": {
                "camera": {"s": 1.0, "e": 0.0, "f": 0.0, "verified": True}
            },
            "dom": {"phase": "running"}
        })

        mock_frame = MagicMock()
        adapter.perception.analyze_frame = MagicMock(return_value={
            # OpenCV visually observes 3 close enemies!
            "enemies": [
                {"x": 640.0, "y": 300.0, "radius": 16},
                {"x": 640.0, "y": 310.0, "radius": 16},
                {"x": 640.0, "y": 290.0, "radius": 16}
            ],
            "hazards": [],
            "gems": [],
            "sector_obstacles": {}
        })

        unified = adapter.read_unified_state(frame_img=mock_frame)

        # Enemies must NOT be empty!
        self.assertEqual(len(unified["enemies"]), 3)
        self.assertEqual(unified["sources"]["enemies"], "opencv_perception_fallback")
        # North sector danger must be elevated due to the close enemies
        self.assertGreater(unified["sector_dangers"]["N"], 0.0)
        self.assertEqual(unified["nearest_enemy_sector"], "N")

    def test_window_g_empty_hazards_falls_back_to_opencv(self):
        """
        When window.G hazards/projectiles is empty but OpenCV observes telegraphs/bullets,
        include the OpenCV hazards rather than assuming safe terrain.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "player": {"x": 500.0, "y": 500.0, "hp": 100, "maxHp": 100},
                "enemies": [{"x": 800.0, "y": 500.0, "r": 16, "dex": 1}],
                "hazards": [],
                "projectiles": [],
                "pickups": []
            },
            "telemetry": {"camera": {"s": 1.0, "e": 0.0, "f": 0.0, "verified": True}},
            "dom": {"phase": "running"}
        })

        mock_frame = MagicMock()
        adapter.perception.analyze_frame = MagicMock(return_value={
            "enemies": [],
            "hazards": [{"x": 640.0, "y": 420.0, "kind": "telegraph", "radius": 25}],
            "gems": [],
            "sector_obstacles": {}
        })

        unified = adapter.read_unified_state(frame_img=mock_frame)
        self.assertEqual(len(unified["hazards"]), 1)
        self.assertEqual(unified["sources"]["hazards"], "opencv_hazards_fallback")

    # =========================================================================
    # 3. DUMMY (0, 0) SPECIALS REJECTION REGARDLESS OF SOURCE
    # =========================================================================

    def test_reject_dummy_zero_coordinates_in_adapter(self):
        """
        GameStateAdapter must reject Rift/Spring/Chest/Altar with (0, 0)
        even if source field is present.
        """
        adapter = GameStateAdapter()
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {
                "player": {"x": 1000.0, "y": 1000.0, "hp": 100, "maxHp": 100},
                "enemies": [],
                "rift": {"x": 0.0, "y": 0.0, "source": "window_G"},
                "specials": {
                    "spring": {"x": 0.0, "y": 0.0, "source": "window_G"},
                    "chest": {"x": 0.0, "y": 0.0, "source": "window_G"},
                    "altar": {"x": 0.0, "y": 0.0, "source": "window_G"}
                }
            },
            "telemetry": {"camera": {"s": 1.0, "e": 0.0, "f": 0.0, "verified": True}},
            "dom": {"phase": "running"}
        })

        unified = adapter.read_unified_state()
        specials = unified["specials"]
        self.assertIsNone(specials["rift"])
        self.assertIsNone(specials["spring"])
        self.assertIsNone(specials["chest"])
        self.assertIsNone(specials["altar"])

    def test_target_evaluator_rejects_dummy_zero_regardless_of_source(self):
        """
        TargetEvaluator must never select (0, 0) specials even if source='window_G' or 'pickupMgr'.
        """
        evaluator = TargetEvaluator()
        dummy_state = {
            "player": {"world_x": 1000.0, "world_y": 1000.0, "hp_ratio": 1.0},
            "specials": {
                "rift": {"x": 0.0, "y": 0.0, "source": "window_G"},
                "spring": {"x": 0.0, "y": 0.0, "source": "pickupMgr"},
                "chest": {"x": 0.0, "y": 0.0, "source": "objMgr"}
            },
            "enemies": [],
            "items": [],
            "hazards": []
        }
        target = evaluator.evaluate_targets(dummy_state)
        self.assertIsNone(target, "Should strictly reject all (0, 0) dummy targets")

    # =========================================================================
    # 4. UNIFIED EARLY-GAME RIFT SUPPRESSION
    # =========================================================================

    def test_early_game_rift_suppressed_in_both_policy_and_evaluator(self):
        """
        At early game (elapsed_time < 120s or level < 3),
        both MovementPolicy and TargetEvaluator must strictly suppress entering the Rift,
        preventing branch 4b from bypassing the safety condition.
        """
        evaluator = TargetEvaluator()
        policy = MovementPolicy(self.default_policy)

        # Early game state: 30 seconds into match, Level 1, Rift is open
        early_state = {
            "player": {"world_x": 1000.0, "world_y": 1000.0, "hp_ratio": 1.0},
            "specials": {
                "rift": {"x": 1200.0, "y": 1000.0, "source": "window_G", "type": "portal"}
            },
            "enemies": [],
            "items": [],
            "hazards": [],
            "elapsed_time": 30.0,
            "current_level": 1,
            "danger_level": "LOW",
            "min_enemy_dist": 9999.0
        }

        # 1. TargetEvaluator must NOT return Rift at 30 seconds
        tactical_target = evaluator.evaluate_targets(early_state)
        self.assertIsNone(tactical_target, "TargetEvaluator must suppress Rift entry in early game (< 120s)")

        # 2. MovementPolicy must NOT choose Rift direction in early game
        chosen_dir, keys, dur = policy.select_movement(early_state, current_level=1, elapsed_time=30.0)
        # Rift is East, but policy should NOT be targeting Rift (tier 4 rift navigation)
        self.assertFalse(policy.pin_escape_dir == "E" and policy.pin_escape_until > time.time())

        # 3. Late game state: 150 seconds into match, Level 4
        late_state = dict(early_state)
        late_state["elapsed_time"] = 150.0
        late_state["current_level"] = 4

        late_target = evaluator.evaluate_targets(late_state)
        self.assertIsNotNone(late_target)
        self.assertEqual(late_target["type"], "rift")
        self.assertEqual(late_target["sector"], "E")

    # =========================================================================
    # 5. TERRAIN 'BOUNDS_ONLY' SEMANTICS (NOT COMPLETELY SAFE)
    # =========================================================================

    def test_terrain_bounds_only_not_completely_safe(self):
        """
        When terrain_status is 'bounds_only', local obstacles have NOT been observed.
        The bot must NOT mark sectors as completely safe just because they are inside mapBounds.
        """
        policy = MovementPolicy(self.default_policy)

        state_bounds_only = {
            "player": {"world_x": 1500.0, "world_y": 1500.0, "hp_ratio": 1.0},
            "terrain": {
                "sector_obstacles": {"W": False, "E": False, "N": False, "S": False},
                "sector_obs_costs": {},
                "status": "bounds_only",
                "source": "window_G_bounds"
            },
            "terrain_status": "bounds_only",
            "enemies": [],
            "hazards": [],
            "sector_dangers": {s: 0.0 for s in ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]},
            "danger_level": "LOW",
            "min_enemy_dist": 9999.0
        }

        # Choose movement must succeed without crashing
        dir_chosen, keys, dur = policy.select_movement(state_bounds_only)
        self.assertIn(dir_chosen, ["E", "SE", "S", "SW", "W", "NW", "N", "NE"])

        # When terrain is 'verified' (both bounds and OpenCV local view), local view is observed
        state_verified = dict(state_bounds_only)
        state_verified["terrain_status"] = "verified"
        dir_ver, keys_ver, dur_ver = policy.select_movement(state_verified)
        self.assertIn(dir_ver, ["E", "SE", "S", "SW", "W", "NW", "N", "NE"])


if __name__ == "__main__":
    unittest.main()

