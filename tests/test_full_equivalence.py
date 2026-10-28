"""
Unit tests verifying complete functional equivalence with PokeSurvivorTool:
1. GameStateAdapter (window.G, Canvas telemetry texts & transforms, DOM, OpenCV fallback)
2. TargetEvaluator (Candidates Tier 5-0, ignore list TTL, emergency survival)
3. CoinTracker (Currency metrics separated from internal fitness)
4. UpgradePolicy (Held item tier list, Choice Band avoidance, Chest move upgrades)
5. MovementPolicy (Low HP double repulsion, tactical target navigation)
"""

import unittest
import time
import os
import tempfile
import json
from pokesurvivor.game_state_adapter import GameStateAdapter
from pokesurvivor.target_evaluator import TargetEvaluator
from pokesurvivor.coin_tracker import CoinTracker
from pokesurvivor.upgrade_policy import UpgradePolicy
from pokesurvivor.movement_policy import MovementPolicy

class TestFullEquivalence(unittest.TestCase):
    def setUp(self):
        self.default_policy = {
            "version": "10.2",
            "movement": {
                "pattern": "kiting",
                "step_duration": 0.22,
                "danger_threshold": 160.0
            },
            "upgrade": {
                "priority_moves": ["Water Gun", "Ember", "Disarming Voice"],
                "buff_moves": ["Aqua Ring", "Swords Dance"],
                "priority_stats": ["Power", "Sprint", "Reflexes"]
            }
        }

    # ------------------------------------------------------------------
    # 1. GameStateAdapter Tests
    # ------------------------------------------------------------------
    def test_adapter_window_g_extraction(self):
        adapter = GameStateAdapter(page=None)
        mock_browser_data = {
            "hasG": True,
            "gState": {
                "player": {
                    "x": 1200.0, "y": 800.0, "r": 16,
                    "hp": 150, "maxHp": 200, "dex": 788,
                    "speed": 120.0, "magnet": 65.0,
                    "moves": ["water gun", "disarming voice"],
                    "stats": ["power"]
                },
                "enemies": [
                    {"x": 1250.0, "y": 800.0, "r": 16, "dex": 19, "hp": 40, "behavior": "chase"}
                ],
                "hazards": [
                    {"kind": "zone", "x": 1220.0, "y": 800.0, "r": 30}
                ],
                "projectiles": [
                    {"x": 1240.0, "y": 800.0, "r": 8, "vx": -5.0, "vy": 0.0, "friendly": False}
                ],
                "pickups": [
                    {"x": 1210.0, "y": 790.0, "kind": "xp"}
                ],
                "rift": {"x": 1500.0, "y": 1000.0, "inArena": False}
            }
        }
        player = adapter._extract_player(mock_browser_data)
        self.assertEqual(player["world_x"], 1200.0)
        self.assertEqual(player["world_y"], 800.0)
        self.assertEqual(player["hp"], 150)
        self.assertEqual(player["max_hp"], 200)
        self.assertAlmostEqual(player["hp_ratio"], 0.75)
        self.assertEqual(player["source"], "window_G")

        entities = adapter._extract_entities(mock_browser_data, frame_img=None, is_moving=True, player=player)
        self.assertEqual(len(entities["enemies"]), 1)
        self.assertEqual(entities["enemies"][0]["dist"], 50.0)
        self.assertEqual(entities["enemies"][0]["sector"], "E")
        self.assertEqual(len(entities["hazards"]), 2) # 1 zone + 1 shot
        self.assertEqual(len(entities["items"]), 1)
        self.assertIsNotNone(entities["specials"]["rift"])

    def test_adapter_canvas_telemetry_extraction(self):
        adapter = GameStateAdapter(page=None)
        mock_browser_data = {
            "hasG": False,
            "telemetry": {
                "texts": [
                    "Tapu Fini", "Lv. 5", "154 / 220", "2:45",
                    "SHADY FOREST · 452 DEFEATED · 85 COINS",
                    "Water Gun", "AIM: AUTO"
                ],
                "camera": {"s": 1.666, "e": 640.0, "f": 360.0}
            }
        }
        player = adapter._extract_player(mock_browser_data)
        self.assertEqual(player["hp"], 154)
        self.assertEqual(player["max_hp"], 220)
        self.assertAlmostEqual(player["hp_ratio"], 154 / 220)
        self.assertEqual(player["level"], 5)
        self.assertEqual(player["source"], "canvas_telemetry_texts")

        metrics = adapter._extract_metrics(mock_browser_data)
        self.assertEqual(metrics["kills"], 452)
        self.assertEqual(metrics["coins"], 85)
        self.assertEqual(metrics["time_str"], "2:45")
        self.assertEqual(metrics["source"], "canvas_hud_texts")

    def test_adapter_unknown_fields_not_fabricated(self):
        adapter = GameStateAdapter(page=None)
        mock_browser_data = {"hasG": False, "telemetry": {"texts": []}}
        metrics = adapter._extract_metrics(mock_browser_data)
        self.assertIsNone(metrics["kills"])
        self.assertIsNone(metrics["coins"])
        self.assertEqual(metrics["source"], "unobserved")

    # ------------------------------------------------------------------
    # 2. TargetEvaluator Tests
    # ------------------------------------------------------------------
    def test_target_evaluator_hierarchy(self):
        evaluator = TargetEvaluator()
        now = time.time()
        game_state = {
            "player": {"hp_ratio": 0.90},
            "enemies": [
                {"type": 1000, "is_thief": True, "name": "Gholdengo", "dist": 200.0, "sector": "E", "x": 100, "y": 200},
                {"type": 52, "is_thief": True, "name": "Meowth", "dist": 120.0, "sector": "W", "x": 50, "y": 50}
            ],
            "specials": {
                "rift": {"sector": "N", "dist": 250.0, "x": 0, "y": -250},
                "chest": {"sector": "S", "dist": 150.0, "x": 0, "y": 150}
            }
        }
        # Golden Thief Gholdengo is Tier 5 (highest priority)
        target = evaluator.evaluate_targets(game_state, now=now)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "thief_gold")
        self.assertEqual(target["tier"], 5)

    def test_target_evaluator_healing_spring_when_low_hp(self):
        evaluator = TargetEvaluator()
        now = time.time()
        game_state = {
            "player": {"hp_ratio": 0.30}, # Hurt
            "specials": {
                "spring": {"sector": "NE", "dist": 180.0, "x": 100, "y": -100},
                "chest": {"sector": "S", "dist": 150.0, "x": 0, "y": 150}
            }
        }
        target = evaluator.evaluate_targets(game_state, now=now)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "spring")
        self.assertEqual(target["mode"], "HEAL_AT_SPRING")

    def test_target_evaluator_ignore_list_ttl(self):
        evaluator = TargetEvaluator()
        now = time.time()
        # Mark chest as ignored for 60s
        evaluator.ignore_list["chest_0_150"] = now + 60.0
        game_state = {
            "player": {"hp_ratio": 0.90},
            "specials": {
                "chest": {"sector": "S", "dist": 150.0, "x": 0, "y": 150}
            }
        }
        target = evaluator.evaluate_targets(game_state, now=now)
        self.assertIsNone(target) # Chest is ignored!

    # ------------------------------------------------------------------
    # 3. CoinTracker Tests
    # ------------------------------------------------------------------
    def test_coin_tracker_accumulation(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            tracker = CoinTracker(filepath=temp_path)
            tracker.record_run(coins=150, kills=300, survival_time=180.0, level=15, bosses=1)
            tracker.record_run(coins=250, kills=500, survival_time=240.0, level=20, bosses=2)

            summary = tracker.get_summary()
            self.assertEqual(summary["total_runs"], 2)
            self.assertEqual(summary["total_coins"], 400)
            self.assertEqual(summary["last_run_coins"], 250)
            self.assertGreater(summary["coins_per_hour"], 0.0)

            # Reload from disk
            tracker2 = CoinTracker(filepath=temp_path)
            self.assertEqual(tracker2.data["total_coins"], 400)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ------------------------------------------------------------------
    # 4. UpgradePolicy Equivalence Tests
    # ------------------------------------------------------------------
    def test_held_item_ranking_and_choice_band_avoidance(self):
        policy = UpgradePolicy(self.default_policy)
        
        # Test 1: Equip Shell Bell over Choice Band
        choices = [
            {"name": "Choice Band", "key": "1"},
            {"name": "Shell Bell", "key": "2"}
        ]
        picked = policy.choose_card(choices, modal_kind="item")
        self.assertIn("shell bell", picked["name"].lower())

        # Test 2: If Choice Band is currently held, ALWAYS switch away!
        policy.held_item = "choice band"
        choices2 = [
            {"name": "Keep Choice Band", "key": "1"},
            {"name": "Lucky Egg", "key": "2"}
        ]
        picked2 = policy.choose_card(choices2, modal_kind="item")
        self.assertIn("lucky egg", picked2["name"].lower())

    def test_chest_build_move_upgrade_priority(self):
        policy = UpgradePolicy(self.default_policy)
        policy.acquired_moves["water gun"] = 2
        choices = [
            {"name": "Rare Candy", "desc": "Gain 1 level", "key": "1"},
            {"name": "Upgrade: Water Gun", "desc": "Power +20%", "key": "2"},
            {"name": "Gacha Ticket x3", "desc": "Currency", "key": "3"}
        ]
        picked = policy.choose_card(choices, modal_kind="chest")
        # Build move upgrade (+900) beats Rare Candy (+700)
        self.assertIn("water gun", picked["name"].lower())

    # ------------------------------------------------------------------
    # 5. MovementPolicy Equivalence Tests
    # ------------------------------------------------------------------
    def test_low_hp_double_repulsion(self):
        policy = MovementPolicy(self.default_policy)
        vision_state = {
            "player_pos": (640, 360),
            "hp_ratio": 0.25, # Low HP -> triggers 2.0x repulsion multiplier
            "min_enemy_dist": 60.0,
            "nearest_enemy_sector": "S",
            "enemies": [
                {"x": 640, "y": 420, "dist": 60.0, "sector": "S"}
            ],
            "sector_enemy_counts": {"S": 1},
            "sector_dangers": {"S": 200.0, "N": 10.0},
            "sector_obstacles": {},
            "sector_obs_costs": {},
            "danger_level": "CRITICAL",
            "is_encircled": False
        }
        chosen_dir, keys, duration = policy.select_movement(vision_state)
        # S has enemy right at 60px; bot must flee strongly away (e.g. N / NW / NE)
        self.assertIn(chosen_dir, ["N", "NW", "NE"])

if __name__ == "__main__":
    unittest.main()
