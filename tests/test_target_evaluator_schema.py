"""
tests/test_target_evaluator_schema.py
--------------------------------------
Verifies schema normalization and candidate evaluation for TargetEvaluator:
- Tier 5: Golden Thief (Gholdengo)
- Tier 4: Purple Rift Portal (when healthy and not emergency)
- Tier 4 (Emergency): Healing Springs when critically hurt (HP < 0.40)
- Tier 3: Coin Thief (Meowth)
- Tier 2: Healing Springs (when hurt 0.40 <= HP < 0.85)
- Tier 2: Big Chests & Altars
- Tier 1: Valuable Ground Pickups (chest, magnet, berry)
- Emergency mode: Sudden burst damage (>= 25% drop) suppresses non-emergency targets
- Ignore list TTL: Depleted springs and opened chests are ignored for 60s
- Cross-schema compatibility: Both nested unified_state and flat vision_state
"""

import unittest
import time
from pokesurvivor.target_evaluator import TargetEvaluator, normalize_game_state

class TestTargetEvaluatorSchema(unittest.TestCase):
    def setUp(self):
        self.evaluator = TargetEvaluator()
        self.now = 1000.0

    def test_schema_normalization_from_flat_vision_state(self):
        """Verifies flat vision_state is normalized into standard schema."""
        flat_state = {
            "hp_ratio": 0.45,
            "player_pos": (600, 350),
            "rift_target": {"sector": "NE", "dist": 220.0, "x": 150, "y": -150},
            "spring": {"sector": "SW", "dist": 180.0, "x": -100, "y": 100},
            "chest": {"sector": "E", "dist": 160.0, "x": 160, "y": 0},
            "enemies": [
                {"name": "Gholdengo", "is_thief": True, "type": 1000, "dist": 190.0, "sector": "SE"}
            ],
            "items_on_ground": [
                {"kind": "magnet", "dist": 90.0, "x": 50, "y": 50}
            ],
            "gems": [
                {"dist": 40.0, "x": 20, "y": 20}
            ]
        }
        norm = normalize_game_state(flat_state)
        self.assertAlmostEqual(norm["player"]["hp_ratio"], 0.45)
        self.assertEqual(norm["player"]["world_x"], 600.0)
        self.assertIsNotNone(norm["specials"]["rift"])
        self.assertIsNotNone(norm["specials"]["spring"])
        self.assertIsNotNone(norm["specials"]["chest"])
        self.assertEqual(len(norm["enemies"]), 1)
        self.assertGreaterEqual(len(norm["items"]), 2)

    def test_schema_normalization_from_unified_state(self):
        """Verifies nested unified_state is normalized cleanly."""
        unified_state = {
            "player": {"hp_ratio": 0.88, "world_x": 700.0, "world_y": 400.0, "source": "window_G"},
            "specials": {
                "rift": {"sector": "N", "dist": 300.0, "x": 0, "y": -300},
                "spring": None,
                "chest": {"sector": "W", "dist": 120.0, "x": -120, "y": 0}
            },
            "enemies": [
                {"dex": 52, "name": "Meowth", "is_thief": True, "dist": 110.0, "sector": "S"}
            ],
            "items": [
                {"kind": "chest", "dist": 80.0, "x": 50, "y": -50}
            ]
        }
        norm = normalize_game_state(unified_state)
        self.assertAlmostEqual(norm["player"]["hp_ratio"], 0.88)
        self.assertEqual(norm["player"]["world_x"], 700.0)
        self.assertIsNotNone(norm["specials"]["rift"])
        self.assertIsNotNone(norm["specials"]["chest"])
        self.assertEqual(len(norm["enemies"]), 1)
        self.assertEqual(len(norm["items"]), 1)

    def test_tier5_gholdengo_top_priority(self):
        """Tier 5: Golden Thief Gholdengo must beat Rift, Meowth, Chests, and Springs."""
        state = {
            "hp_ratio": 0.95,
            "enemies": [
                {"name": "Gholdengo", "is_thief": True, "type": 1000, "dist": 200.0, "sector": "NE", "x": 100, "y": -100},
                {"name": "Meowth", "is_thief": True, "type": 52, "dist": 100.0, "sector": "S", "x": 0, "y": 100}
            ],
            "rift_target": {"sector": "N", "dist": 220.0, "x": 0, "y": -220},
            "chest": {"sector": "E", "dist": 120.0, "x": 120, "y": 0}
        }
        target = self.evaluator.evaluate_targets(state, now=self.now)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "thief_gold")
        self.assertEqual(target["tier"], 5)
        self.assertEqual(target["mode"], "CHASE_GHOLDENGO")

    def test_tier4_rift_portal_when_healthy(self):
        """Tier 4: Purple Rift Portal is selected over Meowth and Chests when healthy."""
        state = {
            "player": {"hp_ratio": 0.90},
            "specials": {
                "rift": {"sector": "NW", "dist": 250.0, "x": -150, "y": -150},
                "chest": {"sector": "E", "dist": 150.0, "x": 150, "y": 0}
            },
            "enemies": [
                {"name": "Rattata", "dist": 180.0, "sector": "S"}
            ]
        }
        target = self.evaluator.evaluate_targets(state, now=self.now)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "rift")
        self.assertEqual(target["tier"], 4)
        self.assertEqual(target["mode"], "ENTER_RIFT")

    def test_tier4_emergency_healing_spring_when_critical(self):
        """Healing spring escalates to Tier 4 when HP < 0.40."""
        state = {
            "player": {"hp_ratio": 0.25}, # Critically low HP!
            "specials": {
                "spring": {"sector": "SW", "dist": 190.0, "x": -100, "y": 100},
                "chest": {"sector": "E", "dist": 120.0, "x": 120, "y": 0}
            }
        }
        target = self.evaluator.evaluate_targets(state, now=self.now)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "spring")
        self.assertEqual(target["tier"], 4)
        self.assertEqual(target["mode"], "HEAL_AT_SPRING")

    def test_tier3_meowth_coin_thief(self):
        """Tier 3: Meowth coin thief prioritized over chests and normal ground items."""
        state = {
            "hp_ratio": 0.85,
            "enemies": [
                {"name": "Meowth", "is_thief": True, "type": 52, "dist": 140.0, "sector": "W", "x": -140, "y": 0}
            ],
            "chest": {"sector": "E", "dist": 150.0, "x": 150, "y": 0}
        }
        target = self.evaluator.evaluate_targets(state, now=self.now)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "thief_coin")
        self.assertEqual(target["tier"], 3)
        self.assertEqual(target["mode"], "CHASE_MEOWTH")

    def test_tier2_chest_and_altar(self):
        """Tier 2: Chests and Altars targeted when safe and healthy."""
        state = {
            "hp_ratio": 1.0,
            "specials": {
                "chest": {"sector": "SE", "dist": 160.0, "x": 100, "y": 100}
            }
        }
        target = self.evaluator.evaluate_targets(state, now=self.now)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "chest")
        self.assertEqual(target["tier"], 2)
        self.assertEqual(target["mode"], "OPEN_CHEST")

    def test_tier1_valuable_ground_pickups(self):
        """Tier 1: High value ground pickups (chests, magnets, berries) evaluated."""
        state = {
            "hp_ratio": 1.0,
            "items": [
                {"kind": "magnet", "dist": 120.0, "x": 50, "y": -50, "sector": "NE"},
                {"kind": "xp", "dist": 80.0, "x": 30, "y": 30, "sector": "SE"}
            ]
        }
        target = self.evaluator.evaluate_targets(state, now=self.now)
        self.assertIsNotNone(target)
        self.assertEqual(target["type"], "item_magnet")
        self.assertEqual(target["tier"], 1)

    def test_emergency_burst_damage_suppression(self):
        """Sudden >= 25% burst damage activates emergency survival and suppresses non-emergency targets."""
        # Frame 1: HP at 90%
        state1 = {"hp_ratio": 0.90, "chest": {"sector": "E", "dist": 100.0, "x": 100, "y": 0}}
        self.evaluator.evaluate_targets(state1, now=self.now)

        # Frame 2 (0.5s later): HP drops to 60% (-30% shock!)
        state2 = {"hp_ratio": 0.60, "chest": {"sector": "E", "dist": 100.0, "x": 100, "y": 0}}
        target = self.evaluator.evaluate_targets(state2, now=self.now + 0.5)
        # Chest should be suppressed because of emergency mode!
        self.assertIsNone(target)

    def test_ignore_list_ttl_prevents_stalling(self):
        """Opened chest or depleted spring is ignored for 60s."""
        state = {
            "hp_ratio": 0.90,
            "chest": {"sector": "E", "dist": 30.0, "x": 100, "y": 0}
        }
        # First arrival at chest (dist < 45px)
        t0 = self.now
        self.evaluator.evaluate_targets(state, now=t0)
        # Stood for 4.0s (chest opened!)
        target_after = self.evaluator.evaluate_targets(state, now=t0 + 4.0)
        # Must return None and chest must be in ignore_list for 60s
        self.assertIsNone(target_after)
        self.assertIn("chest_100_0", self.evaluator.ignore_list)
        self.assertGreater(self.evaluator.ignore_list["chest_100_0"], t0 + 55.0)

if __name__ == "__main__":
    unittest.main()
