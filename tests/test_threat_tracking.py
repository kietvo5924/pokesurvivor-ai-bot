import unittest
import time
import numpy as np
from pokesurvivor.perception import PerceptionEngine
from pokesurvivor.movement_policy import MovementPolicy

class TestThreatTrackingAndEvasion(unittest.TestCase):
    def setUp(self):
        self.perception = PerceptionEngine()
        self.movement = MovementPolicy({})

    def test_hazard_velocity_and_ttc_calculation(self):
        cx, cy = 640, 360
        # Frame 1: bullet at (500, 360), moving East towards player
        h_frame1 = [{"x": 500, "y": 360, "dist": 140.0, "type": "poison_bullet"}]
        self.perception._track_hazards(h_frame1, cx, cy)

        # Simulate 0.1s time elapsed, bullet moves 15px East to (515, 360)
        time.sleep(0.05)
        h_frame2 = [{"x": 515, "y": 360, "dist": 125.0, "type": "poison_bullet"}]
        tracked = self.perception._track_hazards(h_frame2, cx, cy)

        self.assertEqual(len(tracked), 1)
        bullet = tracked[0]
        # Bullet should have positive vx (moving East towards cx=640)
        self.assertGreater(bullet["vx"], 0)
        self.assertAlmostEqual(bullet["vy"], 0.0, delta=20.0)
        self.assertGreater(bullet["speed"], 50.0)
        # Should have reasonable TTC (< 1.5s)
        self.assertLess(bullet["ttc"], 2.0)
        # Should detect as imminent threat since it's on a direct collision course
        self.assertTrue(bullet["is_imminent"])

    def test_first_seen_bullet_does_not_assume_direct_player_velocity(self):
        cx, cy = 640, 360
        # Bullet seen for the very first time at (500, 360), player at (640, 360)
        self.perception.prev_hazards = []
        h_frame = [{"x": 500, "y": 360, "dist": 140.0, "type": "poison_bullet"}]
        tracked = self.perception._track_hazards(h_frame, cx, cy)
        self.assertEqual(len(tracked), 1)
        # Without multi-frame velocity evidence, vx and vy MUST remain 0.0
        self.assertEqual(tracked[0]["vx"], 0.0)
        self.assertEqual(tracked[0]["vy"], 0.0)
        self.assertEqual(tracked[0]["speed"], 0.0)

    def test_ground_items_and_chests_never_in_enemies(self):
        # Create a synthetic HSV frame with a golden chest and a regular purple enemy
        hsv = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Background is walkable grass: H: 60, S: 150, V: 120
        hsv[:, :] = [60, 150, 120]
        
        # 1. Golden chest at (400, 360), size 24x24 (aspect 1.0, area ~576), gold HSV: H=25, S=180, V=220
        hsv[348:372, 388:412] = [25, 180, 220]

        # 2. Purple monster at (800, 360), size 30x40, purple HSV: H=140, S=180, V=150
        hsv[340:380, 785:815] = [140, 180, 150]

        enemies, items = self.perception._detect_entities(hsv, 640, 360)
        
        # Verify items list has the chest
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "chest")
        self.assertAlmostEqual(items[0]["x"], 400, delta=10)

        # Verify enemies list has the monster
        self.assertEqual(len(enemies), 1)
        self.assertEqual(enemies[0]["type"], "enemy")
        self.assertAlmostEqual(enemies[0]["x"], 800, delta=10)

        # Crucial check: items MUST NOT be included in enemies list!
        for e in enemies:
            self.assertNotEqual(e.get("type"), "chest")
            dist_to_chest = ((e["x"] - 400)**2 + (e["y"] - 360)**2)**0.5
            self.assertGreater(dist_to_chest, 50.0)

    def test_multi_hazard_evasion_direction(self):
        cx, cy = 640, 360
        # Two incoming bullets from West (vx > 0)
        hazards = [
            {"x": 560, "y": 360, "vx": 150.0, "vy": 0.0, "ttc": 0.5, "dist": 80.0, "is_imminent": True},
            {"x": 540, "y": 370, "vx": 150.0, "vy": 0.0, "ttc": 0.6, "dist": 100.0, "is_imminent": True}
        ]
        sector_obstacles = {s: False for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
        sector_enemy_counts = {s: 0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
        sector_dangers = {s: 0.0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}

        # West is where bullets come from -> bot should NEVER choose W!
        # Moving E might keep player in the bullet path. Lateral directions (N, S, NE, SE) are safest!
        dodge_res = self.movement._evaluate_multi_hazard_evasion(
            hazards, cx, cy, sector_obstacles, sector_enemy_counts, sector_dangers
        )
        self.assertIsNotNone(dodge_res)
        dodge_dir, hold_t = dodge_res
        self.assertNotEqual(dodge_dir, "W")
        self.assertIn(dodge_dir, ["E", "N", "S", "NE", "SE", "NW", "SW"])

    def test_evasion_avoids_blocked_terrain(self):
        cx, cy = 640, 360
        # Bullet coming from South towards North (vy < 0)
        hazards = [
            {"x": 640, "y": 440, "vx": 0.0, "vy": -140.0, "ttc": 0.5, "dist": 80.0, "is_imminent": True}
        ]
        # North is blocked by cliff wall!
        sector_obstacles = {s: False for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
        sector_obstacles["N"] = True
        sector_enemy_counts = {s: 0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
        sector_dangers = {s: 0.0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}

        dodge_res = self.movement._evaluate_multi_hazard_evasion(
            hazards, cx, cy, sector_obstacles, sector_enemy_counts, sector_dangers
        )
        self.assertIsNotNone(dodge_res)
        dodge_dir, hold_t = dodge_res
        # Bot must not dodge into the blocked wall (North) or into the bullet (South)
        self.assertNotEqual(dodge_dir, "N")
        self.assertNotEqual(dodge_dir, "S")

    def test_swarm_tangential_kiting(self):
        cx, cy = 640, 360
        state = {
            "player_pos": (cx, cy),
            "hp_ratio": 0.9,
            "hp_status": "VERIFIED",
            "min_enemy_dist": 200.0,
            "swarm_center": (cx - 100, cy),  # Swarm is directly West
            "danger_level": "HIGH",
            "hazards": [],
            "sector_obstacles": {s: False for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]},
            "sector_obs_costs": {s: 0.0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]},
            "sector_enemy_counts": {s: 0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]},
            "sector_dangers": {s: 0.0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
        }
        dir_name, keys, dur = self.movement.select_movement(state)
        self.assertIn(dir_name, ["N", "S", "NE", "SE", "E"])

if __name__ == "__main__":
    unittest.main()
