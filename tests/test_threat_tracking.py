import unittest
import time
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

if __name__ == "__main__":
    unittest.main()
