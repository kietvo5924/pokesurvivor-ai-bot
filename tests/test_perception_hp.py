import unittest
import os
import cv2
import numpy as np
from pokesurvivor.perception import PerceptionEngine

class TestPerceptionHP(unittest.TestCase):
    def setUp(self):
        self.perception = PerceptionEngine()
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")

    def test_full_hp_detection(self):
        img_path = os.path.join(self.fixtures_dir, "hp_full.png")
        self.assertTrue(os.path.exists(img_path), f"Missing repo test fixture: {img_path}")
        img = cv2.imread(img_path)
        res = self.perception._detect_hp_ratio(img)
        self.assertIsNotNone(res["hp_ratio"])
        self.assertAlmostEqual(res["hp_ratio"], 1.0, places=1)
        self.assertEqual(res["status"], "VERIFIED")
        self.assertIsNotNone(res["box"])

    def test_damaged_hp_detection(self):
        img_path = os.path.join(self.fixtures_dir, "hp_damaged.png")
        self.assertTrue(os.path.exists(img_path), f"Missing repo test fixture: {img_path}")
        img = cv2.imread(img_path)
        res = self.perception._detect_hp_ratio(img)
        self.assertIsNotNone(res["hp_ratio"])
        # Expected ~0.484 (60/124 HP in screenshot), verified measurement is ~0.483
        self.assertTrue(0.40 <= res["hp_ratio"] <= 0.58)
        self.assertEqual(res["status"], "VERIFIED")
        self.assertIsNotNone(res["box"])

    def test_estimated_hp_when_track_unverified(self):
        # Synthetic frame with active bar but missing dark container track
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Coral active bar: B=80, G=80, R=240, 50px wide
        img[25:35, 45:95] = [80, 80, 240]
        res = self.perception._detect_hp_ratio(img)
        self.assertIsNotNone(res["hp_ratio"])
        self.assertEqual(res["status"], "ESTIMATED")
        self.assertAlmostEqual(res["hp_ratio"], 50.0 / 142.0, places=2)

    def test_match_end_or_hidden_hp(self):
        img_path = os.path.join(self.fixtures_dir, "match_end.png")
        self.assertTrue(os.path.exists(img_path), f"Missing repo test fixture: {img_path}")
        img = cv2.imread(img_path)
        res = self.perception._detect_hp_ratio(img)
        # Never default to 100% when dead or unreadable
        self.assertIsNone(res["hp_ratio"])
        self.assertEqual(res["status"], "UNKNOWN")

    def test_blank_frame_returns_unknown(self):
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        res = self.perception._detect_hp_ratio(blank)
        self.assertIsNone(res["hp_ratio"])
        self.assertEqual(res["status"], "UNKNOWN")

    def test_opencv5_drawing_with_float_coordinates(self):
        """
        Verify that all cv2.circle, cv2.rectangle, cv2.arrowedLine and cv2.putText
        calls in PerceptionEngine handle float coordinates seamlessly without
        raising OpenCV 5.0.0 'Sequencing item with index 0 has a wrong type' error.
        """
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Mask detection with float center
        obs = self.perception._detect_obstacle_mask(hsv, cx=640.0, cy=360.0)
        self.assertEqual(obs.shape, (720, 1280))

        gems, _ = self.perception._detect_gems(hsv, cx=640.0, cy=360.0)
        self.assertIsInstance(gems, list)

        hazards = self.perception._detect_hazards_and_projectiles(hsv, cx=640.0, cy=360.0, enemies=[])
        self.assertIsInstance(hazards, list)

        enemies, items = self.perception._detect_entities(hsv, cx=640.0, cy=360.0)
        self.assertIsInstance(enemies, list)

        rift = self.perception._detect_purple_rift(hsv, cx=640.0, cy=360.0)

        # 2. Debug overlay with world/float coordinates
        unified_state = {
            "coord_space": "world",
            "player": {"world_x": 1000.0, "world_y": 1000.0, "screen_x": 640.0, "screen_y": 360.0},
            "player_pos": (1000.0, 1000.0),
            "hp_ratio": 0.75,
            "hp_status": "VERIFIED",
            "enemies": [
                {"x": 1050.5, "y": 1020.2, "screen_x": 690.5, "screen_y": 380.2, "dist": 53.8},
                {"x": 980.0, "y": 950.0, "dist": 53.8}
            ],
            "hazards": [
                {"x": 1010.5, "y": 1010.5, "screen_x": 650.5, "screen_y": 370.5, "vx": 12.5, "vy": -5.2, "ttc": 2.3, "is_imminent": True}
            ]
        }
        overlay = self.perception.create_debug_overlay(frame, unified_state, chosen_dir="NE", output_path="debug_artifacts/test_float.png")
        self.assertEqual(overlay.shape, (720, 1280, 3))

if __name__ == "__main__":
    unittest.main()

