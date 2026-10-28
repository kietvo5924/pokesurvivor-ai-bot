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

if __name__ == "__main__":
    unittest.main()

