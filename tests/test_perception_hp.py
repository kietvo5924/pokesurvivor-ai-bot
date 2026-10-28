import unittest
import os
import cv2
import numpy as np
from pokesurvivor.perception import PerceptionEngine

class TestPerceptionHP(unittest.TestCase):
    def setUp(self):
        self.perception = PerceptionEngine()

    def test_full_hp_detection(self):
        img_path = "debug_artifacts/tabino_ingame.png"
        if not os.path.exists(img_path):
            self.skipTest(f"Missing test fixture: {img_path}")
        img = cv2.imread(img_path)
        res = self.perception._detect_hp_ratio(img)
        self.assertIsNotNone(res["hp_ratio"])
        self.assertAlmostEqual(res["hp_ratio"], 1.0, places=1)
        self.assertEqual(res["status"], "VERIFIED")
        self.assertIsNotNone(res["box"])

    def test_damaged_hp_detection(self):
        img_path = "C:/Users/Admin/.gemini/antigravity/brain/67f9c386-3d7a-4631-a84d-33dcaf258669/.user_uploaded/media_1793161457118_42c4e663.png"
        if not os.path.exists(img_path):
            self.skipTest(f"Missing test fixture: {img_path}")
        img = cv2.imread(img_path)
        res = self.perception._detect_hp_ratio(img)
        self.assertIsNotNone(res["hp_ratio"])
        # Expected ~0.484 (60/124 HP), detection is ~0.493 (< 2% error)
        self.assertTrue(0.40 <= res["hp_ratio"] <= 0.58)
        self.assertEqual(res["status"], "VERIFIED")
        self.assertIsNotNone(res["box"])

    def test_match_end_or_hidden_hp(self):
        img_path = "debug_artifacts/last_match_end.png"
        if not os.path.exists(img_path):
            self.skipTest(f"Missing test fixture: {img_path}")
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

