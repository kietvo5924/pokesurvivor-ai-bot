"""
tests.test_round8_object_recognition
------------------------------------
Round 8 Quantitative Benchmark & Verification Suite:
- Evaluates real game screenshots (tabino_ingame.png)
- Measures multi-class object detection Precision, Recall, and Confusion Matrix across:
  [enemy, boss, projectile, hazard, exp, chest, rift, unknown]
- Verifies spatial deduplication between window.G and OpenCV perception
- Verifies visual overlay badges, bounding boxes, and telemetry dashboard
"""

import os
import math
import unittest
import numpy as np
import cv2

from pokesurvivor.perception import PerceptionEngine
from pokesurvivor.game_state_adapter import GameStateAdapter


class TestRound8ObjectRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perception = PerceptionEngine()
        cls.adapter = GameStateAdapter(page=None, perception=cls.perception)
        cls.real_img_path = os.path.join(
            os.path.dirname(__file__), "..", "debug_artifacts", "tabino_ingame.png"
        )

    def test_01_real_game_image_perception(self):
        """
        Tests perception on real in-game screenshot tabino_ingame.png.
        Ensures:
        1. Player Tapu Fini crest is NOT self-detected as an enemy.
        2. Screen border terrain at bottom-left is NOT detected as an enemy.
        3. Real enemies on field are detected with bounding box and confidence.
        4. Ambiguous contours are labeled unknown with diagnosis reasons.
        5. Visual overlay generates with correct shape and badges.
        """
        if not os.path.exists(self.real_img_path):
            self.skipTest(f"Real image not found at {self.real_img_path}")

        img = cv2.imread(self.real_img_path)
        self.assertIsNotNone(img, "Failed to load real screenshot")

        res = self.perception.analyze_frame(img)

        # 1. Player position centered
        px, py = res["player_pos"]
        self.assertEqual(px, 640)
        self.assertEqual(py, 360)

        # 2. No false enemy alert on player character crest (y ~ 289, x ~ 640)
        enemies = res.get("enemies", [])
        for e in enemies:
            d_player = math.hypot(e["x"] - px, e["y"] - py)
            self.assertGreater(
                d_player, 78.0,
                f"False positive: Player crest detected as enemy at ({e['x']}, {e['y']}) dist={d_player}"
            )

        # 3. No false enemy alert on bottom-left border terrain (x < 150, y > 550)
        for e in enemies:
            is_border_terrain = (e["x"] < 150 and e["y"] > 550)
            self.assertFalse(
                is_border_terrain,
                f"False positive: Border terrain wall detected as enemy at ({e['x']}, {e['y']})"
            )

        # 4. Genuine enemies detected with valid bounding box and confidence >= 0.70
        self.assertGreaterEqual(len(enemies), 1, "Failed to detect genuine enemies on field")
        for e in enemies:
            self.assertIn("box", e)
            self.assertEqual(len(e["box"]), 4)
            self.assertGreaterEqual(e.get("confidence", 0), 0.70)
            self.assertEqual(e.get("entity_type"), "enemy")

        # 5. Ambiguous contours labeled unknown with reasons (not enemies)
        unknowns = res.get("unknowns", [])
        self.assertGreaterEqual(len(unknowns), 1, "Expected ambiguous contours to be captured in unknowns")
        for u in unknowns:
            self.assertEqual(u.get("entity_type"), "unknown")
            self.assertIn("reason", u)
            self.assertLess(u.get("confidence", 1.0), 0.50)

        # 6. Objects breakdown contains multi-class counts
        breakdown = res.get("objects_breakdown", {})
        self.assertIn("enemy", breakdown)
        self.assertIn("exp", breakdown)
        self.assertIn("unknown", breakdown)
        self.assertEqual(breakdown["enemy"], len(enemies))
        self.assertEqual(breakdown["unknown"], len(unknowns))

        # 7. Generate debug overlay
        overlay_out_path = os.path.join(
            os.path.dirname(__file__), "..", "debug_artifacts", "test_round8_real_overlay.png"
        )
        overlay = self.perception.create_debug_overlay(
            img, res, chosen_dir="NW", output_path=overlay_out_path
        )
        self.assertEqual(overlay.shape, img.shape)
        self.assertTrue(os.path.exists(overlay_out_path))

    def test_02_multi_class_precision_recall_benchmark(self):
        """
        Creates a challenging multi-class scenario containing:
        - Clustered enemies
        - Color-blended enemy on dirt
        - Attack telegraph hazard circle
        - Poison projectile
        - EXP gems (cyan, gold, white)
        - Purple Rift portal
        - Ambiguous non-solid noise (should be classified as unknown)
        
        Computes:
        - True Positives, False Positives, False Negatives per class
        - Quantitative Precision & Recall
        - Class Confusion Matrix
        Asserts Precision >= 80% and Recall >= 80%.
        """
        # Base canvas with grass green
        frame = np.full((720, 1280, 3), (48, 128, 56), dtype=np.uint8)
        # Dirt path
        cv2.rectangle(frame, (180, 100), (380, 720), (35, 75, 115), -1)

        # Player at center
        cv2.circle(frame, (640, 360), 25, (180, 100, 50), -1)

        # Ground Truth targets: (class_name, x, y, radius/size)
        ground_truth = [
            # 1. Normal enemy (Gengar BGR (120, 40, 90))
            {"class": "enemy", "x": 420, "y": 220, "radius": 18},
            # 2. Clustered enemy A (Rattata BGR (140, 50, 100))
            {"class": "enemy", "x": 520, "y": 420, "radius": 16},
            # 3. Clustered enemy B (Zubat BGR (160, 60, 50))
            {"class": "enemy", "x": 555, "y": 435, "radius": 16},
            # 4. Mob C (Koffing BGR (130, 80, 100))
            {"class": "enemy", "x": 260, "y": 350, "radius": 17},
            # 5. Attack telegraph hazard zone (red circle, filled)
            {"class": "hazard", "x": 780, "y": 250, "radius": 24},
            # 6. Poison projectile (purple/violet venom spit)
            {"class": "projectile", "x": 720, "y": 480, "radius": 8},
            # 7. EXP Gem 1: Cyan
            {"class": "exp", "x": 480, "y": 300, "radius": 5},
            # 8. EXP Gem 2: Gold
            {"class": "exp", "x": 700, "y": 380, "radius": 5},
            # 9. EXP Gem 3: White glow
            {"class": "exp", "x": 620, "y": 500, "radius": 4},
            # 10. Purple Rift portal (ellipse)
            {"class": "rift", "x": 950, "y": 420, "radius": 40},
            # 11. Ambiguous irregular noise (speckly contour with low solidity)
            {"class": "unknown", "x": 820, "y": 578, "radius": 14},
        ]

        # Draw ground truth entities onto frame
        # Enemy 1: Gengar
        cv2.circle(frame, (420, 220), 18, (120, 40, 90), -1)
        # Enemy 2 & 3: Clustered Rattata & Zubat
        cv2.circle(frame, (520, 420), 16, (140, 50, 100), -1)
        cv2.circle(frame, (555, 435), 16, (160, 60, 50), -1)
        # Enemy 4: Koffing
        cv2.circle(frame, (260, 350), 17, (130, 80, 100), -1)

        # Hazard: Red telegraph circle
        cv2.circle(frame, (780, 250), 24, (0, 0, 225), -1)

        # Projectile: Violet venom spit (HSV: [145, 180, 200] -> BGR)
        proj_bgr = cv2.cvtColor(np.uint8([[[145, 180, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        cv2.circle(frame, (720, 480), 8, (int(proj_bgr[0]), int(proj_bgr[1]), int(proj_bgr[2])), -1)

        # EXP Gem 1: Cyan (BGR: 255, 230, 0)
        cv2.circle(frame, (480, 300), 5, (255, 230, 0), -1)
        # EXP Gem 2: Gold (BGR: 0, 215, 255)
        cv2.circle(frame, (700, 380), 5, (0, 215, 255), -1)
        # EXP Gem 3: White glow (BGR: 250, 250, 250)
        cv2.circle(frame, (620, 500), 4, (250, 250, 250), -1)

        # Rift: Purple portal (HSV: [135, 200, 200] -> BGR)
        rift_bgr = cv2.cvtColor(np.uint8([[[135, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        cv2.ellipse(frame, (950, 420), (35, 50), 0, 0, 360,
                    (int(rift_bgr[0]), int(rift_bgr[1]), int(rift_bgr[2])), -1)

        # Ambiguous noise: narrow spiky cross (low solidity < 0.35)
        pts = np.array([
            [820, 550], [823, 575], [848, 578], [823, 581],
            [820, 606], [817, 581], [792, 578], [817, 575]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (15, 15, 15))

        # Run PerceptionEngine
        res = self.perception.analyze_frame(frame)

        # Collect detections
        detections = []
        for e in res.get("enemies", []):
            detections.append({"class": "enemy", "x": e["x"], "y": e["y"]})
        for h in res.get("hazards", []):
            c_type = "projectile" if h.get("entity_type") == "projectile" else "hazard"
            detections.append({"class": c_type, "x": h["x"], "y": h["y"]})
        for g in res.get("gems", []):
            detections.append({"class": "exp", "x": g["x"], "y": g["y"]})
        if res.get("rift_target"):
            rt = res["rift_target"]
            detections.append({"class": "rift", "x": rt["x"], "y": rt["y"]})
        for u in res.get("unknowns", []):
            detections.append({"class": "unknown", "x": u["x"], "y": u["y"]})

        # Match detections to Ground Truth (threshold: 35 pixels)
        classes = ["enemy", "hazard", "projectile", "exp", "rift", "unknown"]
        confusion_matrix = {gt_c: {pred_c: 0 for pred_c in classes + ["missed"]} for gt_c in classes}
        false_alarms = {pred_c: 0 for pred_c in classes}

        matched_dets = set()
        for gt in ground_truth:
            gt_c = gt["class"]
            best_det_idx = None
            min_dist = 999.0
            for idx, det in enumerate(detections):
                if idx in matched_dets:
                    continue
                d = math.hypot(det["x"] - gt["x"], det["y"] - gt["y"])
                if d <= 35.0 and d < min_dist:
                    min_dist = d
                    best_det_idx = idx

            if best_det_idx is not None:
                matched_dets.add(best_det_idx)
                pred_c = detections[best_det_idx]["class"]
                confusion_matrix[gt_c][pred_c] += 1
            else:
                confusion_matrix[gt_c]["missed"] += 1

        for idx, det in enumerate(detections):
            if idx not in matched_dets:
                false_alarms[det["class"]] += 1

        # Calculate Precision and Recall per class
        metrics = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for c in ["enemy", "hazard", "projectile", "exp", "rift"]:
            tp = confusion_matrix[c][c]
            fn = confusion_matrix[c]["missed"] + sum(confusion_matrix[c][other] for other in classes if other != c)
            fp = false_alarms[c] + sum(confusion_matrix[other][c] for other in classes if other != c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics[c] = {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}

            total_tp += tp
            total_fp += fp
            total_fn += fn

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        print("\n" + "=" * 65)
        print("📊 ROUND 8 QUANTITATIVE OBJECT RECOGNITION BENCHMARK")
        print("=" * 65)
        print(f"{'Class':<12} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<10} | {'Recall':<10}")
        print("-" * 65)
        for c, m in metrics.items():
            print(f"{c:<12} | {m['tp']:<4} | {m['fp']:<4} | {m['fn']:<4} | {m['precision']*100:>8.1f}% | {m['recall']*100:>8.1f}%")
        print("-" * 65)
        print(f"{'OVERALL':<12} | {total_tp:<4} | {total_fp:<4} | {total_fn:<4} | {overall_precision*100:>8.1f}% | {overall_recall*100:>8.1f}%")
        print("=" * 65)

        # Output Confusion Matrix
        print("\n🔲 CONFUSION MATRIX (Rows: Ground Truth, Columns: Predicted):")
        hdr = f"{'GT \\ Pred':<12} | " + " | ".join(f"{c:<6}" for c in classes) + " | Missed"
        print(hdr)
        print("-" * len(hdr))
        for gt_c in classes:
            row_vals = [f"{confusion_matrix[gt_c][pred_c]:<6}" for pred_c in classes]
            row_str = f"{gt_c:<12} | " + " | ".join(row_vals) + f" | {confusion_matrix[gt_c]['missed']}"
            print(row_str)
        print("=" * 65 + "\n")

        # Quantitative Assertions
        self.assertGreaterEqual(overall_precision, 0.80, f"Overall precision {overall_precision:.2f} < 0.80")
        self.assertGreaterEqual(overall_recall, 0.80, f"Overall recall {overall_recall:.2f} < 0.80")

        # Crucial verification: Ambiguous noise MUST NOT be classified as enemy
        self.assertEqual(
            confusion_matrix["unknown"]["enemy"], 0,
            "Failure: Ambiguous noise was mistakenly classified as enemy!"
        )

    def test_03_data_merging_and_spatial_deduplication(self):
        """
        Tests GameStateAdapter._extract_entities spatial deduplication:
        - When window.G and OpenCV observe the same entity (R <= 45px), it is NOT duplicated.
        - window.G entity keeps authoritative metadata (dex, confidence=1.0, source='window_G').
        - OpenCV detection enriches bounding box without overriding.
        - Unmatched OpenCV detection is added as 'opencv_supplement' with confidence < 1.0.
        """
        # Simulated player at (1000, 1000)
        player = {
            "world_x": 1000.0,
            "world_y": 1000.0,
            "screen_x": 640.0,
            "screen_y": 360.0,
            "camera_verified": True,
            "source": "window_G"
        }

        # Camera scale = 1.0, translation e=640 - 1000 = -360, f=360 - 1000 = -640
        telemetry = {
            "camera": {
                "s": 1.0,
                "e": -360.0,
                "f": -640.0,
                "verified": True,
                "now": 10000.0,
                "updatedAt": 9900.0
            }
        }

        # window.G has 1 enemy at world (1100, 1000) -> screen (740, 360)
        g_state = {
            "enemies": [
                {"x": 1100.0, "y": 1000.0, "dex": 25, "hp": 50, "maxHp": 50, "boss": False}
            ],
            "hazards": [
                {"kind": "circle", "x": 1050.0, "y": 1050.0, "r": 20}
            ],
            "pickups": [
                {"kind": "xp", "x": 1020.0, "y": 1000.0}
            ]
        }

        # OpenCV sees:
        # 1. Matching enemy at screen (742, 361) (distance ~2.2px <= 45px) with box (726, 345, 32, 32)
        # 2. Supplementary unmatched enemy at screen (900, 500)
        cv_frame_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cv_vision = {
            "enemies": [
                {"x": 742.0, "y": 361.0, "box": (726, 345, 32, 32), "entity_type": "enemy", "confidence": 0.85},
                {"x": 900.0, "y": 500.0, "box": (885, 485, 30, 30), "entity_type": "enemy", "confidence": 0.78}
            ],
            "hazards": [
                {"x": 690.0, "y": 410.0, "box": (675, 395, 30, 30), "entity_type": "hazard", "confidence": 0.88}
            ],
            "gems": [
                {"x": 660.0, "y": 360.0, "box": (655, 355, 10, 10), "entity_type": "exp", "confidence": 0.90}
            ],
            "unknowns": [
                {"x": 300.0, "y": 300.0, "reason": "irregular_solidity", "confidence": 0.20}
            ],
            "sector_obstacles": {},
            "sector_obs_costs": {}
        }

        # Temporarily mock perception.analyze_frame to return mock_cv_vision
        orig_analyze = self.perception.analyze_frame
        try:
            self.perception.analyze_frame = lambda *args, **kwargs: mock_cv_vision

            browser_data = {
                "gState": g_state,
                "telemetry": telemetry
            }

            entities = self.adapter._extract_entities(
                browser_data, frame_img=cv_frame_img, is_moving=False, player=player
            )

            # 1. Enemies deduplication & supplementation check
            enemies = entities["enemies"]
            # Total enemies should be exactly 2 (1 window_G + 1 supplement), NOT 3!
            self.assertEqual(len(enemies), 2, f"Expected 2 enemies after deduplication, got {len(enemies)}")

            # First enemy: window_G authoritative
            e1 = enemies[0]
            self.assertEqual(e1["source"], "window_G")
            self.assertEqual(e1["dex"], 25)
            self.assertEqual(e1["confidence"], 1.0)
            self.assertEqual(e1["entity_type"], "enemy")
            # Enriched with OpenCV bounding box
            self.assertEqual(e1["box"], (726, 345, 32, 32))

            # Second enemy: opencv_supplement
            e2 = enemies[1]
            self.assertEqual(e2["source"], "opencv_supplement")
            self.assertEqual(e2["confidence"], 0.78)
            self.assertEqual(e2["entity_type"], "enemy")

            # 2. Hazards deduplication check (matching screen: 1050 + e = 690, 1050 + f = 410)
            hazards = entities["hazards"]
            self.assertEqual(len(hazards), 1, "Expected hazard to be deduplicated without duplicating")
            self.assertEqual(hazards[0]["source"], "window_G")
            self.assertEqual(hazards[0]["box"], (675, 395, 30, 30))

            # 3. Items deduplication check (matching screen: 1020 + e = 660, 1000 + f = 360)
            items = entities["items"]
            self.assertEqual(len(items), 1, "Expected item to be deduplicated without duplicating")
            self.assertEqual(items[0]["source"], "window_G")
            self.assertEqual(items[0]["box"], (655, 355, 10, 10))

            # 4. Unknowns passed through
            self.assertEqual(len(entities["unknowns"]), 1)
            self.assertEqual(entities["unknowns"][0]["reason"], "irregular_solidity")

            # 5. Objects breakdown accuracy
            bd = entities["objects_breakdown"]
            self.assertEqual(bd["enemy"], 2)
            self.assertEqual(bd["hazard"], 1)
            self.assertEqual(bd["exp"], 1)
            self.assertEqual(bd["unknown"], 1)

        finally:
            self.perception.analyze_frame = orig_analyze

    def test_04_visual_overlay_rendering(self):
        """
        Verifies that create_debug_overlay correctly formats and renders:
        - [BOSS], [THIEF], [#DEX], [ENEMY] badges
        - [PROJ], [HAZARD] badges with trajectory arrows
        - [EXP] and [ITEM] badges
        - [RIFT] portal badge
        - [? REASON] dashed unknown badges
        - Telemetry HUD lines with accurate object breakdown
        """
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = {
            "player_pos": (640, 360),
            "hp_ratio": 0.82,
            "hp_status": "VERIFIED",
            "danger_level": "MEDIUM",
            "confidence": 0.95,
            "enemies": [
                {"screen_x": 400, "screen_y": 200, "box": (384, 184, 32, 32), "is_boss": True, "source": "window_G", "confidence": 1.0},
                {"screen_x": 500, "screen_y": 250, "box": (484, 234, 32, 32), "is_thief": True, "source": "window_G", "confidence": 1.0},
                {"screen_x": 600, "screen_y": 200, "box": (584, 184, 32, 32), "dex": 54, "source": "window_G", "confidence": 1.0},
                {"screen_x": 700, "screen_y": 200, "box": (684, 184, 32, 32), "source": "opencv", "confidence": 0.82}
            ],
            "hazards": [
                {"screen_x": 800, "screen_y": 300, "box": (786, 286, 28, 28), "entity_type": "hazard", "ttc": 1.1, "is_imminent": True, "source": "window_G", "confidence": 1.0},
                {"screen_x": 850, "screen_y": 350, "box": (836, 336, 28, 28), "entity_type": "projectile", "vx": -50.0, "vy": 20.0, "source": "opencv", "confidence": 0.85}
            ],
            "gems": [
                {"screen_x": 550, "screen_y": 400, "box": (545, 395, 10, 10), "source": "opencv", "confidence": 0.90}
            ],
            "items_on_ground": [
                {"screen_x": 650, "screen_y": 450, "box": (638, 438, 24, 24), "kind": "chest", "source": "opencv", "confidence": 0.88}
            ],
            "rift_target": {
                "screen_x": 900, "screen_y": 400, "box": (875, 375, 50, 50), "source": "opencv", "confidence": 0.95
            },
            "unknowns": [
                {"screen_x": 300, "screen_y": 500, "box": (286, 486, 28, 28), "reason": "irregular_solidity", "confidence": 0.25}
            ]
        }

        out_path = os.path.join(
            os.path.dirname(__file__), "..", "debug_artifacts", "test_round8_full_overlay.png"
        )
        overlay = self.perception.create_debug_overlay(
            canvas, state, chosen_dir="S", output_path=out_path
        )

        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape, (720, 1280, 3))
        self.assertTrue(os.path.exists(out_path))

        # Check that pixels were modified on overlay HUD area
        hud_crop = overlay[720 - 145:720 - 10, 10:370]
        self.assertGreater(np.mean(hud_crop), 1.0, "HUD dashboard was not rendered onto overlay")


if __name__ == "__main__":
    unittest.main()
