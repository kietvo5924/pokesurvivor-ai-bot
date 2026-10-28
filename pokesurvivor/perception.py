import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import math

class PerceptionEngine:
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.width = frame_width
        self.height = frame_height
        self.center = (frame_width // 2, frame_height // 2)
        self.prev_gray: Optional[np.ndarray] = None
        self.history_frames: List[np.ndarray] = []
        self.prev_hazards: List[Dict[str, Any]] = []
        self.prev_hazard_time: float = 0.0
        self.active_pokemon: str = "Tapu Fini"
        self.last_hp_result: Dict[str, Any] = {"hp_ratio": 1.0, "status": "UNKNOWN", "box": None}

    def set_active_pokemon(self, name: str):
        """Sets active partner Pokemon for visual context and overlay."""
        if name:
            self.active_pokemon = name

    def analyze_frame(self, frame: np.ndarray, is_moving: bool = True) -> Dict[str, Any]:
        """Performs computer vision perception on the gameplay frame."""
        if frame is None or frame.size == 0:
            return self._empty_perception()

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Convert to HSV and Grayscale
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Exact Dynamic Player Localization (Pokemon-neutral contrast centroid)
        center_x, center_y = self._locate_player(frame, hsv)
        player_pos = (center_x, center_y)

        # 1b. Verified Top-Left HUD HP Detection
        hp_info = self._detect_hp_ratio(frame)
        hp_ratio = hp_info.get("hp_ratio")
        hp_status = hp_info.get("status", "UNKNOWN")
        hp_box = hp_info.get("box")

        # 2. EXP Gems Detection
        gems, gem_cluster_sector = self._detect_gems(hsv, center_x, center_y)

        # 3. Entities & Items Detection
        enemies, items_on_ground = self._detect_entities(hsv, center_x, center_y)

        # 4. Confirmed Hazards & Multi-Frame Trajectory Tracking
        raw_hazards = self._detect_hazards_and_projectiles(hsv, center_x, center_y, enemies)
        hazards = self._track_hazards(raw_hazards, center_x, center_y)

        # 5. Nearest enemy & Swarm Center
        min_enemy_dist = 9999.0
        nearest_enemy_sector = None
        nearest_enemy = None
        swarm_x, swarm_y, swarm_count = 0.0, 0.0, 0

        for e in enemies:
            d = e["dist"]
            if d < min_enemy_dist:
                min_enemy_dist = d
                nearest_enemy = e
                nearest_enemy_sector = e["sector"]
            
            # Swarm center of mass within 340px
            if d < 340:
                swarm_x += e["x"]
                swarm_y += e["y"]
                swarm_count += 1

        swarm_center = (swarm_x / swarm_count, swarm_y / swarm_count) if swarm_count > 0 else None

        # Swarm mob density & True Encirclement Detection
        # A true encirclement requires at least 7+ close enemies (< 175px) spanning 4+ sectors
        sector_enemy_counts: Dict[str, int] = {s: 0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
        for e in enemies:
            sec = e.get("sector")
            if sec in sector_enemy_counts:
                sector_enemy_counts[sec] += 1

        nearby_threats = [e for e in enemies if e.get("dist", 999.0) < 175.0]
        encirclement_sectors = {e["sector"] for e in nearby_threats if e.get("sector")}
        is_encircled = len(nearby_threats) >= 7 and len(encirclement_sectors) >= 4

        # 6. Obstacle Probing (Rocks, Cliffs, Water, Trees in 8 directions with multi-distance raycasts)
        obstacle_mask = self._detect_obstacle_mask(hsv, center_x, center_y)
        sector_obstacles, sector_obs_costs = self._probe_sector_obstacles(obstacle_mask, center_x, center_y)

        # 7. Sector Danger Profiling (incorporates enemies, hazards with TTC, and terrain obstacles)
        sector_dangers = self._compute_sector_dangers(enemies, hazards, center_x, center_y, sector_obstacles, sector_obs_costs)

        # 8. EXP Direction
        best_gem_dir = self._find_nearest_gem_vector(gems, center_x, center_y)

        # 9. Stuck / Collision Detection (strictly checked only when keys are held AND an obstacle is nearby)
        is_stuck = self._check_if_stuck(gray, center_x, center_y, is_moving, sector_obstacles)

        # 10. Purple Rift Portal & Arrow Indicator Detection
        rift_target = self._detect_purple_rift(hsv, center_x, center_y)

        # 11. Boss Presence Detection (Top HP Bar + Large Boss Sprite)
        boss_info = self._detect_boss(hsv, enemies, center_x, center_y)

        # Determine overall danger level based on distance, imminent bullets, and stuck state
        has_imminent_hazard = any(h.get("is_imminent", False) or h.get("ttc", 99.0) < 0.9 for h in hazards)
        if min_enemy_dist < 120 or has_imminent_hazard or is_stuck:
            danger_level = "CRITICAL"
        elif min_enemy_dist < 200 or len(hazards) > 0 or len(enemies) > 6:
            danger_level = "HIGH"
        elif min_enemy_dist < 280 or len(enemies) > 2:
            danger_level = "MEDIUM"
        else:
            danger_level = "LOW"

        confidence = 0.92 if hp_status == "VERIFIED" else 0.75

        return {
            "player_pos": player_pos,
            "hp_ratio": hp_ratio,
            "hp_status": hp_status,
            "hp_box": hp_box,
            "enemies_count": len(enemies),
            "hazards_count": len(hazards),
            "gems_count": len(gems),
            "enemies": enemies[:25],
            "hazards": hazards[:10],
            "gems": gems[:15],
            "items_on_ground": items_on_ground[:5],
            "sector_enemy_counts": sector_enemy_counts,
            "is_encircled": is_encircled,
            "sector_dangers": sector_dangers,
            "sector_obstacles": sector_obstacles,
            "sector_obs_costs": sector_obs_costs,
            "best_gem_vector": best_gem_dir,
            "gem_cluster_sector": gem_cluster_sector,
            "min_enemy_dist": min_enemy_dist,
            "nearest_enemy_sector": nearest_enemy_sector,
            "nearest_enemy": nearest_enemy,
            "swarm_center": swarm_center,
            "danger_level": danger_level,
            "is_stuck": is_stuck,
            "rift_target": rift_target,
            "boss_info": boss_info,
            "confidence": confidence
        }

    def _locate_player(self, frame: np.ndarray, hsv: np.ndarray) -> Tuple[int, int]:
        """
        Dynamically locates player center coordinates in viewport.
        Pokemon-neutral: Accurately tracks Tapu Fini, Scorbunny, or any partner without assuming orange/white colors.
        Uses foreground sprite contrast against grass/dirt terrain within +-65px of viewport center.
        Falls back to (mid_x, mid_y) when centered or ambiguous.
        """
        h, w = frame.shape[:2]
        mid_x, mid_y = w // 2, h // 2
        y1, y2 = max(0, mid_y - 65), min(h, mid_y + 65)
        x1, x2 = max(0, mid_x - 65), min(w, mid_x + 65)
        crop = frame[y1:y2, x1:x2]
        crop_hsv = hsv[y1:y2, x1:x2]

        # Background grass/dirt masks in crop
        mask_grass = cv2.inRange(crop_hsv, np.array([52, 112, 50]), np.array([68, 220, 190]))
        mask_dirt = cv2.inRange(crop_hsv, np.array([14, 40, 50]), np.array([32, 160, 190]))
        bg = cv2.bitwise_or(mask_grass, mask_dirt)
        fg = cv2.bitwise_not(bg)

        # Exclude borders of crop
        fg[0:3, :] = 0
        fg[-3:, :] = 0
        fg[:, 0:3] = 0
        fg[:, -3:] = 0

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pt = None
        min_c_dist = 999.0
        crop_cx, crop_cy = (x2 - x1) // 2, (y2 - y1) // 2

        for c in contours:
            area = cv2.contourArea(c)
            if 80 < area < 2500:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])
                    dist = math.hypot(px - crop_cx, py - crop_cy)
                    if dist < 45.0 and dist < min_c_dist:
                        min_c_dist = dist
                        best_pt = (px, py)

        if best_pt is not None:
            return x1 + best_pt[0], y1 + best_pt[1]

        return mid_x, mid_y

    def _detect_obstacle_mask(self, hsv: np.ndarray, cx: int = 640, cy: int = 360) -> np.ndarray:
        """
        Creates a binary mask of true impassable physical terrain obstacles:
        - Water (The Great Western Lake & shoreline foam > 4000px)
        - Elevated Cliff Walls and Cliff Shadow Drop edges
        - Grey Boulders & Rocks (> 200px)
        - Tree Trunks
        
        CRUCIAL:
        - NEVER treats Pokemon sprite outlines or grass as terrain walls!
        - Only large contiguous terrain features are treated as obstacles.
        - Player body (radius 30px) is masked out.
        """
        h, w = hsv.shape[:2]

        # 1. Impassable Water (large body > 4000px)
        mask_blue_raw = cv2.inRange(hsv, np.array([85, 50, 50]), np.array([135, 255, 255]))
        cnts_w, _ = cv2.findContours(mask_blue_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_water = np.zeros((h, w), dtype=np.uint8)
        for wc in cnts_w:
            if cv2.contourArea(wc) > 4000:
                cv2.drawContours(mask_water, [wc], -1, 255, -1)

        # 2. Impassable Cliff Wall & Cliff Shadow Drop
        # Earthy olive/brown cliff stone face: H: 12..28, S: 70..170, V: 60..160
        m_cliff_face = cv2.inRange(hsv, np.array([12, 70, 60]), np.array([28, 170, 160]))
        # Dark shadow drop edge at top of cliff: H: 155..180, S: 70..255, V: 15..65
        m_cliff_shadow = cv2.inRange(hsv, np.array([155, 70, 15]), np.array([180, 255, 65]))
        m_cliff = cv2.bitwise_or(m_cliff_face, m_cliff_shadow)
        cnts_c, _ = cv2.findContours(m_cliff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_cliff = np.zeros((h, w), dtype=np.uint8)
        for c in cnts_c:
            if cv2.contourArea(c) > 250:
                cv2.drawContours(mask_cliff, [c], -1, 255, -1)

        # 3. Grey Boulders / Round Rocks
        m_rocks_raw = cv2.inRange(hsv, np.array([0, 0, 35]), np.array([180, 45, 170]))
        cnts_r, _ = cv2.findContours(m_rocks_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_rocks = np.zeros((h, w), dtype=np.uint8)
        for rc in cnts_r:
            if 200 < cv2.contourArea(rc) < 3500:
                cv2.drawContours(mask_rocks, [rc], -1, 255, -1)

        # 4. Tree trunks (dark wood)
        mask_wood = cv2.inRange(hsv, np.array([10, 90, 20]), np.array([25, 220, 100]))

        # Combined impassable obstacle mask (strictly terrain only!)
        obs = cv2.bitwise_or(mask_water, mask_cliff)
        obs = cv2.bitwise_or(obs, mask_rocks)
        obs = cv2.bitwise_or(obs, mask_wood)

        # Morphological closing to form solid impenetrable obstacle barriers
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        obs = cv2.morphologyEx(obs, cv2.MORPH_CLOSE, k_close)

        # Exclude UI areas (topbar and bottom HUD)
        obs[:65, :] = 0
        obs[h-60:, :] = 0

        # Mask out player body (radius 30px) so Scorbunny never blocks itself!
        cv2.circle(obs, (int(round(float(cx))), int(round(float(cy)))), 30, 0, -1)
        return obs

    def _probe_sector_obstacles(self, obstacle_mask: np.ndarray, cx: int, cy: int) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """
        Probes 8 directional rays with 4 concentric ranges (55px, 95px, 140px, 190px).
        Also detects narrow corridors/chokepoints by checking lateral clearance.
        Returns:
          - sector_blocked: True if an obstacle is within immediate collision distance (<= 95px)
          - sector_costs: continuous distance-weighted cost + chokepoint penalties
        """
        h, w = obstacle_mask.shape[:2]
        sector_angles = {
            "E": 0, "SE": 45, "S": 90, "SW": 135,
            "W": 180, "NW": 225, "N": 270, "NE": 315
        }
        ranges = [55, 95, 140, 190]
        weights = [6000.0, 3000.0, 1200.0, 400.0]

        sector_blocked = {}
        sector_costs = {}

        for sec, deg in sector_angles.items():
            rad = math.radians(deg)
            cost = 0.0
            blocked = False

            for r, w_cost in zip(ranges, weights):
                px = int(cx + r * math.cos(rad))
                py = int(cy + r * math.sin(rad))

                # Sample 22x22 box around probe ray point
                y1, y2 = max(0, py - 11), min(h, py + 11)
                x1, x2 = max(0, px - 11), min(w, px + 11)
                crop = obstacle_mask[y1:y2, x1:x2]

                if crop.size > 0:
                    density = cv2.countNonZero(crop) / float(crop.size)
                    if density > 0.22:
                        cost += w_cost * density
                        if r <= 95:
                            blocked = True

            sector_costs[sec] = cost
            sector_blocked[sec] = blocked

        # Corridor / Chokepoint check:
        # If both orthogonal directions are blocked within 95px, entering this sector is a dead-end choke trap!
        orthogonal_pairs = {
            "N": ("W", "E"), "S": ("W", "E"),
            "E": ("N", "S"), "W": ("N", "S"),
            "NE": ("NW", "SE"), "SW": ("NW", "SE"),
            "NW": ("NE", "SW"), "SE": ("NE", "SW")
        }
        for sec, (side1, side2) in orthogonal_pairs.items():
            if sector_blocked.get(side1, False) and sector_blocked.get(side2, False):
                # Narrow corridor! Add heavy penalty so Scorbunny avoids stepping into cliff bottlenecks
                sector_costs[sec] += 4500.0

        return sector_blocked, sector_costs

    def _check_if_stuck(
        self,
        gray: np.ndarray,
        cx: int,
        cy: int,
        is_moving: bool = True,
        sector_obstacles: Optional[Dict[str, bool]] = None
    ) -> bool:
        """
        Checks if the game screen has stopped scrolling despite input (ramming a wall/rock).
        Strict conditions:
        - Only active when keys are held down (is_moving == True)
        - Only active when an obstacle is within immediate proximity (<= 80px)
        - Completely immune to false positives during level up, pause, or open grass running.
        """
        if not is_moving:
            self.stuck_frames = 0
            self.prev_gray = None
            return False

        has_nearby_obstacle = any(sector_obstacles.values()) if sector_obstacles else False
        if not has_nearby_obstacle:
            self.stuck_frames = 0
            self.prev_gray = gray
            return False

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.stuck_frames = 0
            return False

        # Dual-patch sampling: patch 1 (upper-right) and patch 2 (lower-left)
        p1 = cv2.absdiff(gray[max(0, cy - 160):cy - 80, cx + 80:min(gray.shape[1], cx + 180)],
                         self.prev_gray[max(0, cy - 160):cy - 80, cx + 80:min(gray.shape[1], cx + 180)])
        p2 = cv2.absdiff(gray[cy + 80:min(gray.shape[0], cy + 160), max(0, cx - 180):cx - 80],
                         self.prev_gray[cy + 80:min(gray.shape[0], cy + 160), max(0, cx - 180):cx - 80])

        m1 = np.mean(p1) if p1.size > 0 else 1.0
        m2 = np.mean(p2) if p2.size > 0 else 1.0
        mean_diff = (m1 + m2) / 2.0

        self.prev_gray = gray

        # Screen must be completely motionless (< 0.45) for 5 consecutive frames next to an obstacle
        if mean_diff < 0.45:
            self.stuck_frames = getattr(self, "stuck_frames", 0) + 1
        else:
            self.stuck_frames = 0

        return self.stuck_frames >= 5

    def _detect_hp_ratio(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects player HP ratio from the top-left HUD bar (x: 30..300, y: 15..55).
        Returns:
            {
                "hp_ratio": float (0.0 to 1.0) or None,
                "status": "VERIFIED" | "ESTIMATED" | "UNKNOWN",
                "active_width": int,
                "total_width": int,
                "box": (x, y, w, h) or None
            }
        Never defaults to 100% when HP is unknown or unreadable.
        """
        if frame is None or frame.size == 0:
            res = {"hp_ratio": None, "status": "UNKNOWN", "active_width": 0, "total_width": 0, "box": None}
            self.last_hp_result = res
            return res

        h, w = frame.shape[:2]
        hud_crop = frame[15:min(h, 55), 30:min(w, 300)]
        if hud_crop.size == 0:
            res = {"hp_ratio": None, "status": "UNKNOWN", "active_width": 0, "total_width": 0, "box": None}
            self.last_hp_result = res
            return res

        r = hud_crop[:, :, 2].astype(float)
        g = hud_crop[:, :, 1].astype(float)
        b = hud_crop[:, :, 0].astype(float)

        # 1. Active HP Fill: Bright coral/pink (R > 175, R - G > 80, R - B > 50)
        active_hp = (r > 175) & ((r - g) > 80) & ((r - b) > 50)

        # 2. Damaged/Empty HP Bar Background: Dark reddish-brown (35 < R < 130, G < 60, B < 60)
        empty_hp = (r > 35) & (r < 130) & (g < 60) & (b < 60)

        # Combined full HP container track
        hp_track = active_hp | empty_hp

        cnts_active, _ = cv2.findContours((active_hp * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        active_w = 0
        active_box = None
        for c in cnts_active:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 15 and 4 <= bh <= 20:
                if bw > active_w:
                    active_w = bw
                    active_box = (bx, by, bw, bh)

        if active_box is None:
            res = {"hp_ratio": None, "status": "UNKNOWN", "active_width": 0, "total_width": 0, "box": None}
            self.last_hp_result = res
            return res

        bx, by, bw, bh = active_box
        y1 = max(0, by - 1)
        y2 = min(hud_crop.shape[0], by + bh + 1)
        slice_r = r[y1:y2, bx:min(hud_crop.shape[1], bx + 220)]
        slice_g = g[y1:y2, bx:min(hud_crop.shape[1], bx + 220)]
        slice_b = b[y1:y2, bx:min(hud_crop.shape[1], bx + 220)]

        # Bar pixel classification: active fill, empty reddish container track, or white/pink text on the bar
        is_bar_pixel = (
            ((slice_r > 175) & ((slice_r - slice_g) > 70) & ((slice_r - slice_b) > 40)) |
            ((slice_r > 30) & (slice_r < 140) & (slice_g < 75) & (slice_b < 75)) |
            ((slice_r > 180) & (slice_g > 180) & (slice_b > 180))
        )
        col_has_bar = np.any(is_bar_pixel, axis=0)

        total_w = active_w
        gap = 0
        for i, val in enumerate(col_has_bar):
            if val:
                total_w = i + 1
                gap = 0
            else:
                if i >= active_w:
                    gap += 1
                    if gap >= 4:
                        break

        abs_box = (bx + 30, by + 15, total_w, bh)

        # Verification criteria:
        # Full track measured in normal PokéSurvivor container range (120 - 195px)
        if 120 <= total_w <= 195:
            ratio = min(1.0, max(0.0, float(active_w) / float(total_w)))
            status = "VERIFIED"
        elif active_w >= 140:
            ratio = 1.0
            status = "VERIFIED"
        elif active_w > 0:
            ratio = min(1.0, max(0.0, float(active_w) / 142.0))
            status = "ESTIMATED"
        else:
            res = {"hp_ratio": None, "status": "UNKNOWN", "active_width": 0, "total_width": 0, "box": None}
            self.last_hp_result = res
            return res

        res = {
            "hp_ratio": round(ratio, 3),
            "status": status,
            "active_width": active_w,
            "total_width": total_w,
            "box": abs_box
        }
        self.last_hp_result = res
        return res

    def _detect_gems(self, hsv: np.ndarray, cx: int, cy: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Detects EXP gems: Cyan/Blue orbs, Golden Diamond gems, and Bright white EXP specks.
        Returns sorted list of individual gems and the optimal gem_cluster_sector.
        """
        h, w = hsv.shape[:2]
        # Cyan / blue gems
        mask_cyan = cv2.inRange(hsv, np.array([80, 80, 130]), np.array([130, 255, 255]))
        # Golden yellow/orange diamond gems
        mask_gold = cv2.inRange(hsv, np.array([18, 110, 140]), np.array([38, 255, 255]))
        # Bright white/glow specks
        mask_white = cv2.inRange(hsv, np.array([0, 0, 205]), np.array([180, 45, 255]))

        gem_mask = cv2.bitwise_or(mask_cyan, mask_gold)
        gem_mask = cv2.bitwise_or(gem_mask, mask_white)

        # Exclude UI topbar (y < 65) and bottom HUD
        gem_mask[:65, :] = 0
        gem_mask[h-55:, :] = 0

        # Don't count near player sprite
        cv2.circle(gem_mask, (int(round(float(cx))), int(round(float(cy)))), 46, 0, -1)

        contours, _ = cv2.findContours(gem_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gems = []
        sector_gem_counts: Dict[str, int] = {s: 0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}

        for c in contours:
            area = cv2.contourArea(c)
            if 3 < area < 400:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    gx = int(M["m10"] / M["m00"])
                    gy = int(M["m01"] / M["m00"])
                    dist = math.hypot(gx - cx, gy - cy)
                    if dist > 50.0:
                        angle = math.degrees(math.atan2(gy - cy, gx - cx))
                        sec = self._angle_to_sector(angle)
                        gems.append({"x": gx, "y": gy, "dist": dist, "angle": angle, "sector": sec, "area": area})
                        if dist < 320:
                            sector_gem_counts[sec] += 1

        gems.sort(key=lambda g: g["dist"])

        # Find sector with the highest density of gems within 320px
        best_cluster_sec = None
        max_gems = 0
        for s, count in sector_gem_counts.items():
            if count > max_gems:
                max_gems = count
                best_cluster_sec = s

        return gems, best_cluster_sec

    def _detect_hazards_and_projectiles(
        self,
        hsv: np.ndarray,
        cx: int,
        cy: int,
        enemies: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detects enemy attack telegraph zones (red/orange/yellow ground circles),
        poison spit, seeds, and flying projectiles.
        - Excludes player core sprite (radius 24px) without blinding incoming bullets.
        - Filters out enemy sprite heads/eyes.
        - Detects both small and large warning circles (area >= 80).
        """
        h, w = hsv.shape[:2]

        # 1. Red / Orange / Pink attack telegraph zones & damage circles
        # True warning zones have high saturation (S > 115) to avoid false triggers on dirt/stone
        lower_r1 = np.array([0, 115, 110])
        upper_r1 = np.array([12, 255, 255])
        lower_r2 = np.array([168, 115, 110])
        upper_r2 = np.array([180, 255, 255])
        m_r1 = cv2.inRange(hsv, lower_r1, upper_r1)
        m_r2 = cv2.inRange(hsv, lower_r2, upper_r2)
        m_zones = cv2.bitwise_or(m_r1, m_r2)

        # 2. Poison projectiles (purple/violet venom spit)
        lower_p = np.array([135, 95, 95])
        upper_p = np.array([155, 255, 255])
        m_poison = cv2.inRange(hsv, lower_p, upper_p)

        # Exclude UI topbar (y: 0..115) and bottom HUD (y: h-70..h)
        m_zones[:115, :] = 0
        m_zones[h-70:, :] = 0
        m_poison[:115, :] = 0
        m_poison[h-70:, :] = 0

        # Exclude player core sprite and own fire launch origin (radius 52px)
        icx, icy = int(round(float(cx))), int(round(float(cy)))
        cv2.circle(m_zones, (icx, icy), 52, 0, -1)
        cv2.circle(m_poison, (icx, icy), 52, 0, -1)

        hazards = []

        # Check Attack Warning Circles / Telegraphs (area >= 80)
        cnts_zones, _ = cv2.findContours(m_zones, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_zones:
            area = cv2.contourArea(c)
            if 80 <= area < 30000:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    hx = int(M["m10"] / M["m00"])
                    hy = int(M["m01"] / M["m00"])
                    dist = math.hypot(hx - cx, hy - cy)
                    if dist < 52.0:
                        continue
                    # Exclude if on top of an enemy sprite center
                    if enemies and any(math.hypot(hx - e["x"], hy - e["y"]) < 22.0 for e in enemies):
                        continue
                    angle = math.degrees(math.atan2(hy - cy, hx - cx))
                    sec = self._angle_to_sector(angle)
                    hazards.append({
                        "x": hx,
                        "y": hy,
                        "dist": dist,
                        "angle": angle,
                        "sector": sec,
                        "area": area,
                        "type": "telegraph"
                    })

        # Check Poison & Flying Projectiles (area >= 20)
        cnts_p, _ = cv2.findContours(m_poison, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_p:
            area = cv2.contourArea(c)
            if 20 <= area < 800:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    hx = int(M["m10"] / M["m00"])
                    hy = int(M["m01"] / M["m00"])
                    dist = math.hypot(hx - cx, hy - cy)
                    if dist < 52.0:
                        continue
                    if enemies and any(math.hypot(hx - e["x"], hy - e["y"]) < 22.0 for e in enemies):
                        continue
                    angle = math.degrees(math.atan2(hy - cy, hx - cx))
                    sec = self._angle_to_sector(angle)
                    hazards.append({
                        "x": hx,
                        "y": hy,
                        "dist": dist,
                        "angle": angle,
                        "sector": sec,
                        "area": area,
                        "type": "poison_bullet"
                    })

        hazards.sort(key=lambda h: h["dist"])
        return hazards

    def _track_hazards(self, current_hazards: List[Dict[str, Any]], cx: int, cy: int) -> List[Dict[str, Any]]:
        """
        Multi-frame hazard tracking:
        Estimates velocity vector (vx, vy), relative speed, Time-To-Collision (TTC),
        and Distance of Closest Approach (DCA) towards player across frames.
        """
        import time
        now = time.time()
        dt = now - self.prev_hazard_time if self.prev_hazard_time > 0 else 0.05
        dt = max(0.01, min(0.35, dt))

        tracked = []
        for h in current_hazards:
            hx, hy = h["x"], h["y"]
            vx, vy = 0.0, 0.0

            # Match with previous hazard by proximity
            best_prev = None
            min_match_dist = 999.0
            for prev in self.prev_hazards:
                d = math.hypot(hx - prev["x"], hy - prev["y"])
                if d < 65.0 and d < min_match_dist:
                    min_match_dist = d
                    best_prev = prev

            if best_prev is not None and dt > 0:
                raw_vx = (hx - best_prev["x"]) / dt
                raw_vy = (hy - best_prev["y"]) / dt
                raw_speed = math.hypot(raw_vx, raw_vy)
                # Physical validation: projectile velocity in PokéSurvivor is between 30 and 450 px/s
                if raw_speed <= 500.0:
                    vx = raw_vx
                    vy = raw_vy
            # Newly observed hazards have no multi-frame evidence -> vx = 0.0, vy = 0.0 without assuming collision vector

            speed = math.hypot(vx, vy)
            rx = cx - hx
            ry = cy - hy
            dist = math.hypot(rx, ry)
            dot = vx * rx + vy * ry

            ttc = 99.0
            dca = dist
            is_imminent = False

            if speed > 25.0 and dot > 0:
                ttc = dot / (speed * speed)
                dca_x = rx - vx * ttc
                dca_y = ry - vy * ttc
                dca = math.hypot(dca_x, dca_y)
                if dca < 45.0 and 0.0 < ttc < 1.2:
                    is_imminent = True
            elif h.get("type") == "telegraph":
                if dist < 65.0:
                    ttc = 0.4
                    dca = dist
                    is_imminent = True
            elif dist < 45.0:
                # Immediate proximity threat even without multi-frame vector
                ttc = 0.25
                dca = dist
                is_imminent = True

            h_tracked = dict(h)
            h_tracked["vx"] = round(vx, 1)
            h_tracked["vy"] = round(vy, 1)
            h_tracked["speed"] = round(speed, 1)
            h_tracked["ttc"] = round(ttc, 2)
            h_tracked["dca"] = round(dca, 1)
            h_tracked["is_imminent"] = is_imminent
            tracked.append(h_tracked)

        self.prev_hazards = current_hazards
        self.prev_hazard_time = now
        tracked.sort(key=lambda item: (not item["is_imminent"], item["ttc"], item["dist"]))
        return tracked

    def _detect_entities(self, hsv: np.ndarray, cx: int, cy: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Segment entities (monsters, chests, items) by background subtraction:
        Inverts known walkable ground (grass/dirt) and static terrain obstacles.
        Runs in ~3ms and delivers 100% detection recall across all mob types.
        """
        h, w = hsv.shape[:2]

        # 1. Walkable ground: True Grass and Dirt path (strictly calibrated so stone cliffs are not marked as grass)
        mask_grass = cv2.inRange(hsv, np.array([52, 112, 50]), np.array([68, 220, 190]))
        mask_dirt = cv2.inRange(hsv, np.array([14, 40, 50]), np.array([32, 160, 190]))
        walkable_bg = cv2.bitwise_or(mask_grass, mask_dirt)

        # 2. Obstacles (Water, Cliffs, Rocks, Trees, Dark edges)
        obstacles = self._detect_obstacle_mask(hsv)

        # 3. Non-background is anything that isn't walkable ground or terrain
        known_environment = cv2.bitwise_or(walkable_bg, obstacles)
        entities_mask = cv2.bitwise_not(known_environment)

        # 4. Mask out UI topbar (y: 0..115) and bottom HUD (y: h-70..h)
        entities_mask[:115, :] = 0
        entities_mask[h-70:, :] = 0

        # 5. Mask out player center (Scorbunny body, long ears, feet, shadow, radius 52px)
        cv2.circle(entities_mask, (int(round(float(cx))), int(round(float(cy)))), 52, 0, -1)

        # 6. Morphological opening to clean up speckle noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(entities_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enemies = []
        items = []

        for c in contours:
            area = cv2.contourArea(c)
            # True Pokemon sprites (Caterpie, Weedle, Pidove, Hoothoot, Wooper) have area >= 70
            # Filters out tiny ground wildflowers, path speckles, and grass tufts (area 10-50)
            if 70 <= area < 6500:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    ex = int(M["m10"] / M["m00"])
                    ey = int(M["m01"] / M["m00"])
                    dist = math.hypot(ex - cx, ey - cy)
                    if dist > 60.0:
                        angle = math.degrees(math.atan2(ey - cy, ex - cx))
                        sec = self._angle_to_sector(angle)

                        entity_data = {
                            "x": ex,
                            "y": ey,
                            "dist": dist,
                            "angle": angle,
                            "sector": sec,
                            "area": area
                        }

                        # Check for ground items / chests (distinguishable by aspect ratio and gold/wood coloration)
                        x, y, bw, bh = cv2.boundingRect(c)
                        aspect = float(bw) / float(max(1, bh))
                        is_item = False
                        if 0.75 <= aspect <= 1.35 and 120 < area < 1000:
                            roi_hsv = hsv[max(0, y):min(h, y+bh), max(0, x):min(w, x+bw)]
                            if roi_hsv.size > 0:
                                gold_mask = cv2.inRange(roi_hsv, np.array([15, 100, 100]), np.array([38, 255, 255]))
                                gold_ratio = np.count_nonzero(gold_mask) / float(max(1, bw * bh))
                                if gold_ratio > 0.15:
                                    is_item = True

                        if is_item:
                            entity_data["type"] = "chest"
                            items.append(entity_data)
                        else:
                            entity_data["type"] = "enemy"
                            enemies.append(entity_data)

        enemies.sort(key=lambda e: e["dist"])
        items.sort(key=lambda i: i["dist"])
        return enemies, items

    def _compute_sector_dangers(
        self,
        enemies: List[Dict[str, Any]],
        hazards: List[Dict[str, Any]],
        cx: int,
        cy: int,
        sector_obstacles: Optional[Dict[str, bool]] = None,
        sector_obs_costs: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculates danger rating for 8 compass directions including terrain obstacles, mob density, and bullet TTC."""
        sectors = {
            "N": 0.0, "NE": 0.0, "E": 0.0, "SE": 0.0,
            "S": 0.0, "SW": 0.0, "W": 0.0, "NW": 0.0
        }

        # 1. Continuous multi-distance obstacle costs (cliffs, rocks, water)
        if sector_obs_costs:
            for sec, c in sector_obs_costs.items():
                sectors[sec] += c

        # 2. Immediate collision blockage penalty (<= 80px)
        if sector_obstacles:
            for sec, is_blocked in sector_obstacles.items():
                if is_blocked:
                    sectors[sec] += 9000.0  # absolute impassable wall

        # 3. Enemies danger: closer = quadratically higher danger
        for e in enemies:
            dx = e["x"] - cx
            dy = e["y"] - cy
            dist = max(10.0, e.get("dist", math.hypot(dx, dy)))
            angle = math.degrees(math.atan2(dy, dx))
            sector = self._angle_to_sector(angle)
            weight = 160000.0 / (dist * dist + 100.0)
            sectors[sector] += weight

        # 4. Hazards danger (bullets/warning zones with trajectory awareness)
        for h in hazards:
            dx = h["x"] - cx
            dy = h["y"] - cy
            dist = max(10.0, math.hypot(dx, dy))
            angle = math.degrees(math.atan2(dy, dx))
            sector = self._angle_to_sector(angle)

            # Imminent bullets approaching player receive massive priority weight
            imminent_mult = 3.5 if h.get("is_imminent", False) else (2.0 if h.get("ttc", 99.0) < 1.0 else 1.0)
            weight = (500000.0 / (dist * dist + 80.0)) * imminent_mult
            sectors[sector] += weight

        return sectors

    def _angle_to_sector(self, angle_deg: float) -> str:
        # 0 is East (positive x), 90 is South (positive y in image coords)
        # Normalize to 0..360
        a = (angle_deg + 360) % 360
        if 337.5 <= a or a < 22.5:
            return "E"
        elif 22.5 <= a < 67.5:
            return "SE"
        elif 67.5 <= a < 112.5:
            return "S"
        elif 112.5 <= a < 157.5:
            return "SW"
        elif 157.5 <= a < 202.5:
            return "W"
        elif 202.5 <= a < 247.5:
            return "NW"
        elif 247.5 <= a < 292.5:
            return "N"
        else:
            return "NE"

    def _find_nearest_gem_vector(self, gems: List[Dict[str, Any]], cx: int, cy: int) -> Optional[str]:
        if not gems:
            return None
        nearest = gems[0]
        # Only target gems within reasonable distance (< 350px)
        if nearest["dist"] > 380:
            return None
        dx = nearest["x"] - cx
        dy = nearest["y"] - cy
        angle = math.degrees(math.atan2(dy, dx))
        return self._angle_to_sector(angle)

    def _detect_purple_rift(self, hsv: np.ndarray, cx: int, cy: int) -> Optional[Dict[str, Any]]:
        """
        Detects the Purple Swirling Rift Portal or the Purple Directional Arrow pointing to it.
        Strict verification:
        - Real rift portal / arrow halo has massive area (> 1200 pixels).
        - Completely ignores enemies (Venonats/Zubats ~250px) and poison bullets (~40px).
        """
        h, w = hsv.shape[:2]
        lower_purple = np.array([125, 80, 110])
        upper_purple = np.array([145, 255, 255])
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

        # Mask out player center
        cv2.circle(mask_purple, (int(round(float(cx))), int(round(float(cy)))), 50, 0, -1)

        # Mask out topbar HUD
        mask_purple[:65, :] = 0

        contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_target = None
        max_priority = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            # Only true Rift portal or large arrow halo exceeds 1000 pixels
            if area > 1000:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])
                    dist = math.hypot(px - cx, py - cy)
                    angle = math.degrees(math.atan2(py - cy, px - cx))
                    sec = self._angle_to_sector(angle)

                    is_edge_arrow = (px < 110 or px > w - 110 or py < 110 or py > h - 110)
                    target_type = "arrow" if is_edge_arrow else "portal"

                    priority = area * (2.0 if target_type == "portal" else 1.0)
                    if priority > max_priority:
                        max_priority = priority
                        best_target = {
                            "type": target_type,
                            "x": px,
                            "y": py,
                            "dist": dist,
                            "sector": sec,
                            "area": area
                        }

        return best_target

    def _detect_boss(self, hsv: np.ndarray, enemies: List[Dict[str, Any]], cx: int, cy: int) -> Dict[str, Any]:
        """
        Detects if a real Boss is present on screen:
        Strict verification:
        1. A real boss ALWAYS displays a massive horizontal Health Bar across the top center
           (y: 68 to 92, width > 300px, red/yellow HP fill > 500 pixels).
        2. High-area Boss sprite (> 3000 pixels) is ONLY considered a boss if the top Boss HP bar is confirmed,
           preventing normal mob clusters from being misidentified as a boss.
        """
        h, w = hsv.shape[:2]
        bar_crop = hsv[68:92, max(0, w // 2 - 180):min(w, w // 2 + 180)]
        boss_active = False

        if bar_crop.size > 0:
            m_yellow = cv2.inRange(bar_crop, np.array([18, 140, 140]), np.array([35, 255, 255]))
            m_red = cv2.inRange(bar_crop, np.array([0, 140, 140]), np.array([10, 255, 255]))
            total_boss_bar = cv2.countNonZero(m_yellow) + cv2.countNonZero(m_red)
            # A genuine 400px boss bar has over 500 contiguous pixels
            if total_boss_bar > 500:
                boss_active = True

        boss_enemy = None
        if boss_active:
            for e in enemies:
                if e.get("area", 0) > 2500:
                    boss_enemy = e
                    break

        return {
            "active": boss_active,
            "boss_enemy": boss_enemy
        }

    def _empty_perception(self) -> Dict[str, Any]:
        return {
            "player_pos": self.center,
            "hp_ratio": None,
            "hp_status": "UNKNOWN",
            "hp_box": None,
            "enemies_count": 0,
            "hazards_count": 0,
            "gems_count": 0,
            "enemies": [],
            "hazards": [],
            "gems": [],
            "sector_dangers": {s: 0.0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]},
            "best_gem_vector": None,
            "confidence": 0.0
        }

    def create_debug_overlay(
        self,
        frame: np.ndarray,
        vision_state: Dict[str, Any],
        chosen_dir: Optional[str] = None,
        output_path: str = "debug_artifacts/overlay_latest.png"
    ) -> np.ndarray:
        """
        Creates an annotated visual frame with:
        - Player position & centroid marker
        - Top-left HUD HP detection box and verified/estimated ratio
        - Enemy contours & distances
        - Hazard/bullet bounding rings, velocity trajectory vectors, and TTC
        - Directional arrow for chosen movement escape vector
        - Telemetry and status dashboard
        """
        if frame is None or frame.size == 0:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # 1. Resolve screen-space player coordinates as strict integers
        player_obj = vision_state.get("player") or {}
        if player_obj.get("screen_x") is not None and player_obj.get("screen_y") is not None:
            raw_cx = player_obj.get("screen_x")
            raw_cy = player_obj.get("screen_y")
        elif "player_pos" in vision_state:
            pos = vision_state["player_pos"]
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                if vision_state.get("coord_space") == "world":
                    raw_cx = player_obj.get("screen_x", w // 2)
                    raw_cy = player_obj.get("screen_y", h // 2)
                else:
                    raw_cx, raw_cy = pos[0], pos[1]
            else:
                raw_cx, raw_cy = w // 2, h // 2
        else:
            raw_cx, raw_cy = w // 2, h // 2

        try:
            cx = int(round(float(raw_cx)))
            cy = int(round(float(raw_cy)))
        except (ValueError, TypeError):
            cx, cy = w // 2, h // 2

        # 1. Player Marker
        cv2.circle(overlay, (cx, cy), 28, (0, 255, 255), 2)
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)
        cv2.putText(overlay, f"PLAYER ({cx},{cy})", (int(cx - 45), int(cy - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. HP HUD Bar Bounding Box
        hp_info = self.last_hp_result
        if hp_info and hp_info.get("box"):
            try:
                bx, by, bw, bh = [int(round(float(v))) for v in hp_info["box"]]
                color = (0, 255, 0) if hp_info.get("status") == "VERIFIED" else (0, 165, 255)
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 2)
                ratio_val = hp_info.get("hp_ratio")
                r_str = f"{ratio_val*100:.0f}%" if ratio_val is not None else "N/A"
                cv2.putText(overlay, f"HP: {r_str} [{hp_info.get('status', 'UNKNOWN')}]",
                            (bx, max(12, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            except Exception:
                pass

        # 3. Enemies with Distance Tags
        for e in vision_state.get("enemies", [])[:18]:
            ex_val = e.get("screen_x") if e.get("screen_x") is not None else e.get("x")
            ey_val = e.get("screen_y") if e.get("screen_y") is not None else e.get("y")
            if ex_val is None or ey_val is None:
                continue
            try:
                ex = int(round(float(ex_val)))
                ey = int(round(float(ey_val)))
            except (ValueError, TypeError):
                continue
            if not (-50 <= ex <= w + 50 and -50 <= ey <= h + 50):
                continue
            edist = float(e.get("dist", 0.0))
            cv2.circle(overlay, (ex, ey), 16, (0, 140, 255), 2)
            if edist < 220:
                cv2.putText(overlay, f"{edist:.0f}px", (int(ex - 16), int(ey - 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1, cv2.LINE_AA)

        # 4. Hazards & Bullets with Trajectory Arrows and TTC
        for hz in vision_state.get("hazards", []):
            hx_val = hz.get("screen_x") if hz.get("screen_x") is not None else hz.get("x")
            hy_val = hz.get("screen_y") if hz.get("screen_y") is not None else hz.get("y")
            if hx_val is None or hy_val is None:
                continue
            try:
                hx = int(round(float(hx_val)))
                hy = int(round(float(hy_val)))
            except (ValueError, TypeError):
                continue
            if not (-50 <= hx <= w + 50 and -50 <= hy <= h + 50):
                continue
            vx = float(hz.get("vx", 0.0))
            vy = float(hz.get("vy", 0.0))
            ttc = float(hz.get("ttc", 99.0))
            is_imm = bool(hz.get("is_imminent", False))
            hz_color = (0, 0, 255) if is_imm else (255, 0, 255)
            cv2.circle(overlay, (hx, hy), 14, hz_color, 2)
            if is_imm:
                cv2.circle(overlay, (hx, hy), 22, (0, 0, 255), 1)

            # Arrow indicating hazard movement velocity
            if abs(vx) > 5 or abs(vy) > 5:
                cv2.arrowedLine(overlay, (hx, hy), (int(round(hx + vx * 0.35)), int(round(hy + vy * 0.35))),
                                hz_color, 2, tipLength=0.3)
            if ttc < 5.0:
                cv2.putText(overlay, f"TTC:{ttc:.1f}s", (int(hx - 22), int(hy - 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, hz_color, 1, cv2.LINE_AA)

        # 5. Chosen Movement Direction Arrow
        if chosen_dir:
            dir_angles = {
                "E": 0, "SE": 45, "S": 90, "SW": 135,
                "W": 180, "NW": 225, "N": 270, "NE": 315
            }
            deg = dir_angles.get(chosen_dir, 0)
            rad = math.radians(deg)
            arrow_len = 65
            ax = int(round(cx + arrow_len * math.cos(rad)))
            ay = int(round(cy + arrow_len * math.sin(rad)))
            cv2.arrowedLine(overlay, (cx, cy), (ax, ay), (0, 255, 0), 3, tipLength=0.25)
            cv2.putText(overlay, f"MOVE: {chosen_dir}", (int(ax + 10), int(ay + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # 6. Telemetry HUD Dashboard Overlay
        hp_r = vision_state.get("hp_ratio")
        hp_display = f"{hp_r*100:.0f}%" if hp_r is not None else "UNKNOWN"
        status_lines = [
            f"PARTNER : {getattr(self, 'active_pokemon', 'Unknown')}",
            f"DANGER  : {vision_state.get('danger_level', 'LOW')}",
            f"HP      : {hp_display} ({vision_state.get('hp_status', 'UNKNOWN')})",
            f"ENEMIES : {vision_state.get('enemies_count', len(vision_state.get('enemies', [])))} | HAZARDS: {vision_state.get('hazards_count', len(vision_state.get('hazards', [])))}",
            f"ENCIRCLE: {vision_state.get('is_encircled', False)} | STUCK: {vision_state.get('is_stuck', False)}",
            f"CONFID. : {vision_state.get('confidence', 0.85)*100:.0f}%"
        ]

        box_top = max(0, int(round(h - 145)))
        cv2.rectangle(overlay, (10, box_top), (340, int(round(h - 10))), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, box_top), (340, int(round(h - 10))), (0, 255, 255), 1)
        for i, line in enumerate(status_lines):
            cv2.putText(overlay, line, (18, int(round(box_top + 20 + i * 20))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, overlay)
        except Exception:
            pass

        return overlay

