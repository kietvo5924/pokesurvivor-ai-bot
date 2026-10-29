import time
import math
from typing import Dict, Any, List, Optional, Tuple
from pokesurvivor.danger_estimator import (
    DIR_KEYS, OPPOSITE_DIR,
    DIR16_ANGLES, DIR16_KEYS, DIR16_NAMES, OPPOSITE_DIR16,
    angle_to_16_sector, angle_diff_deg
)
from pokesurvivor.target_evaluator import TargetEvaluator

# Compass angle mapping (0 = East, 90 = South, 180 = West, 270 = North)
DIR_ANGLES = {
    "E": 0, "SE": 45, "S": 90, "SW": 135,
    "W": 180, "NW": 225, "N": 270, "NE": 315
}

# Lateral steering options to preserve momentum without full 180 dead-stops
LATERAL_SLIDES = {
    "E": ["SE", "NE", "S"],
    "SE": ["E", "S", "NE"],
    "S": ["SE", "SW", "E"],
    "SW": ["S", "SE", "W"],
    "W": ["SW", "NW", "S"],
    "NW": ["W", "N", "SW"],
    "N": ["NW", "NE", "E"],
    "NE": ["N", "E", "SE"]
}

def angle_diff(a1: float, a2: float) -> float:
    """Calculates minimal angular difference between two angles in degrees (0 to 180)."""
    return angle_diff_deg(a1, a2)

class PredictiveSurvivalPlanner:
    """
    Multi-horizon Predictive Trajectory Rollout Planner (v5.0):
    - Predicts continuous player trajectory across 16 candidate directions over T = 0.8s horizon.
    - Samples M = 5 future timesteps: t in [0.20, 0.40, 0.60, 0.80, 1.00]s.
    - Integrates verified player speed (with buffs/debuffs) from game state.
    - Accounts for player invulnerability frames (i-frames) to prevent over-evasion.
    - Simulates mob velocities and pursuit vectors, penalizing collision (<42px) and crowding (<85px).
    - Checks projectile trajectory CPA/DCA and graze radii.
    - Accounts for hazard/telegraph activation delay (detonation time-to-contact).
    - Evaluates obstacle clearance, cliffs, and map boundaries at every predicted timestep.
    - Scores secondary utilities: Repulsion potential field, EXP gem gravitation, kiting orbit (200-280px).
    - Detects emergency collision risks to immediately break direction lock.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.update_config(config or {})

    def update_config(self, config: Dict[str, Any]):
        self.horizon: float = float(config.get("rollout_horizon", 0.80))
        self.timesteps: List[float] = [0.20, 0.40, 0.60, 0.80, 1.00]
        self.speed_scale: float = float(config.get("speed_scale", 1.65))
        self.mob_penalty_weight: float = float(config.get("mob_penalty_weight", 150.0))
        self.mob_clear_corridor_bonus: float = float(config.get("mob_clear_corridor_bonus", 550.0))
        self.exp_harvest_weight: float = float(config.get("exp_harvest_weight", 0.70))
        self.wall_avoidance_weight: float = float(config.get("wall_avoidance_weight", 2.5))
        self.danger_threshold: float = float(config.get("danger_threshold", 175.0))

    def evaluate_candidates(
        self,
        vision_state: Dict[str, Any],
        current_dir: Optional[str] = "E",
        candidate_mode: int = 16
    ) -> Tuple[str, float, Dict[str, float], Dict[str, Any], bool]:
        """
        Evaluates 16 candidate directions using multi-timestep trajectory rollout.
        Returns:
            (best_direction, best_score, candidate_scores_dict, risk_breakdown_dict, emergency_collision_flag)
        """
        player = vision_state.get("player") or {}
        if "player_pos" in vision_state:
            cx, cy = vision_state["player_pos"]
        elif "world_x" in player and "world_y" in player:
            cx, cy = float(player["world_x"]), float(player["world_y"])
        elif "x" in player and "y" in player:
            cx, cy = float(player["x"]), float(player["y"])
        else:
            cx, cy = 640.0, 360.0

        raw_speed = float(player.get("speed", player.get("move_speed", 100.0)))
        v_player = max(80.0, raw_speed * self.speed_scale)

        is_invuln = bool(
            player.get("invulnerable") or
            player.get("invincible") or
            player.get("iframe_remaining", 0.0) > 0.0 or
            player.get("is_hit", False)
        )
        iframe_risk_factor = 0.25 if is_invuln else 1.0

        hp_ratio = vision_state.get("hp_ratio")
        if hp_ratio is None and isinstance(vision_state.get("player"), dict):
            hp_ratio = vision_state["player"].get("hp_ratio")
        is_hp_unknown = (hp_ratio is None)
        is_low_hp = (hp_ratio is not None and hp_ratio < 0.50)
        repel_mult = 2.0 if is_low_hp else (1.4 if is_hp_unknown else 1.0)

        enemies = vision_state.get("enemies", [])
        hazards = vision_state.get("hazards", [])
        target_gem_sec = vision_state.get("gem_cluster_sector") or vision_state.get("best_gem_vector")

        sector_obstacles = vision_state.get("sector_obstacles", {})
        sector_obs_costs = vision_state.get("sector_obs_costs", {})
        terrain_info = vision_state.get("terrain") or {}
        map_bounds = terrain_info.get("mapBounds")

        directions = DIR16_NAMES if candidate_mode == 16 else list(DIR_ANGLES.keys())
        angles_map = DIR16_ANGLES if candidate_mode == 16 else DIR_ANGLES

        curr_ang = angles_map.get(current_dir or "E", 0.0)

        # -------------------------------------------------------------
        # 0. Compute Proactive Distance Repulsion Potential Field
        # -------------------------------------------------------------
        repel_vx = 0.0
        repel_vy = 0.0
        nearby_enemy_count = 0
        enemy_ahead_count = 0

        effective_danger_thresh = self.danger_threshold + (25.0 if is_low_hp else (15.0 if is_hp_unknown else 0.0))

        for e in enemies:
            ex = float(e.get("x", cx))
            ey = float(e.get("y", cy))
            edist = float(e.get("dist", math.hypot(cx - ex, cy - ey)))
            if edist < (effective_danger_thresh + 65.0):
                w = 1.0 / max(15.0, edist)
                repel_vx += (cx - ex) * w
                repel_vy += (cy - ey) * w
                nearby_enemy_count += 1
                e_angle = math.degrees(math.atan2(ey - cy, ex - cx)) % 360.0
                if angle_diff_deg(e_angle, curr_ang) < 60.0 and edist < 140.0:
                    enemy_ahead_count += 1

        has_repulsion = False
        repel_angle = 0.0
        if nearby_enemy_count > 0 and (abs(repel_vx) > 1e-4 or abs(repel_vy) > 1e-4):
            repel_angle = math.degrees(math.atan2(repel_vy, repel_vx)) % 360.0
            has_repulsion = True

        # -------------------------------------------------------------
        # 1. Check if current course has imminent collision (Emergency Break)
        # -------------------------------------------------------------
        emergency_collision = False
        if current_dir:
            curr_rad = math.radians(curr_ang)
            v_curr_x = v_player * math.cos(curr_rad)
            v_curr_y = v_player * math.sin(curr_rad)

            for t_chk in [0.15, 0.30, 0.45]:
                p_chk_x = cx + v_curr_x * t_chk
                p_chk_y = cy + v_curr_y * t_chk

                for e in enemies:
                    ex = float(e.get("x", cx))
                    ey = float(e.get("y", cy))
                    evx = float(e.get("vx", 0.0))
                    evy = float(e.get("vy", 0.0))
                    if evx == 0.0 and evy == 0.0:
                        ed = math.hypot(cx - ex, cy - ey)
                        if ed > 0.1:
                            espeed = 95.0 if e.get("is_boss") else 75.0
                            evx = espeed * (cx - ex) / ed
                            evy = espeed * (cy - ey) / ed
                    pred_ex = ex + evx * t_chk
                    pred_ey = ey + evy * t_chk
                    if math.hypot(p_chk_x - pred_ex, p_chk_y - pred_ey) < 42.0:
                        emergency_collision = True
                        break

                for h in hazards:
                    hx = float(h.get("x", cx))
                    hy = float(h.get("y", cy))
                    hvx = float(h.get("vx", 0.0))
                    hvy = float(h.get("vy", 0.0))
                    hr = float(h.get("radius", 65.0))
                    httc = float(h.get("ttc", 0.6))
                    pred_hx = hx + hvx * t_chk
                    pred_hy = hy + hvy * t_chk
                    if (h.get("type") in ("projectile", "bullet") or t_chk >= httc) and math.hypot(p_chk_x - pred_hx, p_chk_y - pred_hy) < hr:
                        emergency_collision = True
                        break

                if emergency_collision:
                    break

        candidate_scores: Dict[str, float] = {}
        risk_breakdowns: Dict[str, Dict[str, float]] = {}

        # -------------------------------------------------------------
        # 2. Multi-Timestep Rollout for all candidate directions
        # -------------------------------------------------------------
        for d in directions:
            deg = angles_map.get(d, 0.0)
            rad = math.radians(deg)
            v_px = v_player * math.cos(rad)
            v_py = v_player * math.sin(rad)

            mob_collision_risk = 0.0
            bullet_risk = 0.0
            hazard_risk = 0.0
            obstacle_risk = 0.0

            d8 = d if d in sector_obstacles else self._to_8_sector(deg)
            if sector_obstacles.get(d8, False):
                obstacle_risk += 75000.0

            obs_cost = sector_obs_costs.get(d8, 0.0)
            if obs_cost > 120.0:
                obstacle_risk += obs_cost * 25.0 * self.wall_avoidance_weight

            for t in self.timesteps:
                t_weight = 1.0 / (t + 0.15)
                px_t = cx + v_px * t
                py_t = cy + v_py * t

                if map_bounds:
                    min_x, max_x = map_bounds.get("minX", 0.0), map_bounds.get("maxX", 9999.0)
                    min_y, max_y = map_bounds.get("minY", 0.0), map_bounds.get("maxY", 9999.0)
                    if px_t < (min_x + 35.0) or px_t > (max_x - 35.0) or py_t < (min_y + 35.0) or py_t > (max_y - 35.0):
                        obstacle_risk += 45000.0 * t_weight

                for e in enemies:
                    ex = float(e.get("x", cx))
                    ey = float(e.get("y", cy))
                    evx = float(e.get("vx", 0.0))
                    evy = float(e.get("vy", 0.0))
                    if evx == 0.0 and evy == 0.0:
                        ed = math.hypot(cx - ex, cy - ey)
                        if ed > 0.1:
                            espeed = 95.0 if e.get("is_boss") else 75.0
                            evx = espeed * (cx - ex) / ed
                            evy = espeed * (cy - ey) / ed

                    pred_ex = ex + evx * t
                    pred_ey = ey + evy * t
                    dist_to_mob = math.hypot(px_t - pred_ex, py_t - pred_ey)
                    boss_mult = 2.5 if e.get("is_boss") else 1.0

                    if dist_to_mob < 42.0:
                        mob_collision_risk += 35000.0 * t_weight * boss_mult * iframe_risk_factor
                    elif dist_to_mob < 85.0:
                        mob_collision_risk += (8000.0 / (dist_to_mob + 1.0)) * t_weight * boss_mult * iframe_risk_factor
                    elif dist_to_mob < 140.0:
                        mob_collision_risk += (1200.0 / (dist_to_mob + 1.0)) * boss_mult * iframe_risk_factor

                for h in hazards:
                    hx = float(h.get("x", cx))
                    hy = float(h.get("y", cy))
                    hvx = float(h.get("vx", 0.0))
                    hvy = float(h.get("vy", 0.0))
                    h_type = h.get("type", "hazard")
                    httc = float(h.get("ttc", 0.6))
                    hr = float(h.get("radius", 65.0))

                    pred_hx = hx + hvx * t
                    pred_hy = hy + hvy * t
                    dist_h = math.hypot(px_t - pred_hx, py_t - pred_hy)

                    if h_type in ("projectile", "bullet", "poison_bullet"):
                        if dist_h < 38.0:
                            bullet_risk += 60000.0 * t_weight * iframe_risk_factor
                        elif dist_h < 68.0:
                            bullet_risk += 16000.0 * t_weight * iframe_risk_factor
                    elif h_type in ("telegraph", "hazard", "blast"):
                        if t >= httc:
                            if dist_h < hr:
                                hazard_risk += 48000.0 * t_weight * iframe_risk_factor
                            elif dist_h < (hr + 30.0):
                                hazard_risk += 12000.0 * t_weight * iframe_risk_factor
                        else:
                            if dist_h < hr:
                                hazard_risk += 4000.0 * iframe_risk_factor

            # ---------------------------------------------------------
            # 3. Secondary Utility & Potential Field Alignment
            # ---------------------------------------------------------
            utility_reward = 0.0

            # A. Potential Field Alignment with Repulsion Vector
            if has_repulsion:
                diff_repel = angle_diff_deg(deg, repel_angle)
                repel_alignment = (180.0 - diff_repel) * 2.6 * repel_mult
                utility_reward += repel_alignment

            # B. EXP Cluster Gravitation
            if target_gem_sec:
                gem_ang = angles_map.get(target_gem_sec)
                if gem_ang is None and target_gem_sec in DIR_ANGLES:
                    gem_ang = float(DIR_ANGLES[target_gem_sec])
                if gem_ang is not None:
                    diff_gem = angle_diff_deg(deg, gem_ang)
                    if diff_gem <= 25.0:
                        utility_reward += self.exp_harvest_weight * 280.0
                    elif diff_gem <= 55.0:
                        utility_reward += self.exp_harvest_weight * 140.0

            # C. Kiting Orbit Sweetspot (200px - 280px buffer from mobs)
            px_end = cx + v_px * self.horizon
            py_end = cy + v_py * self.horizon
            if enemies:
                min_end_dist = min(math.hypot(px_end - float(e.get("x", cx)), py_end - float(e.get("y", cy))) for e in enemies)
                if 200.0 <= min_end_dist <= 280.0:
                    utility_reward += 120.0
                elif min_end_dist < 110.0:
                    utility_reward -= 250.0

            # D. Open field corridor clearance bonus
            d_left = self._shift_angle(deg, -30.0)
            d_right = self._shift_angle(deg, +30.0)
            left_8 = self._to_8_sector(d_left)
            right_8 = self._to_8_sector(d_right)
            if not sector_obstacles.get(left_8, False) and not sector_obstacles.get(right_8, False):
                utility_reward += 60.0

            # E. Momentum continuity & Anti-Twitch
            turn_angle = angle_diff_deg(deg, curr_ang)
            if turn_angle <= 25.0:
                utility_reward += 30.0
            elif turn_angle <= 50.0:
                utility_reward += 10.0
            elif turn_angle >= 155.0 and enemy_ahead_count == 0:
                # Only penalize 180 flip if no mob is blocking the front path
                utility_reward -= 350.0

            total_risk = mob_collision_risk + bullet_risk + hazard_risk + obstacle_risk
            final_score = (1000.0 / (1.0 + total_risk * 0.01)) + utility_reward

            candidate_scores[d] = final_score
            risk_breakdowns[d] = {
                "mob_collision_risk": mob_collision_risk,
                "bullet_risk": bullet_risk,
                "hazard_risk": hazard_risk,
                "obstacle_risk": obstacle_risk,
                "utility_reward": utility_reward
            }

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        best_dir, best_score = ranked[0]

        return best_dir, best_score, candidate_scores, risk_breakdowns.get(best_dir, {}), emergency_collision

    def _to_8_sector(self, angle_deg: float) -> str:
        idx = int(((angle_deg + 22.5) % 360.0) // 45.0)
        names = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
        return names[idx]

    def _shift_angle(self, angle_deg: float, delta: float) -> float:
        return (angle_deg + delta + 360.0) % 360.0


class MovementPolicy:
    """
    Predictive Multi-Horizon Kiting & Survival Planner (v5.0):
    - 16-Candidate Direction Evaluation with 0.8s Trajectory Rollout.
    - Verified State Integration: speeds, mob velocities, project CPA, hazard delays.
    - Zero Twitch / Deceleration: Preserves continuous sprint momentum.
    - Emergency Override: Instantly breaks direction lock when path is compromised.
    - Tactical Gating: Safe Rift entry, Boss combat orbits, and item pickup sweeps.
    """
    def __init__(self, policy_config: Dict[str, Any]):
        self.update_config(policy_config)
        self.current_dir: str = "E"
        self.last_change_time = time.time()
        self.dir_lock_until = 0.0
        self.cruise_hold_duration = 2.8
        self.stuck_eject_time = 0.0
        self.last_dodge_time = 0.0
        self.pos_history: List[Tuple[float, float, float]] = []
        self.pin_escape_until = 0.0
        self.pin_escape_dir: Optional[str] = None
        self.target_evaluator = TargetEvaluator()
        self.planner = PredictiveSurvivalPlanner(self.config)
        self.last_candidate_scores: Dict[str, float] = {}
        self.last_risk_breakdown: Dict[str, Any] = {}

    def update_config(self, policy_config: Dict[str, Any]):
        self.config = policy_config.get("movement", {})
        self.pattern = self.config.get("pattern", "kiting")
        self.danger_threshold = float(self.config.get("danger_threshold", 160.0))
        self.mob_penalty_weight = float(self.config.get("mob_penalty_weight", 85.0))
        self.mob_clear_corridor_bonus = float(self.config.get("mob_clear_corridor_bonus", 350.0))
        self.wall_avoidance_weight = float(self.config.get("wall_avoidance_weight", 1.4))
        self.wall_breakout_thrust = float(self.config.get("wall_breakout_thrust", 0.40))
        self.dodge_hold_duration = float(self.config.get("dodge_hold_duration", 0.35))
        self.dir_commit_duration = float(self.config.get("dir_commit_duration", 0.28))
        self.exp_harvest_weight = float(self.config.get("exp_harvest_weight", 0.45))
        self.evasion_weight = float(self.config.get("evasion_weight", 0.95))
        self.step_duration = float(self.config.get("step_duration", 0.24))
        if hasattr(self, "planner"):
            self.planner.update_config(self.config)

    def select_movement(
        self,
        vision_state: Dict[str, Any],
        current_level: int = 1,
        elapsed_time: float = 0.0
    ) -> Tuple[str, List[str], float]:
        now = time.time()
        player_info = vision_state.get("player") or {}
        if "player_pos" in vision_state:
            cx, cy = vision_state["player_pos"]
        elif "world_x" in player_info and "world_y" in player_info:
            cx, cy = float(player_info["world_x"]), float(player_info["world_y"])
        elif "x" in player_info and "y" in player_info:
            cx, cy = float(player_info["x"]), float(player_info["y"])
        else:
            cx, cy = 640.0, 360.0

        sector_dangers = vision_state.get("sector_dangers", {})
        sector_obstacles = vision_state.get("sector_obstacles", {})
        sector_obs_costs = vision_state.get("sector_obs_costs", {})
        is_stuck = vision_state.get("is_stuck", False)
        
        enemies_list = vision_state.get("enemies", [])
        min_enemy_dist = vision_state.get("min_enemy_dist")
        if min_enemy_dist is None:
            if enemies_list:
                min_enemy_dist = min(
                    float(e.get("dist", math.hypot(cx - float(e.get("x", cx)), cy - float(e.get("y", cy)))))
                    for e in enemies_list
                )
            else:
                min_enemy_dist = 9999.0

        nearest_enemy_sec = vision_state.get("nearest_enemy_sector")
        swarm_center = vision_state.get("swarm_center")
        hazards = vision_state.get("hazards", [])
        gem_sector = vision_state.get("best_gem_vector")
        gem_cluster_sec = vision_state.get("gem_cluster_sector")
        items_on_ground = vision_state.get("items_on_ground", [])
        danger_level = vision_state.get("danger_level", "LOW")

        terrain_info = vision_state.get("terrain") or {}
        terrain_status = vision_state.get("terrain_status") or terrain_info.get("status")
        if not terrain_status:
            terrain_status = "verified" if sector_obstacles else "unobserved"

        is_local_observed = (terrain_status in ("verified", "local_view_only"))

        def is_passable(sec: str) -> bool:
            sec8 = sec if sec in sector_obstacles else self.planner._to_8_sector(DIR16_ANGLES.get(sec, 0.0))
            if sector_obstacles.get(sec8, False):
                return False
            if sector_obs_costs.get(sec8, 0.0) > 320.0:
                return False
            return True

        def is_completely_safe(sec: str) -> bool:
            if not is_local_observed:
                return False
            if not is_passable(sec):
                return False
            sec8 = sec if sec in sector_dangers else self.planner._to_8_sector(DIR16_ANGLES.get(sec, 0.0))
            if sector_dangers.get(sec8, 0.0) > 2500.0:
                return False
            return True

        curr_ang = DIR_ANGLES.get(self.current_dir, 0)
        back_ang = (curr_ang + 180) % 360
        sector_enemy_counts = vision_state.get("sector_enemy_counts", {})
        is_encircled = vision_state.get("is_encircled", False)
        hp_ratio = vision_state.get("hp_ratio")
        if hp_ratio is None and isinstance(vision_state.get("player"), dict):
            hp_ratio = vision_state["player"].get("hp_ratio")
        is_hp_unknown = (hp_ratio is None)
        is_low_hp = (hp_ratio is not None and hp_ratio < 0.50)
        effective_danger_thresh = self.danger_threshold + (25.0 if is_low_hp else (15.0 if is_hp_unknown else 0.0))

        def score_cand(s: str) -> float:
            score = sector_dangers.get(s, 0.0)
            mobs = sector_enemy_counts.get(s, 0)
            score += mobs * self.mob_penalty_weight
            if mobs == 0:
                score -= self.mob_clear_corridor_bonus
            close_in_sec = sum(1 for e in vision_state.get("enemies", []) if e.get("sector") == s and e.get("dist", 999.0) < 85.0)
            if close_in_sec > 0:
                score += close_in_sec * 500.0
            deg = DIR_ANGLES.get(s, 0)
            n1 = self._angle_to_sector((deg - 45 + 360) % 360)
            n2 = self._angle_to_sector((deg + 45) % 360)
            if not sector_obstacles.get(n1, False) and not sector_obstacles.get(n2, False):
                score -= 60.0
            if angle_diff(deg, back_ang) < 45.0:
                score += 800.0
            return score

        def pick_safest_open_direction(fallback: str = "E") -> str:
            candidates = [s for s, blocked in sector_obstacles.items() if not blocked and is_completely_safe(s)]
            if candidates:
                candidates.sort(key=score_cand)
                return candidates[0]
            passable = [s for s, blocked in sector_obstacles.items() if not blocked and is_passable(s)]
            if passable:
                passable.sort(key=score_cand)
                return passable[0]
            all_dirs = list(DIR_ANGLES.keys())
            all_dirs.sort(key=lambda s: (sector_obs_costs.get(s, 0.0), sector_enemy_counts.get(s, 0), sector_dangers.get(s, 0.0)))
            return all_dirs[0]

        # ------------------------------------------------------------------
        # 0. PIN DETECTOR & CORNER ESCAPE
        # ------------------------------------------------------------------
        self.pos_history = [p for p in self.pos_history if now - p[0] <= 1.8]
        self.pos_history.append((now, float(cx), float(cy)))

        if now < self.pin_escape_until and self.pin_escape_dir:
            if not sector_obstacles.get(self.pin_escape_dir, False):
                return self.pin_escape_dir, self._keys_for_dir(self.pin_escape_dir), max(0.10, self.pin_escape_until - now)

        if len(self.pos_history) >= 10 and (now - self.pos_history[0][0]) >= 1.0:
            init_x, init_y = self.pos_history[0][1], self.pos_history[0][2]
            net_disp = math.hypot(cx - init_x, cy - init_y)
            has_obstacle = any(sector_obstacles.values()) or any(c > 150.0 for c in sector_obs_costs.values())

            if net_disp < 32.0 and has_obstacle:
                wall_vx, wall_vy = 0.0, 0.0
                for s, blocked in sector_obstacles.items():
                    if blocked or sector_obs_costs.get(s, 0.0) > 80.0:
                        cost_w = max(1.0, sector_obs_costs.get(s, 0.0) / 80.0)
                        ang = math.radians(DIR_ANGLES.get(s, 0))
                        wall_vx -= math.cos(ang) * cost_w
                        wall_vy -= math.sin(ang) * cost_w

                if abs(wall_vx) > 1e-3 or abs(wall_vy) > 1e-3:
                    escape_angle = math.degrees(math.atan2(wall_vy, wall_vx)) % 360
                    escape_dir = self._angle_to_sector(escape_angle)
                    self.pin_escape_dir = escape_dir
                    self.pin_escape_until = now + 0.75
                    self.pos_history.clear()
                    self._set_dir(escape_dir, now)
                    return escape_dir, self._keys_for_dir(escape_dir), 0.75

        # ------------------------------------------------------------------
        # RUN PREDICTIVE SURVIVAL PLANNER (Candidate Trajectory Rollout)
        # ------------------------------------------------------------------
        pred_dir, pred_score, cand_scores, risk_bkdn, emerg_collision = self.planner.evaluate_candidates(
            vision_state,
            current_dir=self.current_dir,
            candidate_mode=16
        )
        self.last_candidate_scores = cand_scores
        self.last_risk_breakdown = risk_bkdn

        if emerg_collision:
            self.dir_lock_until = 0.0

        is_curr_blocked = sector_obstacles.get(self.current_dir, False)
        has_immed_hazard = any(
            (h.get("is_imminent", False) or h.get("ttc", 99.0) < 0.7 or h.get("dist", 999.0) < 85.0) and
            h.get("sector") == self.current_dir for h in hazards
        )

        if now < self.dir_lock_until and not is_curr_blocked and not has_immed_hazard and not emerg_collision:
            return self.current_dir, self._keys_for_dir(self.current_dir), max(0.10, self.dir_lock_until - now)

        # ------------------------------------------------------------------
        # 1. ANTI-STUCK EMERGENCY EJECTION
        # ------------------------------------------------------------------
        if is_stuck or (now - self.stuck_eject_time < 0.5):
            if is_stuck:
                self.stuck_eject_time = now
            eject_dir = pick_safest_open_direction("E")
            self._set_dir(eject_dir, now)
            self.dir_lock_until = now + self.wall_breakout_thrust
            return eject_dir, self._keys_for_dir(eject_dir), self.wall_breakout_thrust

        # ------------------------------------------------------------------
        # 2. DEAD-END & WALL BREAKOUT
        # ------------------------------------------------------------------
        blocked_secs = [s for s, b in sector_obstacles.items() if b]
        is_wall_pinned = len(blocked_secs) >= 1 and (min_enemy_dist < (effective_danger_thresh + 15.0) or is_encircled)

        if is_wall_pinned:
            wall_vx, wall_vy = 0.0, 0.0
            for s in blocked_secs:
                ang = math.radians(DIR_ANGLES.get(s, 0))
                wall_vx -= math.cos(ang)
                wall_vy -= math.sin(ang)

            wall_escape_angle = math.degrees(math.atan2(wall_vy, wall_vx)) % 360
            best_wall_escape = None
            best_wall_score = -99999.0

            for sec, ang in DIR_ANGLES.items():
                if sector_obstacles.get(sec, False):
                    continue
                diff = angle_diff(ang, wall_escape_angle)
                if diff > 130.0:
                    continue
                align_score = 180.0 - diff
                mobs = sector_enemy_counts.get(sec, 0)
                if mobs == 0:
                    mob_score = self.mob_clear_corridor_bonus * 1.4
                elif mobs <= 2:
                    mob_score = (self.mob_clear_corridor_bonus * 0.5) - (mobs * self.mob_penalty_weight * 0.4)
                else:
                    mob_score = -self.mob_penalty_weight * mobs

                score = (align_score * 2.2) + mob_score
                if score > best_wall_score:
                    best_wall_score = score
                    best_wall_escape = sec

            if not best_wall_escape:
                unblocked = [s for s, b in sector_obstacles.items() if not b]
                if unblocked:
                    unblocked.sort(key=lambda s: (sector_enemy_counts.get(s, 0), sector_dangers.get(s, 0.0)))
                    best_wall_escape = unblocked[0]
                else:
                    best_wall_escape = pick_safest_open_direction("E")

            self.dir_lock_until = now + self.wall_breakout_thrust
            self._set_dir(best_wall_escape, now)
            return best_wall_escape, self._keys_for_dir(best_wall_escape), self.wall_breakout_thrust

        # ------------------------------------------------------------------
        # 3. MULTI-HAZARD & BULLET TRAJECTORY DODGING
        # ------------------------------------------------------------------
        if hazards:
            dodge_eval = self._evaluate_multi_hazard_evasion(
                hazards, cx, cy, sector_obstacles, sector_enemy_counts, sector_dangers
            )
            if dodge_eval is not None:
                dodge_dir, hold_t = dodge_eval
                self.dir_lock_until = now + hold_t
                self._set_dir(dodge_dir, now)
                return dodge_dir, self._keys_for_dir(dodge_dir), hold_t

        # ------------------------------------------------------------------
        # 4. PURPLE RIFT PORTAL ENTRY (Strict Early Game Gating)
        # ------------------------------------------------------------------
        rift_target = vision_state.get("rift_target")
        if not rift_target and isinstance(vision_state.get("specials"), dict):
            rift_target = vision_state["specials"].get("rift")
        if rift_target and danger_level == "LOW" and min_enemy_dist > 190.0:
            target_type = rift_target.get("type", "arrow")
            can_enter_rift = (
                hp_ratio is not None and hp_ratio >= 0.65 and (
                    (target_type == "portal" and elapsed_time > 120.0 and current_level >= 3) or
                    (current_level >= 5 and elapsed_time > 180.0)
                )
            )
            if can_enter_rift:
                rift_sec = rift_target.get("sector")
                if rift_sec and is_passable(rift_sec) and sector_dangers.get(rift_sec, 0.0) < 600.0:
                    self._set_dir(rift_sec, now)
                    return rift_sec, self._keys_for_dir(rift_sec), self.step_duration

        # ------------------------------------------------------------------
        # 5. BOSS ARENA COMBAT KITING
        # ------------------------------------------------------------------
        boss_info = vision_state.get("boss_info", {})
        if boss_info.get("active") and boss_info.get("boss_enemy"):
            boss = boss_info["boss_enemy"]
            bx, by = boss["x"], boss["y"]
            b_dist = max(10.0, boss.get("dist", 250.0))
            if b_dist < 210.0:
                away_angle = math.degrees(math.atan2(cy - by, cx - bx))
                boss_dir = pick_safest_open_direction(self._angle_to_sector(away_angle))
            else:
                t_angle = math.degrees(math.atan2(cx - bx, -(cy - by)))
                boss_dir = pick_safest_open_direction(self._angle_to_sector(t_angle))

            self._set_dir(boss_dir, now)
            return boss_dir, self._keys_for_dir(boss_dir), self.step_duration

        # ------------------------------------------------------------------
        # 6. TACTICAL CANDIDATE NAVIGATION (Thieves, Springs, Chests, Shrines)
        # ------------------------------------------------------------------
        eval_state = dict(vision_state)
        eval_state["elapsed_time"] = elapsed_time
        eval_state["current_level"] = current_level
        tactical_target = self.target_evaluator.evaluate_targets(eval_state, now=now)
        if tactical_target and danger_level in ("LOW", "MEDIUM") and min_enemy_dist > 140.0:
            t_sec = tactical_target.get("sector")
            if t_sec and is_passable(t_sec) and sector_dangers.get(t_sec, 0.0) < 800.0:
                self._set_dir(t_sec, now)
                return t_sec, self._keys_for_dir(t_sec), self.step_duration

        # ------------------------------------------------------------------
        # 7. PREDICTIVE SURVIVAL PLANNER / ACTIVE MOB EVASION
        # ------------------------------------------------------------------
        if emerg_collision or min_enemy_dist < effective_danger_thresh or is_encircled or (danger_level in ("HIGH", "MEDIUM") and swarm_center):
            if pred_dir:
                step_time = 0.32 if is_encircled else self.step_duration
                self.dir_lock_until = now + step_time
                self._set_dir(pred_dir, now)
                return pred_dir, self._keys_for_dir(pred_dir), step_time

        # ------------------------------------------------------------------
        # 8. EXP GEMS & CHESTS HARVESTING
        # ------------------------------------------------------------------
        if not is_low_hp and not is_hp_unknown and danger_level == "LOW":
            target_gem = gem_cluster_sec or gem_sector
            if target_gem and is_passable(target_gem) and sector_dangers.get(target_gem, 0.0) < 1400.0:
                self._set_dir(target_gem, now)
                return target_gem, self._keys_for_dir(target_gem), self.step_duration

        # ------------------------------------------------------------------
        # 9. DYNAMIC OPEN ARENA CRUISE
        # ------------------------------------------------------------------
        current_safe = is_completely_safe(self.current_dir)
        if not current_safe or (now - self.last_change_time > self.cruise_hold_duration):
            new_cruise = pick_safest_open_direction("E")
            self._set_dir(new_cruise, now)

        dir_to_use = self.current_dir if is_passable(self.current_dir) else pick_safest_open_direction("E")
        return dir_to_use, self._keys_for_dir(dir_to_use), self.step_duration

    def _keys_for_dir(self, direction: str) -> List[str]:
        if direction in DIR16_KEYS:
            return DIR16_KEYS[direction]
        if direction in DIR_KEYS:
            return DIR_KEYS[direction]
        return ["KeyD"]

    def _evaluate_multi_hazard_evasion(
        self,
        hazards: List[Dict[str, Any]],
        cx: int,
        cy: int,
        sector_obstacles: Dict[str, bool],
        sector_enemy_counts: Dict[str, int],
        sector_dangers: Dict[str, float]
    ) -> Optional[Tuple[str, float]]:
        urgent_hazards = [h for h in hazards if h.get("is_imminent") or h.get("ttc", 99.0) < 1.2 or h.get("dist", 999.0) < 130.0]
        if not urgent_hazards:
            return None

        v_player_speed = 170.0
        step_dt = 0.32
        sector_hazard_risk = {s: 0.0 for s in DIR_ANGLES.keys()}

        for sec, deg in DIR_ANGLES.items():
            if sector_obstacles.get(sec, False):
                sector_hazard_risk[sec] += 99999.0
                continue

            rad = math.radians(deg)
            v_px = v_player_speed * math.cos(rad)
            v_py = v_player_speed * math.sin(rad)

            for h in urgent_hazards:
                hx = h.get("x", cx)
                hy = h.get("y", cy)
                vx = h.get("vx", 0.0)
                vy = h.get("vy", 0.0)
                h_type = h.get("type", "hazard")

                r0_x = cx - hx
                r0_y = cy - hy
                dv_x = v_px - vx
                dv_y = v_py - vy
                dv_sq = dv_x * dv_x + dv_y * dv_y

                if dv_sq > 1e-4:
                    t_star = -(r0_x * dv_x + r0_y * dv_y) / dv_sq
                    t_eval = max(0.0, min(step_dt, t_star))
                else:
                    t_eval = 0.0

                rel_x = r0_x + dv_x * t_eval
                rel_y = r0_y + dv_y * t_eval
                dca_proj = math.hypot(rel_x, rel_y)

                if h_type == "telegraph":
                    if dca_proj < 65.0:
                        sector_hazard_risk[sec] += 9000.0
                else:
                    if dca_proj < 45.0:
                        sector_hazard_risk[sec] += 15000.0 / max(5.0, dca_proj)
                    elif dca_proj < 85.0:
                        sector_hazard_risk[sec] += 4500.0 / max(10.0, dca_proj)

            mobs = sector_enemy_counts.get(sec, 0)
            sector_hazard_risk[sec] += mobs * self.mob_penalty_weight * 1.2
            sector_hazard_risk[sec] += sector_dangers.get(sec, 0.0) * 0.02

        valid_sectors = [s for s, risk in sector_hazard_risk.items() if risk < 8000.0]
        if valid_sectors:
            valid_sectors.sort(key=lambda s: sector_hazard_risk[s])
            return valid_sectors[0], self.dodge_hold_duration

        passable = [s for s in DIR_ANGLES.keys() if not sector_obstacles.get(s, False)]
        if passable:
            passable.sort(key=lambda s: sector_hazard_risk.get(s, 99999.0))
            return passable[0], self.dodge_hold_duration

        return None

    def _set_dir(self, new_dir: str, now: float):
        if new_dir != self.current_dir:
            self.current_dir = new_dir
            self.last_change_time = now
            self.dir_lock_until = max(self.dir_lock_until, now + 0.24)

    def _angle_to_sector(self, angle_deg: float) -> str:
        return self.planner._to_8_sector(angle_deg)
