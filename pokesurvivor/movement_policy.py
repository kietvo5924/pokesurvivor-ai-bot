import time
import math
from typing import Dict, Any, List, Optional, Tuple
from pokesurvivor.danger_estimator import DIR_KEYS, OPPOSITE_DIR
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
    diff = (a1 - a2 + 180) % 360 - 180
    return abs(diff)

class MovementPolicy:
    """
    Momentum-Preserving Elliptical Kiting & Bullet Dodging Policy (v4.5):
    - Zero Twitch / Deceleration: Never performs 180-degree dead-stop reversals into chasing packs.
    - Continuous Sprint Speed: Holds smooth directional momentum (0.45s - 1.2s) to naturally outrun mobs.
    - Lateral Evasion: Steps orthogonally (45-90 deg) around obstacles, bullets, and oncoming mobs.
    - EXP Vacuuming: Curves tangential kiting arcs directly through dropped gem clusters and chests.
    - Safe Rift Access: Only approaches Rift when strong enough (Level >= 4 or physical portal on-screen).
    """
    def __init__(self, policy_config: Dict[str, Any]):
        self.update_config(policy_config)
        self.current_dir: str = "E"
        self.last_change_time = time.time()
        self.dir_lock_until = 0.0
        self.cruise_hold_duration = 2.8
        self.cruise_dir_index = 0
        self.stuck_eject_time = 0.0
        self.last_dodge_time = 0.0
        # Temporal Pin Detector (Corner Trapping Prevention)
        self.pos_history: List[Tuple[float, float, float]] = []  # (timestamp, x, y)
        self.pin_escape_until = 0.0
        self.pin_escape_dir: Optional[str] = None
        # Tactical Target Prioritization (Thief, Spring, Chest, Escort)
        self.target_evaluator = TargetEvaluator()

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

    def select_movement(self, vision_state: Dict[str, Any], current_level: int = 1, elapsed_time: float = 0.0) -> Tuple[str, List[str], float]:
        now = time.time()
        cx, cy = vision_state.get("player_pos", (640, 360))
        sector_dangers = vision_state.get("sector_dangers", {})
        sector_obstacles = vision_state.get("sector_obstacles", {})
        sector_obs_costs = vision_state.get("sector_obs_costs", {})
        is_stuck = vision_state.get("is_stuck", False)
        min_enemy_dist = vision_state.get("min_enemy_dist", 9999.0)
        nearest_enemy_sec = vision_state.get("nearest_enemy_sector")
        swarm_center = vision_state.get("swarm_center")
        hazards = vision_state.get("hazards", [])
        gem_sector = vision_state.get("best_gem_vector")
        gem_cluster_sec = vision_state.get("gem_cluster_sector")
        items_on_ground = vision_state.get("items_on_ground", [])
        danger_level = vision_state.get("danger_level", "LOW")

        # ------------------------------------------------------------------
        # Helper functions
        # ------------------------------------------------------------------
        def is_passable(sec: str) -> bool:
            if sector_obstacles.get(sec, False):
                return False
            # Ultra-early terrain avoidance: steer away from cliffs and perimeter at 150-200px
            if sector_obs_costs.get(sec, 0.0) > 320.0:
                return False
            return True

        def is_completely_safe(sec: str) -> bool:
            if not is_passable(sec):
                return False
            if sector_dangers.get(sec, 0.0) > 2500.0:
                return False
            return True

        curr_ang = DIR_ANGLES.get(self.current_dir, 0)
        back_ang = (curr_ang + 180) % 360
        sector_enemy_counts = vision_state.get("sector_enemy_counts", {})
        is_encircled = vision_state.get("is_encircled", False)
        hp_ratio = vision_state.get("hp_ratio")
        if hp_ratio is None and isinstance(vision_state.get("player"), dict):
            hp_ratio = vision_state["player"].get("hp_ratio")
        hp_status = vision_state.get("hp_status", "UNKNOWN")
        if hp_status == "UNKNOWN" and isinstance(vision_state.get("player"), dict):
            hp_status = vision_state["player"].get("source", "UNKNOWN")
        is_low_hp = (hp_ratio is not None and hp_ratio < 0.50)
        effective_danger_thresh = self.danger_threshold + (25.0 if is_low_hp else 0.0)

        def score_cand(s: str) -> float:
            score = sector_dangers.get(s, 0.0)
            mobs = sector_enemy_counts.get(s, 0)
            score += mobs * self.mob_penalty_weight
            if mobs == 0:
                score -= self.mob_clear_corridor_bonus
            # Quadratic close contact penalty for any mob in sector within 85px
            close_in_sec = sum(1 for e in vision_state.get("enemies", []) if e.get("sector") == s and e.get("dist", 999.0) < 85.0)
            if close_in_sec > 0:
                score += close_in_sec * 500.0
            # Dynamic open field clearance bonus: reward sectors whose adjacent sectors are also open
            deg = DIR_ANGLES.get(s, 0)
            n1 = self._angle_to_sector((deg - 45 + 360) % 360)
            n2 = self._angle_to_sector((deg + 45) % 360)
            if not sector_obstacles.get(n1, False) and not sector_obstacles.get(n2, False):
                score -= 60.0  # broad open escape corridor
            # Penalize diving backwards into our own pursuit tail
            if angle_diff(deg, back_ang) < 45.0:
                score += 800.0
            return score

        def pick_safest_open_direction(fallback: str = "E") -> str:
            # 1. Look for passable + low danger directions with wide open clearance
            candidates = [s for s, blocked in sector_obstacles.items() if not blocked and is_completely_safe(s)]
            if candidates:
                candidates.sort(key=score_cand)
                return candidates[0]
            # 2. Look for any passable direction
            passable = [s for s, blocked in sector_obstacles.items() if not blocked and is_passable(s)]
            if passable:
                passable.sort(key=score_cand)
                return passable[0]
            # 3. Dynamic least-resistance fallback: sector with lowest terrain obstacle cost & mob count
            all_dirs = list(DIR_ANGLES.keys())
            all_dirs.sort(key=lambda s: (sector_obs_costs.get(s, 0.0), sector_enemy_counts.get(s, 0), sector_dangers.get(s, 0.0)))
            return all_dirs[0]

        # ------------------------------------------------------------------
        # PIN DETECTOR & CORNER ESCAPE (From PokeSurvivorTool analysis)
        # ------------------------------------------------------------------
        self.pos_history = [p for p in self.pos_history if now - p[0] <= 1.8]
        self.pos_history.append((now, float(cx), float(cy)))

        if now < self.pin_escape_until and self.pin_escape_dir:
            if not sector_obstacles.get(self.pin_escape_dir, False):
                return self.pin_escape_dir, DIR_KEYS[self.pin_escape_dir], max(0.10, self.pin_escape_until - now)

        if len(self.pos_history) >= 10 and (now - self.pos_history[0][0]) >= 1.0:
            init_x, init_y = self.pos_history[0][1], self.pos_history[0][2]
            net_disp = math.hypot(cx - init_x, cy - init_y)
            has_obstacle = any(sector_obstacles.values()) or any(c > 150.0 for c in sector_obs_costs.values())
            
            if net_disp < 32.0 and has_obstacle:
                wall_vx, wall_vy = 0.0, 0.0
                for s, blocked in sector_obstacles.items():
                    if blocked or sector_obs_costs.get(s, 0.0) > 80.0:
                        cost_w = max(1.0, sector_obs_costs.get(s, 0.0) / 80.0)
                        ang = math.radians(DIR_ANGLES[s])
                        wall_vx -= math.cos(ang) * cost_w
                        wall_vy -= math.sin(ang) * cost_w

                if abs(wall_vx) > 1e-3 or abs(wall_vy) > 1e-3:
                    escape_angle = math.degrees(math.atan2(wall_vy, wall_vx)) % 360
                    escape_dir = self._angle_to_sector(escape_angle)
                    self.pin_escape_dir = escape_dir
                    self.pin_escape_until = now + 0.75
                    self.pos_history.clear()
                    self._set_dir(escape_dir, now)
                    return escape_dir, DIR_KEYS[escape_dir], 0.75

        # ------------------------------------------------------------------
        # ANTI-OSCILLATION DIRECTION COMMITMENT
        # Prevents rapid micro-reversals while permitting immediate breakout on hazard
        # ------------------------------------------------------------------
        is_curr_blocked = sector_obstacles.get(self.current_dir, False)
        has_immed_hazard = any((h.get("is_imminent", False) or h.get("ttc", 99.0) < 0.7 or h.get("dist", 999.0) < 85.0) and h.get("sector") == self.current_dir for h in hazards)
        if now < self.dir_lock_until and not is_curr_blocked and not has_immed_hazard:
            return self.current_dir, DIR_KEYS[self.current_dir], max(0.10, self.dir_lock_until - now)

        # ------------------------------------------------------------------
        # 0. ANTI-STUCK EMERGENCY EJECTION
        # ------------------------------------------------------------------
        if is_stuck or (now - self.stuck_eject_time < 0.5):
            if is_stuck:
                self.stuck_eject_time = now
            eject_dir = pick_safest_open_direction("E")
            self._set_dir(eject_dir, now)
            self.dir_lock_until = now + self.wall_breakout_thrust
            return eject_dir, DIR_KEYS[eject_dir], self.wall_breakout_thrust

        # ------------------------------------------------------------------
        # 1. DEAD-END & WALL BREAKOUT (When pinned against walls/cliffs with quái nearby)
        # USER MANDATE:
        # "Vô đường cụt thì phải cho điều kiện lao đầu vô quái (nhưng vẫn phải tính đường để xem quái đâu ít nhất
        # hoặc là đường đó gần sát quái chưa đụng hẳn vô quái) để thoát ra chứ nó cứ đâm vô tường"
        # ------------------------------------------------------------------
        blocked_secs = [s for s, b in sector_obstacles.items() if b]
        is_wall_pinned = len(blocked_secs) >= 1 and (min_enemy_dist < (effective_danger_thresh + 15.0) or is_encircled)

        if is_wall_pinned:
            # Outward escape vector pointing AWAY from all blocked terrain walls
            wall_vx = 0.0
            wall_vy = 0.0
            for s in blocked_secs:
                ang = math.radians(DIR_ANGLES[s])
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
                # Lao đầu vô hướng quái ít nhất để mở đường thoát
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
            return best_wall_escape, DIR_KEYS[best_wall_escape], self.wall_breakout_thrust

        # ------------------------------------------------------------------
        # 2. MULTI-HAZARD & BULLET TRAJECTORY DODGING (TTC & DCA Informed)
        # ------------------------------------------------------------------
        if hazards:
            dodge_eval = self._evaluate_multi_hazard_evasion(
                hazards, cx, cy, sector_obstacles, sector_enemy_counts, sector_dangers
            )
            if dodge_eval is not None:
                dodge_dir, hold_t = dodge_eval
                self.dir_lock_until = now + hold_t
                self._set_dir(dodge_dir, now)
                return dodge_dir, DIR_KEYS[dodge_dir], hold_t

        # ------------------------------------------------------------------
        # 3. PROACTIVE DISTANCE REPULSION & ANTI-ENCIRCLEMENT BREAKOUT (enemies < effective_danger_thresh or encircled)
        # ------------------------------------------------------------------
        if min_enemy_dist < effective_danger_thresh or is_encircled:
            repel_vx = 0.0
            repel_vy = 0.0
            nearby_count = 0
            for e in vision_state.get("enemies", []):
                edist = e.get("dist", 999.0)
                if edist < (effective_danger_thresh + 65.0):
                    weight = 1.0 / max(15.0, edist)
                    repel_vx += (cx - e["x"]) * weight
                    repel_vy += (cy - e["y"]) * weight
                    nearby_count += 1

            if nearby_count == 0 and nearest_enemy_sec:
                e_ang = DIR_ANGLES.get(nearest_enemy_sec, 0)
                repel_angle = (e_ang + 180) % 360
            elif abs(repel_vx) < 1e-5 and abs(repel_vy) < 1e-5:
                min_sec_count = 999
                min_count_sec = "E"
                for sec in list(DIR_ANGLES.keys()):
                    if is_passable(sec):
                        cnt = sector_enemy_counts.get(sec, 0)
                        if cnt < min_sec_count:
                            min_sec_count = cnt
                            min_count_sec = sec
                repel_angle = float(DIR_ANGLES.get(min_count_sec, 0))
            else:
                repel_angle = math.degrees(math.atan2(repel_vy, repel_vx))

            best_escape_dir = None
            best_escape_score = -99999.0

            curr_ang = DIR_ANGLES.get(self.current_dir, 0)
            back_ang = (curr_ang + 180) % 360

            for sec, ang in DIR_ANGLES.items():
                if not is_passable(sec):
                    continue
                diff = angle_diff(ang, repel_angle)
                alignment_score = 180.0 - diff

                danger = sector_dangers.get(sec, 0.0)
                obs_cost = sector_obs_costs.get(sec, 0.0) * self.wall_avoidance_weight
                mom_bonus = 15.0 if sec == self.current_dir else (-25.0 if angle_diff(ang, back_ang) < 35.0 else 0.0)

                mobs_in_sec = sector_enemy_counts.get(sec, 0)
                if mobs_in_sec == 0:
                    mob_sparsity_score = self.mob_clear_corridor_bonus
                elif mobs_in_sec == 1:
                    mob_sparsity_score = self.mob_clear_corridor_bonus * 0.35
                else:
                    mob_sparsity_score = -self.mob_penalty_weight * mobs_in_sec

                repel_mult = 2.0 if is_low_hp else 1.0
                if is_encircled:
                    score = (mob_sparsity_score * 2.5) + (alignment_score * 1.5 * repel_mult) - (danger * 0.03) - (obs_cost * 0.04)
                else:
                    score = (alignment_score * 2.2 * repel_mult) + mob_sparsity_score - (danger * 0.04) - (obs_cost * 0.03) + mom_bonus

                if score > best_escape_score:
                    best_escape_score = score
                    best_escape_dir = sec

            if not best_escape_dir:
                best_escape_dir = pick_safest_open_direction("E")

            step_time = 0.32 if is_encircled else self.step_duration
            self.dir_lock_until = now + step_time
            self._set_dir(best_escape_dir, now)
            return best_escape_dir, DIR_KEYS[best_escape_dir], step_time

        # ------------------------------------------------------------------
        # 3. PURPLE RIFT PORTAL ENTRY (Zero-risk Boss Arena: Items + Bonds!)
        # STRICT CONDITIONS: Only enter when powered up (elapsed > 120s and Level >= 3).
        # Never enter at early game (0-2 mins) when we need to farm EXP in open fields!
        # ------------------------------------------------------------------
        rift_target = vision_state.get("rift_target")
        if not rift_target and isinstance(vision_state.get("specials"), dict):
            rift_target = vision_state["specials"].get("rift")
        if rift_target and danger_level == "LOW" and min_enemy_dist > 190.0:
            target_type = rift_target.get("type", "arrow")
            can_enter_rift = (target_type == "portal" and elapsed_time > 120.0 and current_level >= 3) or (current_level >= 5 and elapsed_time > 180.0)
            if can_enter_rift:
                rift_sec = rift_target.get("sector")
                if rift_sec and is_passable(rift_sec) and sector_dangers.get(rift_sec, 0.0) < 600.0:
                    self._set_dir(rift_sec, now)
                    return rift_sec, DIR_KEYS[rift_sec], self.step_duration

        # ------------------------------------------------------------------
        # 4. BOSS ARENA COMBAT KITING (Maintain 220-280px orbit around confirmed Boss)
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
            return boss_dir, DIR_KEYS[boss_dir], self.step_duration

        # ------------------------------------------------------------------
        # 4b. TACTICAL CANDIDATE NAVIGATION (Thieves, Springs, Chests, Shrines)
        # ------------------------------------------------------------------
        tactical_target = self.target_evaluator.evaluate_targets(vision_state, now=now)
        if tactical_target and danger_level in ("LOW", "MEDIUM") and min_enemy_dist > 140.0:
            t_sec = tactical_target.get("sector")
            if t_sec and is_passable(t_sec) and sector_dangers.get(t_sec, 0.0) < 800.0:
                self._set_dir(t_sec, now)
                return t_sec, DIR_KEYS[t_sec], self.step_duration

        # ------------------------------------------------------------------
        # 5. TACTICAL SWARM KITING & EXP VACUUMING (Enemies within 135-260px)
        # Tangential motion holds constant 200-260px buffer and packs mobs
        # in a straight chase line for 100% Ember/Fire Blast pierce hits!
        # Actively curves the kiting arc through EXP gems and chests!
        # ------------------------------------------------------------------
        if (danger_level in ("HIGH", "MEDIUM") or min_enemy_dist < 260.0) and swarm_center:
            sx, sy = swarm_center
            dx = cx - sx
            dy = cy - sy

            t1_angle = math.degrees(math.atan2(dx, -dy))
            t2_angle = math.degrees(math.atan2(-dx, dy))
            t1_sec = self._angle_to_sector(t1_angle)
            t2_sec = self._angle_to_sector(t2_angle)

            t1_ok = is_completely_safe(t1_sec)
            t2_ok = is_completely_safe(t2_sec)

            if t1_ok and not t2_ok:
                kite_dir = t1_sec
            elif t2_ok and not t1_ok:
                kite_dir = t2_sec
            elif t1_ok and t2_ok:
                # Dynamic open field clearance: reward direction with open adjacent flanks
                deg1 = DIR_ANGLES.get(t1_sec, 0)
                deg2 = DIR_ANGLES.get(t2_sec, 0)
                n1_a = self._angle_to_sector((deg1 - 45 + 360) % 360)
                n1_b = self._angle_to_sector((deg1 + 45) % 360)
                n2_a = self._angle_to_sector((deg2 - 45 + 360) % 360)
                n2_b = self._angle_to_sector((deg2 + 45) % 360)

                t1_bias = 25.0 if (not sector_obstacles.get(n1_a, False) and not sector_obstacles.get(n1_b, False)) else 0.0
                t2_bias = 25.0 if (not sector_obstacles.get(n2_a, False) and not sector_obstacles.get(n2_b, False)) else 0.0

                # Favor tangential direction with FEWER enemies (using learned policy weights)
                mobs_t1 = sector_enemy_counts.get(t1_sec, 0)
                mobs_t2 = sector_enemy_counts.get(t2_sec, 0)
                t1_bias += (self.mob_clear_corridor_bonus * 0.45) if mobs_t1 == 0 else (-self.mob_penalty_weight * 0.6 * mobs_t1)
                t2_bias += (self.mob_clear_corridor_bonus * 0.45) if mobs_t2 == 0 else (-self.mob_penalty_weight * 0.6 * mobs_t2)

                # Strongly bias tangential path towards EXP gem clusters (using learned exp_harvest_weight)
                target_gem = gem_cluster_sec or gem_sector
                if target_gem:
                    gem_w = self.exp_harvest_weight * 130.0
                    if t1_sec == target_gem:
                        t1_bias += gem_w
                    elif angle_diff(DIR_ANGLES[t1_sec], DIR_ANGLES[target_gem]) <= 45:
                        t1_bias += gem_w * 0.55
                    if t2_sec == target_gem:
                        t2_bias += gem_w
                    elif angle_diff(DIR_ANGLES[t2_sec], DIR_ANGLES[target_gem]) <= 45:
                        t2_bias += gem_w * 0.55

                # Check nearby ground items/chests
                if items_on_ground:
                    i_sec = items_on_ground[0].get("sector")
                    if i_sec:
                        if t1_sec == i_sec: t1_bias += 50.0
                        if t2_sec == i_sec: t2_bias += 50.0

                # Penalize diving backwards into our own pursuit tail (the pack chasing behind us)
                curr_ang = DIR_ANGLES.get(self.current_dir, 0)
                back_ang = (curr_ang + 180) % 360
                if angle_diff(DIR_ANGLES[t1_sec], back_ang) < 50.0:
                    t1_bias -= 250.0
                if angle_diff(DIR_ANGLES[t2_sec], back_ang) < 50.0:
                    t2_bias -= 250.0

                score1 = sector_dangers.get(t1_sec, 0.0) - t1_bias
                score2 = sector_dangers.get(t2_sec, 0.0) - t2_bias
                kite_dir = t1_sec if score1 <= score2 else t2_sec
            else:
                kite_dir = pick_safest_open_direction("SE")

            # Direct gem sweep when mobs are at a comfortable buffer (> 170px)
            target_gem = gem_cluster_sec or gem_sector
            if target_gem and min_enemy_dist > 170.0 and is_passable(target_gem):
                kite_dir = target_gem

            self._set_dir(kite_dir, now)
            return kite_dir, DIR_KEYS[kite_dir], self.step_duration

        # ------------------------------------------------------------------
        # 6. EXP GEMS & CHESTS HARVESTING (Strictly gated by health and safety)
        # Never dive into mobs for gems if low HP (< 50%) or if danger is elevated!
        # ------------------------------------------------------------------
        if not is_low_hp and danger_level == "LOW":
            target_gem = gem_cluster_sec or gem_sector
            if target_gem and is_passable(target_gem) and sector_dangers.get(target_gem, 0.0) < 1400.0:
                self._set_dir(target_gem, now)
                return target_gem, DIR_KEYS[target_gem], self.step_duration

        # ------------------------------------------------------------------
        # 7. DYNAMIC OPEN ARENA CRUISE (Smooth sweeps without hardcoded direction bias)
        # Evaluates open field clearance, obstacles, and mob absence dynamically.
        # ------------------------------------------------------------------
        current_safe = is_completely_safe(self.current_dir)

        if not current_safe or (now - self.last_change_time > self.cruise_hold_duration):
            new_cruise = pick_safest_open_direction("E")
            self._set_dir(new_cruise, now)

        dir_to_use = self.current_dir if is_passable(self.current_dir) else pick_safest_open_direction("E")
        return dir_to_use, DIR_KEYS[dir_to_use], self.step_duration

    def _evaluate_multi_hazard_evasion(
        self,
        hazards: List[Dict[str, Any]],
        cx: int,
        cy: int,
        sector_obstacles: Dict[str, bool],
        sector_enemy_counts: Dict[str, int],
        sector_dangers: Dict[str, float]
    ) -> Optional[Tuple[str, float]]:
        """
        Multi-bullet and warning zone trajectory evasion:
        Evaluates projected positions across ALL hazards within 1.2s TTC.
        Selects the safest lateral escape sector that minimizes combined bullet risk,
        avoids terrain walls, and minimizes mob density.
        """
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

                # Synchronized continuous closest approach over t in [0, step_dt]
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

                # Player and hazard positions evaluated at identical future time t_eval
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
