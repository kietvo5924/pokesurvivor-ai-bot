import time
import math
from typing import Dict, Any, List, Optional, Tuple
from pokesurvivor.danger_estimator import DIR_KEYS, OPPOSITE_DIR

# Compass angle mapping (0 = East, 90 = South, 180 = West, 270 = North)
DIR_ANGLES = {
    "E": 0, "SE": 45, "S": 90, "SW": 135,
    "W": 180, "NW": 225, "N": 270, "NE": 315
}

# Lateral (orthogonal/tangential) steering options that preserve forward sprint momentum
# and strictly prevent dead-stop 180-degree twitch reversals back into trailing mobs.
LATERAL_SLIDES = {
    "E": ["SE", "NE", "S"],
    "SE": ["E", "S", "NE"],
    "S": ["SE", "SW", "E"],
    "SW": ["S", "SE", "E"],
    "W": ["SE", "S", "E"],    # Turn away from western lake towards open eastern plains
    "NW": ["NE", "E", "SE"],  # Turn away from NW cliffs
    "N": ["NE", "E", "SE"],
    "NE": ["E", "SE", "N"]
}

# Preferred wide-field cruise directions in Shady Forest (vast open plains with wide turning room)
OPEN_FIELD_PRIORITY = ["E", "NE", "N", "SE", "S"]

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

        def score_cand(s: str) -> float:
            score = sector_dangers.get(s, 0.0)
            mobs = sector_enemy_counts.get(s, 0)
            # Parametric candidate scoring from learned policy weights
            score += mobs * self.mob_penalty_weight
            if mobs == 0:
                score -= self.mob_clear_corridor_bonus
            if s in OPEN_FIELD_PRIORITY:
                score -= 40.0
            # Heavily penalize diving backwards into our own pursuit tail!
            if angle_diff(DIR_ANGLES.get(s, 0), back_ang) < 50.0:
                score += 800.0
            return score

        def pick_safest_open_direction(fallback: str = "SE") -> str:
            # 1. Look for passable + low danger directions favoring open grassland without diving backwards
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
        # ANTI-OSCILLATION DIRECTION COMMITMENT
        # Prevents rapid micro-reversals / vibrating in place ("cứ đi qua lại liên tục")
        # ------------------------------------------------------------------
        is_curr_blocked = sector_obstacles.get(self.current_dir, False)
        has_immed_hazard = any(h.get("dist", 999.0) < 85.0 and h.get("sector") == self.current_dir for h in hazards)
        if now < self.dir_lock_until and not is_curr_blocked and not has_immed_hazard:
            return self.current_dir, DIR_KEYS[self.current_dir], max(0.10, self.dir_lock_until - now)

        # ------------------------------------------------------------------
        # 0. ANTI-STUCK EMERGENCY EJECTION
        # ------------------------------------------------------------------
        if is_stuck or (now - self.stuck_eject_time < 0.5):
            if is_stuck:
                self.stuck_eject_time = now
            eject_dir = pick_safest_open_direction("SE")
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
        is_wall_pinned = len(blocked_secs) >= 1 and (min_enemy_dist < (self.danger_threshold + 15.0) or is_encircled)

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
                if diff > 120.0:
                    continue
                align_score = 180.0 - diff

                mobs = sector_enemy_counts.get(sec, 0)
                # Lao đầu vô hướng quái ít nhất để mở đường thoát
                if mobs == 0:
                    mob_score = self.mob_clear_corridor_bonus * 1.3
                elif mobs <= 2:
                    mob_score = (self.mob_clear_corridor_bonus * 0.5) - (mobs * self.mob_penalty_weight * 0.4)
                else:
                    mob_score = -self.mob_penalty_weight * mobs

                open_bonus = 40.0 if sec in OPEN_FIELD_PRIORITY else 0.0
                score = (align_score * 2.2) + mob_score + open_bonus

                if score > best_wall_score:
                    best_wall_score = score
                    best_wall_escape = sec

            if not best_wall_escape:
                unblocked = [s for s, b in sector_obstacles.items() if not b]
                if unblocked:
                    unblocked.sort(key=lambda s: (sector_enemy_counts.get(s, 0), sector_dangers.get(s, 0.0)))
                    best_wall_escape = unblocked[0]
                else:
                    best_wall_escape = pick_safest_open_direction("SE")

            self.dir_lock_until = now + self.wall_breakout_thrust
            self._set_dir(best_wall_escape, now)
            return best_wall_escape, DIR_KEYS[best_wall_escape], self.wall_breakout_thrust

        # ------------------------------------------------------------------
        # 2. BULLET & ATTACK TELEGRAPH DODGING (Hazards < 160px)
        # ------------------------------------------------------------------
        if hazards:
            closest_hz = hazards[0]
            hz_dist = closest_hz.get("dist", 999.0)
            if hz_dist < (165.0 * self.evasion_weight):
                hx = closest_hz.get("x", cx)
                hy = closest_hz.get("y", cy)
                hz_angle = closest_hz.get("angle", 0.0)

                if hz_dist < 80.0:
                    away_angle = math.degrees(math.atan2(cy - hy, cx - hx))
                    cand_dirs = [
                        self._angle_to_sector(away_angle),
                        self._angle_to_sector((away_angle + 45) % 360),
                        self._angle_to_sector((away_angle - 45 + 360) % 360)
                    ]
                else:
                    cand_dirs = [
                        self._angle_to_sector((hz_angle + 90) % 360),
                        self._angle_to_sector((hz_angle - 90 + 360) % 360)
                    ]

                safe_dodges = [s for s in cand_dirs if not sector_obstacles.get(s, False)]
                if safe_dodges:
                    safe_dodges.sort(key=lambda s: (sector_enemy_counts.get(s, 0), sector_dangers.get(s, 0.0)))
                    dodge_dir = safe_dodges[0]
                else:
                    dodge_dir = pick_safest_open_direction("SE")

                self.dir_lock_until = now + self.dodge_hold_duration
                self._set_dir(dodge_dir, now)
                return dodge_dir, DIR_KEYS[dodge_dir], self.dodge_hold_duration

        # ------------------------------------------------------------------
        # 3. PROACTIVE DISTANCE REPULSION & ANTI-ENCIRCLEMENT BREAKOUT (enemies < danger_threshold or encircled)
        # ------------------------------------------------------------------
        if min_enemy_dist < self.danger_threshold or is_encircled:
            repel_vx = 0.0
            repel_vy = 0.0
            nearby_count = 0
            for e in vision_state.get("enemies", []):
                edist = e.get("dist", 999.0)
                if edist < (self.danger_threshold + 65.0):
                    weight = 1.0 / max(15.0, edist)
                    repel_vx += (cx - e["x"]) * weight
                    repel_vy += (cy - e["y"]) * weight
                    nearby_count += 1

            if nearby_count == 0 and nearest_enemy_sec:
                e_ang = DIR_ANGLES.get(nearest_enemy_sec, 0)
                repel_angle = (e_ang + 180) % 360
            elif abs(repel_vx) < 1e-5 and abs(repel_vy) < 1e-5:
                min_sec_count = 999
                min_count_sec = "SE"
                for sec in OPEN_FIELD_PRIORITY + list(DIR_ANGLES.keys()):
                    if is_passable(sec):
                        cnt = sector_enemy_counts.get(sec, 0)
                        if cnt < min_sec_count:
                            min_sec_count = cnt
                            min_count_sec = sec
                repel_angle = float(DIR_ANGLES.get(min_count_sec, 45))
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
                open_bonus = 35.0 if sec in OPEN_FIELD_PRIORITY else 0.0
                mom_bonus = 15.0 if sec == self.current_dir else (-25.0 if angle_diff(ang, back_ang) < 35.0 else 0.0)

                mobs_in_sec = sector_enemy_counts.get(sec, 0)
                if mobs_in_sec == 0:
                    mob_sparsity_score = self.mob_clear_corridor_bonus
                elif mobs_in_sec == 1:
                    mob_sparsity_score = self.mob_clear_corridor_bonus * 0.35
                else:
                    mob_sparsity_score = -self.mob_penalty_weight * mobs_in_sec

                if is_encircled:
                    score = (mob_sparsity_score * 2.5) + (alignment_score * 1.5) - (danger * 0.03) - (obs_cost * 0.04) + open_bonus
                else:
                    score = (alignment_score * 2.2) + mob_sparsity_score - (danger * 0.04) - (obs_cost * 0.03) + open_bonus + mom_bonus

                if score > best_escape_score:
                    best_escape_score = score
                    best_escape_dir = sec

            if not best_escape_dir:
                best_escape_dir = pick_safest_open_direction("SE")

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
                t1_bias = 25.0 if t1_sec in OPEN_FIELD_PRIORITY else 0.0
                t2_bias = 25.0 if t2_sec in OPEN_FIELD_PRIORITY else 0.0

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
        # 6. EXP GEMS & CHESTS HARVESTING (When safe and unblocked)
        # ------------------------------------------------------------------
        target_gem = gem_cluster_sec or gem_sector
        if target_gem and is_passable(target_gem) and sector_dangers.get(target_gem, 0.0) < 1800.0:
            self._set_dir(target_gem, now)
            return target_gem, DIR_KEYS[target_gem], self.step_duration

        # ------------------------------------------------------------------
        # 7. WIDE-FIELD OPEN GRASSLAND CRUISE (Grand scale linear sweeps)
        # Navigates the vast eastern plains; turns away from western lakes early.
        # ------------------------------------------------------------------
        # If currently heading West/Northwest towards the lake, turn back towards East / Southeast open grasslands!
        if self.current_dir in ("W", "NW") and (now - self.last_change_time > 0.8):
            candidate = "SE" if is_completely_safe("SE") else "E"
            if is_passable(candidate):
                self._set_dir(candidate, now)
                return candidate, DIR_KEYS[candidate], self.step_duration

        current_safe = is_completely_safe(self.current_dir)

        if not current_safe or (now - self.last_change_time > self.cruise_hold_duration):
            if not current_safe:
                new_cruise = pick_safest_open_direction("SE")
            else:
                self.cruise_dir_index = (self.cruise_dir_index + 1) % len(OPEN_FIELD_PRIORITY)
                candidate = OPEN_FIELD_PRIORITY[self.cruise_dir_index]
                new_cruise = candidate if is_completely_safe(candidate) else pick_safest_open_direction("SE")

            self._set_dir(new_cruise, now)

        dir_to_use = self.current_dir if is_passable(self.current_dir) else pick_safest_open_direction("SE")
        return dir_to_use, DIR_KEYS[dir_to_use], self.step_duration

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
