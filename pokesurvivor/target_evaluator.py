"""
pokesurvivor.target_evaluator
-----------------------------
Tactical Target Prioritization & Candidate Planner.
Ported and adapted from PokeSurvivorTool (core/bot.js & sites/default/pilot.js).

Prioritizes tactical objectives by value, distance, danger, and accessibility:
- Tier 5: Golden Thief (Gholdengo x10 ticket)
- Tier 4: Purple Rift Portal (evolution & boss arena) [Healthy & strong]
- Tier 4 (Emergency): Healing Springs when critically low HP (< 40%)
- Tier 3: Coin Thief (Meowth) & Quest / Escort objectives
- Tier 2: Buff Altars, Healing Springs (when hurt < 85%), Big Chests
- Tier 1: Valuable Ground Pickups (Chests, Magnets, Tickets, Berries) & EXP Clusters
- Tier 0: Normal Enemy Kiting

Maintains an ignore-list with TTL to prevent oscillating or stalling at dried springs,
opened chests, or unreachable terrain.
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple

DIR_SECTORS = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]

def angle_to_sector(angle_deg: float) -> str:
    """Converts angle in degrees [0, 360) to 8 cardinal sectors."""
    idx = int((angle_deg + 22.5) // 45) % 8
    return DIR_SECTORS[idx]

def normalize_game_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes any state input (whether unified_state from GameStateAdapter,
    raw DOM/OpenCV vision_state, or test fixtures) into a standardized schema:
    {
        "player": {"hp_ratio": float, "world_x": float, "world_y": float},
        "specials": {"rift": dict, "spring": dict, "chest": dict, "altar": dict},
        "enemies": list of dicts,
        "items": list of dicts,
        "hazards": list of dicts
    }
    """
    if not isinstance(state, dict):
        return {
            "player": {"hp_ratio": None, "world_x": 0.0, "world_y": 0.0},
            "specials": {},
            "enemies": [],
            "items": [],
            "hazards": []
        }

    # 1. Player HP ratio & position (preserving unobserved None, never faking 100%)
    player_data = state.get("player")
    if isinstance(player_data, dict):
        hp_r = player_data.get("hp_ratio")
        if hp_r is None:
            hp_r = state.get("hp_ratio")
        px = player_data.get("world_x", player_data.get("x"))
        py = player_data.get("world_y", player_data.get("y"))
        if px is None or py is None:
            pos = state.get("player_pos", (640.0, 360.0))
            px, py = float(pos[0]), float(pos[1])
        else:
            px, py = float(px), float(py)
    else:
        hp_r = state.get("hp_ratio")
        pos = state.get("player_pos", (640.0, 360.0))
        px, py = float(pos[0]), float(pos[1])

    player = {
        "hp_ratio": max(0.0, min(1.0, float(hp_r))) if hp_r is not None else None,
        "world_x": px,
        "world_y": py
    }

    # 2. Specials (Rift, Spring, Chest, Altar)
    specials_in = state.get("specials")
    specials: Dict[str, Any] = dict(specials_in) if isinstance(specials_in, dict) else {}
    
    if not specials.get("rift"):
        if state.get("rift_target"):
            specials["rift"] = state["rift_target"]
        elif state.get("rift"):
            specials["rift"] = state["rift"]

    if not specials.get("spring") and state.get("spring"):
        specials["spring"] = state["spring"]

    if not specials.get("chest") and state.get("chest"):
        specials["chest"] = state["chest"]

    if not specials.get("altar") and state.get("altar"):
        specials["altar"] = state["altar"]

    # 3. Enemies
    enemies = list(state.get("enemies") or [])

    # 4. Items (Gems & Pickups)
    items_in = state.get("items")
    if items_in is not None:
        items = list(items_in)
    else:
        items = list(state.get("items_on_ground") or []) + list(state.get("gems") or [])

    # 5. Hazards
    hazards = list(state.get("hazards") or [])

    # 6. Match Context (Elapsed Time & Level for Rift and tactical gating)
    metrics = state.get("metrics") or {}
    elapsed_time = None
    if isinstance(metrics, dict) and metrics.get("time_seconds") is not None:
        elapsed_time = float(metrics["time_seconds"])
    elif state.get("elapsed_time") is not None:
        elapsed_time = float(state["elapsed_time"])
    elif state.get("time_seconds") is not None:
        elapsed_time = float(state["time_seconds"])

    current_level = None
    if isinstance(metrics, dict) and metrics.get("level") is not None:
        current_level = int(metrics["level"])
    elif state.get("current_level") is not None:
        current_level = int(state["current_level"])
    elif state.get("level") is not None:
        current_level = int(state["level"])
    elif isinstance(state.get("ui"), dict) and state["ui"].get("verified_level") is not None:
        current_level = int(state["ui"]["verified_level"])

    return {
        "player": player,
        "specials": specials,
        "enemies": enemies,
        "items": items,
        "hazards": hazards,
        "elapsed_time": elapsed_time,
        "current_level": current_level
    }

class TargetEvaluator:
    def __init__(self):
        # Ignore list: key -> until_timestamp
        self.ignore_list: Dict[str, float] = {}
        # Dwell tracking: key -> {"start_time": float, "last_hp": float}
        self.dwell_tracker: Dict[str, Dict[str, Any]] = {}
        # Recent HP drop history: [(timestamp, hp_ratio)]
        self.hp_history: List[Tuple[float, float]] = []
        self.emergency_until: float = 0.0

    def evaluate_targets(
        self,
        game_state: Dict[str, Any],
        now: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates and selects the highest priority tactical target.
        Returns a dict: {"target_type", "sector", "x", "y", "dist", "weight", "hold", "mode"}
        or None if standard kiting / free-roam should proceed.
        """
        if now is None:
            now = time.time()

        # Clean expired ignores
        self.ignore_list = {k: until for k, until in self.ignore_list.items() if until > now}

        norm_state = normalize_game_state(game_state)
        player = norm_state["player"]
        hp_ratio = player["hp_ratio"]
        px = player["world_x"]
        py = player["world_y"]
        enemies = norm_state["enemies"]
        specials = norm_state["specials"]
        items = norm_state["items"]
        elapsed_time = norm_state.get("elapsed_time")
        current_level = norm_state.get("current_level")

        def is_valid_coord(x, y) -> bool:
            if x is None or y is None:
                return False
            try:
                fx = float(x)
                fy = float(y)
            except (ValueError, TypeError):
                return False
            # Dummy (0, 0) is uninitialized and invalid regardless of source!
            if abs(fx) < 1e-3 and abs(fy) < 1e-3:
                return False
            return True

        # 1. Check Sudden Burst Damage (Emergency Survival Mode)
        is_emergency = False
        if hp_ratio is not None:
            self.hp_history = [(t, hp) for t, hp in self.hp_history if now - t <= 1.8]
            self.hp_history.append((now, hp_ratio))
            if len(self.hp_history) >= 2:
                max_recent_hp = max(hp for _, hp in self.hp_history)
                if (max_recent_hp - hp_ratio) >= 0.25:
                    # Took >= 25% burst damage recently: enter emergency survival for 2.5s
                    self.emergency_until = now + 2.5
            is_emergency = (hp_ratio < 0.35 or now < self.emergency_until)
        else:
            is_emergency = (now < self.emergency_until)

        # 2. Evaluate Candidates
        candidates = []

        # TIER 5 & 3: Thieves (Gholdengo & Meowth)
        for idx, e in enumerate(enemies):
            is_thief_flag = (
                e.get("is_thief", False) or
                e.get("behavior") == "thief" or
                e.get("type") in [1000, 52] or
                e.get("dex") in [1000, 52]
            )
            name_str = str(e.get("name", "")).lower()
            color_str = str(e.get("color_name", "")).lower()

            if is_thief_flag or "gholdengo" in name_str or "meowth" in name_str or "thief" in color_str:
                is_gold = (
                    e.get("type") == 1000 or
                    e.get("dex") == 1000 or
                    "gholdengo" in name_str or
                    "gold" in color_str
                )
                tier = 5 if is_gold else 3
                val = 14.0 if is_gold else 8.0

                ex = e.get("x")
                ey = e.get("y")
                dist = e.get("dist")
                sec = e.get("sector")

                if ex is not None and ey is not None:
                    if not is_valid_coord(ex, ey):
                        continue
                    dx = float(ex) - px
                    dy = float(ey) - py
                    if dist is None:
                        dist = math.hypot(dx, dy)
                    if not sec:
                        sec = angle_to_sector(math.degrees(math.atan2(dy, dx)) % 360)
                elif dist is not None:
                    dist = float(dist)
                else:
                    dist = None

                if dist is None or not sec:
                    continue

                key = f"thief_{is_gold}_{ex if ex is not None else idx}_{ey if ey is not None else idx}"
                candidates.append({
                    "tier": tier,
                    "value": val,
                    "type": "thief_gold" if is_gold else "thief_coin",
                    "mode": "CHASE_GHOLDENGO" if is_gold else "CHASE_MEOWTH",
                    "x": ex,
                    "y": ey,
                    "dist": float(dist),
                    "sector": sec,
                    "hold": False,
                    "key": key
                })

        # TIER 4: Purple Rift Portal
        # STRICT CONDITIONS:
        # 1. Healthy (hp_ratio >= 0.65) and not in emergency!
        # 2. Match Phase: Never enter in early game (< 120s or Level < 3).
        #    Must match MovementPolicy condition exactly:
        #    (target_type == "portal" and elapsed_time >= 120.0 and current_level >= 3) or
        #    (current_level >= 5 and elapsed_time >= 180.0)
        rift = specials.get("rift")
        target_type = rift.get("type", "portal") if isinstance(rift, dict) else "portal"

        # Check early game constraint if match time or level is tracked
        is_early_match = False
        if elapsed_time is not None and elapsed_time < 120.0:
            is_early_match = True
        if current_level is not None and current_level < 3:
            is_early_match = True
        if target_type != "portal" and current_level is not None and current_level < 5:
            is_early_match = True

        can_enter_rift = (not is_early_match) and (hp_ratio is not None and hp_ratio >= 0.65)

        if rift and not is_emergency and can_enter_rift:
            rx = rift.get("x")
            ry = rift.get("y")
            dist = rift.get("dist")
            r_sec = rift.get("sector")

            if rx is not None and ry is not None:
                # Discard dummy (0, 0) regardless of source
                if not is_valid_coord(rx, ry):
                    dist = None
                    r_sec = None
                else:
                    dx = float(rx) - px
                    dy = float(ry) - py
                    if dist is None:
                        dist = math.hypot(dx, dy)
                    if not r_sec:
                        r_sec = angle_to_sector(math.degrees(math.atan2(dy, dx)) % 360)
            elif dist is not None and r_sec is not None:
                dist = float(dist)
            else:
                dist = None
                r_sec = None

            if dist is not None and r_sec:
                key = f"rift_{rx}_{ry}"
                if key not in self.ignore_list:
                    candidates.append({
                        "tier": 4,
                        "value": 12.0,
                        "type": "rift",
                        "mode": "ENTER_RIFT",
                        "x": rx,
                        "y": ry,
                        "dist": float(dist),
                        "sector": r_sec,
                        "hold": True,
                        "key": key
                    })

        # TIER 4 or 2: Healing Springs (Lifesaver when hurt or cautious when HP unknown)
        spring = specials.get("spring")
        if spring and (hp_ratio is None or hp_ratio < 0.85):
            sx = spring.get("x")
            sy = spring.get("y")
            dist = spring.get("dist")
            s_sec = spring.get("sector")

            if sx is not None and sy is not None:
                if not is_valid_coord(sx, sy):
                    dist = None
                    s_sec = None
                else:
                    dx = float(sx) - px
                    dy = float(sy) - py
                    if dist is None:
                        dist = math.hypot(dx, dy)
                    if not s_sec:
                        s_sec = angle_to_sector(math.degrees(math.atan2(dy, dx)) % 360)
            elif dist is not None and s_sec is not None:
                dist = float(dist)
            else:
                dist = None
                s_sec = None

            if dist is not None and s_sec:
                key = f"spring_{sx}_{sy}"
                if key not in self.ignore_list:
                    spring_tier = 4 if (hp_ratio is not None and hp_ratio < 0.40) else 2
                    spring_val = 16.0 if (hp_ratio is not None and hp_ratio < 0.40) else 7.0
                    candidates.append({
                        "tier": spring_tier,
                        "value": spring_val,
                        "type": "spring",
                        "mode": "HEAL_AT_SPRING",
                        "x": sx,
                        "y": sy,
                        "dist": float(dist),
                        "sector": s_sec,
                        "hold": True,
                        "key": key
                    })

        # TIER 2: Big Chests & Altars (suppressed during emergency)
        chest = specials.get("chest")
        if chest and not is_emergency:
            cx = chest.get("x")
            cy = chest.get("y")
            dist = chest.get("dist")
            c_sec = chest.get("sector")

            if cx is not None and cy is not None:
                if not is_valid_coord(cx, cy):
                    dist = None
                    c_sec = None
                else:
                    dx = float(cx) - px
                    dy = float(cy) - py
                    if dist is None:
                        dist = math.hypot(dx, dy)
                    if not c_sec:
                        c_sec = angle_to_sector(math.degrees(math.atan2(dy, dx)) % 360)
            elif dist is not None and c_sec is not None:
                dist = float(dist)
            else:
                dist = None
                c_sec = None

            if dist is not None and c_sec:
                key = f"chest_{cx}_{cy}"
                if key not in self.ignore_list:
                    candidates.append({
                        "tier": 2,
                        "value": 6.0,
                        "type": "chest",
                        "mode": "OPEN_CHEST",
                        "x": cx,
                        "y": cy,
                        "dist": float(dist),
                        "sector": c_sec,
                        "hold": True,
                        "key": key
                    })

        altar = specials.get("altar")
        if altar and not is_emergency:
            ax = altar.get("x")
            ay = altar.get("y")
            dist = altar.get("dist")
            a_sec = altar.get("sector")

            if ax is not None and ay is not None:
                if not is_valid_coord(ax, ay):
                    dist = None
                    a_sec = None
                else:
                    dx = float(ax) - px
                    dy = float(ay) - py
                    if dist is None:
                        dist = math.hypot(dx, dy)
                    if not a_sec:
                        a_sec = angle_to_sector(math.degrees(math.atan2(dy, dx)) % 360)
            elif dist is not None and a_sec is not None:
                dist = float(dist)
            else:
                dist = None
                a_sec = None

            if dist is not None and a_sec:
                key = f"altar_{ax}_{ay}"
                if key not in self.ignore_list:
                    candidates.append({
                        "tier": 2,
                        "value": 5.0,
                        "type": "altar",
                        "mode": "ACTIVATE_ALTAR",
                        "x": ax,
                        "y": ay,
                        "dist": float(dist),
                        "sector": a_sec,
                        "hold": True,
                        "key": key
                    })

        # TIER 1: Ground pickups / valuable items (suppressed during emergency)
        if not is_emergency:
            for idx, it in enumerate(items[:10]):
                kind = str(it.get("kind", "xp")).lower()
                if kind in ["chest", "magnet", "heal", "berry", "ticket", "bomb"] or it.get("value", 0) > 3.0:
                    val = 4.5 if kind == "chest" else 2.5
                    ix = it.get("x")
                    iy = it.get("y")
                    dist = it.get("dist")
                    i_sec = it.get("sector")

                    if ix is not None and iy is not None:
                        dx = ix - px
                        dy = iy - py
                        if dist is None:
                            dist = math.hypot(dx, dy)
                        if not i_sec:
                            i_sec = angle_to_sector(math.degrees(math.atan2(dy, dx)) % 360)
                    elif dist is not None:
                        dist = float(dist)
                    else:
                        dist = None

                    if dist is not None and i_sec:
                        candidates.append({
                            "tier": 1,
                            "value": val,
                            "type": f"item_{kind}",
                            "mode": "PICKUP_ITEM",
                            "x": ix,
                            "y": iy,
                            "dist": float(dist),
                            "sector": i_sec,
                            "hold": False,
                            "key": f"item_{kind}_{ix}_{iy}_{idx}"
                        })

        # Filter out ignored candidates
        viable = [c for c in candidates if c["key"] not in self.ignore_list]
        if not viable:
            return None

        # Sort primarily by Tier (descending), then by Value / Distance
        viable.sort(key=lambda c: (c["tier"], c["value"] / max(20.0, c["dist"])), reverse=True)
        best = viable[0]

        # Dwell tracking for hold targets (chests, springs)
        if best.get("hold") and best.get("dist", 999.0) < 45.0:
            track = self.dwell_tracker.setdefault(best["key"], {"start": now, "hp": hp_ratio})
            dwell_dur = now - track["start"]

            # Spring dried check: if stood for 2.0s without HP gain, spring is depleted
            if best["type"] == "spring":
                if hp_ratio is not None and track.get("hp") is not None and hp_ratio > track["hp"] + 0.02:
                    track["hp"] = hp_ratio
                    track["start"] = now
                elif dwell_dur > 2.0:
                    self.ignore_list[best["key"]] = now + 60.0
                    del self.dwell_tracker[best["key"]]
                    return None

            # Chest opened check: opening takes ~3.5s
            if best["type"] == "chest" and dwell_dur > 3.8:
                self.ignore_list[best["key"]] = now + 60.0
                del self.dwell_tracker[best["key"]]
                return None

            # Altar activated check: activation takes ~2.5s
            if best["type"] == "altar" and dwell_dur > 2.8:
                self.ignore_list[best["key"]] = now + 60.0
                del self.dwell_tracker[best["key"]]
                return None

        return best
