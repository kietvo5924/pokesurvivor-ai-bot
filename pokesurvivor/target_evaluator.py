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
            "player": {"hp_ratio": 1.0, "world_x": 0.0, "world_y": 0.0},
            "specials": {},
            "enemies": [],
            "items": [],
            "hazards": []
        }

    # 1. Player HP ratio & position
    player_data = state.get("player")
    if isinstance(player_data, dict):
        hp_r = player_data.get("hp_ratio")
        if hp_r is None:
            hp_r = state.get("hp_ratio", 1.0)
        px = player_data.get("world_x", player_data.get("x", 0.0))
        py = player_data.get("world_y", player_data.get("y", 0.0))
    else:
        hp_r = state.get("hp_ratio", 1.0)
        pos = state.get("player_pos", (640, 360))
        px, py = float(pos[0]), float(pos[1])

    player = {
        "hp_ratio": max(0.0, min(1.0, float(hp_r))) if hp_r is not None else 1.0,
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

    return {
        "player": player,
        "specials": specials,
        "enemies": enemies,
        "items": items,
        "hazards": hazards
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
        enemies = norm_state["enemies"]
        specials = norm_state["specials"]
        items = norm_state["items"]

        # 1. Check Sudden Burst Damage (Emergency Survival Mode)
        self.hp_history = [(t, hp) for t, hp in self.hp_history if now - t <= 1.8]
        self.hp_history.append((now, hp_ratio))
        if len(self.hp_history) >= 2:
            max_recent_hp = max(hp for _, hp in self.hp_history)
            if (max_recent_hp - hp_ratio) >= 0.25:
                # Took >= 25% burst damage recently: enter emergency survival for 2.5s
                self.emergency_until = now + 2.5

        is_emergency = hp_ratio < 0.35 or now < self.emergency_until

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
                dist = float(e.get("dist", 999.0))
                sec = e.get("sector")
                if not sec and e.get("x") is not None and e.get("y") is not None:
                    sec = angle_to_sector(math.degrees(math.atan2(e["y"], e["x"])) % 360)

                key = f"thief_{is_gold}_{e.get('x', idx)}_{e.get('y', idx)}"
                candidates.append({
                    "tier": tier,
                    "value": val,
                    "type": "thief_gold" if is_gold else "thief_coin",
                    "mode": "CHASE_GHOLDENGO" if is_gold else "CHASE_MEOWTH",
                    "x": e.get("x"),
                    "y": e.get("y"),
                    "dist": dist,
                    "sector": sec or "N",
                    "hold": False,
                    "key": key
                })

        # TIER 4: Purple Rift Portal (Only when safe and not in emergency survival)
        rift = specials.get("rift")
        if rift and not is_emergency:
            rx = rift.get("x", 0)
            ry = rift.get("y", 0)
            key = f"rift_{rx}_{ry}"
            if key not in self.ignore_list:
                r_sec = rift.get("sector")
                if not r_sec and (rx != 0 or ry != 0):
                    r_sec = angle_to_sector(math.degrees(math.atan2(ry, rx)) % 360)
                candidates.append({
                    "tier": 4,
                    "value": 12.0,
                    "type": "rift",
                    "mode": "ENTER_RIFT",
                    "x": rx,
                    "y": ry,
                    "dist": float(rift.get("dist", 300.0)),
                    "sector": r_sec or "N",
                    "hold": True,
                    "key": key
                })

        # TIER 4 or 2: Healing Springs (Lifesaver when hurt!)
        spring = specials.get("spring")
        if spring and hp_ratio < 0.85:
            sx = spring.get("x", 0)
            sy = spring.get("y", 0)
            key = f"spring_{sx}_{sy}"
            if key not in self.ignore_list:
                s_sec = spring.get("sector")
                if not s_sec and (sx != 0 or sy != 0):
                    s_sec = angle_to_sector(math.degrees(math.atan2(sy, sx)) % 360)
                # When critically hurt (< 40%), healing is Tier 4 top emergency priority!
                spring_tier = 4 if hp_ratio < 0.40 else 2
                spring_val = 16.0 if hp_ratio < 0.40 else 7.0
                candidates.append({
                    "tier": spring_tier,
                    "value": spring_val,
                    "type": "spring",
                    "mode": "HEAL_AT_SPRING",
                    "x": sx,
                    "y": sy,
                    "dist": float(spring.get("dist", 200.0)),
                    "sector": s_sec or "N",
                    "hold": True,
                    "key": key
                })

        # TIER 2: Big Chests & Altars (suppressed during emergency)
        chest = specials.get("chest")
        if chest and not is_emergency:
            cx = chest.get("x", 0)
            cy = chest.get("y", 0)
            key = f"chest_{cx}_{cy}"
            if key not in self.ignore_list:
                c_sec = chest.get("sector")
                if not c_sec and (cx != 0 or cy != 0):
                    c_sec = angle_to_sector(math.degrees(math.atan2(cy, cx)) % 360)
                candidates.append({
                    "tier": 2,
                    "value": 6.0,
                    "type": "chest",
                    "mode": "OPEN_CHEST",
                    "x": cx,
                    "y": cy,
                    "dist": float(chest.get("dist", 250.0)),
                    "sector": c_sec or "N",
                    "hold": True,
                    "key": key
                })

        altar = specials.get("altar")
        if altar and not is_emergency:
            ax = altar.get("x", 0)
            ay = altar.get("y", 0)
            key = f"altar_{ax}_{ay}"
            if key not in self.ignore_list:
                a_sec = altar.get("sector")
                if not a_sec and (ax != 0 or ay != 0):
                    a_sec = angle_to_sector(math.degrees(math.atan2(ay, ax)) % 360)
                candidates.append({
                    "tier": 2,
                    "value": 5.0,
                    "type": "altar",
                    "mode": "ACTIVATE_ALTAR",
                    "x": ax,
                    "y": ay,
                    "dist": float(altar.get("dist", 220.0)),
                    "sector": a_sec or "N",
                    "hold": True,
                    "key": key
                })

        # TIER 1: Ground pickups / chests / valuable items (suppressed during emergency)
        if not is_emergency:
            for idx, it in enumerate(items[:10]):
                kind = str(it.get("kind", "xp")).lower()
                dist = float(it.get("dist", 999.0))
                # Only target high value items as explicit candidates
                if kind in ["chest", "magnet", "heal", "berry", "ticket", "bomb"] or it.get("value", 0) > 3.0:
                    val = 4.5 if kind == "chest" else 2.5
                    ix = it.get("x", 0)
                    iy = it.get("y", 0)
                    i_sec = it.get("sector")
                    if not i_sec and (ix != 0 or iy != 0):
                        i_sec = angle_to_sector(math.degrees(math.atan2(iy, ix)) % 360)
                    candidates.append({
                        "tier": 1,
                        "value": val,
                        "type": f"item_{kind}",
                        "mode": "PICKUP_ITEM",
                        "x": ix,
                        "y": iy,
                        "dist": dist,
                        "sector": i_sec or "N",
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
                if hp_ratio > track["hp"] + 0.02:
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
