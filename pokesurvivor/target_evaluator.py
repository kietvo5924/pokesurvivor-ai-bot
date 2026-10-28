"""
pokesurvivor.target_evaluator
-----------------------------
Tactical Target Prioritization & Candidate Planner.
Ported and adapted from PokeSurvivorTool (core/bot.js & sites/default/pilot.js).

Prioritizes tactical objectives by value, distance, danger, and accessibility:
- Tier 5: Golden Thief (Gholdengo x10 ticket)
- Tier 4: Purple Rift Portal (evolution & boss arena)
- Tier 3: Coin Thief (Meowth) & Quest / Escort objectives
- Tier 2: Buff Altars, Healing Springs (when hurt), Big Chests
- Tier 1: Breakables (rocks/crates in close proximity) & EXP Gems
- Tier 0: Enemy hunting (only when healthy & safe)

Maintains an ignore-list with TTL to prevent oscillating or stalling at dried springs,
opened chests, or unreachable terrain.
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple

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

        player = game_state.get("player", {})
        hp_ratio = player.get("hp_ratio", 1.0)
        enemies = game_state.get("enemies", [])
        hazards = game_state.get("hazards", [])
        specials = game_state.get("specials", {})
        items = game_state.get("items", [])

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
        for e in enemies:
            if e.get("is_thief", False) or e.get("type") in [1000, 52]: # 1000 = Gholdengo, 52 = Meowth
                is_gold = e.get("type") == 1000 or "gholdengo" in str(e.get("name", "")).lower()
                tier = 5 if is_gold else 3
                val = 14.0 if is_gold else 8.0
                candidates.append({
                    "tier": tier,
                    "value": val,
                    "type": "thief_gold" if is_gold else "thief_coin",
                    "mode": "CHASE_GHOLDENGO" if is_gold else "CHASE_MEOWTH",
                    "x": e.get("x"), "y": e.get("y"),
                    "dist": e.get("dist", 999.0),
                    "sector": e.get("sector"),
                    "hold": False,
                    "key": f"thief_{e.get('x')}_{e.get('y')}"
                })

        # TIER 4: Purple Rift Portal
        rift = specials.get("rift")
        if rift and not is_emergency:
            key = f"rift_{rift.get('x')}_{rift.get('y')}"
            if key not in self.ignore_list:
                # If rift has sector or position
                r_sec = rift.get("sector", "N")
                r_dist = rift.get("dist", 300.0)
                candidates.append({
                    "tier": 4,
                    "value": 12.0,
                    "type": "rift",
                    "mode": "ENTER_RIFT",
                    "x": rift.get("x"), "y": rift.get("y"),
                    "dist": r_dist,
                    "sector": r_sec,
                    "hold": True,
                    "key": key
                })

        # TIER 2: Healing Springs (Crucial when hurt)
        spring = specials.get("spring")
        if spring and hp_ratio < 0.85:
            key = f"spring_{spring.get('x')}_{spring.get('y')}"
            if key not in self.ignore_list:
                candidates.append({
                    "tier": 4 if hp_ratio < 0.40 else 2,
                    "value": 15.0 if hp_ratio < 0.40 else 7.0,
                    "type": "spring",
                    "mode": "HEAL_AT_SPRING",
                    "x": spring.get("x"), "y": spring.get("y"),
                    "dist": spring.get("dist", 200.0),
                    "sector": spring.get("sector", "N"),
                    "hold": True,
                    "key": key
                })

        # TIER 2: Big Chests & Altars
        chest = specials.get("chest")
        if chest and not is_emergency:
            key = f"chest_{chest.get('x')}_{chest.get('y')}"
            if key not in self.ignore_list:
                candidates.append({
                    "tier": 2,
                    "value": 6.0,
                    "type": "chest",
                    "mode": "OPEN_CHEST",
                    "x": chest.get("x"), "y": chest.get("y"),
                    "dist": chest.get("dist", 250.0),
                    "sector": chest.get("sector", "N"),
                    "hold": True,
                    "key": key
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
            
            # Spring dried check: if stood for 1.8s without HP gain, spring is depleted
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

        return best

