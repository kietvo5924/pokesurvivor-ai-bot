import time
from typing import Dict, Any, List, Optional

class StateTracker:
    def __init__(self):
        self.active_pokemon = "Unknown"
        self.reset()

    def set_active_pokemon(self, name: str):
        if name:
            self.active_pokemon = name

    def reset(self):
        self.start_time = time.time()
        self.duration = 0.0
        self.current_level = 1
        self.upgrades_chosen: List[Dict[str, Any]] = []
        self.hp_history: List[float] = []
        self.damage_events: int = 0
        self.stuck_events: int = 0
        self.dodge_events: int = 0
        self.buffer_violations: int = 0
        self.encircled_events: int = 0
        self.total_frames: int = 0
        self.min_distance_seen: float = 9999.0
        self.last_hp: Optional[float] = None
        self.last_hp_status: str = "UNKNOWN"
        self.last_danger_level = "LOW"
        self.last_enemies_count = 0
        self.last_stuck = False
        self.end_reason = "unknown"
        self.is_active = False
        self.damage_incidents: List[Dict[str, Any]] = []
        self.trajectory_snapshots: List[Dict[str, Any]] = []
        self.last_snapshot_time = 0.0

    def start_run(self):
        self.reset()
        self.is_active = True
        self.start_time = time.time()

    def update(self, dom_state: Dict[str, Any], vision_state: Dict[str, Any]):
        if not self.is_active:
            return
        
        self.duration = time.time() - self.start_time
        self.total_frames += 1

        if vision_state.get("is_encircled", False):
            self.encircled_events += 1

        # Verified HP tracking: only record real measurements
        hp = vision_state.get("hp_ratio")
        hp_status = vision_state.get("hp_status", "UNKNOWN")
        self.last_hp_status = hp_status

        if hp is not None and hp_status in ("VERIFIED", "ESTIMATED"):
            self.hp_history.append(hp)
            if self.last_hp is not None and hp < self.last_hp - 0.08:
                self.damage_events += 1
                self.damage_incidents.append({
                    "time": round(self.duration, 1),
                    "hp_drop": round(self.last_hp - hp, 2),
                    "hp_after": round(hp, 2),
                    "min_enemy_dist": round(vision_state.get("min_enemy_dist", 999.0), 1),
                    "is_encircled": vision_state.get("is_encircled", False),
                    "near_wall": any(vision_state.get("sector_obstacles", {}).values()),
                    "hazards_count": len(vision_state.get("hazards", [])),
                    "enemies_count": vision_state.get("enemies_count", 0)
                })
                self.last_hp = hp
            elif self.last_hp is not None and hp > self.last_hp:
                self.last_hp = hp
            elif self.last_hp is None:
                self.last_hp = hp

        min_e_dist = vision_state.get("min_enemy_dist", 9999.0)
        if min_e_dist < self.min_distance_seen:
            self.min_distance_seen = min_e_dist
        if min_e_dist < 140.0:
            self.buffer_violations += 1

        hazards = vision_state.get("hazards", [])
        if hazards and (hazards[0].get("dist", 999.0) < 165.0 or hazards[0].get("is_imminent", False)):
            self.dodge_events += 1

        is_stuck = vision_state.get("is_stuck", False)
        if is_stuck and not self.last_stuck:
            self.stuck_events += 1
        self.last_stuck = is_stuck

        self.last_danger_level = vision_state.get("danger_level", "LOW")
        self.last_enemies_count = vision_state.get("enemies_count", 0)

        # Trajectory snapshot
        if self.duration - self.last_snapshot_time >= 1.5:
            self.trajectory_snapshots.append({
                "time": round(self.duration, 1),
                "hp": round(hp, 2) if hp is not None else None,
                "hp_status": hp_status,
                "min_dist": round(min_e_dist, 1),
                "mobs": self.last_enemies_count,
                "hazards": len(hazards),
                "encircled": vision_state.get("is_encircled", False),
                "near_wall": any(vision_state.get("sector_obstacles", {}).values())
            })
            self.last_snapshot_time = self.duration

    def record_upgrade(
        self,
        upgrade: Dict[str, Any],
        modal_kind: str = "level-up",
        verified_level: Optional[int] = None
    ):
        """
        Records chosen upgrade card.
        Strict verification:
        - ONLY increments/updates character level on genuine 'level-up' modals.
        - Uses DOM verified_level if available.
        - Does NOT increment level on chest rewards, held items, or bond combinations!
        """
        card_name = upgrade.get("name", "Unknown")
        now_time = round(self.duration, 1)

        if self.upgrades_chosen:
            last_entry = self.upgrades_chosen[-1]
            if last_entry.get("name") == card_name and abs(now_time - last_entry.get("time", 0.0)) < 1.5:
                return

        if modal_kind == "level-up":
            if verified_level is not None and verified_level > 0:
                self.current_level = max(self.current_level, verified_level)
            else:
                self.current_level += 1

        self.upgrades_chosen.append({
            "modal_kind": modal_kind,
            "level": self.current_level,
            "name": card_name,
            "tag": upgrade.get("tag", ""),
            "time": now_time
        })

    def end_run(self, reason: str = "defeated") -> Dict[str, Any]:
        self.is_active = False
        self.duration = time.time() - self.start_time

        # Honest termination diagnosis: if telemetry is insufficient, record 'unknown'
        if reason == "defeated":
            has_damage_proof = self.damage_events > 0 or (self.last_hp is not None and self.last_hp < 0.25)
            has_swarm_proof = self.last_enemies_count >= 3 or self.last_danger_level == "CRITICAL"
            has_wall_proof = self.last_stuck or self.stuck_events > 0

            if not (has_damage_proof or has_swarm_proof or has_wall_proof):
                self.end_reason = "unknown"
            else:
                self.end_reason = "defeated"
        else:
            self.end_reason = reason

        return {
            "pokemon": self.active_pokemon,
            "duration": round(self.duration, 1),
            "max_level": self.current_level,
            "upgrades": self.upgrades_chosen,
            "damage_events": self.damage_events,
            "damage_incidents": self.damage_incidents,
            "trajectory_snapshots": self.trajectory_snapshots,
            "stuck_events": self.stuck_events,
            "dodge_events": self.dodge_events,
            "buffer_violations": self.buffer_violations,
            "encircled_events": self.encircled_events,
            "total_frames": self.total_frames,
            "min_distance_seen": round(self.min_distance_seen, 1),
            "last_stuck": self.last_stuck,
            "last_hp": self.last_hp,
            "last_hp_status": self.last_hp_status,
            "last_danger_level": self.last_danger_level,
            "last_enemies_count": self.last_enemies_count,
            "end_reason": self.end_reason
        }

