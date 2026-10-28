import time
from typing import Dict, Any, List, Optional

class StateTracker:
    def __init__(self):
        self.reset()

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
        self.last_hp = 1.0
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

        # Update HP and detect damage with smoothing to avoid false jitter
        hp = vision_state.get("hp_ratio", 1.0)
        self.hp_history.append(hp)

        min_e_dist = vision_state.get("min_enemy_dist", 9999.0)
        if min_e_dist < self.min_distance_seen:
            self.min_distance_seen = min_e_dist
        if min_e_dist < 140.0:
            self.buffer_violations += 1

        hazards = vision_state.get("hazards", [])
        if hazards and hazards[0].get("dist", 999.0) < 165.0:
            self.dodge_events += 1

        # Edge-triggered stuck detection: only increment on transition to stuck
        is_stuck = vision_state.get("is_stuck", False)
        if is_stuck and not self.last_stuck:
            self.stuck_events += 1
        self.last_stuck = is_stuck

        self.last_danger_level = vision_state.get("danger_level", "LOW")
        self.last_enemies_count = vision_state.get("enemies_count", 0)

        # Require a clear drop of > 10% to count as taking damage
        if hp < self.last_hp - 0.10:
            self.damage_events += 1
            # Record detailed incident telemetry for Machine Learning diagnostic analysis
            self.damage_incidents.append({
                "time": round(self.duration, 1),
                "hp_drop": round(self.last_hp - hp, 2),
                "hp_after": round(hp, 2),
                "min_enemy_dist": round(min_e_dist, 1),
                "is_encircled": vision_state.get("is_encircled", False),
                "near_wall": any(vision_state.get("sector_obstacles", {}).values()),
                "hazards_count": len(hazards),
                "enemies_count": self.last_enemies_count
            })
            self.last_hp = hp
        elif hp > self.last_hp:
            # Healed or leveling up restores HP
            self.last_hp = hp

        # Periodic trajectory snapshot every 1.5 seconds
        if self.duration - self.last_snapshot_time >= 1.5:
            self.trajectory_snapshots.append({
                "time": round(self.duration, 1),
                "hp": round(hp, 2),
                "min_dist": round(min_e_dist, 1),
                "mobs": self.last_enemies_count,
                "hazards": len(hazards),
                "encircled": vision_state.get("is_encircled", False),
                "near_wall": any(vision_state.get("sector_obstacles", {}).values())
            })
            self.last_snapshot_time = self.duration

    def record_upgrade(self, upgrade: Dict[str, Any]):
        card_name = upgrade.get("name", "Unknown")
        now_time = round(self.duration, 1)

        # Guard against double-recording the same modal selection
        if self.upgrades_chosen:
            last_entry = self.upgrades_chosen[-1]
            if last_entry.get("name") == card_name and abs(now_time - last_entry.get("time", 0.0)) < 1.5:
                return

        self.current_level += 1
        self.upgrades_chosen.append({
            "level": self.current_level,
            "name": card_name,
            "tag": upgrade.get("tag", ""),
            "time": now_time
        })

    def end_run(self, reason: str = "defeated") -> Dict[str, Any]:
        self.is_active = False
        self.end_reason = reason
        self.duration = time.time() - self.start_time

        return {
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
            "last_danger_level": self.last_danger_level,
            "last_enemies_count": self.last_enemies_count,
            "end_reason": self.end_reason
        }

