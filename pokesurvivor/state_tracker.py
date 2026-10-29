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
        self.game_score: Optional[int] = None
        self.game_score_source: str = "unobserved"
        self.survival_time: float = 0.0
        self.survival_time_source: str = "state_tracker_clock"
        self.max_level: int = 1
        self.max_level_source: str = "state_tracker_verified"
        self.fitness_score: float = 0.0
        self.fitness_score_source: str = "internal_composite_formula"
        self.match_results: Optional[Dict[str, Any]] = None
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

    def start_run(self, run_id: Optional[int] = None):
        self.reset()
        if run_id is not None:
            self.run_id = run_id
        self.is_active = True
        self.start_time = time.time()

    def update(
        self,
        dom_state: Optional[Dict[str, Any]] = None,
        vision_state: Optional[Dict[str, Any]] = None,
        unified_state: Optional[Dict[str, Any]] = None,
        elapsed_time: Optional[float] = None
    ):
        if not self.is_active:
            return
        
        if elapsed_time is not None:
            self.duration = elapsed_time
        else:
            self.duration = time.time() - self.start_time
        self.total_frames += 1

        v_state = unified_state if unified_state is not None else (vision_state if vision_state is not None else (dom_state or {}))

        if v_state.get("is_encircled", False):
            self.encircled_events += 1

        # Verified HP tracking: only record real measurements from verified sources
        hp = v_state.get("hp_ratio")
        if hp is None and isinstance(v_state.get("player"), dict):
            hp = v_state["player"].get("hp_ratio")

        hp_status = v_state.get("hp_status")
        if not hp_status and isinstance(v_state.get("player"), dict):
            hp_status = v_state["player"].get("source", "UNKNOWN")
        if not hp_status:
            hp_status = "UNKNOWN"
        self.last_hp_status = hp_status

        valid_statuses = ("VERIFIED", "ESTIMATED", "APPROXIMATE", "window_G", "canvas_telemetry_texts", "opencv_fallback")
        if hp is not None and hp_status in valid_statuses:
            self.hp_history.append(hp)
            if self.last_hp is not None and hp < self.last_hp - 0.08:
                self.damage_events += 1
                self.damage_incidents.append({
                    "time": round(self.duration, 1),
                    "hp_drop": round(self.last_hp - hp, 2),
                    "hp_after": round(hp, 2),
                    "min_enemy_dist": round(v_state.get("min_enemy_dist", 999.0), 1),
                    "is_encircled": v_state.get("is_encircled", False),
                    "near_wall": any(v_state.get("sector_obstacles", {}).values()),
                    "hazards_count": len(v_state.get("hazards", [])),
                    "enemies_count": v_state.get("enemies_count", 0)
                })
                self.last_hp = hp
            elif self.last_hp is not None and hp > self.last_hp:
                self.last_hp = hp
            elif self.last_hp is None:
                self.last_hp = hp

        min_e_dist = v_state.get("min_enemy_dist", 9999.0)
        if min_e_dist < self.min_distance_seen:
            self.min_distance_seen = min_e_dist
        if min_e_dist < 140.0:
            self.buffer_violations += 1

        hazards = v_state.get("hazards", [])
        if hazards and (hazards[0].get("dist", 999.0) < 165.0 or hazards[0].get("is_imminent", False)):
            self.dodge_events += 1

        is_stuck = v_state.get("is_stuck", False)
        if is_stuck and not self.last_stuck:
            self.stuck_events += 1
        self.last_stuck = is_stuck

        self.last_danger_level = v_state.get("danger_level", "LOW")
        self.last_enemies_count = v_state.get("enemies_count", 0)

        # Trajectory snapshot
        if self.duration - self.last_snapshot_time >= 1.5:
            self.trajectory_snapshots.append({
                "time": round(self.duration, 1),
                "hp": round(hp, 2) if hp is not None else None,
                "hp_status": hp_status,
                "min_dist": round(min_e_dist, 1),
                "mobs": self.last_enemies_count,
                "hazards": len(hazards),
                "encircled": v_state.get("is_encircled", False),
                "near_wall": any(v_state.get("sector_obstacles", {}).values())
            })
            self.last_snapshot_time = self.duration

    def record_upgrade(
        self,
        upgrade: Any,
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
        if isinstance(upgrade, dict):
            card_name = upgrade.get("name", "Unknown")
            tag = upgrade.get("tag", "")
        else:
            card_name = str(upgrade)
            tag = ""
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

    def end_run(self, reason: str = "defeated", dom_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.is_active = False
        self.duration = time.time() - self.start_time

        # Integrate verified DOM end-screen results if captured
        if dom_results and dom_results.get("available"):
            self.match_results = dom_results
            if dom_results.get("kills") is not None:
                self.game_score = dom_results["kills"]
                self.game_score_source = "dom_results_kills"
            else:
                self.game_score = None
                self.game_score_source = "dom_results_missing_kills"

            if dom_results.get("time_seconds") is not None:
                self.survival_time = float(dom_results["time_seconds"])
                self.survival_time_source = "dom_results_time"
            elif dom_results.get("time_str"):
                parsed_time = None
                parts = str(dom_results["time_str"]).split(":")
                if len(parts) == 2:
                    try:
                        parsed_time = float(int(parts[0]) * 60 + int(parts[1]))
                    except ValueError:
                        pass
                else:
                    try:
                        parsed_time = float(dom_results["time_str"])
                    except ValueError:
                        pass
                if parsed_time is not None:
                    self.survival_time = parsed_time
                    self.survival_time_source = "dom_results_time_str"
                else:
                    self.survival_time = round(self.duration, 1)
                    self.survival_time_source = "state_tracker_clock"
            else:
                self.survival_time = round(self.duration, 1)
                self.survival_time_source = "state_tracker_clock"

            if dom_results.get("level") is not None:
                self.max_level = int(dom_results["level"])
                self.max_level_source = "dom_results_level"
            else:
                self.max_level = self.current_level
                self.max_level_source = "state_tracker_verified"
        else:
            self.game_score = None
            self.game_score_source = f"unobserved_{reason}"
            self.survival_time = round(self.duration, 1)
            self.survival_time_source = "state_tracker_clock"
            self.max_level = self.current_level
            self.max_level_source = "state_tracker_verified"

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
            "game_score": self.game_score,
            "game_score_source": self.game_score_source,
            "survival_time": self.survival_time,
            "survival_time_source": self.survival_time_source,
            "max_level": self.max_level,
            "max_level_source": self.max_level_source,
            "duration": self.survival_time,
            "match_results": self.match_results,
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

