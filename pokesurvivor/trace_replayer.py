import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

class TraceReplayer:
    """
    Offline Replayer & Decision Validator for PokéSurvivor AI:
    - Replays recorded decision traces from data/decision_trace.jsonl
    - Re-evaluates each state with a candidate policy or planner
    - Compares chosen direction against recomputed optimal direction
    - Audits HP outcomes (post-decision damage incidents)
    - Benchmarks offline decision safety and avoidance metrics
    """
    def __init__(self, trace_file: Optional[str] = None):
        self.trace_file = trace_file

    def replay_traces(
        self,
        planner: Any,
        traces: Optional[List[Dict[str, Any]]] = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Replays recorded traces through the planner and computes fidelity and safety metrics.
        """
        if traces is None:
            if not self.trace_file or not os.path.exists(self.trace_file):
                return {"error": "Trace file not found", "total_traces": 0}
            traces = []
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        traces.append(json.loads(line))
                        if max_samples and len(traces) >= max_samples:
                            break

        if not traces:
            return {"total_traces": 0, "message": "No traces available for replay"}

        total = 0
        matching_decisions = 0
        damage_incidents = 0
        damage_avoided_count = 0
        total_hp_delta = 0.0

        for t in traces:
            total += 1
            rec_dir = t.get("chosen_direction")
            hp_delta = t.get("outcome_hp_delta", 0.0)
            total_hp_delta += hp_delta

            if hp_delta < -0.01:
                damage_incidents += 1
            else:
                damage_avoided_count += 1

            # Reconstruct simplified vision state for policy evaluation
            player = t.get("player", {})
            reconstructed_state = {
                "player": player,
                "player_pos": (player.get("x", 640.0), player.get("y", 360.0)),
                "enemies": [],
                "hazards": [],
                "gems": [],
                "sector_dangers": {},
                "sector_obstacles": {},
                "sector_enemy_counts": {}
            }

            # If planner is a MovementPolicy or PredictiveSurvivalPlanner
            try:
                if hasattr(planner, "select_movement"):
                    recomp_dir, _, _ = planner.select_movement(
                        reconstructed_state,
                        current_level=t.get("level", 1),
                        elapsed_time=t.get("elapsed_time", 0.0)
                    )
                    if recomp_dir == rec_dir:
                        matching_decisions += 1
            except Exception:
                pass

        match_rate = (matching_decisions / total) * 100.0 if total > 0 else 0.0
        safe_outcome_rate = (damage_avoided_count / total) * 100.0 if total > 0 else 0.0

        return {
            "total_traces": total,
            "matching_decisions": matching_decisions,
            "fidelity_rate_pct": round(match_rate, 2),
            "damage_incidents": damage_incidents,
            "damage_avoided_count": damage_avoided_count,
            "safe_outcome_rate_pct": round(safe_outcome_rate, 2),
            "net_hp_delta": round(total_hp_delta, 4)
        }

