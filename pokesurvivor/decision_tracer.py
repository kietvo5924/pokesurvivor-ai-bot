import os
import json
import time
from typing import Dict, Any, List, Optional
from pokesurvivor.config import DATA_DIR

class DecisionTracer:
    """
    Decision & Telemetry Tracer for PokéSurvivor AI:
    Records real-time planner snapshots into data/decision_trace.jsonl:
    - timestamp & elapsed match time
    - player state (pos, hp_ratio, speed, invulnerable)
    - 16 candidate directional scores & risk breakdown
    - chosen direction & dispatched keys
    - delayed HP outcome verification (0.9s post-decision delta)
    - match context (level, kills/score)
    Supports offline replay and safety audit.
    """
    def __init__(self, trace_file: Optional[str] = None, max_pending: int = 50):
        if trace_file is None:
            trace_file = os.path.join(DATA_DIR, "decision_trace.jsonl")
        self.trace_file = trace_file
        self.max_pending = max_pending
        self.pending_traces: List[Dict[str, Any]] = []
        self.trace_counter: int = 0
        os.makedirs(os.path.dirname(os.path.abspath(self.trace_file)), exist_ok=True)

    def record_decision(
        self,
        elapsed_time: float,
        player_state: Dict[str, Any],
        candidate_scores: Dict[str, float],
        chosen_direction: str,
        chosen_keys: List[str],
        risk_breakdown: Optional[Dict[str, Any]] = None,
        level: int = 1,
        game_score: float = 0.0,
        now: Optional[float] = None
    ) -> int:
        """
        Records a decision event and stages it for delayed HP outcome resolution (~0.9s).
        Returns trace_id.
        """
        if now is None:
            now = time.time()
        self.trace_counter += 1
        trace_id = self.trace_counter

        entry = {
            "trace_id": trace_id,
            "timestamp": round(now, 3),
            "elapsed_time": round(elapsed_time, 2),
            "player": {
                "x": round(float(player_state.get("x", player_state.get("world_x", 0.0))), 1),
                "y": round(float(player_state.get("y", player_state.get("world_y", 0.0))), 1),
                "hp_ratio": player_state.get("hp_ratio"),
                "speed": player_state.get("speed", 100.0),
                "invulnerable": bool(player_state.get("invulnerable", False) or player_state.get("invincible", False))
            },
            "candidate_scores": {k: round(v, 2) for k, v in candidate_scores.items()},
            "chosen_direction": chosen_direction,
            "chosen_keys": list(chosen_keys),
            "risk_breakdown": risk_breakdown or {},
            "level": level,
            "game_score": game_score,
            "outcome_hp_delta": 0.0,
            "verified_outcome": False
        }

        self.pending_traces.append(entry)
        if len(self.pending_traces) > self.max_pending:
            self._flush_oldest()

        return trace_id

    def update_outcomes(self, current_hp_ratio: Optional[float], now: Optional[float] = None, delay_window: float = 0.9):
        """
        Resolves delayed HP outcome delta for pending traces older than delay_window (~0.9s).
        """
        if now is None:
            now = time.time()

        flushed = []
        remaining = []

        for entry in self.pending_traces:
            entry_time = entry["timestamp"]
            if (now - entry_time) >= delay_window:
                init_hp = entry["player"]["hp_ratio"]
                if init_hp is not None and current_hp_ratio is not None:
                    entry["outcome_hp_delta"] = round(current_hp_ratio - init_hp, 4)
                    entry["verified_outcome"] = True
                flushed.append(entry)
            else:
                remaining.append(entry)

        self.pending_traces = remaining
        if flushed:
            self._write_entries(flushed)

    def _flush_oldest(self):
        if self.pending_traces:
            oldest = self.pending_traces.pop(0)
            self._write_entries([oldest])

    def flush_all(self):
        """Flushes all pending traces to file at run termination."""
        if self.pending_traces:
            self._write_entries(self.pending_traces)
            self.pending_traces.clear()

    def _write_entries(self, entries: List[Dict[str, Any]]):
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def load_traces(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads recorded decision traces from file."""
        if not os.path.exists(self.trace_file):
            return []
        traces = []
        try:
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        traces.append(json.loads(line))
                        if limit and len(traces) >= limit:
                            break
        except Exception:
            pass
        return traces

