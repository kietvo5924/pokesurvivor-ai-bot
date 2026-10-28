import os
import json
import csv
import copy
from typing import Dict, Any, List, Optional
from pokesurvivor.config import DATA_DIR, DEFAULT_POLICY

class MemoryStore:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.history_file = os.path.join(self.data_dir, "run_history.jsonl")
        self.policy_file = os.path.join(self.data_dir, "policy.json")
        self.best_policy_file = os.path.join(self.data_dir, "best_policy.json")
        self.state_file = os.path.join(self.data_dir, "learning_state.json")
        self.metrics_file = os.path.join(self.data_dir, "metrics.csv")
        
        self._init_files()

    def _init_files(self):
        # Initialize policy if not exists
        if not os.path.exists(self.policy_file):
            self.save_policy(DEFAULT_POLICY)
        if not os.path.exists(self.best_policy_file):
            self.save_best_policy(DEFAULT_POLICY, 0.0)
            
        # Initialize learning state if not exists
        if not os.path.exists(self.state_file):
            initial_state = {
                "total_runs": 0,
                "current_policy_version": "1.0",
                "best_policy_version": "1.0",
                "best_survival_time": 0.0,
                "best_level": 1,
                "recent_scores": []
            }
            self.save_learning_state(initial_state)

        # Initialize metrics CSV if not exists
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_id", "timestamp", "duration_seconds", "max_level", 
                    "upgrades_count", "end_reason", "policy_version", 
                    "movement_pattern", "score", "notes"
                ])

    def load_policy(self) -> Dict[str, Any]:
        try:
            with open(self.policy_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return copy.deepcopy(DEFAULT_POLICY)

    def save_policy(self, policy: Dict[str, Any]):
        with open(self.policy_file, "w", encoding="utf-8") as f:
            json.dump(policy, f, indent=2, ensure_ascii=False)

    def load_best_policy(self) -> Dict[str, Any]:
        try:
            with open(self.best_policy_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return copy.deepcopy(DEFAULT_POLICY)

    def save_best_policy(self, policy: Dict[str, Any], best_score: float):
        best_data = copy.deepcopy(policy)
        best_data["best_score"] = best_score
        with open(self.best_policy_file, "w", encoding="utf-8") as f:
            json.dump(best_data, f, indent=2, ensure_ascii=False)

    def load_learning_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "total_runs": 0,
                "current_policy_version": "1.0",
                "best_policy_version": "1.0",
                "best_survival_time": 0.0,
                "best_level": 1,
                "recent_scores": []
            }

    def save_learning_state(self, state: Dict[str, Any]):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def append_run_history(self, run_data: Dict[str, Any]):
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_data, ensure_ascii=False) + "\n")

    def append_metrics(self, row: List[Any]):
        with open(self.metrics_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_all_runs(self) -> List[Dict[str, Any]]:
        runs = []
        if not os.path.exists(self.history_file):
            return runs
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return runs

