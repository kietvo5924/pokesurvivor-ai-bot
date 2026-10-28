"""
pokesurvivor.coin_tracker
-------------------------
PokéCoin farming & match economy tracker.
Ported and adapted from PokeSurvivorTool (core/farmMoney.js · MIT License).

Tracks genuine currency metrics:
- Coins earned per match
- Total lifetime coins accumulated
- Coin farming rate (Coins / Hour)
- Correlated survival time, defeated kills, and bosses downed.

Keeps economy tracking strictly separated from internal policy fitness scores.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from pokesurvivor.config import DATA_DIR

COIN_HISTORY_FILE = os.path.join(DATA_DIR, "coin_history.json")

class CoinTracker:
    def __init__(self, filepath: str = COIN_HISTORY_FILE):
        self.filepath = filepath
        self.data: Dict[str, Any] = {
            "since": time.time(),
            "total_coins": 0,
            "runs": []
        }
        self.load()

    def load(self):
        """Loads persistent coin history from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def save(self):
        """Saves coin history to disk."""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def record_run(
        self,
        coins: Optional[int],
        kills: Optional[int],
        survival_time: float,
        level: Optional[int] = None,
        bosses: Optional[int] = None,
        policy_version: str = "unknown"
    ):
        """Records economy metrics from a completed match."""
        if coins is None or coins < 0:
            coins = 0

        run_entry = {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coins": coins,
            "kills": kills,
            "survival_time": round(survival_time, 1),
            "level": level,
            "bosses": bosses or 0,
            "policy_version": policy_version
        }

        self.data["runs"].append(run_entry)
        self.data["total_coins"] = sum(r.get("coins", 0) for r in self.data["runs"])
        self.save()

    def get_summary(self) -> Dict[str, Any]:
        """Calculates total coins, runs, and coins per hour."""
        runs = self.data.get("runs", [])
        total_coins = self.data.get("total_coins", 0)
        since = self.data.get("since", time.time())
        hours = max(0.01, (time.time() - since) / 3600.0)
        coins_per_hour = round(total_coins / hours, 1)

        last_run = runs[-1] if runs else None

        return {
            "total_runs": len(runs),
            "total_coins": total_coins,
            "coins_per_hour": coins_per_hour,
            "last_run_coins": last_run.get("coins", 0) if last_run else 0,
            "last_run_time": last_run.get("survival_time", 0.0) if last_run else 0.0
        }

    def format_status_line(self) -> str:
        """Returns a concise summary string for logging."""
        summary = self.get_summary()
        return (
            f"💰 Farm: {summary['total_coins']} xu | "
            f"{summary['total_runs']} trận | "
            f"{summary['coins_per_hour']} xu/h"
        )

