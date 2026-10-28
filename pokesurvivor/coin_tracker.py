"""
pokesurvivor.coin_tracker
-------------------------
PokéCoin farming & match economy tracker.
Ported and adapted from PokeSurvivorTool (core/farmMoney.js · MIT License).

Tracks genuine currency metrics:
- Coins earned per match (differentiating observed 0 from unobserved None)
- Total lifetime coins accumulated across observed runs
- In-game active farming rate (coins / in-game play hour)
- Differentiates active gameplay time from idle / between-match time
- Correlated survival time, defeated kills, and bosses downed.

Keeps economy tracking strictly separated from internal policy fitness scores.
"""

import os
import json
import time
import shutil
from typing import Dict, Any, List, Optional
from pokesurvivor.config import DATA_DIR

COIN_HISTORY_FILE = os.path.join(DATA_DIR, "coin_history.json")

class CoinTracker:
    def __init__(self, filepath: str = COIN_HISTORY_FILE):
        self.filepath = filepath
        self._backup_existing()
        self.data: Dict[str, Any] = {
            "since": time.time(),
            "total_coins": 0,
            "runs": []
        }
        self.load()

    def _backup_existing(self):
        """Creates a safety backup (.bak) before modifying schema or writing data."""
        if os.path.exists(self.filepath):
            bak_path = self.filepath + ".bak"
            if not os.path.exists(bak_path):
                try:
                    shutil.copy2(self.filepath, bak_path)
                except Exception:
                    pass

    def load(self):
        """Loads persistent coin history from disk preserving backward compatibility."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                # Recalculate total_coins strictly summing observed runs
                self.data["total_coins"] = sum(
                    r.get("coins", 0) for r in self.data.get("runs", []) if r.get("coins") is not None
                )
            except Exception:
                pass

    def save(self):
        """Saves coin history to disk safely."""
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
        """
        Records economy metrics from a completed match.
        CRITICAL: Never converts coins=None to 0.
        Differentiates genuinely earning 0 coins from unobserved match results.
        """
        coins_observed = (coins is not None and coins >= 0)
        coin_val = int(coins) if coins_observed else None

        run_entry = {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coins": coin_val,
            "coins_observed": coins_observed,
            "kills": kills,
            "survival_time": round(survival_time, 1),
            "level": level,
            "bosses": bosses or 0,
            "policy_version": policy_version
        }

        self.data["runs"].append(run_entry)
        # Sum only observed runs
        self.data["total_coins"] = sum(
            r.get("coins", 0) for r in self.data["runs"] if r.get("coins") is not None
        )
        self.save()

    def get_summary(self) -> Dict[str, Any]:
        """
        Calculates total coins, run breakdown, and coins per hour.
        
        Definition of coins_per_hour (In-Game Active Rate):
        Total coins earned divided by total match play time (in hours).
        This distinguishes active in-game farming speed from idle wall-clock time
        (e.g. overnight gaps or pause menus).
        """
        runs = self.data.get("runs", [])
        observed_runs = [r for r in runs if r.get("coins") is not None]
        unobserved_runs = len(runs) - len(observed_runs)
        zero_coin_runs = sum(1 for r in observed_runs if r.get("coins") == 0)

        total_coins = sum(r.get("coins", 0) for r in observed_runs)
        total_play_time = sum(r.get("survival_time", 0.0) for r in observed_runs)

        play_hours = total_play_time / 3600.0
        coins_per_play_hour = round(total_coins / max(1e-4, play_hours), 1) if play_hours > 0 else 0.0

        last_run = runs[-1] if runs else None

        return {
            "total_runs": len(runs),
            "observed_runs": len(observed_runs),
            "unobserved_runs": unobserved_runs,
            "zero_coin_runs": zero_coin_runs,
            "total_coins": total_coins,
            "total_play_time_seconds": round(total_play_time, 1),
            "coins_per_hour": coins_per_play_hour,
            "coins_per_play_hour": coins_per_play_hour,
            "rate_definition": "in_game_play_rate (total_coins / total_play_hours)",
            "last_run_coins": last_run.get("coins") if last_run else None,
            "last_run_time": last_run.get("survival_time", 0.0) if last_run else 0.0
        }

    def format_status_line(self) -> str:
        """Returns a concise summary string for logging."""
        summary = self.get_summary()
        unobs_info = f" (+{summary['unobserved_runs']} chưa rõ)" if summary["unobserved_runs"] > 0 else ""
        return (
            f"💰 Farm: {summary['total_coins']} xu | "
            f"{summary['observed_runs']}{unobs_info} trận | "
            f"{summary['coins_per_hour']} xu/h (thời gian chơi)"
        )
