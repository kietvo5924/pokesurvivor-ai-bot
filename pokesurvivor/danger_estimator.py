from typing import Dict, Any, List, Tuple, Optional
import math

# Direction mapping to keyboard keys
DIR_KEYS = {
    "N": ["KeyW"],
    "NE": ["KeyW", "KeyD"],
    "E": ["KeyD"],
    "SE": ["KeyS", "KeyD"],
    "S": ["KeyS"],
    "SW": ["KeyS", "KeyA"],
    "W": ["KeyA"],
    "NW": ["KeyW", "KeyA"]
}

OPPOSITE_DIR = {
    "N": "S", "S": "N", "E": "W", "W": "E",
    "NE": "SW", "SW": "NE", "NW": "SE", "SE": "NW"
}

class DangerEstimator:
    def __init__(self, evasion_weight: float = 0.6, exp_weight: float = 0.4):
        self.evasion_weight = evasion_weight
        self.exp_weight = exp_weight

    def evaluate_directions(
        self,
        sector_dangers: Dict[str, float],
        gem_sector: Optional[str],
        current_dir: Optional[str],
        hp_ratio: float = 1.0
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Evaluates 8 directions and returns sorted list of (direction, score, keys).
        Higher score is better (safer + higher utility).
        """
        scored = []

        # If HP is low, prioritize evasion even more
        effective_evasion = self.evasion_weight if hp_ratio > 0.4 else 0.85
        effective_exp = self.exp_weight if hp_ratio > 0.4 else 0.15

        for d, keys in DIR_KEYS.items():
            danger = sector_dangers.get(d, 0.0)
            
            # Base safety: lower danger -> higher safety score
            # Score decreases sharply as danger rises
            safety_score = 100.0 / (1.0 + danger * 10.0)
            
            # Bonus if moving towards gem (if safe)
            exp_bonus = 0.0
            if gem_sector == d:
                exp_bonus = 25.0 if danger < 1.0 else 5.0
            
            # Momentum bonus: penalize instant 180 degree flip unless danger is critical
            continuity_bonus = 0.0
            if current_dir:
                if current_dir == d:
                    continuity_bonus = 8.0  # keep running smoothly
                elif OPPOSITE_DIR.get(current_dir) == d:
                    continuity_bonus = -5.0 # avoid rapid twitching back and forth

            total_score = (safety_score * effective_evasion) + (exp_bonus * effective_exp) + continuity_bonus
            scored.append((d, total_score, keys))

        # Sort descending: best direction first
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

