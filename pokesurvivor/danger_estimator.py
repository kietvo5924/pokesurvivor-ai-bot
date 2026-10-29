from typing import Dict, Any, List, Tuple, Optional
import math

# Direction mapping to keyboard keys (8 directions)
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

# 16-Direction Expanded Compass Mapping
# 0 = East, 90 = South, 180 = West, 270 = North
DIR16_ANGLES: Dict[str, float] = {
    "E": 0.0,
    "ESE": 22.5,
    "SE": 45.0,
    "SSE": 67.5,
    "S": 90.0,
    "SSW": 112.5,
    "SW": 135.0,
    "WSW": 157.5,
    "W": 180.0,
    "WNW": 202.5,
    "NW": 225.0,
    "NNW": 247.5,
    "N": 270.0,
    "NNE": 292.5,
    "NE": 315.0,
    "ENE": 337.5
}

DIR16_NAMES: List[str] = [
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
    "N", "NNE", "NE", "ENE"
]

DIR16_KEYS: Dict[str, List[str]] = {
    "E": ["KeyD"],
    "ESE": ["KeyD"],
    "SE": ["KeyD", "KeyS"],
    "SSE": ["KeyS"],
    "S": ["KeyS"],
    "SSW": ["KeyS"],
    "SW": ["KeyS", "KeyA"],
    "WSW": ["KeyA"],
    "W": ["KeyA"],
    "WNW": ["KeyA"],
    "NW": ["KeyW", "KeyA"],
    "NNW": ["KeyW"],
    "N": ["KeyW"],
    "NNE": ["KeyW"],
    "NE": ["KeyW", "KeyD"],
    "ENE": ["KeyD"]
}

OPPOSITE_DIR16: Dict[str, str] = {
    "E": "W", "ESE": "WNW", "SE": "NW", "SSE": "NNW",
    "S": "N", "SSW": "NNE", "SW": "NE", "WSW": "ENE",
    "W": "E", "WNW": "ESE", "NW": "SE", "NNW": "SSE",
    "N": "S", "NNE": "SSW", "NE": "SW", "ENE": "WSW"
}

def angle_to_16_sector(angle_deg: float) -> str:
    """Maps continuous angle in degrees [0, 360) to nearest 16-compass sector."""
    normalized = (angle_deg % 360.0 + 360.0) % 360.0
    idx = int(((normalized + 11.25) % 360.0) // 22.5)
    return DIR16_NAMES[idx]

def angle_diff_deg(a1: float, a2: float) -> float:
    """Minimal angular difference between two angles in degrees (0 to 180)."""
    diff = (a1 - a2 + 180.0) % 360.0 - 180.0
    return abs(diff)

class DangerEstimator:
    def __init__(self, evasion_weight: float = 0.6, exp_weight: float = 0.4):
        self.evasion_weight = evasion_weight
        self.exp_weight = exp_weight

    def evaluate_directions(
        self,
        sector_dangers: Dict[str, float],
        gem_sector: Optional[str],
        current_dir: Optional[str],
        hp_ratio: float = 1.0,
        candidate_mode: int = 16
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Evaluates candidate directions (8 or 16) and returns sorted list of (direction, score, keys).
        Higher score is better (safer + higher utility).
        """
        scored = []
        keys_map = DIR16_KEYS if candidate_mode == 16 else DIR_KEYS
        opposite_map = OPPOSITE_DIR16 if candidate_mode == 16 else OPPOSITE_DIR

        # If HP is low, prioritize evasion even more
        effective_evasion = self.evasion_weight if hp_ratio > 0.4 else 0.85
        effective_exp = self.exp_weight if hp_ratio > 0.4 else 0.15

        for d, keys in keys_map.items():
            # If 16 sectors are evaluated against an 8-sector danger map, interpolate/fallback
            danger = sector_dangers.get(d)
            if danger is None:
                # Fallback to parent 8-sector if available
                deg = DIR16_ANGLES.get(d, 0.0)
                # Map to closest 8-sector
                idx8 = int(((deg + 22.5) % 360.0) // 45.0)
                names8 = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
                parent_8 = names8[idx8]
                danger = sector_dangers.get(parent_8, 0.0)
            
            # Base safety: lower danger -> higher safety score
            safety_score = 100.0 / (1.0 + danger * 10.0)
            
            # Bonus if moving towards gem (if safe)
            exp_bonus = 0.0
            if gem_sector:
                if gem_sector == d:
                    exp_bonus = 25.0 if danger < 1.0 else 5.0
                elif candidate_mode == 16:
                    gem_ang = DIR16_ANGLES.get(gem_sector)
                    d_ang = DIR16_ANGLES.get(d)
                    if gem_ang is not None and d_ang is not None and angle_diff_deg(gem_ang, d_ang) <= 22.5:
                        exp_bonus = 18.0 if danger < 1.0 else 3.0
            
            # Momentum bonus: penalize instant 180 degree flip unless danger is critical
            continuity_bonus = 0.0
            if current_dir:
                if current_dir == d:
                    continuity_bonus = 8.0  # keep running smoothly
                elif opposite_map.get(current_dir) == d:
                    continuity_bonus = -5.0 # avoid rapid twitching back and forth
                elif candidate_mode == 16:
                    curr_ang = DIR16_ANGLES.get(current_dir)
                    d_ang = DIR16_ANGLES.get(d)
                    if curr_ang is not None and d_ang is not None:
                        diff = angle_diff_deg(curr_ang, d_ang)
                        if diff <= 25.0:
                            continuity_bonus = 6.0

            total_score = (safety_score * effective_evasion) + (exp_bonus * effective_exp) + continuity_bonus
            scored.append((d, total_score, keys))

        # Sort descending: best direction first
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
