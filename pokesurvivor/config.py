import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

GAME_URL = "https://pokesurvivor.doofcraft.com/"
STORAGE_STATE_PATH = os.path.join(BASE_DIR, "auth_state.json")

# Screen dimensions
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720

# Hotkeys
HOTKEY_TOGGLE = "f8"
HOTKEY_EMERGENCY_STOP = "f9"

# Default survival duration: unconstrained (3600s = 60 mins) to farm late-game gold/tickets/EXP
DEFAULT_MAX_DURATION = 3600.0

# Researched Meta Strategy for PokéSurvivor (Doofcraft Edition)
# 1. Scorbunny starts with Ember -> Rush Ember to Level 5.
# 2. Pair with Power card -> Unlocks Golden Evolution: Fire Blast!
# 3. Enter Purple Rift -> Defeat Boss to get Item + Bond 2 moves together!
# 4. Target Bond Loadout: 1 Buff Move + 3 Attack Moves (fired simultaneously!)
# 5. Dying in the Rift boss arena does NOT end the run (ejected back to main map with no penalty).
DEFAULT_POLICY = {
    "version": "3.5",
    "description": "Meta Fire Blast & Rift Boss Hunting Loadout (1 Buff + 3 Attack)",
    "movement": {
        "pattern": "kiting",
        "step_duration": 0.22,
        "evasion_weight": 0.88,
        "exp_weight": 0.42,
        "danger_threshold": 160.0,
        "enter_rift_priority": True
    },
    "upgrade": {
        "priority_moves": ["Ember", "Fire Blast", "Will-O-Wisp", "Fire Spin", "Take Down", "Tackle", "Flamethrower"],
        "buff_moves": ["Swords Dance", "Nasty Plot", "Harden", "Agility", "Iron Defense", "Growth"],
        "priority_stats": ["Power", "Sprint", "Vigor", "Reflexes", "Synthesis"],
        "rush_primary_move": "Ember",
        "primary_move_max_level": 5,
        "target_bonded_moves": 4, # 4 bonded move groups
        "target_buff_count": 1,   # 1 buff move
        "target_attack_count": 3, # 3 attack moves
        "prefer_evolution": True,
        "prefer_new_skills": False
    },
    "items": {
        "priority": [
            "leftovers", "shell-bell", "focus-band", # S-Tier Survival
            "charcoal", "choice-specs", "choice-band", "muscle-band", # S-Tier Damage
            "amulet-coin", "exp-share", "lucky-egg" # Economy / Farming
        ]
    }
}
