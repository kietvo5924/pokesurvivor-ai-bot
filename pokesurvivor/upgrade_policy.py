import os
import json
from typing import Dict, Any, List, Optional, Tuple

class UpgradePolicy:
    """
    PokéSurvivor Meta Upgrade & Fusion Policy (v5.0):
    - Contextual to Active Pokémon (STAB & Element Synergy: Tapu Fini Water/Fairy vs Scorbunny Fire).
    - Eliminates bias toward off-element moves (e.g., Flamethrower on Tapu Fini).
    - Rushes primary starter attack to Level 5.
    - Secures evolution-enabling stat passives based on 87-move evolution database.
    - Selects Golden Evolution cards with highest priority.
    - Rift Boss Reward: Bonds 2 moves together (max 4 loadout: 1 Buff + 3 Attack moves).
    - Held Item: Manages single held item slot (swapping for top-tier survival/damage items).
    - Chests: Prioritizes Rare Candy (+1 level) > Meta Upgrades > Gacha Tickets.
    - Reroll: Triggers reroll when no card reaches the viable threshold (< 500 score).
    """
    def __init__(self, policy_config: Dict[str, Any]):
        self.active_pokemon: str = "Unknown"
        self.dex: Optional[int] = None
        self.rush_primary_move: str = ""
        self.acquired_moves: Dict[str, int] = {}
        self.acquired_stats: List[str] = []
        self.held_item: Optional[str] = None
        self.bonded_groups: List[str] = []
        self.evolutions_db: Dict[str, Any] = self._load_evolutions_db()
        self.update_config(policy_config)
        self.reset_match()

    def _load_evolutions_db(self) -> Dict[str, Any]:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolutions_db.json")
        try:
            if os.path.exists(db_path):
                with open(db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def get_pokemon_types(self) -> List[str]:
        """Returns elemental types for the active Pokemon."""
        name = self.active_pokemon.lower()
        if "fini" in name:
            return ["water", "fairy"]
        elif "scorbunny" in name or "raboot" in name or "cinderace" in name:
            return ["fire"]
        elif "sobble" in name or "drizzile" in name or "inteleon" in name or "squirtle" in name or "blastoise" in name:
            return ["water"]
        elif "bulbasaur" in name or "ivysaur" in name or "venusaur" in name:
            return ["grass", "poison"]
        elif "pikachu" in name or "raichu" in name:
            return ["electric"]
        elif "gastly" in name or "haunter" in name or "gengar" in name:
            return ["ghost", "poison"]
        return []

    def set_active_pokemon(self, name: str, starting_move: str = "", dex: Optional[int] = None):
        """Dynamically adapts upgrade prioritization to the active Pokemon and its starter move."""
        if name and name != "Unknown":
            self.active_pokemon = name
        if dex is not None and dex > 0:
            self.dex = dex
        if starting_move:
            self.rush_primary_move = starting_move.lower().strip()
            self.acquired_moves[self.rush_primary_move] = 1
        elif "fini" in self.active_pokemon.lower():
            self.rush_primary_move = "water gun"
            self.acquired_moves["water gun"] = 1
        elif "scorbunny" in self.active_pokemon.lower():
            self.rush_primary_move = "ember"
            self.acquired_moves["ember"] = 1

    def reset_match(self):
        self.acquired_moves = {}
        if self.rush_primary_move:
            self.acquired_moves[self.rush_primary_move] = 1
        self.acquired_stats = []
        self.held_item = None
        self.bonded_groups = []

    def update_config(self, policy_config: Dict[str, Any]):
        self.config = policy_config.get("upgrade", {})
        self.priority_moves = [m.lower() for m in self.config.get("priority_moves", [
            "Water Gun", "Scald", "Ice Beam", "Blizzard", "Hydro Pump", "Hydro Cannon",
            "Shadow Ball", "Phantom Force", "Dazzling Gleam", "Grass Knot", "Toxic",
            "Ember", "Fire Blast", "Flamethrower", "Overheat", "Fire Spin", "Sacred Fire",
            "Tackle", "Take Down"
        ])]
        self.buff_moves = [b.lower() for b in self.config.get("buff_moves", ["Aqua Ring", "Swords Dance", "Nasty Plot", "Harden", "Agility", "Iron Defense", "Growth"])]
        self.priority_stats = [s.lower() for s in self.config.get("priority_stats", ["Power", "Sprint", "Vigor", "Reflexes", "Synthesis"])]
        self.rush_primary_move = self.config.get("rush_primary_move", "Ember").lower()
        self.primary_move_max_level = self.config.get("primary_move_max_level", 5)
        self.skill_ratings: Dict[str, float] = self.config.get("skill_ratings", {})

        self.item_scores: Dict[str, float] = {
            "shell bell": 110.0,
            "shell-bell": 110.0,
            "leftovers": 100.0,
            "quick claw": 90.0,
            "quick-claw": 90.0,
            "focus sash": 80.0,
            "focus-sash": 80.0,
            "focus band": 75.0,
            "focus-band": 75.0,
            "assault vest": 70.0,
            "assault-vest": 70.0,
            "rocky helmet": 60.0,
            "rocky-helmet": 60.0,
            "lucky egg": 50.0,
            "lucky-egg": 50.0,
            "amulet coin": 40.0,
            "amulet-coin": 40.0,
            "choice scarf": 30.0,
            "choice-scarf": 30.0,
            "charcoal": 20.0,
            "mystic water": 20.0,
            "life orb": 10.0,
            "choice band": -100.0,
            "choice-band": -100.0
        }

    def choose_card(
        self,
        choices: List[Dict[str, Any]],
        current_hp_ratio: Optional[float] = None,
        current_skills: List[str] = None,
        modal_kind: str = "level-up"
    ) -> Optional[Dict[str, Any]]:
        if not choices:
            return None

        if modal_kind == "item" or any("item" in c.get("name", "").lower() for c in choices):
            return self._choose_item(choices)

        if modal_kind == "bond" or any("bond" in c.get("name", "").lower() or "bond" in c.get("tag", "").lower() for c in choices):
            return self._choose_bond(choices, current_skills)

        if modal_kind == "chest":
            return self._choose_chest(choices)

        return self._choose_standard_upgrade(choices, current_hp_ratio)

    def _choose_item(self, choices: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(choices) == 1:
            return choices[0]

        keep_card = choices[0]
        switch_card = choices[1]
        new_name = switch_card.get("name", "").lower().strip()
        held_name = (self.held_item or keep_card.get("name", "")).lower().strip()

        def get_item_score(item_name: str) -> float:
            clean = item_name.replace("-", " ").strip()
            for key, val in self.item_scores.items():
                if key.replace("-", " ") in clean:
                    return val
            return 20.0

        new_score = get_item_score(new_name)
        held_score = get_item_score(held_name) if self.held_item else -999.0

        if "choice band" in held_name:
            held_score = -999.0

        if new_score > held_score:
            self.held_item = new_name
            switch_card["selection_score"] = 500.0 + new_score
            switch_card["selection_reason"] = f"EQUIP_SUPERIOR_ITEM({new_name}, score={new_score:.0f})"
            return switch_card
        else:
            keep_card["selection_score"] = 400.0 + held_score
            keep_card["selection_reason"] = f"KEEP_SUPERIOR_ITEM({held_name}, score={held_score:.0f})"
            return keep_card

    def _choose_bond(self, choices: List[Dict[str, Any]], current_skills: List[str] = None) -> Dict[str, Any]:
        scored_bonds = []
        poke_types = self.get_pokemon_types()
        is_fire_poke = "fire" in poke_types
        is_water_poke = "water" in poke_types

        for card in choices:
            name = card.get("name", "").strip().lower()
            score = 100.0
            reasons = []

            is_buff = any(b in name for b in self.buff_moves)
            is_pri_attack = any(a in name for a in self.priority_moves)

            if len(self.bonded_groups) == 0:
                if is_water_poke and ("water" in name or "ice" in name):
                    score += 350.0
                    reasons.append("BOND_CORE_WATER_ATTACK")
                elif is_fire_poke and ("fire" in name or "ember" in name):
                    score += 350.0
                    reasons.append("BOND_CORE_FIRE_ATTACK")
                elif is_pri_attack:
                    score += 280.0
                    reasons.append("BOND_CORE_ATTACK")
            elif len(self.bonded_groups) == 1 and is_buff:
                score += 350.0
                reasons.append("BOND_BUFF_SYNERGY")
            elif is_pri_attack:
                score += 200.0
                reasons.append("BOND_ATTACK_COMPLEMENT")

            scored_bonds.append((card, score, ", ".join(reasons) or "bond_option"))

        scored_bonds.sort(key=lambda x: x[1], reverse=True)
        best_card, best_score, best_reason = scored_bonds[0]
        best_card["selection_score"] = best_score
        best_card["selection_reason"] = best_reason
        self.bonded_groups.append(best_card.get("name", "Unknown"))
        return best_card

    def _choose_chest(self, choices: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = []
        for card in choices:
            name = card.get("name", "").lower().strip()
            desc = (card.get("desc") or card.get("description") or "").lower().strip()
            score = 100.0
            reason = "chest_reward"

            owned_match = None
            for m in self.acquired_moves.keys():
                if m in name or m in desc:
                    owned_match = m
                    break

            if owned_match:
                score = 900.0
                reason = f"UPGRADE_BUILD_MOVE_{owned_match.upper()}"
            elif "candy" in name or "rare" in name:
                score = 700.0
                reason = "RARE_CANDY_INSTANT_LEVEL"
            elif "evolution" in name or "gold" in name:
                score = 500.0
                reason = "EVOLUTION_UPGRADE"
            elif "ticket" in name:
                score = 150.0
                reason = "GACHA_TICKETS_REWARD"
            else:
                score = 250.0
                reason = "CHEST_UPGRADE"

            scored.append((card, score, reason))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_card, best_score, best_reason = scored[0]
        best_card["selection_score"] = best_score
        best_card["selection_reason"] = best_reason
        return best_card

    def _choose_standard_upgrade(self, choices: List[Dict[str, Any]], current_hp_ratio: Optional[float] = None) -> Dict[str, Any]:
        scored_cards: List[Tuple[Dict[str, Any], float, str]] = []

        is_damaged = (current_hp_ratio is not None and current_hp_ratio < 0.50)
        is_unknown_hp = (current_hp_ratio is None)

        poke_types = self.get_pokemon_types()
        is_fire_poke = "fire" in poke_types
        is_water_poke = "water" in poke_types or "fini" in self.active_pokemon.lower()

        for card in choices:
            score = 10.0
            reasons = []
            name = card.get("name", "").strip().lower()
            tag = card.get("tag", "").strip().lower()
            desc = (card.get("desc") or card.get("description") or "").strip().lower()

            # 1. GOLDEN EVOLUTION CARDS
            is_evo = (
                "evolution" in tag or "evolv" in name or "gold" in tag or "evo" in tag or "★" in name or "⭐" in name or
                "scald" in name or "blizzard" in name or "hydro cannon" in name or
                "phantom force" in name or "frenzy plant" in name or "poison gas" in name or
                "fire blast" in name or "overheat" in name or "sacred fire" in name or
                "take down" in name or "inferno" in name or "extreme speed" in name or "surf" in name or
                "water gun+" in name or "water gun →" in name
            )
            if is_evo:
                score += 2000.0
                reasons.append("GOLDEN_EVOLUTION_MAX_PRIORITY")

            # 2. HP RECOVERY & HEALING SELECTION
            elif "sitrus berry" in name or "berry" in name or "heal" in desc or "recovery" in desc:
                if is_damaged:
                    score += 1500.0
                    reasons.append("EMERGENCY_HEAL_LOW_HP")
                elif is_unknown_hp:
                    score += 850.0
                    reasons.append("DEFENSIVE_HEAL_UNKNOWN_HP")
                else:
                    score += 10.0
                    reasons.append("preventative_heal_demoted")

            # 3. CORE BUFF MOVES (Aqua Ring provides up to 6.4 HP/s passive heal!)
            elif "aqua ring" in name:
                score += 1250.0
                reasons.append("AQUA_RING_PASSIVE_REGEN_CORE_BUFF")

            # 4. RUSH ACTIVE STARTER ATTACK TO LEVEL 5
            elif self.rush_primary_move and self.rush_primary_move in name:
                cur_lvl = self.acquired_moves.get(self.rush_primary_move, 1)
                if cur_lvl < 5:
                    score += 1050.0
                    reasons.append(f"RUSH_{self.rush_primary_move.upper()}_TO_LV5(cur={cur_lvl})")
                else:
                    score += 80.0
                    reasons.append(f"{self.rush_primary_move}_already_max_lv5")

            elif "water gun" in name:
                wg_lvl = self.acquired_moves.get("water gun", 1)
                if wg_lvl < 5:
                    bonus = 1050.0 if is_water_poke else 450.0
                    score += bonus
                    reasons.append(f"RUSH_WATER_GUN_TO_LV5(cur={wg_lvl})")
                else:
                    score += 80.0
                    reasons.append("water_gun_already_max_lv5")

            elif "ember" in name:
                ember_lvl = self.acquired_moves.get("ember", 1)
                if ember_lvl < 5:
                    bonus = 1050.0 if is_fire_poke else 150.0
                    score += bonus
                    reasons.append(f"RUSH_EMBER_TO_LV5(cur={ember_lvl})")
                else:
                    score += 80.0
                    reasons.append("ember_already_max_lv5")

            # 5. DYNAMIC EVOLUTION KEY: Check if required passive for owned moves
            elif self._is_required_passive_for_owned_moves(name):
                matched_move, req_p = self._get_required_passive_context(name)
                score += 950.0
                reasons.append(f"{req_p.upper()}_REQUIRED_TO_EVOLVE_{matched_move.upper()}")

            # 6. HIGH-TIER ATTACK MOVES CONTEXTUAL TO ACTIVE POKEMON
            elif "ice beam" in name:
                ib_lvl = self.acquired_moves.get("ice beam", 0)
                if ib_lvl < 5:
                    score += 960.0
                    reasons.append(f"RUSH_ICE_BEAM_TO_LV5(cur={ib_lvl})")
                else:
                    score += 80.0
                    reasons.append("ice_beam_already_max_lv5")
            elif "hydro pump" in name:
                hp_lvl = self.acquired_moves.get("hydro pump", 0)
                if hp_lvl < 5:
                    bonus = 940.0 if is_water_poke else 500.0
                    score += bonus
                    reasons.append(f"RUSH_HYDRO_PUMP_TO_LV5(cur={hp_lvl})")
                else:
                    score += 80.0
                    reasons.append("hydro_pump_already_max_lv5")
            elif "shadow ball" in name:
                sb_lvl = self.acquired_moves.get("shadow ball", 0)
                if sb_lvl < 5:
                    score += 920.0
                    reasons.append(f"RUSH_SHADOW_BALL_TO_LV5(cur={sb_lvl})")
                else:
                    score += 80.0
                    reasons.append("shadow_ball_already_max_lv5")
            elif "flamethrower" in name:
                flame_lvl = self.acquired_moves.get("flamethrower", 0)
                if flame_lvl < 5:
                    # Contextual: only rush Flamethrower if playing Fire Pokemon or already owned!
                    if is_fire_poke or flame_lvl > 0:
                        score += 900.0
                        reasons.append(f"RUSH_FLAMETHROWER_TO_LV5(cur={flame_lvl})")
                    else:
                        score += 220.0
                        reasons.append(f"OFF_TYPE_FLAMETHROWER_NEUTRAL(cur={flame_lvl})")
                else:
                    score += 80.0
                    reasons.append("flamethrower_already_max_lv5")

            # 7. UPGRADE OWNED MOVES THAT CAN STILL EVOLVE
            elif any(m in name for m in self.acquired_moves if self.acquired_moves[m] < 5):
                matching_m = next(m for m in self.acquired_moves if m in name)
                c_lvl = self.acquired_moves[matching_m]
                score += 650.0 + (c_lvl * 40.0)
                reasons.append(f"UPGRADE_{matching_m.upper()}_TOWARD_EVOLUTION(cur={c_lvl}/5)")

            # 8. DEFENSIVE ARMOR PASSIVE
            elif "armor" in name or "defense" in desc:
                armor_count = self.acquired_stats.count("armor")
                if armor_count < 12:
                    score += 570.0
                    reasons.append(f"ARMOR_DEFENSE_REDUCTION_UNTIL_CAP(cur={armor_count}/12)")
                else:
                    score += 350.0
                    reasons.append("armor_defense_capped")

            # 9. ESSENTIAL STAT PASSIVES
            elif "power" in name:
                score += 560.0
                reasons.append("POWER_ATTACK_DAMAGE_TIER1")
            elif "reflexes" in name or "cooldown" in desc:
                score += 555.0
                reasons.append("REFLEXES_COOLDOWN_TIER1")
            elif "focus" in name or "sniper" in name or "crit" in desc:
                score += 550.0
                reasons.append("FOCUS_CRIT_TIER2")
            elif "vigor" in name or "max hp" in desc:
                score += 540.0
                reasons.append("VIGOR_MAX_HP_TIER2")
            elif "synthesis" in name or "regen" in desc:
                score += 535.0
                reasons.append("SYNTHESIS_REGEN_TIER2")
            elif "magnet" in name or "pickup" in desc or "range" in desc:
                score += 530.0
                reasons.append("MAGNET_PICKUP_RANGE_TIER2")
            elif "learning" in name or "exp" in desc or "growth" in desc:
                score += 520.0
                reasons.append("LEARNING_EXP_BOOST_TIER2")
            elif "sprint" in name or "speed" in desc:
                score += 510.0
                reasons.append("SPRINT_MOVE_SPEED_TIER2")
            else:
                score += 150.0
                reasons.append("baseline_upgrade")

            scored_cards.append((card, score, ", ".join(reasons)))

        scored_cards.sort(key=lambda x: x[1], reverse=True)
        best_card, best_score, best_reason = scored_cards[0]
        best_card["selection_score"] = best_score
        best_card["selection_reason"] = best_reason
        best_card["should_reroll"] = (best_score < 500.0)

        # Update acquired moves & stats
        card_name = best_card.get("name", "").strip().lower()
        self._record_acquisition(card_name)

        return best_card

    def _is_required_passive_for_owned_moves(self, card_name: str) -> bool:
        """Checks if card_name is the required evolution passive for ANY currently owned move."""
        matched_m, req_p = self._get_required_passive_context(card_name)
        return bool(matched_m and req_p)

    def _get_required_passive_context(self, card_name: str) -> Tuple[str, str]:
        """Returns (owned_move_name, required_passive_name) if card_name satisfies evolution prerequisite."""
        c_name = card_name.lower().strip()
        for owned_m in self.acquired_moves.keys():
            evo_info = self.evolutions_db.get(owned_m.lower())
            if evo_info and evo_info.get("requires_passive"):
                req_p = evo_info["requires_passive"].lower()
                if req_p in c_name and req_p not in [s.lower() for s in self.acquired_stats]:
                    return owned_m, evo_info["requires_passive"]
        return "", ""

    def _record_acquisition(self, card_name: str):
        """Records acquired card into acquired_moves or acquired_stats."""
        c_name = card_name.lower().strip()
        if "→" in c_name:
            c_name = c_name.split("→")[-1].strip()
        elif "->" in c_name:
            c_name = c_name.split("->")[-1].strip()

        evo_info = self.evolutions_db.get(c_name)
        is_known_move = evo_info and (evo_info.get("kind") is not None or evo_info.get("buff_stat") is not None)
        
        if is_known_move or any(p in c_name for p in self.priority_moves) or any(b in c_name for b in self.buff_moves):
            matched_key = c_name
            for m in list(self.acquired_moves.keys()) + self.priority_moves:
                if m in c_name:
                    matched_key = m
                    break
            self.acquired_moves[matched_key] = self.acquired_moves.get(matched_key, 1) + 1
        else:
            for s in ["power", "reflexes", "focus", "sniper", "vigor", "synthesis", "learning", "magnet", "sprint", "armor"]:
                if s in c_name:
                    self.acquired_stats.append(s)
                    break
