import os
import json
from typing import Dict, Any, List, Optional, Tuple

class UpgradePolicy:
    """
    PokéSurvivor Meta Upgrade & Fusion Policy:
    Built on official community meta strategies to push past 15-20+ minutes:
    - Rushes primary attack (Water Gun / Ember) to Level 5.
    - Dynamically secures evolution-enabling stat cards based on 87-move evolution database.
    - Immediately selects Golden Evolution cards.
    - Rift Boss Reward: Bonds 2 moves together (max 4 loadout: 1 Buff + 3 Attack moves firing simultaneously).
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
        """Loads verified move evolutions database extracted from PokéSurvivor client."""
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolutions_db.json")
        try:
            if os.path.exists(db_path):
                with open(db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

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
        self.acquired_moves: Dict[str, int] = {}
        if self.rush_primary_move:
            self.acquired_moves[self.rush_primary_move] = 1
        self.acquired_stats: List[str] = []
        self.held_item: Optional[str] = None
        self.bonded_groups: List[str] = []

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

        # Item Tier list from verified game mechanics (BoundlessEgret / PokeSurvivorTool)
        # Note: Choice Band locks weapon slot and breaks multi-weapon E-cycling, so it is strictly avoided!
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
            "choice band": -100.0, # Breaks multi-slot firing
            "choice-band": -100.0
        }
        self.item_priority = [
            "shell bell", "leftovers", "quick claw", "focus sash", "assault vest",
            "rocky helmet", "lucky egg", "amulet coin", "choice scarf"
        ]

    def choose_card(
        self,
        choices: List[Dict[str, Any]],
        current_hp_ratio: Optional[float] = None,
        current_skills: List[str] = None,
        modal_kind: str = "level-up"
    ) -> Optional[Dict[str, Any]]:
        if not choices:
            return None

        # -------------------------------------------------------------
        # 1. HELD ITEM MODAL (Option 1 = Keep, Option 2 = Switch to found)
        # -------------------------------------------------------------
        if modal_kind == "item" or any("item" in c.get("name", "").lower() for c in choices):
            return self._choose_item(choices)

        # -------------------------------------------------------------
        # 2. BOND / MOVE COMBINATION MODAL (From Rift Boss!)
        # -------------------------------------------------------------
        if modal_kind == "bond" or any("bond" in c.get("name", "").lower() or "bond" in c.get("tag", "").lower() for c in choices):
            return self._choose_bond(choices, current_skills)

        # -------------------------------------------------------------
        # 3. CHEST REWARD MODAL (Candy vs Tickets vs Upgrades)
        # -------------------------------------------------------------
        if modal_kind == "chest":
            return self._choose_chest(choices)

        # -------------------------------------------------------------
        # 4. STANDARD LEVEL-UP UPGRADE CARDS
        # -------------------------------------------------------------
        return self._choose_standard_upgrade(choices, current_hp_ratio)

    def _choose_item(self, choices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Manages the 1 single item slot in PokéSurvivor based on survival tier list."""
        if len(choices) == 1:
            return choices[0]

        keep_card = choices[0]
        switch_card = choices[1]
        new_name = switch_card.get("name", "").lower().strip()
        held_name = (self.held_item or keep_card.get("name", "")).lower().strip()

        # Score function helper
        def get_item_score(item_name: str) -> float:
            clean = item_name.replace("-", " ").strip()
            for key, val in self.item_scores.items():
                if key.replace("-", " ") in clean:
                    return val
            return 20.0  # default baseline for unknown items

        new_score = get_item_score(new_name)
        held_score = get_item_score(held_name) if self.held_item else -999.0

        # Choice Band must ALWAYS be dropped immediately
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
        """
        Bonds 2 moves together (max 4 groups fired simultaneously!).
        Target loadout: 1 Buff move + 3 Attack moves.
        """
        scored_bonds = []
        for card in choices:
            name = card.get("name", "").strip().lower()
            score = 100.0
            reasons = []

            # Check if this move is a buff move
            is_buff = any(b in name for b in self.buff_moves)
            is_pri_attack = any(a in name for a in self.priority_moves)

            # Rule: Pair primary attack with a buff move
            if len(self.bonded_groups) == 0:
                if is_pri_attack or "ember" in name or "fire" in name or "water" in name:
                    score += 300.0
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
        """
        Selects best chest reward (Adapted from PokeSurvivorTool):
        1. Upgrade for an owned build move (score: +900)
        2. Rare Candy (instant level-up pick, score: +700)
        3. General Evolution / Move Upgrade (score: +500)
        4. Gacha Tickets (score: +150)
        """
        scored = []
        for card in choices:
            name = card.get("name", "").lower().strip()
            desc = (card.get("desc") or card.get("description") or "").lower().strip()
            score = 100.0
            reason = "chest_reward"

            # Check if this card upgrades an owned move
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
        """
        PokéSurvivor Dynamic Starter & Evolution Meta Picker:
        1. Golden Evolution Cards (Scald, Blizzard, Fire Blast, Overheat, Sacred Fire, Take Down) -> +2000
        2. Emergency Sitrus Berry if HP < 50% -> +1500; Defensive heal if HP is unknown -> +650
        3. Core Buffs: Aqua Ring (+1250 for Tapu Fini passive regen), Swords Dance, Agility
        4. Rush Starter Move (Water Gun / Ember / detected starter) to Level 5 -> +1050
        5. Evolution Prerequisite Stat: 'Power' (attack) for Scald/Fire Blast -> +950
        6. Synergy Combos (Ice Beam + Power, Shadow Ball + Reflexes, Flamethrower + Reflexes)
        7. Essential Passives: Magnet (+XP Range), Learning (+XP), Sprint (+Speed)
        """
        scored_cards: List[Tuple[Dict[str, Any], float, str]] = []

        has_power = "power" in self.acquired_stats
        has_reflexes = "reflexes" in self.acquired_stats
        has_synthesis = "synthesis" in self.acquired_stats
        has_vigor = "vigor" in self.acquired_stats

        is_damaged = (current_hp_ratio is not None and current_hp_ratio < 0.50)
        is_unknown_hp = (current_hp_ratio is None)

        for card in choices:
            score = 10.0
            reasons = []
            name = card.get("name", "").strip().lower()
            tag = card.get("tag", "").strip().lower()
            desc = (card.get("desc") or card.get("description") or "").strip().lower()

            # 1. GOLDEN EVOLUTION CARDS (Supreme priority - clears entire screens!)
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

            # 2. HP RECOVERY & HEALING SELECTION (HP-sensitive)
            elif "sitrus berry" in name or "berry" in name or "heal" in desc or "recovery" in desc:
                if is_damaged:
                    score += 1500.0
                    reasons.append("EMERGENCY_HEAL_LOW_HP")
                elif is_unknown_hp:
                    score += 850.0
                    reasons.append("DEFENSIVE_HEAL_UNKNOWN_HP")
                else:
                    # When healthy (HP >= 50%), sharply demote Sitrus Berry so an upgrade is not wasted!
                    score += 10.0
                    reasons.append("preventative_heal_demoted")

            # 3. CORE BUFF MOVES (Aqua Ring provides 1.6 HP/s up to 6.4 HP/s passive heal!)
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
                    score += 1000.0
                    reasons.append(f"RUSH_WATER_GUN_TO_LV5(cur={wg_lvl})")
                else:
                    score += 80.0
                    reasons.append("water_gun_already_max_lv5")

            elif "ember" in name:
                ember_lvl = self.acquired_moves.get("ember", 1)
                if ember_lvl < 5:
                    score += 1000.0
                    reasons.append(f"RUSH_PRIMARY_EMBER_TO_LV5(cur={ember_lvl})")
                else:
                    score += 80.0
                    reasons.append("ember_already_max_lv5")

            # 5. DYNAMIC EVOLUTION KEY: Check if this card is the REQUIRED PASSIVE for ANY owned move
            elif self._is_required_passive_for_owned_moves(name):
                matched_move, req_p = self._get_required_passive_context(name)
                score += 950.0
                reasons.append(f"{req_p.upper()}_REQUIRED_TO_EVOLVE_{matched_move.upper()}")

            # 6. PRIORITY HIGH-TIER ATTACK MOVES
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
                    score += 940.0
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
                    score += 900.0
                    reasons.append(f"RUSH_FLAMETHROWER_TO_LV5(cur={flame_lvl})")
                else:
                    score += 80.0
                    reasons.append("flamethrower_already_max_lv5")

            # 7. UPGRADE OWNED MOVES THAT CAN STILL EVOLVE
            elif any(m in name for m in self.acquired_moves if self.acquired_moves[m] < 5):
                matching_m = next(m for m in self.acquired_moves if m in name)
                c_lvl = self.acquired_moves[matching_m]
                score += 650.0 + (c_lvl * 40.0)
                reasons.append(f"UPGRADE_{matching_m.upper()}_TOWARD_EVOLUTION(cur={c_lvl}/5)")

            # 8. DEFENSIVE ARMOR PASSIVE (Up to 12-20 cap)
            elif "armor" in name or "defense" in desc:
                armor_count = self.acquired_stats.count("armor")
                if armor_count < 12:
                    score += 570.0
                    reasons.append(f"ARMOR_DEFENSE_REDUCTION_UNTIL_CAP(cur={armor_count}/12)")
                else:
                    score += 350.0
                    reasons.append("armor_defense_capped")

            # 9. ESSENTIAL STAT PASSIVES (BoundlessEgret Meta Tier)
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
            elif "learning" in name or "experience" in desc:
                score += 530.0
                reasons.append("LEARNING_XP_MULTIPLIER_TIER3")
            elif "magnet" in name or "pickup" in desc:
                score += 525.0
                reasons.append("MAGNET_EXP_VACUUM_TIER3")
            elif "sprint" in name or "movement speed" in desc:
                score += 515.0
                reasons.append("SPRINT_KITING_SPEED_TIER3")

            # 10. BUFF MOVES (Swords Dance / Nasty Plot / Harden)
            elif any(b in name for b in self.buff_moves):
                if len(self.acquired_moves) >= 2:
                    score += 480.0
                    reasons.append("BUFF_MOVE_SYNERGY")
                else:
                    score += 80.0
                    reasons.append("defer_buff_until_attacks_ready")

            # 11. GENERAL ATTACK MOVES
            elif any(m in name for m in self.priority_moves):
                score += 450.0
                reasons.append("PRIORITY_NEW_ATTACK")
            else:
                score += 430.0
                reasons.append("general_card_upgrade")

            # 12. LEARNED VALUE NETWORK BONUS
            learned_r = self.skill_ratings.get(name, 0.0)
            if learned_r > 0:
                learned_bonus = learned_r * 40.0
                score += learned_bonus
                reasons.append(f"LEARNED_VALUE(+{learned_bonus:.0f})")

            # 13. RARITY TIERS
            if "legendary" in tag:
                score += 60.0
                reasons.append("legendary")
            elif "epic" in tag:
                score += 40.0
                reasons.append("epic")
            elif "rare" in tag:
                score += 25.0
                reasons.append("rare")

            scored_cards.append((card, score, ", ".join(reasons)))

        scored_cards.sort(key=lambda x: x[1], reverse=True)
        best_card, best_score, best_reason = scored_cards[0]
        best_card["selection_score"] = best_score
        best_card["selection_reason"] = best_reason or "meta_priority"
        best_card["should_reroll"] = bool(best_score < 500.0)

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
        c_name = card_name.lower()
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
        # Clean evolution arrow if present
        if "→" in c_name:
            c_name = c_name.split("→")[-1].strip()
        elif "->" in c_name:
            c_name = c_name.split("->")[-1].strip()

        # Check if known move in database
        evo_info = self.evolutions_db.get(c_name)
        is_known_move = evo_info and (evo_info.get("kind") is not None or evo_info.get("buff_stat") is not None)
        
        if is_known_move or any(p in c_name for p in self.priority_moves) or any(b in c_name for b in self.buff_moves):
            # Key under primary move name
            matched_key = c_name
            for m in list(self.acquired_moves.keys()) + self.priority_moves:
                if m in c_name:
                    matched_key = m
                    break
            self.acquired_moves[matched_key] = self.acquired_moves.get(matched_key, 1) + 1
        else:
            # Check if stat passive
            for s in ["power", "reflexes", "focus", "sniper", "vigor", "synthesis", "learning", "magnet", "sprint", "armor"]:
                if s in c_name:
                    self.acquired_stats.append(s)
                    break
