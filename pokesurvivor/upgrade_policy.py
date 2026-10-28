from typing import Dict, Any, List, Optional, Tuple

class UpgradePolicy:
    """
    PokéSurvivor Meta Upgrade & Fusion Policy:
    Built on official community meta strategies to push past 15-20+ minutes:
    - Rushes primary attack (Ember) to Level 5.
    - Secures evolution-enabling stat cards (Power for Fire Blast, Sprint for Speed, Vigor for HP).
    - Immediately selects Golden Evolution cards (Fire Blast, Take Down).
    - Rift Boss Reward: Bonds 2 moves together (max 4 loadout: 1 Buff + 3 Attack moves firing simultaneously).
    - Held Item: Manages single held item slot (swapping for top-tier survival/damage items).
    - Chests: Prioritizes Rare Candy (+1 level) > Meta Upgrades > Gacha Tickets.
    """
    def __init__(self, policy_config: Dict[str, Any]):
        self.active_pokemon: str = "Tapu Fini"
        self.dex: int = 788
        self.update_config(policy_config)
        self.reset_match()

    def set_active_pokemon(self, name: str, starting_move: str = "", dex: Optional[int] = 0):
        """Dynamically adapts upgrade prioritization to the active Pokemon and its starter move."""
        if name:
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

        self.item_priority = [
            "leftovers", "shell-bell", "focus-band", # S-Tier Survival
            "charcoal", "choice-specs", "choice-band", "muscle-band", # S-Tier Damage
            "amulet-coin", "exp-share", "lucky-egg" # S-Tier Economy
        ]

    def choose_card(
        self,
        choices: List[Dict[str, Any]],
        current_hp_ratio: float = 1.0,
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
        """Manages the 1 single item slot in PokéSurvivor."""
        # Typically choices[0] is Keep, choices[1] is Switch To
        if len(choices) == 1:
            return choices[0]

        keep_card = choices[0]
        switch_card = choices[1]
        new_item_name = switch_card.get("name", "").lower()

        # If we have no item yet, ALWAYS take the new item!
        if not self.held_item:
            self.held_item = new_item_name
            switch_card["selection_score"] = 500.0
            switch_card["selection_reason"] = f"EQUIP_FIRST_ITEM: {new_item_name}"
            return switch_card

        # Compare tier of new item vs current item
        new_rank = 99
        held_rank = 99
        for idx, item_id in enumerate(self.item_priority):
            if item_id in new_item_name:
                new_rank = idx
            if item_id in self.held_item.lower():
                held_rank = idx

        # Lower rank index = higher tier
        if new_rank < held_rank:
            self.held_item = new_item_name
            switch_card["selection_score"] = 400.0
            switch_card["selection_reason"] = f"UPGRADE_ITEM_TO_TIER_{new_rank+1}: {new_item_name}"
            return switch_card
        else:
            keep_card["selection_score"] = 350.0
            keep_card["selection_reason"] = f"KEEP_SUPERIOR_ITEM: {self.held_item}"
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
                if is_pri_attack or "ember" in name or "fire" in name:
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
        """Selects best chest reward: Rare Candy > Evolution Upgrade > Tickets."""
        scored = []
        for card in choices:
            name = card.get("name", "").lower()
            desc = (card.get("desc") or card.get("description") or "").lower()
            score = 100.0
            reason = "chest_reward"

            if "candy" in name or "rare" in name:
                score = 450.0
                reason = "RARE_CANDY_INSTANT_LEVEL"
            elif "evolution" in name or "gold" in name:
                score = 400.0
                reason = "EVOLUTION_UPGRADE"
            elif "ticket" in name:
                score = 300.0
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

    def _choose_standard_upgrade(self, choices: List[Dict[str, Any]], current_hp_ratio: Optional[float] = 1.0) -> Dict[str, Any]:
        """
        PokéSurvivor Dynamic Starter & Evolution Meta Picker:
        1. Golden Evolution Cards (Scald, Blizzard, Fire Blast, Overheat, Sacred Fire, Take Down) -> +2000
        2. Emergency Sitrus Berry if HP < 50% -> +1500 (Saves match)
        3. Core Buffs: Aqua Ring (+1250 for Tapu Fini passive regen), Swords Dance, Agility
        4. Rush Starter Move (Water Gun / Ember) to Level 5 -> +1050
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

        for card in choices:
            score = 10.0
            reasons = []
            name = card.get("name", "").strip().lower()
            tag = card.get("tag", "").strip().lower()
            desc = (card.get("desc") or card.get("description") or "").strip().lower()

            # 1. GOLDEN EVOLUTION CARDS (Supreme priority - clears entire screens!)
            is_evo = (
                "evolution" in tag or "evolv" in name or "gold" in tag or
                "scald" in name or "blizzard" in name or "hydro cannon" in name or
                "phantom force" in name or "frenzy plant" in name or "poison gas" in name or
                "fire blast" in name or "overheat" in name or "sacred fire" in name or
                "take down" in name or "inferno" in name or "extreme speed" in name or "surf" in name
            )
            if is_evo:
                score += 2000.0
                reasons.append("GOLDEN_EVOLUTION_MAX_PRIORITY")

            # 2. EMERGENCY HP RECOVERY
            elif is_damaged and ("sitrus berry" in name or "berry" in name or "heal" in desc or "recovery" in desc):
                score += 1500.0
                reasons.append("EMERGENCY_HEAL_LOW_HP")

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

            # 5. EVOLUTION PREREQUISITE STAT CARD: POWER (Attack - Evolves Water Gun -> Scald, Ember -> Fire Blast)
            elif "power" in name:
                if not has_power:
                    score += 950.0
                    reasons.append("POWER_REQUIRED_FOR_EVOLUTION")
                else:
                    score += 350.0
                    reasons.append("power_additional_damage")

            # 5. SECONDARY EVOLUTION PAIRS
            # Flamethrower + Reflexes -> Overheat
            elif "flamethrower" in name:
                flame_lvl = self.acquired_moves.get("flamethrower", 0)
                if flame_lvl < 5:
                    score += 900.0
                    reasons.append(f"RUSH_FLAMETHROWER_TO_LV5(cur={flame_lvl})")
                else:
                    score += 80.0
                    reasons.append("flamethrower_already_max_lv5")
            elif "reflexes" in name or "cooldown" in desc:
                if ("flamethrower" in self.acquired_moves or "shadow ball" in self.acquired_moves) and not has_reflexes:
                    score += 920.0
                    reasons.append("REFLEXES_REQUIRED_FOR_EVOLUTION")
                else:
                    score += 400.0
                    reasons.append("reflexes_cooldown_reduction")

            # Fire Spin + Synthesis -> Sacred Fire
            elif "fire spin" in name:
                score += 700.0
                reasons.append("FIRE_SPIN_ORBIT_SHIELD")
            elif "synthesis" in name or "regen" in desc:
                if "fire spin" in self.acquired_moves and not has_synthesis:
                    score += 680.0
                    reasons.append("SYNTHESIS_REQUIRED_FOR_SACRED_FIRE")
                else:
                    score += 320.0
                    reasons.append("synthesis_hp_regen")

            # Tackle + Vigor -> Take Down
            elif "tackle" in name:
                score += 650.0
                reasons.append("TACKLE_KNOCKBACK")
            elif "vigor" in name or "max hp" in desc:
                if "tackle" in self.acquired_moves and not has_vigor:
                    score += 630.0
                    reasons.append("VIGOR_REQUIRED_FOR_TAKE_DOWN")
                else:
                    score += 280.0
                    reasons.append("vigor_max_hp_boost")

            # Aqua Ring & Other crowd-clear moves
            elif "aqua ring" in name:
                score += 780.0
                reasons.append("AQUA_RING_REGEN_TANK")
            elif "will-o-wisp" in name or "heat wave" in name or "swift" in name:
                score += 550.0
                reasons.append("AOE_CROWD_CLEAR")

            # 6. MOBILITY & ECONOMY PASSIVE STATS (Fast leveling engine)
            elif "magnet" in name or "pickup" in desc:
                score += 880.0
                reasons.append("MAGNET_EXP_VACUUM_PRIORITY")
            elif "learning" in name or "experience" in desc:
                score += 830.0
                reasons.append("LEARNING_XP_MULTIPLIER")
            elif "sprint" in name or "movement speed" in desc:
                score += 800.0
                reasons.append("SPRINT_CRITICAL_KITING_SPEED")
            elif "armor" in name or "defense" in desc:
                score += 350.0
                reasons.append("ARMOR_DAMAGE_REDUCTION")

            # 7. BUFF MOVES (Swords Dance / Nasty Plot / Harden)
            elif any(b in name for b in self.buff_moves):
                if len(self.acquired_moves) >= 2:
                    score += 300.0
                    reasons.append("BUFF_MOVE_SYNERGY")
                else:
                    score += 80.0
                    reasons.append("defer_buff_until_attacks_ready")

            # 8. LEARNED VALUE NETWORK BONUS
            learned_r = self.skill_ratings.get(name, 0.0)
            if learned_r > 0:
                learned_bonus = learned_r * 40.0
                score += learned_bonus
                reasons.append(f"LEARNED_VALUE(+{learned_bonus:.0f})")

            # 9. RARITY TIERS
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

        # Update acquired moves & stats
        card_name = best_card.get("name", "").strip().lower()
        matched_move = False
        for p_move in self.priority_moves:
            if p_move in card_name:
                self.acquired_moves[p_move] = self.acquired_moves.get(p_move, 1) + 1
                matched_move = True
                break

        if not matched_move:
            for p_stat in self.priority_stats:
                if p_stat in card_name:
                    if p_stat not in self.acquired_stats:
                        self.acquired_stats.append(p_stat)
                    break

        return best_card
