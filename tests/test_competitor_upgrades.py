import unittest
from pokesurvivor.upgrade_policy import UpgradePolicy

class TestCompetitorUpgrades(unittest.TestCase):
    def setUp(self):
        self.policy = UpgradePolicy({})

    def test_evolution_prerequisite_bubble_needs_sprint(self):
        """Verifies that an owned move (Bubble) dynamically prioritizes its required passive (Sprint)."""
        self.policy.set_active_pokemon("Poliwag", starting_move="Bubble", dex=60)
        self.assertEqual(self.policy.rush_primary_move, "bubble")

        choices = [
            {"name": "Sprint", "tag": "Stat", "description": "+10% Move speed"},
            {"name": "Vigor", "tag": "Stat", "description": "+15% Max HP"},
            {"name": "Tackle", "tag": "Move", "description": "Charges in an arc"}
        ]
        chosen = self.policy.choose_card(choices, current_hp_ratio=1.0)
        self.assertEqual(chosen["name"], "Sprint")
        self.assertIn("REQUIRED_TO_EVOLVE_BUBBLE", chosen.get("selection_reason", ""))

    def test_evolution_prerequisite_disarming_voice_needs_reflexes(self):
        """Verifies that Disarming Voice dynamically prioritizes Reflexes for evolution."""
        self.policy.set_active_pokemon("Tapu Fini", starting_move="Water Gun", dex=788)
        # Simulate having acquired Disarming Voice
        self.policy.acquired_moves["disarming voice"] = 2

        choices = [
            {"name": "Reflexes", "tag": "Stat", "description": "-15% Cooldowns"},
            {"name": "Learning", "tag": "Stat", "description": "+20% Experience"},
            {"name": "Magnet", "tag": "Stat", "description": "+25% Pickup Range"}
        ]
        chosen = self.policy.choose_card(choices, current_hp_ratio=1.0)
        self.assertEqual(chosen["name"], "Reflexes")
        self.assertIn("REQUIRED_TO_EVOLVE", chosen.get("selection_reason", ""))

    def test_sitrus_berry_demoted_when_healthy(self):
        """Verifies that Sitrus Berry is demoted to low priority when HP >= 50% so level is not wasted."""
        self.policy.set_active_pokemon("Tapu Fini", starting_move="Water Gun", dex=788)
        choices = [
            {"name": "Sitrus Berry", "tag": "Item", "description": "Heals 50% HP"},
            {"name": "Magnet", "tag": "Stat", "description": "+25% Pickup Range"},
            {"name": "Vigor", "tag": "Stat", "description": "+15% Max HP"}
        ]
        chosen = self.policy.choose_card(choices, current_hp_ratio=0.85)
        # Should choose Magnet or Vigor, NOT Sitrus Berry!
        self.assertNotEqual(chosen["name"], "Sitrus Berry")
        self.assertIn(chosen["name"], ["Magnet", "Vigor"])

    def test_reroll_flag_when_choices_under_threshold(self):
        """Verifies should_reroll is True when choices contain no viable card (score < 500)."""
        self.policy.set_active_pokemon("Tapu Fini", starting_move="Water Gun", dex=788)
        self.policy.acquired_moves["water gun"] = 5  # Already maxed
        
        choices = [
            {"name": "Weak Strike", "tag": "Common", "description": "Minor scratch"},
            {"name": "Scratch", "tag": "Common", "description": "Normal scratch"}
        ]
        chosen = self.policy.choose_card(choices, current_hp_ratio=1.0)
        self.assertTrue(chosen.get("should_reroll", False))

if __name__ == "__main__":
    unittest.main()

