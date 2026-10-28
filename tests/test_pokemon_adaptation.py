import unittest
from pokesurvivor.upgrade_policy import UpgradePolicy

class TestPokemonAdaptation(unittest.TestCase):
    def test_tapu_fini_adaptation(self):
        policy = UpgradePolicy({})
        policy.set_active_pokemon("Tapu Fini", starting_move="Water Gun", dex=788)
        self.assertEqual(policy.rush_primary_move, "water gun")

        choices = [
            {"name": "Water Gun", "tag": "Move", "description": "Shoots pressurized water"},
            {"name": "Ember", "tag": "Move", "description": "Shoots small fire sparks"},
            {"name": "Tackle", "tag": "Move", "description": "Physical strike"}
        ]
        # Should rush Water Gun
        chosen = policy.choose_card(choices, current_hp_ratio=1.0)
        self.assertEqual(chosen["name"], "Water Gun")

    def test_tapu_fini_aqua_ring_priority(self):
        policy = UpgradePolicy({})
        policy.set_active_pokemon("Tapu Fini", starting_move="Water Gun", dex=788)
        choices = [
            {"name": "Aqua Ring", "tag": "Buff", "description": "Restores 1.6 HP every second"},
            {"name": "Tackle", "tag": "Move", "description": "Physical strike"},
            {"name": "Armor", "tag": "Stat", "description": "Reduces damage"}
        ]
        chosen = policy.choose_card(choices, current_hp_ratio=1.0)
        self.assertEqual(chosen["name"], "Aqua Ring")

    def test_scorbunny_adaptation(self):
        policy = UpgradePolicy({})
        policy.set_active_pokemon("Scorbunny", starting_move="Ember", dex=813)
        self.assertEqual(policy.rush_primary_move, "ember")

        choices = [
            {"name": "Water Gun", "tag": "Move", "desc": "Water attack"},
            {"name": "Ember", "tag": "Move", "desc": "Fire attack"},
            {"name": "Scratch", "tag": "Move", "desc": "Normal attack"}
        ]
        chosen = policy.choose_card(choices, current_hp_ratio=1.0)
        self.assertEqual(chosen["name"], "Ember")

    def test_emergency_healing_when_low_hp(self):
        policy = UpgradePolicy({})
        choices = [
            {"name": "Sitrus Berry", "tag": "Item", "description": "Instantly heals 50% HP"},
            {"name": "Tackle", "tag": "Move", "description": "Physical strike"},
            {"name": "Sprint", "tag": "Stat", "description": "+10% Move speed"}
        ]
        # With HP < 50%, Sitrus Berry MUST be selected to survive
        chosen = policy.choose_card(choices, current_hp_ratio=0.35)
        self.assertEqual(chosen["name"], "Sitrus Berry")

    def test_description_and_desc_field_compatibility(self):
        policy = UpgradePolicy({})
        # Card with 'description' instead of 'desc'
        choices = [
            {"name": "Swift", "tag": "Move", "description": "Increases movement speed by 15%"},
            {"name": "Scratch", "tag": "Move", "desc": "Weak normal strike"}
        ]
        chosen = policy.choose_card(choices, current_hp_ratio=1.0)
        self.assertIsNotNone(chosen)

if __name__ == "__main__":
    unittest.main()

