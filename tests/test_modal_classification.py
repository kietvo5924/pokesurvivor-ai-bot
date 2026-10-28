import unittest
from pokesurvivor.state_tracker import StateTracker
from pokesurvivor.upgrade_policy import UpgradePolicy

class TestModalClassificationAndLevel(unittest.TestCase):
    def setUp(self):
        self.tracker = StateTracker()
        self.tracker.start_run()
        self.policy = UpgradePolicy({})

    def test_level_up_increments_level(self):
        self.assertEqual(self.tracker.current_level, 1)
        card = {"name": "Water Gun", "tag": "Move"}
        self.tracker.record_upgrade(card, modal_kind="level-up")
        self.assertEqual(self.tracker.current_level, 2)

    def test_verified_level_overrides_counter(self):
        self.assertEqual(self.tracker.current_level, 1)
        card = {"name": "Water Gun", "tag": "Move"}
        # Game UI says Level 6!
        self.tracker.record_upgrade(card, modal_kind="level-up", verified_level=6)
        self.assertEqual(self.tracker.current_level, 6)

    def test_chest_modal_does_not_increment_level(self):
        self.tracker.current_level = 3
        card = {"name": "Gacha Ticket", "tag": "Reward"}
        self.tracker.record_upgrade(card, modal_kind="chest")
        # Level must remain 3!
        self.assertEqual(self.tracker.current_level, 3)
        self.assertEqual(self.tracker.upgrades_chosen[-1]["modal_kind"], "chest")

    def test_item_modal_does_not_increment_level(self):
        self.tracker.current_level = 4
        card = {"name": "Leftovers", "tag": "Item"}
        self.tracker.record_upgrade(card, modal_kind="item")
        # Level must remain 4!
        self.assertEqual(self.tracker.current_level, 4)
        self.assertEqual(self.tracker.upgrades_chosen[-1]["modal_kind"], "item")

    def test_bond_modal_does_not_increment_level(self):
        self.tracker.current_level = 5
        card = {"name": "Water Gun + Scald", "tag": "Bond"}
        self.tracker.record_upgrade(card, modal_kind="bond")
        # Level must remain 5!
        self.assertEqual(self.tracker.current_level, 5)
        self.assertEqual(self.tracker.upgrades_chosen[-1]["modal_kind"], "bond")

    def test_upgrade_policy_routes_correct_modal_handler(self):
        # 1. Item choices
        item_choices = [
            {"name": "Keep: Leftovers", "selector": "#c1"},
            {"name": "Switch: Shell Bell", "selector": "#c2"}
        ]
        choice = self.policy.choose_card(item_choices, modal_kind="item")
        self.assertIn("Shell Bell", choice["name"])

        # 2. Chest choices
        chest_choices = [
            {"name": "Tickets x2", "desc": "Gacha tickets"},
            {"name": "Rare Candy", "desc": "+1 Level instantly"}
        ]
        choice = self.policy.choose_card(chest_choices, modal_kind="chest")
        self.assertIn("Candy", choice["name"])

        # 3. Bond choices
        bond_choices = [
            {"name": "Aqua Ring", "tag": "Bond Move"},
            {"name": "Water Gun", "tag": "Bond Move"}
        ]
        choice = self.policy.choose_card(bond_choices, modal_kind="bond")
        self.assertIsNotNone(choice)

if __name__ == "__main__":
    unittest.main()

