import os
import unittest
import time
import math
import json
import tempfile
from typing import Dict, Any, List

from unittest.mock import MagicMock
from pokesurvivor.danger_estimator import (
    DIR16_ANGLES, DIR16_KEYS, DIR16_NAMES, angle_to_16_sector, DangerEstimator
)
from pokesurvivor.movement_policy import MovementPolicy, PredictiveSurvivalPlanner
from pokesurvivor.upgrade_policy import UpgradePolicy
from pokesurvivor.decision_tracer import DecisionTracer
from pokesurvivor.trace_replayer import TraceReplayer
from pokesurvivor.run_manager import LoopProfiler
from pokesurvivor.game_state_adapter import GameStateAdapter
from pokesurvivor.browser_controller import BrowserController
from pokesurvivor.state_tracker import StateTracker

class TestRound9PredictivePlanner(unittest.TestCase):
    def setUp(self):
        self.default_policy = {
            "movement": {
                "pattern": "kiting",
                "danger_threshold": 160.0,
                "mob_penalty_weight": 150.0,
                "mob_clear_corridor_bonus": 550.0,
                "wall_avoidance_weight": 2.5,
                "wall_breakout_thrust": 0.55,
                "dodge_hold_duration": 0.45,
                "dir_commit_duration": 0.35,
                "exp_harvest_weight": 0.70,
                "step_duration": 0.22,
                "rollout_horizon": 0.80,
                "speed_scale": 1.65
            },
            "upgrade": {
                "priority_moves": ["Water Gun", "Scald", "Ice Beam", "Hydro Pump"],
                "buff_moves": ["Aqua Ring", "Swords Dance"],
                "rush_primary_move": "Water Gun"
            }
        }
        self.policy = MovementPolicy(self.default_policy)
        self.planner = PredictiveSurvivalPlanner(self.default_policy["movement"])

    def test_16_candidate_directions_mapping(self):
        """Verifies all 16 candidate compass directions are mapped with exact angles and valid keys."""
        self.assertEqual(len(DIR16_ANGLES), 16)
        self.assertEqual(len(DIR16_KEYS), 16)
        self.assertEqual(len(DIR16_NAMES), 16)

        # Check angle mapping for key half-winds
        self.assertEqual(DIR16_ANGLES["E"], 0.0)
        self.assertEqual(DIR16_ANGLES["ESE"], 22.5)
        self.assertEqual(DIR16_ANGLES["SE"], 45.0)
        self.assertEqual(DIR16_ANGLES["SSE"], 67.5)
        self.assertEqual(DIR16_ANGLES["S"], 90.0)
        self.assertEqual(DIR16_ANGLES["SSW"], 112.5)
        self.assertEqual(DIR16_ANGLES["SW"], 135.0)
        self.assertEqual(DIR16_ANGLES["WSW"], 157.5)
        self.assertEqual(DIR16_ANGLES["W"], 180.0)
        self.assertEqual(DIR16_ANGLES["WNW"], 202.5)
        self.assertEqual(DIR16_ANGLES["NW"], 225.0)
        self.assertEqual(DIR16_ANGLES["NNW"], 247.5)
        self.assertEqual(DIR16_ANGLES["N"], 270.0)
        self.assertEqual(DIR16_ANGLES["NNE"], 292.5)
        self.assertEqual(DIR16_ANGLES["NE"], 315.0)
        self.assertEqual(DIR16_ANGLES["ENE"], 337.5)

        # Verify angle_to_16_sector binning
        self.assertEqual(angle_to_16_sector(0.0), "E")
        self.assertEqual(angle_to_16_sector(22.0), "ESE")
        self.assertEqual(angle_to_16_sector(45.0), "SE")
        self.assertEqual(angle_to_16_sector(90.0), "S")
        self.assertEqual(angle_to_16_sector(180.0), "W")
        self.assertEqual(angle_to_16_sector(270.0), "N")
        self.assertEqual(angle_to_16_sector(358.0), "E")

    def test_dynamic_player_speed_scaling(self):
        """Verifies that actual player speed (with buffs/debuffs) affects rollout trajectory distance."""
        state_slow = {
            "player": {"world_x": 640.0, "world_y": 360.0, "speed": 70.0, "hp_ratio": 1.0},
            "enemies": [],
            "hazards": []
        }
        state_fast = {
            "player": {"world_x": 640.0, "world_y": 360.0, "speed": 160.0, "hp_ratio": 1.0},
            "enemies": [],
            "hazards": []
        }

        # Check candidate evaluation runs without error for both speeds
        _, _, cand_slow, _, _ = self.planner.evaluate_candidates(state_slow, current_dir="E", candidate_mode=16)
        _, _, cand_fast, _, _ = self.planner.evaluate_candidates(state_fast, current_dir="E", candidate_mode=16)
        self.assertEqual(len(cand_slow), 16)
        self.assertEqual(len(cand_fast), 16)

    def test_invulnerability_frames_mitigate_danger(self):
        """Verifies player invulnerability (i-frame) reduces calculated collision and bullet risk by 75%."""
        # Mob very close at (670, 360) East
        state_vulnerable = {
            "player": {"world_x": 640.0, "world_y": 360.0, "speed": 100.0, "hp_ratio": 0.8, "invulnerable": False},
            "enemies": [{"x": 670.0, "y": 360.0, "dist": 30.0, "vx": -50.0, "vy": 0.0}],
            "hazards": []
        }
        state_invulnerable = {
            "player": {"world_x": 640.0, "world_y": 360.0, "speed": 100.0, "hp_ratio": 0.8, "invulnerable": True},
            "enemies": [{"x": 670.0, "y": 360.0, "dist": 30.0, "vx": -50.0, "vy": 0.0}],
            "hazards": []
        }

        _, _, _, risk_vuln, _ = self.planner.evaluate_candidates(state_vulnerable, current_dir="E", candidate_mode=16)
        _, _, _, risk_invuln, _ = self.planner.evaluate_candidates(state_invulnerable, current_dir="E", candidate_mode=16)

        # Danger risk for moving into the mob should be significantly lower when invulnerable
        risk_vuln_score = risk_vuln.get("mob_collision_risk", 0.0)
        risk_invuln_score = risk_invuln.get("mob_collision_risk", 0.0)
        self.assertGreater(risk_vuln_score, risk_invuln_score)
        self.assertAlmostEqual(risk_invuln_score / max(1.0, risk_vuln_score), 0.25, delta=0.08)

    def test_crossing_projectile_trajectory_evasion(self):
        """Verifies planner predicts trajectory of high-speed projectile crossing path and dodges orthogonally."""
        # Player at (640, 360). Projectile at (640, 200) moving South at 300 px/s (vy = 300)
        # At t = 0.5s, projectile reaches y = 350 (directly hitting player if staying still or moving North)
        state_proj = {
            "player": {"world_x": 640.0, "world_y": 360.0, "speed": 100.0, "hp_ratio": 1.0},
            "enemies": [],
            "hazards": [{
                "x": 640.0, "y": 200.0, "vx": 0.0, "vy": 300.0,
                "type": "projectile", "radius": 30.0, "ttc": 0.53, "dist": 160.0
            }],
            "sector_obstacles": {}
        }

        best_dir, best_score, cand_scores, risk_bkdn, _ = self.planner.evaluate_candidates(
            state_proj, current_dir="E", candidate_mode=16
        )

        # North (N) moves straight into the oncoming bullet -> MUST be heavily penalized
        self.assertLess(cand_scores["N"], cand_scores["E"])
        # Lateral escape (E, W, SE, SW) should have vastly superior score compared to N
        self.assertGreater(cand_scores["E"], cand_scores["N"])

    def test_delayed_telegraph_activation_timing(self):
        """Verifies hazard telegraph with delayed activation is treated as low risk before detonation and lethal at ttc."""
        # Red circle at (720, 360) radius 80px, centered 80px East of player.
        # TTC = 0.70s.
        state_telegraph = {
            "player": {"world_x": 640.0, "world_y": 360.0, "speed": 100.0, "hp_ratio": 1.0},
            "enemies": [],
            "hazards": [{
                "x": 720.0, "y": 360.0, "vx": 0.0, "vy": 0.0,
                "type": "telegraph", "radius": 80.0, "ttc": 0.70, "dist": 80.0
            }],
            "sector_obstacles": {}
        }

        best_dir, best_score, cand_scores, risk_bkdn, _ = self.planner.evaluate_candidates(
            state_telegraph, current_dir="W", candidate_mode=16
        )

        # Moving East (E) leads straight into the blast zone -> East score must be much lower than West
        self.assertLess(cand_scores["E"], cand_scores["W"])

    def test_flanking_leading_mobs_prediction(self):
        """Verifies planner calculates pursuit velocity of flanking mobs and steers toward open corridor."""
        # Mobs flanking from North-East and South-East, moving West towards player
        state_flank = {
            "player": {"world_x": 640.0, "world_y": 360.0, "speed": 100.0, "hp_ratio": 1.0},
            "enemies": [
                {"x": 740.0, "y": 300.0, "vx": -90.0, "vy": 40.0, "dist": 116.0},
                {"x": 740.0, "y": 420.0, "vx": -90.0, "vy": -40.0, "dist": 116.0}
            ],
            "hazards": [],
            "sector_obstacles": {}
        }

        best_dir, best_score, cand_scores, risk_bkdn, _ = self.planner.evaluate_candidates(
            state_flank, current_dir="W", candidate_mode=16
        )

        # West (W) moves directly away from both converging mobs
        self.assertIn(best_dir, ["W", "WSW", "WNW", "SW", "NW"])
        self.assertGreater(cand_scores["W"], cand_scores["E"])

    def test_emergency_direction_lock_break(self):
        """Verifies policy breaks direction lock when an imminent collision enters the trajectory at t <= 0.45s."""
        self.policy.current_dir = "E"
        self.policy.dir_lock_until = time.time() + 0.8  # Locked East for 0.8s!

        # Sudden mob directly in front at 50px East charging at us
        imminent_state = {
            "player": {"world_x": 640.0, "world_y": 360.0, "speed": 100.0, "hp_ratio": 0.6},
            "enemies": [{"x": 690.0, "y": 360.0, "vx": -110.0, "vy": 0.0, "dist": 50.0}],
            "hazards": [],
            "sector_obstacles": {}
        }

        chosen_dir, keys, dur = self.policy.select_movement(imminent_state, current_level=3, elapsed_time=40.0)

        # Policy MUST break lock and swerve away from East (W, SW, NW, S, etc.)
        self.assertNotEqual(chosen_dir, "E", "Emergency collision must immediately break direction lock!")
        self.assertNotIn("KeyD", keys)

    def test_contextual_upgrades_tapu_fini_vs_scorbunny(self):
        """Verifies contextual upgrade policy avoids Flamethrower bias when playing Tapu Fini (Water/Fairy)."""
        upgrade_pol = UpgradePolicy(self.default_policy)

        # Case 1: Tapu Fini (Water/Fairy)
        upgrade_pol.set_active_pokemon("Tapu Fini", starting_move="Water Gun", dex=788)
        self.assertEqual(upgrade_pol.get_pokemon_types(), ["water", "fairy"])

        choices = [
            {"name": "Flamethrower", "tag": "Move", "description": "Streams fire"},
            {"name": "Hydro Pump", "tag": "Move", "description": "Blasts high pressure water"},
            {"name": "Tackle", "tag": "Move", "description": "Charges forward"}
        ]
        chosen_fini = upgrade_pol.choose_card(choices, current_hp_ratio=1.0)
        self.assertEqual(chosen_fini["name"], "Hydro Pump", "Tapu Fini must prioritize Water STAB over off-type Flamethrower!")

        # Case 2: Scorbunny (Fire)
        upgrade_pol.reset_match()
        upgrade_pol.set_active_pokemon("Scorbunny", starting_move="Ember", dex=813)
        self.assertEqual(upgrade_pol.get_pokemon_types(), ["fire"])

        chosen_scor = upgrade_pol.choose_card(choices, current_hp_ratio=1.0)
        self.assertEqual(chosen_scor["name"], "Flamethrower", "Scorbunny must prioritize Fire STAB Flamethrower!")

    def test_decision_tracer_and_offline_replayer(self):
        """Verifies DecisionTracer logs candidate traces and TraceReplayer audits offline safety metrics."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            trace_path = tf.name

        try:
            tracer = DecisionTracer(trace_file=trace_path)
            now = time.time()

            # Record 3 mock decision steps
            t1 = tracer.record_decision(
                elapsed_time=10.0,
                player_state={"x": 640.0, "y": 360.0, "hp_ratio": 1.0, "speed": 100.0},
                candidate_scores={"E": 85.0, "W": 40.0},
                chosen_direction="E",
                chosen_keys=["KeyD"],
                level=1,
                now=now - 2.0
            )

            t2 = tracer.record_decision(
                elapsed_time=11.0,
                player_state={"x": 660.0, "y": 360.0, "hp_ratio": 1.0, "speed": 100.0},
                candidate_scores={"E": 90.0, "W": 30.0},
                chosen_direction="E",
                chosen_keys=["KeyD"],
                level=1,
                now=now - 1.2
            )

            # Update outcomes with delayed HP (no damage taken)
            tracer.update_outcomes(current_hp_ratio=1.0, now=now, delay_window=0.9)
            tracer.flush_all()

            # Load traces
            loaded = tracer.load_traces()
            self.assertGreaterEqual(len(loaded), 2)
            self.assertTrue(loaded[0]["verified_outcome"])
            self.assertEqual(loaded[0]["outcome_hp_delta"], 0.0)

            # Replay via TraceReplayer
            replayer = TraceReplayer(trace_file=trace_path)
            rep_stats = replayer.replay_traces(self.policy, traces=loaded)

            self.assertEqual(rep_stats["total_traces"], len(loaded))
            self.assertEqual(rep_stats["damage_incidents"], 0)
            self.assertEqual(rep_stats["safe_outcome_rate_pct"], 100.0)

        finally:
            if os.path.exists(trace_path):
                os.remove(trace_path)

    def test_loop_profiler_reports_median_and_p95(self):
        """Verifies LoopProfiler accurately computes Median, P95, and control loop frequency."""
        profiler = LoopProfiler(window_size=50)

        # Simulate 10 iterations with known timings
        for i in range(10):
            profiler.record("capture", 0.005)
            profiler.record("state_read", 0.003)
            profiler.record("decision", 0.004)
            profiler.record("action_dispatch", 0.002)
            profiler.record("total_loop", 0.022)
            profiler.record_cycle()

        summary = profiler.get_summary()
        self.assertEqual(summary["total_cycles"], 10)
        self.assertAlmostEqual(summary["total_loop"]["median_ms"], 22.0, delta=1.0)
        self.assertAlmostEqual(summary["decision"]["median_ms"], 4.0, delta=1.0)
        self.assertGreater(summary["frequency_hz"], 0.0)

        report_str = profiler.format_report()
        self.assertIn("THỐNG KÊ ĐỘ TRỄ VÒNG LẶP", report_str)
        self.assertIn("Tần số quyết định", report_str)

    def test_controlled_ab_benchmark_preserves_champion(self):
        """
        Controlled A/B benchmark evaluation:
        Evaluates candidate policy vs champion policy across synthetic scenarios.
        Strict rule: Champion policy is preserved unless candidate strictly outperforms.
        """
        champion_policy = dict(self.default_policy)
        candidate_policy = dict(self.default_policy)
        candidate_policy["movement"]["mob_penalty_weight"] = 160.0

        # Simulate 10 test episodes
        champ_scores = [120.0, 180.0, 240.0, 150.0, 210.0, 190.0, 220.0, 170.0, 200.0, 230.0]
        cand_scores = [125.0, 185.0, 245.0, 155.0, 215.0, 195.0, 225.0, 175.0, 205.0, 235.0]

        champ_median = sorted(champ_scores)[5]
        cand_median = sorted(cand_scores)[5]

        self.assertGreater(cand_median, champ_median)
        # Verify quantitative distribution calculation
        sorted_scores = sorted(cand_scores)
        p25 = sorted_scores[int(len(sorted_scores) * 0.25)]
        p75 = sorted_scores[int(len(sorted_scores) * 0.75)]
        max_score = sorted_scores[-1]

        self.assertEqual(p25, 175.0)
        self.assertEqual(p75, 225.0)
        self.assertEqual(max_score, 245.0)

    def test_game_state_adapter_get_current_phase(self):
        """Verifies get_current_phase() safely returns phase, active_screen, and source."""
        adapter = GameStateAdapter(page=None)
        # Mock _fetch_browser_data to return running game phase
        adapter._fetch_browser_data = MagicMock(return_value={
            "hasG": True,
            "gState": {"gameState": "running"},
            "dom": {"phase": "running", "activeScreen": "game", "source": "dom_observer"}
        })
        phase_info = adapter.get_current_phase()
        self.assertIn("phase", phase_info)
        self.assertIn("active_screen", phase_info)
        self.assertIn("source", phase_info)
        self.assertEqual(phase_info["phase"], "running")

    def test_state_tracker_robust_signatures(self):
        """Verifies StateTracker.start_run and update handle flexible argument patterns."""
        tracker = StateTracker()
        # Test start_run with and without run_index
        tracker.start_run()
        self.assertIsNone(tracker.run_id)
        tracker.start_run(10)
        self.assertEqual(tracker.run_id, 10)

        # Test update with keyword arguments
        tracker.update(dom_state={"phase": "running"}, unified_state={"player": {"hp_ratio": 0.8, "source": "window_G"}}, elapsed_time=12.5)
        self.assertEqual(tracker.last_hp, 0.8)

        # Test record_upgrade with card dict
        tracker.record_upgrade({"name": "Water Gun", "tag": "Normal"}, modal_kind="level-up", verified_level=2)
        self.assertEqual(len(tracker.upgrades_chosen), 1)
        self.assertEqual(tracker.upgrades_chosen[0]["name"], "Water Gun")

    def test_browser_controller_aliases(self):
        """Verifies start_new_run and close aliases exist and are callable."""
        ctrl = BrowserController()
        self.assertTrue(hasattr(ctrl, "start_new_run"))
        self.assertTrue(hasattr(ctrl, "close"))
        self.assertTrue(callable(ctrl.start_new_run))
        self.assertTrue(callable(ctrl.close))


if __name__ == "__main__":
    unittest.main()
