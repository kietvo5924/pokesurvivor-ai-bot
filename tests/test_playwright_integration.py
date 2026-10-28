"""
tests/test_playwright_integration.py
------------------------------------
Real Playwright Integration Test Suite:
1. document_start injection of CANVAS_HOOK_SCRIPT
2. window.G setter trap capture
3. Canvas telemetry fillText/drawImage hook and atomic buffer flushing
4. DOM modals (level-up, held item swap, chest) parsing
5. Genuine DOM match results modal parsing
6. OpenCV visual perception fallback and strict unobserved handling (no fabricated numbers)
"""

import unittest
import time
import numpy as np
import cv2
from playwright.sync_api import sync_playwright
from pokesurvivor.game_state_adapter import GameStateAdapter, CANVAS_HOOK_SCRIPT
from pokesurvivor.perception import PerceptionEngine

class TestPlaywrightIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding"
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 720})
        # Inject hook at document_start exactly as BrowserController does
        self.context.add_init_script(CANVAS_HOOK_SCRIPT)
        self.page = self.context.new_page()
        self.perception = PerceptionEngine()
        self.adapter = GameStateAdapter(page=self.page, perception=self.perception)

    def tearDown(self):
        self.page.close()
        self.context.close()

    def test_1_document_start_injection_and_window_g_trap(self):
        """Verifies hook is active before page scripts run and traps Object.prototype.Game / window.G."""
        # Load page where game engine defines Game on prototype and instantiates window.G
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>PokeSurvivor Game</title></head>
        <body>
            <div data-testid="game-phase">running</div>
            <script>
                // Simulate PokeSurvivor engine initializing Game
                const engine = {
                    BUILT: true,
                    LIFT: true,
                    Game: {
                        state: 'running',
                        player: {
                            x: 1050.0,
                            y: 650.0,
                            r: 16,
                            hp: 175,
                            maxHp: 200,
                            dex: 788,
                            baseSpd: 120,
                            spdMul: 1.0,
                            magnet: 70.0,
                            moves: ['water gun', 'bubble'],
                            stats: ['power', 'armor']
                        }
                    },
                    EnemyMgr: {
                        all: () => [
                            { x: 1100.0, y: 650.0, r: 16, dex: 19, hp: 50, maxHp: 50, atkType: 'normal', behavior: 'chase', shiny: false, boss: false },
                            { x: 1050.0, y: 750.0, r: 24, dex: 1000, hp: 300, maxHp: 300, atkType: 'steel', behavior: 'thief', shiny: false, boss: false }
                        ]
                    },
                    Hazards: {
                        all: [{ kind: 'zone', x: 1080.0, y: 650.0, r: 35, angle: 0, len: 0, w: 0 }]
                    },
                    Projectiles: {
                        all: [{ x: 1070.0, y: 650.0, r: 8, vx: -3.0, vy: 0.0, friendly: false }]
                    },
                    Pickups: {
                        all: [{ x: 1060.0, y: 640.0, kind: 'xp' }]
                    },
                    Rift: {
                        rift: { x: 1200.0, y: 800.0 },
                        inArena: false
                    }
                };
                // Triggers the setter trap on Object.prototype.Game via window.Game
                window.Game = engine;
            </script>
        </body>
        </html>
        """
        self.page.set_content(html_content)

        # Verify window.G was captured
        has_g = self.page.evaluate("() => !!window.G")
        self.assertTrue(has_g, "window.G must be captured by document_start trap setter")

        # Read unified state via adapter
        state = self.adapter.read_unified_state()
        self.assertEqual(state["player"]["source"], "window_G")
        self.assertEqual(state["player"]["hp"], 175)
        self.assertEqual(state["player"]["max_hp"], 200)
        self.assertAlmostEqual(state["player"]["hp_ratio"], 175 / 200)
        self.assertEqual(len(state["enemies"]), 2)
        # Verify Golden thief detected
        gholdengo = [e for e in state["enemies"] if e.get("dex") == 1000]
        self.assertEqual(len(gholdengo), 1)
        self.assertTrue(gholdengo[0]["is_thief"])
        self.assertEqual(len(state["hazards"]), 2) # 1 zone + 1 shot
        self.assertEqual(len(state["items"]), 1)
        self.assertIsNotNone(state["specials"]["rift"])

    def test_2_canvas_telemetry_hud_and_buffer_flushing(self):
        """Verifies Canvas fillText captures HUD and atomically flushes buffer to prevent stale data."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <canvas class="ps-canvas" id="game-canvas" width="1280" height="720"></canvas>
            <script>
                const canvas = document.getElementById('game-canvas');
                const ctx = canvas.getContext('2d');
                window.renderHUD = function(hpStr, killCount, coinCount) {
                    ctx.clearRect(0, 0, 1280, 720);
                    ctx.fillText('Tapu Fini', 20, 20);
                    ctx.fillText('Lv. 6', 20, 40);
                    ctx.fillText(hpStr, 20, 60);
                    ctx.fillText('2:30', 20, 80);
                    ctx.fillText('SHADY FOREST · ' + killCount + ' DEFEATED · ' + coinCount + ' COINS', 20, 100);
                };
                // Initial frame render
                window.renderHUD('182 / 182', 150, 40);
            </script>
        </body>
        </html>
        """
        self.page.set_content(html_content)

        # First read: Full HP (182 / 182), 150 kills, 40 coins
        state1 = self.adapter.read_unified_state()
        self.assertEqual(state1["player"]["source"], "canvas_telemetry_texts")
        self.assertEqual(state1["player"]["hp"], 182)
        self.assertEqual(state1["player"]["max_hp"], 182)
        self.assertAlmostEqual(state1["player"]["hp_ratio"], 1.0)
        self.assertEqual(state1["player"]["level"], 6)
        self.assertEqual(state1["metrics"]["kills"], 150)
        self.assertEqual(state1["metrics"]["coins"], 40)
        self.assertEqual(state1["metrics"]["source"], "canvas_hud_texts")

        # Second step: Player took damage -> HUD now draws "120 / 182", 155 kills, 45 coins
        self.page.evaluate("() => window.renderHUD('120 / 182', 155, 45)")

        state2 = self.adapter.read_unified_state()
        self.assertEqual(state2["player"]["hp"], 120)
        self.assertEqual(state2["player"]["max_hp"], 182)
        self.assertAlmostEqual(state2["player"]["hp_ratio"], 120 / 182)
        self.assertEqual(state2["metrics"]["kills"], 155)
        self.assertEqual(state2["metrics"]["coins"], 45)

        # Third step: Buffer flush verification: if game ceases drawing HUD text (e.g. cutscene),
        # next read must return None/unobserved rather than repeating old numbers!
        self.page.evaluate("() => { const ctx = document.getElementById('game-canvas').getContext('2d'); ctx.clearRect(0, 0, 1280, 720); }")
        state3 = self.adapter.read_unified_state()
        self.assertIsNone(state3["player"]["hp"])
        self.assertIsNone(state3["metrics"]["kills"])
        self.assertIsNone(state3["metrics"]["coins"])
        self.assertEqual(state3["metrics"]["source"], "unobserved")

    def test_3_dom_modals_parsing(self):
        """Verifies DOM Level-Up and Held Item Swap modals are parsed correctly."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <div data-testid="game-phase">running</div>
            <div data-testid="choice" class="ps-choice">
                <h2>Level 4</h2>
                <button class="ps-card" data-testid="option-0">
                    <span class="nm">Water Gun</span>
                    <span class="tag-chip">Lv.1 > 2</span>
                    <span class="ds">Shoots piercing water jets</span>
                    <div class="key"><span class="kb-only">1</span></div>
                </button>
                <button class="ps-card" data-testid="option-1">
                    <span class="nm">Reflexes</span>
                    <span class="tag-chip">Common</span>
                    <span class="ds">Evasion +10%</span>
                    <div class="key"><span class="kb-only">2</span></div>
                </button>
            </div>
        </body>
        </html>
        """
        self.page.set_content(html_content)

        state = self.adapter.read_unified_state()
        ui = state["ui"]
        self.assertTrue(ui["choice_open"])
        self.assertEqual(ui["modal_kind"], "level-up")
        self.assertEqual(ui["verified_level"], 4)
        self.assertEqual(len(ui["choices"]), 2)
        self.assertEqual(ui["choices"][0]["name"], "Water Gun")
        self.assertEqual(ui["choices"][0]["tag"], "Lv.1 > 2")
        self.assertEqual(ui["choices"][0]["key"], "1")

        # Mutate to Held Item Modal
        self.page.evaluate("""() => {
            const choice = document.querySelector('[data-testid="choice"]');
            choice.querySelector('h2').innerText = 'Held Item Swap (Hold only 1)';
        }""")

        state_item = self.adapter.read_unified_state()
        self.assertEqual(state_item["ui"]["modal_kind"], "item")

    def test_4_dom_match_results_modal(self):
        """Verifies genuine DOM end-of-match stats modal parsing."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <div data-testid="game-phase">ended</div>
            <div data-testid="results" class="ps-results">
                <h2>Game Over</h2>
                <div data-testid="run-stat-kills"><b>428</b></div>
                <div data-testid="run-stat-level"><b>19</b></div>
                <div data-testid="run-stat-damage"><b>62,500</b></div>
                <div data-testid="run-stat-downs"><b>0</b></div>
                <div class="stats">
                    <div><span>Time</span><b>3:12</b></div>
                    <div><span>Coins</span><b>185</b></div>
                    <div><span>Bosses Defeated</span><b>1</b></div>
                </div>
                <button data-testid="game-exit">Exit</button>
            </div>
        </body>
        </html>
        """
        self.page.set_content(html_content)

        state = self.adapter.read_unified_state()
        metrics = state["metrics"]
        self.assertTrue(metrics["is_match_over"])
        self.assertEqual(metrics["kills"], 428)
        self.assertEqual(metrics["level"], 19)
        self.assertEqual(metrics["coins"], 185)
        self.assertEqual(metrics["bosses"], 1)
        self.assertEqual(metrics["time_str"], "3:12")
        self.assertEqual(metrics["damage"], 62500)
        self.assertEqual(metrics["source"], "dom_results")
        self.assertTrue(state["ui"]["exit_available"])

    def test_5_opencv_fallback_and_strict_unobserved_metrics(self):
        """Verifies graceful fallback to OpenCV when window.G is null and telemetry is unobserved."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <div data-testid="game-phase">running</div>
            <canvas class="ps-canvas" id="game-canvas" width="1280" height="720"></canvas>
        </body>
        </html>
        """
        self.page.set_content(html_content)

        # Create mock gameplay frame (green grass with red enemy blob)
        frame = np.full((720, 1280, 3), (40, 120, 40), dtype=np.uint8)
        # Red enemy at (700, 360)
        cv2.circle(frame, (700, 360), 16, (0, 0, 220), -1)

        # Feed frame to adapter
        state = self.adapter.read_unified_state(frame_img=frame)
        self.assertEqual(state["sources"]["enemies"], "opencv_perception")
        self.assertEqual(state["sources"]["hazards"], "opencv_hazards")
        # Metrics not drawn on canvas or results modal -> strictly unobserved (not fabricated 0)
        self.assertIsNone(state["metrics"]["kills"])
        self.assertIsNone(state["metrics"]["coins"])
        self.assertEqual(state["metrics"]["source"], "unobserved")

if __name__ == "__main__":
    unittest.main()
