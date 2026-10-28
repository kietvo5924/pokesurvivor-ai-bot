"""
pokesurvivor.game_state_adapter
-------------------------------
Unified Game State Adapter bridging:
1. window.G (if running on pokesurvivor.com / /dev.html)
2. Injected Canvas Hook & Telemetry (Doofcraft / Canvas-based engine)
3. DOM state & genuine match results
4. OpenCV perception engine (fallback & visual confirmation)

Produces a standardized GameState structure so policies and learning engines
never depend on the extraction mechanism.
"""

import time
import math
import re
from typing import Dict, Any, List, Optional, Tuple
from playwright.sync_api import Page
from pokesurvivor.perception import PerceptionEngine

# Injected hook executed at document_start in the browser context
CANVAS_HOOK_SCRIPT = """
(() => {
  // 1. One-shot setter hook for window.G on pokesurvivor.com / /dev.html
  // (Reference: PokeSurvivorTool sites/default/hook.js · MIT License)
  if (!window.G && !Object.getOwnPropertyDescriptor(Object.prototype, 'Game')) {
    const trap = {
      configurable: true,
      get() { return undefined; },
      set(v) {
        Object.defineProperty(this, 'Game', { value: v, writable: true, enumerable: true, configurable: true });
        if (this && this.BUILT === true && 'LIFT' in this) {
          if (Object.getOwnPropertyDescriptor(Object.prototype, 'Game')?.set === trap.set) {
            delete Object.prototype.Game;
          }
          window.G = this;
        }
      },
    };
    Object.defineProperty(Object.prototype, 'Game', trap);
    addEventListener('load', () => {
      if (!window.G && Object.getOwnPropertyDescriptor(Object.prototype, 'Game')?.set === trap.set) {
        delete Object.prototype.Game;
      }
    }, { once: true });
  }

  // 2. Real-time telemetry buffer for canvas-based engines (Doofcraft / generic)
  window.__psTelemetry = {
    sprites: [],
    shapes: [],
    texts: [],
    camera: { s: 1.0, e: 0.0, f: 0.0 },
    lastFrameAt: performance.now(),
    frameCount: 0
  };

  const P = CanvasRenderingContext2D.prototype;
  const origDrawImage = P.drawImage;
  const origFillText = P.fillText;
  const origFill = P.fill;
  const origStroke = P.stroke;

  // Intercept Canvas drawImage to capture sprites & transforms
  P.drawImage = function(img, ...args) {
    try {
      if (this.canvas && (this.canvas.classList.contains('ps-canvas') || this.canvas.id === 'game-canvas')) {
        const m = this.getTransform ? this.getTransform() : { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
        window.__psTelemetry.camera = { s: m.a, e: m.e, f: m.f };
        
        const src = img.src || (img.tagName ? img.tagName : 'canvas');
        const dw = args.length === 8 ? args[6] : args.length === 4 ? args[2] : (img.width || 0);
        const dh = args.length === 8 ? args[7] : args.length === 4 ? args[3] : (img.height || 0);
        const dx = args.length >= 4 ? args[args.length === 8 ? 4 : 0] : 0;
        const dy = args.length >= 4 ? args[args.length === 8 ? 5 : 1] : 0;
        const cx = m.a * (dx + dw / 2) + m.e;
        const cy = m.d * (dy + dh / 2) + m.f;

        if (window.__psTelemetry.sprites.length < 400) {
          window.__psTelemetry.sprites.push({
            src: src.slice(-60),
            cx: cx,
            cy: cy,
            w: dw * m.a,
            h: dh * m.d
          });
        }
        window.__psTelemetry.lastFrameAt = performance.now();
      }
    } catch(e) {}
    return origDrawImage.apply(this, [img, ...args]);
  };

  // Intercept fillText for live HUD information (HP, level, kills, coins, time)
  P.fillText = function(text, ...args) {
    try {
      if (this.canvas && (this.canvas.classList.contains('ps-canvas') || this.canvas.id === 'game-canvas')) {
        const str = String(text).trim();
        if (str && window.__psTelemetry.texts.length < 60) {
          window.__psTelemetry.texts.push(str);
        }
      }
    } catch(e) {}
    return origFillText.apply(this, [text, ...args]);
  };
})();
"""

class GameStateAdapter:
    def __init__(self, page: Optional[Page] = None, perception: Optional[PerceptionEngine] = None):
        self.page = page
        self.perception = perception or PerceptionEngine()
        self.last_state: Dict[str, Any] = {}
        self.last_read_time = 0.0

    def read_unified_state(self, frame_img: Optional[Any] = None, is_moving: bool = True) -> Dict[str, Any]:
        """
        Reads, verifies, and harmonizes state from all available layers:
        Layer 1: window.G internal state (pokesurvivor.com / /dev.html)
        Layer 2: window.__psTelemetry (canvas hook for Doofcraft)
        Layer 3: DOM state & modals
        Layer 4: OpenCV perception (visual verification & fallback)
        """
        now = time.time()
        self.last_read_time = now

        # 1. Fetch Browser Internal & Telemetry State
        browser_data = self._fetch_browser_data()

        # 2. Extract Player Data
        player = self._extract_player(browser_data)

        # 3. Extract UI & Modals
        ui = self._extract_ui(browser_data)

        # 4. Extract Metrics (Kills, Time, Level, Coins)
        metrics = self._extract_metrics(browser_data)

        # 5. Extract Entities (Enemies, Hazards, Items, Specials, Map)
        entities = self._extract_entities(browser_data, frame_img, is_moving, player)

        unified_state = {
            "timestamp": now,
            "player": player,
            "enemies": entities["enemies"],
            "hazards": entities["hazards"],
            "items": entities["items"],
            "specials": entities["specials"],
            "terrain": entities["terrain"],
            "ui": ui,
            "metrics": metrics,
            "danger_level": entities["danger_level"],
            "is_encircled": entities["is_encircled"],
            "is_stuck": entities["is_stuck"],
            "sources": {
                "player": player.get("source"),
                "metrics": metrics.get("source"),
                "enemies": entities.get("source"),
                "hazards": entities.get("hazard_source")
            }
        }
        self.last_state = unified_state
        return unified_state

    def _fetch_browser_data(self) -> Dict[str, Any]:
        """Executes a single consolidated evaluate call in the page."""
        if not self.page:
            return {}

        try:
            return self.page.evaluate('''() => {
                const res = {
                    hasG: !!window.G,
                    gState: null,
                    telemetry: null,
                    dom: null,
                    results: null
                };

                // 1. window.G state
                if (window.G && window.G.Game && window.G.Game.player) {
                    try {
                        const P = window.G.Game.player;
                        const G = window.G;
                        res.gState = {
                            player: {
                                x: P.x,
                                y: P.y,
                                r: P.r,
                                hp: P.hp,
                                maxHp: P.maxHp,
                                dex: P.dex,
                                speed: (P.baseSpd || 100) * (P.spdMul || 1),
                                magnet: P.magnet,
                                moves: P.moves || [],
                                stats: P.stats || []
                            },
                            gameState: G.Game.state,
                            enemies: (G.EnemyMgr ? G.EnemyMgr.all() : []).filter(e => !e.dead).slice(0, 30).map(e => ({
                                x: e.x, y: e.y, r: e.r, dex: e.dex, hp: e.hp, maxHp: e.maxHp,
                                atkType: e.atkType, behavior: e.behavior, shiny: !!e.shiny,
                                boss: !!e.boss
                            })),
                            hazards: (G.Hazards ? G.Hazards.all : []).slice(0, 20).map(z => ({
                                kind: z.kind, x: z.x, y: z.y, r: z.r, angle: z.angle, len: z.len, w: z.w
                            })),
                            projectiles: (G.Projectiles ? G.Projectiles.all() : []).filter(p => p.friendly === false).slice(0, 20).map(p => ({
                                x: p.x, y: p.y, r: p.r, vx: p.vx, vy: p.vy
                            })),
                            pickups: (G.Pickups ? G.Pickups.all() : []).slice(0, 30).map(p => ({
                                x: p.x, y: p.y, kind: p.kind
                            })),
                            rift: (G.Rift && G.Rift.rift) ? { x: G.Rift.rift.x, y: G.Rift.rift.y, inArena: !!G.Rift.inArena } : null
                        };
                    } catch(e) {}
                }

                // 2. window.__psTelemetry (Canvas hook)
                if (window.__psTelemetry) {
                    try {
                        const tel = window.__psTelemetry;
                        res.telemetry = {
                            texts: tel.texts ? tel.texts.slice(-40) : [],
                            camera: tel.camera || { s: 1, e: 0, f: 0 },
                            spriteCount: tel.sprites ? tel.sprites.length : 0,
                            lastFrameAt: tel.lastFrameAt || 0
                        };
                        // Flush transient text buffer to prevent unbounded growth
                        if (tel.texts && tel.texts.length > 50) {
                            tel.texts.splice(0, tel.texts.length - 20);
                        }
                    } catch(e) {}
                }

                // 3. DOM state & Modals
                try {
                    const screensEl = document.querySelector('main.screens');
                    const activeScreen = screensEl ? screensEl.getAttribute('data-screen') : 'unknown';
                    const phaseEl = document.querySelector('[data-testid="game-phase"]');
                    const phase = phaseEl ? phaseEl.innerText.trim() : 'none';

                    const choiceEl = document.querySelector('[data-testid="choice"]') || document.querySelector('.ps-choice:not(.hidden) .ps-box');
                    const choiceOpen = choiceEl ? (choiceEl.offsetParent !== null && !choiceEl.classList.contains('hidden')) : false;

                    let modalKind = 'level-up';
                    let modalTitle = '';
                    let verifiedLevel = null;
                    const choices = [];

                    if (choiceOpen && choiceEl) {
                        const titleEl = choiceEl.querySelector('h2');
                        modalTitle = titleEl ? titleEl.innerText.trim() : '';
                        const titleLow = modalTitle.toLowerCase();

                        if (titleLow.includes('item') || titleLow.includes('hold only')) {
                            modalKind = 'item';
                        } else if (titleLow.includes('bond') || choiceEl.querySelector('.bond-title, .bond-hint')) {
                            modalKind = 'bond';
                        } else if (titleLow.includes('chest') || choiceEl.querySelector('.waves-chest')) {
                            modalKind = 'chest';
                        } else {
                            modalKind = 'level-up';
                            const lvlMatch = modalTitle.match(/Level\\s+(\\d+)/i);
                            if (lvlMatch) verifiedLevel = parseInt(lvlMatch[1]);
                        }

                        const cards = choiceEl.querySelectorAll('button.ps-card, [data-testid^="option-"], .cell, .option');
                        cards.forEach((card, idx) => {
                            const nameEl = card.querySelector('.nm, h3, .title, strong');
                            const tagEl = card.querySelector('.tag-chip, .badge, .tag');
                            const descEl = card.querySelector('.ds, .sub, .desc, p');
                            const keyEl = card.querySelector('.key .kb-only, .kb-only, div.key');

                            let name = '';
                            if (nameEl) {
                                name = Array.from(nameEl.childNodes).filter(n => n.nodeType === 3).map(n => n.textContent).join('').trim() || nameEl.innerText.split('\\n')[0].trim();
                            }
                            if (!name) name = card.innerText.split('\\n')[0].trim();

                            const tag = tagEl ? tagEl.innerText.trim() : '';
                            const desc = descEl ? descEl.innerText.trim() : '';
                            const key = keyEl ? keyEl.innerText.trim() : String(idx + 1);
                            const testid = card.getAttribute('data-testid') || `option-${idx}`;

                            choices.push({
                                index: idx,
                                name: name || `Option ${idx+1}`,
                                tag: tag,
                                desc: desc,
                                description: desc,
                                key: key,
                                testid: testid,
                                selector: `[data-testid="${testid}"]`
                            });
                        });
                    }

                    const exitBtn = document.querySelector('[data-testid="game-exit"]');
                    const exitAvailable = exitBtn ? exitBtn.offsetParent !== null : false;

                    res.dom = {
                        activeScreen,
                        phase,
                        choiceOpen,
                        modalKind,
                        modalTitle,
                        verifiedLevel,
                        choices,
                        exitAvailable
                    };
                } catch(e) {}

                // 4. Match Results (if finished)
                try {
                    const resultsEl = document.querySelector('[data-testid="results"]') || document.querySelector('.ps-results');
                    if (resultsEl && resultsEl.offsetParent !== null) {
                        const killsEl = resultsEl.querySelector('[data-testid="run-stat-kills"] b');
                        const levelEl = resultsEl.querySelector('[data-testid="run-stat-level"] b');
                        const damageEl = resultsEl.querySelector('[data-testid="run-stat-damage"] b');
                        const downsEl = resultsEl.querySelector('[data-testid="run-stat-downs"] b');

                        let kills = killsEl ? parseInt(killsEl.innerText.replace(/[^\\d]/g, '')) : null;
                        let level = levelEl ? parseInt(levelEl.innerText.replace(/[^\\d]/g, '')) : null;
                        let damage = damageEl ? parseInt(damageEl.innerText.replace(/[^\\d]/g, '')) : null;
                        let downs = downsEl ? parseInt(downsEl.innerText.replace(/[^\\d]/g, '')) : null;

                        // Parse statsMap for time, coins, bosses
                        const statRows = Array.from(resultsEl.querySelectorAll('.stats > div'));
                        let timeStr = '';
                        let coins = null;
                        let bosses = null;

                        for (let row of statRows) {
                            const span = row.querySelector('span');
                            const b = row.querySelector('b');
                            if (span && b) {
                                const k = span.innerText.trim().toLowerCase();
                                const val = b.innerText.trim();
                                if (k.includes('time')) timeStr = val;
                                if (k.includes('defeated') && kills === null) kills = parseInt(val.replace(/[^\\d]/g, ''));
                                if (k.includes('level') && level === null) level = parseInt(val.replace(/[^\\d]/g, ''));
                                if (k.includes('coin')) coins = parseInt(val.replace(/[^\\d]/g, ''));
                                if (k.includes('boss')) bosses = parseInt(val.replace(/[^\\d]/g, ''));
                            }
                        }

                        res.results = {
                            available: true,
                            kills: kills,
                            level: level,
                            time_str: timeStr,
                            coins: coins,
                            bosses: bosses,
                            damage: damage,
                            downs: downs
                        };
                    }
                } catch(e) {}

                return res;
            }''')
        except Exception:
            return {}

    def _extract_player(self, browser_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts player attributes, combining window.G, canvas texts, and OpenCV."""
        # 1. From window.G
        g_player = browser_data.get("gState", {}).get("player")
        if g_player:
            hp = g_player.get("hp", 100)
            max_hp = g_player.get("maxHp", 100)
            return {
                "world_x": g_player.get("x", 0.0),
                "world_y": g_player.get("y", 0.0),
                "hp": hp,
                "max_hp": max_hp,
                "hp_ratio": max(0.0, min(1.0, hp / max(1, max_hp))),
                "dex": g_player.get("dex"),
                "speed": g_player.get("speed", 100.0),
                "magnet": g_player.get("magnet", 50.0),
                "moves": g_player.get("moves", []),
                "stats": g_player.get("stats", []),
                "source": "window_G"
            }

        # 2. From Canvas telemetry texts (e.g. Doofcraft "182 / 182", "Tapu Fini", "Lv. 1")
        telemetry = browser_data.get("telemetry", {})
        texts = telemetry.get("texts", [])
        hp_ratio = None
        current_hp = None
        max_hp = None
        player_name = None
        level = None

        for t in texts:
            # Match HP format "182 / 182"
            hp_m = re.match(r"^(\d+)\s*/\s*(\d+)$", t)
            if hp_m:
                current_hp = int(hp_m.group(1))
                max_hp = int(hp_m.group(2))
                if max_hp > 0:
                    hp_ratio = current_hp / max_hp

            # Match Level "Lv. 1"
            lvl_m = re.match(r"^Lv\.?\s*(\d+)$", t, re.IGNORECASE)
            if lvl_m:
                level = int(lvl_m.group(1))

        # 3. Fallback to OpenCV HP ratio if text is not found
        source = "canvas_telemetry_texts" if hp_ratio is not None else "opencv_fallback"
        if hp_ratio is None:
            hp_ratio = self.perception.last_hp_result.get("hp_ratio", 1.0)

        cam = telemetry.get("camera", {"s": 1.0, "e": 0.0, "f": 0.0})
        # Screen center = 640, 360 in 1280x720
        world_x = (640.0 - cam.get("e", 0.0)) / max(0.01, cam.get("s", 1.0))
        world_y = (360.0 - cam.get("f", 0.0)) / max(0.01, cam.get("s", 1.0))

        return {
            "world_x": world_x,
            "world_y": world_y,
            "hp": current_hp,
            "max_hp": max_hp,
            "hp_ratio": hp_ratio,
            "level": level,
            "dex": None,
            "speed": 100.0,
            "magnet": 50.0,
            "source": source
        }

    def _extract_ui(self, browser_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts UI phase and choice dialog status."""
        dom = browser_data.get("dom", {})
        return {
            "phase": dom.get("phase", "none"),
            "active_screen": dom.get("activeScreen", "unknown"),
            "choice_open": dom.get("choiceOpen", False),
            "modal_kind": dom.get("modalKind", "level-up"),
            "modal_title": dom.get("modalTitle", ""),
            "verified_level": dom.get("verifiedLevel"),
            "choices": dom.get("choices", []),
            "exit_available": dom.get("exitAvailable", False)
        }

    def _extract_metrics(self, browser_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts genuine match results or HUD counters."""
        results = browser_data.get("results")
        if results and results.get("available"):
            return {
                "kills": results.get("kills"),
                "time_str": results.get("time_str"),
                "coins": results.get("coins"),
                "level": results.get("level"),
                "bosses": results.get("bosses"),
                "damage": results.get("damage"),
                "downs": results.get("downs"),
                "is_match_over": True,
                "source": "dom_results"
            }

        # Otherwise read live counters from Canvas telemetry texts
        telemetry = browser_data.get("telemetry", {})
        texts = telemetry.get("texts", [])
        kills = None
        coins = None
        time_str = ""

        for t in texts:
            # Match "SHADY FOREST · 0 DEFEATED · 0 COINS" or "0 DEFEATED"
            def_m = re.search(r"(\d+)\s+DEFEATED", t, re.IGNORECASE)
            if def_m:
                kills = int(def_m.group(1))
            coin_m = re.search(r"(\d+)\s+COINS", t, re.IGNORECASE)
            if coin_m:
                coins = int(coin_m.group(1))
            time_m = re.match(r"^(\d+:\d\d)$", t)
            if time_m:
                time_str = time_m.group(1)

        return {
            "kills": kills,
            "time_str": time_str,
            "coins": coins,
            "level": None,
            "bosses": None,
            "damage": None,
            "downs": None,
            "is_match_over": False,
            "source": "canvas_hud_texts" if kills is not None else "unobserved"
        }

    def _extract_entities(
        self,
        browser_data: Dict[str, Any],
        frame_img: Optional[Any],
        is_moving: bool,
        player: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Harmonizes entities from window.G, canvas telemetry, and OpenCV."""
        g_state = browser_data.get("gState")
        
        # Branch 1: Complete window.G available
        if g_state:
            enemies = []
            px, py = player["world_x"], player["world_y"]
            for e in g_state.get("enemies", []):
                dx = e["x"] - px
                dy = e["y"] - py
                dist = math.hypot(dx, dy)
                deg = math.degrees(math.atan2(dy, dx)) % 360
                sector = self._angle_to_sector(deg)
                enemies.append({
                    "x": e["x"],
                    "y": e["y"],
                    "dist": dist,
                    "sector": sector,
                    "radius": e.get("r", 16),
                    "type": e.get("dex"),
                    "is_boss": e.get("boss", False),
                    "is_shiny": e.get("shiny", False),
                    "is_thief": e.get("behavior") == "thief",
                    "hp": e.get("hp"),
                    "source": "window_G"
                })

            hazards = []
            for z in g_state.get("hazards", []):
                hazards.append({
                    "kind": z.get("kind"),
                    "x": z.get("x"),
                    "y": z.get("y"),
                    "dist": math.hypot(z.get("x", 0) - px, z.get("y", 0) - py),
                    "radius": z.get("r", 20),
                    "angle": z.get("angle", 0.0),
                    "source": "window_G"
                })
            for p in g_state.get("projectiles", []):
                hazards.append({
                    "kind": "shot",
                    "x": p.get("x"),
                    "y": p.get("y"),
                    "dist": math.hypot(p.get("x", 0) - px, p.get("y", 0) - py),
                    "vx": p.get("vx", 0.0),
                    "vy": p.get("vy", 0.0),
                    "source": "window_G"
                })

            items = []
            for it in g_state.get("pickups", []):
                items.append({
                    "x": it.get("x"),
                    "y": it.get("y"),
                    "dist": math.hypot(it.get("x", 0) - px, it.get("y", 0) - py),
                    "kind": it.get("kind", "xp"),
                    "source": "window_G"
                })

            specials = {
                "rift": g_state.get("rift"),
                "spring": None,
                "altar": None,
                "chest": None
            }

            return {
                "enemies": enemies,
                "hazards": hazards,
                "items": items,
                "specials": specials,
                "terrain": {"sector_obstacles": {}, "sector_obs_costs": {}},
                "danger_level": "HIGH" if len(hazards) > 0 else "LOW",
                "is_encircled": len([e for e in enemies if e["dist"] < 150]) >= 6,
                "is_stuck": False,
                "source": "window_G",
                "hazard_source": "window_G"
            }

        # Branch 2: OpenCV Visual Perception (Rock-Solid for Doofcraft & Generic Canvas)
        if frame_img is not None:
            vision = self.perception.analyze_frame(frame_img, is_moving=is_moving)
            specials = {
                "rift": vision.get("rift_target"),
                "spring": None,
                "altar": None,
                "chest": None
            }
            return {
                "enemies": vision.get("enemies", []),
                "hazards": vision.get("hazards", []),
                "items": vision.get("gems", []),
                "specials": specials,
                "terrain": {
                    "sector_obstacles": vision.get("sector_obstacles", {}),
                    "sector_obs_costs": vision.get("sector_obs_costs", {})
                },
                "danger_level": vision.get("danger_level", "LOW"),
                "is_encircled": vision.get("is_encircled", False),
                "is_stuck": vision.get("is_stuck", False),
                "source": "opencv_perception",
                "hazard_source": "opencv_hazards"
            }

        # Empty fallback
        return {
            "enemies": [],
            "hazards": [],
            "items": [],
            "specials": {},
            "terrain": {"sector_obstacles": {}, "sector_obs_costs": {}},
            "danger_level": "LOW",
            "is_encircled": False,
            "is_stuck": False,
            "source": "empty",
            "hazard_source": "empty"
        }

    def _angle_to_sector(self, angle_deg: float) -> str:
        """Converts angle in degrees [0, 360) to 8 cardinal sectors."""
        sectors = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
        idx = int((angle_deg + 22.5) // 45) % 8
        return sectors[idx]

