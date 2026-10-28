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
  // Prevent double-installation
  if (window.__psHookInstalled) return;
  window.__psHookInstalled = true;

  // 1. One-shot setter hook for window.G on pokesurvivor.com / /dev.html
  // (Reference: PokeSurvivorTool sites/default/hook.js · MIT License)
  if (!window.G && !Object.getOwnPropertyDescriptor(Object.prototype, 'Game')) {
    const trap = {
      configurable: true,
      get() { return undefined; },
      set(v) {
        Object.defineProperty(this, 'Game', { value: v, writable: true, enumerable: true, configurable: true });
        const target = (v && typeof v === 'object' && ('BUILT' in v || 'LIFT' in v || 'player' in v)) ? v :
                       (this && typeof this === 'object' && ('BUILT' in this || 'LIFT' in this || 'player' in this)) ? this : null;
        if (target) {
          if (Object.getOwnPropertyDescriptor(Object.prototype, 'Game')?.set === trap.set) {
            delete Object.prototype.Game;
          }
          window.G = target;
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
  // Strictly hooks only CanvasRenderingContext2D.prototype.drawImage and fillText
  window.__psTelemetry = {
    sprites: [],
    texts: [],
    camera: { s: 1.0, e: 0.0, f: 0.0 },
    lastFrameAt: performance.now(),
    frameCount: 0
  };

  const P = CanvasRenderingContext2D.prototype;
  const origDrawImage = P.drawImage;
  const origFillText = P.fillText;

  // Intercept Canvas drawImage to capture camera transform & sprites
  P.drawImage = function(img, ...args) {
    try {
      if (this.canvas && (this.canvas.classList?.contains('ps-canvas') || this.canvas.id === 'game-canvas' || this.canvas.tagName === 'CANVAS')) {
        const m = this.getTransform ? this.getTransform() : { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
        window.__psTelemetry.camera = { s: m.a, e: m.e, f: m.f };

        const src = img ? (img.src || (img.tagName ? img.tagName : 'canvas')) : '';
        const dw = args.length === 8 ? args[6] : args.length === 4 ? args[2] : (img ? (img.width || 0) : 0);
        const dh = args.length === 8 ? args[7] : args.length === 4 ? args[3] : (img ? (img.height || 0) : 0);
        const dx = args.length >= 4 ? args[args.length === 8 ? 4 : 0] : 0;
        const dy = args.length >= 4 ? args[args.length === 8 ? 5 : 1] : 0;
        const cx = m.a * (dx + dw / 2) + m.e;
        const cy = m.d * (dy + dh / 2) + m.f;

        if (window.__psTelemetry.sprites.length < 150) {
          window.__psTelemetry.sprites.push({
            src: String(src).slice(-60),
            cx: cx,
            cy: cy,
            w: dw * m.a,
            h: dh * m.d
          });
        }
        window.__psTelemetry.lastFrameAt = performance.now();
      }
    } catch(e) {}
    return origDrawImage.apply(this, arguments);
  };

  // Intercept fillText for live HUD information (HP, level, kills, coins, time)
  P.fillText = function(text, ...args) {
    try {
      if (this.canvas && (this.canvas.classList?.contains('ps-canvas') || this.canvas.id === 'game-canvas' || this.canvas.tagName === 'CANVAS')) {
        const str = String(text).trim();
        if (str && window.__psTelemetry.texts.length < 100) {
          window.__psTelemetry.texts.push(str);
        }
      }
    } catch(e) {}
    return origFillText.apply(this, arguments);
  };
})();
"""

class GameStateAdapter:
    def __init__(self, page: Optional[Page] = None, perception: Optional[PerceptionEngine] = None):
        self.page = page
        self.perception = perception or PerceptionEngine()
        self.last_state: Dict[str, Any] = {}
        self.last_read_time = 0.0
        self.last_cached_entities: Optional[Dict[str, Any]] = None
        self.last_frame_time: float = 0.0
        self.max_frame_staleness: float = 0.85  # Cached vision entities valid up to 850ms between captures

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

        hp_ratio = player.get("hp_ratio")
        rift_target = entities.get("specials", {}).get("rift")

        px = float(player.get("world_x", 640.0))
        py = float(player.get("world_y", 360.0))
        player["pos"] = (px, py)

        confidence = 0.98 if player.get("source") == "window_G" else (
            0.90 if player.get("source") == "canvas_telemetry_texts" else (
                0.80 if player.get("source") == "opencv_fallback" else 0.50
            )
        )

        unified_state = {
            "timestamp": now,
            "player": player,
            "player_pos": (px, py),
            "enemies": entities["enemies"],
            "hazards": entities["hazards"],
            "items": entities["items"],
            "specials": entities["specials"],
            "terrain": entities["terrain"],
            "ui": ui,
            "metrics": metrics,
            "danger_level": entities.get("danger_level", "LOW"),
            "is_encircled": entities.get("is_encircled", False),
            "is_stuck": entities.get("is_stuck", False),
            # Direct tactical shortcuts for MovementPolicy & TargetEvaluator
            "hp_ratio": hp_ratio,
            "rift_target": rift_target,
            "items_on_ground": entities.get("items_on_ground", []),
            "sector_dangers": entities.get("sector_dangers", {}),
            "sector_enemy_counts": entities.get("sector_enemy_counts", {}),
            "sector_obstacles": entities.get("terrain", {}).get("sector_obstacles", {}),
            "sector_obs_costs": entities.get("terrain", {}).get("sector_obs_costs", {}),
            "min_enemy_dist": entities.get("min_enemy_dist", 9999.0),
            "nearest_enemy_sector": entities.get("nearest_enemy_sector"),
            "swarm_center": entities.get("swarm_center"),
            "gem_cluster_sector": entities.get("gem_cluster_sector"),
            "best_gem_vector": entities.get("best_gem_vector"),
            "boss_info": entities.get("boss_info", {}),
            "confidence": confidence,
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
                const G = window.G;
                let P = null;
                let container = G;
                if (G) {
                    if (G.player) {
                        P = G.player;
                    } else if (G.Game && G.Game.player) {
                        P = G.Game.player;
                        container = G.Game;
                    }
                }
                if (P) {
                    try {
                        const enemyMgr = G.EnemyMgr || (G.Game ? G.Game.EnemyMgr : null) || (container ? container.EnemyMgr : null);
                        const hazardsMgr = G.Hazards || (G.Game ? G.Game.Hazards : null) || (container ? container.Hazards : null);
                        const projMgr = G.Projectiles || (G.Game ? G.Game.Projectiles : null) || (container ? container.Projectiles : null);
                        const pickupMgr = G.Pickups || (G.Game ? G.Game.Pickups : null) || (container ? container.Pickups : null);
                        const riftMgr = G.Rift || (G.Game ? G.Game.Rift : null) || (container ? container.Rift : null);
                        const mapMgr = G.Map || (G.Game ? G.Game.Map : null) || (container ? container.Map : null);
                        let mapBounds = null;
                        if (mapMgr) {
                            mapBounds = {
                                w: mapMgr.w || mapMgr.width || 3000,
                                h: mapMgr.h || mapMgr.height || 3000,
                                minX: (mapMgr.minX !== undefined) ? mapMgr.minX : 0,
                                minY: (mapMgr.minY !== undefined) ? mapMgr.minY : 0,
                                maxX: mapMgr.maxX || mapMgr.w || mapMgr.width || 3000,
                                maxY: mapMgr.maxY || mapMgr.h || mapMgr.height || 3000
                            };
                        }

                        const getList = (mgr) => {
                            if (!mgr) return [];
                            if (typeof mgr.all === 'function') {
                                try { return mgr.all(); } catch(e) { return []; }
                            }
                            if (Array.isArray(mgr.all)) return mgr.all;
                            if (Array.isArray(mgr)) return mgr;
                            return [];
                        };

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
                            gameState: (G.Game ? G.Game.state : G.state) || 'running',
                            enemies: getList(enemyMgr).filter(e => !e.dead).slice(0, 30).map(e => ({
                                x: e.x, y: e.y, r: e.r, dex: e.dex, hp: e.hp, maxHp: e.maxHp,
                                atkType: e.atkType, behavior: e.behavior, shiny: !!e.shiny,
                                boss: !!e.boss
                            })),
                            hazards: getList(hazardsMgr).slice(0, 20).map(z => ({
                                kind: z.kind, x: z.x, y: z.y, r: z.r, angle: z.angle, len: z.len, w: z.w
                            })),
                            projectiles: getList(projMgr).filter(p => p.friendly === false).slice(0, 20).map(p => ({
                                x: p.x, y: p.y, r: p.r, vx: p.vx, vy: p.vy
                            })),
                            pickups: getList(pickupMgr).slice(0, 30).map(p => ({
                                x: p.x, y: p.y, kind: p.kind
                            })),
                            rift: (riftMgr && riftMgr.rift) ? { x: riftMgr.rift.x, y: riftMgr.rift.y, inArena: !!riftMgr.inArena } : null,
                            mapBounds: mapBounds
                        };
                    } catch(e) {}
                }

                // 2. window.__psTelemetry (Canvas hook)
                if (window.__psTelemetry) {
                    try {
                        const tel = window.__psTelemetry;
                        // Copy and atomically FLUSH active buffers so old texts/sprites never linger!
                        const texts = tel.texts ? tel.texts.slice(0) : [];
                        tel.texts = []; // Flush on read!
                        const sprites = tel.sprites ? tel.sprites.slice(0) : [];
                        tel.sprites = []; // Flush on read!
                        res.telemetry = {
                            texts: texts,
                            camera: tel.camera || { s: 1, e: 0, f: 0 },
                            spriteCount: sprites.length,
                            lastFrameAt: tel.lastFrameAt || 0
                        };
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

                        let timeSeconds = null;
                        if (timeStr) {
                            const parts = timeStr.split(':');
                            if (parts.length === 2) {
                                const mins = parseInt(parts[0]);
                                const secs = parseInt(parts[1]);
                                if (!isNaN(mins) && !isNaN(secs)) {
                                    timeSeconds = mins * 60 + secs;
                                }
                            } else {
                                const parsed = parseFloat(timeStr);
                                if (!isNaN(parsed)) timeSeconds = parsed;
                            }
                        }

                        res.results = {
                            available: true,
                            kills: kills,
                            level: level,
                            time_str: timeStr,
                            time_seconds: timeSeconds,
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
        """Extracts player attributes, combining window.G, canvas texts, and OpenCV fallback."""
        # 1. From window.G
        g_state = browser_data.get("gState") or {}
        g_player = g_state.get("player")
        if g_player:
            hp = g_player.get("hp")
            max_hp = g_player.get("maxHp")
            hp_ratio = None
            if hp is not None and max_hp and max_hp > 0:
                hp_ratio = max(0.0, min(1.0, float(hp) / float(max_hp)))
            return {
                "world_x": float(g_player.get("x", 0.0)),
                "world_y": float(g_player.get("y", 0.0)),
                "hp": hp,
                "max_hp": max_hp,
                "hp_ratio": hp_ratio,
                "dex": g_player.get("dex"),
                "speed": g_player.get("speed", 100.0),
                "magnet": g_player.get("magnet", 50.0),
                "moves": g_player.get("moves", []),
                "stats": g_player.get("stats", []),
                "source": "window_G"
            }

        # 2. From Canvas telemetry texts (e.g. Doofcraft "182 / 182", "Tapu Fini", "Lv. 1")
        telemetry = browser_data.get("telemetry") or {}
        texts = telemetry.get("texts") or []
        hp_ratio = None
        current_hp = None
        max_hp = None
        level = None

        for t in texts:
            # Match HP format "182 / 182"
            hp_m = re.match(r"^(\d+)\s*/\s*(\d+)$", t)
            if hp_m:
                current_hp = int(hp_m.group(1))
                max_hp = int(hp_m.group(2))
                if max_hp > 0:
                    hp_ratio = max(0.0, min(1.0, current_hp / max_hp))

            # Match Level "Lv. 1"
            lvl_m = re.match(r"^Lv\.?\s*(\d+)$", t, re.IGNORECASE)
            if lvl_m:
                level = int(lvl_m.group(1))

        # 3. Fallback to OpenCV HP ratio if text is not found
        if hp_ratio is not None:
            source = "canvas_telemetry_texts"
        else:
            last_hp = getattr(self.perception, "last_hp_result", {})
            if last_hp.get("hp_ratio") is not None and last_hp.get("status") in ("VERIFIED", "APPROXIMATE"):
                hp_ratio = float(last_hp["hp_ratio"])
                source = "opencv_fallback"
            else:
                hp_ratio = None
                source = "unobserved"

        cam = telemetry.get("camera") or {"s": 1.0, "e": 0.0, "f": 0.0}
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
        """Extracts UI phase and choice dialog status, with fallback to gState.gameState."""
        dom = browser_data.get("dom") or {}
        g_state = browser_data.get("gState") or {}

        phase = dom.get("phase", "none")
        if phase in ("none", "unknown", "", None):
            g_phase = g_state.get("gameState")
            if g_phase:
                g_str = str(g_phase).lower()
                if g_str in ("running", "play", "playing", "battle"):
                    phase = "running"
                elif g_str in ("gameover", "ended", "death", "victory"):
                    phase = "ended"
                elif g_str in ("choice", "levelup", "pause", "paused"):
                    phase = "choice"
                else:
                    phase = g_str

        return {
            "phase": phase,
            "active_screen": dom.get("activeScreen", "unknown"),
            "choice_open": dom.get("choiceOpen", False),
            "modal_kind": dom.get("modalKind", "level-up"),
            "modal_title": dom.get("modalTitle", ""),
            "verified_level": dom.get("verifiedLevel"),
            "choices": dom.get("choices", []),
            "exit_available": dom.get("exitAvailable", False)
        }

    def _extract_metrics(self, browser_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts genuine match results or HUD counters.
        Returns None/null when fields are not observed; never fabricates numbers.
        Includes available: True and time_seconds for seamless StateTracker.end_run compatibility.
        """
        results = browser_data.get("results")
        if results and results.get("available"):
            time_seconds = results.get("time_seconds")
            time_str = results.get("time_str")
            if time_seconds is None and time_str:
                parts = str(time_str).split(":")
                if len(parts) == 2:
                    try:
                        time_seconds = float(int(parts[0]) * 60 + int(parts[1]))
                    except ValueError:
                        pass
                else:
                    try:
                        time_seconds = float(time_str)
                    except ValueError:
                        pass

            return {
                "available": True,
                "kills": results.get("kills"),
                "time_str": time_str,
                "time_seconds": time_seconds,
                "coins": results.get("coins"),
                "level": results.get("level"),
                "bosses": results.get("bosses"),
                "damage": results.get("damage"),
                "downs": results.get("downs"),
                "is_match_over": True,
                "source": "dom_results"
            }

        # Otherwise read live counters from Canvas telemetry texts
        telemetry = browser_data.get("telemetry") or {}
        texts = telemetry.get("texts") or []
        kills = None
        coins = None
        time_str = None
        time_seconds = None

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
                parts = time_str.split(":")
                try:
                    time_seconds = float(int(parts[0]) * 60 + int(parts[1]))
                except ValueError:
                    pass

        source = "canvas_hud_texts" if (kills is not None or coins is not None or time_seconds is not None) else "unobserved"
        return {
            "available": (kills is not None or coins is not None or time_seconds is not None),
            "kills": kills,
            "time_str": time_str,
            "time_seconds": time_seconds,
            "coins": coins,
            "level": None,
            "bosses": None,
            "damage": None,
            "downs": None,
            "is_match_over": False,
            "source": source
        }

    def _extract_entities(
        self,
        browser_data: Dict[str, Any],
        frame_img: Optional[Any],
        is_moving: bool,
        player: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Harmonizes entities and tactical metrics from window.G, canvas telemetry, and OpenCV."""
        g_state = browser_data.get("gState")

        # Branch 1: Complete window.G available
        if g_state:
            enemies = []
            px, py = player["world_x"], player["world_y"]
            sector_enemy_counts = {s: 0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
            min_enemy_dist = 9999.0
            nearest_enemy_sec = None
            swarm_x, swarm_y, swarm_count = 0.0, 0.0, 0

            for e in g_state.get("enemies", []):
                dx = e["x"] - px
                dy = e["y"] - py
                dist = math.hypot(dx, dy)
                deg = math.degrees(math.atan2(dy, dx)) % 360
                sector = self._angle_to_sector(deg)
                if sector in sector_enemy_counts:
                    sector_enemy_counts[sector] += 1
                if dist < min_enemy_dist:
                    min_enemy_dist = dist
                    nearest_enemy_sec = sector
                if dist < 340.0:
                    swarm_x += e["x"]
                    swarm_y += e["y"]
                    swarm_count += 1

                enemies.append({
                    "x": e["x"],
                    "y": e["y"],
                    "dist": dist,
                    "sector": sector,
                    "radius": e.get("r", 16),
                    "type": e.get("dex"),
                    "dex": e.get("dex"),
                    "is_boss": e.get("boss", False),
                    "is_shiny": e.get("shiny", False),
                    "is_thief": e.get("behavior") == "thief" or e.get("dex") in [1000, 52],
                    "hp": e.get("hp"),
                    "source": "window_G"
                })

            swarm_center = (swarm_x / swarm_count, swarm_y / swarm_count) if swarm_count > 0 else None

            # Terrain calculation from map bounds (if available)
            sector_obstacles = {s: False for s in ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]}
            sector_obs_costs = {s: 0.0 for s in ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]}
            map_bounds = g_state.get("mapBounds")
            terrain_source = "window_G_bounds" if map_bounds else "unobserved"

            if map_bounds:
                min_x = float(map_bounds.get("minX", 0.0))
                max_x = float(map_bounds.get("maxX", 3000.0))
                min_y = float(map_bounds.get("minY", 0.0))
                max_y = float(map_bounds.get("maxY", 3000.0))
                wall_buffer = 180.0

                if px - min_x < wall_buffer:
                    dist_to_wall = max(1.0, px - min_x)
                    cost = max(0.0, (wall_buffer - dist_to_wall) * 4.0)
                    for s in ["W", "NW", "SW"]:
                        sector_obstacles[s] = True
                        sector_obs_costs[s] = max(sector_obs_costs[s], cost)
                if max_x - px < wall_buffer:
                    dist_to_wall = max(1.0, max_x - px)
                    cost = max(0.0, (wall_buffer - dist_to_wall) * 4.0)
                    for s in ["E", "NE", "SE"]:
                        sector_obstacles[s] = True
                        sector_obs_costs[s] = max(sector_obs_costs[s], cost)
                if py - min_y < wall_buffer:
                    dist_to_wall = max(1.0, py - min_y)
                    cost = max(0.0, (wall_buffer - dist_to_wall) * 4.0)
                    for s in ["N", "NW", "NE"]:
                        sector_obstacles[s] = True
                        sector_obs_costs[s] = max(sector_obs_costs[s], cost)
                if max_y - py < wall_buffer:
                    dist_to_wall = max(1.0, max_y - py)
                    cost = max(0.0, (wall_buffer - dist_to_wall) * 4.0)
                    for s in ["S", "SW", "SE"]:
                        sector_obstacles[s] = True
                        sector_obs_costs[s] = max(sector_obs_costs[s], cost)

            hazards = []
            for z in g_state.get("hazards", []):
                zx = float(z.get("x", 0.0))
                zy = float(z.get("y", 0.0))
                z_dist = math.hypot(zx - px, zy - py)
                z_deg = math.degrees(math.atan2(zy - py, zx - px)) % 360
                z_sec = self._angle_to_sector(z_deg)
                hazards.append({
                    "kind": z.get("kind"),
                    "type": "telegraph",
                    "x": zx,
                    "y": zy,
                    "dist": z_dist,
                    "sector": z_sec,
                    "radius": z.get("r", 20),
                    "angle": z.get("angle", 0.0),
                    "is_imminent": z_dist < 110.0,
                    "source": "window_G"
                })
            for p in g_state.get("projectiles", []):
                px_b = float(p.get("x", 0.0))
                py_b = float(p.get("y", 0.0))
                dx = px_b - px
                dy = py_b - py
                p_dist = math.hypot(dx, dy)
                p_deg = math.degrees(math.atan2(dy, dx)) % 360
                p_sec = self._angle_to_sector(p_deg)
                vx = float(p.get("vx", 0.0))
                vy = float(p.get("vy", 0.0))
                v_sq = vx * vx + vy * vy
                ttc = 99.0
                dca = p_dist
                is_imminent = False
                if v_sq > 1e-4:
                    rx = -dx
                    ry = -dy
                    t_star = (rx * vx + ry * vy) / v_sq
                    if t_star > 0:
                        ttc = t_star
                        rel_x = rx - vx * t_star
                        rel_y = ry - vy * t_star
                        dca = math.hypot(rel_x, rel_y)
                        if ttc < 1.2 and dca < 55.0:
                            is_imminent = True
                hazards.append({
                    "kind": "shot",
                    "type": "bullet",
                    "x": px_b,
                    "y": py_b,
                    "dist": p_dist,
                    "sector": p_sec,
                    "vx": vx,
                    "vy": vy,
                    "ttc": ttc,
                    "dca": dca,
                    "is_imminent": is_imminent,
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

            close_threats = [e for e in enemies if e["dist"] < 175.0]
            is_encircled = len(close_threats) >= 7 and len({e["sector"] for e in close_threats}) >= 4

            # Sector danger profiles from window.G
            sector_dangers = {}
            for s, cnt in sector_enemy_counts.items():
                sector_dangers[s] = cnt * 80.0

            danger_level = "CRITICAL" if (min_enemy_dist < 120.0 or is_encircled) else (
                "HIGH" if (min_enemy_dist < 200.0 or len(hazards) > 0) else (
                    "MEDIUM" if min_enemy_dist < 280.0 else "LOW"
                )
            )

            res_g = {
                "enemies": enemies,
                "hazards": hazards,
                "items": items,
                "items_on_ground": items,
                "specials": specials,
                "terrain": {
                    "sector_obstacles": sector_obstacles,
                    "sector_obs_costs": sector_obs_costs,
                    "source": terrain_source
                },
                "sector_dangers": sector_dangers,
                "sector_enemy_counts": sector_enemy_counts,
                "min_enemy_dist": min_enemy_dist,
                "nearest_enemy_sector": nearest_enemy_sec,
                "swarm_center": swarm_center,
                "gem_cluster_sector": None,
                "best_gem_vector": None,
                "boss_info": {},
                "danger_level": danger_level,
                "is_encircled": is_encircled,
                "is_stuck": False,
                "source": "window_G",
                "hazard_source": "window_G"
            }
            self.last_cached_entities = res_g
            self.last_frame_time = time.time()
            return res_g

        # Branch 2: OpenCV Visual Perception (Doofcraft & Generic Canvas)
        if frame_img is not None:
            vision = self.perception.analyze_frame(frame_img, is_moving=is_moving)
            specials = {
                "rift": vision.get("rift_target"),
                "spring": None,
                "altar": None,
                "chest": None
            }
            items = list(vision.get("gems", [])) + list(vision.get("items_on_ground", []))
            res_vision = {
                "enemies": vision.get("enemies", []),
                "hazards": vision.get("hazards", []),
                "items": items,
                "items_on_ground": vision.get("items_on_ground", []),
                "specials": specials,
                "terrain": {
                    "sector_obstacles": vision.get("sector_obstacles", {}),
                    "sector_obs_costs": vision.get("sector_obs_costs", {})
                },
                "sector_dangers": vision.get("sector_dangers", {}),
                "sector_enemy_counts": vision.get("sector_enemy_counts", {}),
                "min_enemy_dist": vision.get("min_enemy_dist", 9999.0),
                "nearest_enemy_sector": vision.get("nearest_enemy_sector"),
                "swarm_center": vision.get("swarm_center"),
                "gem_cluster_sector": vision.get("gem_cluster_sector"),
                "best_gem_vector": vision.get("best_gem_vector"),
                "boss_info": vision.get("boss_info", {}),
                "danger_level": vision.get("danger_level", "LOW"),
                "is_encircled": vision.get("is_encircled", False),
                "is_stuck": vision.get("is_stuck", False),
                "source": "opencv_perception",
                "hazard_source": "opencv_hazards"
            }
            self.last_cached_entities = res_vision
            self.last_frame_time = time.time()
            return res_vision

        # Branch 3: Cached Vision Entities across skipped visual frames
        now = time.time()
        if self.last_cached_entities is not None and (now - self.last_frame_time) <= self.max_frame_staleness:
            cached = dict(self.last_cached_entities)
            orig_src = self.last_cached_entities.get("source", "opencv")
            cached["source"] = f"{orig_src}_cached"
            cached["cached_at"] = self.last_frame_time
            cached["staleness"] = round(now - self.last_frame_time, 3)
            return cached

        # Empty fallback (expired or unobserved)
        return {
            "enemies": [],
            "hazards": [],
            "items": [],
            "items_on_ground": [],
            "specials": {"rift": None, "spring": None, "altar": None, "chest": None},
            "terrain": {"sector_obstacles": {}, "sector_obs_costs": {}, "source": "unobserved"},
            "sector_dangers": {},
            "sector_enemy_counts": {s: 0 for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]},
            "min_enemy_dist": 9999.0,
            "nearest_enemy_sector": None,
            "swarm_center": None,
            "gem_cluster_sector": None,
            "best_gem_vector": None,
            "boss_info": {},
            "danger_level": "LOW",
            "is_encircled": False,
            "is_stuck": False,
            "source": "unobserved_expired",
            "hazard_source": "unobserved_expired"
        }

    def _angle_to_sector(self, angle_deg: float) -> str:
        """Converts angle in degrees [0, 360) to 8 cardinal sectors."""
        sectors = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
        idx = int((angle_deg + 22.5) // 45) % 8
        return sectors[idx]
