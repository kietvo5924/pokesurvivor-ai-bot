import time
import base64
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from playwright.sync_api import Page

class GameObserver:
    def __init__(self, page: Page):
        self.page = page

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Captures gameplay canvas via base64 toDataURL.
        This runs in ~13ms, causes ZERO DOM layout recalculation,
        and completely eliminates screen pulsing/resizing jitter.
        """
        if not self.page:
            return None
        try:
            data_url = self.page.evaluate('''() => {
                const c = document.querySelector('canvas.ps-canvas');
                if (!c) return null;
                return c.toDataURL('image/jpeg', 0.6);
            }''')

            if data_url and ',' in data_url:
                img_b64 = data_url.split(',', 1)[1]
                raw_bytes = base64.b64decode(img_b64)
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img_bgr

            # Fallback to page screenshot if canvas not directly accessible
            img_bytes = self.page.screenshot()
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def get_selected_partner(self) -> Dict[str, Any]:
        """Reads selected partner Pokemon details from the Solo screen DOM."""
        if not self.page:
            return {"name": "Unknown", "startsWith": "", "starting_move": "", "dex": None}

        try:
            return self.page.evaluate('''() => {
                let startsWithMove = '';
                let dex = null;
                let partnerName = '';
                
                // 1. Search text containing "Starts with"
                const allDivs = Array.from(document.querySelectorAll('div, section, aside, p, span, h2, h3'));
                for (let el of allDivs) {
                    const txt = el.innerText || '';
                    if (txt.includes('Starts with')) {
                        const match = txt.match(/Starts with\\s+([^\\n]+)/i);
                        if (match) startsWithMove = match[1].trim();
                        const dexMatch = txt.match(/#(\\d+)/);
                        if (dexMatch) dex = parseInt(dexMatch[1].trim());
                    }
                }

                // 2. Selected cell
                const selectedCell = document.querySelector('.cell.on');
                if (selectedCell) {
                    const raw = selectedCell.innerText.replace(/\\n/g, ' ').trim();
                    // Example: "788 Tapu Fini +83" or "813 Scorbunny"
                    const parts = raw.split(/\\s+/);
                    if (parts.length >= 2) {
                        if (!dex && parts[0].match(/^\\d+$/)) {
                            dex = parseInt(parts[0]);
                        }
                        // Filter out numbers and +83
                        const nameParts = parts.filter(p => !p.match(/^\\d+$/) && !p.startsWith('+'));
                        if (nameParts.length > 0) {
                            partnerName = nameParts.join(' ');
                        }
                    } else if (raw) {
                        partnerName = raw;
                    }
                }

                return {
                    name: partnerName || (dex ? `Pokemon #${dex}` : 'Unknown'),
                    starting_move: startsWithMove,
                    startsWith: startsWithMove,
                    dex: dex
                };
            }''')
        except Exception as e:
            return {"name": "Unknown", "startsWith": "", "starting_move": "", "dex": None, "error": str(e)}

    def read_dom_state(self) -> Dict[str, Any]:
        """Reads clean structured game state from the DOM."""
        if not self.page:
            return {"phase": "disconnected", "active_screen": "none"}

        try:
            return self.page.evaluate('''() => {
                const screensEl = document.querySelector('main.screens');
                const activeScreen = screensEl ? screensEl.getAttribute('data-screen') : 'unknown';

                const phaseEl = document.querySelector('[data-testid="game-phase"]');
                let phase = phaseEl ? phaseEl.innerText.trim() : 'none';
                if ((phase === 'none' || !phase) && activeScreen === 'game') {
                    phase = 'running';
                }
                if (phase === 'none' || !phase) {
                    if (typeof window !== 'undefined' && window.G && window.G.gState && window.G.gState.gameState) {
                        const gState = String(window.G.gState.gameState).toLowerCase();
                        if (['running', 'play', 'playing', 'battle'].includes(gState)) phase = 'running';
                        else if (['gameover', 'ended', 'death', 'victory'].includes(gState)) phase = 'ended';
                        else if (['choice', 'levelup', 'pause', 'paused'].includes(gState)) phase = 'choice';
                        else phase = gState;
                    }
                }

                const choiceEl = document.querySelector('[data-testid="choice"]');
                const choiceOpen = choiceEl ? !choiceEl.classList.contains('hidden') : false;

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
                        if (lvlMatch) {
                            verifiedLevel = parseInt(lvlMatch[1]);
                        }
                    }

                    const cards = choiceEl.querySelectorAll('.ps-card, [data-testid^="option-"], .cell, .option');
                    cards.forEach((card, idx) => {
                        const nameEl = card.querySelector('.nm, h3, .title, strong');
                        const tagEl = card.querySelector('.tag-chip, .badge, .tag');
                        const descEl = card.querySelector('.ds, .sub, .desc, p');
                        const keyEl = card.querySelector('.key .kb-only, .kb-only');

                        let name = '';
                        if (nameEl) {
                            name = nameEl.childNodes[0]?.nodeValue?.trim() || nameEl.innerText.split('\\n')[0].trim();
                        }
                        if (!name) {
                            name = card.innerText.split('\\n')[0].trim();
                        }

                        const tag = tagEl ? tagEl.innerText.trim() : '';
                        const desc = descEl ? descEl.innerText.trim() : '';
                        const key = keyEl ? keyEl.innerText.trim() : String(idx + 1);
                        const testid = card.getAttribute('data-testid') || `option-${idx}`;

                        choices.push({
                            index: idx,
                            name: name || `Option ${idx+1}`,
                            tag,
                            desc,
                            description: desc,
                            key,
                            testid,
                            selector: `[data-testid="${testid}"]`
                        });
                    });
                }

                // Check results / exit
                const exitBtn = document.querySelector('[data-testid="game-exit"]');
                const exitAvailable = exitBtn ? exitBtn.offsetParent !== null : false;
                
                // Read topbar gems & tickets
                const gemsEl = document.querySelector('[data-testid="topbar-gems"]');
                const gems = gemsEl ? parseInt(gemsEl.innerText) || 0 : 0;

                // Check Rift & Arena status
                const bodyText = document.body.innerText;
                const riftBanner = bodyText.includes('A Rift Has Opened') || bodyText.includes('Rift');
                const isArena = bodyText.includes('ARENA') || bodyText.includes('Rift Boss');

                return {
                    activeScreen,
                    phase,
                    choiceOpen,
                    modalKind,
                    modalTitle,
                    verifiedLevel,
                    choices,
                    exitAvailable,
                    gems,
                    riftBanner,
                    isArena
                };
            }''')
        except Exception as e:
            return {
                "phase": "error",
                "activeScreen": "unknown",
                "choiceOpen": False,
                "choices": [],
                "exitAvailable": False,
                "error": str(e)
            }

    def read_match_results(self) -> Dict[str, Any]:
        """
        Reads genuine game end-of-match statistics from the DOM:
        - [data-testid="results"] / .ps-results
        - [data-testid="run-stat-kills"] b (Defeated monsters count)
        - [data-testid="run-stat-level"] b (Final character level)
        - [data-testid="run-stat-damage"] b (Total damage dealt)
        - [data-testid="run-stat-last-hits"] b (Last hits)
        - [data-testid="run-stat-meters"] b (Meters ran)
        - [data-testid="run-stat-downs"] b (Times down)
        - .stats (Round, Time, Level, Defeated, Bosses, PokeCoins, Gems, Tickets)
        """
        if not self.page:
            return {"available": False, "source": "no_page"}

        try:
            return self.page.evaluate('''() => {
                const resultsEl = document.querySelector('[data-testid="results"]') || document.querySelector('.ps-results');
                if (!resultsEl || resultsEl.offsetParent === null) {
                    return { available: false, source: "no_results_modal" };
                }

                const titleEl = resultsEl.querySelector('h2');
                const outcome = titleEl ? titleEl.innerText.trim() : 'Unknown';

                // 1. Try reading directly from data-testid="run-stat-*" elements
                const killsEl = resultsEl.querySelector('[data-testid="run-stat-kills"] b');
                const levelEl = resultsEl.querySelector('[data-testid="run-stat-level"] b');
                const damageEl = resultsEl.querySelector('[data-testid="run-stat-damage"] b');
                const lastHitsEl = resultsEl.querySelector('[data-testid="run-stat-last-hits"] b');
                const metersEl = resultsEl.querySelector('[data-testid="run-stat-meters"] b');
                const downsEl = resultsEl.querySelector('[data-testid="run-stat-downs"] b');

                // 2. Read key-value pairs from .stats div
                const statRows = Array.from(resultsEl.querySelectorAll('.stats > div'));
                const statsMap = {};
                for (let row of statRows) {
                    const span = row.querySelector('span');
                    const b = row.querySelector('b');
                    if (span && b) {
                        const key = span.innerText.trim().toLowerCase();
                        statsMap[key] = b.innerText.trim();
                    }
                }

                // Parse kills (defeated count)
                let kills = null;
                if (killsEl && killsEl.innerText.trim()) {
                    const parsed = parseInt(killsEl.innerText.replace(/[^0-9]/g, ''));
                    if (!isNaN(parsed)) kills = parsed;
                } else if (statsMap['defeated']) {
                    const parsed = parseInt(statsMap['defeated'].replace(/[^0-9]/g, ''));
                    if (!isNaN(parsed)) kills = parsed;
                }

                // Parse level
                let level = null;
                if (levelEl && levelEl.innerText.trim()) {
                    const parsed = parseInt(levelEl.innerText.replace(/[^0-9]/g, ''));
                    if (!isNaN(parsed)) level = parsed;
                } else if (statsMap['level']) {
                    const parsed = parseInt(statsMap['level'].replace(/[^0-9]/g, ''));
                    if (!isNaN(parsed)) level = parsed;
                }

                // Parse time
                let timeSeconds = null;
                const timeStr = statsMap['time'];
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

                // Parse coins
                let coins = null;
                const coinStr = statsMap['pokécoins earned'] || statsMap['pokecoins earned'];
                if (coinStr) {
                    const parsed = parseInt(coinStr.replace(/[^0-9]/g, ''));
                    if (!isNaN(parsed)) coins = parsed;
                }

                // Parse bosses
                let bosses = null;
                if (statsMap['bosses']) {
                    const parsed = parseInt(statsMap['bosses'].replace(/[^0-9]/g, ''));
                    if (!isNaN(parsed)) bosses = parsed;
                }

                return {
                    available: true,
                    source: "dom_results_modal",
                    outcome,
                    kills,
                    level,
                    time_seconds: timeSeconds,
                    damage: damageEl ? damageEl.innerText.trim() : null,
                    last_hits: lastHitsEl ? lastHitsEl.innerText.trim() : null,
                    meters: metersEl ? metersEl.innerText.trim() : null,
                    downs: downsEl ? downsEl.innerText.trim() : null,
                    coins,
                    bosses,
                    raw_stats: statsMap
                };
            }''')
        except Exception as e:
            return {"available": False, "source": f"error: {str(e)}"}

