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

    def read_dom_state(self) -> Dict[str, Any]:
        """Reads clean structured game state from the DOM."""
        if not self.page:
            return {"phase": "disconnected", "active_screen": "none"}

        try:
            return self.page.evaluate('''() => {
                const screensEl = document.querySelector('main.screens');
                const activeScreen = screensEl ? screensEl.getAttribute('data-screen') : 'unknown';

                const phaseEl = document.querySelector('[data-testid="game-phase"]');
                const phase = phaseEl ? phaseEl.innerText.trim() : 'none';

                const choiceEl = document.querySelector('[data-testid="choice"]');
                const choiceOpen = choiceEl ? !choiceEl.classList.contains('hidden') : false;

                let modalKind = 'level-up';
                let modalTitle = '';
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

