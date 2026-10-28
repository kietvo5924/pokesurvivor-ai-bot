import time
import os
from typing import Optional, Set
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from pokesurvivor.config import GAME_URL, STORAGE_STATE_PATH, VIEWPORT_WIDTH, VIEWPORT_HEIGHT
from pokesurvivor.game_state_adapter import CANVAS_HOOK_SCRIPT

class BrowserController:
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.held_keys: Set[str] = set()

    def start(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding"
            ]
        )
        
        context_opts = {
            "viewport": {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        }
        if os.path.exists(STORAGE_STATE_PATH):
            context_opts["storage_state"] = STORAGE_STATE_PATH

        self.context = self.browser.new_context(**context_opts)
        # Inject Canvas hook & window.G setter at document_start
        self.context.add_init_script(CANVAS_HOOK_SCRIPT)
        self.page = self.context.new_page()

    def stop(self):
        self.release_all_keys()
        if self.page:
            try:
                self.page.close()
            except Exception:
                pass
        if self.context:
            try:
                self.context.close()
            except Exception:
                pass
        if self.browser:
            try:
                self.browser.close()
            except Exception:
                pass
        if self.playwright:
            try:
                self.playwright.stop()
            except Exception:
                pass

    def navigate_home(self):
        if not self.page:
            return
        self.page.goto(GAME_URL, wait_until="networkidle")
        time.sleep(1.5)

    def is_logged_in(self) -> bool:
        if not self.page:
            return False
        # If username input exists, we are not logged in
        return self.page.query_selector('input[name="username"]') is None

    def start_solo_game(self, on_solo_screen_callback=None) -> bool:
        """Navigates from menu to Solo and starts the match."""
        if not self.page:
            return False
        
        try:
            # 1. Wait for and click Solo menu button
            self.page.wait_for_selector('[data-testid="menu-solo"]', state="visible", timeout=10000)
            self.page.click('[data-testid="menu-solo"]')
            time.sleep(1.0)

            # 2. Ensure partner pokemon is selected
            selected = self.page.query_selector('.cell.on')
            if not selected:
                cell = self.page.query_selector('.cell:not(.cell-favourite)')
                if cell:
                    cell.click()
                    time.sleep(0.4)

            # Callback while on solo screen (e.g. detect partner info)
            if on_solo_screen_callback:
                try:
                    on_solo_screen_callback()
                except Exception:
                    pass

            # 3. Wait for and click Play [solo-start] button
            self.page.wait_for_selector('[data-testid="solo-start"]', state="visible", timeout=10000)
            self.page.click('[data-testid="solo-start"]')
            time.sleep(1.0)

            # 4. Handle "End saved run & start new" dialog if present
            try:
                confirm_btn = self.page.query_selector('[data-testid="dialog-confirm"]')
                if confirm_btn and confirm_btn.is_visible():
                    print("[*] Phát hiện hộp thoại ván đã lưu: Đang bấm 'End saved run & start new'...")
                    confirm_btn.click()
                    time.sleep(1.2)
            except Exception:
                pass

            return True
        except Exception as e:
            print(f"[!] Lỗi khi vào trận Solo: {e}")
            return False

    def return_to_menu(self) -> bool:
        """Exits cleanly from any screen back to main menu (#menu)."""
        if not self.page:
            return False
        self.release_all_keys()
        
        # 1. Check if results exit button is visible or wait briefly for it
        try:
            for _ in range(6):
                exit_btn = self.page.query_selector('[data-testid="game-exit"]')
                if exit_btn and exit_btn.is_visible():
                    exit_btn.click()
                    time.sleep(1.0)
                    break
                time.sleep(0.4)
        except Exception:
            pass

        # 2. Check if menu-solo is visible (smooth return without reload)
        for _ in range(10):
            try:
                if self.page.is_visible('[data-testid="menu-solo"]'):
                    return True
            except Exception:
                pass
            time.sleep(0.3)

        # 3. Fallback: navigate directly to #menu
        try:
            self.page.goto(f"{GAME_URL}#menu", wait_until="networkidle")
            time.sleep(1.0)
            self.page.wait_for_selector('[data-testid="menu-solo"]', state="visible", timeout=6000)
            return True
        except Exception:
            return False

    def key_down(self, key: str):
        if not self.page:
            return
        if key not in self.held_keys:
            try:
                self.page.keyboard.down(key)
                self.held_keys.add(key)
            except Exception:
                pass

    def key_up(self, key: str):
        if not self.page:
            return
        if key in self.held_keys:
            try:
                self.page.keyboard.up(key)
                self.held_keys.remove(key)
            except Exception:
                pass

    def press_key(self, key: str):
        if not self.page:
            return
        try:
            self.page.keyboard.press(key)
        except Exception:
            pass

    def release_all_keys(self):
        """Emergency safe release of all keys currently held down."""
        if not self.page:
            return
        keys_to_release = list(self.held_keys)
        for key in keys_to_release:
            try:
                self.page.keyboard.up(key)
            except Exception:
                pass
        self.held_keys.clear()

    def click_selector(self, selector: str) -> bool:
        if not self.page:
            return False
        try:
            el = self.page.query_selector(selector)
            if el:
                el.click()
                return True
        except Exception:
            pass
        return False

