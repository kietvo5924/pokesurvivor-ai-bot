import time
import os
import sys
import threading
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from pynput import keyboard

from pokesurvivor.browser_controller import BrowserController
from pokesurvivor.game_observer import GameObserver
from pokesurvivor.perception import PerceptionEngine
from pokesurvivor.state_tracker import StateTracker
from pokesurvivor.movement_policy import MovementPolicy
from pokesurvivor.upgrade_policy import UpgradePolicy
from pokesurvivor.learning_engine import LearningEngine
from pokesurvivor.memory_store import MemoryStore
from pokesurvivor.metrics_reporter import MetricsReporter
from pokesurvivor.config import HOTKEY_TOGGLE, HOTKEY_EMERGENCY_STOP

class RunManager:
    def __init__(
        self,
        max_runs: int = 5,
        max_duration_per_run: float = 300.0,
        observe_only: bool = False,
        headless: bool = False
    ):
        self.max_runs = max_runs
        self.max_duration_per_run = max_duration_per_run
        self.observe_only = observe_only
        self.headless = headless

        self.memory_store = MemoryStore()
        self.learning_engine = LearningEngine(self.memory_store)
        self.metrics_reporter = MetricsReporter(self.memory_store)
        self.state_tracker = StateTracker()
        self.perception = PerceptionEngine()

        self.policy = self.learning_engine.current_policy
        self.movement_policy = MovementPolicy(self.policy)
        self.upgrade_policy = UpgradePolicy(self.policy)

        self.browser_ctrl: Optional[BrowserController] = None
        self.observer: Optional[GameObserver] = None

        self.is_running = True
        self.is_paused = False
        self.emergency_stopped = False

        self._init_hotkeys()

    def _init_hotkeys(self):
        def on_press(key):
            try:
                if key == keyboard.Key.f8:
                    self.is_paused = not self.is_paused
                    status = "TẠM DỪNG (PAUSED)" if self.is_paused else "TIẾP TỤC (RESUMED)"
                    print(f"\n[HOTKEY F8] Đã chuyển trạng thái bot: {status}")
                    if self.browser_ctrl and self.is_paused:
                        self.browser_ctrl.release_all_keys()
                elif key == keyboard.Key.f9:
                    print("\n[HOTKEY F9] DỪNG KHẨN CẤP (EMERGENCY STOP)! Đang nhả toàn bộ phím và dừng...")
                    self.emergency_stopped = True
                    self.is_running = False
                    if self.browser_ctrl:
                        self.browser_ctrl.release_all_keys()
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

    def run_session(self):
        print("\n" + "=" * 60)
        print("🚀 KHỞI ĐỘNG HỆ THỐNG POKÉSURVIVOR AUTOPLAY BOT")
        print("=" * 60)
        print(f"📌 Chế độ: {'Chỉ quan sát (Observe Only)' if self.observe_only else 'Tự động chơi hoàn toàn'}")
        print(f"📌 Số lượt tối đa: {self.max_runs}")
        print(f"📌 Thời gian tối đa/lượt: {self.max_duration_per_run}s")
        print(f"📌 Phím F8: Bật / Tạm dừng gửi thao tác")
        print(f"📌 Phím F9: Dừng khẩn cấp toàn bộ hệ thống")
        print(f"📌 Chính sách hiện tại: v{self.policy.get('version', '1.0')} ({self.policy.get('description', '')})")
        print("=" * 60 + "\n")

        self.browser_ctrl = BrowserController(headless=self.headless)
        self.browser_ctrl.start()
        self.observer = GameObserver(self.browser_ctrl.page)

        try:
            self.browser_ctrl.navigate_home()
            time.sleep(1.5)

            if not self.browser_ctrl.is_logged_in():
                print("[!] Chưa đăng nhập. Vui lòng đảm bảo auth_state.json hợp lệ.")
                return

            completed_runs = 0
            while self.is_running and completed_runs < self.max_runs and not self.emergency_stopped:
                completed_runs += 1
                print(f"\n--- BẮT ĐẦU VÁN #{completed_runs}/{self.max_runs} ---")
                
                # Refresh policy
                self.policy = self.learning_engine.current_policy
                self.movement_policy.update_config(self.policy)
                self.upgrade_policy.update_config(self.policy)

                success = self._run_single_match(completed_runs)
                if not success or self.emergency_stopped:
                    break

                # Immediately proceed to next match without delay (nghỉ 3 giây đã được bỏ)
                if completed_runs < self.max_runs and self.is_running:
                    pass

            print("\n" + "=" * 60)
            print("🏁 HOÀN TẤT PHIÊN CHẠY (SESSION FINISHED)")
            print(self.metrics_reporter.generate_comparison_table())

        except Exception as e:
            print(f"[!] Lỗi trong phiên chạy: {e}")
        finally:
            if self.browser_ctrl:
                self.browser_ctrl.release_all_keys()
                self.browser_ctrl.stop()

    def _run_single_match(self, run_index: int) -> bool:
        """Executes a single game match from start to game-over."""
        def on_solo_ready():
            partner = self.observer.get_selected_partner()
            if partner:
                p_name = partner.get("name", "Tapu Fini")
                p_move = partner.get("starting_move", "Water Gun")
                p_dex = partner.get("dex", 788)
                print(f"[*] 🎯 Nhận diện Pokémon xuất trận: No. {p_dex} {p_name} | Chiêu đầu: '{p_move}'")
                self.upgrade_policy.set_active_pokemon(p_name, starting_move=p_move, dex=p_dex)
                self.state_tracker.set_active_pokemon(p_name)
                self.perception.set_active_pokemon(p_name)

        print("[*] Đang vào trận Solo...")
        if not self.browser_ctrl.start_solo_game(on_solo_screen_callback=on_solo_ready):
            print("[!] Không thể bấm bắt đầu Solo. Kiểm tra lại giao diện.")
            return False

        # 2. Wait for game-phase == 'running'
        print("[*] Chờ trận đấu sẵn sàng (phase: running)...")
        ready = False
        for wait_i in range(100):  # wait up to 35 seconds
            if self.emergency_stopped:
                return False
            dom_state = self.observer.read_dom_state()
            current_phase = dom_state.get("phase", "none")
            if current_phase == "running":
                ready = True
                break
            if wait_i % 10 == 0 and wait_i > 0:
                print(f"  ...Đang tải trận (Current phase: '{current_phase}') [{wait_i * 0.35:.1f}s]")
            time.sleep(0.35)

        if not ready:
            print("[!] Quá thời gian chờ game bắt đầu.")
            return False

        print("[*] TRẬN ĐẤU ĐÃ BẮT ĐẦU! Bắt đầu chu trình quan sát & điều khiển...")
        self.state_tracker.start_run()
        self.upgrade_policy.reset_match()

        # Ensure native Auto-Aim is strictly in AUTO mode (never toggle off)
        try:
            aim_mode = self.browser_ctrl.page.evaluate('''() => {
                try {
                    const raw = localStorage.getItem('ps.settings');
                    if (raw) {
                        const s = JSON.parse(raw);
                        if (s.aimMode !== 'auto') {
                            s.aimMode = 'auto';
                            localStorage.setItem('ps.settings', JSON.stringify(s));
                            return 'toggled_to_auto';
                        }
                        return 'already_auto';
                    }
                } catch(e) {}
                return 'unknown';
            }''')
            if aim_mode == 'toggled_to_auto':
                # Synchronize in-memory settings with KeyT
                self.browser_ctrl.press_key("t")
                print("[*] Native Auto-Aim đã kích hoạt chế độ AUTO (tự động khóa mục tiêu và bắn).")
            else:
                print("[*] Native Auto-Aim đã ở chế độ AUTO (tự động khóa mục tiêu và bắn).")
        except Exception:
            pass

        active_keys = []
        match_start_time = time.time()
        last_move_time = time.time()
        last_vision_time = 0.0
        last_rift_log_time = 0.0
        last_boss_log_time = 0.0
        last_status_log_time = 0.0
        last_dodge_log_time = 0.0
        last_buffer_log_time = 0.0
        last_overlay_time = 0.0
        vision_state = None

        while self.is_running and not self.emergency_stopped:
            loop_start = time.time()
            elapsed = loop_start - match_start_time

            if elapsed > self.max_duration_per_run:
                print(f"[*] Đạt giới hạn thời gian ván ({self.max_duration_per_run}s).")
                self.state_tracker.end_reason = "timeout"
                break

            # 3. Read DOM state
            dom_state = self.observer.read_dom_state()
            phase = dom_state.get("phase", "running")
            exit_available = dom_state.get("exitAvailable", False)

            # Check match end
            if phase == "ended" or exit_available:
                try:
                    self.browser_ctrl.page.screenshot(path="debug_artifacts/last_match_end.png")
                except Exception:
                    pass
                print(f"[*] Trận đấu kết thúc! Phase: {phase}")
                break

            # 4. Handle Modals: Level Up, Rift Boss Bond/Fusion, Item Swap, or Chest
            if dom_state.get("choiceOpen"):
                # Release movement keys during modal
                self.browser_ctrl.release_all_keys()
                active_keys = []
                choices = dom_state.get("choices", [])
                modal_kind = dom_state.get("modalKind", "level-up")
                modal_title = dom_state.get("modalTitle", "")
                
                print(f"[{elapsed:.1f}s] 🌟 GIAO DIỆN [{modal_kind.upper()}]: '{modal_title}' ({len(choices)} lựa chọn)")
                
                if choices:
                    best_card = self.upgrade_policy.choose_card(
                        choices,
                        current_hp_ratio=self.state_tracker.last_hp,
                        current_skills=[u["name"] for u in self.state_tracker.upgrades_chosen],
                        modal_kind=modal_kind
                    )
                    if best_card:
                        print(f"  -> Bot chọn: '{best_card.get('name')}' ({best_card.get('tag')}) - Lý do: {best_card.get('selection_reason')}")
                        if not self.observe_only and not self.is_paused:
                            # Try clicking selector and pressing number key
                            self.browser_ctrl.click_selector(best_card.get("selector"))
                            key = best_card.get("key", "1")
                            if key:
                                self.browser_ctrl.press_key(key)
                        self.state_tracker.record_upgrade(
                            best_card,
                            modal_kind=modal_kind,
                            verified_level=dom_state.get("verifiedLevel")
                        )

                        # Actively wait until the choice modal is completely closed to prevent duplicate processing
                        modal_wait_start = time.time()
                        while time.time() - modal_wait_start < 1.6:
                            time.sleep(0.08)
                            st = self.observer.read_dom_state()
                            if not st.get("choiceOpen"):
                                break
                continue

            # 5. Capture visual frame (ultra-fast toDataURL without screen jitter)
            now_t = time.time()
            if now_t - last_vision_time >= 0.08 or vision_state is None:
                frame = self.observer.capture_frame()
                if frame is not None:
                    vision_state = self.perception.analyze_frame(frame, is_moving=bool(active_keys))
                    last_vision_time = now_t

            if vision_state:
                self.state_tracker.update(dom_state, vision_state)

                # Announce Purple Rift Portal
                rift_target = vision_state.get("rift_target")
                if rift_target and (now_t - last_rift_log_time > 8.0):
                    t_type = rift_target.get("type", "portal")
                    if t_type == "portal" and elapsed > 120.0 and self.state_tracker.current_level >= 3:
                        print(f"[{elapsed:.1f}s] 🌀 CỔNG RIFT TÍM: Đủ cấp và thực lực! Tiến vào cổng {rift_target.get('sector')} diệt Boss săn quà...")
                    else:
                        print(f"[{elapsed:.1f}s] 🌀 CỔNG RIFT TÍM: Mũi tên chỉ hướng {rift_target.get('sector')} (Tiếp tục kiting farm ngọc EXP & tích lũy tiến hóa)")
                    last_rift_log_time = now_t

                # Announce Boss Arena Combat
                boss_info = vision_state.get("boss_info", {})
                if boss_info.get("active") and (now_t - last_boss_log_time > 8.0):
                    print(f"[{elapsed:.1f}s] ⚔️ ĐANG ĐÁNH BOSS! Kiting vòng tròn quanh Boss và auto-aim xả chiêu...")
                    last_boss_log_time = now_t

            # 6. Movement decision & execution with Obstacle Avoidance & Dodging
            if not self.observe_only and not self.is_paused and vision_state:
                dir_name, target_keys, step_dur = self.movement_policy.select_movement(
                    vision_state,
                    current_level=self.state_tracker.current_level,
                    elapsed_time=elapsed
                )
                
                # Check for critical tactical events to log
                min_e_dist = vision_state.get("min_enemy_dist", 9999.0)
                e_sec = vision_state.get("nearest_enemy_sector", "?")
                hazards = vision_state.get("hazards", [])
                sector_obstacles = vision_state.get("sector_obstacles", {})
                blocked_walls = [s for s, b in sector_obstacles.items() if b]

                # Wall / Dead-end breakout alert
                if len(blocked_walls) >= 1 and (min_e_dist < 210.0 or vision_state.get("is_encircled", False)) and (now_t - last_buffer_log_time > 2.5):
                    fewest_mobs = vision_state.get("sector_enemy_counts", {}).get(dir_name, 0)
                    print(f"[{elapsed:.1f}s] 💥 PHÁ VÒNG VÂY TƯỜNG CỤT: Tường chặn {blocked_walls}! Lao đầu bứt tốc hướng {dir_name} ({fewest_mobs} quái - đường thoáng nhất) thoát hiểm!")
                    last_buffer_log_time = now_t

                # Encirclement breakout alert
                elif vision_state.get("is_encircled", False) and (now_t - last_buffer_log_time > 2.5):
                    fewest_mobs = vision_state.get("sector_enemy_counts", {}).get(dir_name, 0)
                    print(f"[{elapsed:.1f}s] ⚡ ĐỘT PHÁ VÒNG VÂY: Quái vây quanh! Bứt tốc hướng {dir_name} ({fewest_mobs} quái - hướng thoáng nhất) thoát hiểm!")
                    last_buffer_log_time = now_t

                # Bullet dodge alert
                elif hazards and hazards[0].get("dist", 999.0) < 165.0 and (now_t - last_dodge_log_time > 3.0):
                    h_dist = hazards[0].get("dist", 0.0)
                    h_sec = hazards[0].get("sector", "?")
                    print(f"[{elapsed:.1f}s] ⚡ NÉ ĐẠN: Phát hiện đạn ở {h_dist:.0f}px hướng {h_sec}! Lạng sang hướng {dir_name} né đòn!")
                    last_dodge_log_time = now_t

                # Buffer alert (Repulsion Field active)
                elif min_e_dist < 175.0 and (now_t - last_buffer_log_time > 3.0):
                    print(f"[{elapsed:.1f}s] 🛡️ GIỮ CỰ LY (REPULSION): Quái áp sát {min_e_dist:.0f}px hướng {e_sec}! Kích hoạt lực đẩy phản xạ lùi hướng {dir_name} duy trì cự ly an toàn!")
                    last_buffer_log_time = now_t

                # Regular tactical telemetry every 4.0 seconds
                elif now_t - last_status_log_time > 4.0:
                    n_enemies = vision_state.get("enemies_count", 0)
                    n_hazards = len(hazards)
                    hp_r = vision_state.get("hp_ratio")
                    hp_str = f"{hp_r*100:.0f}% ({vision_state.get('hp_status', 'UNKNOWN')})" if hp_r is not None else "UNKNOWN"
                    dist_str = f"{min_e_dist:.0f}px ({e_sec})" if min_e_dist < 9000 else "Không có quái gần"
                    print(f"[{elapsed:.1f}s] 🏃 Kiting: Hướng {dir_name} | Quái gần nhất: {dist_str} | Quái quanh vùng: {n_enemies} | Đạn: {n_hazards} | HP: {hp_str}")
                    last_status_log_time = now_t

                # Periodically render and save annotated debug visual overlay
                if frame is not None and (now_t - last_overlay_time > 4.0 or (hazards and hazards[0].get("is_imminent", False))):
                    self.perception.create_debug_overlay(frame, vision_state, chosen_dir=dir_name)
                    last_overlay_time = now_t

                # Update held keys if direction changed
                if target_keys != active_keys:
                    for k in active_keys:
                        if k not in target_keys:
                            self.browser_ctrl.key_up(k)
                    for k in target_keys:
                        if k not in active_keys:
                            self.browser_ctrl.key_down(k)
                    active_keys = target_keys

            # Throttle loop to ~25 FPS to maintain responsive control
            processing_time = time.time() - loop_start
            sleep_time = max(0.01, 0.04 - processing_time)
            time.sleep(sleep_time)

        # 7. Match Cleanup & Learning
        self.browser_ctrl.release_all_keys()

        # If ended via timeout while still running, cleanly open pause to exit
        if not exit_available and phase != "ended":
            try:
                self.browser_ctrl.press_key("Escape")
                time.sleep(0.6)
                self.browser_ctrl.click_selector('[data-testid="game-exit"]')
                time.sleep(1.2)
            except Exception:
                pass

        run_result = self.state_tracker.end_run(
            reason="timeout" if elapsed > self.max_duration_per_run else "defeated"
        )

        # Learning & adaptation
        analysis = self.learning_engine.evaluate_run(run_result)
        
        # Save history & metrics
        run_record = {
            "run_id": analysis.get("total_runs", run_index),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": run_result["duration"],
            "max_level": run_result["max_level"],
            "damage_events": run_result["damage_events"],
            "upgrades": run_result["upgrades"],
            "end_reason": run_result["end_reason"],
            "score": analysis["score"],
            "policy_version": self.policy.get("version", "1.0"),
            "movement_pattern": self.movement_policy.pattern
        }
        self.memory_store.append_run_history(run_record)
        self.memory_store.append_metrics([
            run_record["run_id"],
            run_record["timestamp"],
            run_record["duration"],
            run_record["max_level"],
            len(run_record["upgrades"]),
            run_record["end_reason"],
            run_record["policy_version"],
            run_record["movement_pattern"],
            run_record["score"],
            "; ".join(analysis.get("adaptations", []))
        ])

        # Print summary
        self.metrics_reporter.print_run_summary(run_result, analysis)

        # 8. Return to menu
        print("[*] Đang trở về Menu để chuẩn bị lượt tiếp theo...")
        self.browser_ctrl.return_to_menu()
        time.sleep(2.0)

        return True
