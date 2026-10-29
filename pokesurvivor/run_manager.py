import time
import os
import sys
import threading
from typing import Dict, Any, Optional, List

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
from pokesurvivor.game_state_adapter import GameStateAdapter
from pokesurvivor.coin_tracker import CoinTracker
from pokesurvivor.decision_tracer import DecisionTracer
from pokesurvivor.config import HOTKEY_TOGGLE, HOTKEY_EMERGENCY_STOP

class LoopProfiler:
    """
    Real-time latency profiler measuring:
    - capture: frame grab latency
    - state_read: unified state extraction latency
    - decision: predictive planner decision latency
    - action_dispatch: keyboard input dispatch latency
    - total_loop: full loop cycle latency
    Computes Median (P50), P95, and control loop frequency (Hz).
    """
    def __init__(self, window_size: int = 120):
        self.window_size = window_size
        self.samples: Dict[str, List[float]] = {
            "capture": [],
            "state_read": [],
            "decision": [],
            "action_dispatch": [],
            "total_loop": []
        }
        self.total_cycles: int = 0
        self.start_time: float = time.time()

    def record(self, stage: str, duration_sec: float):
        if stage in self.samples:
            ms = duration_sec * 1000.0
            buf = self.samples[stage]
            buf.append(ms)
            if len(buf) > self.window_size:
                buf.pop(0)

    def record_cycle(self):
        self.total_cycles += 1

    def get_summary(self) -> Dict[str, Any]:
        elapsed = max(0.1, time.time() - self.start_time)
        freq_hz = self.total_cycles / elapsed
        summary = {"frequency_hz": round(freq_hz, 1), "total_cycles": self.total_cycles}

        for stage, vals in self.samples.items():
            if vals:
                s_sorted = sorted(vals)
                n = len(s_sorted)
                p50 = s_sorted[int(n * 0.50)]
                p95 = s_sorted[min(n - 1, int(n * 0.95))]
                summary[stage] = {
                    "median_ms": round(p50, 2),
                    "p95_ms": round(p95, 2),
                    "min_ms": round(s_sorted[0], 2),
                    "max_ms": round(s_sorted[-1], 2)
                }
            else:
                summary[stage] = {"median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        return summary

    def format_report(self) -> str:
        s = self.get_summary()
        freq = s.get("frequency_hz", 0.0)
        c = s.get("capture", {})
        sr = s.get("state_read", {})
        d = s.get("decision", {})
        ad = s.get("action_dispatch", {})
        tot = s.get("total_loop", {})
        return (
            f"\n============================================================\n"
            f"⏱️ THỐNG KÊ ĐỘ TRỄ VÒNG LẶP & ĐIỀU KHIỂN (LOOP LATENCY PROFILE)\n"
            f"============================================================\n"
            f"⚡ Tần số quyết định (Control Loop Freq): {freq:.1f} Hz ({s.get('total_cycles', 0)} chu kỳ)\n"
            f"🔄 Chu kỳ tổng (Total Loop Cycle)     : Median = {tot.get('median_ms', 0):.1f} ms | P95 = {tot.get('p95_ms', 0):.1f} ms\n"
            f"📸 Thu thập ảnh (Visual Capture)     : Median = {c.get('median_ms', 0):.1f} ms | P95 = {c.get('p95_ms', 0):.1f} ms\n"
            f"🔍 Trích xuất trạng thái (State Read) : Median = {sr.get('median_ms', 0):.1f} ms | P95 = {sr.get('p95_ms', 0):.1f} ms\n"
            f"🧠 Lập kế hoạch (Decision Planner)    : Median = {d.get('median_ms', 0):.1f} ms | P95 = {d.get('p95_ms', 0):.1f} ms\n"
            f"⌨️ Gửi lệnh phím (Action Dispatch)    : Median = {ad.get('median_ms', 0):.1f} ms | P95 = {ad.get('p95_ms', 0):.1f} ms\n"
            f"============================================================"
        )


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
        self.coin_tracker = CoinTracker()
        self.decision_tracer = DecisionTracer()

        self.browser_ctrl: Optional[BrowserController] = None
        self.observer: Optional[GameObserver] = None
        self.game_state_adapter: Optional[GameStateAdapter] = None

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
        print("🚀 KHỞI ĐỘNG HỆ THỐNG POKÉSURVIVOR AUTOPLAY BOT (ROUND 9)")
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
        self.game_state_adapter = GameStateAdapter(self.browser_ctrl.page, self.perception)

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
                
                self.policy = self.learning_engine.current_policy
                self.movement_policy.update_config(self.policy)
                self.upgrade_policy.update_config(self.policy)

                success = self._run_single_match(completed_runs)
                if not success or self.emergency_stopped:
                    break

            print("\n" + "=" * 60)
            print("🏁 HOÀN TẤT PHIÊN CHẠY (SESSION FINISHED)")
            print("=" * 60)

        finally:
            self.browser_ctrl.close()

    def _run_single_match(self, run_index: int) -> bool:
        if not self.browser_ctrl.start_new_run():
            print("[!] Không thể bắt đầu ván mới qua UI.")
            return False

        # Read active Pokemon context from DOM or G
        try:
            p_data = self.browser_ctrl.page.evaluate('''() => {
                let name = "Unknown";
                let move = "";
                let dex = null;
                if (window.G && window.G.player) {
                    name = window.G.player.name || window.G.player.pokemon || name;
                    dex = window.G.player.dex || null;
                }
                const nameEl = document.querySelector('[data-testid="selected-pokemon-name"], .pokemon-select-name, .character-name');
                if (nameEl && nameEl.innerText) name = nameEl.innerText.trim();
                return { name: name, move: move, dex: dex };
            }''')
            p_name = p_data.get("name", "Unknown")
            p_move = p_data.get("move", "")
            p_dex = p_data.get("dex")
            if p_name and p_name != "Unknown":
                self.upgrade_policy.set_active_pokemon(p_name, starting_move=p_move, dex=p_dex)
                self.state_tracker.set_active_pokemon(p_name)
                self.perception.set_active_pokemon(p_name)
                print(f"[*] Đã nhận diện Pokemon xuất trận: {p_name} (Dex: {p_dex})")
            else:
                self.upgrade_policy.set_active_pokemon("Unknown", starting_move="", dex=None)
                self.state_tracker.set_active_pokemon("Unknown")
                self.perception.set_active_pokemon("Unknown")
        except Exception:
            pass

        self.state_tracker.start_run(run_index)
        self.upgrade_policy.reset_match()

        print("[*] Đang chờ trận đấu load xong (chờ canvas sẵn sàng)...")
        start_wait = time.time()
        while time.time() - start_wait < 20.0:
            if not self.is_running or self.emergency_stopped:
                return False

            phase_info = self.game_state_adapter.get_current_phase()
            phase = phase_info.get("phase", "none")
            source = phase_info.get("source", "none")

            if phase == "running":
                print(f"[*] Trận đấu đã bắt đầu! Sẵn sàng điều khiển. (Phát hiện qua: {source})")
                break
            
            raw_data = self.game_state_adapter._fetch_browser_data()
            if raw_data.get("hasG") and raw_data.get("gState"):
                g_phase = raw_data["gState"].get("gameState")
                if g_phase in ("running", "active", "combat"):
                    print(f"[*] Trận đấu đã bắt đầu qua window.G ({g_phase})!")
                    break

            time.sleep(0.3)
        else:
            print("[!] Timeout chờ trận đấu bắt đầu. Thử bỏ qua và tiếp tục...")

        # Setup Canvas Telemetry Hook
        try:
            self.game_state_adapter.setup_canvas_telemetry()
        except Exception:
            pass

        active_keys: List[str] = []
        match_start_time = time.time()
        last_vision_time = 0.0
        last_rift_log_time = 0.0
        last_boss_log_time = 0.0
        last_status_log_time = 0.0
        last_buffer_log_time = 0.0
        last_overlay_time = 0.0
        last_slot_cycle_time = 0.0
        last_zmove_time = 0.0
        last_profiler_log_time = 0.0
        last_consumable_times = {"5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0}

        has_rerolled_this_choice = False
        unified_state = None
        dom_results = None
        loop_counter = 0

        # Real-time latency profiler
        profiler = LoopProfiler()

        while self.is_running and not self.emergency_stopped:
            loop_start = time.time()
            loop_counter += 1
            elapsed = loop_start - match_start_time

            if elapsed > self.max_duration_per_run:
                print(f"[*] Đạt giới hạn thời gian ván ({self.max_duration_per_run}s).")
                self.state_tracker.end_reason = "timeout"
                break

            # Check if memory state is verified from window.G
            is_mem_verified = False
            if unified_state is not None:
                p_src = unified_state.get("sources", {}).get("player", "")
                is_mem_verified = (p_src == "window_G" or unified_state.get("is_memory_verified", False))

            # 1. Capture visual frame (Decoupled fast loop: 1.0s when memory is verified, 0.08s on vision fallback)
            now_t = time.time()
            vision_interval = 1.0 if is_mem_verified else 0.08
            frame = None
            if (now_t - last_vision_time >= vision_interval) or (unified_state is None):
                t_cap_start = time.time()
                frame = self.observer.capture_frame()
                profiler.record("capture", time.time() - t_cap_start)
                last_vision_time = now_t
            else:
                profiler.record("capture", 0.0)

            # 2. Read Harmonized Unified State (window.G, Canvas hook, DOM, OpenCV)
            t_sr_start = time.time()
            unified_state = self.game_state_adapter.read_unified_state(
                frame_img=frame,
                is_moving=bool(active_keys)
            )
            profiler.record("state_read", time.time() - t_sr_start)

            # 3. Extract UI & Phase
            dom_state = unified_state.get("ui", {})
            phase = dom_state.get("phase", "running")
            exit_available = dom_state.get("exit_available", False)
            metrics = unified_state.get("metrics", {})

            # Check match end
            if phase == "ended" or exit_available or metrics.get("is_match_over"):
                print(f"[*] Trận đấu kết thúc! Phase: {phase}")
                wait_results_start = time.time()
                while time.time() - wait_results_start < 2.5:
                    dom_results = self.observer.read_match_results()
                    if dom_results and dom_results.get("available"):
                        break
                    fresh_metrics = self.game_state_adapter._extract_metrics(self.game_state_adapter._fetch_browser_data())
                    if fresh_metrics.get("is_match_over"):
                        dom_results = fresh_metrics
                        break
                    time.sleep(0.25)

                try:
                    self.browser_ctrl.page.screenshot(path="debug_artifacts/last_match_end.png")
                except Exception:
                    pass
                break

            # 4. Handle Modals: Level Up, Rift Boss Bond, Item Swap, Chest
            if dom_state.get("choice_open"):
                self.browser_ctrl.release_all_keys()
                active_keys = []
                choices = dom_state.get("choices", [])
                modal_kind = dom_state.get("modal_kind", "level-up")
                modal_title = dom_state.get("modal_title", "")

                print(f"[{elapsed:.1f}s] 🌟 GIAO DIỆN [{modal_kind.upper()}]: '{modal_title}' ({len(choices)} lựa chọn)")

                if choices:
                    best_card = self.upgrade_policy.choose_card(
                        choices,
                        current_hp_ratio=self.state_tracker.last_hp,
                        current_skills=[u["name"] for u in self.state_tracker.upgrades_chosen],
                        modal_kind=modal_kind
                    )
                    if best_card:
                        if best_card.get("should_reroll") and not has_rerolled_this_choice and not self.observe_only and not self.is_paused:
                            try:
                                reroll_avail = self.browser_ctrl.page.evaluate('''() => {
                                    const rr = document.querySelector('[data-testid="reroll"]');
                                    return rr && !rr.disabled && rr.style.display !== 'none';
                                }''')
                                if reroll_avail:
                                    print(f"[{elapsed:.1f}s] 🔄 REROLL: Thẻ điểm quá thấp (<500), kích hoạt Reroll!")
                                    self.browser_ctrl.click_selector('[data-testid="reroll"]')
                                    has_rerolled_this_choice = True
                                    time.sleep(0.4)
                                    continue
                            except Exception:
                                pass

                        choice_idx = best_card["index"]
                        choice_name = best_card.get("name", "Unknown")
                        sel_reason = best_card.get("selection_reason", "standard_pick")
                        print(f"[{elapsed:.1f}s] 🎯 CHỌN KỸ NĂNG: [{choice_name}] (Thẻ #{choice_idx + 1} | Lý do: {sel_reason})")

                        if not self.observe_only and not self.is_paused:
                            self.browser_ctrl.select_upgrade_choice(choice_idx)
                            self.state_tracker.record_upgrade(choice_name, choice_idx)

                        has_rerolled_this_choice = False
                        time.sleep(0.35)
                continue

            # 5. Track state metrics & health
            self.state_tracker.update(
                unified_state=unified_state,
                elapsed_time=elapsed
            )

            # 6. Movement & Combat Action Planning
            t_dec_start = time.time()
            dir_name, target_keys, hold_duration = self.movement_policy.select_movement(
                vision_state=unified_state,
                current_level=metrics.get("level", 1) or 1,
                elapsed_time=elapsed
            )
            profiler.record("decision", time.time() - t_dec_start)

            # Telemetry Trace Recording (data/decision_trace.jsonl)
            player_info = unified_state.get("player") or {}
            self.decision_tracer.record_decision(
                elapsed_time=elapsed,
                player_state=player_info,
                candidate_scores=getattr(self.movement_policy, "last_candidate_scores", {}),
                chosen_direction=dir_name,
                chosen_keys=target_keys,
                risk_breakdown=getattr(self.movement_policy, "last_risk_breakdown", {}),
                level=metrics.get("level", 1) or 1,
                game_score=metrics.get("kills", 0.0) or 0.0,
                now=now_t
            )

            # Delayed HP Outcome Resolution (~0.9s post decision)
            if loop_counter % 12 == 0:
                cur_hp = player_info.get("hp_ratio")
                self.decision_tracer.update_outcomes(cur_hp, now=now_t, delay_window=0.9)

            # Tactical telemetry logging
            min_e_dist = unified_state.get("min_enemy_dist", 9999.0)
            e_sec = unified_state.get("nearest_enemy_sector", "None")
            hazards = unified_state.get("hazards", [])

            if not self.observe_only and not self.is_paused:
                t_act_start = time.time()
                # Update held keys if direction changed
                if target_keys != active_keys:
                    for k in active_keys:
                        if k not in target_keys:
                            self.browser_ctrl.key_up(k)
                    for k in target_keys:
                        if k not in active_keys:
                            self.browser_ctrl.key_down(k)
                    active_keys = target_keys
                profiler.record("action_dispatch", time.time() - t_act_start)

            # Regular status print
            if now_t - last_status_log_time > 4.0:
                n_enemies = len(unified_state.get("enemies", []))
                n_hazards = len(hazards)
                hp_r = player_info.get("hp_ratio")
                hp_src = unified_state.get("sources", {}).get("player", "unobserved")
                conf = unified_state.get("confidence", 0.0)
                hp_str = f"{hp_r*100:.0f}%" if hp_r is not None else "UNKNOWN"
                dist_str = f"{min_e_dist:.0f}px ({e_sec})" if min_e_dist < 9000 else "Không có quái gần"
                print(f"[{elapsed:.1f}s] 🏃 Kiting: Hướng {dir_name} | Quái gần: {dist_str} | Quái: {n_enemies} | Đạn: {n_hazards} | HP: {hp_str} (Nguồn: {hp_src}, Tin cậy: {conf*100:.0f}%)")
                last_status_log_time = now_t

            # Periodically report loop profiler stats (every 8.0s)
            if now_t - last_profiler_log_time > 8.0:
                p_sum = profiler.get_summary()
                tot_p50 = p_sum.get("total_loop", {}).get("median_ms", 0.0)
                tot_p95 = p_sum.get("total_loop", {}).get("p95_ms", 0.0)
                freq_now = p_sum.get("frequency_hz", 0.0)
                print(f"[{elapsed:.1f}s] ⏱️ Vòng lặp: {freq_now:.1f}Hz | P50={tot_p50:.1f}ms, P95={tot_p95:.1f}ms | Chế độ: {'BỘ NHỚ NHANH (Fast Memory)' if is_mem_verified else 'THỊ GIÁC FALLBACK (OpenCV)'}")
                last_profiler_log_time = now_t

            # 6b. Active Combat Actions (Weapon slot cycling, Emergency Z-Move, Condition-based Consumables)
            if not self.observe_only and not self.is_paused and phase == "running":
                # A. Weapon slot cycling ('e'): optimized to 0.18s to continuously fire all unlocked weapons without input lag
                if now_t - last_slot_cycle_time >= 0.18:
                    t_slot_start = time.time()
                    self.browser_ctrl.press_key("e")
                    last_slot_cycle_time = now_t
                    profiler.record("action_dispatch", time.time() - t_slot_start)

                # B. Emergency Z-Move ('f'): triggers 400px burst when critically low HP or surrounded
                if now_t - last_zmove_time >= 4.0 and unified_state:
                    hp_r = player_info.get("hp_ratio")
                    is_crit_hp = (hp_r is not None and hp_r < 0.40)
                    is_heavy_encircled = unified_state.get("is_encircled", False)
                    n_close_mobs = len([e for e in unified_state.get("enemies", []) if e.get("dist", 999.0) < 180.0])
                    if is_crit_hp or is_heavy_encircled or n_close_mobs >= 8:
                        self.browser_ctrl.press_key("f")
                        last_zmove_time = now_t

                # C. Condition-based Consumables (Keys 5-8): Replaces blind periodic spam with tactical triggers
                self._evaluate_and_trigger_consumables(now_t, unified_state, last_consumable_times)

            # Throttle loop: 35-50 Hz when memory is verified (target dt = 0.022s), 25 Hz on vision fallback (0.040s)
            target_dt = 0.022 if is_mem_verified else 0.040
            processing_time = time.time() - loop_start
            profiler.record("total_loop", processing_time)
            profiler.record_cycle()
            sleep_time = max(0.004, target_dt - processing_time)
            time.sleep(sleep_time)

        # 7. Match Cleanup & Final Reporting
        self.browser_ctrl.release_all_keys()
        self.decision_tracer.flush_all()
        print(profiler.format_report())

        if not dom_results or not dom_results.get("available"):
            dom_results = self.observer.read_match_results()
        if (not dom_results or not dom_results.get("available")) and unified_state:
            metrics = unified_state.get("metrics") or {}
            if metrics.get("available") or metrics.get("is_match_over"):
                dom_results = {
                    "available": True,
                    "source": metrics.get("source", "adapter_metrics"),
                    "kills": metrics.get("kills"),
                    "level": metrics.get("level"),
                    "time_seconds": metrics.get("time_seconds"),
                    "time_str": metrics.get("time_str"),
                    "coins": metrics.get("coins"),
                    "bosses": metrics.get("bosses")
                }

        if not exit_available and phase != "ended":
            try:
                self.browser_ctrl.press_key("Escape")
                time.sleep(0.6)
                self.browser_ctrl.click_selector('[data-testid="game-exit"]')
                time.sleep(1.2)
            except Exception:
                pass

        run_result = self.state_tracker.end_run(
            reason="timeout" if elapsed > self.max_duration_per_run else "defeated",
            dom_results=dom_results
        )

        coins_earned = dom_results.get("coins") if dom_results else None
        if coins_earned is None and unified_state:
            coins_earned = unified_state.get("metrics", {}).get("coins")
        self.coin_tracker.record_run(
            coins=coins_earned,
            kills=run_result.get("game_score"),
            survival_time=run_result.get("survival_time", elapsed),
            level=run_result.get("max_level"),
            bosses=dom_results.get("bosses") if dom_results else None,
            policy_version=self.policy.get("version", "1.0")
        )

        analysis = self.learning_engine.evaluate_run(run_result)
        
        self.policy = self.learning_engine.current_policy
        self.movement_policy.update_config(self.policy)
        self.upgrade_policy.update_config(self.policy)

        run_record = {
            "run_id": analysis.get("total_runs", run_index),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pokemon": run_result.get("pokemon", "Unknown"),
            "game_score": run_result.get("game_score"),
            "game_score_source": run_result.get("game_score_source"),
            "duration": run_result["survival_time"],
            "survival_time": run_result["survival_time"],
            "survival_time_source": run_result.get("survival_time_source"),
            "max_level": run_result["max_level"],
            "max_level_source": run_result.get("max_level_source"),
            "damage_events": run_result["damage_events"],
            "upgrades": run_result["upgrades"],
            "end_reason": run_result["end_reason"],
            "score": analysis["fitness_score"],
            "fitness_score": analysis["fitness_score"],
            "fitness_score_source": analysis.get("fitness_score_source", "internal_composite_formula"),
            "policy_version": analysis.get("policy_version", self.policy.get("version", "1.0")),
            "policy_hash": analysis.get("policy_hash", ""),
            "movement_pattern": self.movement_policy.pattern
        }
        self.memory_store.append_run_history(run_record)
        self.memory_store.append_metrics([
            run_record["run_id"],
            run_record["timestamp"],
            run_record["pokemon"],
            run_record["game_score"],
            run_record["game_score_source"],
            run_record["duration"],
            run_record["survival_time_source"],
            run_record["max_level"],
            run_record["max_level_source"],
            run_record["fitness_score"],
            len(run_record["upgrades"]),
            run_record["end_reason"],
            run_record["policy_version"],
            run_record["policy_hash"],
            run_record["movement_pattern"],
            "; ".join(analysis.get("adaptations", []))
        ])

        self.metrics_reporter.print_run_summary(run_result, analysis)
        print(f"[*] {self.coin_tracker.format_status_line()}")

        print("[*] Đang trở về Menu để chuẩn bị lượt tiếp theo...")
        self.browser_ctrl.return_to_menu()
        time.sleep(2.0)

        return True

    def _evaluate_and_trigger_consumables(
        self,
        now_t: float,
        unified_state: Dict[str, Any],
        last_times: Dict[str, float]
    ):
        """
        Condition-based tactical consumable activation:
        Key 5: Healing Potion / Sitrus Berry (Trigger when HP < 65%)
        Key 6: Combat Burst / Attack Orb (Trigger when Boss active or dense melee combat)
        Key 7: Crowd Control / Speed / Pokédoll (Trigger when surrounded or mob swarm >= 6)
        Key 8: Emergency Shield / Wall Escape / Revive (Trigger when wall-pinned or critical HP < 25%)
        """
        player = unified_state.get("player") or {}
        hp_r = player.get("hp_ratio")
        enemies = unified_state.get("enemies", [])
        boss_active = unified_state.get("boss_info", {}).get("active", False)
        is_encircled = unified_state.get("is_encircled", False)
        min_dist = unified_state.get("min_enemy_dist", 9999.0)
        close_mobs = sum(1 for e in enemies if e.get("dist", 999.0) < 160.0)
        blocked_secs = [s for s, b in unified_state.get("sector_obstacles", {}).items() if b]
        is_pinned = len(blocked_secs) >= 2 and min_dist < 180.0

        # Key 5: Healing
        if hp_r is not None and hp_r < 0.65:
            if now_t - last_times["5"] >= 3.5:
                self.browser_ctrl.press_key("5")
                last_times["5"] = now_t

        # Key 6: Combat Burst
        if boss_active or (min_dist < 110.0 and close_mobs >= 3):
            if now_t - last_times["6"] >= 4.0:
                self.browser_ctrl.press_key("6")
                last_times["6"] = now_t

        # Key 7: Crowd Control / Speed
        if is_encircled or close_mobs >= 6:
            if now_t - last_times["7"] >= 5.0:
                self.browser_ctrl.press_key("7")
                last_times["7"] = now_t

        # Key 8: Emergency Shield / Wall Escape
        if is_pinned or (hp_r is not None and hp_r < 0.25):
            if now_t - last_times["8"] >= 6.0:
                self.browser_ctrl.press_key("8")
                last_times["8"] = now_t
