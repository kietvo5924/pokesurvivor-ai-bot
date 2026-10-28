import copy
import time
import sys
from typing import Dict, Any, List, Optional
from pokesurvivor.memory_store import MemoryStore

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

class LearningEngine:
    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store
        self.current_policy = self.memory.load_policy()
        self.best_policy = self.memory.load_best_policy()
        self.learning_state = self.memory.load_learning_state()

    def evaluate_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trajectory-Informed Policy Gradient (TIPG) & Evolutionary Optimization:
        1. Analyzes telemetry snapshots and collision incidents from the match.
        2. Diagnoses exact failure modes (Wall trapping vs Swarm encirclement vs Bullet attrition).
        3. Allows realistic ML compute/thinking time (2.5 - 3.5s) to backpropagate weight deltas.
        4. Updates and clamps parametric policy weights (danger_threshold, mob_penalty_weight,
           mob_clear_corridor_bonus, wall_avoidance_weight, wall_breakout_thrust, dodge_hold_duration).
        5. Saves evolved policy to disk for the next match.
        """
        duration = run_data.get("duration", 0.0)
        level = run_data.get("max_level", 1)
        damage_events = run_data.get("damage_events", 0)
        stuck_events = run_data.get("stuck_events", 0)
        last_stuck = run_data.get("last_stuck", False)
        last_danger = run_data.get("last_danger_level", "LOW")
        last_enemies = run_data.get("last_enemies_count", 0)
        incidents = run_data.get("damage_incidents", [])
        snapshots = run_data.get("trajectory_snapshots", [])

        # Composite fitness score:
        # Duration weight (+3.5/s) heavily rewards reaching late-game hordes (5-15+ mins).
        # Level attainment (+35.0/lvl) heavily rewards XP farming and evolution spikes.
        # Penalties: actual damage incidents (-2.5), and edge-triggered wall traps clamped to max 6 (-8.0).
        score = (duration * 3.5) + (level * 35.0) - (damage_events * 2.5) - (min(stuck_events, 6) * 8.0)
        score = max(10.0, score)

        total_runs = self.learning_state.get("total_runs", 0) + 1
        self.learning_state["total_runs"] = total_runs

        recent_scores = self.learning_state.get("recent_scores", [])
        recent_scores.append(round(score, 2))
        if len(recent_scores) > 20:
            recent_scores.pop(0)
        self.learning_state["recent_scores"] = recent_scores

        best_score = self.best_policy.get("best_score", 0.0)
        best_duration = self.learning_state.get("best_survival_time", 0.0)
        is_new_best = duration > best_duration or score > best_score

        # --- 1. TRAJECTORY ERROR DIAGNOSTICS ---
        wall_damage_count = sum(1 for inc in incidents if inc.get("near_wall"))
        swarm_damage_count = sum(1 for inc in incidents if inc.get("is_encircled") or inc.get("min_enemy_dist", 999.0) < 140.0)
        hazard_damage_count = sum(1 for inc in incidents if inc.get("hazards_count", 0) > 0)
        encircled_events = run_data.get("encircled_events", 0)
        total_frames = max(1, run_data.get("total_frames", 1))
        encircle_rate = float(encircled_events) / float(total_frames)

        if last_stuck or wall_damage_count >= 3:
            death_cause = "OBSTACLE_TRAP"
            death_summary = f"Bị kẹt góc/vách đá địa hình ({wall_damage_count} lần dính đòn sát tường)"
        elif last_enemies > 4 or last_danger == "CRITICAL" or swarm_damage_count > 4:
            death_cause = "SWARM_OVERWHELM"
            death_summary = f"Bị quái dồn ép và bao vây ({swarm_damage_count} lần bị áp sát < 140px)"
        elif damage_events > 2:
            death_cause = "ATTRITION_DAMAGE"
            death_summary = f"Mất máu tích lũy qua thời gian kiting ({damage_events} lần trúng đòn)"
        elif run_data.get("end_reason") == "timeout":
            death_cause = "TIMEOUT"
            death_summary = f"Hết thời gian tối đa ván ({duration:.1f}s)"
        else:
            # Honest telemetry: never assume cause without conclusive data
            death_cause = "unknown"
            death_summary = f"Không đủ dữ liệu xác định nguyên nhân chết tại mốc {duration:.1f}s (Level {level})"

        print("\n" + "=" * 60)
        print("🧠 BẮT ĐẦU CHU KỲ HỌC MÁY & TỐI ƯU HÓA CHÍNH SÁCH (RL ENGINE)")
        print("=" * 60)
        print(f"📥 Nạp dữ liệu quỹ đạo: {len(snapshots)} snapshots, {len(incidents)} điểm va chạm mất máu.")
        print(f"🔍 Chẩn đoán sự cố thực chiến:")
        print(f"   - Va chạm quái cự ly gần / vòng vây : {swarm_damage_count} lần (Tỷ lệ vây: {encircle_rate*100:.1f}%)")
        print(f"   - Va chạm góc cụt / vách đá        : {wall_damage_count} lần")
        print(f"   - Đòn đánh từ chiêu thức / đạn nguy : {hazard_damage_count} lần")
        print(f"   - Kết luận nguyên nhân kết thúc    : {death_cause} ({death_summary})")
        print("⚙️ Đang tính toán Gradient thích nghi & Cập nhật tham số...")

        # --- 3. INDIVIDUALIZED SKILL ATTRIBUTION (EMA VALUE NETWORK) ---
        skill_ratings = self.learning_state.setdefault("skill_ratings", {})
        base_run_reward = (duration / 60.0) + (level * 1.2) - (damage_events * 0.08)
        base_run_reward = max(0.2, base_run_reward)

        upgrades = run_data.get("upgrades", [])
        learned_skill_logs = []
        for u in upgrades:
            s_name = u.get("name", "").strip().lower()
            if not s_name:
                continue
            t_picked = float(u.get("time", 0.0))
            f_active = max(0.1, min(1.0, (duration - t_picked) / max(1.0, duration)))

            # Role multiplier
            tag = u.get("tag", "").lower()
            if any(evo in s_name for evo in ["scald", "blizzard", "fire blast", "overheat", "sacred fire", "take down"]) or "evolution" in tag:
                role_mult = 1.35
            elif s_name in ["water gun", "ember", "ice beam", "hydro pump", "shadow ball", "flamethrower"]:
                role_mult = 1.20
            elif s_name in ["aqua ring", "sitrus berry", "berry", "sprint"]:
                role_mult = 1.15
            else:
                role_mult = 0.90

            skill_reward = base_run_reward * (0.35 + 0.65 * f_active) * role_mult
            old_r = skill_ratings.get(s_name, 1.0)
            new_r = round((1.0 - 0.25) * old_r + (0.25 * skill_reward), 3)
            skill_ratings[s_name] = new_r
            delta = "+" if new_r >= old_r else ""
            learned_skill_logs.append(f"{s_name.title()}: {new_r:.2f} ({delta}{new_r-old_r:.2f}, active={f_active*100:.0f}%)")

        self.current_policy.setdefault("upgrade", {})["skill_ratings"] = skill_ratings

        # --- 4. MULTI-MATCH EVALUATION BLOCK & STATISTICAL ROLLBACK ---
        eval_block = self.learning_state.setdefault("eval_block_runs", [])
        eval_block.append({
            "duration": duration,
            "score": score,
            "level": level,
            "damage_events": damage_events,
            "swarm_damage_count": swarm_damage_count,
            "wall_damage_count": wall_damage_count,
            "hazard_damage_count": hazard_damage_count,
            "encircle_rate": encircle_rate,
            "last_stuck": last_stuck
        })

        current_v = float(self.learning_state.get("current_policy_version", "5.0"))
        is_rollback = False
        is_promoted = False
        param_deltas = []

        if len(eval_block) < 3:
            print(f"📈 Khối đánh giá: Đã thu thập {len(eval_block)}/3 trận của chính sách v{current_v}. (Chính sách được giữ cố định để đo lường công bằng).")
            if learned_skill_logs:
                print(f"  ⭐ Xếp hạng kỹ năng trận này: {', '.join(learned_skill_logs)}")
            self.memory.save_learning_state(self.learning_state)
        else:
            # Complete 3-run evaluation block under identical policy
            block_avg = sum((r["duration"] if isinstance(r, dict) else float(r)) for r in eval_block) / float(len(eval_block))
            best_eval_avg = self.learning_state.get("best_eval_avg", 0.0)
            if best_eval_avg <= 0.0:
                self.learning_state["best_eval_avg"] = block_avg
                best_eval_avg = block_avg

            print(f"\n📊 ĐÁNH GIÁ KHỐI 3 TRẬN: Thời gian sống sót TB = {block_avg:.1f}s (Baseline chuẩn: {best_eval_avg:.1f}s)")

            # Check for regression (> 15% drop compared to best baseline)
            if block_avg < best_eval_avg * 0.85 and best_eval_avg > 60.0:
                print(f"⚠️ [ROLLBACK] Chính sách v{current_v} làm giảm thời gian sống ({block_avg:.1f}s < {best_eval_avg:.1f}s). TỰ ĐỘNG ROLLBACK VỀ BEST POLICY!")
                self.current_policy = copy.deepcopy(self.best_policy)
                is_rollback = True
            else:
                if block_avg > best_eval_avg * 1.05:
                    new_v = round(current_v + 0.1, 1)
                    self.current_policy["version"] = str(new_v)
                    self.learning_state["current_policy_version"] = str(new_v)
                    self.learning_state["best_policy_version"] = str(new_v)
                    self.learning_state["best_eval_avg"] = block_avg
                    self.best_policy = copy.deepcopy(self.current_policy)
                    self.memory.save_best_policy(self.best_policy, score)
                    is_promoted = True
                    print(f"🏆 [THĂNG CẤP] Khối 3 trận đạt TB {block_avg:.1f}s (+{(block_avg-best_eval_avg):.1f}s)! Chính thức thăng cấp chính sách lên v{new_v}!")

                # Formulate next candidate policy using aggregated telemetry
                adapted_policy = copy.deepcopy(self.current_policy)
                mov = adapted_policy.setdefault("movement", {})
                cur_danger_thresh = float(mov.get("danger_threshold", 160.0))
                cur_mob_penalty = float(mov.get("mob_penalty_weight", 85.0))
                cur_corridor_bonus = float(mov.get("mob_clear_corridor_bonus", 350.0))
                cur_wall_avoid = float(mov.get("wall_avoidance_weight", 1.4))
                cur_wall_thrust = float(mov.get("wall_breakout_thrust", 0.40))
                cur_dodge_hold = float(mov.get("dodge_hold_duration", 0.35))
                cur_dir_commit = float(mov.get("dir_commit_duration", 0.28))
                cur_exp_weight = float(mov.get("exp_harvest_weight", 0.45))
                cur_evasion = float(mov.get("evasion_weight", 0.95))

                agg_swarm = sum(r.get("swarm_damage_count", 0) for r in eval_block if isinstance(r, dict))
                agg_wall = sum(r.get("wall_damage_count", 0) for r in eval_block if isinstance(r, dict))
                agg_hazard = sum(r.get("hazard_damage_count", 0) for r in eval_block if isinstance(r, dict))
                agg_encircle = sum(r.get("encircle_rate", 0.0) for r in eval_block if isinstance(r, dict)) / float(max(1, len(eval_block)))
                agg_stuck = any(r.get("last_stuck", False) for r in eval_block if isinstance(r, dict))

                # Controlled Search with Parameter Ceiling Handling:
                # Swarm Penalty & Corridor
                if cur_mob_penalty >= 150.0 - 0.001:
                    if agg_swarm > 0 or agg_encircle > 0.08:
                        new_corridor = min(550.0, cur_corridor_bonus + 40.0)
                        new_wall_thrust = min(0.55, cur_wall_thrust + 0.04)
                        param_deltas.append(f"[TRẦN THAM SỐ] mob_penalty_weight đã đạt trần (150). Kích hoạt đòn bẩy phụ: corridor_bonus ({cur_corridor_bonus:.0f} -> {new_corridor:.0f}) & wall_thrust ({cur_wall_thrust:.2f} -> {new_wall_thrust:.2f}s)!")
                        mov["mob_clear_corridor_bonus"] = new_corridor
                        mov["wall_breakout_thrust"] = new_wall_thrust
                elif agg_swarm > 0 or agg_encircle > 0.08:
                    delta_mob = min(25.0, 5.0 + agg_swarm * 1.5)
                    new_mob_pen = min(150.0, cur_mob_penalty + delta_mob)
                    param_deltas.append(f"Trọng số phạt đâm quái (mob_penalty): {cur_mob_penalty:.0f} -> {new_mob_pen:.0f} (+{new_mob_pen - cur_mob_penalty:.0f})")
                    mov["mob_penalty_weight"] = new_mob_pen

                # Danger Threshold
                if cur_danger_thresh >= 185.0 - 0.001:
                    if agg_swarm > 0:
                        new_commit = min(0.40, cur_dir_commit + 0.03)
                        param_deltas.append(f"[TRẦN THAM SỐ] danger_threshold đã đạt trần (185px). Kích hoạt đòn bẩy phụ: dir_commit ({cur_dir_commit:.2f} -> {new_commit:.2f}s)!")
                        mov["dir_commit_duration"] = new_commit
                elif agg_swarm > 0 or agg_encircle > 0.08:
                    delta_danger = min(12.0, 4.0 + agg_swarm * 1.0)
                    new_danger = min(185.0, cur_danger_thresh + delta_danger)
                    param_deltas.append(f"Vành đai an toàn (danger_threshold): {cur_danger_thresh:.0f}px -> {new_danger:.0f}px (+{new_danger - cur_danger_thresh:.0f}px)")
                    mov["danger_threshold"] = new_danger

                # Wall Avoidance & Thrust
                if cur_wall_avoid >= 2.5 - 0.001:
                    if agg_wall > 0 or agg_stuck:
                        new_wall_thrust = min(0.55, cur_wall_thrust + 0.05)
                        param_deltas.append(f"[TRẦN THAM SỐ] wall_avoidance_weight đã đạt trần (2.5). Kích hoạt đòn bẩy phụ: wall_thrust ({cur_wall_thrust:.2f} -> {new_wall_thrust:.2f}s)!")
                        mov["wall_breakout_thrust"] = new_wall_thrust
                elif agg_wall > 0 or agg_stuck:
                    new_wall_avoid = min(2.5, cur_wall_avoid + 0.20)
                    new_wall_thrust = min(0.52, cur_wall_thrust + 0.03)
                    param_deltas.append(f"Độ né tránh góc tường (wall_avoidance): {cur_wall_avoid:.2f} -> {new_wall_avoid:.2f} (+0.20)")
                    param_deltas.append(f"Lực phóng bứt vách (wall_breakout_thrust): {cur_wall_thrust:.2f}s -> {new_wall_thrust:.2f}s (+0.03s)")
                    mov["wall_avoidance_weight"] = new_wall_avoid
                    mov["wall_breakout_thrust"] = new_wall_thrust

                # Evasion & Dodge Hold
                if agg_hazard > 0:
                    new_dodge_hold = min(0.45, cur_dodge_hold + 0.02)
                    new_evasion = min(0.99, cur_evasion + 0.02)
                    param_deltas.append(f"Thời gian giữ sải né đạn (dodge_hold): {cur_dodge_hold:.2f}s -> {new_dodge_hold:.2f}s (+0.02s)")
                    mov["dodge_hold_duration"] = new_dodge_hold
                    mov["evasion_weight"] = new_evasion

                # EXP Acceleration
                if block_avg > 75.0:
                    new_exp = min(0.65, cur_exp_weight + 0.04)
                    param_deltas.append(f"Ưu tiên hút ngọc EXP (exp_harvest_weight): {cur_exp_weight:.2f} -> {new_exp:.2f} (+0.04)")
                    mov["exp_harvest_weight"] = new_exp

                self.current_policy = adapted_policy

            # Reset evaluation block runs for next 3-match cycle
            self.learning_state["eval_block_runs"] = []
            self.memory.save_policy(self.current_policy)
            self.memory.save_learning_state(self.learning_state)

            if param_deltas:
                print("\n📊 BẢNG CẬP NHẬT TRỌNG SỐ CHO KHỐI TIẾP THEO:")
                for delta_msg in param_deltas:
                    print(f"  ⚡ {delta_msg}")
            if learned_skill_logs:
                print(f"  ⭐ Xếp hạng kỹ năng: {', '.join(learned_skill_logs)}")

        if is_new_best and not is_rollback:
            self.learning_state["best_survival_time"] = max(best_duration, duration)
            self.learning_state["best_level"] = max(self.learning_state.get("best_level", 1), level)
            self.memory.save_learning_state(self.learning_state)

        print(f"💾 Chính sách sẵn sàng cho ván tiếp theo (v{self.current_policy.get('version', '5.0')}).")
        print("=" * 60 + "\n")

        analysis = {
            "score": round(score, 2),
            "is_new_best": is_new_best,
            "is_rollback": is_rollback,
            "is_promoted": is_promoted,
            "previous_best_score": round(best_score, 2),
            "previous_best_duration": round(best_duration, 1),
            "total_runs": total_runs,
            "death_cause": death_cause,
            "death_summary": death_summary,
            "adaptations": param_deltas + ([f"Kỹ năng: {', '.join(learned_skill_logs)}"] if learned_skill_logs else [])
        }
        return analysis
