import copy
import time
import sys
import json
import hashlib
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

def compute_policy_hash(policy: Dict[str, Any]) -> str:
    """Computes a deterministic short SHA-256 hash of core behavioral policy parameters."""
    if not policy:
        return "empty"
    core = {
        "movement": policy.get("movement", {}),
        "upgrade": {
            "tier_priorities": policy.get("upgrade", {}).get("tier_priorities", {}),
            "priority_skills": policy.get("upgrade", {}).get("priority_skills", []),
            "skill_ratings": policy.get("upgrade", {}).get("skill_ratings", {})
        },
        "items": policy.get("items", {})
    }
    raw = json.dumps(core, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]

class LearningEngine:
    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store
        self.current_policy = self.memory.load_policy()
        self.best_policy = self.memory.load_best_policy()
        self.learning_state = self.memory.load_learning_state()
        self._sanitize_eval_block_schema()

    def _sanitize_eval_block_schema(self):
        """
        Sanitizes eval_block_runs to handle legacy formats cleanly.
        Never mixes incompatible records or primitives into an active eval block.
        """
        eval_runs = self.learning_state.get("eval_block_runs", [])
        if not eval_runs:
            return
        
        has_legacy = False
        current_hash = compute_policy_hash(self.current_policy)
        for r in eval_runs:
            if not isinstance(r, dict) or "policy_hash" not in r:
                has_legacy = True
                break

        if has_legacy:
            history = self.learning_state.setdefault("eval_block_history", [])
            history.append({
                "archived_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": copy.deepcopy(eval_runs)
            })
            self.learning_state["eval_block_runs"] = []
            self.memory.save_learning_state(self.learning_state)

    def evaluate_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Policy Gradient & Evolutionary Optimization Cycle:
        1. Separates genuine game_score (kills/defeated), survival_time, max_level, and fitness_score.
        2. Strictly freezes current policy (including skill_ratings) during an evaluation block.
        3. At block completion: evaluates candidate against champion, updates skill ratings with
           context-aware attribution, and formulates candidate policy with ceiling-aware exploration.
        """
        # Separate individual score metrics with explicit provenance
        game_score = run_data.get("game_score")
        game_score_source = run_data.get("game_score_source", "unobserved")
        duration = float(run_data.get("survival_time", run_data.get("duration", 0.0)))
        survival_time_source = run_data.get("survival_time_source", "state_tracker_clock")
        level = int(run_data.get("max_level", 1))
        max_level_source = run_data.get("max_level_source", "state_tracker_verified")
        damage_events = int(run_data.get("damage_events", 0))
        stuck_events = int(run_data.get("stuck_events", 0))
        last_stuck = bool(run_data.get("last_stuck", False))
        last_danger = run_data.get("last_danger_level", "LOW")
        last_enemies = int(run_data.get("last_enemies_count", 0))
        incidents = run_data.get("damage_incidents", [])
        snapshots = run_data.get("trajectory_snapshots", [])
        pokemon = run_data.get("pokemon", "Unknown")
        upgrades = run_data.get("upgrades", [])

        # Internal composite fitness score
        fitness_score = (duration * 3.5) + (level * 35.0) - (damage_events * 2.5) - (min(stuck_events, 6) * 8.0)
        fitness_score = max(10.0, round(fitness_score, 2))
        fitness_score_source = "internal_composite_formula"

        total_runs = self.learning_state.get("total_runs", 0) + 1
        self.learning_state["total_runs"] = total_runs

        recent_scores = self.learning_state.get("recent_scores", [])
        recent_scores.append(round(fitness_score, 2))
        if len(recent_scores) > 20:
            recent_scores.pop(0)
        self.learning_state["recent_scores"] = recent_scores

        best_score = self.best_policy.get("best_score", 0.0)
        best_duration = self.learning_state.get("best_survival_time", 0.0)
        is_new_best = duration > best_duration or (game_score is not None and game_score > self.learning_state.get("best_game_score", 0))

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

        # Current policy metadata
        current_v = str(self.learning_state.get("current_policy_version", self.current_policy.get("version", "5.0")))
        current_hash = compute_policy_hash(self.current_policy)

        # --- 2. RECORD RUN IN FIXED-POLICY EVALUATION BLOCK ---
        eval_block = self.learning_state.setdefault("eval_block_runs", [])
        eval_block.append({
            "run_id": total_runs,
            "policy_version": current_v,
            "policy_hash": current_hash,
            "pokemon": pokemon,
            "game_score": game_score,
            "game_score_source": game_score_source,
            "survival_time": duration,
            "survival_time_source": survival_time_source,
            "max_level": level,
            "max_level_source": max_level_source,
            "fitness_score": fitness_score,
            "fitness_score_source": fitness_score_source,
            "duration": duration,
            "damage_events": damage_events,
            "swarm_damage_count": swarm_damage_count,
            "wall_damage_count": wall_damage_count,
            "hazard_damage_count": hazard_damage_count,
            "encircle_rate": encircle_rate,
            "last_stuck": last_stuck,
            "end_reason": run_data.get("end_reason", "unknown"),
            "upgrades": upgrades
        })

        is_rollback = False
        is_promoted = False
        param_deltas = []
        learned_skill_logs = []

        # Tentative skill attribution for diagnostic learning state
        # NOTE: current_policy["upgrade"]["skill_ratings"] is strictly NOT updated during the block!
        skill_ratings = self.learning_state.setdefault("skill_ratings", {})
        base_run_reward = (duration / 60.0) + (level * 1.2) - (damage_events * 0.08)
        base_run_reward = max(0.2, base_run_reward)

        for u in upgrades:
            s_name = u.get("name", "").strip().lower()
            if not s_name:
                continue
            t_picked = float(u.get("time", 0.0))
            f_active = max(0.0, min(1.0, (duration - t_picked) / max(1.0, duration)))
            effective_active = (f_active ** 1.35)

            tag = u.get("tag", "").lower()
            if any(evo in s_name for evo in ["scald", "blizzard", "fire blast", "overheat", "sacred fire", "take down"]) or "evolution" in tag:
                role_mult = 1.35
            elif s_name in ["water gun", "ember", "ice beam", "hydro pump", "shadow ball", "flamethrower"]:
                role_mult = 1.20
            elif s_name in ["aqua ring", "sitrus berry", "berry", "sprint"]:
                role_mult = 1.15
            else:
                role_mult = 0.90

            skill_reward = base_run_reward * (0.15 + 0.85 * effective_active) * role_mult
            old_r = skill_ratings.get(s_name, 1.0)
            new_r = round((1.0 - 0.20) * old_r + (0.20 * skill_reward), 3)
            if abs(new_r - old_r) > 0.005:
                skill_ratings[s_name] = new_r
                delta = "+" if new_r >= old_r else ""
                learned_skill_logs.append(f"{s_name.title()}: {new_r:.2f} ({delta}{new_r-old_r:.2f}, active={f_active*100:.0f}%)")

        # If evaluation block is still in progress, keep active policy COMPLETELY fixed
        if len(eval_block) < 3:
            print(f"📈 Khối đánh giá: Đã thu thập {len(eval_block)}/3 trận của chính sách v{current_v} [hash: {current_hash[:8]}]. (Chính sách & kỹ năng được giữ cố định hoàn toàn để đo lường công bằng).")
            if learned_skill_logs:
                print(f"  ⭐ Đánh giá kỹ năng lượt này (chưa áp dụng vào chính sách cố định): {', '.join(learned_skill_logs)}")
            self.memory.save_learning_state(self.learning_state)
        else:
            # --- 3. EVALUATION BLOCK COMPLETION ---
            # Evaluate using primary metric: game_score (kills) if available across runs; else survival_time
            valid_game_scores = [r["game_score"] for r in eval_block if r.get("game_score") is not None]
            has_consistent_game_scores = len(valid_game_scores) == len(eval_block) and len(eval_block) > 0

            block_dur_avg = sum((r["duration"] if isinstance(r, dict) else float(r)) for r in eval_block) / float(len(eval_block))
            best_eval_avg = float(self.learning_state.get("best_eval_avg", 0.0))
            if best_eval_avg <= 0.0:
                self.learning_state["best_eval_avg"] = block_dur_avg
                best_eval_avg = block_dur_avg

            if has_consistent_game_scores:
                block_score_avg = sum(valid_game_scores) / float(len(valid_game_scores))
                best_game_score_avg = float(self.learning_state.get("best_game_score_avg", 0.0))
                if best_game_score_avg <= 0.0:
                    self.learning_state["best_game_score_avg"] = block_score_avg
                    best_game_score_avg = block_score_avg

                print(f"\n📊 ĐÁNH GIÁ KHỐI 3 TRẬN (GAME SCORE): Điểm Game TB = {block_score_avg:.1f} (Baseline: {best_game_score_avg:.1f}) | Thời gian TB = {block_dur_avg:.1f}s")
                
                # Primary comparison on game_score
                if block_score_avg < best_game_score_avg * 0.85 and best_game_score_avg > 50.0:
                    print(f"⚠️ [ROLLBACK] Chính sách v{current_v} làm giảm điểm game ({block_score_avg:.1f} < {best_game_score_avg:.1f}). TỰ ĐỘNG ROLLBACK VỀ BEST POLICY (CHAMPION)!")
                    self.current_policy = copy.deepcopy(self.best_policy)
                    self.learning_state["current_policy_version"] = self.best_policy.get("version", current_v)
                    is_rollback = True
                elif block_score_avg > best_game_score_avg * 1.05:
                    new_v = round(float(current_v) + 0.1, 1)
                    self.current_policy["version"] = str(new_v)
                    self.learning_state["current_policy_version"] = str(new_v)
                    self.learning_state["best_policy_version"] = str(new_v)
                    self.learning_state["best_game_score_avg"] = block_score_avg
                    self.learning_state["best_eval_avg"] = block_dur_avg
                    self.best_policy = copy.deepcopy(self.current_policy)
                    self.memory.save_best_policy(self.best_policy, block_score_avg)
                    is_promoted = True
                    print(f"🏆 [THĂNG CẤP] Khối 3 trận đạt điểm Game TB {block_score_avg:.1f} (+{(block_score_avg - best_game_score_avg):.1f})! Chính thức thăng cấp chính sách lên v{new_v}!")
            else:
                print(f"\n📊 ĐÁNH GIÁ KHỐI 3 TRẬN: Thời gian sống sót TB = {block_dur_avg:.1f}s (Baseline chuẩn: {best_eval_avg:.1f}s)")
                if block_dur_avg < best_eval_avg * 0.85 and best_eval_avg > 60.0:
                    print(f"⚠️ [ROLLBACK] Chính sách v{current_v} làm giảm thời gian sống ({block_dur_avg:.1f}s < {best_eval_avg:.1f}s). TỰ ĐỘNG ROLLBACK VỀ BEST POLICY (CHAMPION)!")
                    self.current_policy = copy.deepcopy(self.best_policy)
                    self.learning_state["current_policy_version"] = self.best_policy.get("version", current_v)
                    is_rollback = True
                elif block_dur_avg > best_eval_avg * 1.05:
                    new_v = round(float(current_v) + 0.1, 1)
                    self.current_policy["version"] = str(new_v)
                    self.learning_state["current_policy_version"] = str(new_v)
                    self.learning_state["best_policy_version"] = str(new_v)
                    self.learning_state["best_eval_avg"] = block_dur_avg
                    self.best_policy = copy.deepcopy(self.current_policy)
                    self.memory.save_best_policy(self.best_policy, fitness_score)
                    is_promoted = True
                    print(f"🏆 [THĂNG CẤP] Khối 3 trận đạt TB {block_dur_avg:.1f}s (+{(block_dur_avg - best_eval_avg):.1f}s)! Chính thức thăng cấp chính sách lên v{new_v}!")

            # --- 4. CONTEXT-AWARE SKILL ATTRIBUTION (EMA VALUE NETWORK) ---
            # Evaluated ONLY when the block finishes
            skill_ratings = self.learning_state.setdefault("skill_ratings", {})
            for r in eval_block:
                r_dur = float(r.get("duration", 0.0) if isinstance(r, dict) else r)
                r_lvl = float(r.get("max_level", 1) if isinstance(r, dict) else 1)
                r_dmg = float(r.get("damage_events", 0) if isinstance(r, dict) else 0)
                base_run_reward = (r_dur / 60.0) + (r_lvl * 1.2) - (r_dmg * 0.08)
                base_run_reward = max(0.2, base_run_reward)

                r_upgrades = r.get("upgrades", []) if isinstance(r, dict) else []
                for u in r_upgrades:
                    s_name = u.get("name", "").strip().lower()
                    if not s_name:
                        continue
                    t_picked = float(u.get("time", 0.0))
                    f_active = max(0.0, min(1.0, (r_dur - t_picked) / max(1.0, r_dur)))
                    # Non-linear active penalty: skills picked near the end receive sharply discounted credit
                    effective_active = (f_active ** 1.35)

                    tag = u.get("tag", "").lower()
                    if any(evo in s_name for evo in ["scald", "blizzard", "fire blast", "overheat", "sacred fire", "take down"]) or "evolution" in tag:
                        role_mult = 1.35
                    elif s_name in ["water gun", "ember", "ice beam", "hydro pump", "shadow ball", "flamethrower"]:
                        role_mult = 1.20
                    elif s_name in ["aqua ring", "sitrus berry", "berry", "sprint"]:
                        role_mult = 1.15
                    else:
                        role_mult = 0.90

                    skill_reward = base_run_reward * (0.15 + 0.85 * effective_active) * role_mult
                    old_r = skill_ratings.get(s_name, 1.0)
                    new_r = round((1.0 - 0.20) * old_r + (0.20 * skill_reward), 3)
                    if abs(new_r - old_r) > 0.005:
                        skill_ratings[s_name] = new_r
                        delta = "+" if new_r >= old_r else ""
                        learned_skill_logs.append(f"{s_name.title()}: {new_r:.2f} ({delta}{new_r-old_r:.2f}, active={f_active*100:.0f}%)")

            self.current_policy.setdefault("upgrade", {})["skill_ratings"] = skill_ratings

            # --- 5. CANDIDATE POLICY GENERATION (CONTROLLED SEARCH WITH CEILING HANDLING) ---
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

            # Diagnostic trigger analysis:
            # Swarm Penalty & Secondary Levers
            if cur_mob_penalty >= 150.0 - 0.001:
                if agg_swarm > 0 or agg_encircle > 0.08:
                    new_corridor = min(550.0, cur_corridor_bonus + 40.0)
                    new_wall_thrust = min(0.55, cur_wall_thrust + 0.04)
                    
                    # Verify real delta before logging
                    changed = False
                    if abs(new_corridor - cur_corridor_bonus) > 1e-4:
                        mov["mob_clear_corridor_bonus"] = new_corridor
                        changed = True
                    if abs(new_wall_thrust - cur_wall_thrust) > 1e-4:
                        mov["wall_breakout_thrust"] = new_wall_thrust
                        changed = True

                    if changed:
                        param_deltas.append(f"[TRẦN THAM SỐ] mob_penalty đã đạt trần (150). Kích hoạt đòn bẩy phụ: corridor_bonus ({cur_corridor_bonus:.0f} -> {new_corridor:.0f}) & wall_thrust ({cur_wall_thrust:.2f} -> {new_wall_thrust:.2f}s)")
                    else:
                        # Both primary and secondary at ceiling: explore alternative parameter (compact safe zone)
                        if cur_danger_thresh > 155.0:
                            new_danger = max(155.0, cur_danger_thresh - 10.0)
                            mov["danger_threshold"] = new_danger
                            param_deltas.append(f"[THỬ NGHIỆM CHIẾN THUẬT] Tránh né quá mức khiến kiting dao động: giảm danger_threshold ({cur_danger_thresh:.0f}px -> {new_danger:.0f}px) để di chuyển ổn định tiêu diệt quái")
            elif agg_swarm > 0 or agg_encircle > 0.08:
                delta_mob = min(25.0, 5.0 + agg_swarm * 1.5)
                new_mob_pen = min(150.0, cur_mob_penalty + delta_mob)
                if abs(new_mob_pen - cur_mob_penalty) > 1e-4:
                    param_deltas.append(f"Trọng số phạt đâm quái (mob_penalty): {cur_mob_penalty:.0f} -> {new_mob_pen:.0f} (+{new_mob_pen - cur_mob_penalty:.0f})")
                    mov["mob_penalty_weight"] = new_mob_pen

            # Danger Threshold
            if cur_danger_thresh >= 185.0 - 0.001:
                if agg_swarm > 0:
                    new_commit = min(0.40, cur_dir_commit + 0.03)
                    if abs(new_commit - cur_dir_commit) > 1e-4:
                        param_deltas.append(f"[TRẦN THAM SỐ] danger_threshold đã đạt trần (185px). Kích hoạt đòn bẩy phụ: dir_commit ({cur_dir_commit:.2f} -> {new_commit:.2f}s)")
                        mov["dir_commit_duration"] = new_commit
            elif agg_swarm > 0 or agg_encircle > 0.08:
                delta_danger = min(12.0, 4.0 + agg_swarm * 1.0)
                new_danger = min(185.0, cur_danger_thresh + delta_danger)
                if abs(new_danger - cur_danger_thresh) > 1e-4:
                    param_deltas.append(f"Vành đai an toàn (danger_threshold): {cur_danger_thresh:.0f}px -> {new_danger:.0f}px (+{new_danger - cur_danger_thresh:.0f}px)")
                    mov["danger_threshold"] = new_danger

            # Wall Avoidance & Thrust
            if cur_wall_avoid >= 2.5 - 0.001:
                if agg_wall > 0 or agg_stuck:
                    new_wall_thrust = min(0.55, cur_wall_thrust + 0.05)
                    if abs(new_wall_thrust - cur_wall_thrust) > 1e-4:
                        param_deltas.append(f"[TRẦN THAM SỐ] wall_avoidance đã đạt trần (2.5). Kích hoạt đòn bẩy phụ: wall_thrust ({cur_wall_thrust:.2f} -> {new_wall_thrust:.2f}s)")
                        mov["wall_breakout_thrust"] = new_wall_thrust
            elif agg_wall > 0 or agg_stuck:
                new_wall_avoid = min(2.5, cur_wall_avoid + 0.20)
                new_wall_thrust = min(0.52, cur_wall_thrust + 0.03)
                if abs(new_wall_avoid - cur_wall_avoid) > 1e-4:
                    param_deltas.append(f"Độ né tránh góc tường (wall_avoidance): {cur_wall_avoid:.2f} -> {new_wall_avoid:.2f} (+0.20)")
                    mov["wall_avoidance_weight"] = new_wall_avoid
                if abs(new_wall_thrust - cur_wall_thrust) > 1e-4:
                    param_deltas.append(f"Lực phóng bứt vách (wall_breakout_thrust): {cur_wall_thrust:.2f}s -> {new_wall_thrust:.2f}s (+0.03s)")
                    mov["wall_breakout_thrust"] = new_wall_thrust

            # Evasion & Dodge Hold
            if agg_hazard > 0:
                new_dodge_hold = min(0.45, cur_dodge_hold + 0.02)
                new_evasion = min(0.99, cur_evasion + 0.02)
                if abs(new_dodge_hold - cur_dodge_hold) > 1e-4:
                    param_deltas.append(f"Thời gian giữ sải né đạn (dodge_hold): {cur_dodge_hold:.2f}s -> {new_dodge_hold:.2f}s (+0.02s)")
                    mov["dodge_hold_duration"] = new_dodge_hold
                if abs(new_evasion - cur_evasion) > 1e-4:
                    mov["evasion_weight"] = new_evasion

            # EXP Vacuuming
            if block_dur_avg > 75.0:
                new_exp = min(0.70, cur_exp_weight + 0.04)
                if abs(new_exp - cur_exp_weight) > 1e-4:
                    param_deltas.append(f"Ưu tiên hút ngọc EXP (exp_harvest_weight): {cur_exp_weight:.2f} -> {new_exp:.2f} (+0.04)")
                    mov["exp_harvest_weight"] = new_exp

            self.current_policy = adapted_policy

            # Reset evaluation block runs for next cycle
            self.learning_state["eval_block_runs"] = []
            self.memory.save_policy(self.current_policy)
            self.memory.save_learning_state(self.learning_state)
            self.memory.save_candidate_policy(self.current_policy)

            if param_deltas:
                print("\n📊 BẢNG CẬP NHẬT TRỌNG SỐ CHO KHỐI TIẾP THEO:")
                for delta_msg in param_deltas:
                    print(f"  ⚡ {delta_msg}")
            if learned_skill_logs:
                print(f"  ⭐ Xếp hạng kỹ năng: {', '.join(learned_skill_logs)}")

        if is_new_best and not is_rollback:
            self.learning_state["best_survival_time"] = max(best_duration, duration)
            self.learning_state["best_level"] = max(self.learning_state.get("best_level", 1), level)
            if game_score is not None:
                self.learning_state["best_game_score"] = max(self.learning_state.get("best_game_score", 0), game_score)
            self.memory.save_learning_state(self.learning_state)

        next_hash = compute_policy_hash(self.current_policy)
        print(f"💾 Chính sách sẵn sàng cho ván tiếp theo (v{self.current_policy.get('version', '5.0')} [hash: {next_hash[:8]}]).")
        print("=" * 60 + "\n")

        if len(eval_block) < 3:
            block_adaptations = [f"[Khối đánh giá {len(eval_block)}/3] Chính sách & kỹ năng được giữ cố định hoàn toàn"]
        else:
            block_adaptations = param_deltas + ([f"Kỹ năng: {', '.join(learned_skill_logs)}"] if learned_skill_logs else [])

        analysis = {
            "score": round(fitness_score, 2),
            "fitness_score": round(fitness_score, 2),
            "game_score": game_score,
            "game_score_source": game_score_source,
            "survival_time": duration,
            "survival_time_source": survival_time_source,
            "max_level": level,
            "max_level_source": max_level_source,
            "policy_version": current_v,
            "policy_hash": current_hash,
            "is_new_best": is_new_best,
            "is_rollback": is_rollback,
            "is_promoted": is_promoted,
            "previous_best_score": round(best_score, 2),
            "previous_best_duration": round(best_duration, 1),
            "total_runs": total_runs,
            "death_cause": death_cause,
            "death_summary": death_summary,
            "adaptations": block_adaptations
        }
        return analysis