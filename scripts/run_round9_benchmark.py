import os
import sys
import time
import math
import random
import json
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokesurvivor.danger_estimator import DIR16_ANGLES, DIR16_KEYS
from pokesurvivor.movement_policy import MovementPolicy
from pokesurvivor.decision_tracer import DecisionTracer
from pokesurvivor.trace_replayer import TraceReplayer
from pokesurvivor.run_manager import LoopProfiler
from pokesurvivor.config import DATA_DIR

def run_controlled_simulation_benchmark(episodes: int = 10) -> Dict[str, Any]:
    """
    Controlled A/B benchmark evaluation:
    Runs simulated match episodes with dynamic enemy spawns, crossing projectiles,
    hazard detonations, and EXP gems.
    Records decisions into data/decision_trace.jsonl and calculates statistical distribution.
    """
    policy_config = {
        "movement": {
            "pattern": "kiting",
            "danger_threshold": 175.0,
            "mob_penalty_weight": 150.0,
            "mob_clear_corridor_bonus": 550.0,
            "wall_avoidance_weight": 2.5,
            "wall_breakout_thrust": 0.55,
            "dodge_hold_duration": 0.45,
            "dir_commit_duration": 0.35,
            "exp_harvest_weight": 0.70,
            "step_duration": 0.22,
            "rollout_horizon": 0.80,
            "speed_scale": 1.65
        }
    }
    policy = MovementPolicy(policy_config)
    tracer = DecisionTracer(os.path.join(DATA_DIR, "decision_trace.jsonl"))
    profiler = LoopProfiler(window_size=200)

    results = []

    print(f"🚀 BẮT ĐẦU BENCHMARK THỰC NGHIỆM MÔ PHỎNG ({episodes} VÁN ĐỐI CHỨNG)")
    print("=" * 65)

    random.seed(42)  # Deterministic seed for reproducible evaluation

    for ep in range(1, episodes + 1):
        # Episode simulation parameters
        # Mobs scale with time: early game (few mobs) -> late game (dense swarm)
        sim_time = 0.0
        dt = 0.025  # 40Hz simulation tick
        cx, cy = 640.0, 360.0
        hp = 100.0
        max_hp = 100.0
        score = 0
        kills = 0
        level = 1
        exp = 0.0
        exp_next = 20.0

        # Map bounds: 0 to 1280 x, 0 to 720 y
        map_bounds = {"minX": 0.0, "maxX": 1280.0, "minY": 0.0, "maxY": 720.0}

        # Entities in arena
        mobs = []
        hazards = []
        gems = []

        is_alive = True
        collisions_count = 0
        grazes_count = 0

        while is_alive and sim_time < 1800.0:  # Max 30 minutes (1800s)
            sim_time += dt

            # Spawn mobs periodically based on match stage
            target_mob_count = min(60, int(6 + (sim_time / 45.0) * 3))
            while len(mobs) < target_mob_count:
                # Spawn at border
                side = random.choice(["top", "bottom", "left", "right"])
                if side == "top": mx, my = random.uniform(50, 1230), 20.0
                elif side == "bottom": mx, my = random.uniform(50, 1230), 700.0
                elif side == "left": mx, my = 20.0, random.uniform(50, 670)
                else: mx, my = 1260.0, random.uniform(50, 670)
                m_speed = random.uniform(65.0, 95.0) + min(35.0, sim_time * 0.02)
                is_boss = (sim_time > 180.0 and random.random() < 0.05 and not any(m.get("is_boss") for m in mobs))
                mobs.append({
                    "x": mx, "y": my, "speed": m_speed, "is_boss": is_boss,
                    "hp": 250.0 if is_boss else 35.0
                })

            # Spawn projectiles periodically
            if sim_time > 30.0 and random.random() < (0.015 + min(0.04, sim_time * 0.00003)):
                p_angle = random.uniform(0, 2 * math.pi)
                spawn_d = random.uniform(220, 320)
                px = cx + math.cos(p_angle) * spawn_d
                py = cy + math.sin(p_angle) * spawn_d
                to_player_ang = math.atan2(cy - py, cx - px) + random.uniform(-0.25, 0.25)
                p_spd = random.uniform(160.0, 240.0)
                hazards.append({
                    "x": px, "y": py,
                    "vx": p_spd * math.cos(to_player_ang),
                    "vy": p_spd * math.sin(to_player_ang),
                    "type": "projectile",
                    "radius": 22.0,
                    "ttc": spawn_d / p_spd
                })

            # Update mobs motion towards player
            for m in mobs:
                mdx = cx - m["x"]
                mdy = cy - m["y"]
                md = math.hypot(mdx, mdy)
                if md > 1.0:
                    m["vx"] = (mdx / md) * m["speed"]
                    m["vy"] = (mdy / md) * m["speed"]
                    m["x"] += m["vx"] * dt
                    m["y"] += m["vy"] * dt
                    m["dist"] = md

            # Update hazards motion
            active_hazards = []
            for h in hazards:
                h["x"] += h["vx"] * dt
                h["y"] += h["vy"] * dt
                h["dist"] = math.hypot(cx - h["x"], cy - h["y"])
                h["ttc"] = max(0.0, h["ttc"] - dt)
                if 0 <= h["x"] <= 1280 and 0 <= h["y"] <= 720:
                    active_hazards.append(h)
            hazards = active_hazards

            # Construct perception vision state
            vision_state = {
                "player": {
                    "world_x": cx, "world_y": cy, "x": cx, "y": cy,
                    "speed": 100.0, "hp_ratio": hp / max_hp, "invulnerable": False
                },
                "player_pos": (cx, cy),
                "enemies": mobs,
                "hazards": hazards,
                "gems": gems,
                "sector_obstacles": {},
                "sector_obs_costs": {},
                "terrain": {"mapBounds": map_bounds, "status": "verified"},
                "terrain_status": "verified",
                "danger_level": "HIGH" if any(m["dist"] < 90 for m in mobs) else "LOW"
            }

            # Planner decision
            t_dec_start = time.perf_counter()
            chosen_dir, chosen_keys, dur = policy.select_movement(vision_state, current_level=level, elapsed_time=sim_time)
            profiler.record("decision", time.perf_counter() - t_dec_start)

            # Move player in chosen direction
            deg = DIR16_ANGLES.get(chosen_dir, 0.0)
            rad = math.radians(deg)
            pv_spd = 165.0
            cx = max(35.0, min(1245.0, cx + pv_spd * math.cos(rad) * dt))
            cy = max(35.0, min(685.0, cy + pv_spd * math.sin(rad) * dt))

            # Record decision trace periodically (every 0.5s)
            if int(sim_time / 0.5) != int((sim_time - dt) / 0.5):
                tracer.record_decision(
                    elapsed_time=sim_time,
                    player_state={"x": cx, "y": cy, "hp_ratio": hp / max_hp, "speed": 100.0},
                    candidate_scores=getattr(policy, "last_candidate_scores", {}),
                    chosen_direction=chosen_dir,
                    chosen_keys=chosen_keys,
                    risk_breakdown=getattr(policy, "last_risk_breakdown", {}),
                    level=level,
                    game_score=score
                )
                tracer.update_outcomes(hp / max_hp, delay_window=0.9)

            # Damage check: body contact with mob
            for m in mobs:
                if m["dist"] < 32.0:
                    dmg = 8.0 * dt * (2.2 if m.get("is_boss") else 1.0)
                    hp -= dmg
                    collisions_count += 1
                    if hp <= 0:
                        is_alive = False
                        break

            # Projectile hit
            for h in hazards:
                if h["dist"] < 24.0:
                    hp -= 25.0
                    collisions_count += 1
                    if hp <= 0:
                        is_alive = False
                        break

            # Automated combat weapon damage
            for m in mobs:
                if m["dist"] < 190.0:
                    m["hp"] -= 120.0 * dt
            surviving_mobs = []
            for m in mobs:
                if m["hp"] <= 0:
                    kills += 1
                    score += 15 if m.get("is_boss") else 5
                    exp += 25 if m.get("is_boss") else 8
                    if exp >= exp_next:
                        level += 1
                        exp -= exp_next
                        exp_next *= 1.25
                        hp = min(max_hp, hp + 15.0)  # Level up heal
                else:
                    surviving_mobs.append(m)
            mobs = surviving_mobs

        tracer.flush_all()

        ep_duration = round(sim_time, 1)
        results.append({
            "episode": ep,
            "survival_time": ep_duration,
            "score": score,
            "kills": kills,
            "level": level,
            "collisions": collisions_count
        })

        print(f"  [Ván #{ep:02d}] Sống sót: {ep_duration:6.1f}s | Điểm: {score:4d} | Hạ gục: {kills:4d} | Cấp: {level:2d} | Va chạm: {collisions_count}")

    # Statistical distribution
    durations = [r["survival_time"] for r in results]
    scores = [r["score"] for r in results]

    sorted_dur = sorted(durations)
    sorted_sco = sorted(scores)
    n = len(results)

    median_time = sorted_dur[int(n * 0.50)]
    p25_time = sorted_dur[int(n * 0.25)]
    p75_time = sorted_dur[int(n * 0.75)]
    max_time = sorted_dur[-1]

    median_score = sorted_sco[int(n * 0.50)]
    p25_score = sorted_sco[int(n * 0.25)]
    p75_score = sorted_sco[int(n * 0.75)]
    max_score = sorted_sco[-1]

    milestones = {
        "5min (300s)": sum(1 for d in durations if d >= 300.0),
        "10min (600s)": sum(1 for d in durations if d >= 600.0),
        "15min (900s)": sum(1 for d in durations if d >= 900.0),
        "20min (1200s)": sum(1 for d in durations if d >= 1200.0),
        "25min (1500s)": sum(1 for d in durations if d >= 1500.0),
        "30min (1800s)": sum(1 for d in durations if d >= 1800.0)
    }

    report = {
        "episodes": n,
        "survival_time": {
            "median": median_time, "p25": p25_time, "p75": p75_time, "max": max_time
        },
        "score": {
            "median": median_score, "p25": p25_score, "p75": p75_score, "max": max_score
        },
        "milestones": milestones,
        "profiler": profiler.get_summary()
    }

    print("\n" + "=" * 65)
    print("📊 BÁO CÁO PHÂN BỐ HIỆU SUẤT THỰC NGHIỆM (SIMULATION BENCHMARK)")
    print("=" * 65)
    print(f"⏱️ THỜI GIAN SỐNG SÓT:")
    print(f"   - Median (P50) : {median_time:.1f}s")
    print(f"   - P25 / P75    : {p25_time:.1f}s / {p75_time:.1f}s")
    print(f"   - Kỷ lục (Max) : {max_time:.1f}s")
    print(f"🎯 ĐIỂM SỐ GAME (GAME SCORE):")
    print(f"   - Median (P50) : {median_score:.1f}")
    print(f"   - P25 / P75    : {p25_score:.1f} / {p75_score:.1f}")
    print(f"   - Kỷ lục (Max) : {max_score:.1f}")
    print(f"🏆 PHÂN BỐ CÁC CỘT MỐC SỐNG SÓT:")
    for m, count in milestones.items():
        pct = (count / n) * 100.0
        print(f"   - Mốc {m:14s}: {count:2d}/{n} ván ({pct:5.1f}%)")
    print("=" * 65)

    return report

if __name__ == "__main__":
    run_controlled_simulation_benchmark(episodes=10)
