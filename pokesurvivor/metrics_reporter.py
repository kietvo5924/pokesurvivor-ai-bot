from typing import Dict, Any, List, Optional
import statistics

class MetricsReporter:
    def __init__(self, memory_store):
        self.memory = memory_store

    def print_run_summary(self, run_data: Dict[str, Any], analysis: Dict[str, Any]):
        g_score = run_data.get("game_score")
        g_src = run_data.get("game_score_source", "unobserved")
        dur = float(run_data.get("survival_time", run_data.get("duration", 0.0)))
        dur_src = run_data.get("survival_time_source", "state_tracker_clock")
        lvl = run_data.get("max_level", 1)
        lvl_src = run_data.get("max_level_source", "state_tracker_verified")
        fit_score = analysis.get("fitness_score", analysis.get("score", 0.0))
        p_ver = analysis.get("policy_version", "1.0")
        p_hash = analysis.get("policy_hash", "")[:8]

        print("\n" + "=" * 60)
        print(f"📊 KẾT QUẢ LƯỢT CHƠI #{analysis.get('total_runs', 1)}")
        print("=" * 60)
        print(f"🎯 Điểm Game thực tế (Kills) : {g_score if g_score is not None else 'N/A (Chưa ghi nhận)'} [Nguồn: {g_src}]")
        print(f"⏱️  Thời gian sống sót      : {dur:.1f} giây [Nguồn: {dur_src}]")
        print(f"⭐ Cấp độ đạt được          : Level {lvl} [Nguồn: {lvl_src}]")
        print(f"📈 Điểm Fitness nội bộ      : {fit_score:.1f} pts [Nguồn: internal_composite_formula]")
        print(f"🏷️  Chính sách áp dụng       : v{p_ver} (hash: {p_hash})")
        print(f"💥 Số lần trúng đòn         : {run_data.get('damage_events', 0)}")
        print(f"🏁 Nguyên nhân kết thúc     : {analysis.get('death_summary', run_data.get('end_reason', 'unknown'))}")
        
        if analysis.get('is_new_best'):
            print("🏆 ĐẠT KỶ LỤC MỚI (NEW BEST RECORD)!")
        
        upgrades = run_data.get("upgrades", [])
        if upgrades:
            print("\n📦 Nâng cấp đã chọn:")
            for u in upgrades:
                print(f"  - Lvl {u.get('level')}: {u.get('name')} ({u.get('tag')}) at {u.get('time')}s")

        adaptations = analysis.get("adaptations", [])
        if adaptations:
            print("\n🧠 Điều chỉnh chiến thuật (Learning Adaptation):")
            for a in adaptations:
                print(f"  ⚡ {a}")
        print("=" * 60 + "\n")

    def generate_comparison_table(self) -> str:
        runs = self.memory.get_all_runs()
        if not runs:
            return "Chưa có dữ liệu lượt chơi nào."

        durations = [float(r.get("duration", 0.0)) for r in runs]
        levels = [int(r.get("max_level", 1)) for r in runs]
        game_scores = [r.get("game_score") for r in runs if r.get("game_score") is not None]
        fitness_scores = [float(r.get("fitness_score", r.get("score", 0.0))) for r in runs]

        avg_dur = statistics.mean(durations)
        med_dur = statistics.median(durations)
        max_dur = max(durations)
        avg_lvl = statistics.mean(levels)
        max_lvl = max(levels)

        report = []
        report.append("\n📈 BẢNG SO SÁNH HIỆU SUẤT TỔNG THỂ")
        report.append("-" * 75)
        report.append(f"Tổng số lượt chơi đã lưu : {len(runs)}")
        if game_scores:
            avg_gs = statistics.mean(game_scores)
            med_gs = statistics.median(game_scores)
            max_gs = max(game_scores)
            report.append(f"Điểm Game thực tế (Kills): Mean: {avg_gs:.1f} | Median: {med_gs:.1f} | Max: {max_gs}")
        else:
            report.append("Điểm Game thực tế (Kills): Chưa có dữ liệu DOM ghi nhận từ các lượt cũ")
        report.append(f"Thời gian sống sót (Time): Mean: {avg_dur:.1f}s | Median: {med_dur:.1f}s | Best: {max_dur:.1f}s")
        report.append(f"Cấp độ (Max Level)       : Mean: Level {avg_lvl:.1f} | Best: Level {max_lvl}")
        report.append("-" * 75)
        report.append(f"{'Run':<5} | {'GameScore':<10} | {'Thời gian':<10} | {'Level':<6} | {'Hits':<5} | {'Policy':<8} | {'Fitness':<8}")
        report.append("-" * 75)
        
        for r in runs[-10:]: # last 10 runs
            gs_val = r.get("game_score")
            gs_str = f"{gs_val}" if gs_val is not None else "N/A"
            fit_val = float(r.get("fitness_score", r.get("score", 0.0)))
            p_str = f"v{r.get('policy_version', '1.0')}"
            report.append(
                f"#{r.get('run_id', '?'):<4} | {gs_str:<10} | {float(r.get('duration', 0.0)):<9.1f}s | {r.get('max_level', 1):<6} | "
                f"{r.get('damage_events', 0):<5} | {p_str:<8} | {fit_val:<8.1f}"
            )
        report.append("-" * 75 + "\n")

        return "\n".join(report)
