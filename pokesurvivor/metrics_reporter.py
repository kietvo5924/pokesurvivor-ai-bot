from typing import Dict, Any, List
import statistics

class MetricsReporter:
    def __init__(self, memory_store):
        self.memory = memory_store

    def print_run_summary(self, run_data: Dict[str, Any], analysis: Dict[str, Any]):
        print("\n" + "=" * 60)
        print(f"📊 KẾT QUẢ LƯỢT CHƠI #{analysis.get('total_runs', 1)}")
        print("=" * 60)
        print(f"⏱️  Thời gian sống sót: {run_data.get('duration', 0):.1f} giây")
        print(f"⭐ Cấp độ đạt được : Level {run_data.get('max_level', 1)}")
        print(f"💥 Số lần trúng đòn : {run_data.get('damage_events', 0)}")
        print(f"🏁 Nguyên nhân kết thúc : {analysis.get('death_summary', run_data.get('end_reason', 'unknown'))}")
        print(f"🎯 Điểm tổng hợp   : {analysis.get('score', 0):.1f} pts")
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

        durations = [r.get("duration", 0.0) for r in runs]
        levels = [r.get("max_level", 1) for r in runs]
        scores = [r.get("score", 0.0) for r in runs]

        avg_dur = statistics.mean(durations)
        med_dur = statistics.median(durations)
        max_dur = max(durations)
        avg_lvl = statistics.mean(levels)
        max_lvl = max(levels)

        baseline_dur = durations[0] if durations else 0.0
        improvement = ((avg_dur - baseline_dur) / baseline_dur * 100.0) if baseline_dur > 0 else 0.0

        report = []
        report.append("\n📈 BẢNG SO SÁNH HIỆU SUẤT TỔNG THỂ")
        report.append("-" * 65)
        report.append(f"Tổng số lượt chơi đã lưu : {len(runs)}")
        report.append(f"Lượt cơ sở (Baseline #1) : {baseline_dur:.1f}s | Level {levels[0] if levels else 1}")
        report.append(f"Kỷ lục tốt nhất (Best)   : {max_dur:.1f}s | Level {max_lvl}")
        report.append(f"Thời gian TB (Mean)      : {avg_dur:.1f}s | Median: {med_dur:.1f}s")
        report.append(f"Cấp độ trung bình        : Level {avg_lvl:.1f}")
        report.append(f"Tăng trưởng so baseline  : {improvement:+.1f}%")
        report.append("-" * 65)
        report.append(f"{'Run':<5} | {'Thời gian':<10} | {'Level':<6} | {'Hits':<5} | {'Policy':<8} | {'Score':<8}")
        report.append("-" * 65)
        
        for r in runs[-10:]: # last 10 runs
            report.append(
                f"#{r.get('run_id', '?'):<4} | {r.get('duration', 0.0):<9.1f}s | {r.get('max_level', 1):<6} | "
                f"{r.get('damage_events', 0):<5} | v{r.get('policy_version', '1.0'):<6} | {r.get('score', 0.0):<8.1f}"
            )
        report.append("-" * 65 + "\n")

        return "\n".join(report)

