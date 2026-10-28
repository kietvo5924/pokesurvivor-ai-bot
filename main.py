import sys
import os

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pokesurvivor.run_manager import RunManager
from pokesurvivor.memory_store import MemoryStore
from pokesurvivor.metrics_reporter import MetricsReporter

def main():
    parser = argparse.ArgumentParser(description="PokéSurvivor Self-Learning Autoplay Bot")
    parser.add_argument("--runs", type=int, default=3, help="Số lượng ván chơi (mặc định: 3)")
    parser.add_argument("--max-duration", type=float, default=3600.0, help="Thời gian tối đa mỗi ván tính bằng giây (mặc định: 3600s = 60 phút, không giới hạn để tối đa vàng/vé/EXP)")
    parser.add_argument("--observe-only", action="store_true", help="Chế độ chỉ quan sát (không gửi phím điều khiển)")
    parser.add_argument("--headless", action="store_true", help="Chạy trình duyệt ẩn (không mở cửa sổ UI)")
    parser.add_argument("--report", action="store_true", help="Chỉ hiển thị bảng thống kê dữ liệu các ván đã lưu rồi thoát")

    args = parser.parse_args()

    if args.report:
        memory = MemoryStore()
        reporter = MetricsReporter(memory)
        print(reporter.generate_comparison_table())
        return

    manager = RunManager(
        max_runs=args.runs,
        max_duration_per_run=args.max_duration,
        observe_only=args.observe_only,
        headless=args.headless
    )

    try:
        manager.run_session()
    except KeyboardInterrupt:
        print("\n[!] Đã nhận tín hiệu dừng từ bàn phím (Ctrl+C). Đang dọn dẹp...")
        if manager.browser_ctrl:
            manager.browser_ctrl.release_all_keys()
            manager.browser_ctrl.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
